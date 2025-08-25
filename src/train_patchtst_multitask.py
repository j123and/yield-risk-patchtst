#!/usr/bin/env python3
"""
Train a simple PatchTST-like multi-task model (quantile τ=0.05 + log-variance) and write holdout predictions.
- Adds full reproducibility seeding.
- Uses custom collate_fns to avoid default_collate choking on numpy.datetime64.
- NO side effects on import.

Usage:
  python src/train_patchtst_multitask.py --npz outputs/spy_seq_120.npz --split_date 2023-01-02 --epochs 10
"""
from __future__ import annotations
import argparse
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


def seed_everything(seed: int = 1337) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def pinball_loss(y_hat: torch.Tensor, y: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
    e = y - y_hat
    return torch.mean(torch.maximum(tau * e, (tau - 1.0) * e))


class SeqDataset(Dataset):
    def __init__(self, npz_path: str, split_date: str | None, train: bool, seq_len: int | None = None):
        z = np.load(npz_path, allow_pickle=True)
        self.X = z["X"].astype(np.float32)
        self.y_ret = z["y_ret"].astype(np.float32)
        self.y_lrv = z["y_lrv"].astype(np.float32)  # log variance
        self.dates = z["dates"].astype("datetime64[D]")
        if seq_len is not None and self.X.shape[1] != seq_len:
            raise ValueError(f"seq_len mismatch: data has {self.X.shape[1]}, arg was {seq_len}")
        if split_date is None:
            split = int(0.8 * len(self.X))
            idx_tr = np.arange(split)
            idx_te = np.arange(split, len(self.X))
        else:
            d = np.datetime64(split_date)
            idx_tr = np.where(self.dates < d)[0]
            idx_te = np.where(self.dates >= d)[0]
        self.idx = idx_tr if train else idx_te

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        j = self.idx[i]
        return self.X[j], self.y_ret[j], self.y_lrv[j], self.dates[j]


def collate_train(batch):
    """Custom collate that drops dates for training."""
    X, y_ret, y_lrv, _ = zip(*batch)
    X = torch.as_tensor(np.stack(X))
    y_ret = torch.as_tensor(np.stack(y_ret))
    y_lrv = torch.as_tensor(np.stack(y_lrv))
    return X, y_ret, y_lrv, None


def collate_with_dates(batch):
    """Custom collate that returns dates as a plain Python list (avoids default_collate on datetime64)."""
    X, y_ret, y_lrv, d = zip(*batch)
    X = torch.as_tensor(np.stack(X))
    y_ret = torch.as_tensor(np.stack(y_ret))
    y_lrv = torch.as_tensor(np.stack(y_lrv))
    d = list(d)  # list of numpy.datetime64[D]
    return X, y_ret, y_lrv, d


class PatchTSTMulti(pl.LightningModule):
    def __init__(self, seq_len=120, n_feat=2, patch_len=20, d_model=64, nhead=4, nlayers=2, dropout=0.1, lr=1e-3, tau=0.05, w_q=1.0, w_v=1.0):
        super().__init__()
        self.save_hyperparameters()
        self.tau = tau
        self.lr = lr
        self.w_q = w_q
        self.w_v = w_v
        pad = (patch_len - (seq_len % patch_len)) % patch_len
        L = seq_len + pad
        self.n_patches = L // patch_len
        self.patch_len = patch_len
        self.pad = pad
        self.n_feat = n_feat

        self.proj = nn.Linear(n_feat * patch_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head_q = nn.Linear(d_model, 1)
        self.head_lrv = nn.Linear(d_model, 1)

        self.mse = nn.MSELoss()

    def forward(self, x: torch.Tensor):
        if self.pad:
            pad = torch.zeros((x.size(0), self.pad, x.size(2)), device=x.device, dtype=x.dtype)
            x = torch.cat([pad, x], dim=1)
        B, L, F = x.shape
        x = x.reshape(B, self.n_patches, self.patch_len * F)
        x = self.proj(x)
        z = self.encoder(x)
        z = z.mean(dim=1)
        q = self.head_q(z).squeeze(-1)
        lrv = self.head_lrv(z).squeeze(-1)
        return q, lrv

    def training_step(self, batch, batch_idx):
        X, y_ret, y_lrv, _ = batch
        qhat, lrvhat = self(X)
        loss_q = pinball_loss(qhat, y_ret, self.tau)
        loss_v = self.mse(lrvhat, y_lrv)
        loss = self.w_q * loss_q + self.w_v * loss_v
        self.log_dict({"train_loss": loss, "loss_q": loss_q, "loss_v": loss_v}, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def train(npz_path: str, split_date: str | None, out_csv: str, epochs=10, batch=64, seed=1337,
          seq_len=120, n_feat=2, patch_len=20, d_model=64, nhead=4, nlayers=2, dropout=0.1, lr=1e-3, tau=0.05, w_q=1.0, w_v=1.0) -> None:
    seed_everything(seed)

    ds_tr = SeqDataset(npz_path, split_date, train=True, seq_len=seq_len)
    ds_te = SeqDataset(npz_path, split_date, train=False, seq_len=seq_len)
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, num_workers=0, collate_fn=collate_train)
    dl_te = DataLoader(ds_te, batch_size=batch, shuffle=False, num_workers=0, collate_fn=collate_with_dates)

    model = PatchTSTMulti(seq_len=seq_len, n_feat=n_feat, patch_len=patch_len, d_model=d_model,
                          nhead=nhead, nlayers=nlayers, dropout=dropout, lr=lr, tau=tau, w_q=w_q, w_v=w_v)

    trainer = pl.Trainer(max_epochs=epochs, log_every_n_steps=10, enable_checkpointing=False, enable_model_summary=False)
    trainer.fit(model, dl_tr)

    # Predict on holdout
    model.eval()
    preds_q, preds_lrv, dates, y_true = [], [], [], []
    with torch.no_grad():
        for X, y_ret, y_lrv, d in dl_te:
            q, lrv = model(X)
            preds_q.extend(q.cpu().numpy().tolist())
            preds_lrv.extend(lrv.cpu().numpy().tolist())
            dates.extend(d)  # list of numpy.datetime64[D]
            y_true.extend(y_ret.cpu().numpy().tolist())

    sigma2 = np.exp(np.array(preds_lrv, dtype=np.float64))  # ensure positivity
    date_arr = np.array(dates, dtype="datetime64[D]")
    out = pd.DataFrame({
        "date": date_arr,
        "ret_true": y_true,
        "q05_ret_pred": preds_q,
        "sigma2_pred": sigma2,
    })
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}  (rows={len(out)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="outputs/spy_seq_120.npz")
    ap.add_argument("--split_date", default=None, help="YYYY-MM-DD; if omitted, 80/20 split")
    ap.add_argument("--out_csv", default="outputs/patch_preds.csv")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seq_len", type=int, default=120)
    ap.add_argument("--patch_len", type=int, default=20)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--w_q", type=float, default=1.0)
    ap.add_argument("--w_v", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    train(args.npz, args.split_date, args.out_csv, epochs=args.epochs, batch=args.batch, seed=args.seed,
          seq_len=args.seq_len, n_feat=2, patch_len=args.patch_len, d_model=args.d_model,
          nhead=args.nhead, nlayers=args.nlayers, dropout=args.dropout, lr=args.lr, tau=args.tau, w_q=args.w_q, w_v=args.w_v)
