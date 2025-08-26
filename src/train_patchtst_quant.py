#!/usr/bin/env python3
# Train a simple PatchTST-ish quantile model (tau=0.05) and write holdout predictions.
# Junior-level: simple seeding, simple encoder, minimal checks.

import argparse, os, random
from pathlib import Path
import numpy as np, pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

def seed_everything(seed=1337):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

def pinball_loss(yhat, y, tau=0.05):
    e = y - yhat
    return torch.mean(torch.maximum(tau*e, (tau-1.0)*e))

class SeqDataset(Dataset):
    def __init__(self, npz_path, split_date, train, seq_len=None):
        z = np.load(npz_path, allow_pickle=True)
        self.X = z["X"].astype(np.float32)          # (N, T, F)
        self.y_ret = z["y_ret"].astype(np.float32)  # (N,)
        self.y_lrv = z["y_lrv"].astype(np.float32)  # (N,)
        self.dates = z["dates"].astype("datetime64[D]")
        if seq_len is not None and self.X.shape[1] != seq_len:
            raise ValueError(f"seq_len mismatch: data has {self.X.shape[1]}, arg {seq_len}")
        if split_date is None:
            cut = int(0.8 * len(self.X))
            idx_tr = np.arange(cut)
            idx_te = np.arange(cut, len(self.X))
        else:
            d = np.datetime64(split_date)
            idx_tr = np.where(self.dates < d)[0]
            idx_te = np.where(self.dates >= d)[0]
            if idx_tr.size == 0 or idx_te.size == 0:
                raise ValueError("Empty train or test after time split; check --split_date.")
            if self.dates[idx_tr].max() >= self.dates[idx_te].min():
                raise ValueError("Train/test overlap; bad split.")
        self.idx = idx_tr if train else idx_te

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        return self.X[j], self.y_ret[j], self.y_lrv[j], self.dates[j]

def collate_train(batch):
    X, y_ret, y_lrv, _ = zip(*batch)
    return (torch.as_tensor(np.stack(X)),
            torch.as_tensor(np.stack(y_ret)),
            torch.as_tensor(np.stack(y_lrv)),
            None)

def collate_with_dates(batch):
    X, y_ret, y_lrv, d = zip(*batch)
    return (torch.as_tensor(np.stack(X)),
            torch.as_tensor(np.stack(y_ret)),
            torch.as_tensor(np.stack(y_lrv)),
            list(d))

class PatchTSTQuant(pl.LightningModule):
    def __init__(self, seq_len=120, n_feat=2, patch_len=20,
                 d_model=64, nhead=4, nlayers=2, dropout=0.1, lr=1e-3, tau=0.05):
        super().__init__()
        if seq_len % patch_len != 0:
            raise ValueError("seq_len must be divisible by patch_len (no zero padding).")
        self.tau = tau
        self.lr = lr
        self.n_patches = seq_len // patch_len
        self.patch_len = patch_len
        self.n_feat = n_feat

        # flatten each patch: (patch_len * n_feat) -> d_model
        self.proj = nn.Linear(n_feat * patch_len, d_model)

        # simplest way to give the model order: learnable patch positions
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.normal_(self.pos, 0.0, 0.02)

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head_q = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, T, F) with T % patch_len == 0
        B, T, F = x.shape
        x = x.reshape(B, self.n_patches, self.patch_len * F)  # (B, P, patch_len*F)
        x = self.proj(x)                                      # (B, P, d_model)
        x = x + self.pos                                      # (B, P, d_model)
        z = self.encoder(x)                                   # (B, P, d_model)
        z = z.mean(dim=1)                                     # (B, d_model)
        q = self.head_q(z).squeeze(-1)                        # (B,)
        return q

    def training_step(self, batch, _):
        X, y_ret, _, _ = batch
        yhat = self(X)
        loss = pinball_loss(yhat, y_ret, self.tau)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def train(npz_path, split_date, out_csv, epochs=10, batch=64, seed=1337,
          seq_len=120, n_feat=2, patch_len=20, d_model=64, nhead=4, nlayers=2,
          dropout=0.1, lr=1e-3, tau=0.05):
    seed_everything(seed)

    ds_tr = SeqDataset(npz_path, split_date, train=True,  seq_len=seq_len)
    ds_te = SeqDataset(npz_path, split_date, train=False, seq_len=seq_len)
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True,  num_workers=0, collate_fn=collate_train)
    dl_te = DataLoader(ds_te, batch_size=batch, shuffle=False, num_workers=0, collate_fn=collate_with_dates)

    model = PatchTSTQuant(seq_len=seq_len, n_feat=n_feat, patch_len=patch_len,
                          d_model=d_model, nhead=nhead, nlayers=nlayers,
                          dropout=dropout, lr=lr, tau=tau)

    trainer = pl.Trainer(max_epochs=epochs, log_every_n_steps=10,
                         enable_checkpointing=False, enable_model_summary=False)
    trainer.fit(model, dl_tr)

    # predict on holdout
    model.eval()
    preds, dates, ytrue = [], [], []
    with torch.no_grad():
        for X, y_ret, _y_lrv, d in dl_te:
            q = model(X)
            preds += q.cpu().numpy().tolist()
            dates += d
            ytrue += y_ret.cpu().numpy().tolist()

    df = pd.DataFrame({
        "date": np.array(dates, dtype="datetime64[D]"),
        "ret_true": ytrue,
        "q05_ret_pred": preds,
        "sigma2_pred": np.nan,  # multitask will fill
    })
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} (rows={len(df)})")

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
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    train(args.npz, args.split_date, args.out_csv, epochs=args.epochs, batch=args.batch, seed=args.seed,
          seq_len=args.seq_len, n_feat=2, patch_len=args.patch_len, d_model=args.d_model,
          nhead=args.nhead, nlayers=args.nlayers, dropout=args.dropout, lr=args.lr, tau=args.tau)
