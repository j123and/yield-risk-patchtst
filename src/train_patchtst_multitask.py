#!/usr/bin/env python3
"""
PatchTST multitask: τ-quantile + log-variance, trained end-to-end on RETURNS.

We predict:
  - q05_z_pred  : 5% quantile in z-space (latent)
  - lrv_pred    : log variance

Quantile loss is applied in RETURN space using our own σ̂:

    sigma_hat = exp(0.5 * lrv_pred)
    q05_ret   = mu_ret + q05_z_pred * sigma_hat
    loss_q    = pinball(q05_ret, y_ret, τ)

This removes post-hoc calibration.

Writes CSV:
  date, ret_true, q05_ret_pred, sigma2_pred, q05_z_pred
"""
import argparse, os, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
import pytorch_lightning as pl


# ---------------- utils ----------------
def seed_everything(seed: int = 1337) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

def pinball_loss(y_hat: torch.Tensor, y: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
    e = y - y_hat
    return torch.where(e >= 0, tau * e, (tau - 1.0) * e).mean()


# ---------------- data ----------------
class SeqDataset(Dataset):
    """Loads .npz from build_sequences.py and applies a leak-safe time split."""
    def __init__(self, npz_path: str, split_date: str | None, train: bool, seq_len: int | None = None):
        z = np.load(npz_path, allow_pickle=True)
        self.X = z["X"].astype(np.float32)
        self.y_ret = z["y_ret"].astype(np.float32)
        self.y_lrv = z["y_lrv"].astype(np.float32)
        self.dates = z["dates"].astype("datetime64[D]")

        if seq_len is not None and self.X.shape[1] != seq_len:
            raise ValueError(f"seq_len mismatch: data has {self.X.shape[1]}, arg was {seq_len}")

        if split_date is None:
            cut = int(0.8 * len(self.X))
            idx_tr = np.arange(cut)
            idx_te = np.arange(cut, len(self.X))
        else:
            d = np.datetime64(split_date)
            idx_tr = np.where(self.dates < d)[0]
            idx_te = np.where(self.dates >= d)[0]
            if idx_tr.size == 0 or idx_te.size == 0:
                raise ValueError("Time split produced empty train or test; adjust --split_date.")
            if self.dates[idx_tr].max() >= self.dates[idx_te].min():
                raise ValueError("Train/test temporal overlap detected; check split logic.")

        self.idx_train, self.idx_test = idx_tr, idx_te
        self.use_train = train

    def __len__(self) -> int:
        return len(self.idx_train) if self.use_train else len(self.idx_test)

    def __getitem__(self, i: int):
        j = self.idx_train[i] if self.use_train else self.idx_test[i]
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


# ---------------- model ----------------
class PatchTSTMulti(pl.LightningModule):
    def __init__(self, seq_len=120, n_feat=2, patch_len=20, d_model=128, nhead=4, nlayers=3,
                 dropout=0.1, lr=3e-4, tau=0.05, w_q=12.0, w_v=1.0):
        super().__init__()
        if seq_len % patch_len != 0:
            raise ValueError("seq_len must be divisible by patch_len.")
        self.tau, self.lr, self.w_q, self.w_v = tau, lr, w_q, w_v

        self.n_patches = seq_len // patch_len
        self.patch_len = patch_len
        self.n_feat = n_feat

        self.proj = nn.Linear(n_feat * patch_len, d_model)
        self.pos  = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.normal_(self.pos, 0.0, 0.02)

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)

        self.head_qz  = nn.Linear(d_model, 1)  # τ-quantile in z-space
        self.head_lrv = nn.Linear(d_model, 1)  # log variance
        self.mse = nn.MSELoss()

        # mean return (set from train data)
        self.register_buffer("mu_ret", torch.tensor(0.0), persistent=False)

    def forward(self, x: torch.Tensor):
        B, T, F = x.shape
        x = x.reshape(B, self.n_patches, self.patch_len * F)
        h = self.encoder(self.proj(x) + self.pos).mean(dim=1)
        qz  = self.head_qz(h).squeeze(-1)
        lrv = self.head_lrv(h).squeeze(-1)
        return qz, lrv

    @staticmethod
    def _sigma_from_lrv(lrv: torch.Tensor) -> torch.Tensor:
        # No in-place ops; stable: sigma = exp(0.5 * lrv) with out-of-place clamp.
        sigma = torch.exp(0.5 * lrv)
        return torch.clamp(sigma, min=1e-10, max=1e3)

    def training_step(self, batch, _):
        X, y_ret, y_lrv, _ = batch
        qz, lrvhat = self(X)
        sigma_hat = self._sigma_from_lrv(lrvhat)
        q_ret = self.mu_ret + qz * sigma_hat

        loss_q = pinball_loss(q_ret, y_ret, self.tau)   # return-space
        loss_v = self.mse(lrvhat, y_lrv)
        loss   = self.w_q * loss_q + self.w_v * loss_v

        with torch.no_grad():
            exc_ret = (y_ret < q_ret).float().mean()
        self.log_dict({"train_loss": loss, "train_q": loss_q, "train_v": loss_v,
                       "train_exc_ret": exc_ret},
                      prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        X, y_ret, _y_lrv, _ = batch
        qz, lrvhat = self(X)
        sigma_hat = self._sigma_from_lrv(lrvhat)
        q_ret = self.mu_ret + qz * sigma_hat
        loss_q = pinball_loss(q_ret, y_ret, self.tau)
        exc_ret = (y_ret < q_ret).float().mean()
        self.log_dict({"val_q": loss_q, "val_exc_ret": exc_ret},
                      prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


# ---------------- train + predict ----------------
def train(npz_path: str, split_date: str | None, out_csv: str, epochs=60, batch=128, seed=1337,
          seq_len=120, n_feat=2, patch_len=20, d_model=128, nhead=4, nlayers=3,
          dropout=0.1, lr=3e-4, tau=0.05, w_q=12.0, w_v=1.0) -> None:
    seed_everything(seed)

    ds_all_tr = SeqDataset(npz_path, split_date, train=True,  seq_len=seq_len)
    ds_te     = SeqDataset(npz_path, split_date, train=False, seq_len=seq_len)

    ntr = len(ds_all_tr.idx_train)
    cut = max(1, int(0.9 * ntr))
    ds_tr  = Subset(ds_all_tr, list(range(0,   cut)))
    ds_val = Subset(ds_all_tr, list(range(cut, ntr)))

    dl_tr  = DataLoader(ds_tr,  batch_size=batch, shuffle=True,  num_workers=0, collate_fn=collate_train)
    dl_val = DataLoader(ds_val, batch_size=batch, shuffle=False, num_workers=0, collate_fn=collate_train)
    dl_te  = DataLoader(ds_te,  batch_size=batch, shuffle=False, num_workers=0, collate_fn=collate_with_dates)

    model = PatchTSTMulti(seq_len=seq_len, n_feat=n_feat, patch_len=patch_len, d_model=d_model,
                          nhead=nhead, nlayers=nlayers, dropout=dropout, lr=lr, tau=tau,
                          w_q=w_q, w_v=w_v)

    # set μ_ret from the actual train chunk used for fitting
    y_train = torch.as_tensor(ds_all_tr.y_ret[ds_all_tr.idx_train[:cut]], dtype=torch.float32)
    mu_ret = float(y_train.mean().item())
    with torch.no_grad():
        model.mu_ret.copy_(torch.tensor(mu_ret, dtype=torch.float32, device=model.mu_ret.device))
    print(f"[ret mean] mu_ret_train={mu_ret:.6g}")

    # bias-init qz toward empirical τ-quantile in z using σ_true (warm start only)
    y_lrv_tr = torch.as_tensor(ds_all_tr.y_lrv[ds_all_tr.idx_train[:cut]], dtype=torch.float32)
    sigma_true_tr = torch.exp(0.5 * y_lrv_tr).clamp(min=1e-10)  # out-of-place clamp, no grad here anyway
    z_tr = (y_train - mu_ret) / sigma_true_tr
    qz_emp = float(np.quantile(z_tr.numpy(), tau))
    with torch.no_grad():
        model.head_qz.bias.fill_(qz_emp)

    trainer = pl.Trainer(max_epochs=epochs, log_every_n_steps=10,
                         enable_checkpointing=False, enable_model_summary=False)
    trainer.fit(model, dl_tr, dl_val)

    # predict holdout
    model.eval()
    dates, y_true, q05_ret, q05_z, sigma2 = [], [], [], [], []
    with torch.no_grad():
        for X, y_ret, _y_lrv, d in dl_te:
            qz, lrvhat = model(X)
            sigma_hat = model._sigma_from_lrv(lrvhat)
            q_ret = model.mu_ret + qz * sigma_hat

            dates  += d
            y_true += y_ret.cpu().numpy().tolist()
            q05_ret += q_ret.cpu().numpy().tolist()
            q05_z   += qz.cpu().numpy().tolist()
            sigma2  += torch.exp(lrvhat).cpu().numpy().tolist()

    out = pd.DataFrame({
        "date": np.array(dates, dtype="datetime64[D]"),
        "ret_true": y_true,
        "q05_ret_pred": q05_ret,
        "sigma2_pred": sigma2,
        "q05_z_pred": q05_z,
    })
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}  (rows={len(out)})")


# ---------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="outputs/spy_seq_120.npz")
    ap.add_argument("--split_date", default=None, help="YYYY-MM-DD; if omitted, 80/20 split")
    ap.add_argument("--out_csv", default="outputs/patch_preds.csv")

    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seq_len", type=int, default=120)
    ap.add_argument("--patch_len", type=int, default=20)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)

    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--w_q", type=float, default=12.0)
    ap.add_argument("--w_v", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1337)

    args = ap.parse_args()
    train(args.npz, args.split_date, args.out_csv, epochs=args.epochs, batch=args.batch, seed=args.seed,
          seq_len=args.seq_len, n_feat=2, patch_len=args.patch_len, d_model=args.d_model,
          nhead=args.nhead, nlayers=args.nlayers, dropout=args.dropout, lr=args.lr,
          tau=args.tau, w_q=args.w_q, w_v=args.w_v)
