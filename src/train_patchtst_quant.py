#!/usr/bin/env python3
import argparse, math
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

# ---------- utils ----------
def pinball_loss(y_hat, y, tau=0.05):
    # y_hat, y: (B,)
    e = y - y_hat
    return torch.mean(torch.maximum(tau*e, (tau-1)*e))

class SeqDataset(Dataset):
    def __init__(self, npz_path, split_date=None):
        z = np.load(npz_path, allow_pickle=True)
        self.X = z["X"].astype(np.float32)         # (N,T,F)
        self.y_ret = z["y_ret"].astype(np.float32) # (N,)
        self.dates = z["dates"].astype("datetime64[D]")
        if split_date is not None:
            d = np.datetime64(split_date)
            mask = self.dates < d
            self.train_idx = np.where(mask)[0]
            self.val_idx   = np.where(~mask)[0]
        else:
            n = len(self.X)
            split = int(n*0.8)
            self.train_idx = np.arange(split)
            self.val_idx   = np.arange(split, n)

    def get_split(self, train=True):
        idx = self.train_idx if train else self.val_idx
        X = torch.from_numpy(self.X[idx])          # (N,T,F)
        y = torch.from_numpy(self.y_ret[idx])
        d = torch.from_numpy(idx.astype(np.int64))
        return torch.utils.data.TensorDataset(X, y, d)

# ---------- model (PatchTST-lite) ----------
class PatchTSTLite(pl.LightningModule):
    def __init__(self, seq_len=120, n_feat=2, patch_len=20, d_model=64, nhead=4, nlayers=2, dropout=0.1, lr=1e-3, tau=0.05):
        super().__init__()
        self.save_hyperparameters()
        self.tau = tau
        self.lr = lr
        self.patch_len = patch_len
        self.pad = (patch_len - (seq_len % patch_len)) % patch_len
        L = seq_len + self.pad
        self.n_patches = L // patch_len

        self.proj = nn.Linear(n_feat*patch_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, batch_first=True, dropout=dropout, activation="gelu")
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):  # x: (B,T,F)
        B,T,F = x.shape
        if self.pad:
            pad = torch.zeros(B, self.pad, F, device=x.device, dtype=x.dtype)
            x = torch.cat([pad, x], dim=1)
            T = x.shape[1]
        # reshape into patches (B, n_patches, patch_len*F)
        x = x.view(B, self.n_patches, self.hparams.patch_len, F).contiguous().view(B, self.n_patches, -1)
        x = self.proj(x)                    # (B, P, d_model)
        x = self.encoder(x)                 # (B, P, d_model)
        x = x.mean(dim=1)                   # global average over patches
        q = self.head(x).squeeze(-1)        # (B,)
        return q

    def training_step(self, batch, _):
        x,y,_ = batch
        q = self(x)
        loss = pinball_loss(q, y, self.tau)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x,y,idx = batch
        q = self(x)
        loss = pinball_loss(q, y, self.tau)
        self.log("val_loss", loss, prog_bar=True)
        return {"q": q.detach(), "y": y.detach(), "idx": idx.detach()}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

def train(npz_path, split_date, out_csv, epochs=60, batch=64, **h):
    ds = SeqDataset(npz_path, split_date)
    train_dl = DataLoader(ds.get_split(True), batch_size=batch, shuffle=True, drop_last=True, num_workers=2)
    val_dl   = DataLoader(ds.get_split(False), batch_size=batch, shuffle=False, num_workers=2)

    model = PatchTSTLite(**h)
    ckpt = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
    trainer = pl.Trainer(max_epochs=epochs, log_every_n_steps=10, callbacks=[ckpt],
                         enable_checkpointing=True, accelerator="auto", devices=1)
    trainer.fit(model, train_dl, val_dl)
    best_path = ckpt.best_model_path or ckpt.last_model_path
    print("Best ckpt:", best_path)

    # ---- Make validation predictions and align them to dates ----
    z = np.load(npz_path, allow_pickle=True)
    dates = z["dates"]  # full date array for all windows

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PatchTSTLite.load_from_checkpoint(best_path, **h).to(device)
    model.eval()

    idx_list, y_list, q_list = [], [], []
    with torch.no_grad():
        for xb, yb, ib in val_dl:           # xb: (B,T,F), yb: (B,), ib: original indices
            xb = xb.to(device)
            qb = model(xb).cpu().numpy()    # predicted q05 next-day return
            idx_list.append(ib.numpy())
            y_list.append(yb.numpy())
            q_list.append(qb)

    idx = np.concatenate(idx_list)
    yv  = np.concatenate(y_list)
    qv  = np.concatenate(q_list)

    order = np.argsort(idx)  # restore chronological order
    val_dates = dates[ds.val_idx][order].astype("datetime64[D]")

    out = pd.DataFrame({
        "date": val_dates,
        "ret_true": yv[order],
        "q05_ret_pred": qv[order],
    })
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(out)} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--seq_len", type=int, default=120)
    ap.add_argument("--split_date", default="2023-01-02", help="validation start (holdout)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    # model hparams
    ap.add_argument("--patch_len", type=int, default=20)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    npz = f"outputs/{args.symbol.lower()}_seq_{args.seq_len}.npz"
    Path("outputs").mkdir(exist_ok=True, parents=True)
    out_csv = f"outputs/patch_preds.csv"
    train(npz, args.split_date, out_csv, epochs=args.epochs, batch=args.batch,
          seq_len=args.seq_len, n_feat=2,
          patch_len=args.patch_len, d_model=args.d_model,
          nhead=args.nhead, nlayers=args.nlayers, dropout=args.dropout, lr=args.lr, tau=0.05)
