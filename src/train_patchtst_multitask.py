#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd, torch
from pathlib import Path
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

def pinball_loss(y_hat, y, tau=0.05):
    e = y - y_hat
    return torch.mean(torch.maximum(tau*e, (tau-1)*e))

class SeqDataset(Dataset):
    """
    Loads outputs/<sym>_seq_<L>.npz with arrays:
      X (N,T,2) features: [ret, sigma_gk]
      y_ret (N,) next-day return
      y_lrv (N,) next-day log RV
      dates (N,) window target dates
    Splits by date; computes z-score scaling from TRAIN split only.
    """
    def __init__(self, npz_path, split_date=None, train=True):
        z = np.load(npz_path, allow_pickle=True)
        self.X = z["X"].astype(np.float32)
        self.y_ret = z["y_ret"].astype(np.float32)
        self.y_lrv = z["y_lrv"].astype(np.float32)
        self.dates = z["dates"].astype("datetime64[D]")
        if split_date is None:
            split = int(len(self.X)*0.8)
            self.train_idx = np.arange(split)
            self.val_idx   = np.arange(split, len(self.X))
        else:
            d = np.datetime64(split_date)
            self.train_idx = np.where(self.dates < d)[0]
            self.val_idx   = np.where(self.dates >= d)[0]
        self.train = train
        # fit scaler on train subset
        mu = self.X[self.train_idx].mean(axis=(0,1), keepdims=True)
        sd = self.X[self.train_idx].std(axis=(0,1), keepdims=True) + 1e-8
        self.mu, self.sd = mu.astype(np.float32), sd.astype(np.float32)

    def __len__(self):
        idx = self.train_idx if self.train else self.val_idx
        return len(idx)

    def __getitem__(self, i):
        idx = (self.train_idx if self.train else self.val_idx)[i]
        x = (self.X[idx] - self.mu) / self.sd
        return (torch.from_numpy(x),
                torch.tensor(self.y_ret[idx]),
                torch.tensor(self.y_lrv[idx]),
                torch.tensor(idx, dtype=torch.long))

class PatchTSTMulti(pl.LightningModule):
    def __init__(self, seq_len=120, n_feat=2, patch_len=20, d_model=64, nhead=4, nlayers=2, dropout=0.1,
                 lr=1e-3, tau=0.05, beta=0.3, pad_left=True):
        super().__init__()
        self.save_hyperparameters()
        self.tau, self.lr, self.beta = tau, lr, beta
        L = seq_len
        self.pad = (patch_len - (L % patch_len)) % patch_len if pad_left else 0
        L_eff = L + self.pad
        self.n_patches = L_eff // patch_len
        self.patch_len = patch_len
        self.pad_left = pad_left

        self.proj = nn.Linear(n_feat*patch_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
                                               batch_first=True, dropout=dropout, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head_q = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))   # q05 return
        self.head_v = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))   # log variance

    def forward(self, x):
        # x expected (B, T, F). If a stray dim sneaks in (e.g., (B,1,T,F)), squeeze it.
        if x.dim() > 3:
            x = x.view(x.shape[0], x.shape[-2], x.shape[-1])
        B, T, F = x.size(0), x.size(1), x.size(2)

        # left-pad to make T divisible by patch_len
        if self.pad:
            pad = torch.zeros(B, self.pad, F, device=x.device, dtype=x.dtype)
            x = torch.cat([pad, x], dim=1)
            T = x.size(1)

        # patchify: (B, P, patch_len*F)
        x = x.view(B, self.n_patches, self.patch_len, F).contiguous().view(B, self.n_patches, -1)
        x = self.proj(x)
        x = self.encoder(x)
        h = x.mean(dim=1)
        q05 = self.head_q(h).squeeze(-1)
        lrv = self.head_v(h).squeeze(-1)
        return q05, lrv


    def training_step(self, batch, _):
        x, y_ret, y_lrv, _ = batch
        q05, lrv_hat = self(x)
        loss_q = pinball_loss(q05, y_ret, self.hparams.tau)
        loss_v = torch.mean((lrv_hat - y_lrv)**2)
        loss = loss_q + self.hparams.beta * loss_v
        self.log_dict({"train_pinball":loss_q, "train_mse_lrv":loss_v, "train_loss":loss}, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y_ret, y_lrv, idx = batch
        q05, lrv_hat = self(x)
        loss_q = pinball_loss(q05, y_ret, self.hparams.tau)
        loss_v = torch.mean((lrv_hat - y_lrv)**2)
        loss = loss_q + self.hparams.beta * loss_v
        self.log_dict({"val_pinball":loss_q, "val_mse_lrv":loss_v, "val_loss":loss}, prog_bar=True)
        return {"q05": q05.detach(), "lrv_hat": lrv_hat.detach(), "y_ret": y_ret.detach(), "y_lrv": y_lrv.detach(), "idx": idx.detach()}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

def train(npz_path, split_date, out_csv, epochs=80, batch=64, num_workers=2, **h):
    ds_tr = SeqDataset(npz_path, split_date, train=True)
    ds_va = SeqDataset(npz_path, split_date, train=False)
    train_dl = DataLoader(ds_tr, batch_size=batch, shuffle=True, drop_last=True, num_workers=num_workers)
    val_dl   = DataLoader(ds_va, batch_size=batch, shuffle=False, num_workers=num_workers)

    model = PatchTSTMulti(**h)
    ckpt = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)
    trainer = pl.Trainer(max_epochs=epochs, log_every_n_steps=10, callbacks=[ckpt],
                         enable_checkpointing=True, accelerator="auto", devices=1)
    trainer.fit(model, train_dl, val_dl)
    best = ckpt.best_model_path or ckpt.last_model_path
    print("Best ckpt:", best)

    # write validation predictions aligned to dates
    z = np.load(npz_path, allow_pickle=True)
    dates = z["dates"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PatchTSTMulti.load_from_checkpoint(best, **h).to(device)
    model.eval()
    idx_list, q_list, v_list, yret_list = [], [], [], []
    with torch.no_grad():
        for xb, yret, ylv, ib in val_dl:
            xb = xb.to(device)
            q05, lrv_hat = model(xb)
            idx_list.append(ib.numpy())
            q_list.append(q05.cpu().numpy())
            v_list.append(lrv_hat.cpu().numpy())
            yret_list.append(yret.cpu().numpy())
    idx = np.concatenate(idx_list)
    qv  = np.concatenate(q_list).ravel()
    lv  = np.concatenate(v_list).ravel()
    yrv = np.concatenate(yret_list).ravel()
    ord = np.argsort(idx)

    df = pd.DataFrame({
        "date": dates[ds_va.val_idx][ord].astype("datetime64[D]"),
        "ret_true": yrv[ord],
        "q05_ret_pred": qv[ord],
        "sigma2_pred": np.exp(lv[ord])  # back-transform log variance -> variance
    })
    Path("outputs").mkdir(exist_ok=True, parents=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(df)} rows")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--seq_len", type=int, default=120)
    ap.add_argument("--split_date", default="2023-01-02")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    # model hparams
    ap.add_argument("--patch_len", type=int, default=20)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--beta", type=float, default=0.3, help="weight for variance-head MSE")
    args = ap.parse_args()

    npz = f"outputs/{args.symbol.lower()}_seq_{args.seq_len}.npz"
    out_csv = "outputs/patch_preds.csv"
    train(npz, args.split_date, out_csv, epochs=args.epochs, batch=args.batch, num_workers=args.num_workers,
          seq_len=args.seq_len, n_feat=2, patch_len=args.patch_len, d_model=args.d_model,
          nhead=args.nhead, nlayers=args.nlayers, dropout=args.dropout, lr=args.lr,
          tau=args.tau, beta=args.beta)
