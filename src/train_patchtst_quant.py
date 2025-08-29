#!/usr/bin/env python3
import argparse, os, random
from pathlib import Path
import numpy as np, pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
import pytorch_lightning as pl

def seed_everything(seed=1337):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

def pinball_loss(yhat, y, tau=0.05):
    e = y - yhat
    return torch.where(e >= 0, tau * e, (tau - 1.0) * e).mean()

class SeqDataset(Dataset):
    def __init__(self, npz_path, split_date, train, seq_len=None):
        z = np.load(npz_path, allow_pickle=True)
        self.X = z["X"].astype(np.float32)
        self.y_ret = z["y_ret"].astype(np.float32)
        self.y_lrv = z["y_lrv"].astype(np.float32)
        self.dates = z["dates"].astype("datetime64[D]")
        if seq_len is not None and self.X.shape[1] != seq_len:
            raise ValueError(f"seq_len mismatch: data {self.X.shape[1]} vs arg {seq_len}")
        if split_date is None:
            cut = int(0.8 * len(self.X)); idx_tr = np.arange(cut); idx_te = np.arange(cut, len(self.X))
        else:
            d = np.datetime64(split_date)
            idx_tr = np.where(self.dates < d)[0]
            idx_te = np.where(self.dates >= d)[0]
            if idx_tr.size == 0 or idx_te.size == 0:
                raise ValueError("Empty train or test after time split.")
            if self.dates[idx_tr].max() >= self.dates[idx_te].min():
                raise ValueError("Train/test overlap; bad split.")
        self.idx_train = idx_tr
        self.idx_test  = idx_te
        self.use_train = train
    def __len__(self): return len(self.idx_train) if self.use_train else len(self.idx_test)
    def __getitem__(self, i):
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

class PatchTSTQuant(pl.LightningModule):
    def __init__(self, seq_len=120, n_feat=2, patch_len=20,
                 d_model=64, nhead=4, nlayers=2, dropout=0.1, lr=1e-3, tau=0.05):
        super().__init__()
        if seq_len % patch_len != 0:
            raise ValueError("seq_len must be divisible by patch_len.")
        self.tau = tau; self.lr = lr
        self.n_patches = seq_len // patch_len
        self.patch_len = patch_len

        # target scaler
        self.register_buffer("y_mu", torch.tensor(0.0), persistent=False)
        self.register_buffer("y_sd", torch.tensor(1.0), persistent=False)
        # input scaler
        self.register_buffer("x_mu", torch.zeros(n_feat), persistent=False)
        self.register_buffer("x_sd", torch.ones(n_feat),  persistent=False)

        self.proj = nn.Linear(n_feat * patch_len, d_model)
        self.pos  = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.normal_(self.pos, 0.0, 0.02)

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head_q  = nn.Linear(d_model, 1)

    def set_target_scaler(self, mu: float, sd: float):
        sd = float(sd) if sd > 0 else 1.0
        self.y_mu.data = torch.tensor(mu, dtype=torch.float32, device=self.y_mu.device)
        self.y_sd.data = torch.tensor(sd, dtype=torch.float32, device=self.y_sd.device)

    def set_input_scaler(self, mu_vec: np.ndarray, sd_vec: np.ndarray):
        sd_vec = np.where(sd_vec <= 0, 1.0, sd_vec)
        self.x_mu.data = torch.tensor(mu_vec, dtype=torch.float32, device=self.x_mu.device)
        self.x_sd.data = torch.tensor(sd_vec, dtype=torch.float32, device=self.x_sd.device)

    def _scale_y(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.y_mu) / self.y_sd
    def _unscale_y(self, y_scaled: torch.Tensor) -> torch.Tensor:
        return y_scaled * self.y_sd + self.y_mu

    def forward(self, x):
        x = (x - self.x_mu) / self.x_sd
        B, T, F = x.shape
        x = x.reshape(B, self.n_patches, self.patch_len * F)
        x = self.proj(x) + self.pos
        z = self.encoder(x).mean(dim=1)
        q = self.head_q(z).squeeze(-1)
        return q

    def training_step(self, batch, _):
        X, y_ret, _, _ = batch
        yhat_scaled = self(X)
        y_scaled = self._scale_y(y_ret)
        loss = pinball_loss(yhat_scaled, y_scaled, self.tau)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

def train(npz_path, split_date, out_csv, epochs=10, batch=64, seed=1337,
          seq_len=120, n_feat=2, patch_len=20, d_model=64, nhead=4, nlayers=2,
          dropout=0.1, lr=1e-3, tau=0.05):
    seed_everything(seed)

    ds_all_tr = SeqDataset(npz_path, split_date, train=True,  seq_len=seq_len)
    ds_te     = SeqDataset(npz_path, split_date, train=False, seq_len=seq_len)

    ntr = len(ds_all_tr.idx_train)
    cut = max(1, int(0.9 * ntr))
    ds_tr  = Subset(ds_all_tr, list(range(0, cut)))
    ds_val = Subset(ds_all_tr, list(range(cut, ntr)))

    dl_tr  = DataLoader(ds_tr,  batch_size=batch, shuffle=True,  num_workers=0, collate_fn=collate_train)
    dl_val = DataLoader(ds_val, batch_size=batch, shuffle=False, num_workers=0, collate_fn=collate_train)
    dl_te  = DataLoader(ds_te,  batch_size=batch, shuffle=False, num_workers=0, collate_fn=collate_with_dates)

    model = PatchTSTQuant(seq_len=seq_len, n_feat=2, patch_len=patch_len,
                          d_model=d_model, nhead=nhead, nlayers=nlayers,
                          dropout=dropout, lr=lr, tau=tau)

    # target scaling (returns)
    y_train = torch.as_tensor(ds_all_tr.y_ret[ds_all_tr.idx_train[:cut]], dtype=torch.float32)
    y_mu = float(y_train.mean().item())
    y_sd = float(y_train.std(unbiased=False).item() + 1e-12)
    model.set_target_scaler(y_mu, y_sd)
    print(f"[standardize y] mu={y_mu:.6g}, sd={y_sd:.6g}")

    # input scaling (features)
    X_tr = ds_all_tr.X[ds_all_tr.idx_train[:cut]]
    x_mu = X_tr.mean(axis=(0, 1))
    x_sd = X_tr.std(axis=(0, 1)) + 1e-12
    model.set_input_scaler(x_mu, x_sd)
    print(f"[standardize X] mu={x_mu.tolist()}, sd={x_sd.tolist()}")

    # bias-init at empirical τ in scaled space
    q_emp_unscaled = float(np.quantile(y_train.numpy(), tau))
    q_emp_scaled = (q_emp_unscaled - y_mu) / y_sd
    with torch.no_grad():
        model.head_q.bias.fill_(q_emp_scaled)

    trainer = pl.Trainer(max_epochs=epochs, log_every_n_steps=10,
                         enable_checkpointing=False, enable_model_summary=False)
    trainer.fit(model, dl_tr, dl_val)

    # predict
    model.eval()
    preds, dates, ytrue = [], [], []
    with torch.no_grad():
        for X, y_ret, _y_lrv, d in dl_te:
            q_scaled = model(X)
            q = model._unscale_y(q_scaled)
            preds += q.cpu().numpy().tolist()
            dates += d
            ytrue += y_ret.cpu().numpy().tolist()

    q_arr = np.asarray(preds, dtype=np.float64)
    if float(np.std(q_arr)) < 1e-4:
        print("WARNING: holdout q05_ret_pred looks nearly constant; check input scaling and capacity.")

    df = pd.DataFrame({
        "date": np.array(dates, dtype="datetime64[D]"),
        "ret_true": ytrue,
        "q05_ret_pred": preds,
        "sigma2_pred": np.nan,
    })
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} (rows={len(df)})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="outputs/spy_seq_120.npz")
    ap.add_argument("--split_date", default=None)
    ap.add_argument("--out_csv", default="outputs/patch_quant.csv")
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
