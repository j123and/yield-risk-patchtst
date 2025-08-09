#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np, pandas as pd

def make_windows(df, seq_len=120):
    # features: past returns and sigma_gk (vol proxy)
    X = []
    y_ret = []   # next-day return (for quantile VaR)
    y_lrv = []   # next-day log RV (for MSE/QLIKE comparisons)
    dates = []
    for i in range(seq_len, len(df)-1):
        sl = df.iloc[i-seq_len:i]
        nxt = df.iloc[i]        # predict for this row
        X.append(sl[["ret","sigma_gk"]].to_numpy(dtype=np.float32))
        y_ret.append(float(nxt["ret"]))
        y_lrv.append(float(np.log(max(nxt["RV_GK"], 1e-12))))
        dates.append(pd.Timestamp(nxt["date"]))
    X = np.stack(X)               # (N, T, F)
    y_ret = np.array(y_ret, dtype=np.float32)
    y_lrv = np.array(y_lrv, dtype=np.float32)
    dates = np.array(dates)
    return X, y_ret, y_lrv, dates

def main(symbol, seq_len):
    Path("outputs").mkdir(exist_ok=True, parents=True)
    df = pd.read_parquet(f"data/{symbol.lower()}_rv_clean.parquet")
    df = df.sort_values("date").reset_index(drop=True)
    df["ret"] = np.log(df["Close"]).diff()          # daily log return
    df = df.dropna(subset=["ret","RV_GK","sigma_gk"])

    X, y_ret, y_lrv, dates = make_windows(df, seq_len)
    np.savez_compressed(f"outputs/{symbol.lower()}_seq_{seq_len}.npz",
                        X=X, y_ret=y_ret, y_lrv=y_lrv, dates=dates.astype('datetime64[D]'))
    print(f"Wrote outputs/{symbol.lower()}_seq_{seq_len}.npz  →  X{X.shape}, y_ret{y_ret.shape}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--seq_len", type=int, default=120)
    args = ap.parse_args()
    main(args.symbol, args.seq_len)
