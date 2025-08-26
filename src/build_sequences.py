#!/usr/bin/env python3
"""
Build leak-safe supervised sequences for VaR/variance modeling.

Inputs (from audit step):
  data/<symbol>_rv_clean.parquet  with at least:
    - date (unique, increasing)
    - Close (adjusted close preferred) or adj_close
    - sigma_gk (sqrt of GK RV)
    - RV_GK

Outputs:
  outputs/<symbol>_seq_<seq_len>.npz with:
    - X: (N, T, F) float32, features over past T days
    - y_ret: (N,) float32, next-day log return
    - y_lrv: (N,) float32, next-day log(RV_GK) with clamp at 1e-12
    - dates: (N,) datetime64[D], target day dates
    - feature_names: array of dtype '<U...' listing column order in X
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

CLAMP_RV = 1e-12

def _assert_monotonic_unique_dates(df: pd.DataFrame) -> None:
    if not df["date"].is_monotonic_increasing:
        raise ValueError("date column must be strictly increasing (sort before building).")
    if df["date"].duplicated().any():
        dups = df.loc[df["date"].duplicated(), "date"].astype(str).tolist()
        raise ValueError(f"duplicate dates found (clean the data first): {dups[:5]}...")

def _choose_close_col(df: pd.DataFrame, allow_raw: bool) -> str:
    # Prefer explicit adj_close for returns; else accept Close only if allow_raw=True
    if "adj_close" in df.columns:
        return "adj_close"
    if "Close" in df.columns and allow_raw:
        return "Close"
    raise ValueError(
        "Adjusted close not found. Provide 'adj_close' in the clean parquet "
        "or rerun with --allow_raw_close if you intentionally want raw Close."
    )

def make_windows(df: pd.DataFrame, seq_len: int, close_col: str):
    # Features: past log returns (from chosen close) and sigma_gk
    feat_cols = ["ret", "sigma_gk"]
    X_list, y_ret, y_lrv, dates = [], [], [], []

    # past returns
    df = df.copy()
    df["ret"] = np.log(df[close_col]).diff()
    df = df.dropna(subset=["ret", "RV_GK", "sigma_gk"])

    for i in range(seq_len, len(df) - 1):
        sl = df.iloc[i - seq_len : i]   # past T days
        nxt = df.iloc[i]                # target day
        X_list.append(sl[["ret", "sigma_gk"]].to_numpy(dtype=np.float32))
        y_ret.append(np.float32(nxt["ret"]))
        y_lrv.append(np.float32(np.log(max(float(nxt["RV_GK"]), CLAMP_RV))))
        dates.append(pd.Timestamp(nxt["date"]))

    if not X_list:
        raise ValueError("Not enough rows to build any sequences; reduce --seq_len or check data.")

    X = np.stack(X_list, axis=0)  # (N, T, F)
    y_ret = np.asarray(y_ret, dtype=np.float32)
    y_lrv = np.asarray(y_lrv, dtype=np.float32)
    dates = np.asarray(dates, dtype="datetime64[D]")

    return X, y_ret, y_lrv, dates, np.array(feat_cols, dtype="U")

def main(symbol: str, seq_len: int, allow_raw_close: bool):
    Path("outputs").mkdir(parents=True, exist_ok=True)
    in_path = Path("data") / f"{symbol.lower()}_rv_clean.parquet"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing {in_path}. Run the ingest/audit steps first.")

    df = pd.read_parquet(in_path)
    df = df.sort_values("date").reset_index(drop=True)
    _assert_monotonic_unique_dates(df)

    close_col = _choose_close_col(df, allow_raw=allow_raw_close)

    X, y_ret, y_lrv, dates, feature_names = make_windows(df, seq_len, close_col)

    out_path = Path("outputs") / f"{symbol.lower()}_seq_{seq_len}.npz"
    np.savez_compressed(
        out_path,
        X=X, y_ret=y_ret, y_lrv=y_lrv, dates=dates, feature_names=feature_names
    )
    print(f"Wrote {out_path}  →  X{X.shape}, y_ret{y_ret.shape}, y_lrv{y_lrv.shape}")
    print(f"Features order: {list(feature_names)} | close column: {close_col}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--seq_len", type=int, default=120)
    ap.add_argument("--allow_raw_close", action="store_true",
                    help="Allow using raw 'Close' for returns if 'adj_close' is missing (not recommended for SPY).")
    args = ap.parse_args()
    main(args.symbol, args.seq_len, args.allow_raw_close)
