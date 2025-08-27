#!/usr/bin/env python3
"""
Build leak-safe supervised sequences for Value at Risk (VaR) / variance modeling.

Inputs (from audit step):
  data/<symbol>_rv_clean.parquet  with at least:
    - date (unique, increasing)
    - Close (adjusted close preferred) or adj_close
    - sigma_gk (sqrt of Garman–Klass realized variance)
    - RV_GK

Outputs:
  outputs/<symbol>_seq_<seq_len>.npz with:
    - X: (N, T, F) float32, features over past T days
    - y_ret: (N,) float32, next-day log return
    - y_lrv: (N,) float32, next-day log(RV_GK) clamped at 1e-12
    - dates: (N,) datetime64[D], target-day dates
    - feature_names: array[str] giving column order in X
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

CLAMP_RV = 1e-12


def _assert_monotonic_unique_dates(df: pd.DataFrame) -> None:
    if "date" not in df.columns:
        raise ValueError("missing required 'date' column")
    if not df["date"].is_monotonic_increasing:
        raise ValueError("date column must be strictly increasing (sort before building).")
    if df["date"].duplicated().any():
        dups = df.loc[df["date"].duplicated(), "date"].astype(str).tolist()
        raise ValueError(f"duplicate dates found (clean the data first): {dups[:5]}...")


def _choose_close_col(df: pd.DataFrame, allow_raw: bool) -> str:
    """
    Prefer adjusted close for returns; allow raw Close only if the flag is set.
    """
    if "adj_close" in df.columns:
        return "adj_close"
    if "Close" in df.columns and allow_raw:
        return "Close"
    raise ValueError(
        "Adjusted close not found. Provide 'adj_close' in the clean parquet "
        "or rerun with --allow_raw_close if you intentionally want raw Close."
    )


def make_windows(df: pd.DataFrame, seq_len: int, close_col: str):
    """
    Build sliding windows of features and next-day targets.

    Returns (EXACTLY four items):
      X      : (N, seq_len, 2) with features [log return, sigma_gk]
      y_ret  : (N,)            next-day log return
      y_lrv  : (N,)            next-day log realized variance (log RV_GK)
      meta   : dict            {"target_dates": [...], "feature_names": ["ret","sigma_gk"]}

    Notes:
      - We avoid leakage by using only past seq_len days to predict the next day.
      - We DO NOT drop the first NaN from the return diff globally; instead we start
        our first window at index 1, so all windows contain valid returns.
    """
    need = {"date", close_col, "sigma_gk", "RV_GK"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    n = len(df)
    if n < seq_len + 2:
        # not enough rows to form any window + one-step target
        emptyX = np.empty((0, seq_len, 2), dtype=np.float32)
        empty = np.empty((0,), dtype=np.float32)
        return emptyX, empty, empty, {"target_dates": [], "feature_names": ["ret", "sigma_gk"]}

    # Compute daily log returns
    r = np.log(df[close_col].to_numpy(dtype=np.float64))
    r = np.diff(r, prepend=np.nan)  # r[0] is NaN by construction

    s = df["sigma_gk"].to_numpy(dtype=np.float64)
    lrv = np.log(np.maximum(df["RV_GK"].to_numpy(dtype=np.float64), CLAMP_RV))
    dates = pd.to_datetime(df["date"]).dt.tz_localize(None).to_numpy(dtype="datetime64[D]")

    # N = n - seq_len - 1 gives windows i = seq_len .. n-2, target day i+1
    N = n - seq_len - 1
    X = np.empty((N, seq_len, 2), dtype=np.float32)
    y_ret = np.empty(N, dtype=np.float32)
    y_lrv = np.empty(N, dtype=np.float32)
    target_dates = []

    # first window starts at indices 1..seq_len (avoid r[0]==NaN)
    for k, i in enumerate(range(seq_len, n - 1)):
        sl = slice(i - seq_len + 1, i + 1)  # inclusive of i
        X[k, :, 0] = r[sl]
        X[k, :, 1] = s[sl]
        y_ret[k] = r[i + 1]
        y_lrv[k] = lrv[i + 1]
        target_dates.append(dates[i + 1])

    meta = {"target_dates": target_dates, "feature_names": ["ret", "sigma_gk"]}
    return X, y_ret, y_lrv, meta


def main(symbol: str, seq_len: int, allow_raw_close: bool):
    Path("outputs").mkdir(parents=True, exist_ok=True)
    in_path = Path("data") / f"{symbol.lower()}_rv_clean.parquet"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing {in_path}. Run the ingest/audit steps first.")

    df = pd.read_parquet(in_path)
    df = df.sort_values("date").reset_index(drop=True)
    _assert_monotonic_unique_dates(df)

    close_col = _choose_close_col(df, allow_raw=allow_raw_close)

    X, y_ret, y_lrv, meta = make_windows(df, seq_len, close_col)
    dates = np.asarray(meta["target_dates"], dtype="datetime64[D]")
    feature_names = np.asarray(meta["feature_names"], dtype="U")

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
                    help="Allow using raw 'Close' for returns if 'adj_close' is missing.")
    args = ap.parse_args()
    main(args.symbol, args.seq_len, args.allow_raw_close)
