#!/usr/bin/env python3
"""
Ingest daily OHLC from Yahoo Finance and compute a daily realized-volatility proxy
using the Garman–Klass (GK) estimator. Saves Parquet under data/<symbol>_rv.parquet.

Important:
- yfinance(auto_adjust=True) returns OHLC already adjusted for splits/dividends.
  GK uses same-day ratios (H/L, C/O) so it is invariant to that scaling.
- The saved 'Close' is adjusted. We also write an 'adj_close' mirror for clarity.

Usage:
  python src/ingest_yahoo_gk.py --symbol SPY --start 2015-01-02 --end 2025-07-31
"""
import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

GK_K = 2 * math.log(2) - 1  # ≈ 0.38629436112

def select_ohlc(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Return a DataFrame with 'Open','High','Low','Close' (float)."""
    if not isinstance(df.columns, pd.MultiIndex):
        need = ["Open", "High", "Low", "Close"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        out = df[need].copy()
    else:
        cols0, cols1 = df.columns.get_level_values(0), df.columns.get_level_values(1)
        if symbol in cols0:
            out = df[symbol][["Open", "High", "Low", "Close"]].copy()
        elif symbol in cols1:
            out = df.xs(symbol, axis=1, level=1)[["Open", "High", "Low", "Close"]].copy()
        else:
            flat = df.copy()
            flat.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            out = flat[["Open", "High", "Low", "Close"]].copy()
    return out.astype(float)

def gk_variance(ohlc: pd.DataFrame) -> pd.Series:
    """Garman–Klass daily variance estimator from OHLC (natural logs)."""
    log_hl = np.log(ohlc["High"].to_numpy() / ohlc["Low"].to_numpy())
    log_co = np.log(ohlc["Close"].to_numpy() / ohlc["Open"].to_numpy())
    var = 0.5 * (log_hl ** 2) - GK_K * (log_co ** 2)
    return pd.Series(var, index=ohlc.index, name="RV_GK")

def main(symbol: str, start: str, end: str):
    Path("data").mkdir(parents=True, exist_ok=True)

    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise SystemExit(f"No data returned for {symbol}. Check ticker or dates.")

    ohlc = select_ohlc(df, symbol)

    # Hard data sanity: GK requires strictly positive prices
    if (ohlc <= 0).any().any():
        bad = ohlc[(ohlc <= 0).any(axis=1)]
        raise SystemExit(f"Non-positive OHLC encountered on rows:\n{bad}")

    out = ohlc.copy()
    out["RV_GK"] = gk_variance(ohlc).clip(lower=0.0)
    out["sigma_gk"] = np.sqrt(out["RV_GK"])

    # Mirror adjusted close explicitly for downstream code clarity
    out["adj_close"] = out["Close"]

    # Index → column, sort, drop NaNs, drop duplicate dates if any
    out = out.rename_axis("date").reset_index().sort_values("date")
    out = out.dropna().drop_duplicates(subset="date", keep="last").reset_index(drop=True)

    out_path = Path("data") / f"{symbol.lower()}_rv.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} with {len(out)} rows")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--start", default="2015-01-02")
    ap.add_argument("--end",   default="2025-07-31")
    args = ap.parse_args()
    main(args.symbol, args.start, args.end)
