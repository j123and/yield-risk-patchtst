#!/usr/bin/env python3
"""
Ingest daily OHLC from Yahoo Finance and compute a daily realized-volatility proxy
using the Garman–Klass (GK) estimator. Saves Parquet under data/<symbol>_rv.parquet.

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
    """Return a DataFrame with plain 'Open','High','Low','Close' columns.
    Handles both single-index and MultiIndex columns from yfinance."""
    if not isinstance(df.columns, pd.MultiIndex):
        # single symbol, plain columns
        need = ["Open", "High", "Low", "Close"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return df[need].copy()

    # MultiIndex: either level 0 = fields or level 0 = ticker
    cols0, cols1 = df.columns.get_level_values(0), df.columns.get_level_values(1)
    if symbol in cols0:
        sub = df[symbol]
        return sub[["Open", "High", "Low", "Close"]].copy()
    if symbol in cols1:
        sub = df.xs(symbol, axis=1, level=1)
        return sub[["Open", "High", "Low", "Close"]].copy()

    # Fallback: flatten names like ('Open','SPY') -> 'Open'
    flat = df.copy()
    flat.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return flat[["Open", "High", "Low", "Close"]].copy()

def gk_variance(ohlc: pd.DataFrame) -> pd.Series:
    """Garman–Klass daily variance estimator from OHLC."""
    # ensure float dtype
    ohlc = ohlc.astype(float)
    # vectorised logs
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
    out = ohlc.copy()
    out["RV_GK"] = gk_variance(ohlc).clip(lower=0)
    out["sigma_gk"] = np.sqrt(out["RV_GK"])

    out = out.rename_axis("date").reset_index()
    out = out.sort_values("date").dropna().reset_index(drop=True)

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
