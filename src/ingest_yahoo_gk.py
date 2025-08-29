#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

GK_K = 2 * math.log(2) - 1.0

def select_ohlc(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
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
            flat = df.copy(); flat.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            out = flat[["Open", "High", "Low", "Close"]].copy()
    return out.astype(float)

def gk_variance(ohlc: pd.DataFrame) -> pd.Series:
    log_hl = np.log(ohlc["High"].to_numpy() / ohlc["Low"].to_numpy())
    log_co = np.log(ohlc["Close"].to_numpy() / ohlc["Open"].to_numpy())
    var = 0.5 * (log_hl ** 2) - GK_K * (log_co ** 2)
    return pd.Series(var, index=ohlc.index, name="RV_GK")

def main(symbol: str, start: str, end: str):
    Path("data").mkdir(parents=True, exist_ok=True)
    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty: raise SystemExit(f"No data returned for {symbol}.")
    ohlc = select_ohlc(df, symbol)
    if (ohlc <= 0).any().any():
        bad = ohlc[(ohlc <= 0).any(axis=1)]
        raise SystemExit(f"Non-positive OHLC encountered:\n{bad}")

    out = ohlc.copy()
    out["RV_GK"] = gk_variance(ohlc).clip(lower=0.0)
    out["sigma_gk"] = np.sqrt(out["RV_GK"])
    out["adj_close"] = out["Close"]

    out = out.rename_axis("date").reset_index().sort_values("date")
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
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
