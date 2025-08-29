#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import re
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import pandas_market_calendars as mcal
except Exception:
    mcal = None
try:
    from statsmodels.tsa.stattools import adfuller
except Exception:
    adfuller = None

TABLES = Path("tables")
FIGS = Path("figs")

def _infer_symbol_from_path(p: str) -> str:
    m = re.search(r"([A-Za-z0-9]+)_rv\.parquet$", p)
    return (m.group(1).upper() if m else "SPY")

def nyse_trading_days(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    if mcal is None:
        return []
    cal = mcal.get_calendar("XNYS")
    days = cal.valid_days(start_date=start, end_date=end).tz_localize(None)
    return [pd.Timestamp(d).normalize() for d in days]

def main(in_parquet: str = "data/spy_rv.parquet") -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)

    sym = _infer_symbol_from_path(in_parquet)
    df = pd.read_parquet(in_parquet)
    if "date" not in df.columns: raise ValueError("Expected 'date'")
    if "RV_GK" not in df.columns: raise ValueError("Expected 'RV_GK'")

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date")
    df = df[pd.notna(df["RV_GK"])].copy()

    dup_mask = df["date"].duplicated(keep="last")
    dups = df.loc[dup_mask, ["date", "RV_GK"]]
    if not dups.empty:
        dups.to_csv(TABLES / "duplicates.csv", index=False)
    df = df.drop_duplicates(subset="date", keep="last").reset_index(drop=True)

    if "sigma_gk" not in df.columns:
        df["sigma_gk"] = np.sqrt(df["RV_GK"].clip(lower=0.0))
    if "adj_close" not in df.columns and "Close" in df.columns:
        df["adj_close"] = df["Close"]

    start_date = df["date"].min().normalize()
    end_date = df["date"].max().normalize()
    expected_days = nyse_trading_days(start_date, end_date)
    if expected_days:
        expected = pd.Series(expected_days, name="date")
        mode = "XNYS calendar"
    else:
        expected = pd.Series(df["date"].dt.normalize().unique(), name="date")
        mode = "observed-days fallback (no calendar)"

    df_days = pd.Series(df["date"].dt.normalize().unique(), name="date")
    missing = expected[~expected.isin(df_days)].sort_values()
    pd.DataFrame({"missing_trading_dates": missing}).to_csv(TABLES / "missing_dates.csv", index=False)

    rv = df["RV_GK"].clip(lower=0).replace([np.inf, -np.inf], np.nan).dropna()
    sigma = np.sqrt(rv)

    q99 = rv.quantile(0.99) if len(rv) > 100 else rv.max()
    top_idx = rv[rv >= q99].index
    top = df.loc[top_idx, ["date", "RV_GK"]].sort_values("RV_GK", ascending=False).head(20)
    top.to_csv(TABLES / "top_outliers.csv", index=False)

    pval = np.nan
    nobs = int(rv.replace(0, np.nan).dropna().shape[0])
    if adfuller is not None and nobs > 10:
        try:
            pval = float(adfuller(np.log(rv.replace(0, np.nan).dropna()))[1])
        except Exception:
            pval = np.nan

    summary = {
        "symbol": sym,
        "rows_after_clean": int(len(df)),
        "date_start": str(df["date"].min().date()),
        "date_end": str(df["date"].max().date()),
        "missing_trading_days": int(missing.size),
        "missing_mode": mode,
        "duplicates_dropped": int(len(dups)),
        "rv_mean": float(rv.mean()),
        "rv_median": float(rv.median()),
        "rv_p95": float(rv.quantile(0.95)),
        "adf_pvalue_logrv": None if np.isnan(pval) else float(pval),
        "adf_nobs": nobs,
    }
    pd.DataFrame([summary]).to_csv(TABLES / "summary.csv", index=False)

    plt.figure()
    plt.plot(df["date"], df["RV_GK"])
    plt.title(f"{sym} GK Realized Variance")
    plt.xlabel("Date"); plt.ylabel("RV_GK")
    plt.tight_layout(); plt.savefig(FIGS / "rv_timeseries.png", dpi=140); plt.close()

    plt.figure()
    plt.hist(sigma, bins=50)
    plt.title(f"{sym} Realized Volatility (σ = sqrt(RV_GK))")
    plt.xlabel("σ"); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(FIGS / "rv_hist.png", dpi=140); plt.close()

    clean_path = Path("data") / f"{sym.lower()}_rv_clean.parquet"
    df.to_parquet(clean_path, index=False)
    print(f"Wrote {clean_path}")
    print("Audit complete:")
    print(f"  Calendar check mode: {mode}")
    print(f"  Missing trading days: {int(missing.size)}")
    print(f"  Duplicates dropped: {int(len(dups))}")
    print(f"  ADF p-value (log RV): {pval if not np.isnan(pval) else 'N/A'} (n={nobs})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_parquet", default="data/spy_rv.parquet")
    args = ap.parse_args()
    main(args.in_parquet)
