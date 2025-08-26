#!/usr/bin/env python3
"""
Audit daily realized-volatility proxy (GK).

Checks:
- Trading-day completeness vs NYSE calendar (or observed-days fallback).
- Basic stats and top outliers.
- ADF stationarity test on log RV.
- Saves a "clean" Parquet with an explicit minimal policy (sort, dropna, dedup-by-date).

Outputs:
  tables/summary.csv
  tables/missing_dates.csv
  tables/top_outliers.csv
  tables/duplicates.csv
  figs/rv_timeseries.png
  figs/rv_hist.png
  data/<symbol>_rv_clean.parquet

Usage:
  python src/audit_rv.py --in data/spy_rv.parquet
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import pandas_market_calendars as mcal  # optional
except Exception:  # pragma: no cover
    mcal = None

try:
    from statsmodels.tsa.stattools import adfuller  # optional
except Exception:  # pragma: no cover
    adfuller = None

TABLES = Path("tables")
FIGS = Path("figs")

def _infer_symbol_from_path(p: str) -> str:
    # try to extract ticker from ".../<symbol>_rv.parquet"
    m = re.search(r"([A-Za-z0-9]+)_rv\.parquet$", p)
    return (m.group(1).upper() if m else "SPY")

def nyse_trading_days(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    """Return expected NYSE trading days [start, end].
    If calendar not available, return an EMPTY list to signal fallback mode.
    """
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

    # Expect at least: date, RV_GK; if sigma_gk missing, derive from RV_GK
    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column in the input parquet")
    if "RV_GK" not in df.columns:
        raise ValueError("Expected an 'RV_GK' column in the input parquet")

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    # Minimal cleaning policy (explicit):
    # 1) sort; 2) drop rows with NaN RV_GK; 3) deduplicate by date (keep last)
    df = df.sort_values("date")
    df = df[pd.notna(df["RV_GK"])].copy()

    # duplicate detection
    dup_mask = df["date"].duplicated(keep="last")
    dups = df.loc[dup_mask, ["date", "RV_GK"]].copy()
    if not dups.empty:
        dups.to_csv(TABLES / "duplicates.csv", index=False)

    df = df.drop_duplicates(subset="date", keep="last").reset_index(drop=True)

    # Completeness vs NYSE calendar
    start_date = df["date"].min().normalize()
    end_date = df["date"].max().normalize()

    expected_days = nyse_trading_days(start_date, end_date)
    if expected_days:
        expected = pd.Series(expected_days, name="date")
        mode = "XNYS calendar"
    else:
        # Fallback: use observed trading days as the expected set (no holiday false-positives)
        expected = pd.Series(df["date"].dt.normalize().unique(), name="date")
        mode = "observed-days fallback (no calendar)"

    df_days = pd.Series(df["date"].dt.normalize().unique(), name="date")
    missing = expected[~expected.isin(df_days)].sort_values()

    pd.DataFrame({"missing_trading_dates": missing}).to_csv(TABLES / "missing_dates.csv", index=False)

    # RV series for stats/plots
    rv = df["RV_GK"].clip(lower=0).replace([np.inf, -np.inf], np.nan).dropna()
    sigma = np.sqrt(rv)

    # Outliers (top 20 by RV)
    q99 = rv.quantile(0.99) if len(rv) > 100 else rv.max()
    top_idx = rv[rv >= q99].index
    top = df.loc[top_idx, ["date", "RV_GK"]].sort_values("RV_GK", ascending=False).head(20)
    top.to_csv(TABLES / "top_outliers.csv", index=False)

    # ADF on log RV
    pval = np.nan
    nobs = int(rv.replace(0, np.nan).dropna().shape[0])
    if adfuller is not None and nobs > 10:
        try:
            pval = float(adfuller(np.log(rv.replace(0, np.nan).dropna()))[1])
        except Exception:
            pval = np.nan

    # Summary
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

    # Plots
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

    # Save clean parquet next to raw, symbol-aware name
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
