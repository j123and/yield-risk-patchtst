#!/usr/bin/env python3
"""
Audit daily realized-volatility proxy (GK) for SPY.

Checks:
- Trading-day completeness vs NYSE calendar (optional, soft dependency).
- Basic stats and top outliers.
- ADF stationarity test on log RV.
- Saves a "clean" Parquet (explicit policy).

Outputs:
  tables/summary.csv
  tables/missing_dates.csv
  tables/top_outliers.csv
  figs/rv_timeseries.png
  figs/rv_hist.png
  data/spy_rv_clean.parquet

Usage:
  python src/audit_rv.py --in data/spy_rv.parquet
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, List
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
DATA_OUT = Path("data/spy_rv_clean.parquet")


def nyse_trading_days(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    if mcal is None:
        # Fallback: just use observed dates as "expected"
        return pd.date_range(start, end, freq="B").to_list()
    cal = mcal.get_calendar("XNYS")
    days = cal.valid_days(start_date=start, end_date=end).tz_localize(None)
    return [pd.Timestamp(d).normalize() for d in days]


def main(in_parquet: str = "data/spy_rv.parquet") -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_parquet)
    # Expect at least: date, RV_GK; if sigma_gk missing, derive from RV_GK
    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column in the input parquet")
    if "RV_GK" not in df.columns:
        raise ValueError("Expected an 'RV_GK' column in the input parquet")

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop=True)
    if "sigma_gk" not in df.columns:
        df["sigma_gk"] = np.sqrt(df["RV_GK"].clip(lower=0))

    # Completeness vs NYSE calendar (soft)
    start, end = df["date"].min().date(), df["date"].max().date()
    expected = pd.Series(nyse_trading_days(pd.Timestamp(start), pd.Timestamp(end)), name="date")
    df_days = pd.Series(df["date"].dt.normalize().unique(), name="date")
    missing = expected[~expected.isin(df_days)]
    pd.DataFrame({"missing_trading_dates": missing}).to_csv(TABLES / "missing_dates.csv", index=False)

    # Basic stats & outliers
    rv = df["RV_GK"].clip(lower=0).replace([np.inf, -np.inf], np.nan).dropna()
    sigma = np.sqrt(rv)
    q99 = rv.quantile(0.99) if len(rv) > 100 else rv.max()
    top = df.loc[rv[rv >= q99].index, ["date", "RV_GK"]].copy()
    top = top.sort_values("RV_GK", ascending=False).head(20)
    top.to_csv(TABLES / "top_outliers.csv", index=False)

    # ADF on log RV (if available)
    pval = np.nan
    if adfuller is not None:
        try:
            pval = float(adfuller(np.log(rv.replace(0, np.nan).dropna()))[1])
        except Exception:
            pval = np.nan

    # Summary
    summary = {
        "rows": int(len(df)),
        "date_start": str(df["date"].min().date()),
        "date_end": str(df["date"].max().date()),
        "missing_trading_days": int(missing.size),
        "rv_mean": float(rv.mean()),
        "rv_median": float(rv.median()),
        "rv_p95": float(rv.quantile(0.95)),
        "adf_pvalue_logrv": float(pval) if not np.isnan(pval) else None,
    }
    pd.DataFrame([summary]).to_csv(TABLES / "summary.csv", index=False)

    # Plots
    plt.figure()
    plt.plot(df["date"], df["RV_GK"])
    plt.title("SPY GK Realized Variance")
    plt.xlabel("Date"); plt.ylabel("RV_GK")
    plt.tight_layout(); plt.savefig(FIGS / "rv_timeseries.png", dpi=140); plt.close()

    plt.figure()
    plt.hist(np.sqrt(rv), bins=50)
    plt.title("SPY Realized Volatility (σ = sqrt(RV_GK))")
    plt.xlabel("σ"); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(FIGS / "rv_hist.png", dpi=140); plt.close()

    # Cleaning policy (explicit, even if NOP)
    df_clean = df.copy()
    df_clean.to_parquet(DATA_OUT, index=False)
    print(f"Wrote {DATA_OUT}")
    print("Audit complete:")
    print(f"  Missing trading days: {int(missing.size)}")
    print(f"  ADF p-value (log RV): {pval if not np.isnan(pval) else 'N/A'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_parquet", default="data/spy_rv.parquet")
    args = ap.parse_args()
    main(args.in_parquet)
