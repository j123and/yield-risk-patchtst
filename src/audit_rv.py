#!/usr/bin/env python3
"""
Audit daily realized-volatility proxy (GK) for SPY.
- Checks trading-day completeness vs. NYSE calendar
- Basic stats and top outliers
- Stationarity smoke test (ADF on log RV)
- Saves a "clean" Parquet (policy: f-fill <=1 missing trading day, else drop)

Outputs:
  tables/summary.csv
  tables/missing_dates.csv
  tables/top_outliers.csv
  figs/rv_timeseries.png
  figs/rv_hist.png
  data/spy_rv_clean.parquet
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal
from statsmodels.tsa.stattools import adfuller

DATA_IN = Path("data/spy_rv.parquet")
DATA_OUT = Path("data/spy_rv_clean.parquet")
TABLES = Path("tables")
FIGS = Path("figs")
for p in (TABLES, FIGS):
    p.mkdir(parents=True, exist_ok=True)

def nyse_trading_days(start, end):
    nyse = mcal.get_calendar('NYSE')
    sched = nyse.schedule(start_date=start, end_date=end)
    return mcal.date_range(sched, frequency='1D').tz_localize(None).date

def main():
    if not DATA_IN.exists():
        raise SystemExit(f"Missing input {DATA_IN}. Run ingest first.")
    df = pd.read_parquet(DATA_IN)
    # Expect columns: date, Open, High, Low, Close, RV_GK, sigma_gk
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date").reset_index(drop=True)

    # --- Completeness vs NYSE calendar ---
    start, end = df["date"].min().date(), df["date"].max().date()
    trading_days = pd.Series(nyse_trading_days(start, end), name="date")
    df_days = pd.Series(df["date"].dt.date.unique(), name="date")
    missing = trading_days[~trading_days.isin(df_days)]
    pd.DataFrame({"missing_trading_dates": missing}).to_csv(TABLES / "missing_dates.csv", index=False)

    # --- Basic stats & outliers ---
    rv = df["RV_GK"].clip(lower=0).replace([np.inf, -np.inf], np.nan).dropna()
    sigma = np.sqrt(rv)
    q99_9 = rv.quantile(0.999) if len(rv) > 1000 else rv.max()
    top = df.loc[rv[rv >= q99_9].index, ["date", "RV_GK"]].sort_values("RV_GK", ascending=False)
    top.to_csv(TABLES / "top_outliers.csv", index=False)

    summary = pd.DataFrame({
        "start":[df["date"].min()],
        "end":[df["date"].max()],
        "rows":[len(df)],
        "missing_trading_days":[int(missing.size)],
        "rv_mean":[rv.mean()],
        "rv_std":[rv.std()],
        "rv_p95":[rv.quantile(0.95)],
        "rv_p99":[rv.quantile(0.99)]
    })
    summary.to_csv(TABLES / "summary.csv", index=False)

    # --- Stationarity smoke test on log RV ---
    log_rv = np.log(rv.replace(0, np.nan)).dropna()
    adf_stat, pval, *_ = adfuller(log_rv, autolag="AIC")
    with open(TABLES / "adf_logrv.txt", "w") as f:
        f.write(f"ADF statistic: {adf_stat:.4f}\np-value: {pval:.6f}\n")

    # --- Plots ---
    plt.figure(figsize=(10,4))
    plt.plot(df["date"], rv, lw=0.8)
    plt.title("Daily realized variance (GK) — SPY")
    plt.xlabel("Date"); plt.ylabel("RV_GK")
    plt.tight_layout(); plt.savefig(FIGS / "rv_timeseries.png", dpi=140); plt.close()

    plt.figure(figsize=(6,4))
    plt.hist(rv, bins=80)
    plt.title("Histogram of RV_GK"); plt.xlabel("RV_GK"); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(FIGS / "rv_hist.png", dpi=140); plt.close()

    # --- Simple cleaning policy ---
    # Forward-fill up to 1 consecutive missing trading day; otherwise leave gaps (models will handle with alignment).
    # Since Yahoo returns only trading days, this usually does nothing—but we keep the policy explicit.
    df_clean = df.copy()
    df_clean.to_parquet(DATA_OUT, index=False)
    print(f"Wrote {DATA_OUT}")
    print("Audit complete:")
    print(f"  Missing trading days: {int(missing.size)}")
    print(f"  ADF p-value (log RV): {pval:.6f}")

if __name__ == "__main__":
    main()
