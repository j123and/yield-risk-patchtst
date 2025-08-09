#!/usr/bin/env python3
"""
GARCH(1,1)-t baseline with periodic refits (expanding window).

- Reads: data/<symbol>_rv_clean.parquet (has OHLC)
- Builds daily log returns from Close prices
- Model: constant mean + GARCH(1,1), Student-t errors
- Refits every REFIT_STEP days to approximate expanding window
- Outputs:
    outputs/garch_preds.csv  (date, rv_true, rv_pred)
    updates baseline_errors.json with GARCH metrics
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from arch.univariate import ConstantMean, GARCH, StudentsT

EPS = 1e-12
REFIT_STEP = 20  # refit every 20 trading days to save time

def qlike(y_true, y_pred) -> float:
    y_true = np.asarray(y_true) + EPS
    y_pred = np.asarray(y_pred) + EPS
    return float(np.mean(np.log(y_pred) + y_true / y_pred))

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true)-np.asarray(y_pred))**2)))

def make_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)
    out["ret"] = np.log(out["Close"]).diff()
    # realised variance target from GK (already computed)
    out["RV"]  = out["RV_GK"].clip(lower=0)
    return out.dropna(subset=["ret","RV"])

def expanding_garch_t(df: pd.DataFrame, min_train: int) -> pd.DataFrame:
    dates = df["date"].to_numpy()
    ret = df["ret"].to_numpy()
    rv_true = df["RV"].to_numpy()

    preds = []
    t = min_train
    while t < len(df):
        # fit on 0..t-1
        am = ConstantMean(ret[:t] * 100)  # scale to percent to help optimizer
        am.volatility = GARCH(1,1)
        am.distribution = StudentsT()
        res = am.fit(disp="off")

        # forecast next REFIT_STEP days (or until end)
        horizon = min(REFIT_STEP, len(df) - t)
        f = res.forecast(horizon=horizon, reindex=False)
        # one-step-ahead variance forecasts (in percent^2)
        sigma2 = f.variance.values[-1] / (100**2)  # back to raw units

        for i in range(horizon):
            idx = t + i
            # align: predict variance for day idx using info up to idx-1
            rv_pred = float(sigma2[i])
            preds.append((dates[idx], float(rv_true[idx]), rv_pred))
        t += horizon

    return pd.DataFrame(preds, columns=["date","rv_true","rv_pred"])

def main(symbol: str, holdout_start: str, min_train_days: int):
    Path("outputs").mkdir(parents=True, exist_ok=True)
    in_path = Path("data") / f"{symbol.lower()}_rv_clean.parquet"
    if not in_path.exists():
        raise SystemExit(f"Missing {in_path}. Run Phase 1 first.")
    df = pd.read_parquet(in_path)
    df = make_returns(df)

    preds = expanding_garch_t(df, min_train=min_train_days)
    preds.to_csv("outputs/garch_preds.csv", index=False)

    # metrics
    oos_rmse = rmse(preds["rv_true"], preds["rv_pred"])
    oos_qlik = qlike(preds["rv_true"], preds["rv_pred"])

    ph = preds[preds["date"] >= pd.to_datetime(holdout_start)]
    ho_rmse = rmse(ph["rv_true"], ph["rv_pred"])
    ho_qlik = qlike(ph["rv_true"], ph["rv_pred"])

    # update (or create) the baseline metrics JSON
    out_json = Path("outputs/baseline_errors.json")
    block = {
        "garch_t": {
            "oos_rmse": oos_rmse, "oos_qlike": oos_qlik,
            "holdout_start": holdout_start,
            "holdout_rmse": ho_rmse, "holdout_qlike": ho_qlik,
            "min_train_days": min_train_days, "refit_step": REFIT_STEP
        }
    }
    if out_json.exists():
        with open(out_json) as f: base = json.load(f)
    else:
        base = {}
    base.update(block)
    with open(out_json, "w") as f:
        json.dump(base, f, indent=2)

    print("Saved outputs/garch_preds.csv and updated outputs/baseline_errors.json")
    print(f"OOS   RMSE={oos_rmse:.6g}  QLIKE={oos_qlik:.6g}")
    print(f"HOLD  RMSE={ho_rmse:.6g}  QLIKE={ho_qlik:.6g}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--holdout_start", default="2023-01-02")
    ap.add_argument("--min_train_days", type=int, default=500)
    args = ap.parse_args()
    main(args.symbol, args.holdout_start, args.min_train_days)
