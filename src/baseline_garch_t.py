#!/usr/bin/env python3
"""
GARCH(1,1)-t baseline with expanding-window 1-step-ahead forecasts.

- Reads: data/<symbol>_rv_clean.parquet (has OHLC; prefer 'adj_close' for returns)
- Builds daily log returns from adjusted close when available
- Model: ConstantMean + GARCH(1,1), Student-t errors (arch)
- Refits every day on an expanding window, predicts next day (true 1-step ahead)
- Outputs:
    outputs/garch_preds.csv  (date, rv_true, rv_pred)
    updates outputs/baseline_errors.json with GARCH metrics
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from arch.univariate import ConstantMean, GARCH, StudentsT

EPS = 1e-12

def qlike(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float) + EPS
    y_pred = np.asarray(y_pred, dtype=float) + EPS
    return float(np.mean(np.log(y_pred) + y_true / y_pred))

def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def make_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)
    # prefer adjusted close if present
    if "adj_close" in out.columns:
        px = out["adj_close"].astype(float)
        src = "adj_close"
    else:
        px = out["Close"].astype(float)
        src = "Close (WARNING: adjusted close not found)"
        print(f"[GARCH] Using {src} for returns.")
    out["ret"] = np.log(px).diff()
    out["RV"] = out["RV_GK"].astype(float).clip(lower=0.0)
    return out.dropna(subset=["ret","RV"])

def expanding_garch_t(df: pd.DataFrame, min_train: int) -> pd.DataFrame:
    dates = df["date"].to_numpy()
    ret = df["ret"].to_numpy()
    rv_true = df["RV"].to_numpy()

    if len(df) <= min_train:
        raise ValueError(f"Not enough rows for min_train={min_train} (have {len(df)}).")

    preds = []
    # loop over forecast origins; each step fits on [0 .. t-1] and forecasts t
    for t in range(min_train, len(df)):
        # scale to percent to help optimizer
        am = ConstantMean(ret[:t] * 100.0)
        am.volatility = GARCH(1, 1)
        am.distribution = StudentsT()
        res = am.fit(disp="off")

        f = res.forecast(horizon=1, reindex=False)
        # arch returns variance in the forecast object; handle shape robustly
        v = f.variance.values
        if np.ndim(v) == 0:
            sigma2_next = float(v)
        elif v.ndim == 1:
            sigma2_next = float(v[-1])
        else:
            sigma2_next = float(v[-1, 0])
        sigma2_next = sigma2_next / (100.0 ** 2)  # back to raw return units

        preds.append((dates[t], float(rv_true[t]), sigma2_next))

    return pd.DataFrame(preds, columns=["date","rv_true","rv_pred"])

def main(symbol: str, holdout_start: str, min_train_days: int):
    Path("outputs").mkdir(parents=True, exist_ok=True)
    in_path = Path("data") / f"{symbol.lower()}_rv_clean.parquet"
    if not in_path.exists():
        raise SystemExit(f"Missing {in_path}. Run ingest/audit first.")
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

    # update/create baseline metrics JSON
    out_json = Path("outputs/baseline_errors.json")
    block = {
        "garch_t": {
            "oos_rmse": oos_rmse, "oos_qlike": oos_qlik,
            "holdout_start": holdout_start,
            "holdout_rmse": ho_rmse, "holdout_qlike": ho_qlik,
            "min_train_days": min_train_days, "refit_step": 1  # daily refit
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
