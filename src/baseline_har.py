#!/usr/bin/env python3
"""
HAR-RV baseline with expanding-window forecasts.

- Reads: data/<symbol>_rv_clean.parquet
- Target: log(RV_GK)
- Features (log-HAR): log_RV_{t-1}, mean(log_RV) over t-5..t-1, mean(log_RV) over t-22..t-1
- Model: OLS on log(RV), refit each day (expanding window)
- Outputs:
    outputs/har_preds.csv  (date, rv_true, rv_pred, lrv_pred)
    outputs/baseline_errors.json (RMSE, QLIKE for OOS + holdout)
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

EPS = 1e-12  # to avoid log(0)

def make_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df = df.copy()
    df["RV"] = df[target_col].astype(float).clip(lower=0.0)
    # log-RV once
    df["log_RV"] = np.log(df["RV"] + EPS)

    # daily (lag 1), weekly (mean of logs over 5 prior days), monthly (mean of logs over 22 prior days)
    df["log_RV_l1"] = df["log_RV"].shift(1)
    df["log_RV_w"]  = df["log_RV"].rolling(5).mean().shift(1)   # mean of log(RV)
    df["log_RV_m"]  = df["log_RV"].rolling(22).mean().shift(1)  # mean of log(RV)

    keep = ["date", "RV", "log_RV", "log_RV_l1", "log_RV_w", "log_RV_m"]
    df = df.sort_values("date").dropna(subset=keep).loc[:, keep].reset_index(drop=True)
    return df

def expanding_har_forecast(df_feat: pd.DataFrame, min_train: int) -> pd.DataFrame:
    dates = df_feat["date"].to_numpy()
    y_log = df_feat["log_RV"].to_numpy()
    X = df_feat[["log_RV_l1", "log_RV_w", "log_RV_m"]].to_numpy()

    if len(df_feat) <= min_train:
        raise ValueError(f"Not enough rows for min_train={min_train} (have {len(df_feat)}).")

    preds = []
    for t in range(min_train, len(df_feat)):
        X_train = sm.add_constant(X[:t], has_constant="add")
        y_train = y_log[:t]
        X_next  = sm.add_constant(X[t:t+1], has_constant="add")

        res = sm.OLS(y_train, X_train, missing="drop").fit()
        lrv_hat = float(res.predict(X_next)[0])   # predicted log variance
        rv_pred = float(np.exp(lrv_hat))          # back-transform to variance

        preds.append((dates[t], float(df_feat["RV"].iloc[t]), rv_pred, lrv_hat))

    return pd.DataFrame(preds, columns=["date","rv_true","rv_pred","lrv_pred"])

def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def qlike(y_true, y_pred) -> float:
    # standard variance QLIKE: log σ² + y/σ²
    y_true = np.asarray(y_true, dtype=float) + EPS
    y_pred = np.asarray(y_pred, dtype=float) + EPS
    return float(np.mean(np.log(y_pred) + y_true / y_pred))

def main(symbol: str, holdout_start: str, min_train_days: int):
    Path("outputs").mkdir(parents=True, exist_ok=True)
    in_path = Path("data") / f"{symbol.lower()}_rv_clean.parquet"
    if not in_path.exists():
        raise SystemExit(f"Missing {in_path}. Run ingest/audit first.")

    df = pd.read_parquet(in_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    feat = make_features(df, target_col="RV_GK")

    # expanding forecasts on the full timeline (OOS notion is driven by min_train)
    preds = expanding_har_forecast(feat, min_train=min_train_days)
    preds.to_csv("outputs/har_preds.csv", index=False)

    # metrics (overall OOS + holdout subset)
    oos_rmse = rmse(preds["rv_true"], preds["rv_pred"])
    oos_qlik = qlike(preds["rv_true"], preds["rv_pred"])

    ph = preds[preds["date"] >= pd.to_datetime(holdout_start)]
    ho_rmse = rmse(ph["rv_true"], ph["rv_pred"])
    ho_qlik = qlike(ph["rv_true"], ph["rv_pred"])

    with open("outputs/baseline_errors.json","w") as f:
        json.dump({
            "oos_rmse": oos_rmse,
            "oos_qlike": oos_qlik,
            "holdout_start": holdout_start,
            "holdout_rmse": ho_rmse,
            "holdout_qlike": ho_qlik,
            "min_train_days": min_train_days
        }, f, indent=2)

    print("Saved outputs/har_preds.csv and outputs/baseline_errors.json")
    print(f"OOS   RMSE={oos_rmse:.6g}  QLIKE={oos_qlik:.6g}")
    print(f"HOLD  RMSE={ho_rmse:.6g}  QLIKE={ho_qlik:.6g}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--holdout_start", default="2023-01-02",
                    help="Date from which to compute holdout metrics")
    ap.add_argument("--min_train_days", type=int, default=500,
                    help="Initial expanding-window length (business days)")
    args = ap.parse_args()
    main(args.symbol, args.holdout_start, args.min_train_days)
