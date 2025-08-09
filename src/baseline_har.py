#!/usr/bin/env python3
"""
HAR-RV baseline with expanding-window forecasts.

- Reads: data/<symbol>_rv_clean.parquet (from Phase 1b)
- Target: realized variance proxy column 'RV_GK'
- Features: log-lag(1), log-weekly-avg(5), log-monthly-avg(22), all lagged by 1 day
- Model: OLS on log(RV), refit each day (expanding window)
- Outputs:
    outputs/har_preds.csv  (date, rv_true, rv_pred)
    outputs/baseline_errors.json (RMSE, QLIKE for holdout)
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
    df["RV"] = df[target_col].astype(float).clip(lower=0)
    # lags and averages (lagged by one day to prevent leakage)
    df["RV_l1"] = df["RV"].shift(1)
    df["RV_w"]  = df["RV"].rolling(5).mean().shift(1)
    df["RV_m"]  = df["RV"].rolling(22).mean().shift(1)

    # log-transform features and target as per HAR literature
    for c in ["RV", "RV_l1", "RV_w", "RV_m"]:
        df[f"log_{c}"] = np.log(df[c] + EPS)

    # keep needed cols
    keep = ["date", "RV", "log_RV_l1", "log_RV_w", "log_RV_m"]
    return df.dropna().loc[:, keep]

def expanding_har_forecast(df_feat: pd.DataFrame, min_train: int) -> pd.DataFrame:
    dates = df_feat["date"].to_numpy()
    y_log = np.log(df_feat["RV"].to_numpy() + EPS)
    X = df_feat[["log_RV_l1", "log_RV_w", "log_RV_m"]].to_numpy()

    preds = []
    for t in range(min_train, len(df_feat)):
        # Force an intercept column on BOTH matrices
        X_train = sm.add_constant(X[:t], has_constant="add")
        y_train = y_log[:t]
        X_next  = sm.add_constant(X[t:t+1], has_constant="add")

        model = sm.OLS(y_train, X_train, missing="drop")
        res = model.fit()
        yhat_log = res.predict(X_next)[0]
        rv_pred  = float(np.exp(yhat_log))  # back-transform

        preds.append((dates[t], float(df_feat["RV"].iloc[t]), rv_pred))

    return pd.DataFrame(preds, columns=["date","rv_true","rv_pred"])

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true)-np.asarray(y_pred))**2)))

def qlike(y_true, y_pred) -> float:
    y_true = np.asarray(y_true) + EPS
    y_pred = np.asarray(y_pred) + EPS
    return float(np.mean(np.log(y_pred) + y_true / y_pred))

def main(symbol: str, holdout_start: str, min_train_days: int):
    Path("outputs").mkdir(parents=True, exist_ok=True)
    in_path = Path("data") / f"{symbol.lower()}_rv_clean.parquet"
    if not in_path.exists():
        raise SystemExit(f"Missing {in_path}. Run Phase 1 first.")

    df = pd.read_parquet(in_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    feat = make_features(df, target_col="RV_GK")
    # expanding forecasts
    preds = expanding_har_forecast(feat, min_train=min_train_days)
    preds.to_csv("outputs/har_preds.csv", index=False)

    # metrics (overall OOS + holdout subset)
    oos_rmse = rmse(preds["rv_true"], preds["rv_pred"])
    oos_qlik = qlike(preds["rv_true"], preds["rv_pred"])

    # holdout
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--holdout_start", default="2023-01-02",
                    help="Date from which to compute holdout metrics")
    ap.add_argument("--min_train_days", type=int, default=500,
                    help="Initial expanding-window length (business days)")
    args = ap.parse_args()
    main(args.symbol, args.holdout_start, args.min_train_days)
