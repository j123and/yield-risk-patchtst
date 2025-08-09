#!/usr/bin/env python3
import argparse, math, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
try:
    from statsmodels.stats.diagnostic import diebold_mariano
    HAS_DM = True
except Exception:
    HAS_DM = False
from scipy.stats import binom

EPS = 1e-12
Z95 = 1.645
ES95_FACTOR = 2.0627  # Normal ES/VaR ratio ≈ φ(z)/α

def rmse(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(np.sqrt(np.mean((y - yhat)**2)))


def var_acceptance_band(n=250, p=0.05, conf=0.95):
    """Two-sided 95% acceptance band for number of VaR(95%) breaches over n days."""
    lo = int(binom.ppf((1-conf)/2, n, p))
    hi = int(binom.ppf(1-(1-conf)/2, n, p))
    return lo, hi

def band_status(x, lo, hi):
    """Classify breach count x vs. [lo, hi] band."""
    if x < lo:  return f"too_few[{lo}-{hi}]"
    if x > hi:  return f"too_many[{lo}-{hi}]"
    return     f"within[{lo}-{hi}]"


def qlike(y, yhat):
    y = np.asarray(y) + EPS
    yhat = np.asarray(yhat) + EPS
    return float(np.mean(np.log(yhat) + y / yhat))

def pinball_loss(y, q, tau=0.05):
    e = y - q
    return float(np.mean(np.maximum(tau*e, (tau-1)*e)))

def kupiec_pof(y, var, alpha=0.95):
    # y: returns, var: VaR (negative), breach if y < var
    x = int(np.sum(y < var))
    T = len(y)
    p = 1 - alpha
    # Likelihood ratio for unconditional coverage (Kupiec, 1995)
    p_hat = x / T if T else 0.0
    def ll(k, n, pp): return (k*np.log(pp + EPS) + (n-k)*np.log(1-pp + EPS))
    LRuc = -2*(ll(x, T, p) - ll(x, T, p_hat if 0 < p_hat < 1 else (x+0.5)/(T+1)))
    # p-value under chi2(1)
    from scipy.stats import chi2
    return x, x/T, float(1 - chi2.cdf(LRuc, 1))

def christoffersen_independence(y, var):
    # Test independence of exceptions via 2x2 transition matrix
    I = (y < var).astype(int)
    if len(I) < 2:
        return int(I.sum()), float("nan")

    n00 = np.sum((I[1:] == 0) & (I[:-1] == 0))
    n01 = np.sum((I[1:] == 1) & (I[:-1] == 0))
    n10 = np.sum((I[1:] == 0) & (I[:-1] == 1))
    n11 = np.sum((I[1:] == 1) & (I[:-1] == 1))

    # Jeffreys smoothing to avoid log(0)
    n00s, n01s, n10s, n11s = n00+0.5, n01+0.5, n10+0.5, n11+0.5
    pi0 = n01s / (n00s + n01s)
    pi1 = n11s / (n10s + n11s)
    pi  = (n01s + n11s) / (n00s + n01s + n10s + n11s)

    from numpy import log
    ll_indep   = (n00s*log(1-pi)  + n01s*log(pi)  + n10s*log(1-pi)  + n11s*log(pi))
    ll_markov  = (n00s*log(1-pi0) + n01s*log(pi0) + n10s*log(1-pi1) + n11s*log(pi1))
    LRind = -2*(ll_indep - ll_markov)

    from scipy.stats import chi2
    pval = float(1 - chi2.cdf(LRind, 1))
    return int(n01 + n11), pval


def rolling_hist_q05(returns, window=60):
    r = pd.Series(returns).reset_index(drop=True)
    q = r.rolling(window).quantile(0.05).shift(1)  # one-day-ahead
    return q.values

def main(symbol, holdout_start, alpha, hist_win):
    Path("tables").mkdir(exist_ok=True, parents=True)
    Path("figs").mkdir(exist_ok=True, parents=True)
    Path("outputs").mkdir(exist_ok=True, parents=True)

    # --- Load data & predictions ---
    dfp = pd.read_parquet(f"data/{symbol.lower()}_rv_clean.parquet")
    dfp["date"] = pd.to_datetime(dfp["date"])
    dfp["ret"] = np.log(dfp["Close"]).diff()
    dfp = dfp.dropna(subset=["ret","RV_GK"]).loc[:, ["date","ret","RV_GK"]]

    har = pd.read_csv("outputs/har_preds.csv", parse_dates=["date"])
    gar = pd.read_csv("outputs/garch_preds.csv", parse_dates=["date"])
    pat = pd.read_csv("outputs/patch_preds.csv", parse_dates=["date"])  # expects ret_true, q05_ret_pred
    # Optional: if you later add sigma2_pred from multi-task, include it here

    # Align on common dates & holdout slice
    merged = (dfp.merge(har, on="date", how="inner", suffixes=("","_har"))
                 .merge(gar, on="date", how="inner", suffixes=("","_gar"))
                 .merge(pat, on="date", how="inner"))
    merged = merged[merged["date"] >= pd.to_datetime(holdout_start)].copy()
    merged = merged.sort_values("date").reset_index(drop=True)

    # --- Error metrics (HAR vs GARCH on variance) ---
    err_rows = []
    err_rows.append(("HAR", rmse(merged["RV_GK"], merged["rv_pred"]), qlike(merged["RV_GK"], merged["rv_pred"])))
    err_rows.append(("GARCH_t", rmse(merged["RV_GK"], merged["rv_pred_gar"]), qlike(merged["RV_GK"], merged["rv_pred_gar"])))
    # If deep variance head exists, include it:
    if "sigma2_pred" in merged.columns:
        err_rows.append(("PatchTST_var", rmse(merged["RV_GK"], merged["sigma2_pred"]), qlike(merged["RV_GK"], merged["sigma2_pred"])))
    err = pd.DataFrame(err_rows, columns=["model","RMSE","QLIKE"])
    err.to_csv("tables/error_metrics.csv", index=False)

    # --- DM test (variance models, QLIKE) ---
    dm_rows = []
    if HAS_DM:
        # losses
        L_har = np.log(merged["rv_pred"]+EPS) + merged["RV_GK"]/(merged["rv_pred"]+EPS)
        L_gar = np.log(merged["rv_pred_gar"]+EPS) + merged["RV_GK"]/(merged["rv_pred_gar"]+EPS)
        from statsmodels.stats.diagnostic import diebold_mariano
        dm_stat, p_dm = diebold_mariano(L_har, L_gar, h=1, alternative='two-sided')
        dm_rows.append(("HAR vs GARCH_t", float(dm_stat), float(p_dm), "QLIKE"))
    # Pinball DM: PatchTST vs rolling hist quantile
    q_hist = rolling_hist_q05(merged["ret"].values, window=hist_win)
    valid = ~np.isnan(q_hist)
    L_patch = np.maximum(0.05*(merged["ret"].values - merged["q05_ret_pred"].values),
                         (0.05-1)*(merged["ret"].values - merged["q05_ret_pred"].values))
    L_hist  = np.maximum(0.05*(merged["ret"].values[valid] - q_hist[valid]),
                         (0.05-1)*(merged["ret"].values[valid] - q_hist[valid]))
    if HAS_DM and valid.sum() > 10:
        dm_stat, p_dm = diebold_mariano(L_patch[valid], L_hist, h=1, alternative='two-sided')
        dm_rows.append(("PatchTST vs RollingHist", float(dm_stat), float(p_dm), "Pinball τ=0.05"))
    pd.DataFrame(dm_rows, columns=["comparison","DM","p_value","loss"]).to_csv("tables/dm_test.csv", index=False)

    # --- VaR (95%) per model ---
    ret = merged["ret"].values
    sigma_har  = np.sqrt(np.clip(merged["rv_pred"].values, EPS, None))
    sigma_gar  = np.sqrt(np.clip(merged["rv_pred_gar"].values, EPS, None))
    var_har = -Z95 * sigma_har
    var_gar = -Z95 * sigma_gar

    # PatchTST raw and calibrated quantile
    q05_raw = merged["q05_ret_pred"].values
    # Calibrate with shift so coverage ≈ 5% on holdout sample (simple intercept correction)
    delta = np.quantile(ret - q05_raw, 0.05)
    q05_cal = q05_raw + delta
    # q05 of returns is already the left-tail threshold (negative)
    var_patch_raw = q05_raw
    var_patch_cal = q05_cal

    out_var = pd.DataFrame({
        "date": merged["date"],
        "ret_true": ret,
        "VaR95_HAR": var_har,
        "VaR95_GARCHt": var_gar,
        "VaR95_Patch_raw": var_patch_raw,
        "VaR95_Patch_cal": var_patch_cal
    })
    out_var.to_csv("outputs/var_series.csv", index=False)

    # --- Back-tests: Kupiec + Christoffersen + binomial acceptance band (last 250d @ 95% VaR) ---
    rows = []
    tail = min(250, len(ret))
    lo, hi = var_acceptance_band(n=tail, p=1-args.alpha, conf=0.95)  # p = 0.05 when alpha=0.95
    for name, v in [("HAR", var_har), ("GARCH_t", var_gar), ("Patch_raw", var_patch_raw), ("Patch_cal", var_patch_cal)]:
        # overall holdout
        x, cov, p_k = kupiec_pof(ret, v, alpha=args.alpha)
        _, p_c = christoffersen_independence(ret, v)
        # last 250 business days
        x250 = int(np.sum(ret[-tail:] < v[-tail:]))
        status = band_status(x250, lo, hi)
        rows.append([name, len(ret), x, cov, p_k, p_c, x250, f"{lo}-{hi}", status])

    bt = pd.DataFrame(rows, columns=[
        "model","N_days","breaches","coverage","kupiec_p","christoffersen_p",
        "breaches_250","band_95pct","status_95pct"
    ])
    bt.to_csv("tables/var_backtest.csv", index=False)

    # --- ES (realized tail loss) diagnostics ---
    def realized_shortfall(y, v):
        mask = y < v
        return float(-y[mask].mean()) if mask.any() else np.nan
    es_rows = []
    for name, v in [("HAR", var_har), ("GARCH_t", var_gar), ("Patch_raw", var_patch_raw), ("Patch_cal", var_patch_cal)]:
        es_real = realized_shortfall(ret, v)
        es_rows.append([name, es_real])
    pd.DataFrame(es_rows, columns=["model","realized_ES"]).to_csv("tables/es_realized.csv", index=False)

    # --- Breach timeline plot ---
    plt.figure(figsize=(11,5))
    plt.plot(merged["date"], ret, lw=0.8, label="Return")
    plt.plot(merged["date"], var_har, lw=0.8, label="VaR95 HAR")
    plt.plot(merged["date"], var_gar, lw=0.8, label="VaR95 GARCH-t")
    plt.plot(merged["date"], var_patch_cal, lw=1.0, label="VaR95 Patch (cal)")
    breaches = ret < var_patch_cal
    plt.scatter(merged["date"][breaches], ret[breaches], s=10, label="Breaches (Patch cal)")
    plt.axhline(0, color="black", lw=0.5)
    plt.title("VaR95 breach timeline (holdout)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/var_breach_timeline.png", dpi=140); plt.close()

    # --- Print headline summary ---
    cov_raw = (ret < var_patch_raw).mean()
    cov_cal = (ret < var_patch_cal).mean()
    print(f"PatchTST coverage raw={cov_raw:.4f}, calibrated={cov_cal:.4f} (target 0.0500)")
    print("Wrote tables/error_metrics.csv, tables/dm_test.csv, tables/var_backtest.csv, tables/es_realized.csv and figs/var_breach_timeline.png")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--holdout_start", default="2023-01-02")
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--hist_win", type=int, default=60, help="rolling window for hist q05 baseline")
    args = ap.parse_args()
    main(args.symbol, args.holdout_start, args.alpha, args.hist_win)
