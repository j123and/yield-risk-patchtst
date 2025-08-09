#!/usr/bin/env python3
import argparse, math, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

EPS = 1e-12
Z95 = 1.645

# ------------------------- Metrics & tests -------------------------

def rmse(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(np.sqrt(np.mean((y - yhat)**2)))

def qlike(y, yhat):
    y = np.asarray(y) + EPS
    yhat = np.asarray(yhat) + EPS
    return float(np.mean(np.log(yhat) + y / yhat))

def pinball_loss(y, q, tau=0.05):
    e = y - q
    return float(np.mean(np.maximum(tau*e, (tau-1)*e)))

def kupiec_pof(y, var, alpha=0.95):
    # y: returns, var: VaR (negative; threshold), breach if y < var
    y = np.asarray(y); var = np.asarray(var)
    mask = ~np.isnan(var)
    y = y[mask]; var = var[mask]
    T = len(y)
    if T == 0:
        return 0, float("nan"), float("nan")
    x = int(np.sum(y < var))
    p = 1 - alpha
    p_hat = x / T if T else 0.0
    def ll(k, n, pp): return (k*np.log(pp + EPS) + (n-k)*np.log(1-pp + EPS))
    # guard p_hat ∈ (0,1)
    ph = p_hat if 0 < p_hat < 1 else (x + 0.5) / (T + 1)
    LRuc = -2*(ll(x, T, p) - ll(x, T, ph))
    from scipy.stats import chi2
    pval = float(1 - chi2.cdf(LRuc, 1))
    return x, x/T if T else float("nan"), pval

def christoffersen_independence(y, var):
    # Smoothed (Jeffreys 0.5 add) to avoid log(0)
    y = np.asarray(y); var = np.asarray(var)
    mask = ~np.isnan(var)
    y = y[mask]; var = var[mask]
    if len(y) < 2:
        return 0, float("nan")
    I = (y < var).astype(int)
    n00 = np.sum((I[1:] == 0) & (I[:-1] == 0))
    n01 = np.sum((I[1:] == 1) & (I[:-1] == 0))
    n10 = np.sum((I[1:] == 0) & (I[:-1] == 1))
    n11 = np.sum((I[1:] == 1) & (I[:-1] == 1))
    n00s, n01s, n10s, n11s = n00+0.5, n01+0.5, n10+0.5, n11+0.5
    pi0 = n01s / (n00s + n01s)
    pi1 = n11s / (n10s + n11s)
    pi  = (n01s + n11s) / (n00s + n01s + n10s + n11s)
    from numpy import log
    ll_indep  = (n00s*log(1-pi)  + n01s*log(pi)  + n10s*log(1-pi)  + n11s*log(pi))
    ll_markov = (n00s*log(1-pi0) + n01s*log(pi0) + n10s*log(1-pi1) + n11s*log(pi1))
    LRind = -2*(ll_indep - ll_markov)
    from scipy.stats import chi2
    pval = float(1 - chi2.cdf(LRind, 1))
    return int(n01+n11), pval

def var_acceptance_band(n=250, p=0.05, conf=0.95):
    from scipy.stats import binom
    lo = int(binom.ppf((1-conf)/2, n, p))
    hi = int(binom.ppf(1-(1-conf)/2, n, p))
    return lo, hi

def band_status(x, lo, hi):
    if x < lo:  return f"too_few[{lo}-{hi}]"
    if x > hi:  return f"too_many[{lo}-{hi}]"
    return f"within[{lo}-{hi}]"

# ------------------------- Rolling calibration -------------------------

def rolling_calibrated_q05(y, q_raw, window=250, p=0.05, min_hist=100, ema=None):
    """
    For each t >= window, compute delta_t = Q_p( y - q_raw | last 'window' obs ),
    then q_cal[t] = q_raw[t] + delta_t.
    - p: target quantile (default 0.05)
    - min_hist: require at least this many non-NaN residuals before calibrating
    - ema: optional smoothing factor in [0,1]; if set, delta_t := ema*delta_t + (1-ema)*delta_{t-1}
    """
    y = np.asarray(y, dtype=float)
    q_raw = np.asarray(q_raw, dtype=float)
    n = len(y)
    q_cal = np.full(n, np.nan, dtype=float)
    last_delta = np.nan
    for t in range(window, n):
        resid = y[t-window:t] - q_raw[t-window:t]
        resid = resid[~np.isnan(resid)]
        if resid.size < min_hist:
            continue
        delta = np.nanquantile(resid, p)
        if ema is not None and not np.isnan(last_delta):
            delta = ema*delta + (1.0-ema)*last_delta
        q_cal[t] = q_raw[t] + delta
        last_delta = delta
    return q_cal

def rolling_backtest(y, var_dict, alpha=0.95, window=250, min_eff=200):
    """
    Slide a window of 'window' days; for each model in var_dict, compute stats only
    if there are at least 'min_eff' non-NaN VaR values in that window.
    """
    dates = pd.to_datetime(var_dict["dates"])
    rows = []
    p = 1 - alpha
    for start in range(0, len(y) - window + 1):
        end = start + window
        y_win = y[start:end]
        for name, v in var_dict.items():
            if name == "dates":
                continue
            v_win = v[start:end]
            mask = ~np.isnan(v_win)
            n = int(mask.sum())
            if n < min_eff:
                continue
            x, cov, p_k = kupiec_pof(y_win[mask], v_win[mask], alpha=alpha)
            _, p_c = christoffersen_independence(y_win[mask], v_win[mask])
            lo, hi = var_acceptance_band(n=n, p=p, conf=0.95)
            rows.append([dates[start], dates[end-1], name, n, x, cov, p_k, p_c, f"{lo}-{hi}", band_status(x, lo, hi)])
    cols = ["window_start","window_end","model","N","breaches","coverage","kupiec_p","christoffersen_p","band_95pct","status_95pct"]
    return pd.DataFrame(rows, columns=cols)


# ------------------------- Main -------------------------
def main(symbol, holdout_start, alpha, hist_win,
         calib_mode, calib_window, roll_window,
         calib_quantile=None, calib_ema=None):

    Path("tables").mkdir(exist_ok=True, parents=True)
    Path("figs").mkdir(exist_ok=True, parents=True)
    Path("outputs").mkdir(exist_ok=True, parents=True)

    # Load data & predictions
    dfp = pd.read_parquet(f"data/{symbol.lower()}_rv_clean.parquet")
    dfp["date"] = pd.to_datetime(dfp["date"])
    dfp["ret"] = np.log(dfp["Close"]).diff()
    dfp = dfp.dropna(subset=["ret","RV_GK"]).loc[:, ["date","ret","RV_GK"]]

    har = pd.read_csv("outputs/har_preds.csv", parse_dates=["date"])
    gar = pd.read_csv("outputs/garch_preds.csv", parse_dates=["date"])
    pat = pd.read_csv("outputs/patch_preds.csv", parse_dates=["date"])  # ret_true, q05_ret_pred, optional sigma2_pred

    merged = (dfp.merge(har, on="date", how="inner", suffixes=("","_har"))
                 .merge(gar, on="date", how="inner", suffixes=("","_gar"))
                 .merge(pat, on="date", how="inner"))
    merged = merged[merged["date"] >= pd.to_datetime(holdout_start)].copy()
    merged = merged.sort_values("date").reset_index(drop=True)

    # Variance metrics (HAR, GARCH, optional deep variance head)
    err_rows = []
    err_rows.append(("HAR", rmse(merged["RV_GK"], merged["rv_pred"]), qlike(merged["RV_GK"], merged["rv_pred"])))
    err_rows.append(("GARCH_t", rmse(merged["RV_GK"], merged["rv_pred_gar"]), qlike(merged["RV_GK"], merged["rv_pred_gar"])))
    if "sigma2_pred" in merged.columns:
        err_rows.append(("PatchTST_var", rmse(merged["RV_GK"], merged["sigma2_pred"]), qlike(merged["RV_GK"], merged["sigma2_pred"])))
    pd.DataFrame(err_rows, columns=["model","RMSE","QLIKE"]).to_csv("tables/error_metrics.csv", index=False)

    # VaR series
    ret = merged["ret"].values
    dates = merged["date"].values

    sigma_har = np.sqrt(np.clip(merged["rv_pred"].values, EPS, None))
    sigma_gar = np.sqrt(np.clip(merged["rv_pred_gar"].values, EPS, None))
    var_har = -Z95 * sigma_har
    var_gar = -Z95 * sigma_gar

    q05_raw = merged["q05_ret_pred"].values  # left-tail returns threshold already
    # Fixed intercept calibration (across full holdout)
    delta_fixed = np.quantile(ret - q05_raw, 0.05)
    q05_cal_fixed = q05_raw + delta_fixed
    # Rolling intercept calibration
    p_q  = (1 - alpha) if calib_quantile is None else calib_quantile
    ema  = calib_ema
    q05_cal_roll = rolling_calibrated_q05(
        ret, q05_raw,
        window=calib_window,
        p=p_q,
        min_hist=max(100, calib_window // 2),
        ema=ema
    )



    # Choose which PatchTST VaR to treat as "calibrated" in tables
    if calib_mode == "rolling":
        var_patch_cal = q05_cal_roll
    elif calib_mode == "none":
        var_patch_cal = q05_raw
    else:  # "fixed"
        var_patch_cal = q05_cal_fixed

    # Save VaR series
    out_var = pd.DataFrame({
        "date": merged["date"],
        "ret_true": ret,
        "VaR95_HAR": var_har,
        "VaR95_GARCHt": var_gar,
        "VaR95_Patch_raw": q05_raw,
        "VaR95_Patch_cal": var_patch_cal,
        "VaR95_Patch_rollcal": q05_cal_roll
    })
    out_var.to_csv("outputs/var_series.csv", index=False)

    # Overall back-tests on the full holdout
    rows = []
    for name, v in [("HAR", var_har), ("GARCH_t", var_gar), ("Patch_raw", q05_raw), ("Patch_cal", var_patch_cal)]:
        mask = ~np.isnan(v)
        n_eff = int(mask.sum())
        x, cov, p_k = kupiec_pof(ret[mask], v[mask], alpha=alpha)
        _, p_c = christoffersen_independence(ret[mask], v[mask])
        # last 250 days within the *effective* part
        tail = min(250, n_eff)
        y_tail = ret[mask][-tail:]
        v_tail = v[mask][-tail:]
        x250 = int(np.sum(y_tail < v_tail))
        lo, hi = var_acceptance_band(n=tail, p=1-alpha, conf=0.95)
        rows.append([name, n_eff, x, cov, p_k, p_c, x250, f"{lo}-{hi}", band_status(x250, lo, hi)])

    bt = pd.DataFrame(rows, columns=[
        "model","N_effective","breaches","coverage","kupiec_p","christoffersen_p","breaches_lastN","band_95pct","status_95pct"
    ])

    bt.to_csv("tables/var_backtest.csv", index=False)

    # Rolling backtest over sliding windows
    vb = {
        "dates": dates,
        "HAR": var_har,
        "GARCH_t": var_gar,
        "Patch_raw": q05_raw,
        "Patch_cal": var_patch_cal
    }
    bt_roll = rolling_backtest(ret, vb, alpha=alpha, window=roll_window, min_eff=min(roll_window, 200))

    bt_roll.to_csv("tables/var_backtest_rolling.csv", index=False)

    # Headline printout
    cov_raw = (ret < q05_raw).mean()
    cov_fix = (ret < q05_cal_fixed).mean()
    mask_roll = ~np.isnan(q05_cal_roll)
    cov_roll = np.nanmean(ret[mask_roll] < q05_cal_roll[mask_roll])
    print(f"PatchTST coverage raw={cov_raw:.4f}, fixed={cov_fix:.4f}, rolling={cov_roll:.4f} "
        f"(N_eff={mask_roll.sum()}, target {(1-alpha):.4f})")

    # Breach timeline figure (keep your visual; show Patch_cal selection)
    plt.figure(figsize=(11,5))
    plt.plot(merged["date"], ret, lw=0.8, label="Return")
    plt.plot(merged["date"], var_har, lw=0.8, label="VaR95 HAR")
    plt.plot(merged["date"], var_gar, lw=0.8, label="VaR95 GARCH-t")
    plt.plot(merged["date"], var_patch_cal, lw=1.0, label=f"VaR95 Patch ({calib_mode})")
    breaches = ret < var_patch_cal
    plt.scatter(merged["date"][breaches], ret[breaches], s=10, label="Breaches (Patch)")
    plt.axhline(0, color="black", lw=0.5)
    plt.title("VaR95 breach timeline (holdout)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/var_breach_timeline.png", dpi=140); plt.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--holdout_start", default="2023-01-02")
    ap.add_argument("--alpha", type=float, default=0.95)
    ap.add_argument("--hist_win", type=int, default=60)
    ap.add_argument("--calib_mode", choices=["fixed","rolling","none"], default="fixed")
    ap.add_argument("--calib_window", type=int, default=250, help="days of past residuals for rolling calibration")
    ap.add_argument("--roll_window", type=int, default=250, help="window size for rolling backtest")
    ap.add_argument("--calib_quantile", type=float, default=None,
                help="Quantile used for rolling intercept shift (default: 1-alpha; e.g., 0.045)")
    ap.add_argument("--calib_ema", type=float, default=None,
                help="EMA smoothing factor in [0,1] for rolling calibration (e.g., 0.1)")

    args = ap.parse_args()
    main(args.symbol, args.holdout_start, args.alpha, args.hist_win,
     args.calib_mode, args.calib_window, args.roll_window,
     args.calib_quantile, args.calib_ema)
