#!/usr/bin/env python3
"""
Evaluate Value at Risk (VaR) at level alpha for PatchTST predictions with optional calibration:
  - none/raw
  - fixed: single intercept delta learned on a calibration window
  - rolling: time-varying intercept from a rolling past window (optional EMA smoothing)

Outputs:
  - tables/var_backtest.csv
  - figs/var_breach_timeline.png

Assumes outputs/patch_preds.csv with:
  date (datetime-like), ret_true (float), q05_ret_pred (float), sigma2_pred (float, optional)
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2, binom  # chi-squared survival + exact binomial interval


def _to_datetime(x: pd.Series) -> pd.Series:
    return pd.to_datetime(x).dt.tz_localize(None)


def _exception_indicator(ret_true: np.ndarray, q_pred: np.ndarray) -> np.ndarray:
    # VaR exception (breach) if realized return is below predicted α-quantile
    return (ret_true < q_pred).astype(np.int32)


def _calibrate_fixed(ret: np.ndarray, q: np.ndarray, alpha: float, window: int):
    """
    Learn a constant delta as the alpha-quantile of residuals on the first `window` points,
    then evaluate only on the post-warmup region.
    """
    n = len(ret)
    if n <= window:
        exc = _exception_indicator(ret, q)
        rate = float(exc.mean()) if len(exc) else np.nan
        return exc, rate, len(exc)
    resid = ret[:window] - q[:window]
    delta = np.quantile(resid, alpha)
    q_adj = q + delta
    exc = _exception_indicator(ret[window:], q_adj[window:])
    rate = float(exc.mean()) if len(exc) else np.nan
    return exc, rate, len(exc)


def _calibrate_rolling(ret: np.ndarray, q: np.ndarray, alpha: float, window: int, ema: Optional[float]):
    """
    For each t >= window, estimate delta_t from previous `window` residuals.
    Optionally smooth deltas using an exponential moving average (EMA).
    """
    n = len(ret)
    if n <= window:
        exc = _exception_indicator(ret, q)
        return exc, float(exc.mean()) if len(exc) else np.nan, len(exc)
    deltas = np.zeros(n, dtype=np.float64)
    last = 0.0
    for t in range(window, n):
        resid = ret[t - window:t] - q[t - window:t]
        delta_t = np.quantile(resid, alpha)
        if ema is not None and 0.0 < ema < 1.0 and t > window:
            delta_t = ema * delta_t + (1.0 - ema) * last
        deltas[t] = delta_t
        last = delta_t
    q_adj = q + deltas
    exc = _exception_indicator(ret[window:], q_adj[window:])
    rate = float(exc.mean()) if len(exc) else np.nan
    return exc, rate, len(exc)


def _lr_uc(ex: np.ndarray, alpha: float) -> float:
    """
    Log-likelihood ratio for Kupiec unconditional coverage (UC).
    Returns the LR statistic; p-value is chi2.sf(LR, df=1).
    """
    e = np.asarray(ex, dtype=int)
    n = int(e.size)
    if n == 0:
        return np.nan
    x = int(e.sum())
    # log-likelihoods
    ll_alpha = (n - x) * np.log1p(-alpha) + x * np.log(alpha)
    # guard against log(0) by clipping
    eps = 1e-12
    pi = min(max(x / n, eps), 1.0 - eps)
    ll_pi = (n - x) * np.log1p(-pi) + x * np.log(pi)
    return -2.0 * (ll_alpha - ll_pi)


def kupiec_pvalue(exceptions: np.ndarray, alpha: float) -> float:
    """Kupiec UC p-value: tests if breach frequency equals alpha (χ² with df=1)."""
    lr = _lr_uc(exceptions, alpha)
    return float(chi2.sf(lr, df=1))


def christoffersen_ind_pvalue(exceptions: np.ndarray) -> float:
    """
    Christoffersen independence p-value: tests no clustering of breaches.
    Uses χ² with df=1 on the LR_ind statistic.
    """
    e = np.asarray(exceptions, dtype=int)
    if e.size <= 1:
        return np.nan

    n00 = n01 = n10 = n11 = 0
    for i in range(1, e.size):
        a, b = int(e[i - 1]), int(e[i])
        if   a == 0 and b == 0: n00 += 1
        elif a == 0 and b == 1: n01 += 1
        elif a == 1 and b == 0: n10 += 1
        else:                    n11 += 1

    n0 = n00 + n01
    n1 = n10 + n11
    total = n0 + n1
    if total == 0:
        return np.nan

    pi01 = 0.0 if n0 == 0 else n01 / n0
    pi11 = 0.0 if n1 == 0 else n11 / n1
    pi1  = (n01 + n11) / total

    def _ll(p, s, f):
        eps = 1e-12
        p = min(max(p, eps), 1.0 - eps)
        return s * np.log(p) + f * np.log1p(-p)

    ll_restricted   = _ll(pi1,  n01 + n11, n00 + n10)
    ll_unrestricted = _ll(pi01, n01,       n00) + _ll(pi11, n11, n10)

    lr_ind = -2.0 * (ll_restricted - ll_unrestricted)
    return float(chi2.sf(lr_ind, df=1))


def christoffersen_cc_pvalue(exceptions: np.ndarray, alpha: float) -> float:
    """
    Christoffersen conditional coverage p-value.
    LR_cc = LR_uc + LR_ind, tested against χ² with df=2.
    """
    lr_uc = _lr_uc(exceptions, alpha)
    # Recompute LR_ind to avoid numerical inversion from p
    e = np.asarray(exceptions, dtype=int)
    if e.size <= 1:
        return np.nan

    n00 = n01 = n10 = n11 = 0
    for i in range(1, e.size):
        a, b = int(e[i - 1]), int(e[i])
        if   a == 0 and b == 0: n00 += 1
        elif a == 0 and b == 1: n01 += 1
        elif a == 1 and b == 0: n10 += 1
        else:                    n11 += 1

    n0 = n00 + n01
    n1 = n10 + n11
    total = n0 + n1
    if total == 0:
        return np.nan

    pi01 = 0.0 if n0 == 0 else n01 / n0
    pi11 = 0.0 if n1 == 0 else n11 / n1
    pi1  = (n01 + n11) / total

    def _ll(p, s, f):
        eps = 1e-12
        p = min(max(p, eps), 1.0 - eps)
        return s * np.log(p) + f * np.log1p(-p)

    ll_r  = _ll(pi1,  n01 + n11, n00 + n10)
    ll_ur = _ll(pi01, n01,       n00) + _ll(pi11, n11, n10)
    lr_ind = -2.0 * (ll_r - ll_ur)

    lr_cc = lr_uc + lr_ind
    return float(chi2.sf(lr_cc, df=2))


def last250_band(alpha: float, n: int = 250, conf: float = 0.95) -> Tuple[int, int]:
    """
    Binomial acceptance band on last n points at confidence level `conf`.
    What it tells: whether the count of recent breaches is consistent with alpha.
    """
    if n <= 0:
        return (0, 0)
    lo, hi = binom.interval(conf, n, alpha)
    return int(lo), int(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--holdout_start", default=None, help="YYYY-MM-DD")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--calib_mode", choices=["fixed", "rolling", "none"], default="none")
    ap.add_argument("--calib_window", type=int, default=250)
    ap.add_argument("--roll_window", type=int, default=250)
    ap.add_argument("--calib_ema", type=float, default=0.0)  # exponential moving average weight
    args = ap.parse_args()

    ROOT = Path(".")
    OUT = ROOT / "outputs"
    TAB = ROOT / "tables"
    FIG = ROOT / "figs"
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    # ---- Load predictions
    df = pd.read_csv(OUT / "patch_preds.csv")
    need = {"date", "ret_true", "q05_ret_pred"}
    if not need.issubset(df.columns):
        raise ValueError("Expected outputs/patch_preds.csv with columns: date, ret_true, q05_ret_pred")
    df["date"] = _to_datetime(df["date"])
    df = df.sort_values("date")
    if args.holdout_start:
        df = df[df["date"] >= pd.to_datetime(args.holdout_start)]
    df = df.dropna(subset=["ret_true", "q05_ret_pred"]).reset_index(drop=True)

    ret = df["ret_true"].to_numpy(dtype=np.float64)
    q05 = df["q05_ret_pred"].to_numpy(dtype=np.float64)

    # ---- Raw (no calibration)
    exc_raw = _exception_indicator(ret, q05)
    br_raw = float(exc_raw.mean()) if len(exc_raw) else np.nan
    cov_raw = 1.0 - br_raw if not np.isnan(br_raw) else np.nan
    N_raw = len(exc_raw)

    # ---- Fixed calibration
    exc_fix, br_fix, N_fix = _calibrate_fixed(ret, q05, args.alpha, args.calib_window)
    cov_fix = 1.0 - br_fix if not np.isnan(br_fix) else np.nan

    # ---- Rolling calibration
    exc_roll, br_roll, N_roll = _calibrate_rolling(ret, q05, args.alpha, args.roll_window, args.calib_ema)
    cov_roll = 1.0 - br_roll if not np.isnan(br_roll) else np.nan

    # ---- Tests
    k_raw  = kupiec_pvalue(exc_raw,  args.alpha); ci_raw  = christoffersen_ind_pvalue(exc_raw);  cc_raw  = christoffersen_cc_pvalue(exc_raw,  args.alpha)
    k_fix  = kupiec_pvalue(exc_fix,  args.alpha); ci_fix  = christoffersen_ind_pvalue(exc_fix);  cc_fix  = christoffersen_cc_pvalue(exc_fix,  args.alpha)
    k_roll = kupiec_pvalue(exc_roll, args.alpha); ci_roll = christoffersen_ind_pvalue(exc_roll); cc_roll = christoffersen_cc_pvalue(exc_roll, args.alpha)

    # ---- Last-250 acceptance band (rolling)
    n_band = min(250, len(exc_roll))
    lo, hi = last250_band(args.alpha, n_band, conf=0.95)
    last250 = int(exc_roll[-n_band:].sum()) if n_band > 0 else 0
    band_str = f"{lo}–{hi}"

    # ---- Save table
    table = pd.DataFrame([
        {"model":"Patch_raw","mode":"none","breach_rate":br_raw,"coverage":cov_raw,
         "kupiec_p":k_raw,"christoffersen_ind_p":ci_raw,"christoffersen_cc_p":cc_raw,
         "effective_n":N_raw,"last250_breaches":int(exc_raw[-n_band:].sum()) if n_band>0 else 0,"band_95pct":band_str},
        {"model":"Patch_fixed","mode":"fixed","breach_rate":br_fix,"coverage":cov_fix,
         "kupiec_p":k_fix,"christoffersen_ind_p":ci_fix,"christoffersen_cc_p":cc_fix,
         "effective_n":N_fix,"last250_breaches":int(exc_fix[-n_band:].sum()) if n_band>0 else 0,"band_95pct":band_str},
        {"model":"Patch_cal","mode":"rolling","breach_rate":br_roll,"coverage":cov_roll,
         "kupiec_p":k_roll,"christoffersen_ind_p":ci_roll,"christoffersen_cc_p":cc_roll,
         "effective_n":N_roll,"last250_breaches":last250,"band_95pct":band_str},
    ])
    table_path = TAB / "var_backtest.csv"
    table.to_csv(table_path, index=False)

    # ---- Plot breach timeline (rolling)
    if len(df) > 0:
        plt.figure(figsize=(9, 3))
        dates = df["date"].to_numpy()
        y = np.zeros_like(dates, dtype=float)
        offset = len(df) - len(exc_roll)
        y[offset:] = exc_roll
        plt.plot(dates, y, drawstyle="steps-mid")
        plt.title(f"VaR breaches (alpha={args.alpha:.3f}) — rolling window={args.roll_window}, EMA={args.calib_ema}")
        plt.ylabel("breach (1=yes)"); plt.xlabel("date")
        plt.tight_layout(); plt.savefig(FIG / "var_breach_timeline.png", dpi=140); plt.close()

    # ---- Print summary
    alpha = args.alpha
    print(f"PatchTST VaR{int((1-alpha)*100)} exception_rate raw={br_raw:.4f}, fixed={br_fix:.4f}, rolling={br_roll:.4f} (N_eff={N_roll}, target={alpha:.4f})")
    print(f"             coverage         raw={cov_raw:.4f}, fixed={cov_fix:.4f}, rolling={cov_roll:.4f} (target={1.0-alpha:.4f})")
    print(f"             Kupiec p         raw={k_raw:.3f}, fixed={k_fix:.3f}, rolling={k_roll:.3f}")
    print(f"             Christoffersen p (ind) raw={ci_raw:.3f}, fixed={ci_fix:.3f}, rolling={ci_roll:.3f}")
    print(f"             Christoffersen p (cc)  raw={cc_raw:.3f}, fixed={cc_fix:.3f}, rolling={cc_roll:.3f}")
    print(f"             last-{n_band} breaches (rolling) = {last250} in [{band_str}]")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
