#!/usr/bin/env python3
"""
Evaluate VaR (alpha) for PatchTST predictions with optional calibration:
  - none/raw
  - fixed: one intercept delta from a calibration window
  - rolling: time-varying intercept using a rolling past window (optional EMA smoothing)

Outputs:
  - tables/var_backtest.csv
  - figs/var_breach_timeline.png
And prints:
  PatchTST VaR95 exception_rate raw=..., fixed=..., rolling=... (N_eff=..., target=...)

Assumptions:
  outputs/patch_preds.csv with columns:
    date (datetime-like), ret_true (float), q05_ret_pred (float), sigma2_pred (float, optional)

CLI matches prior usage in your logs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# SciPy is nice-to-have for exact binomial intervals; fallback to normal approx if missing.
try:
    from scipy.stats import binom
except Exception:
    binom = None


def _to_datetime(x: pd.Series) -> pd.Series:
    return pd.to_datetime(x).dt.tz_localize(None)


def _exception_indicator(ret_true: np.ndarray, q_pred: np.ndarray) -> np.ndarray:
    # VaR_0.95 is the 5% left-tail quantile → exception if realized return is below predicted quantile.
    return (ret_true < q_pred).astype(np.int32)


def _calibrate_fixed(ret: np.ndarray, q: np.ndarray, alpha: float, window: int) -> Tuple[np.ndarray, float, int]:
    """
    Find a constant delta so that exceptions in the first 'window' points hit alpha.
    Apply that delta to the remaining points; return exceptions over the post-warmup region,
    the breach rate, and effective N.
    """
    if len(ret) <= window:
        return _exception_indicator(ret, q), float(np.mean(ret < q)), len(ret)

    # Solve for delta: mean(ret[:W] < q[:W] + delta) = alpha
    # This is equivalent to choosing the alpha-quantile of residuals (ret - q) over the calibration window.
    resid = ret[:window] - q[:window]
    delta = np.quantile(resid, alpha)
    q_adj = q + delta

    exc = _exception_indicator(ret[window:], q_adj[window:])
    rate = float(exc.mean()) if len(exc) else np.nan
    return exc, rate, len(exc)


def _calibrate_rolling(ret: np.ndarray, q: np.ndarray, alpha: float, window: int, ema: Optional[float]) -> Tuple[np.ndarray, float, int]:
    """
    At each t >= window, compute delta_t from the previous 'window' residuals so that
    mean(ret < q + delta_t) ≈ alpha. Optionally smooth deltas with EMA.
    Exceptions are evaluated on q + delta_t (using the delta from *past* window only).
    """
    n = len(ret)
    if n <= window:
        # Not enough for rolling; fall back to raw on what we have
        exc = _exception_indicator(ret, q)
        return exc, float(exc.mean()), n

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


def kupiec_pvalue(exceptions: np.ndarray, alpha: float) -> float:
    """
    Kupiec unconditional coverage test (LR_uc).
    """
    n = len(exceptions)
    if n == 0:
        return np.nan
    x = int(exceptions.sum())
    # Likelihood ratio statistic:
    # LR_uc = -2 [ ln((1 - alpha)^(n-x) * alpha^x) - ln((1 - pi)^(n-x) * pi^x) ]
    # with pi = x/n
    pi = x / n if n > 0 else 0.0
    if pi in (0.0, 1.0):
        # degenerate; return conservative p=1.0 if exactly equals alpha, else 0.0
        return 1.0 if abs(pi - alpha) < 1e-12 else 0.0
    ll_alpha = (n - x) * np.log1p(-alpha) + x * np.log(alpha)
    ll_pi = (n - x) * np.log1p(-pi) + x * np.log(pi)
    lr = -2.0 * (ll_alpha - ll_pi)
    # Approx chi2_1
    from math import exp
    # p-value = 1 - CDF_chi2_1(lr) = exp(-lr/2) approximately for 1 dof tail
    return float(np.exp(-lr / 2.0))


def christoffersen_pvalue(exceptions: np.ndarray) -> float:
    """
    Christoffersen independence (LR_cc): 2x2 transition matrix of exceptions.
    """
    n = len(exceptions)
    if n <= 1:
        return np.nan
    e = exceptions.astype(int)
    n00 = n01 = n10 = n11 = 0
    for i in range(1, n):
        a, b = e[i - 1], e[i]
        if a == 0 and b == 0:
            n00 += 1
        elif a == 0 and b == 1:
            n01 += 1
        elif a == 1 and b == 0:
            n10 += 1
        else:
            n11 += 1

    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    pi01 = safe_div(n01, n00 + n01)
    pi11 = safe_div(n11, n10 + n11)
    pi = safe_div(n01 + n11, n00 + n01 + n10 + n11)

    # Log-likelihoods
    def ll(p, n0, n1):
        if p in (0.0, 1.0):
            if (p == 0.0 and n1 == 0) or (p == 1.0 and n0 == 0):
                return 0.0
            return -np.inf
        return n0 * np.log1p(-p) + n1 * np.log(p)

    ll_ind = ll(pi, n00 + n10, n01 + n11)
    ll_dep = ll(pi01, n00, n01) + ll(pi11, n10, n11)

    lr = -2.0 * (ll_ind - ll_dep)
    # p ≈ exp(-lr/2) for 1 dof tail (chi2_1)
    return float(np.exp(-lr / 2.0))


def last250_band(alpha: float, n: int = 250, conf: float = 0.95) -> Tuple[int, int]:
    """
    Binomial acceptance band at (1 - conf) for the number of breaches in last n points.
    """
    if n <= 0:
        return (0, 0)
    if binom is not None:
        lo, hi = binom.interval(conf, n, alpha)
        return int(lo), int(hi)
    # Normal approximation
    mu = n * alpha
    sigma = np.sqrt(n * alpha * (1 - alpha))
    # 1.96 ~ 95%
    lo = int(np.floor(mu - 1.96 * sigma))
    hi = int(np.ceil(mu + 1.96 * sigma))
    lo = max(0, lo)
    hi = min(n, hi)
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--holdout_start", default=None, help="YYYY-MM-DD")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--hist_win", type=int, default=250, help="reserved; not used")
    ap.add_argument("--calib_mode", choices=["fixed", "rolling", "none"], default="none")
    ap.add_argument("--calib_window", type=int, default=250)
    ap.add_argument("--roll_window", type=int, default=250)
    ap.add_argument("--calib_quantile", type=float, default=None, help="reserved; not used (alpha is used)")
    ap.add_argument("--calib_ema", type=float, default=0.20)
    args = ap.parse_args()

    ROOT = Path(".")
    OUT = ROOT / "outputs"
    TAB = ROOT / "tables"
    FIG = ROOT / "figs"
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    # ---- Load predictions
    df = pd.read_csv(OUT / "patch_preds.csv")
    if "date" not in df.columns or "ret_true" not in df.columns or "q05_ret_pred" not in df.columns:
        raise ValueError("Expected outputs/patch_preds.csv with columns: date, ret_true, q05_ret_pred")
    df["date"] = _to_datetime(df["date"])
    df = df.sort_values("date")
    if args.holdout_start:
        df = df[df["date"] >= pd.to_datetime(args.holdout_start)]
    df = df.reset_index(drop=True)

    ret = df["ret_true"].to_numpy(dtype=np.float64)
    q05 = df["q05_ret_pred"].to_numpy(dtype=np.float64)

    # ---- Raw (no calibration)
    exc_raw = _exception_indicator(ret, q05)
    br_raw = float(exc_raw.mean()) if len(exc_raw) else np.nan  # breach rate
    cov_raw = 1.0 - br_raw if not np.isnan(br_raw) else np.nan
    N_raw = len(exc_raw)

    # ---- Fixed calibration
    exc_fix, br_fix, N_fix = _calibrate_fixed(ret, q05, args.alpha, args.calib_window)
    cov_fix = 1.0 - br_fix if not np.isnan(br_fix) else np.nan

    # ---- Rolling calibration
    exc_roll, br_roll, N_roll = _calibrate_rolling(ret, q05, args.alpha, args.roll_window, args.calib_ema)
    cov_roll = 1.0 - br_roll if not np.isnan(br_roll) else np.nan

    # ---- Statistical tests on the evaluated regions
    k_raw = kupiec_pvalue(exc_raw, args.alpha)
    c_raw = christoffersen_pvalue(exc_raw)

    k_fix = kupiec_pvalue(exc_fix, args.alpha)
    c_fix = christoffersen_pvalue(exc_fix)

    k_roll = kupiec_pvalue(exc_roll, args.alpha)
    c_roll = christoffersen_pvalue(exc_roll)

    # ---- Last-250 acceptance band & count for rolling series (typical for reporting)
    n_band = min(250, len(exc_roll))
    lo, hi = last250_band(args.alpha, n_band, conf=0.95)
    last250 = int(exc_roll[-n_band:].sum()) if n_band > 0 else 0
    band_str = f"{lo}–{hi}"

    # ---- Save table
    table = pd.DataFrame([
        {
            "model": "Patch_raw", "mode": "none",
            "breach_rate": br_raw, "coverage": cov_raw,
            "kupiec_p": k_raw, "christoffersen_p": c_raw,
            "effective_n": N_raw,
            "last250_breaches": int(exc_raw[-n_band:].sum()) if n_band > 0 else 0,
            "band_95pct": band_str,
        },
        {
            "model": "Patch_fixed", "mode": "fixed",
            "breach_rate": br_fix, "coverage": cov_fix,
            "kupiec_p": k_fix, "christoffersen_p": c_fix,
            "effective_n": N_fix,
            "last250_breaches": int(exc_fix[-n_band:].sum()) if n_band > 0 else 0,
            "band_95pct": band_str,
        },
        {
            "model": "Patch_cal", "mode": "rolling",
            "breach_rate": br_roll, "coverage": cov_roll,
            "kupiec_p": k_roll, "christoffersen_p": c_roll,
            "effective_n": N_roll,
            "last250_breaches": last250,
            "band_95pct": band_str,
        },
    ])
    table_path = TAB / "var_backtest.csv"
    table.to_csv(table_path, index=False)

    # ---- Plot breach timeline (rolling)
    if len(df) > 0:
        plt.figure(figsize=(9, 3))
        dates = df["date"].to_numpy()
        y = np.zeros_like(dates, dtype=float)
        # mark rolling-evaluated region; align length
        offset = len(df) - len(exc_roll)
        y[offset:] = exc_roll
        plt.plot(dates, y, drawstyle="steps-mid")
        plt.title(f"VaR breaches (alpha={args.alpha:.3f}) — rolling calibration window={args.roll_window}, EMA={args.calib_ema}")
        plt.ylabel("breach (1=yes)")
        plt.xlabel("date")
        plt.tight_layout()
        plt.savefig(FIG / "var_breach_timeline.png", dpi=140)
        plt.close()

    # ---- Print normalized, unambiguous summary
    alpha = args.alpha
    print(
        f"PatchTST VaR{int((1-alpha)*100):d} exception_rate "
        f"raw={br_raw:.4f}, fixed={br_fix:.4f}, rolling={br_roll:.4f} "
        f"(N_eff={N_roll}, target={alpha:.4f})"
    )
    print(
        f"             coverage "
        f"raw={cov_raw:.4f}, fixed={cov_fix:.4f}, rolling={cov_roll:.4f} "
        f"(target={1.0-alpha:.4f})"
    )
    print(f"             Kupiec p raw={k_raw:.3f}, fixed={k_fix:.3f}, rolling={k_roll:.3f}")
    print(f"             Christoffersen p raw={c_raw:.3f}, fixed={c_fix:.3f}, rolling={c_roll:.3f}")
    print(f"             last-{n_band} breaches (rolling) = {last250} in [{band_str}]")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
