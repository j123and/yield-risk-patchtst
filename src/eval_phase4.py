#!/usr/bin/env python3
"""
Evaluate VaR at level alpha for PatchTST predictions with leak-safe calibration.

Calibrators:
  none
  fixed                   : q' = q + δ̂
  rolling                 : rolling δ̂ (EMA optional)
  fixed_affine            : q' = a + b q
  rolling_affine          : a_t + b_t q_t on [t-W..t-1]
  fixed_affine3           : a + b q + c s
  rolling_affine3         : a_t + b_t q_t + c_t s_t (leak-safe)

Scale regressor s_t:
  --sigma_source predicted : sqrt(sigma2_pred) from patch_preds.csv
  --sigma_source gk        : GK sigma **lagged by one day** (σ_{t-1})
  --sigma_source none      : 1
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import chi2, binom
except Exception:
    chi2 = None; binom = None

def _to_datetime(x: pd.Series) -> pd.Series: return pd.to_datetime(x).dt.tz_localize(None)
def _exc(ret: np.ndarray, q: np.ndarray) -> np.ndarray: return (ret < q).astype(np.int32)

def _chisq_sf(lr: float, df: int) -> float:
    lr = float(max(lr, 0.0))
    if chi2 is not None: return float(chi2.sf(lr, df))
    if df == 1:
        from math import erfc, sqrt
        return float(erfc(sqrt(lr/2)))
    return float(np.exp(-lr/2))

def _lr_uc(e: np.ndarray, a: float) -> float:
    n = int(e.size); x = int(e.sum())
    if n == 0: return 0.0
    pi = x / n
    if pi in (0.0, 1.0): return 0.0 if abs(pi - a) < 1e-12 else 1e9
    ll_a = (n-x)*np.log1p(-a) + x*np.log(a)
    ll_p = (n-x)*np.log1p(-pi) + x*np.log(pi)
    return -2.0*(ll_a - ll_p)

def _lr_ind(e: np.ndarray) -> float:
    n = int(e.size)
    if n <= 1: return 0.0
    e = e.astype(np.int8)
    n00=n01=n10=n11=0
    for i in range(1,n):
        a,b = int(e[i-1]), int(e[i])
        if a==0 and b==0: n00+=1
        elif a==0 and b==1: n01+=1
        elif a==1 and b==0: n10+=1
        else: n11+=1
    total0, total1 = n00+n10, n01+n11
    pi = (total1/(total0+total1)) if (total0+total1)>0 else 0.0
    pi01 = n01/(n00+n01) if (n00+n01)>0 else 0.0
    pi11 = n11/(n10+n11) if (n10+n11)>0 else 0.0
    def ll(p,n0,n1):
        if p in (0.0,1.0): return 0.0 if ((p==0.0 and n1==0) or (p==1.0 and n0==0)) else -np.inf
        return n0*np.log1p(-p)+n1*np.log(p)
    return -2.0*((ll(pi,total0,total1)) - (ll(pi01,n00,n01)+ll(pi11,n10,n11)))

def kupiec_pvalue(e: np.ndarray, a: float) -> float: return _chisq_sf(_lr_uc(e,a),1)
def christoffersen_ind_pvalue(e: np.ndarray) -> float: return _chisq_sf(_lr_ind(e),1)
def christoffersen_cc_pvalue(e: np.ndarray, a: float) -> float: return _chisq_sf(_lr_uc(e,a)+_lr_ind(e),2)

def lastn_band(alpha: float, n: int, conf: float=0.95) -> Tuple[int,int]:
    if n<=0: return (0,0)
    if binom is not None:
        lo, hi = binom.interval(conf, n, alpha)
        return int(lo), int(hi)
    mu, sd = n*alpha, np.sqrt(n*alpha*(1-alpha))
    return int(np.floor(mu-1.96*sd)), int(np.ceil(mu+1.96*sd))

def _fit_affine(y: np.ndarray, q: np.ndarray, tau: float, iters=400, lr=0.01):
    a,b = 0.0,1.0; y=y.astype(np.float64); q=q.astype(np.float64); n=max(1,y.size)
    for _ in range(iters):
        e = y - (a + b*q); g = (e<0).astype(np.float64) - tau
        a -= lr*(g.sum()/n); b -= lr*(np.sum(g*q)/n)
    return float(a), float(b)

def _fit_affine3(y: np.ndarray, q: np.ndarray, s: np.ndarray, tau: float, iters=600, lr=0.01):
    a,b,c = 0.0,1.0,0.0; n=max(1,y.size)
    y=y.astype(np.float64); q=q.astype(np.float64); s=s.astype(np.float64)
    for _ in range(iters):
        pred = a + b*q + c*s; g = (y - pred < 0).astype(np.float64) - tau
        a -= lr*(g.sum()/n); b -= lr*(np.sum(g*q)/n); c -= lr*(np.sum(g*s)/n)
    return float(a), float(b), float(c)

def _calibrate_fixed(ret, q, a, W): 
    if len(ret)<=W: e=_exc(ret,q); return e, float(e.mean()), len(e)
    d = np.quantile(ret[:W]-q[:W], a); q2 = q + d; e=_exc(ret[W:], q2[W:]); return e, float(e.mean()), len(e)
def _calibrate_rolling(ret,q,a,W,ema: Optional[float]):
    n=len(ret)
    if n<=W: e=_exc(ret,q); return e,float(e.mean()),n
    d=np.zeros(n); last=0.0
    for t in range(W,n):
        delta=np.quantile(ret[t-W:t]-q[t-W:t], a)
        if ema and 0.0<ema<1.0 and t>W: delta = ema*delta + (1-ema)*last
        d[t]=delta; last=delta
    q2=q+d; e=_exc(ret[W:], q2[W:]); return e,float(e.mean()),len(e)
def _calibrate_fixed_affine(ret,q,a,W,iters=300,lr=0.02):
    if len(ret)<=W: e=_exc(ret,q); return e,float(e.mean()),len(e)
    A,B=_fit_affine(ret[:W],q[:W],a,iters,lr); q2=A+B*q; e=_exc(ret[W:],q2[W:]); return e,float(e.mean()),len(e)
def _calibrate_rolling_affine(ret,q,a,W,iters=200,lr=0.02):
    n=len(ret)
    if n<=W: e=_exc(ret,q); return e,float(e.mean()),n
    A=np.zeros(n); B=np.ones(n)
    for t in range(W,n): A[t],B[t]=_fit_affine(ret[t-W:t],q[t-W:t],a,iters,lr)
    q2=A+B*q; e=_exc(ret[W:],q2[W:]); return e,float(e.mean()),len(e)
def _calibrate_fixed_affine3(ret,q,s,a,W,iters=400,lr=0.02):
    if len(ret)<=W: e=_exc(ret,q); return e,float(e.mean()),len(e)
    A,B,C=_fit_affine3(ret[:W],q[:W],s[:W],a,iters,lr); q2=A+B*q+C*s; e=_exc(ret[W:],q2[W:]); return e,float(e.mean()),len(e)
def _calibrate_rolling_affine3(ret,q,s,a,W,iters=300,lr=0.02):
    n=len(ret)
    if n<=W: e=_exc(ret,q); return e,float(e.mean()),n
    A=np.zeros(n); B=np.ones(n); C=np.zeros(n)
    for t in range(W,n):
        A[t],B[t],C[t]=_fit_affine3(ret[t-W:t],q[t-W:t],s[t-W:t],a,iters,lr)
    q2=A+B*q+C*s; e=_exc(ret[W:],q2[W:]); return e,float(e.mean()),len(e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--holdout_start", default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--calib_mode", choices=[
        "none","fixed","rolling","fixed_affine","rolling_affine","fixed_affine3","rolling_affine3"
    ], default="none")
    ap.add_argument("--calib_window", type=int, default=250)
    ap.add_argument("--roll_window", type=int, default=250)
    ap.add_argument("--calib_ema", type=float, default=0.0)
    ap.add_argument("--affine_lr", type=float, default=0.02)
    ap.add_argument("--affine_iters", type=int, default=300)
    ap.add_argument("--sigma_source", choices=["predicted","gk","none"], default="predicted")
    args = ap.parse_args()

    ROOT=Path("."); OUT=ROOT/"outputs"; TAB=ROOT/"tables"; FIG=ROOT/"figs"
    TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(OUT/"patch_preds.csv")
    need={"date","ret_true","q05_ret_pred"}
    if not need.issubset(df.columns): raise ValueError("patch_preds.csv missing required columns.")
    df["date"]=_to_datetime(df["date"]); df=df.sort_values("date")
    if args.holdout_start: df=df[df["date"]>=pd.to_datetime(args.holdout_start)]
    df=df.dropna(subset=["ret_true","q05_ret_pred"]).reset_index(drop=True)

    ret=df["ret_true"].to_numpy(np.float64)
    q  =df["q05_ret_pred"].to_numpy(np.float64)

    # scale regressor s_t (leak-safe)
    s=np.ones(len(df),dtype=np.float64)
    if args.sigma_source=="predicted" and "sigma2_pred" in df.columns:
        s=np.sqrt(np.clip(df["sigma2_pred"].to_numpy(np.float64),0.0,None))
    elif args.sigma_source=="gk":
        p=Path("data")/f"{str(args.symbol).lower()}_rv_clean.parquet"
        if p.exists():
            gk=pd.read_parquet(p)[["date","sigma_gk"]]
            gk["date"]=_to_datetime(gk["date"])
            df=df.merge(gk, on="date", how="left")
            df["sigma_gk_lag1"]=df["sigma_gk"].shift(1).ffill().bfill()
            s=df["sigma_gk_lag1"].to_numpy(np.float64)
        else:
            print(f"WARNING: {p} not found; using s=1.")

    # raw
    exc_raw=_exc(ret,q); br_raw=float(exc_raw.mean()) if exc_raw.size else np.nan
    cov_raw=1.0 - br_raw if np.isfinite(br_raw) else np.nan

    # calibrated
    mode=args.calib_mode
    if mode=="none": exc_cal,br_cal,N_cal=exc_raw,br_raw,len(exc_raw)
    elif mode=="fixed": exc_cal,br_cal,N_cal=_calibrate_fixed(ret,q,args.alpha,args.calib_window)
    elif mode=="rolling": exc_cal,br_cal,N_cal=_calibrate_rolling(ret,q,args.alpha,args.roll_window,args.calib_ema)
    elif mode=="fixed_affine": exc_cal,br_cal,N_cal=_calibrate_fixed_affine(ret,q,args.alpha,args.calib_window,args.affine_iters,args.affine_lr)
    elif mode=="rolling_affine": exc_cal,br_cal,N_cal=_calibrate_rolling_affine(ret,q,args.alpha,args.roll_window,args.affine_iters,args.affine_lr)
    elif mode=="fixed_affine3": exc_cal,br_cal,N_cal=_calibrate_fixed_affine3(ret,q,s,args.alpha,args.calib_window,args.affine_iters,args.affine_lr)
    elif mode=="rolling_affine3": exc_cal,br_cal,N_cal=_calibrate_rolling_affine3(ret,q,s,args.alpha,args.roll_window,args.affine_iters,args.affine_lr)
    else: raise ValueError(mode)
    cov_cal=1.0 - br_cal if np.isfinite(br_cal) else np.nan

    k_raw=kupiec_pvalue(exc_raw,args.alpha); ci_raw=christoffersen_ind_pvalue(exc_raw); cc_raw=christoffersen_cc_pvalue(exc_raw,args.alpha)
    k_cal=kupiec_pvalue(exc_cal,args.alpha); ci_cal=christoffersen_ind_pvalue(exc_cal); cc_cal=christoffersen_cc_pvalue(exc_cal,args.alpha)

    m=min(250,len(exc_cal)); last=int(exc_cal[-m:].sum()) if m>0 else 0; lo,hi=lastn_band(args.alpha,m)

    # write table
    out=pd.DataFrame([{
        "mode":mode,"breach_rate_raw":br_raw,"breach_rate_cal":br_cal,
        "coverage_raw":cov_raw,"coverage_cal":cov_cal,
        "kupiec_p_raw":k_raw,"kupiec_p_cal":k_cal,
        "christoffersen_ind_p_raw":ci_raw,"christoffersen_ind_p_cal":ci_cal,
        "christoffersen_cc_p_raw":cc_raw,"christoffersen_cc_p_cal":cc_cal,
        "effective_n":int(N_cal),"lastn":m,"lastn_breaches":last,"band_95pct":f"{lo}–{hi}",
    }])
    (TAB/"var_backtest.csv").write_text(out.to_csv(index=False))

    # plot
    # plot
    # --- plot: rolling 250-day breach count with 95% acceptance band ---
    # --- plot: rolling 250-day breach rate with 95% acceptance band ---
    if len(df) > 0:
        dates = df["date"].to_numpy()
        # choose which series to show (use calibrated series for whatever mode you evaluated)
        e_plot = exc_cal.astype(int)

        # window size (250 if available)
        W_plot = min(250, len(e_plot))
        roll_count = (
            pd.Series(e_plot, index=dates)
            .rolling(W_plot, min_periods=W_plot)
            .sum()
            .dropna()
        )

        # convert to rate
        roll_rate = roll_count / W_plot
        lo, hi = lastn_band(args.alpha, W_plot)
        lo_r, hi_r = lo / W_plot, hi / W_plot

        fig, ax = plt.subplots(figsize=(10, 3.5))
        # acceptance band across the whole x-range
        ax.axhspan(lo_r, hi_r, alpha=0.18, label=f"95% band (W={W_plot}) [{lo_r:.3f}–{hi_r:.3f}]")
        ax.axhline(args.alpha, lw=1.2, linestyle="--", label=f"target α={args.alpha:.2f}")

        ax.plot(roll_rate.index, roll_rate.values, lw=1.6, label=f"breach rate (last {W_plot}d)")
        ax.set_title(f"VaR rolling breach rate (α={args.alpha:.2f}) — mode={args.calib_mode}")
        ax.set_ylabel(f"breach rate (last {W_plot} days)")
        ax.set_xlabel("date")

        ymax = max(float(roll_rate.max() or 0), float(hi_r))
        ax.set_ylim(0, ymax + 0.05)

        ax.legend(loc="upper left", frameon=False)
        fig.tight_layout()
        (Path("figs")/"var_breach_timeline.png").parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(Path("figs")/"var_breach_timeline.png", dpi=140)
        plt.close(fig)

    a=args.alpha
    print(f"PatchTST VaR{int((1-a)*100)} exception_rate raw={br_raw:.4f}, {mode}={br_cal:.4f} (N_eff={int(N_cal)}, target={a:.4f})")
    print(f"             coverage         raw={cov_raw:.4f}, {mode}={cov_cal:.4f} (target={1.0-a:.4f})")
    print(f"             Kupiec p         raw={k_raw:.3f}, {mode}={k_cal:.3f}")
    print(f"             Christoffersen p (ind) raw={ci_raw:.3f}, {mode}={ci_cal:.3f}")
    print(f"             Christoffersen p (cc)  raw={cc_raw:.3f}, {mode}={cc_cal:.3f}")
    print(f"             last-{m} breaches ({mode}) = {last} in [{lo}–{hi}]")
    print("Wrote tables/var_backtest.csv")

if __name__=="__main__":
    main()
