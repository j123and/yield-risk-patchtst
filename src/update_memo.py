#!/usr/bin/env python3
from pathlib import Path
from datetime import date
import subprocess
import pandas as pd

# Load results
err = pd.read_csv("tables/error_metrics.csv")
bt  = pd.read_csv("tables/var_backtest.csv")

# Helpers
def pct(x): return f"{100*float(x):.2f}%"
def r6(x):  return f"{float(x):.6f}"
def r3(x):  return f"{float(x):.3f}"

# Choose Patch row
row_patch = bt.loc[bt["model"]=="Patch_cal"].iloc[0]

# Column compatibility
n_eff_col  = "N_effective" if "N_effective" in bt.columns else "N_days"
lastN_col  = "breaches_lastN" if "breaches_lastN" in bt.columns else ("breaches_250" if "breaches_250" in bt.columns else None)

n_eff  = int(row_patch[n_eff_col])
b_last = int(row_patch[lastN_col]) if lastN_col else None
cov    = pct(row_patch["coverage"])
kupiec = r3(row_patch["kupiec_p"])
christ = r3(row_patch["christoffersen_p"])
band   = str(row_patch.get("band_95pct", "—")).replace("-", "–")
status = str(row_patch.get("status_95pct", "—")).split('[')[0]

# Best QLIKE
best_qlike_row = err.loc[err["QLIKE"].idxmin()]
best_qlike_model, best_qlike_val = best_qlike_row["model"], r3(best_qlike_row["QLIKE"])

# Pretty tables
err_out = err.copy()
err_out["RMSE"]  = err_out["RMSE"].map(r6)
err_out["QLIKE"] = err_out["QLIKE"].map(r3)
err_md = err_out.to_markdown(index=False)

bt_main = bt[bt["model"].isin(["HAR","GARCH_t","Patch_cal"])].copy()
bt_main["coverage"] = bt_main["coverage"].map(pct)
for c in ("kupiec_p","christoffersen_p"):
    if c in bt_main: bt_main[c] = bt_main[c].map(r3)
if "band_95pct" in bt_main:
    bt_main["band_95pct"] = bt_main["band_95pct"].astype(str).str.replace("-", "–")
if "status_95pct" in bt_main:
    bt_main["status_95pct"] = bt_main["status_95pct"].astype(str).str.replace(r"\[.*\]","",regex=True)
bt_md = bt_main.to_markdown(index=False)

# Commit hash
try:
    commit = subprocess.check_output(["git","rev-parse","--short","HEAD"]).decode().strip()
except Exception:
    commit = "N/A"

CALIB_NOTE = f"Rolling 250-day intercept calibration on residuals (quantile p=0.035, EMA=0.20); effective N = {n_eff}"
b_last_str = f"; last-window breaches = **{b_last}**" if b_last is not None else ""

md = f"""# Intraday Volatility: VaR/ES Decision Memo

**Date:** {date.today()}  
**Commit:** {commit}  
**Asset:** SPY (2015–2025 holdout from 2023-01-02)

## 1. Executive Summary
- PatchTST (calibrated) coverage **{cov}** at VaR95 (**Kupiec p = {kupiec}**, **Christoffersen p = {christ}**), status **{status}** the **{band}** acceptance band{b_last_str}.
- Calibration: {CALIB_NOTE}.
- Variance: **{best_qlike_model}** has best **QLIKE** (**{best_qlike_val}**); PatchTST variance head is competitive.

## 2. Methods
Daily realized-variance target from OHLC (Garman–Klass). Baselines: HAR-RV and GARCH(1,1)-t (map σ̂→VaR with Normal). Deep model: PatchTST with two heads, τ=0.05 return quantile (direct VaR) and log-variance. Calibration uses a rolling intercept shift fit on trailing 250 days. Evaluation uses RMSE/QLIKE for σ² and Kupiec/Christoffersen for VaR; rolling back-tests reported.

## 3. Results
### 3.1 Variance forecast error (holdout)
{err_md}

### 3.2 VaR(95%) back-tests (holdout)
{bt_md}

**Breach timeline:**  
![VaR95 breaches](../figs/var_breach_timeline.png)

## 4. Assumptions & Limitations
- GK variance proxy from daily OHLC; Oxford-Man realized measures are a drop-in replacement.
- Baselines use Normal mapping; PatchTST predicts tail quantiles directly.
- Calibration is rolling and causal; parameters as stated above.

## 5. Recommendation
Adopt PatchTST with rolling calibration as the VaR(95) engine; monitor weekly and alert if last-window breaches exit the acceptance band. Retain HAR as a variance benchmark.
"""
Path("docs").mkdir(parents=True, exist_ok=True)
Path("docs/var_decision_memo.md").write_text(md, encoding="utf-8")
print("Wrote docs/var_decision_memo.md")
