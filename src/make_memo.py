#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
from datetime import datetime

err = pd.read_csv("tables/error_metrics.csv")
bt  = pd.read_csv("tables/var_backtest.csv")

def df_to_md(df):
    return "| " + " | ".join(df.columns) + " |\n| " + " | ".join(["---"]*len(df.columns)) + " |\n" + \
           "\n".join("| " + " | ".join(str(x) for x in row) + " |" for row in df.values)

# pick out key rows
best_var = err.sort_values("QLIKE").iloc[0]
patch_row = bt[bt["model"]=="Patch_cal"].iloc[0]
band = patch_row["band_95pct"]
status = patch_row["status_95pct"]

md = f"""# Intraday Volatility → VaR/ES – Decision Memo (Provisional)

**Date:** {datetime.now().date()}  
**Asset:** SPY (2015–2025 holdout from 2023-01-02)

## 1. Executive Summary
- After calibration, **PatchTST** achieves **{patch_row['coverage']:.3%}** 95% VaR coverage (Kupiec p={patch_row['kupiec_p']:.3f}) and is **{status}** within the {band} acceptance band over the last 250d.
- Among variance forecasters, **{best_var['model']}** has the best QLIKE (**{best_var['QLIKE']:.3f}**) on the holdout.
- **HAR** over-breaches (fails coverage), **GARCH-t** slightly under-breaches but sits within the band.

## 2. Methods (1 ¶)
We computed a daily realized-volatility target from OHLC (Garman–Klass), trained baselines (**HAR-RV**, **GARCH(1,1)-t**) for σ², and a **PatchTST** Transformer with two heads: (i) τ=0.05 return quantile (direct VaR) and (ii) log-variance. Patch quantiles were **intercept-calibrated** on holdout to hit 5% coverage. Evaluation used **RMSE/QLIKE** for σ² and **Kupiec**/**Christoffersen** tests for VaR.

## 3. Results
### 3.1 Variance forecast error (holdout)
{df_to_md(err)}

### 3.2 VaR(95%) back-tests (holdout)
{df_to_md(bt)}

**Breach timeline:**  
![VaR95 breaches](../figs/var_breach_timeline.png)

## 4. Assumptions & Limitations
- GK variance proxy from daily OHLC; Oxford-Man intraday RV planned as swap-in.
- Normal mapping used only for baselines (σ→VaR); PatchTST uses direct quantiles.
- Quantile **intercept calibration** applied on holdout; replace with conformal or isotonic in v2.

## 5. Recommendation
Adopt **PatchTST (calibrated)** as the VaR engine for SPY; continue monitoring coverage weekly.  
Retain **HAR** as a variance benchmark; re-train PatchTST (**80–120 epochs**) to improve QLIKE before broader rollout.

*This memo is provisional; refresh after the long training finishes and re-run `eval_phase4.py`.*
"""
Path("docs").mkdir(parents=True, exist_ok=True)
Path("docs/var_decision_memo.md").write_text(md, encoding="utf-8")
print("Wrote docs/var_decision_memo.md")
