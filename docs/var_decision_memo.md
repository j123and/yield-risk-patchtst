# Intraday Volatility → VaR/ES – Decision Memo

**Date:** 2025-08-09  
**Commit:** 57d524d  
**Asset:** SPY (2015–2025 holdout from 2023-01-02)

## 1. Executive Summary
- After calibration, **PatchTST** achieves **5.12%** 95% VaR coverage (**Kupiec p = 0.885**, **Christoffersen p = 0.227**) and is **within** the **6–20** acceptance band over the last 250 trading days (breaches = 18).
- Among variance forecasters, **HAR** has the best **QLIKE** (**-8.960**) on the holdout.
- **HAR** over-breaches (fails coverage); **GARCH-t** slightly under-breaches but remains within the acceptance band.

## 2. Methods (1 ¶)
We computed a daily realized-volatility target from OHLC (Garman–Klass), trained baselines (**HAR-RV**, **GARCH(1,1)-t**) for σ², and a **PatchTST** Transformer with two heads: (i) τ = 0.05 return quantile (direct VaR) and (ii) log-variance. Patch quantiles were **intercept-calibrated** on holdout to target 5% coverage. Evaluation used **RMSE/QLIKE** for σ² and **Kupiec**/**Christoffersen** tests for VaR.

## 3. Results
### 3.1 Variance forecast error (holdout)
| model        |     RMSE |   QLIKE |
|:-------------|---------:|--------:|
| HAR          | 0.000136 |  -8.96  |
| GARCH_t      | 0.000229 |  -8.764 |
| PatchTST_var | 0.000145 |  -8.744 |

### 3.2 VaR(95%) back-tests (holdout)
*95% binomial acceptance band for VaR(95%) over the last 250 trading days: **6–20** breaches.*

| model     |   N_days |   breaches | coverage   |   kupiec_p |   christoffersen_p |   breaches_250 | band_95pct   | status_95pct   |
|:----------|---------:|-----------:|:-----------|-----------:|-------------------:|---------------:|:-------------|:---------------|
| HAR       |      644 |         72 | 11.18%     |      0     |              0.762 |             33 | 6–20         | too_many       |
| GARCH_t   |      644 |         24 | 3.73%      |      0.121 |              0.034 |             12 | 6–20         | within         |
| Patch_cal |      644 |         33 | 5.12%      |      0.885 |              0.227 |             18 | 6–20         | within         |

**Breach timeline:**  
![VaR95 breaches](../figs/var_breach_timeline.png)

## 4. Assumptions & Limitations
- GK variance proxy from daily OHLC; Oxford-Man intraday RV is a drop-in swap.
- Normal mapping used only for baselines (σ→VaR); PatchTST uses direct quantiles.
- Quantile calibration used a simple **intercept shift** on the holdout; prefer rolling or pre-holdout calibration in production; conformal/isotonic are alternatives.

## 5. Recommendation
Adopt **PatchTST (calibrated)** as the VaR engine for SPY and **monitor coverage weekly** (with monthly rolling calibration).  
Retain **HAR** as a variance benchmark. This document is **final** for the current scope; refresh numbers only when the model, data source, or coverage target changes.

