# Intraday Volatility → VaR/ES (SPY) — HAR/GARCH vs PatchTST

Deep PatchTST (quantile + variance heads) versus classic HAR/GARCH for daily risk on SPY.  
Objective: forecast variance and produce a well-calibrated 95% VaR that passes standard back-tests.

![VaR95 breach timeline](figs/var_breach_timeline.png)

## Headline results (holdout 2023-01-02 → 2025-07-29)
- VaR(95%) — PatchTST (calibrated): coverage 5.12%, Kupiec p = 0.885, Christoffersen p = 0.227; last-250 breaches 18/250, within the 6–20 acceptance band.
- Variance forecasting: HAR has the best QLIKE; PatchTST variance head is competitive.

| model        | RMSE       | QLIKE  |
|--------------|------------|--------|
| HAR          | 0.000136   | -8.960 |
| GARCH_t      | 0.000229   | -8.764 |
| PatchTST_var | 0.000145   | -8.744 |

Full write-up: **[docs/var_decision_memo.md](docs/var_decision_memo.md)**

## What’s inside
- Data: Yahoo daily OHLC; realized variance via Garman–Klass proxy.
- Baselines: HAR-RV and GARCH(1,1) with Student-t errors.
- Deep model: PatchTST encoder with two heads — τ = 0.05 return quantile (direct VaR) and log-variance.
- Evaluation: RMSE/QLIKE for σ²; VaR back-tests (Kupiec, Christoffersen) + 95% binomial acceptance band; breach-timeline figure.
