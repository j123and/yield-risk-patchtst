# Project spec — Intraday Volatility → VaR/ES (PatchTST-P vs HAR-RV/GARCH-t)

## 0.1 Research question
Can a probabilistic Transformer (PatchTST-P) reduce 1-day 95% VaR breaches versus a tuned HAR-RV and/or GARCH-t baseline for the SPY ETF over 2015-01-02 → 2025-07-31?

## 0.2 Asset universe
Primary: SPY (S&P 500 ETF)
Stretch: AAPL (single name), QQQ (Nasdaq-100)
Data source: Oxford-Man realised-variance via FRED (free). (Polygon minute bars optional later.)

## 0.3 Model pair
Baseline: HAR-RV (lags 1,5,22) **and/or** GARCH-t (Student-t errors)
Upgrade: PatchTST-P (probabilistic Transformer; τ = 0.95 quantile head or MDN head)

## 0.4 Metrics & tests
Point error: RMSE, QLIKE (on σ²)
Quantile loss: Pinball loss at τ = 0.95
Risk: 95% VaR & ES (from σ̂ for baselines; direct τ-quantile from PatchTST-P)
Back-tests: Kupiec LR_uc, Christoffersen LR_cc
Comparative test: Diebold–Mariano on QLIKE or pinball losses

## 0.5 Success criteria
≥ 20% reduction in 95% VaR breaches with p < 0.05 (DM test), or equal breaches with ≥ 2× faster inference / simpler ops.

## 0.6 Timeline & resources
Phases 1–7 over ~5 weeks. Train PatchTST-P on Kaggle GPU; no paid data unless Polygon is added later.

## 0.7 Risks & mitigations
- Data gaps / calendar issues → audit & gap policy
- GPU quota overruns → mixed precision, early stopping, resume checkpoints
- Convergence issues → LR range test, gradient clipping, patience

## 0.8 Project board
Link will be added in README after board creation.
