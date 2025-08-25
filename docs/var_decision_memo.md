# Intraday Volatility → VaR/ES – Decision Memo (Provisional)

**Date:** 2025-08-25

## Headline
- VaR95 (PatchTST calibrated): coverage 100.00% (Kupiec p≈0.000, Christoffersen p≈1.000); last-250 breaches 0 ∈ 6–20; effective N=644.
- Variance: HAR RMSE≈1.355251e-04, QLIKE≈-8.960.

## Notes
- VaR calibration: rolling 250-day intercept on residuals, EMA=0.20; stats reported after warm-up.
- σ² evaluated with RMSE and QLIKE on holdout.
- No look-ahead in sequence construction.

*Regenerate this memo after retraining or updating evaluation tables.*
