# Intraday Volatility → VaR/ES – Decision Memo (Provisional)

**Date:** 2025-08-25

## Headline
- VaR95 (PatchTST calibrated): exceptions 6.85% (Kupiec p≈0.277, Christoffersen p≈0.325); last-250 breaches 20 ∈ [6–20]; effective N=394.
- Variance: HAR RMSE≈1.355251e-04, QLIKE≈-8.960.

## Notes
- VaR calibration: rolling intercept on residuals (see eval configs); stats reported after warm-up.
- σ² evaluated with RMSE and QLIKE on holdout.
- No look-ahead in sequence construction.

*Regenerate this memo after retraining or updating evaluation tables.*
