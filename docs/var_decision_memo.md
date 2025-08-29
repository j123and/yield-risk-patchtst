# Daily SPY VaR/Vol — Decision Memo

**Date:** 2025-08-29

## Headline (RAW, no calibration)
- VaR95 (PatchTST raw): exceptions **N/A**; Kupiec p≈N/A, Christoffersen ind p≈N/A, cc p≈N/A; N_eff=645.
- Last-250 breaches: **0** ∉ [6–20].


## Notes
- Sequences are leak-safe (past T→next day).
- Evaluation uses unconditional (Kupiec) and independence (Christoffersen) tests.
- No post-hoc calibration was applied to the quantile.
