here’s a no-frills, readable flow of how the pieces fit together — from raw data to the final headline + figures.

```
[Yahoo Finance: SPY daily OHLC]
            │
            ▼
┌───────────────────────────┐
│ src/ingest_yahoo_gk.py    │  downloads OHLC (auto-adjusted), computes GK RV
│  → data/spy_ohlc.parquet  │  RV_GK, sigma_gk
│  → data/spy_rv.parquet    │
└───────────────────────────┘
            │
            ▼
┌───────────────────────────┐
│ src/audit_rv.py           │  checks trading-day completeness, outliers, ADF(log RV),
│  → tables/summary.csv     │  dedups/sorts, minimal cleaning
│  → tables/missing_dates…  │
│  → figs/rv_*.png          │
│  → data/spy_rv_clean.parquet
└───────────────────────────┘
            │
            ├──────────────► (Baselines use the clean RV directly)
            │
            ▼
┌───────────────────────────┐
│ src/build_sequences.py    │  builds leak-safe windows (past T days → next day)
│  (uses adj_close returns) │  features: [ret, sigma_gk]
│  → outputs/spy_seq_120.npz│  targets: y_ret, y_lrv=log(RV_GK)
└───────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────────────────┐
│   Deep model branch (PatchTST)                                  │
│                                                                │
│  ┌───────────────────────────────┐   ┌────────────────────────┐ │
│  │ src/train_patchtst_quant.py   │   │ src/train_patchtst_…   │ │
│  │  τ=0.05 pinball on y_ret      │   │  multitask: y_ret +    │ │
│  │  → outputs/patch_preds.csv    │   │  MSE on y_lrv          │ │
│  │     (q05_ret_pred only)       │   │  → outputs/patch_preds.csv
│  └───────────────────────────────┘   │     (q05_ret_pred +    │ │
│                                      │      sigma2_pred)       │ │
│                                      └────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────┐
│ src/eval_phase4.py        │  VaR evaluation on PatchTST:
│  - modes: raw/fixed/rolling│  exceptions, Kupiec p, Christoffersen (ind, cc),
│  - leak-safe rolling shift │  last-250 breaches + binomial band
│  → tables/var_backtest.csv │
│  → figs/var_breach_timeline.png
└───────────────────────────┘
            │
            ├───────────────────────────────►┌───────────────────────────┐
            │                                 │ src/update_readme.py      │
            │                                 │  reads var_backtest.csv   │
            │                                 │  → updates README between │
            │                                 │     VAR_HEAD_START/END    │
            │                                 └───────────────────────────┘
            │
            └───────────────────────────────►┌───────────────────────────┐
                                              │ src/update_memo.py        │
                                              │  reads var_backtest.csv   │
                                              │  → fills {{…}} in docs/   │
                                              │     var_decision_memo.md  │
                                              └───────────────────────────┘
```

and the **baseline branch** in parallel:

```
data/spy_rv_clean.parquet
            │
            ├──► src/baseline_har.py
            │      log-HAR on log(RV): D (t-1), W (mean log over 5), M (mean log over 22)
            │      expanding refit, 1-step ahead
            │      → outputs/har_preds.csv
            │      → outputs/baseline_errors.json  (adds HAR block)
            │
            └──► src/baseline_garch_t.py
                   ConstantMean + GARCH(1,1) with Student-t, returns from adj_close
                   expanding refit, 1-step ahead
                   → outputs/garch_preds.csv
                   → outputs/baseline_errors.json  (adds GARCH block)
```

### leak-safety fences (where you can accidentally cheat)

* **Sequence build:** inputs only from `[t-T … t-1]`; targets at `t`. No overlap/duplicates allowed.
* **Train/holdout split:** `--split_date` enforces `train < split ≤ test`. We added assertions.
* **Calibration (eval\_phase4):**

  * **fixed:** δ from the first `calib_window` days of the evaluated slice; score only after that window.
  * **rolling:** δₜ from residuals on `[t-W … t-1]`; never includes day `t`.
* **Baselines:** both are **expanding** and **1-step ahead** (fit ≤ `t-1`, predict `t`). No multi-step forecasts.

### main artifacts you’ll look at

* `outputs/patch_preds.csv` — PatchTST VaR and (from multitask) σ² predictions per day.
* `tables/var_backtest.csv` — breach rate, Kupiec/Christoffersen p-values, N\_eff, last-250.
* `figs/var_breach_timeline.png` — step plot of rolling exceptions.
* `outputs/har_preds.csv`, `outputs/garch_preds.csv` — baseline variance forecasts.
* `outputs/baseline_errors.json` — HAR/GARCH holdout RMSE & QLIKE.
* `README.md` (headline updated), `docs/var_decision_memo.md` (placeholders filled).
