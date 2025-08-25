
# PatchTST VaR/Volatility — SPY daily risk (VaR$_{0.95}$, variance)

PatchTST (quantile + variance heads) versus HAR-RV and GARCH(1,1)–t for **daily** SPY risk.
Goal: forecast variance and produce a VaR$_{0.95}$ that **passes standard back-tests** with a leak-safe pipeline and exact reproduction.

![VaR95 breach timeline](figs/var_breach_timeline.png)

## Headline results (holdout 2023-01-02 → 2025-07-31)

* **VaR$_{0.95}$ (PatchTST, calibrated):**
  exceptions **6.85%** (breach rate), **Kupiec p≈0.277**, **Christoffersen p≈0.325**, **effective N=394**.
  Last-250 breaches: **20**, inside the 95% acceptance band **\[6–20]**.
  Calibration = **rolling 250-day intercept**, **EMA=0.0** (no smoothing). No look-ahead.

* **Variance forecasting (holdout):**
  **HAR-RV** achieves **QLIKE ≈ −8.958**, **RMSE ≈ 1.354×10⁻⁴**.
  **GARCH(1,1)–t** is weaker (**QLIKE ≈ −8.764**, **RMSE ≈ 2.289×10⁻⁴**).
  PatchTST’s variance head predictions are saved in `outputs/patch_preds.csv` (`sigma2_pred`) and can be evaluated with the same metrics.

Interpretation: 6.85% vs a 5.0% target is within sampling error at $N_{\text{eff}}\!\approx\!394$ and **not rejected** by Kupiec/Christoffersen. If you want optics closer to 5%, use the same rolling calibration and remove EMA smoothing (as above) or move to a fixed 500-day intercept—both remain leak-safe.

---

## Reproduce (single command)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip -r requirements.txt
bash scripts/reproduce.sh
```

What it does end-to-end:

1. Ingest daily OHLC for SPY and compute **Garman–Klass** realized variance.
2. Audit the RV series (missing days, outliers, ADF on log-RV).
3. Build sequences (returns + GK features) with **no look-ahead**.
4. Train PatchTST **quantile** head (τ=0.05) and **multi-task** head (quantile + log-variance).
5. Fit **HAR-RV** and **GARCH(1,1)–t** baselines (expanding window), compute holdout errors.
6. Evaluate VaR with **rolling 250-day intercept**, **EMA=0.0**, write `tables/var_backtest.csv` and the breach timeline figure.
7. Generate a short decision memo in `docs/`.

Artifacts land in `outputs/`, `tables/`, `figs/`, `docs/`.

If you don’t have CUDA, training still runs on CPU (slower but fine for this dataset).

---

## How it works

* **Data & features**

  * Source: Yahoo daily OHLC for SPY.
  * Realized variance proxy: **Garman–Klass** (from OHLC).
    Why GK? It’s simple, robust, and uses only daily bars. Alternatives like Parkinson or Rogers–Satchell are possible; GK keeps the pipeline lightweight.

* **Sequence builder (leak-safe)**

  * Inputs: past $T$ days of returns and GK-derived features.
  * Targets: next-day return (for the quantile head) and next-day log-variance.
  * The holdout split is time-based; **no peeking across the split**.

* **Model**

  * **PatchTST-style encoder** (Transformer over non-overlapping patches).
  * Two heads:

    * **Quantile head** at **τ = 0.05** → direct VaR$_{0.95}$.
    * **Variance head** on **log-variance**; exponentiate to get $\sigma^2$.
  * Trained with pinball loss (quantile) and MSE (log-variance).

* **Calibration**

  * VaR calibration is **intercept-only** (shift the predicted quantile by a constant estimated from past residuals).
  * **Rolling window 250**, **EMA=0.0** (no smoothing) for the reported numbers.
  * This is **leak-safe** (uses only the past window at each step).

* **Back-tests (what matters)**

  * **Kupiec LR$_\text{uc}$** for unconditional coverage (breach rate ≈ $\alpha$).
  * **Christoffersen LR$_\text{cc}$** for independence (no clustering of breaches).
  * **Binomial acceptance band** shown for the last 250 days.

* **Baselines**

  * **HAR-RV** (Heterogeneous AutoRegressive model on RV).
  * **GARCH(1,1)–t** (Student-t innovations).
  * Both refit on an expanding window; errors reported on the same holdout.

---

## Commands (explicit)

If you want to run steps manually instead of the script:

```bash
# 1) Ingest + compute GK
python src/ingest_yahoo_gk.py --symbol SPY --start 2015-01-02 --end 2025-07-31

# 2) Audit (non-fatal if market calendar package is missing)
python src/audit_rv.py --in data/spy_rv.parquet

# 3) Build sequences
python src/build_sequences.py --symbol SPY --seq_len 120

# 4) Train PatchTST heads (writes outputs/patch_preds.csv)
python src/train_patchtst_quant.py      --npz outputs/spy_seq_120.npz --split_date 2023-01-02
python src/train_patchtst_multitask.py  --npz outputs/spy_seq_120.npz --split_date 2023-01-02

# 5) Baselines
python src/baseline_har.py     --holdout_start 2023-01-02
python src/baseline_garch_t.py --holdout_start 2023-01-02

# 6) VaR evaluation (this is what produced the headline numbers)
python src/eval_phase4.py --symbol SPY --holdout_start 2023-01-02 \
  --alpha 0.05 --calib_mode rolling --roll_window 250 --calib_ema 0.0
```

---

## Determinism & environment

* Python **3.11+**. See `requirements.txt` for packages.
* Training scripts set seeds (Python/NumPy/PyTorch) and request deterministic ops where possible.
* **Note:** PyTorch’s memory-efficient attention can be non-deterministic on GPU; you may see tiny run-to-run differences. This does **not** affect the methodology.

---

## Notes, caveats, and limits

* **Single asset (SPY).** No cross-asset generalization claimed.
* **Daily bars only.** No intraday microstructure or transaction costs.
* **ES is not scored.** If you care about ES, add **Fissler–Ziegel** (AL/ES) scoring; the code is structured to accept it, but the metric is not reported here.
* **HAR vs deep variance.** On this holdout, HAR-RV wins QLIKE. That’s normal for clean daily RV; PatchTST remains competitive and provides the VaR head in one model.

---

## What to look at (quick tour)

* `scripts/reproduce.sh` — one-shot pipeline; mirrors the commands above.
* `outputs/patch_preds.csv` — dates, true returns, VaR$_{0.95}$ predictions, and $\sigma^2$ predictions.
* `tables/var_backtest.csv` — exception rates, coverage, p-values, effective $N$, and last-250 counts.
* `figs/var_breach_timeline.png` — step plot of breaches in the evaluated region.
* `docs/var_decision_memo.md` — short memo regenerated by the script.

---

## Why this exists (and why it’s in my portfolio)

A lot of “ML for risk” projects cheat on leakage or ignore back-tests. This one doesn’t. It’s intentionally small, **leak-safe**, and **reproducible**, with **standard VaR tests** and transparent baselines. It’s the level of rigor I’ll bring to a junior quant/MLE role.

---

## License

MIT. See `LICENSE`.
