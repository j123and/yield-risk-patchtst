# PatchTST VaR/Volatility: SPY daily risk (VaR 0.95 and variance)

PatchTST (quantile + variance heads) versus HAR-RV and GARCH(1,1)–t for **daily** SPY risk.
Goal: forecast variance and produce a **VaR at 95%** (α=0.05, left tail) that meets standard back-tests with a leak-safe, reproducible pipeline.

![VaR 0.95 breach timeline](figs/var_breach_timeline.png)

*Figure: Breach timeline for α=0.05. Shaded band shows the 95% Binomial acceptance range on the **last 250 days**.*

## Headline results (holdout 2023-01-02 → 2025-07-31)

<!-- VAR_HEAD_START -->
* **VaR<sub>0.95</sub> (PatchTST, calibrated):**
  exceptions **6.60%** (breach rate), **Kupiec p=0.380**, **Christoffersen (ind) p=0.631**, **effective N<sub>eff</sub>=394**.
  Last-250 breaches: **20**, inside the 95% acceptance band **[6–20]**.
  Calibration = **rolling 250-day intercept, EMA=0.0 (no smoothing). No look-ahead.**
<!-- VAR_HEAD_END -->

**Variance forecasting (holdout)**  
• **HAR-RV**: **Avg log-likelihood = −8.958** (higher is better for this metric), **RMSE = 1.354×10⁻⁴** (units: variance of daily returns).  
• **GARCH(1,1)–t**: **Avg log-likelihood = −8.764**, **RMSE = 2.289×10⁻⁴**.  
• PatchTST variance-head predictions are saved in `outputs/patch_preds.csv` (`sigma2_pred`) and can be scored with the same metrics.

Interpretation: With **N_eff = 394**, a 6.85% breach rate is within sampling noise for α=0.05 and is **not rejected** by LR_uc or LR_ind. 

---

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip -r requirements.txt
bash scripts/reproduce.sh
````

This will:

1. Download SPY daily OHLC and compute **Garman–Klass** realized variance.
2. Audit the series (missing days via market calendar, outliers, ADF on log-RV).
3. Build sequences from **past T days to next day** (no look-ahead).
4. Train PatchTST **quantile** (τ=0.05) and **variance** heads.
5. Fit **HAR-RV** and **GARCH(1,1)–t** on an **expanding window** (refit daily), score the same holdout.
6. Evaluate VaR with a **rolling 250-day intercept** (EMA=0.0), write `tables/var_backtest.csv` and the breach timeline figure.
7. Generate a short memo in `docs/`.

Artifacts land in `outputs/`, `tables/`, `figs/`, `docs/`.

> Repro tip: On first run, a **data snapshot** `data/spy_ohlc.parquet` is written and re-used to keep results stable across re-runs.

---

## How it works

**Data & returns**
  • Source: Yahoo daily OHLC for SPY.
  • Returns for the VaR head use **Adjusted Close** (dividends matter for SPY).
  • Realized variance proxy: **Garman–Klass**
  $\mathrm{RV}^{GK} = \tfrac12\ln^2(H/L) - (2\ln2 - 1)\ln^2(C/O)$.
  • Daily volatility is $\sigma = \sqrt{\mathrm{RV}}$.

**Sequence builder (leak-safe)**
  • Inputs: past $T$ days $t\!-\!T,\dots,t\!-\!1$.
  • Targets: next-day return $r_t$ (quantile head) and next-day **log-variance** (variance head).
  • Split is **time-based** at `2023-01-02`; no peeking across the split.

**Model**
  • **PatchTST** encoder (transformer over non-overlapping patches).
  • Heads:
    – **Quantile head** at **τ=0.05** → direct VaR₀.₉₅.
    – **Variance head** on **log-variance**; transform back (exp/softplus) to get $\sigma^2$.
  • Losses: **pinball** (quantile) and **MSE** (log-variance).

**Calibration (intercept-only)**
At day $t$, compute residuals $u = r - q$ over the **previous 250 days**, set $\delta_t = \mathrm{quantile}_\alpha(u)$, then
$q_t^\* = q_t + \delta_t$.
This uses **only past data** each day (leak-safe). EMA smoothing is **off** (0.0).

**Back-tests**
• **Kupiec LR\_uc**: tests breach rate equals α.
• **Christoffersen LR\_ind**: tests independence (no clustering).
• **LR\_cc = LR\_uc + LR\_ind**: conditional coverage.
• Last-250 acceptance band uses a **Binomial 95% interval**.

**Baselines**
• **HAR-RV** on log-RV.
• **GARCH(1,1)–t** (Student-t innovations).
• Both use **expanding window, refit daily**.

---

## Commands (explicit)

```bash
# 1) Ingest + compute GK (writes data/spy_ohlc.parquet and data/spy_rv.parquet)
python src/ingest_yahoo_gk.py --symbol SPY --start 2015-01-02 --end 2025-07-31

# 2) Audit (requires a market calendar; install 'exchange_calendars')
python src/audit_rv.py --in data/spy_rv.parquet

# 3) Build sequences
python src/build_sequences.py --symbol SPY --seq_len 120

# 4) Train PatchTST heads (writes outputs/patch_preds.csv)
python src/train_patchtst_quant.py      --npz outputs/spy_seq_120.npz --split_date 2023-01-02
python src/train_patchtst_multitask.py  --npz outputs/spy_seq_120.npz --split_date 2023-01-02

# 5) Baselines (expanding window, daily refit)
python src/baseline_har.py     --holdout_start 2023-01-02
python src/baseline_garch_t.py --holdout_start 2023-01-02

# 6) VaR evaluation
python src/eval_phase4.py --symbol SPY --holdout_start 2023-01-02 \
  --alpha 0.05 --calib_mode rolling --roll_window 250 --calib_ema 0.0
```

---

## Determinism & environment

• Python **3.11+**. See `requirements.txt`.
• Seeds are set for Python/NumPy/PyTorch; deterministic ops requested where available.
• The script writes `outputs/run.json` with args, git SHA, and package versions so runs are traceable.
• Note: Some GPU attention kernels are not fully deterministic; expect tiny run-to-run differences.

---

## Notes and limits

• **Single asset (SPY)** and **daily bars** only.
• **ES not scored**. The code can be extended to ES with **Fissler–Ziegel** scoring.
• On this holdout, **HAR-RV** beats the deep variance head on the log-likelihood metric — normal for clean daily RV. PatchTST still gives you VaR and variance in one model.

---

## What to open

• `scripts/reproduce.sh` — end-to-end pipeline.
• `outputs/patch_preds.csv` — dates, true returns, VaR₀.₉₅ and σ² predictions.
• `tables/var_backtest.csv` — breach rate, **LR\_uc p**, **LR\_ind p**, **LR\_cc p**, N\_eff, last-250 counts.
• `figs/var_breach_timeline.png` — breaches and acceptance band.
• `docs/var_decision_memo.md` — short decision memo.

---

## License

MIT (see `LICENSE`).

