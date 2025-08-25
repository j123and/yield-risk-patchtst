#!/usr/bin/env python3
"""
Generate a brief decision memo from the evaluation CSVs.
- NO side effects on import.
- Usage:
    python src/make_memo.py --out docs/var_decision_memo.md
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import argparse
import pandas as pd


def build_memo(err: pd.DataFrame, bt: pd.DataFrame) -> str:
    # Pick best variance model by QLIKE (lower is better)
    best_var = err.sort_values("QLIKE").iloc[0] if "QLIKE" in err else None

    # Patch (calibrated) VaR row if present
    patch_row = None
    if "model" in bt.columns:
        rows = bt.loc[bt["model"].astype(str).str.contains("Patch", case=False, na=False)]
        if not rows.empty:
            patch_row = rows.sort_values("coverage_diff_abs").iloc[0] if "coverage_diff_abs" in rows else rows.iloc[0]

    now = datetime.utcnow().strftime("%Y-%m-%d")

    va_line = ""
    if patch_row is not None:
        kupiec_p = patch_row.get("kupiec_p", float("nan"))
        christ_p = patch_row.get("christoffersen_p", float("nan"))
        cov = patch_row.get("coverage", float("nan"))
        last250 = patch_row.get("last250_breaches", "")
        band = patch_row.get("band_95pct", "")
        eff_n = patch_row.get("effective_n", "")
        va_line = (
            f"- VaR95 (PatchTST calibrated): coverage {cov:.2%} "
            f"(Kupiec p≈{float(kupiec_p):.3f}, Christoffersen p≈{float(christ_p):.3f}); "
            f"last-250 breaches {last250} ∈ {band}; effective N={eff_n}."
        )

    var_line = ""
    if best_var is not None:
        var_line = (
            f"- Variance: {best_var.get('model','(unknown)')} "
            f"RMSE≈{float(best_var.get('RMSE', float('nan'))):.6e}, "
            f"QLIKE≈{float(best_var.get('QLIKE', float('nan'))):.3f}."
        )

    md = f"""# Intraday Volatility → VaR/ES – Decision Memo (Provisional)

**Date:** {now}

## Headline
{va_line}
{var_line}

## Notes
- VaR calibration: rolling 250-day intercept on residuals, EMA=0.20; stats reported after warm-up.
- σ² evaluated with RMSE and QLIKE on holdout.
- No look-ahead in sequence construction.

*Regenerate this memo after retraining or updating evaluation tables.*
"""
    return md


def main(out_path: str = "docs/var_decision_memo.md") -> None:
    err = pd.read_csv("tables/error_metrics.csv")
    bt = pd.read_csv("tables/var_backtest.csv")
    md = build_memo(err, bt)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(md, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/var_decision_memo.md")
    args = ap.parse_args()
    main(args.out)
