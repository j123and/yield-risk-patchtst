#!/usr/bin/env python3
"""
Generate a brief decision memo from the evaluation CSVs.

Fixes vs prior version:
- Explicitly picks the rolling-calibrated Patch row (not raw).
- Reports VaR *exceptions* (breach rate), not coverage.
- Correctly marks last-250 membership as ∈ or ∉ the acceptance band.
- Uses timezone-aware date (no deprecation warning).
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, UTC
import argparse
import re
import pandas as pd


def _pick_patch_row(bt: pd.DataFrame) -> pd.Series:
    """Prefer rolling-calibrated Patch row; then fixed; then raw."""
    if bt.empty:
        raise ValueError("tables/var_backtest.csv is empty")

    # Normalize string columns for robust matching
    bt_norm = bt.copy()
    for col in ("model", "mode"):
        if col in bt_norm.columns:
            bt_norm[col] = bt_norm[col].astype(str).str.lower()

    # Priority order
    prefs = [
        ("mode", "rolling"),
        ("model", "patch_cal"),
        ("mode", "fixed"),
        ("model", "patch_fixed"),
        ("mode", "none"),
        ("model", "patch_raw"),
    ]
    for col, val in prefs:
        if col in bt_norm.columns:
            hit = bt_norm[bt_norm[col] == val]
            if not hit.empty:
                return bt.loc[hit.index[0]]

    # Fallback: first row
    return bt.iloc[0]


def _parse_band(band_str: str) -> tuple[int | None, int | None]:
    """Parse '6–20' or '6-20' → (6, 20); return (None, None) if not parseable."""
    if not isinstance(band_str, str):
        return None, None
    s = band_str.strip().replace("–", "-")
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", s)
    if not m:
        return None, None
    lo, hi = int(m.group(1)), int(m.group(2))
    return lo, hi


def build_memo(err: pd.DataFrame, bt: pd.DataFrame) -> str:
    # Choose the Patch row to report (rolling if available)
    row = _pick_patch_row(bt)

    # Extract metrics with safe fallbacks
    br = float(row.get("breach_rate", float("nan")))   # exceptions
    cov = float(row.get("coverage", float("nan")))
    kup = float(row.get("kupiec_p", float("nan")))
    chi = float(row.get("christoffersen_p", float("nan")))
    eff_n = int(row.get("effective_n", 0))
    last250 = int(row.get("last250_breaches", 0))
    band_str = str(row.get("band_95pct", "") or "")
    lo, hi = _parse_band(band_str)
    in_band = (lo is not None and hi is not None and lo <= last250 <= hi)
    membership = "∈" if in_band else "∉"

    # Variance model summary (best QLIKE if present)
    var_line = ""
    if not err.empty:
        err_sorted = err.sort_values("QLIKE") if "QLIKE" in err.columns else err
        best = err_sorted.iloc[0]
        var_line = (
            f"- Variance: {best.get('model','HAR')} "
            f"RMSE≈{float(best.get('RMSE', float('nan'))):.6e}, "
            f"QLIKE≈{float(best.get('QLIKE', float('nan'))):.3f}."
        )

    today = datetime.now(UTC).date().isoformat()

    md = f"""# Intraday Volatility → VaR/ES – Decision Memo (Provisional)

**Date:** {today}

## Headline
- VaR95 (PatchTST calibrated): exceptions {br*100:.2f}% (Kupiec p≈{kup:.3f}, Christoffersen p≈{chi:.3f}); last-250 breaches {last250} {membership} [{band_str}]; effective N={eff_n}.
{var_line}

## Notes
- VaR calibration: rolling intercept on residuals (see eval configs); stats reported after warm-up.
- σ² evaluated with RMSE and QLIKE on holdout.
- No look-ahead in sequence construction.

*Regenerate this memo after retraining or updating evaluation tables.*
"""
    return md


def main(out_path: str = "docs/var_decision_memo.md") -> None:
    err = pd.read_csv("tables/error_metrics.csv") if Path("tables/error_metrics.csv").exists() else pd.DataFrame()
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
