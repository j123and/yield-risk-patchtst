#!/usr/bin/env python3
"""
Update fields in the memo after evaluation runs.
- NO side effects on import.
- Usage:
    python src/update_memo.py --memo docs/var_decision_memo.md
"""
from __future__ import annotations
from pathlib import Path
from datetime import date
import argparse
import subprocess
import pandas as pd


def update_memo(memo_path: str) -> None:
    # Load results (non-fatal if columns differ)
    err = pd.read_csv("tables/error_metrics.csv")
    bt = pd.read_csv("tables/var_backtest.csv")

    # Example: extract Patch_cal row (if present)
    row_patch = None
    if "model" in bt.columns:
        sel = bt.loc[bt["model"].astype(str).str.contains("Patch", case=False, na=False)]
        if not sel.empty:
            row_patch = sel.sort_values("coverage_diff_abs").iloc[0] if "coverage_diff_abs" in sel else sel.iloc[0]

    # Commit hash (best-effort)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        commit = "N/A"

    p = Path(memo_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = p.read_text(encoding="utf-8")

    # Replace simple placeholders if present
    text = text.replace("{{DATE}}", str(date.today()))
    text = text.replace("{{GIT_COMMIT}}", commit)

    # Optionally inject a tiny status line
    if row_patch is not None:
        inject = (
            f" (Kupiec p≈{float(row_patch.get('kupiec_p', float('nan'))):.3f}, "
            f"Christoffersen p≈{float(row_patch.get('christoffersen_p', float('nan'))):.3f})"
        )
        text = text.replace("{{PATCH_VaR_TESTS}}", inject)

    p.write_text(text, encoding="utf-8")
    print(f"Updated {memo_path} (commit {commit})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--memo", default="docs/var_decision_memo.md")
    args = ap.parse_args()
    update_memo(args.memo)
