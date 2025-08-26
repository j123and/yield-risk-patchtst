#!/usr/bin/env python3
"""
Update fields in the memo after evaluation runs.
Usage:
    python src/update_memo.py --memo docs/var_decision_memo.md
Fills:
    {{DATE}}, {{GIT_COMMIT}}, {{PATCH_VaR_TESTS}}
"""
from __future__ import annotations
from pathlib import Path
from datetime import date
import argparse
import subprocess
import pandas as pd
import numpy as np

def _safe_fmt(x, fmt="{:.3f}"):
    try:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return "N/A"
        return fmt.format(float(x))
    except Exception:
        return "N/A"

def update_memo(memo_path: str) -> None:
    # Optional: variance/other metrics (not used here; keep for future)
    err = None
    try:
        err_path = Path("tables/error_metrics.csv")
        if err_path.exists():
            err = pd.read_csv(err_path)
    except Exception:
        err = None

    # VaR backtest (required for inject)
    bt = None
    bt_path = Path("tables/var_backtest.csv")
    if bt_path.exists():
        bt = pd.read_csv(bt_path)
    else:
        bt = None

    # Pick Patch row: prefer rolling ("Patch_cal"), else any "Patch"
    row = None
    if bt is not None and "model" in bt.columns and len(bt) > 0:
        sel_roll = bt.loc[bt["model"].astype(str).str.lower() == "patch_cal"]
        if not sel_roll.empty:
            row = sel_roll.iloc[0]
        else:
            sel_any = bt.loc[bt["model"].astype(str).str.contains("patch", case=False, na=False)]
            if not sel_any.empty:
                row = sel_any.iloc[0]

    # Git commit (best-effort)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        commit = "N/A"

    p = Path(memo_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        # create a minimal memo so replace() won’t crash
        p.write_text("# VaR decision memo\n\nDate: {{DATE}}\nCommit: {{GIT_COMMIT}}\n\nTests: {{PATCH_VaR_TESTS}}\n", encoding="utf-8")

    text = p.read_text(encoding="utf-8")

    # Basic replacements
    text = text.replace("{{DATE}}", str(date.today()))
    text = text.replace("{{GIT_COMMIT}}", commit)

    # VaR tests injection
    inject = " (tests unavailable)"
    if row is not None:
        kupiec = _safe_fmt(row.get("kupiec_p"))
        # new columns from eval: independence and conditional coverage
        ch_ind = _safe_fmt(row.get("christoffersen_ind_p"))
        ch_cc  = _safe_fmt(row.get("christoffersen_cc_p"))
        br     = _safe_fmt(row.get("breach_rate"), "{:.3%}")  # as percent
        neff   = row.get("effective_n")
        neff_s = str(int(neff)) if pd.notna(neff) else "N/A"
        inject = f" (breach rate={br}, Kupiec p≈{kupiec}, Christoffersen ind p≈{ch_ind}, cc p≈{ch_cc}, N_eff={neff_s})"

    text = text.replace("{{PATCH_VaR_TESTS}}", inject)

    p.write_text(text, encoding="utf-8")
    print(f"Updated {memo_path} (commit {commit})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--memo", default="docs/var_decision_memo.md")
    args = ap.parse_args()
    update_memo(args.memo)
