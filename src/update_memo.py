#!/usr/bin/env python3
"""
Update fields in the memo after evaluation runs.

Fills:
    {{DATE}}, {{GIT_COMMIT}}, {{PATCH_VaR_TESTS}}
Now prefers RAW Patch row (no calibration).
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
    # VaR backtest
    bt_path = Path("tables/var_backtest.csv")
    bt = pd.read_csv(bt_path) if bt_path.exists() else None

    row = None
    if bt is not None and "model" in bt.columns and len(bt) > 0:
        df = bt.copy()
        for c in ("model", "mode"):
            if c in df.columns:
                df[c] = df[c].astype(str).str.lower()
        # prefer raw, then rolling, then fixed
        for col, val in [("mode","none"),("model","patch_raw"),
                         ("mode","rolling"),("model","patch_cal"),
                         ("mode","fixed"),("model","patch_fixed")]:
            if col in df.columns:
                hit = df[df[col]==val]
                if not hit.empty:
                    row = bt.loc[hit.index[0]]
                    break
        if row is None:
            row = bt.iloc[0]

    # Git commit (best-effort)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        commit = "N/A"

    p = Path(memo_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text("# VaR decision memo\n\nDate: {{DATE}}\nCommit: {{GIT_COMMIT}}\n\nTests: {{PATCH_VaR_TESTS}}\n",
                     encoding="utf-8")

    text = p.read_text(encoding="utf-8")
    text = text.replace("{{DATE}}", str(date.today()))
    text = text.replace("{{GIT_COMMIT}}", commit)

    inject = " (tests unavailable)"
    if row is not None:
        kupiec = _safe_fmt(row.get("kupiec_p"))
        ch_ind = _safe_fmt(row.get("christoffersen_ind_p"))
        ch_cc  = _safe_fmt(row.get("christoffersen_cc_p"))
        br     = _safe_fmt(row.get("breach_rate"), "{:.3%}")
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
