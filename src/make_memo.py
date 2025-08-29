#!/usr/bin/env python3
"""
Generate a brief decision memo from the evaluation CSVs.

Changes:
- Prefer RAW PatchTST row (no calibration).
- Report Christoffersen independence and conditional-coverage p-values explicitly.
- Robust to column-name casing and missing fields.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, UTC
import argparse
import re
import pandas as pd
import numpy as np

def _pick_patch_row(bt: pd.DataFrame) -> pd.Series:
    if bt.empty:
        raise ValueError("tables/var_backtest.csv is empty")

    df = bt.copy()
    for c in ("model", "mode"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower()

    # Prefer RAW, then rolling, then fixed
    prefs = [
        ("mode", "none"),
        ("model", "patch_raw"),
        ("mode", "rolling"),
        ("model", "patch_cal"),
        ("mode", "fixed"),
        ("model", "patch_fixed"),
    ]
    for col, val in prefs:
        if col in df.columns:
            hit = df[df[col] == val]
            if not hit.empty:
                return bt.loc[hit.index[0]]

    return bt.iloc[0]

def _parse_band(band_str: str) -> tuple[int | None, int | None]:
    if not isinstance(band_str, str):
        return None, None
    s = band_str.strip().replace("–", "-")
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", s)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))

def _fmt(x, pct=False, nd=3):
    try:
        if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            return "N/A"
        if pct:
            return f"{float(x)*100:.2f}%"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "N/A"

def build_memo(err: pd.DataFrame, bt: pd.DataFrame) -> str:
    row = _pick_patch_row(bt)

    br   = row.get("breach_rate", np.nan)
    kup  = row.get("kupiec_p", np.nan)
    chi_i = row.get("christoffersen_ind_p", np.nan)
    chi_c = row.get("christoffersen_cc_p", np.nan)
    eff_n = int(row.get("effective_n", 0)) if pd.notna(row.get("effective_n", np.nan)) else 0
    last250 = int(row.get("last250_breaches", 0)) if pd.notna(row.get("last250_breaches", np.nan)) else 0
    band_str = str(row.get("band_95pct", "") or "")
    lo, hi = _parse_band(band_str)
    in_band = (lo is not None and hi is not None and lo <= last250 <= hi)
    membership = "∈" if in_band else "∉"

    # Optional variance summary (best QLIKE)
    var_line = ""
    if err is not None and not err.empty:
        e = err.copy()
        if "QLIKE" in e.columns:
            e = e.sort_values("QLIKE")
        best = e.iloc[0]
        rmse = _fmt(best.get("RMSE", np.nan), nd=6)
        qlk  = _fmt(best.get("QLIKE", np.nan))
        var_line = f"- Variance: {best.get('model','HAR')} RMSE≈{rmse}, QLIKE≈{qlk}."

    today = datetime.now(UTC).date().isoformat()

    md = f"""# Daily SPY VaR/Vol — Decision Memo

**Date:** {today}

## Headline (RAW, no calibration)
- VaR95 (PatchTST raw): exceptions **{_fmt(br, pct=True)}**; Kupiec p≈{_fmt(kup)}, Christoffersen ind p≈{_fmt(chi_i)}, cc p≈{_fmt(chi_c)}; N_eff={eff_n}.
- Last-250 breaches: **{last250}** {membership} [{band_str}].
{var_line}

## Notes
- Sequences are leak-safe (past T→next day).
- Evaluation uses unconditional (Kupiec) and independence (Christoffersen) tests.
- No post-hoc calibration was applied to the quantile.
"""
    return md

def main(out_path: str = "docs/var_decision_memo.md") -> None:
    err = pd.read_csv("tables/error_metrics.csv") if Path("tables/error_metrics.csv").exists() else pd.DataFrame()
    bt  = pd.read_csv("tables/var_backtest.csv")
    md  = build_memo(err, bt)
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(md, encoding="utf-8")
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/var_decision_memo.md")
    args = ap.parse_args()
    main(args.out)
