#!/usr/bin/env python3
"""
Update the VaR headline block in README.md from tables/var_backtest.csv.

- Looks for markers:
    <!-- VAR_HEAD_START -->
    <!-- VAR_HEAD_END -->
  and replaces the block in between with fresh Markdown.

- Picks the Patch row in priority: rolling -> fixed -> raw.
- Reports exceptions (breach rate), Kupiec p, Christoffersen (ind) p, N_eff, and last-250 with acceptance band.
- You can pass a calibration note to print under the headline.

Usage:
  python src/update_readme.py --readme README.md \
    --calib-note "rolling 250-day intercept, EMA=0.0 (no smoothing). No look-ahead."
"""
from __future__ import annotations
from pathlib import Path
import argparse
import re
import math
import pandas as pd

def _fmt(x, pct=False, nd=3):
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return "N/A"
        if pct:
            return f"{float(x)*100:.2f}%"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "N/A"

def pick_patch_row(bt: pd.DataFrame) -> pd.Series:
    if bt.empty:
        raise SystemExit("tables/var_backtest.csv is empty.")
    df = bt.copy()
    for col in ("model", "mode"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower()

    # Prefer rolling-calibrated, then fixed, then raw
    prefs = [
        ("mode", "rolling"), ("model", "patch_cal"),
        ("mode", "fixed"),   ("model", "patch_fixed"),
        ("mode", "none"),    ("model", "patch_raw"),
    ]
    for col, val in prefs:
        if col in df.columns:
            hit = df[df[col] == val]
            if not hit.empty:
                return bt.loc[hit.index[0]]
    return bt.iloc[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--var_table", default="tables/var_backtest.csv")
    ap.add_argument("--calib-note", default="rolling 250-day intercept, EMA=0.0 (no smoothing). No look-ahead.")
    args = ap.parse_args()

    readme_path = Path(args.readme)
    table_path = Path(args.var_table)

    if not readme_path.exists():
        raise SystemExit(f"README not found: {readme_path}")
    if not table_path.exists():
        raise SystemExit(f"Var table not found: {table_path}")

    bt = pd.read_csv(table_path)
    row = pick_patch_row(bt)

    # Extract fields with safe fallbacks
    br = row.get("breach_rate", float("nan"))          # fraction
    kup = row.get("kupiec_p", float("nan"))
    chi_ind = row.get("christoffersen_ind_p", float("nan"))
    # chi_cc = row.get("christoffersen_cc_p", float("nan"))  # uncomment if you want to show CC too
    eff_n = row.get("effective_n", 0)
    last250 = row.get("last250_breaches", 0)
    band = str(row.get("band_95pct", "") or "")

    mode = str(row.get("mode", "")).lower()
    tag = "calibrated" if mode in ("rolling", "fixed") else "raw"

    # Build the new Markdown block
    block_lines = [
        f"* **VaR<sub>0.95</sub> (PatchTST, {tag}):**",
        f"  exceptions **{_fmt(br, pct=True)}** (breach rate), **Kupiec p≈{_fmt(kup)}**, **Christoffersen (ind) p≈{_fmt(chi_ind)}**, **effective N<sub>eff</sub>={int(eff_n) if pd.notna(eff_n) else 'N/A'}**.",
        f"  Last-250 breaches: **{int(last250) if pd.notna(last250) else 0}**, inside the 95% acceptance band **[{band}]**.",
        f"  Calibration = **{args.calib_note}**",
    ]
    new_block = "\n".join(block_lines)

    text = readme_path.read_text(encoding="utf-8")

    start_marker = "<!-- VAR_HEAD_START -->"
    end_marker = "<!-- VAR_HEAD_END -->"
    if start_marker not in text or end_marker not in text:
        raise SystemExit(
            f"Markers not found in {readme_path}. Please add:\n{start_marker}\n... block ...\n{end_marker}"
        )

    pattern = re.compile(rf"({re.escape(start_marker)})(.*?){re.escape(end_marker)}", re.DOTALL | re.IGNORECASE)
    updated = pattern.sub(lambda m: f"{m.group(1)}\n{new_block}\n{end_marker}", text)

    readme_path.write_text(updated, encoding="utf-8")
    print(f"Updated {readme_path} with latest VaR headline.")

if __name__ == "__main__":
    main()
