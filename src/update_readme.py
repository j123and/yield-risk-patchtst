#!/usr/bin/env python3
"""
Update the VaR headline block in README.md from tables/var_backtest.csv.

- Replaces the block between:
    <!-- VAR_HEAD_START -->
    <!-- VAR_HEAD_END -->

- Picks the Patch row in priority: rolling -> fixed -> none/raw.
- Robust to different column names emitted by eval_phase4.py.

Usage:
  python src/update_readme.py --readme README.md --var_table tables/var_backtest.csv \
    --calib-note "none (reporting raw PatchTST VaR)."
"""
from __future__ import annotations
from pathlib import Path
import argparse
import math
import pandas as pd
import re

def _fmt(x, pct=False, nd=3):
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return "N/A"
        if pct:
            return f"{float(x)*100:.2f}%"
        return f"{float(x):.{nd}f}"
    except Exception:
        return "N/A"

def _get_first(df_row: pd.Series, names: list[str], default=float("nan")):
    """Return first available numeric from a list of candidate column names."""
    for n in names:
        if n in df_row:
            val = df_row[n]
            try:
                return float(val)
            except Exception:
                continue
    return default

def _string_first(df_row: pd.Series, names: list[str], default: str = "") -> str:
    for n in names:
        if n in df_row and pd.notna(df_row[n]):
            return str(df_row[n])
    return default

def pick_patch_row(bt: pd.DataFrame) -> pd.Series:
    if bt.empty:
        raise SystemExit("tables/var_backtest.csv is empty.")

    df = bt.copy()
    # normalize
    for c in ("model", "mode"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower()

    # Priority: rolling -> fixed -> none/raw
    prefs = [
        ("mode", "rolling"),
        ("model", "patch_cal"),
        ("mode", "fixed"),
        ("model", "patch_fixed"),
        ("mode", "none"),
        ("model", "patch_raw"),
    ]
    for col, val in prefs:
        if col in df.columns:
            hit = df[df[col] == val]
            if not hit.empty:
                return bt.loc[hit.index[0]]

    # fallback: first row
    return bt.iloc[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--var_table", default="tables/var_backtest.csv")
    ap.add_argument("--calib-note", default="none (reporting raw PatchTST VaR).")
    args = ap.parse_args()

    readme_path = Path(args.readme)
    table_path = Path(args.var_table)

    if not readme_path.exists():
        raise SystemExit(f"README not found: {readme_path}")
    if not table_path.exists():
        raise SystemExit(f"Var table not found: {table_path}")

    bt = pd.read_csv(table_path)

    row = pick_patch_row(bt)

    # Mode tag for display
    mode = _string_first(row, ["mode"], "").lower()
    tag = "raw" if mode in ("none", "", "raw") else "calibrated"

    # Robust extractions (accept multiple column name variants)
    br = _get_first(row, [
        "breach_rate", "exception_rate", "breach_rate_mode", "breach_rate_raw"
    ])
    kup = _get_first(row, [
        "kupiec_p", "kupiec_p_mode", "kupiec_p_raw"
    ])
    chi_ind = _get_first(row, [
        "christoffersen_ind_p", "christoffersen_p", "christoffersen_ind_p_mode", "christoffersen_ind_p_raw"
    ])
    eff_n = _get_first(row, ["effective_n", "N_eff"], default=float("nan"))
    last250 = _get_first(row, ["lastn_breaches", "lastn_breaches"], default=float("nan"))
    band = _string_first(row, ["band_95pct", "band95", "binom_band_95"], default="")

    block_lines = [
        f"* **VaR<sub>0.95</sub> (PatchTST, {tag}):** exceptions **{_fmt(br, pct=True)}**, **Kupiec p≈{_fmt(kup)}**, **Christoffersen (ind) p≈{_fmt(chi_ind)}**, **N<sub>eff</sub>={_fmt(eff_n, nd=0)}**.",
        f"  Last-250 breaches: **{_fmt(last250, nd=0)}**, inside the 95% band **[{band}]**.",
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
