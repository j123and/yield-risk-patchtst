#!/usr/bin/env bash
# Reproduce SPY VaR/volatility results end-to-end.
# Ingest → audit → sequences → train PatchTST (quant + multitask) → baselines → eval → memo.
# Uses rolling VaR calibration (since eval_phase4.py supports: fixed|rolling|none).
set -euo pipefail

# ---- defaults (override via flags) ----
SYMBOL="SPY"
START="2015-01-02"
END="2025-07-31"
HOLDOUT="2023-01-02"
SEQ_LEN=120
PATCH_LEN=20
D_MODEL=64
NHEAD=4
NLAYERS=2
DROPOUT=0.1
EPOCHS=10
BATCH=64
ALPHA=0.05
SEED=1337
CALIB_WINDOW=250
ROLL_WINDOW=250
CALIB_EMA=0.20

# ---- paths ----
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"
OUT_DIR="${ROOT_DIR}/outputs"
TAB_DIR="${ROOT_DIR}/tables"
FIG_DIR="${ROOT_DIR}/figs"
DOC_DIR="${ROOT_DIR}/docs"
MEMO="${DOC_DIR}/var_decision_memo.md"

SYMBOL_LC="$(echo "${SYMBOL}" | tr '[:upper:]' '[:lower:]')"
NPZ_DEFAULT="${OUT_DIR}/${SYMBOL_LC}_seq_${SEQ_LEN}.npz"
PATCH_PREDS="${OUT_DIR}/patch_preds.csv"

# ---- helpers ----
usage() {
  cat <<EOF
Usage: $(basename "$0") [options]
  --symbol TICKER         (default: ${SYMBOL})
  --start YYYY-MM-DD      (default: ${START})
  --end   YYYY-MM-DD      (default: ${END})
  --holdout YYYY-MM-DD    (default: ${HOLDOUT})
  --seq-len N             (default: ${SEQ_LEN})
  --epochs N              (default: ${EPOCHS})
  --batch N               (default: ${BATCH})
  --alpha A               (default: ${ALPHA})
  --seed  N               (default: ${SEED})
EOF
}

die() { echo "ERROR: $*" >&2; exit 1; }
run() { echo "+ $*"; "$@"; }

# ---- parse args ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --symbol) SYMBOL="$2"; shift 2;;
    --start) START="$2"; shift 2;;
    --end) END="$2"; shift 2;;
    --holdout) HOLDOUT="$2"; shift 2;;
    --seq-len) SEQ_LEN="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --batch) BATCH="$2"; shift 2;;
    --alpha) ALPHA="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) usage; die "Unknown arg: $1";;
  esac
done

# Recompute derived names after arg parsing
SYMBOL_LC="$(echo "${SYMBOL}" | tr '[:upper:]' '[:lower:]')"
NPZ_DEFAULT="${OUT_DIR}/${SYMBOL_LC}_seq_${SEQ_LEN}.npz"

mkdir -p "${DATA_DIR}" "${OUT_DIR}" "${TAB_DIR}" "${FIG_DIR}" "${DOC_DIR}"

# ---- reproducibility env (training scripts also seed internally) ----
export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=":16:8"
export CUDA_LAUNCH_BLOCKING=1

echo "=== Reproducing results for ${SYMBOL} ==="
echo "Range: ${START} → ${END} | Holdout start: ${HOLDOUT} | Seed: ${SEED}"
echo

# 1) Ingest OHLC & compute GK RV
run python "${ROOT_DIR}/src/ingest_yahoo_gk.py" \
  --symbol "${SYMBOL}" --start "${START}" --end "${END}"

# 2) Audit RV dataset
run python "${ROOT_DIR}/src/audit_rv.py" --in "${DATA_DIR}/${SYMBOL_LC}_rv.parquet"

# 3) Build sequences (writes outputs/*_seq_${SEQ_LEN}.npz)
run python "${ROOT_DIR}/src/build_sequences.py" \
  --symbol "${SYMBOL}" --seq_len "${SEQ_LEN}"

# 3b) Locate NPZ
NPZ_FOUND=""
if [[ -f "${NPZ_DEFAULT}" ]]; then
  NPZ_FOUND="${NPZ_DEFAULT}"
else
  CAND=$(ls -1t "${OUT_DIR}/${SYMBOL_LC}"*_seq_"${SEQ_LEN}".npz 2>/dev/null | head -n1 || true)
  [[ -n "${CAND}" && -f "${CAND}" ]] && NPZ_FOUND="${CAND}"
fi
[[ -z "${NPZ_FOUND}" ]] && die "Could not find NPZ in ${OUT_DIR} (looked for ${NPZ_DEFAULT})."
echo "+ Using NPZ: ${NPZ_FOUND}"

# 4) Train PatchTST quantile head (τ=0.05)
run python "${ROOT_DIR}/src/train_patchtst_quant.py" \
  --npz "${NPZ_FOUND}" --split_date "${HOLDOUT}" --epochs "${EPOCHS}" --batch "${BATCH}" \
  --seq_len "${SEQ_LEN}" --patch_len "${PATCH_LEN}" --d_model "${D_MODEL}" \
  --nhead "${NHEAD}" --nlayers "${NLAYERS}" --dropout "${DROPOUT}" --seed "${SEED}" \
  --out_csv "${PATCH_PREDS}"

# 5) Train PatchTST multi-task head (quantile + log-variance)
run python "${ROOT_DIR}/src/train_patchtst_multitask.py" \
  --npz "${NPZ_FOUND}" --split_date "${HOLDOUT}" --epochs "${EPOCHS}" --batch "${BATCH}" \
  --seq_len "${SEQ_LEN}" --patch_len "${PATCH_LEN}" --d_model "${D_MODEL}" \
  --nhead "${NHEAD}" --nlayers "${NLAYERS}" --dropout "${DROPOUT}" --seed "${SEED}" \
  --out_csv "${PATCH_PREDS}"

# 6) Baselines + evaluation
run python "${ROOT_DIR}/src/baseline_har.py"       --holdout_start "${HOLDOUT}"
run python "${ROOT_DIR}/src/baseline_garch_t.py"   --holdout_start "${HOLDOUT}"
run python "${ROOT_DIR}/src/eval_phase4.py" \
  --symbol "${SYMBOL}" --holdout_start "${HOLDOUT}" --alpha "${ALPHA}" \
  --calib_mode rolling --calib_window "${CALIB_WINDOW}" --roll_window "${ROLL_WINDOW}" \
  --calib_ema "${CALIB_EMA}"

# 7) Memo
run python "${ROOT_DIR}/src/make_memo.py" --out "${MEMO}"
run python "${ROOT_DIR}/src/update_memo.py" --memo "${MEMO}"

echo
echo "=== Done. Key artifacts ==="
echo "- Tables: ${TAB_DIR}"
echo "- Figures: ${FIG_DIR}"
echo "- Predictions: ${PATCH_PREDS}"
echo "- Memo: ${MEMO}"
