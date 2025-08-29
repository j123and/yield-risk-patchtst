#!/usr/bin/env bash
set -euo pipefail

SYMBOL="SPY"
START="2015-01-02"
END="2025-07-31"
HOLDOUT="2023-01-02"

SEQ_LEN=120
PATCH_LEN=20
D_MODEL=128
NHEAD=4
NLAYERS=3
DROPOUT=0.1
EPOCHS=80            # 60–100 is fine; 80 is a good default
PRE_V=0              # set >0 only if you really want variance pretraining
BATCH=128
LR=3e-4
ALPHA=0.05
SEED=1337

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"
OUT_DIR="${ROOT_DIR}/outputs"
TAB_DIR="${ROOT_DIR}/tables"
FIG_DIR="${ROOT_DIR}/figs"
DOC_DIR="${ROOT_DIR}/docs"
mkdir -p "${DATA_DIR}" "${OUT_DIR}" "${TAB_DIR}" "${FIG_DIR}" "${DOC_DIR}"

export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=":16:8"
export CUDA_LAUNCH_BLOCKING=1

echo "=== Ingest + audit ==="
python "${ROOT_DIR}/src/ingest_yahoo_gk.py" --symbol "${SYMBOL}" --start "${START}" --end "${END}"
python "${ROOT_DIR}/src/audit_rv.py" --in "${DATA_DIR}/${SYMBOL,,}_rv.parquet"
python "${ROOT_DIR}/src/build_sequences.py" --symbol "${SYMBOL}" --seq_len "${SEQ_LEN}"

NPZ="${OUT_DIR}/${SYMBOL,,}_seq_${SEQ_LEN}.npz"

echo "=== Train PatchTST (quantile in return space + optional variance)==="
python "${ROOT_DIR}/src/train_patchtst_multitask.py" \
  --npz "${OUT_DIR}/${SYMBOL,,}_seq_${SEQ_LEN}.npz" \
  --split_date "${HOLDOUT}" \
  --epochs "${EPOCHS}" \
  --batch "${BATCH}" \
  --seq_len "${SEQ_LEN}" --patch_len "${PATCH_LEN}" \
  --d_model "${D_MODEL}" --nhead "${NHEAD}" --nlayers "${NLAYERS}" \
  --dropout "${DROPOUT}" --lr "${LR}" --w_q 12 --w_v 1.0 \
  --seed "${SEED}" --out_csv "${OUT_DIR}/patch_preds.csv"

echo "=== Evaluate VaR (RAW, no calibration) ==="
python "${ROOT_DIR}/src/eval_phase4.py" \
  --symbol "${SYMBOL}" --holdout_start "${HOLDOUT}" \
  --alpha "${ALPHA}" --calib_mode none

echo "=== Update memo + README ==="
python "${ROOT_DIR}/src/make_memo.py" --out "${DOC_DIR}/var_decision_memo.md"
python "${ROOT_DIR}/src/update_memo.py" --memo "${DOC_DIR}/var_decision_memo.md"
python "${ROOT_DIR}/src/update_readme.py" --readme "${ROOT_DIR}/README.md" \
  --calib-note "none (reporting raw PatchTST VaR)."

echo
echo "Done:"
echo " - Tables: ${TAB_DIR}/var_backtest.csv"
echo " - Figure: ${FIG_DIR}/var_breach_timeline.png"
echo " - Predictions: ${OUT_DIR}/patch_preds.csv"
echo " - Memo: ${DOC_DIR}/var_decision_memo.md"
