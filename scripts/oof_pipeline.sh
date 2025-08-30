#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
PYBIN="$ROOT_DIR/venv/bin/python"
TISSUES=("Breast" "Colon" "Adrenal_gland" "Esophagus" "Bile-duct")
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-5e-4}"
N_SPLITS="${N_SPLITS:-3}"

for T in "${TISSUES[@]}"; do
  echo "[oof-train] $T"
  PYTHONPATH="$ROOT_DIR" "$PYBIN" "$ROOT_DIR/src/training/train_oof.py" --tissue "$T" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --lr "$LR" --n_splits "$N_SPLITS"
done

echo "[oof-eval] Collecting per-image OOF metrics"
PYTHONPATH="$ROOT_DIR" "$PYBIN" "$ROOT_DIR/src/evaluation/eval_oof.py" --n_splits "$N_SPLITS" --batch_size "$BATCH_SIZE"

echo "[oof] Done."


