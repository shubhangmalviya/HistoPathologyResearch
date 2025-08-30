#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
PYBIN="$ROOT_DIR/venv/bin/python"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-5e-4}"

TISSUES=("Breast" "Colon" "Adrenal_gland" "Esophagus" "Bile-duct")

for T in "${TISSUES[@]}"; do
  echo "[train expert] $T"
  PYTHONPATH="$ROOT_DIR" "$PYBIN" "$ROOT_DIR/src/training/train_expert_unet.py" --tissue "$T" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --lr "$LR"
done

echo "[train expert] Done."


