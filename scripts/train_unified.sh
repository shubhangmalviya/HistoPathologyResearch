#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
PYBIN="$ROOT_DIR/venv/bin/python"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-5e-4}"
TISSUES=("Breast" "Colon" "Adrenal_gland" "Esophagus" "Bile-duct")

echo "[train unified] tissues: ${TISSUES[*]}"
"$PYBIN" "$ROOT_DIR/src/train_unified_unet.py" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --lr "$LR" --tissues "${TISSUES[@]}"

echo "[train unified] Done."


