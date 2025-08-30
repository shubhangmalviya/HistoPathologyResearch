#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
PYBIN="$ROOT_DIR/venv/bin/python"

echo "[split] Creating unified dataset splits into dataset/"
PYTHONPATH="$ROOT_DIR" "$PYBIN" "$ROOT_DIR/src/datasets/split_pannuke.py"

echo "[split] Creating per-tissue dataset splits into dataset_tissues/ (top-5)"
PYTHONPATH="$ROOT_DIR" "$PYBIN" "$ROOT_DIR/src/datasets/split_by_tissue.py"

echo "[split] Done."


