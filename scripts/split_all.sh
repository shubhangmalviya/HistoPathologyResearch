#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
PYBIN="$ROOT_DIR/venv/bin/python"

echo "[split] Creating unified dataset splits into dataset/"
"$PYBIN" "$ROOT_DIR/src/split_pannuke.py"

echo "[split] Creating per-tissue dataset splits into dataset_tissues/ (top-5)"
"$PYBIN" "$ROOT_DIR/src/split_by_tissue.py"

echo "[split] Done."


