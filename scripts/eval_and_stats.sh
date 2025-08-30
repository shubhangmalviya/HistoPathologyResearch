#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
PYBIN="$ROOT_DIR/venv/bin/python"
BATCH_SIZE="${BATCH_SIZE:-16}"

echo "[eval] Computing metrics for expert and unified models"
PYTHONPATH="$ROOT_DIR/src" "$PYBIN" "$ROOT_DIR/src/evaluation/eval_rq2.py" --batch_size "$BATCH_SIZE"

echo "[stats] Running statistical tests"
PYTHONPATH="$ROOT_DIR/src" "$PYBIN" "$ROOT_DIR/src/stats/compare_expert_vs_unified.py"

echo "[eval+stats] Done. See results/"


