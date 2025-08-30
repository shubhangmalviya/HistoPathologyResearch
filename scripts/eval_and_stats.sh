#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
PYBIN="$ROOT_DIR/venv/bin/python"
BATCH_SIZE="${BATCH_SIZE:-16}"

echo "[eval] Computing metrics for expert and unified models"
PYTHONPATH="$ROOT_DIR" "$PYBIN" "$ROOT_DIR/src/evaluation/eval_rq2.py" --batch_size "$BATCH_SIZE"

echo "[collect] Per-image paired metrics parquet"
PYTHONPATH="$ROOT_DIR" "$PYBIN" "$ROOT_DIR/src/evaluation/collect_per_image_metrics.py" --batch_size "$BATCH_SIZE"

echo "[stats] Running statistical tests (Wilcoxon one-sided PQ + BH + mixed effects)"
PYTHONPATH="$ROOT_DIR" "$PYBIN" "$ROOT_DIR/src/stats/analyze_rq2.py"

echo "[eval+stats] Done. See results/"


