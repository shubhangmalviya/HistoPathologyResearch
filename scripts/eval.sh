#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
PYBIN="$ROOT_DIR/venv/bin/python"

PYTHONPATH="$ROOT_DIR/src" "$PYBIN" "$ROOT_DIR/src/evaluation/eval_rq2.py" "$@"


