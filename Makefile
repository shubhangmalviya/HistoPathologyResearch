SHELL := /bin/bash

PY := venv/bin/python

.PHONY: splits experts unified eval stats all

splits:
	./scripts/split_all.sh

experts:
	./scripts/train_experts.sh

unified:
	./scripts/train_unified.sh

eval:
	./scripts/eval_and_stats.sh

stats:
	PYTHONPATH="$(shell pwd)/src" $(PY) src/stats/compare_expert_vs_unified.py

all: splits experts unified eval


