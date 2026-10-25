SHELL := /bin/bash
PY := .venv/bin/python
PIP := .venv/bin/pip

# Primary and companion runs are explicit so `make eval/package` cannot
# accidentally pick a smoke-test or companion run by directory mtime.
# Override on the command line, e.g. `make eval RUN=artifacts/runs/tinyshell-2m-run2-docmask`.
RUN ?= artifacts/runs/tinyshell-xl-10m
EVAL_DIR ?= artifacts/eval/tinyshell-xl-10m
CORE_RUN ?= artifacts/runs/tinyshell-core-2m
CORE_EVAL_DIR ?= artifacts/eval/tinyshell-core-2m
PORT ?= 8585

.PHONY: help setup data tokenizer core smoke train core-train eval core-eval package plots core-plots results core-results dashboard demo test lint format clean

help:
	@echo "TinyShell targets:"
	@echo "  make setup      Create .venv and install the package + dev dependencies"
	@echo "  make data       Download tldr-pages and build train/val/test JSONL"
	@echo "  make tokenizer  Train the 4k BPE tokenizer"
	@echo "  make core       Build the focused core-command subset"
	@echo "  make smoke      Run a 30-step pipeline smoke test"
	@echo "  make train      Run the full training job"
	@echo "  make core-train  Train the focused core-command model"
	@echo "  make eval       Evaluate the broad run and write reports"
	@echo "  make core-eval  Evaluate the core-command model"
	@echo "  make package    Package the broad run into artifacts/model"
	@echo "  make plots      Generate training and evaluation plots"
	@echo "  make core-plots Generate plots for the core run"
	@echo "  make results    Print the broad model vs BM25 comparison"
	@echo "  make core-results Print the core model vs BM25 comparison"

	@echo "  make dashboard  Serve the training/eval dashboard (http://localhost:$(PORT))"
	@echo "  make demo       Ask the trained model for a command"
	@echo "  make test       Run the test suite"
	@echo "  make lint       Run ruff"

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

data:
	$(PY) -m tinyshell.data --config configs/tinyshell.yaml

tokenizer:
	$(PY) -m tinyshell.tokenizer --config configs/tinyshell.yaml

core:
	$(PY) scripts/build_core.py

dashboard:
	$(PY) scripts/dashboard.py --serve --port $(PORT)


smoke:
	$(PY) -m tinyshell.train --config configs/smoke.yaml

train:
	$(PY) -m tinyshell.train --config configs/tinyshell.yaml

core-train:
	$(PY) -m tinyshell.train --config configs/core.yaml

eval:
	$(PY) -m tinyshell.evaluate --run $(RUN)

core-eval:
	$(PY) -m tinyshell.evaluate --run $(CORE_RUN)

package:
	$(PY) -m tinyshell.package --run $(RUN)

plots:
	$(PY) -m tinyshell.plots --run $(RUN) --eval-dir $(EVAL_DIR)

core-plots:
	$(PY) -m tinyshell.plots --run $(CORE_RUN) --eval-dir $(CORE_EVAL_DIR)

results:
	$(PY) scripts/summarize_eval.py $(EVAL_DIR)/metrics.json

core-results:
	$(PY) scripts/summarize_eval.py $(CORE_EVAL_DIR)/metrics.json

demo:
	$(PY) -m tinyshell.cli "list all files including hidden ones, with human-readable sizes"

test:
	$(PY) -m pytest

lint:
	.venv/bin/ruff check .

format:
	.venv/bin/ruff check --fix .
	.venv/bin/ruff format .

clean:
	rm -rf data/processed artifacts/runs artifacts/eval .pytest_cache .ruff_cache
