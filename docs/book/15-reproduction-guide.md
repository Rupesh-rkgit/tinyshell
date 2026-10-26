# Chapter 15 — Reproduction Guide: Every Command, Config, and Number

This appendix is the operator's reference: exact commands to rebuild data,
tokenizer, packed corpus, any of the four runs, and the full evaluation; plus
the resolved configuration tables and environment facts.

## 15.1 Environment (as recorded for run 3)

| Field | Value |
|---|---|
| Platform | Linux (Oracle Cloud) aarch64, 4 vCPU Neoverse-N1 |
| Python | 3.12.3 |
| PyTorch | 2.14.1+cpu (no CUDA) |
| NumPy | 2.5.4 |
| Threads | 4 (`torch.set_num_threads`) |
| Precision | fp32 everywhere |
| Tokenizer data | tldr-pages @ `95062b79e30fc3539246399b1dbfcfabbccc79cf` (common + linux) |
| Seed | 42 (data, tokenizer sampling, torch init, DataLoader shuffling) |

Install once:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

## 15.2 Full pipeline, from empty machine to shipped model

```bash
# 1. Data (clones tldr-pages sparsely; needs network once)
make data                       # -> data/raw/tldr, data/processed/*.jsonl + manifest

# 2. Tokenizer (4,096 byte-level BPE)
make tokenizer                  # -> artifacts/tokenizer/tokenizer.json + stats

# 3. Optional: smoke test the whole pipeline (~1 min, tiny model, 30 steps)
make smoke

# 4a. Broad 2.1M model (runs 1–2 lineage; run 2 = doc-aligned packing)
make train                      # configs/tinyshell.yaml -> artifacts/runs/tinyshell-2m
make eval
make plots
make package

# 4b. The shipped 9.6M model (run 3)
.venv/bin/python -m tinyshell.prepare --config configs/tinyshell-xl.yaml
.venv/bin/python -m tinyshell.dataset --config configs/tinyshell-xl.yaml --force
.venv/bin/python -m tinyshell.train    --config configs/tinyshell-xl.yaml
.venv/bin/python -m tinyshell.evaluate --run artifacts/runs/tinyshell-xl-10m
.venv/bin/python -m tinyshell.package  --run  artifacts/runs/tinyshell-xl-10m

# 5. Core diagnostic experiment (2.1M on 111 command families)
make core                       # scripts/build_core.py -> data/core/
make core-train                 # configs/core.yaml
make core-eval
make core-plots
```

Expected wall-clock on the reference 4-vCPU host: tokenizer ~minutes;
broad 2.1M ~3.2 h; core ~2.2 h; XL run-3 ~27.7 h; six-split generation eval
~1 h (9,000 greedy requests, single-batch).

## 15.3 Rebuilding a specific historical run

| Run | Command sketch | Key facts to match |
|---|---|---|
| Run 1 (naive packing) | historic code path; not reconstructable from current code by default | packed without doc alignment; see `docs/EXPERIMENTS.md` and run dir for its config |
| Run 2 | `tinyshell.dataset --config configs/tinyshell.yaml --force` then train with run-id `tinyshell-2m-run2-docmask` | 4 epochs, batch 64, warmup 200, seed 42 |
| Core | `make core && make core-train` | 20 epochs, dropout 0.1, `first_token_weight: 5.0` (note: shipped run used the partial-tagging loss version) |
| Run 3 | commands in §15.2 step 4b | variants 150, styles 12–15 in train pool, stricter filters, batch 128, warmup 300, 1 epoch |

Each run directory carries `config.yaml`, `env.json`, `git_commit.txt`, and
a tokenizer copy — diff these first if your numbers drift.

## 15.4 Resolved configurations

**Model**

| Field | 2.1M (`tinyshell.yaml`, `core.yaml`) | 9.6M (`tinyshell-xl.yaml`) |
|---|---|---|
| vocab_size | 4096 | 4096 |
| d_model | 128 | 256 |
| n_layers | 6 | 10 |
| n_heads | 4 | 8 |
| intermediate_size | 512 | 768 |
| max_seq_len | 256 | 256 |
| rope_theta | 10000.0 | 10000.0 |
| dropout | 0.05 (core: 0.1) | 0.05 |
| tie_embeddings | true | true |

**Training**

| Field | 2.1M broad | Core | XL (run 3) |
|---|---|---|---|
| batch_size | 64 | 64 | 128 |
| grad_accum_steps | 1 | 1 | 1 |
| lr / min_lr | 3e-4 / 3e-5 | same | same |
| weight_decay / betas | 0.1 / (0.9, 0.95) | same | same |
| grad_clip | 1.0 | 1.0 | 1.0 |
| warmup_steps | 200 | 80 | 300 |
| epochs | 4 | 20 | 1 |
| first_token_weight | 1.0 | 5.0 | 1.0 |
| eval_interval / sample_interval | 250 / 250 | 50 / 100 | 250 / 250 |
| save_interval | 1000 | 500 | 500 |
| num_threads | 4 | 4 | 4 |

**Data**

| Field | v1 (`tinyshell.yaml`) | core | XL (`tinyshell-xl.yaml`) |
|---|---|---|---|
| processed_dir | data/processed | data/core | data/processed-xl |
| variants_per_example | 6 | 6 | 150 |
| eval_examples | 1500 | 1500 | 1500 |
| train/val/test | 0.8/0.1/0.1 | same | same |
| max_command_chars | 120 | 120 | 120 |
| description bounds | 8–160 chars | same | same |

**Eval** (all configs): greedy (temperature 0.0, top_k 1), max_new_tokens
64, batch 1; splits: six broad / three core; num_failures 50 per split.

## 15.5 Evaluation-only rebuilds and comparisons

```bash
# Full six-split eval + BM25 for any checkpoint
.venv/bin/python -m tinyshell.evaluate --run artifacts/runs/<run-id>

# Evaluate the BROAD 2.1M checkpoint on the CORE splits (no retraining)
.venv/bin/python -m tinyshell.evaluate --run artifacts/runs/tinyshell-2m-run2-docmask \
  --data-dir data/core

# Model-vs-BM25 markdown, error census, side-by-side runs
.venv/bin/python scripts/summarize_eval.py latest
.venv/bin/python scripts/analyze_errors.py artifacts/eval/<run>
.venv/bin/python scripts/compare_runs.py <runA> <runB>

# Refresh README/model-card metric blocks from the newest eval
.venv/bin/python scripts/update_docs.py
```

## 15.6 Where every number in this book lives

| Claim | Artifact |
|---|---|
| Corpus statistics, filters, splits, slot/risk counts | `data/processed*/dataset_manifest.json`, `DATA_CARD.md` |
| Tokenizer stats | `artifacts/tokenizer/stats.json` |
| Packing totals (chunks, tokens, supervised labels) | `data/processed*/packed/manifest.json` |
| Training curves, throughput, grad norms | `artifacts/runs/<run>/metrics.jsonl`, `train.log` |
| Best/final losses, step counts | `artifacts/runs/<run>/final_metrics.json` |
| Generation metrics (model + BM25) | `artifacts/eval/<run>/metrics.json`, `report.md` |
| Raw predictions / failure samples | `artifacts/eval/<run>/*_predictions.jsonl`, `*_failures.jsonl` |
| Parameter counts | `artifacts/runs/<run>/model_summary.json` |
| Environment provenance | `artifacts/runs/<run>/env.json`, `git_commit.txt` |

## 15.7 Checkpoint sizes and what to distribute

Full checkpoints include optimizer state (2× fp32 moment buffers):

| Checkpoint | Full | Model-only |
|---|---:|---:|
| 2.1M runs (best.pt) | ~25 MB | ~8.4 MB |
| 9.6M run 3 (best.pt) | ~115 MB | ~38 MB |

`scripts/strip_checkpoint.py` writes `<name>-model.pt` files containing only
the model state dict (plus config/step metadata) — the format the CLI and
evaluator load happily (they read `ckpt["model"]`). For GitHub, model-only
files fit under the 100 MB per-file limit; full optimizer checkpoints are
for resuming long runs and belong in Git LFS or release assets if you need
them remotely (commands in Chapter 17).
