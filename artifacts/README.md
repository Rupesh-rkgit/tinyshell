# TinyShell artifacts

This directory holds generated, machine-specific outputs. Large files
(`artifacts/runs/`, `artifacts/model/*.pt`) are gitignored, but the shipped
working tree contains them so the trained model can be used immediately.

## Layout

```text
artifacts/
├── tokenizer/
│   ├── tokenizer.json          # 4,096-token byte-level BPE
│   ├── tokenizer_config.json   # special tokens + chat template
│   ├── stats.json              # vocab size, token-length percentiles, round trips
│   └── vocab_preview.txt       # first 200 vocabulary entries
├── runs/
│   ├── tinyshell-2m-run1/      # naive-packing ablation (kept for comparison)
│   ├── tinyshell-2m-run2-docmask/  # shipped broad model
│   └── tinyshell-core-2m/      # focused core-command diagnostic
│       ├── config.yaml         # exact config used for this run
│       ├── env.json            # OS, Python, torch, CPU, git state
│       ├── model_summary.json  # parameter counts + architecture config
│       ├── train.log           # human-readable training log
│       ├── metrics.jsonl       # train/val loss, LR, grad norm, throughput
│       ├── samples.jsonl       # generated examples every sample interval
│       ├── final_metrics.json  # final train/val losses and elapsed time
│       ├── checkpoints/
│       │   ├── best.pt         # best validation loss
│       │   ├── last.pt         # periodic / end-of-run snapshot
│       │   └── final.pt        # end-of-run snapshot
│       ├── tokenizer/          # copy of the tokenizer used
│       └── plots/              # loss.png, lr.png, throughput.png
├── eval/
│   ├── tinyshell-2m-run1/
│   ├── tinyshell-2m-run2-docmask/
│   └── tinyshell-core-2m/
│       ├── metrics.json        # all splits + BM25 baseline
│       ├── report.md           # human-readable tables
│       ├── *_predictions.jsonl # every prediction with scores and latency
│       ├── *_failures.jsonl    # non-exact examples
│       └── plots/              # eval_metrics.png
└── model/                      # self-contained packaged broad model
    ├── model.pt                # best checkpoint
    ├── config.yaml
    ├── model_summary.json
    ├── final_metrics.json
    ├── manifest.json
    ├── README.md
    ├── MODEL_CARD.md
    ├── DATA_CARD.md
    ├── eval_report.md
    ├── eval_metrics.json
    └── tokenizer/
```

Run 1 is retained because `docs/EXPERIMENTS.md` uses it for the packing
ablation. Run 2 is the broad model used by the CLI and packaged under
`artifacts/model/`. The core run is a diagnostic: it narrows the task to 655
pages and ~100 command families, and is not the default model.

## Using a checkpoint

```bash
# Packaged broad model (default when artifacts/model/model.pt exists)
tinyshell --model artifacts/model "list open ports"

# Broad training run directory
tinyshell --model artifacts/runs/tinyshell-2m-run2-docmask "find large files"

# Focused core run directory
tinyshell --model artifacts/runs/tinyshell-core-2m "show disk usage"

# Explicit checkpoint
tinyshell --model artifacts/runs/tinyshell-2m-run2-docmask/checkpoints/best.pt "show disk usage"

# Retrieval-only baseline (stronger on commands seen in training)
tinyshell --retrieve "stream pod logs. Assume the pod name is alice"
```

Checkpoints are PyTorch dictionaries with `model`, `optimizer`, `scheduler`,
`step`, `epoch`, `batch_in_epoch`, `best_val`, and `config`. Only `model` and
the config are needed for inference.

## Reproducing

```bash
make data
make tokenizer
make train
make eval
make plots
make package

# Focused core experiment
make core
make core-train
make core-eval
make core-plots
```

The broad targets default to the shipped `tinyshell-2m-run2-docmask` run;
override with `RUN=...` (for example, `make eval RUN=latest`). The core targets
use `CORE_RUN=artifacts/runs/tinyshell-core-2m`. `make package` copies the best
checkpoint and the current root docs to `artifacts/model/`; rerun it after
editing `MODEL_CARD.md`, `DATA_CARD.md`, or `docs/EXPERIMENTS.md`.

To evaluate a checkpoint against the focused data without retraining, override
the data directory:

```bash
.venv/bin/python -m tinyshell.evaluate \
  --run artifacts/runs/tinyshell-2m-run2-docmask \
  --data-dir data/core --splits val \
  --output-dir /tmp/broad-on-core
```
