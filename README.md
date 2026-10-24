# TinyShell

**A 2.1M-parameter language model, trained from scratch on a CPU, for
natural-language → Linux shell command suggestions.**

TinyShell is an end-to-end, education-oriented LLM project: data collection and
filtering, a byte-level BPE tokenizer, a Llama-style decoder trained from random
initialization, honest held-out evaluation, retrieval baselines, a CLI, tests,
plots, model packaging, and model/data cards.

> **Status: educational prototype.** The neural model is not a reliable command
> oracle. On the broad 6,600-page dataset it usually chooses the wrong command
> family (≈7% command match on held-out requests) even though it produces valid
> bash 98–99% of the time and copies some arguments. BM25 retrieval is much
> stronger on seen commands. A focused 111-command-label variant lifts command
> match to 74–92% but stays near zero on exact match, confirming the small
> model's bottleneck is argument composition, not family selection. The
> project's value is the complete, reproducible pipeline and the documented
> failure analysis, not production accuracy. See [Results](#results) and
> [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

The model **only suggests** commands. It never runs them. A retrieval-only mode
is available for practical use on commands that appear in the training data:

```text
$ tinyshell --retrieve "stream pod logs. Assume the pod name is alice"
$ kubectl logs --follow alice

  valid bash | risk: safe | 6 ms
  mode: BM25 retrieval baseline (nearest training request)
```

The neural model's output on the same request is shown in the
[failure analysis](docs/EXPERIMENTS.md): it copies `alice` but picks the wrong
`kubectl` subcommand. That gap is the point of the experiment.

## Why this task?

Natural language → shell command is a useful, bounded task for a small model:

- The output is short (median 26 characters), so a 256-token context is plenty.
- Correctness is checkable (exact match, command match, token F1, `bash -n`).
- Generalization can be measured honestly: hold out entire commands/pages, then
  test whether the model can still compose a command it has never seen.
- It is a realistic local, offline "what was that flag?" target; the retrieval
  mode is already usable for commands covered by the training data.

The dataset is built from [tldr-pages](https://github.com/tldr-pages/tldr)
(CC-BY-4.0). A single tldr example is expanded into varied natural-language
requests by filling placeholders with realistic paths, files, hosts, users, and
packages, then rendering the same intent in several prompt styles.

## Results

Trained on 4 vCPU ARM64 (no GPU) for 4 epochs / 2,608 optimizer steps
(≈36.7M real-token passes) with document-aligned packing and block-diagonal
attention. Full evaluation uses 1,500 examples per split.

<!-- RESULTS:START -->
Trained model: **2,098,816 parameters**, checkpoint `best`. Run: `tinyshell-2m-run2-docmask`.

| Split | N | Exact | Command | Token F1 | Valid bash | p50 latency |
|---|---:|---:|---:|---:|---:|---:|
| `val` | 1500 | 0.001 | 0.073 | 0.174 | 0.985 | 155 ms |
| `val_unseen` | 1500 | 0.001 | 0.080 | 0.174 | 0.991 | 133 ms |
| `test_seen` | 1500 | 0.000 | 0.081 | 0.173 | 0.987 | 151 ms |
| `test_paraphrase` | 1500 | 0.002 | 0.089 | 0.187 | 0.995 | 140 ms |
| `test_unseen` | 1500 | 0.000 | 0.069 | 0.170 | 0.995 | 139 ms |
| `test_unseen_paraphrase` | 1500 | 0.000 | 0.062 | 0.172 | 0.991 | 141 ms |

### Model vs BM25 retrieval baseline

| Split | BM25 exact | BM25 command | BM25 token F1 | Model exact | Δ exact |
|---|---:|---:|---:|---:|---:|
| `val` | 0.601 | 0.916 | 0.811 | 0.001 | -0.601 |
| `val_unseen` | 0.006 | 0.131 | 0.259 | 0.001 | -0.005 |
| `test_seen` | 0.601 | 0.915 | 0.815 | 0.000 | -0.601 |
| `test_paraphrase` | 0.599 | 0.921 | 0.825 | 0.002 | -0.597 |
| `test_unseen` | 0.006 | 0.113 | 0.248 | 0.000 | -0.006 |
| `test_unseen_paraphrase` | 0.006 | 0.111 | 0.254 | 0.000 | -0.006 |

Averaged over seen-command splits, BM25 exact match is **0.600** vs **0.001** for the model. On unseen-command splits both are weak: **0.006** vs **0.000**. BM25 returns a memorized training command; the model can in principle compose a new one, but this 2.1M-parameter prototype has not learned to do so reliably. Treat the model as an educational baseline, not a command oracle.
<!-- RESULTS:END -->

### Focused core experiment (diagnostic)

A side variant narrows training to five command families (git, docker,
kubectl, cargo, systemctl) with the same 2.1M architecture trained for 20
epochs and a 5× first-command-token loss weight:

| Split | N | Exact | Command | Token F1 | Valid bash |
|---|---:|---:|---:|---:|---:|
| `val` | 187 | 0.048 | 0.866 | 0.227 | 0.952 |
| `test_seen` | 179 | 0.028 | 0.737 | 0.155 | 0.983 |
| `test_paraphrase` | 211 | 0.057 | 0.924 | 0.260 | 0.981 |

Command-family match rises from ≈7% to 74–92% and sudo over-prediction drops
from 42% to 6.7% — but exact match stays at 3–6% and BM25 still leads exact
match ~3×. The experiment isolates the 2.1M model's real bottleneck: family
selection is label-space limited, argument composition is capacity limited.
This diagnostic motivated the 10M-parameter run below; the core model is not
packaged as the default. Details: [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md),
artifacts in `artifacts/eval/tinyshell-core-2m/`.

Metric definitions:

- **Exact** — the normalized prediction matches the reference exactly.
- **Command** — the first executable token matches (e.g. both are `find`).
- **Token F1** — multiset F1 over arguments, ignoring the command name.
- **Valid bash** — `bash -n` parses the prediction. Syntax only; never executed.
- **p50 latency** — single-request CPU latency including tokenization/decoding.

Evaluation splits (page-level split, so commands do not leak across splits):

| Split | Commands | Prompt styles | What it measures |
|---|---|---|---|
| `val` | seen | held-out style 8 | checkpoint selection |
| `val_unseen` | unseen | seen styles | command generalization |
| `test_seen` | seen | held-out style 9, new slots | slot filling + style robustness |
| `test_paraphrase` | seen | unseen styles 10–11 | paraphrase robustness |
| `test_unseen` | unseen | seen styles | command generalization |
| `test_unseen_paraphrase` | unseen | unseen styles | hardest setting |

A BM25 retrieval baseline (returns the command of the most similar training
request) is evaluated on the same splits. It is expected to be strong when the
command was seen and weak on unseen commands, which is exactly the gap a
generative model is supposed to close.

## How it works

```text
tldr Markdown
   │  parse examples, filter unsafe/complex commands
   ▼
28,659 base examples ──► slot filling + 12 prompt styles
   │                       ├─ train: 135,860 requests (6 variants/example)
   │                       └─ eval:  6 × 1,500 requests (page-level holdout)
   ▼
4,096-token byte-level BPE (67 tokens/example median)
   │  prompt tokens masked with -100; loss only on command + EOS
   │  document-aligned packing + block-diagonal causal attention
   ▼
Llama-style decoder (2,098,816 params)
   RMSNorm · RoPE · causal SDPA · SwiGLU · tied embeddings
   │
   ▼
greedy decoding → clean command → risk warning → CLI output
```

For a guided tour of tokenization, packing, RoPE, attention, and the training
math, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). For the packing bug
found in the first run, the ablation that diagnosed it, the fixed second run,
and the focused core experiment, see
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

`scripts/build_core.py` derives the focused core subset from the same splits;
it does not change the tokenizer or the model architecture.

### Model

| | |
|---|---|
| Parameters | 2,098,816 total, 1,574,528 non-embedding |
| Layers | 6 |
| `d_model` / heads | 128 / 4 (head dim 32) |
| MLP | SwiGLU, intermediate size 512 |
| Context | 256 tokens |
| Vocab | 4,096 byte-level BPE |
| Positions | RoPE (θ = 10,000) |
| Norm | RMSNorm (pre-norm) |
| Embeddings | tied input/output |
| Dropout | 0.05 |

### Training

| | |
|---|---|
| Objective | next-token cross-entropy on command + EOS only |
| Optimizer | AdamW (β = 0.9, 0.95), weight decay 0.1 on matrices |
| LR | 3e-4 → 3e-5, cosine decay, 200-step warmup |
| Batch | 64 packed sequences × 256 tokens |
| Epochs | 4 |
| Steps | 2,608 |
| Tokens/epoch | 9,164,892 real tokens (10,686,208 positions incl. padding) |
| Gradient clipping | 1.0 |
| Hardware | 4 vCPU ARM64, CPU-only PyTorch |

Training logs, checkpoints, samples, and plots live in
`artifacts/runs/<run-id>/`. Best checkpoint is selected by validation loss.

## Quickstart

```bash
cd tinyshell
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"

# Build data (clones tldr-pages) and train the tokenizer
make data
make tokenizer

# Fast end-to-end sanity check (~1 minute)
make smoke

# Full CPU training (several hours on 4 vCPU)
make train

# Evaluate, plot, package
make eval
make plots
make package

# Focused core-command experiment (~1.5 hours on 4 vCPU)
make core
make core-train
make core-eval

# Use the model (or the retrieval baseline)
tinyshell "find all Python files modified in the last day"
tinyshell --json "show listening TCP ports"
tinyshell --retrieve "stream pod logs. Assume the pod name is alice"
tinyshell --interactive
```

If `artifacts/model/` exists it is used by default; otherwise the newest run
with `checkpoints/best.pt` is selected. You can also pass a run directory,
packaged model directory, or checkpoint file:

```bash
tinyshell --model artifacts/runs/tinyshell-2m-run2-docmask "unzip an archive here"
tinyshell --model artifacts/runs/tinyshell-2m-run2-docmask/checkpoints/best.pt "show disk usage"
```

### Commands

| Command | Purpose |
|---|---|
| `make data` | Clone/filter tldr-pages, build splits and manifest |
| `make tokenizer` | Train 4k BPE tokenizer |
| `make core` | Build the focused core-command subset |
| `make smoke` | 30-step training smoke test (small model) |
| `make train` | Full broad training run |
| `make core-train` | Train the focused core-command model |
| `make eval` | Evaluate the broad checkpoint + BM25 baseline |
| `make core-eval` | Evaluate the core-command model |
| `make package` | Copy the broad checkpoint/config/tokenizer/docs to `artifacts/model` |
| `make plots` | Broad-run training curves and eval bar chart |
| `make core-plots` | Core-run training curves and eval bar chart |
| `make results` | Print the broad model vs BM25 comparison |
| `make core-results` | Print the core model vs BM25 comparison |
| `make demo` | One example request |
| `make test` | Run the test suite |
| `make lint` | Ruff |

Equivalent Python entry points: `tinyshell-prepare`, `tinyshell-tokenizer`,
`tinyshell-train`, `tinyshell-eval`, `tinyshell-package`, `tinyshell-plots`,
and `tinyshell`.

The broad-run targets default to the shipped `tinyshell-2m-run2-docmask`
directory; override with `RUN=...` (for example, `make eval RUN=latest`). The
core targets use `CORE_RUN=artifacts/runs/tinyshell-core-2m`.

## Project layout

```text
tinyshell/
├── configs/
│   ├── tinyshell.yaml          # full broad run
│   ├── core.yaml               # focused core-command run
│   └── smoke.yaml              # 30-step pipeline test
├── data/                       # raw tldr clone + generated JSONL (gitignored)
├── artifacts/
│   ├── tokenizer/              # tokenizer.json, stats, vocab preview
│   ├── runs/                   # broad + core logs, checkpoints, samples
│   ├── eval/                   # metrics, report, predictions, failures
│   └── model/                  # self-contained packaged model
├── src/tinyshell/
│   ├── data.py                 # parse/filter/slot-fill/split tldr
│   ├── tokenizer.py            # train/load BPE
│   ├── dataset.py              # prompt masking + packing
│   ├── model.py                # Llama-style decoder
│   ├── train.py                # training loop, scheduler, checkpoints
│   ├── evaluate.py             # metrics, per-command report, BM25 comparison
│   ├── baseline.py             # BM25 retriever
│   ├── generate.py             # checkpoint loading + generation
│   ├── cli.py                  # command-line interface
│   ├── package.py              # build artifacts/model
│   ├── plots.py                # training/eval figures
│   ├── text.py                 # prompts, cleaning, scoring, risk checks
│   ├── config.py               # typed YAML config
│   └── utils.py                # seeding, logging, serialization
├── scripts/
│   ├── build_core.py           # build the focused core-command subset
│   ├── summarize_eval.py       # model vs BM25 markdown comparison
│   ├── analyze_errors.py       # command-confusion and failure samples
│   ├── compare_runs.py         # side-by-side checkpoint comparison
│   └── update_docs.py          # insert latest metrics into README/model card
├── tests/                      # fast unit tests (no training required)
├── docs/
│   ├── ARCHITECTURE.md         # learning-oriented design walkthrough
│   └── EXPERIMENTS.md          # packing bug, ablation, and run comparison
├── MODEL_CARD.md
├── DATA_CARD.md
└── Makefile
```

## Evaluation details

`make eval` writes:

```text
artifacts/eval/<run>/
├── metrics.json                 # all splits + BM25 baseline + config
├── report.md                    # tables and worst per-command results
├── <split>_predictions.jsonl    # every request, prediction, score, latency
└── <split>_failures.jsonl       # up to 50 non-exact examples per split
```

To evaluate a broad checkpoint on the focused core splits without retraining,
pass `--data-dir data/core` to `tinyshell.evaluate`; the BM25 baseline is then
built from `data/core/train.jsonl` as well.

Predictions are normalized only cosmetically (whitespace, wrapping backticks,
leading `$`) before scoring. No semantic equivalence is granted: different but
valid commands count as errors, so exact match is a lower bound on usefulness.
`token_f1` gives partial credit for argument overlap; `command_match` measures
whether the right tool was selected.

## Safety

- The model **never executes** generated commands.
- The CLI warns on `caution` and `destructive` patterns (e.g. `rm -rf`, `dd
  of=/dev/...`, `mkfs`, `shutdown`, piping downloads to a shell).
- `bash -n` validates syntax during evaluation; it does not validate semantics.
- Risk classification is a heuristic aid, **not a security boundary**. Review
  every command before running it.

## Limitations

- 2.1M parameters and 37M training tokens: this is a small model, not a
  replacement for a general assistant.
- One command per request; no pipelines unless the tldr example itself has one,
  no multi-turn context, no filesystem inspection.
- Training only on `pages/common` and `pages/linux`, in English. macOS, Windows,
  and BSD-specific commands are out of scope.
- Requests are synthetic paraphrases of tldr descriptions. Real users phrase
  things differently, and slot values are drawn from a fixed realistic pool.
- Exact match is strict. Different-yet-correct commands are counted as errors;
  token F1 and command match provide a more forgiving view.
- The model can hallucinate flags or produce plausible-but-wrong commands.
- The safety classifier is pattern-based and can miss dangerous commands or
  flag benign ones.
- BM25 is a retrieval baseline, not an oracle; it cannot compose unseen commands.

## License and attribution

Code and documentation: MIT (see `LICENSE`).

Training data: [tldr-pages](https://github.com/tldr-pages/tldr), licensed under
[CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). The exact source
commit used for the shipped run is recorded in
`data/processed/dataset_manifest.json` and `DATA_CARD.md`. tldr-pages is a
community project and is not affiliated with TinyShell.

Model weights and evaluation artifacts are released for research and learning.
See `MODEL_CARD.md` for intended use and out-of-scope uses.
