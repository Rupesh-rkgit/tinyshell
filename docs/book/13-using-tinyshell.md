# Chapter 13 — Using TinyShell: CLI, Retrieval Mode, Packaging, and Safety

This chapter is the user manual for the finished system. Everything here
works from a fresh clone after `make data` + `make tokenizer` (and either a
shipped `artifacts/model/` or your own `make train`).

## 13.1 Installation

```bash
cd tinyshell
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"

make data        # clone tldr (pinned commit), parse, filter, split
make tokenizer   # train the 4,096-token BPE
make smoke       # 30-step end-to-end sanity check (~1 minute)
```

`make smoke` trains a tiny model for 30 steps and runs the eval plumbing —
the fastest way to verify the environment before committing to a full run.

## 13.2 The CLI

```bash
# Neural model (default: packaged model, else newest run with best.pt)
tinyshell "find all Python files modified in the last day"
tinyshell --json "show listening TCP ports"
tinyshell --interactive

# Explicit model selection
tinyshell --model artifacts/runs/tinyshell-xl-10m "unzip an archive here"
tinyshell --model artifacts/runs/tinyshell-xl-10m/checkpoints/best.pt "show disk usage"

# Retrieval baseline (BM25 over training requests)
tinyshell --retrieve "stream pod logs. Assume the pod name is alice"
```

Output shape:

```text
$ kubectl logs --follow alice

  valid bash | risk: safe | 6 ms
  mode: BM25 retrieval baseline (nearest training request)
```

The neural equivalent prints the generated command with the same banner
(mode `neural`, latency in ms, generated-token count in `--json`). `--json`
emits the full record: request, command, raw generation, `valid_bash`,
`risk`, `risk_reasons`, latency, mode, and (retrieval mode) the retrieved
source request — everything the evaluator scores, available interactively.

Model resolution (`_default_model` / `load_model`): packaged
`artifacts/model/` if present; otherwise the newest run with
`checkpoints/best.pt`; else pass anything the loader understands — a run
directory, a packaged directory, or a checkpoint file.

## 13.3 Which mode should you use?

Honest guidance from the measured results (Chapter 11):

| Situation | Mode | Why |
|---|---|---|
| Command likely in the training corpus (tldr coverage) | `--retrieve` | 0.84–0.88 exact, 0 ms, perfectly faithful to its source |
| Modification of a known command (new slots/phrasing) | neural | 0.52–0.56 exact; composes instead of copying |
| Command from a page never in the corpus | neither, really | 0.5% exact; read output as a *guess*, verify manually |

The README states it plainly: broad-task suggestion at 9.6M parameters works
inside the training distribution; the retrieval mode stays stronger there;
the neural path's unique value is composition and the research platform.

## 13.4 Safety model

Three layers, in order of authority:

1. **The model never executes anything.** There is no code path from a
   generated command to a shell. The CLI prints; the human runs.
2. **Risk warnings** (`assess_risk` in `text.py`): pattern classes over the
   generated command —
   - *destructive*: `rm -rf`, `dd of=/dev/…`, `mkfs`, `shutdown/reboot`,
     `wipefs/fdisk/parted`, raw device writes, `chmod -R 777 /`, fork bombs,
     `truncate -s 0`, `shred`;
   - *caution*: `sudo`, `rm`, `mv`/`cp` overwrites, `chmod`/`chown`,
     `kill/pkill`, service changes, firewall changes, git history rewrites,
     `curl … | sh`, `find … -delete`, `xargs … rm`, redirection.
   The CLI prints `risk: caution|destructive` with reasons.
3. **`bash -n` validity** is displayed — syntax check only.

The risk classifier is explicitly **not a security boundary** (its docstring
says so): it has false negatives and false positives, and the model card's
out-of-scope uses (autonomous execution, incident response, destructive
operations) flow from that.

## 13.5 Packaging and distribution

```bash
make eval      # six splits + BM25, artifacts/eval/<run>/
make plots     # loss curves + eval bars
make package   # -> artifacts/model/
```

`artifacts/model/` is a **self-contained drop**: `model.pt` (best
checkpoint), `config.yaml`, `tokenizer/`, `model_summary.json`,
`final_metrics.json`, `eval_report.md`, `MODEL_CARD.md`, `DATA_CARD.md`,
`EXPERIMENTS.md`, and a generated `manifest.json` (run name, creation time,
checkpoint provenance, model config, data attribution). Anyone with this
directory plus the repo's CLI can reproduce every eval number — no data
pipeline required for inference.

## 13.6 The dashboard

```bash
make dashboard   # http://localhost:8585
```

`scripts/dashboard.py` is a dependency-free (stdlib only) HTTP server that
renders: all runs with parameter counts and status, live loss curves from
`metrics.jsonl`, latest eval tables, and step-rate/ETA. A `--poll` mode
refreshes while training runs. Built during run 3; the entire 27.7-hour run
was monitored through it.

## 13.7 Tests and linting

```bash
make test   # pytest: data, tokenizer, packing, model shapes, config, CLI, eval plumbing
make lint   # ruff
```

Notable test subjects (Chapter 5.5 lists the packing invariants;
`tests/test_model.py` checks mask construction, parameter counts, tied
weights, and generation stopping). All tests are fast and require no
training — they run in CI-style on every change.

## 13.8 Make targets, complete map

| Target | Purpose |
|---|---|
| `make data` | clone + parse + split + manifest (tldr) |
| `make tokenizer` | train 4k BPE |
| `make core` | build the focused core subset |
| `make smoke` | 30-step pipeline test |
| `make train` | full broad training (config `configs/tinyshell.yaml`; `RUN=` to override) |
| `make train-xl` | run-3 config (`configs/tinyshell-xl.yaml`) |
| `make core-train` / `make core-eval` | core experiment |
| `make eval` | six-split generation eval + BM25 |
| `make plots` / `make core-plots` | figures |
| `make package` | build `artifacts/model` |
| `make dashboard` | live dashboard |
| `make results` / `make core-results` | model-vs-BM25 markdown |
| `make demo` | one example request |
| `make test` / `make lint` | quality gates |

Python entry points behind them: `tinyshell-prepare`, `tinyshell-tokenizer`,
`tinyshell-train`, `tinyshell-eval`, `tinyshell-package`,
`tinyshell-plots`, and the `tinyshell` CLI.
