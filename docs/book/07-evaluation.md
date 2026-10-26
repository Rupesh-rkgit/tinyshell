# Chapter 7 — Evaluation Design: Measuring a Command Generator Honestly

Evaluation is where language-model projects quietly lie to themselves, so
TinyShell's evaluation (`src/tinyshell/evaluate.py`) is built around one rule:
**never score a single number when the interesting quantity is a gap.** This
chapter defines the metrics, the splits, the baseline, and the artifacts the
evaluator writes.

## 7.1 The six splits and what each isolates

| Split | Pages | Prompt style | What it measures |
|---|---|---|---|
| `val` | seen | 8 (held out) | checkpoint selection under unseen phrasing |
| `val_unseen` | unseen | 0–7 (seen) | loss-side view of command generalization |
| `test_seen` | seen | 9 (held out) + new slots | slot filling + style robustness |
| `test_paraphrase` | seen | 10–11 (unseen) | phrasing robustness |
| `test_unseen` | unseen | 0–7 (seen) | command generalization |
| `test_unseen_paraphrase` | unseen | 10–11 | both axes held out |

All test splits have 1,500 examples (val: 1,499 after dedup). Because the
page split is deterministic and identical across runs, the *same* held-out
pages appear in run 2 and run 3 — the property that makes the scaling
comparison in Chapter 11 meaningful.

## 7.2 The metrics

**Exact match.** After cosmetic normalization (whitespace collapse, unwrap
backticks, strip a leading `$`/`>`, strip trailing `;`), prediction must equal
the reference string. No semantic equivalence is granted: `ls -la` vs
`ls -lah` is a *miss*. Exact match is therefore a **lower bound on
usefulness** — the right primary metric for a benchmark, the wrong one for
judging usefulness alone.

**Command match.** The first executable token matches, skipping wrappers
(`sudo`, `env`, `time`, `nohup`, …) and `VAR=value` prefixes
(`command_name`). This separates *"chose the right tool"* from *"got the
details right"* — the axis that separated run 2 (0.08) from run 3 (0.69).

**Token F1.** Multiset F1 over `shlex`-tokenized arguments, excluding the
command name. Gives partial credit when the model gets 3 of 4 arguments right.
The harmonic balance of precision and recall punishes both invented flags
(precision) and dropped ones (recall).

**Valid bash.** `bash -n` parses the prediction (3 s timeout, stdin). Syntax
only, never executed. A sanity floor: a model that emits unparseable junk
fails this regardless of the other metrics.

**Latency p50.** Wall-clock per request, tokenization + decoding included.
The shipped model generates in ~230–290 ms on the 4-vCPU host.

## 7.3 The BM25 retrieval baseline

`src/tinyshell/baseline.py` implements BM25 (k1 = 1.5, b = 0.75; Robertson &
Zaragoza, 2009) over the **training requests**:

- Tokenization is regex `[a-z0-9_./*+-]+` lowercased — shell-aware tokens.
- An inverted index is precomputed where each posting already carries its
  full BM25 weight, so a query is one vector addition per distinct query
  token. Full-corpus queries take single-digit milliseconds (p50 latency 0 ms
  in the eval tables).
- `search` returns the top-1 training row; the evaluator scores that row's
  **command** against the split's reference command.

Why this baseline and why it is *expected* to be brutally strong: evaluation
requests are **generated paraphrases of training requests**. Retrieving the
nearest training request and copying its command is close to an oracle on
seen-command splits (measured: 0.84–0.88 exact). On unseen pages the
retriever returns some unrelated training command (measured: ~0.8% exact).
The baseline's role is calibration: a generative model claiming usefulness on
seen commands must be compared against 0.85, and the *interesting* number for
a generative model is unseen-command behavior, where retrieval structurally
cannot help.

## 7.4 The evaluation procedure

For each split, for each row:

1. Render the prompt, encode, truncate the left if over context (rare).
2. Greedy decode up to 64 new tokens, stop at `<|eos|>`.
3. Clean (`clean_generation`): strip special tokens/code fences, keep the
   first non-empty line, strip shell prompts and backticks.
4. Score all four metrics; time the request.
5. Append the full record to `<split>_predictions.jsonl`; if not exact, to
   `<split>_failures.jsonl` (capped at 50 per split for eyeballing).

Outputs per run in `artifacts/eval/<run>/`:

| File | Content |
|---|---|
| `metrics.json` | every split × metric, model + BM25, config snapshot |
| `report.md` | tables + per-command worst lists |
| `<split>_predictions.jsonl` | every request, prediction, scores, latency |
| `<split>_failures.jsonl` | failure samples for qualitative analysis |

The predictions files are the raw material for the error census in Chapter
11; keeping them (14 MB across runs) made every qualitative claim in this
book re-derivable.

## 7.5 Per-command breakdowns

`report.md` also aggregates metrics **per command** (page title). Two uses:

- **Worst lists.** For run 3's `test_unseen`, pages like `hatch`,
  `install-tl`, `krunvm`, `rustscan`, `wait4x` score 0/3 with 0 command
  match — the concrete face of the generalization gap.
- **Near-miss families.** On seen splits the errors concentrate in sibling
  tools: `phpcs` ↔ `phpcbf`, `7zr` ↔ `7za`, `bun` ↔ `npm` — wrong-family
  errors that a human reviewer would catch instantly, but exact match
  (correctly) counts as failures.

## 7.6 Metric caveats, stated on the record

From `MODEL_CARD.md`, worth repeating verbatim in spirit:

- Exact match is strict: a different-but-valid command counts as wrong.
- `bash -n` is syntax only; it cannot detect dangerous or semantically wrong
  commands.
- Test prompts are synthetic paraphrases; real-user performance is likely
  lower.
- BM25 retrieves from training requests — strong on seen commands *by
  construction*, and unable to compose unseen commands.

## 7.7 The dashboard and the plot scripts

Two helpers make the numbers visible without a notebook:

- `make dashboard` (`scripts/dashboard.py`): a dependency-free stdlib HTTP
  server on `localhost:8585` listing runs, live loss curves from
  `metrics.jsonl`, eval tables, and per-run ETA. Built mid-run-3 precisely
  because staring at `tail -f train.log` was not scaling.
- `tinyshell-plots` (`plots.py`): training curves and eval bar charts from
  the same JSONL/JSON artifacts into each run's `plots/`.

Neither is required for the pipeline; both were repeatedly load-bearing for
the decisions documented in Part III.
