# Chapter 10 — The Core Experiment: Label Space vs Capacity

> Run ID: `tinyshell-core-2m` · 2,098,816 params · 20 epochs · 1,620 steps ·
> ~2.2 h · dropout 0.1 · `first_token_weight` 5.0 · best val **1.5472**
> Data: `data/core/` (655 pages, 17,317 requests, 111 first-command tokens)
> Artifacts: `artifacts/runs/tinyshell-core-2m/`, `artifacts/eval/tinyshell-core-2m/`

After run 2 the question was sharp: the 2.1M model has learned *forms*
(copying, flag syntax) but fails *selection* across 3,651 command families.
Is that because the task is too hard for the model, or because the label
space is too big for the data and parameters? This chapter describes the
controlled experiment that answers it — and the two conclusions it forced.

## 10.1 Design

Build a focused subset from the same splits and pools (Chapter 4.9): keep
pages whose title starts with one of ~100 curated common tools. Result:

- 655 pages; 17,317 training requests; 5,186 packed chunks;
- 1,143,753 real tokens per epoch; 176,359 supervised label tokens per epoch;
- **111 distinct first command tokens** (91 curated; ~20 shell snippets or
  aliases that slipped through the title-based filter — e.g. a crontab
  schedule line and an `xargs` pipeline; run 3's stricter example-level
  filter would reject these);
- eval: `val` 187, `test_seen` 179, `test_paraphrase` 211 requests (unseen
  core splits alias the seen ones — every retained page is in training).

Training: the same 2.1M architecture, 20 epochs, dropout raised 0.05 → 0.1
(more epochs on less data), `first_token_weight` 5.0 to upweight the
command-family decision, batch 64, 1,620 steps, ~2.2 h.

## 10.2 Results

| Split | N | Exact | Command | Token F1 | Valid bash | p50 |
|---|---:|---:|---:|---:|---:|---:|
| `val` | 187 | 0.048 | **0.866** | 0.227 | 0.952 | 161 ms |
| `test_seen` | 179 | 0.028 | **0.737** | 0.155 | 0.983 | 161 ms |
| `test_paraphrase` | 211 | 0.057 | **0.924** | 0.260 | 0.981 | 156 ms |

BM25 on the same splits (from the core eval report):

| Split | BM25 exact | BM25 command | BM25 token F1 |
|---|---:|---:|---:|
| `val` | 0.674 | 1.000 | 0.878 |
| `test_seen` | 0.642 | 0.983 | 0.857 |
| `test_paraphrase` | 0.640 | 0.976 | 0.864 |

**Family selection became a nearly solved subproblem.** On the same 187
`val` examples, the broad run-2 model scores 0.358 command match / 0.144
token F1 (evaluated against the core data; predictions in
`artifacts/eval/tinyshell-2m-run2-coreval/`) versus **0.866 / 0.227** for the
core-trained model. The comparison is not a fully controlled ablation — the
core training set is a 17,317-request slice of the broad corpus — but the
effect size (0.36 → 0.87 command match) is far beyond that confound.

**The `sudo` prior collapsed.** 12 of 179 `test_seen` core predictions
(6.7%) begin with `sudo`, versus 626/1,500 (42%) for run 2. With a small
label space plus a 5× first-token weight, `sudo` becomes a *conditional*
decision (some requests warrant it) instead of a constant prior.

**Invalid-bash rate essentially unchanged** (1.7% vs 1.3% on test_seen) —
packing was never the source of malformed output; the remaining invalids are
template-substitution artifacts inherited from tldr.

## 10.3 The bottleneck moves to arguments

On `test_seen`, 127 of 179 predictions (71%) choose the correct command
family but compose wrong arguments; only 5/179 are exact. Signature failure:

```text
request:    stream pod logs. Assume the pod name is `alice`.
reference:  kubectl logs --follow alice
prediction: kubectl logs --image alpine:3.12-project
```

Right family, right entity, invented flag semantics. And BM25 still triples
the model's exact match (0.642 vs 0.028): retrieval copies a complete
training command; the model must compose one.

**Conclusion (as recorded in `docs/EXPERIMENTS.md`):** *family selection is
label-space limited; argument composition is model/data limited.* The first
is solved by narrowing the label space; the second is not solved by anything
tried so far — which is precisely the hypothesis run 3 was designed to test
with 4.6× the capacity.

## 10.4 Implementation caveats, on the record

`docs/EXPERIMENTS.md` documents two honest footnotes:

1. The shipped core run used the **first version** of the weighted loss,
   which tagged only the first active token *per packed chunk* (~one example
   in three). The implementation now tags the first command token of *every*
   example (Chapter 6.7); retraining the core run would apply the full 5×
   weight everywhere, so the published numbers are a slightly conservative
   estimate of the weighting's effect.
2. Title-based filtering (vs example-based) is the root cause of the ~20
   non-command first tokens in the 111; noted as a data-quality caveat and
   fixed at the corpus level in run 3's stricter filters.

## 10.5 What this experiment contributes methodologically

It demonstrates a pattern worth generalizing: when a small model fails at a
composite task, **decompose the failure and bound each component's
difficulty**. Here one surgical data change (label space 3,651 → 111)
converted command match from 0.36 to 0.87, proving selection was never a
capacity problem. That diagnosis, not the core model itself (which is not
packaged as a default), is the deliverable — and it is what directed run 3's
design budget toward capacity and data volume rather than architecture
novelty.
