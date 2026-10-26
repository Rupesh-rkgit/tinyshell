# Chapter 14 — Research Frontiers: Where TinyShell Points Next

This chapter is the bridge between the book and the paper: the measured gaps
of Part III, restated as open research problems concrete enough to start on.
Each direction names the *evidence in this project* that motivates it, the
hypothesis to test, and the first experiment we would run.

## 14.1 Retrieval-augmented generation (RAG) for command composition

**Evidence.** BM25 beats the model on seen splits by 0.34 exact (0.857 vs
0.515); the model beats BM25 on unseen *token F1* by 0.001–0.029. The two
strengths are complementary: retrieval has the knowledge, generation has the
composition.

**Hypothesis.** Feeding retrieved examples into the prompt (or a small
cross-attention adapter) lets even a ~10M model compose commands for *unseen*
tools by conditioning on the retrieved page's flag vocabulary — turning
"unseen page" into "unseen *arrangement* of retrieved evidence."

**First experiment.** Ship the tldr page content (description+commands) of
the *eval-time* tool into the prompt for `test_unseen`, with the same 9.6M
model fine-tuned on such augmented prompts. Watch exact match on
`test_unseen` (0.005 baseline) and, crucially, whether seen-split performance
holds (no over-reliance on the retrieved command verbatim).

## 14.2 Page-level diversity and curriculum in the data mix

**Evidence.** Run 3's val_unseen loss *regressed* (3.80 → 4.15) while
seen-page loss collapsed — 150 variants per page hyper-fit known pages.

**Hypothesis.** The regression is a mixing artifact: cap variants per page
and spend the budget on more *pages* (or oversample rare first tokens), and
unseen-page loss should stop regressing at equal total tokens.

**First experiment.** Two corpora at fixed 100M tokens: (A) 150 variants ×
same pages (run 3 replication), (B) ~30 variants × ~3× pages (or rare-token
oversampling). Compare val/val_unseen losses and unseen-page exact match.
This is the cheapest high-information experiment in the list.

## 14.3 Execution-grounded training signals

**Evidence.** `bash -n` validity is 99.7%+ everywhere — syntax was never the
problem; semantics and flag correctness are (71% of core-run errors were
right-family-wrong-arguments). The pipeline never executes anything, so the
only supervision is string match.

**Hypothesis.** Cheap executable ground truth exists for a subset of
commands (file-system ops in a sandbox, `--help` round-trips, shellcheck
pass/fail). RL or filtered SFT on *execution-verifiable* requests should
improve flag precision without a human in the loop.

**First experiment.** Build a sandboxed executor for the ~100 core tools
(Chapter 10's list); run a rejection-sampling loop: sample k=8 commands per
request, keep those that execute correctly on a fixture filesystem, fine-tune
on winners. Metric: `test_seen` right-family-wrong-arguments rate (16.7%
baseline).

## 14.4 Calibration and uncertainty for safe suggestions

**Evidence.** The `sudo` prior (14.7% of test_seen predictions vs 5.5% of
references) concentrates on obscure pages — the model is most confident
*where it is most wrong*. The risk UI currently warns on the *output
string*, not on model uncertainty.

**Hypothesis.** Token-level entropy (or sequence log-prob) separates
"known-command suggestion" from "prior-driven guess," and thresholding it
would route uncertain cases to retrieval or to an explicit "I don't know."

**First experiment.** Recompute per-request log-prob under the run-3
checkpoint for all predictions; plot entropy vs correctness on seen vs
unseen splits; fit a routing threshold and report coverage/accuracy curves.
Artifacts to start from are already in `artifacts/eval/…/*_predictions.jsonl`.

## 14.5 Semantic-equivalent evaluation and multiple references

**Evidence.** Exact match is deliberately strict (different-but-valid = 0);
BM25's 0.857 is inflated by paraphrase-generation; token F1 rewards
invented-but-plausible flags partially.

**Hypothesis.** A graded rubric — executable-equivalence in a sandbox,
`--help` flag validation, tool-man pages for flag legality — correlates with
human usefulness better than exact match and removes the retrieval-vs-model
paradox (where the "oracle" is a duplicate generator).

**First experiment.** Hand-label 200 `test_seen` predictions as
{correct, valid-but-different, wrong}; compute agreement of each existing
metric and of the sandbox rubric with the labels. Publish the rubric with the
next data card.

## 14.6 Scaling the *right* axis: parameters vs pages

**Evidence.** 4.6× parameters fixed composition on *seen* pages; nothing
fixed unseen pages. The core experiment showed family selection is
label-space limited.

**Hypothesis.** Unseen-page exact match is a function of (pages seen,
parameters) roughly independently — i.e. a "Chinchilla for coverage" law
exists where the sample axis that matters is *distinct tools*, not tokens.

**First experiment.** Train the 9.6M model on 4 corpora: {100M tokens,
1,303k requests, same pages} (replication), {100M tokens, fewer pages}, and
two variants capping variants per page; fit seen/exact and unseen/loss
against page count. If the hypothesis holds, the corpus builder gains a
principled allocation rule between variants and coverage.

## 14.7 Byte/char-level tokenization at tiny scale

**Evidence.** The 4,096 vocab keeps embeddings at 11% of parameters, but
median 67-token examples and 229-token maxes inflate attention cost; rare
tool names fragment into many tokens (`mysql_secure_installation`), and the
model must spell commands letter-for-letter.

**Hypothesis.** At this scale a byte-level model (or a hybrid: byte LM with
a small BPE assist) improves unseen-page copying of exact tool names — the
failure mode `zeroclaw agent → hangman` involves tokenization-adjacent
spelling uncertainty.

**First experiment.** Same data, byte-level tokens (vocab 512 + specials),
context 512, ~9.6M params; compare exact/F1 and *spelling* error rates on
tool names.

## 14.8 Small-model interpretability: where composition lives (or doesn't)

**Evidence.** The design doc hypothesizes head specializations (lexical
copying, family selection, flag continuation) but no attribution was run;
the mask ablation nevertheless proved *context-level* mechanisms are
readable at this scale.

**Hypothesis.** Tiny models are cheap enough for exhaustive intervention:
head ablation, activation patching between seen/unseen requests, and
dictionary learning over residual streams can localize why argument
composition fails.

**First experiment.** Ablate single heads at inference on 200 `test_seen`
pairs where the only error is the flag set; publish which heads/events flip
the flags. The repo's 272-line model makes this tractable in a weekend.

## 14.9 From suggestion to supervised agency (with the safety rails)

**Evidence.** The model card's out-of-scope list forbids autonomous
execution; the CLI never runs anything. That is correct *for this model*,
whose 0.5% unseen exact match would make an agent dangerous.

**Hypothesis.** The agent setting is still the natural end state for the
task — but it needs (a) execution-verified training signals (14.3),
(b) calibrated uncertainty (14.4), and (c) sandboxed environments with
revert. Benchmarks in this spirit exist for general agents (e.g. InterCode,
Yang et al. 2023); a tiny-shell analogue with hard sandboxing would be a
useful, safe testbed.

**First experiment.** Wrap the executor of 14.3 into a stateful sandbox
(fixture filesystem, network off, snapshot/rollback), and measure task
success of a tiny model with tool feedback versus the one-shot setting.

## 14.10 The meta-frontier: honest small-model evaluation as a contribution

Every finding in this book is a measurement-design result as much as a model
result: orthogonal generalization axes, a by-construction-strong retrieval
baseline, eval-time mechanism ablations, per-command error censuses. Small
models are the ideal substrate for this kind of *evaluation science* — cheap
enough to run many controls — and the community has few templates for it at
this scale. Formalizing the six-split design, the BM25 oracle caveat, and
the census methodology into a reusable benchmark suite (different target
languages: SQL, regex, Makefile, PowerShell) is itself a publishable,
extension-shaped continuation of this project.
