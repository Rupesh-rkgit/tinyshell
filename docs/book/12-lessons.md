# Chapter 12 — What We Learned: Findings, Lessons, and Engineering Notes

Part III told the story chronologically. This chapter consolidates it into
transferable conclusions — first about language models, then about the
practice of building them on tiny budgets.

## 12.1 Findings about models and learning

**1. Objective-conditioning mismatches are the commonest silent failure.**
Run 1 optimized `P(command | request, previous example)` while serving
`P(command | request)`. No loss curve reveals this; only generation probes
and mask ablations do. The general lesson: *any* discrepancy between the
training conditional and the serving conditional — packing, templates,
position conventions, special tokens — is a bug you must actively hunt.

**2. Diagnosability is an architectural feature.** Making `doc_ids` and
`position_ids` runtime inputs is what turned a hypothesis into a 15-minute
measurement (eval-time mask ablation: 3.866 → 3.306 on a frozen checkpoint).
Design models so that suspected mechanisms can be *switched at inference*.

**3. Benchmarks that orthogonalize axes of generalization pay for
themselves.** Because slot values, phrasings, and command pages were held
out independently, the project could say precisely: *scale fixed
interpolation (seen pages, 0.00 → 0.53 exact) and did nothing for
composition (unseen pages, 0.00 → 0.005 exact, loss regressed).* A single
 IID test set would have reported one number and taught nothing.

**4. Label-space size and model capacity are separable bottlenecks.** The
core experiment (Chapter 10) proved family selection was label-space
limited (0.36 → 0.87 command match from narrowing 3,651 → 111 first
tokens), while argument composition was capacity limited (exact match
stayed 3–6%). Different failures have different cures; measure before
scaling.

**5. More phrasings of known knowledge ≠ more knowledge.** Run 3's 150
variants improved seen splits and *worsened* unseen-page loss (3.80 →
4.15). Data scaling laws have a direction: diversity along the axis you
want to generalize along (pages), not along axes you already saturate
(phrasings).

**6. A retrieval baseline is a floor, a ceiling, and a mirror.** Floor: no
generative model is "useful on seen commands" while trailing BM25's 0.857
by 0.34. Ceiling: BM25's unseen-page ~0.8% shows what pure memorization
achieves. Mirror: the model's unseen-page token F1 edging past BM25's is
the first quantitative signature of composition. Report both, always.

**7. Frequency priors are data, not bugs — but they must be surfaced.** The
`sudo` bias (72% → 42% → 14.7% across runs 1→2→3) shrank with objective
fixes and capacity, but never disappeared; it concentrates exactly where
the model is most uncertain (obscure admin pages). Uncertainty-aware
warning UIs and calibration are product requirements, not research
afterthoughts.

**8. Loss fluency is cheap; correctness is expensive.** Valid-bash was
99.5% in the *broken* run. Perplexity 1.93 on seen pages coexists with 47%
wrong commands on the same distribution. Only task metrics measure the
task.

## 12.2 Findings about building on a CPU budget

**1. Small vocabularies are correct at small scale.** 4,096 BPE tokens keep
embeddings at 11% of the 9.6M model (vs 50%+ at 32k), while byte fallback
keeps arbitrary slot values representable. Median 67-token examples leave
context to spare.

**2. Throughput comes from layout, not heroics.** Fused QKV, bias-free
matrices, packed batches, `num_workers=0` with mmap'd arrays, and fp32 on
4 threads produced a steady 1,190 tok/s — enough for 100M tokens in a day.
No CUDA, no mixed precision, no distributed tricks.

**3. Complete-example packing costs ~14–17% positions and is worth 1,000×
that.** The padding budget is the explicit price of document-aligned
attention; run 1 quantified the alternative.

**4. Resume-grade checkpoints enable long runs on fallible machines.**
Storing optimizer + scheduler + progress counters (115 MB for run 3) made a
27.7-hour run survivable. For *distribution*, strip to model weights
(38 MB) — Chapter 15 shows the script.

**5. Run metadata is cheap insurance.** `env.json`, `git_commit.txt`,
per-run config and tokenizer snapshots cost kilobytes and made every
comparison in this book auditable months later. One caveat bit us anyway
(a history rewrite changed the recorded pre-launch commit hash), which is
itself the lesson: write down *everything*, including the exceptions.

**6. Dashboards change behavior.** `scripts/dashboard.py` (stdlib HTTP,
loss curves + ETA + eval tables) was built mid-run-3 and immediately paid
for itself; live generation samples caught run 1's pathology long before
the formal eval.

**7. Tests protect the science.** The packing invariants (no split
examples, doc-id resets, label masking, round-trips) are unit-tested
because run 1 vs run 2 differences *are* these details. Fast tests, no
training required, run on every change.

## 12.3 What we would do differently

- **Add a `val_seen_style_seen` split** (seen pages, seen styles) so
  checkpoint selection is not entangled with style generalization.
- **Save an eval-time attention-hook** for head-level attribution; the
  "lexical copying head" hypothesis in Chapter 3 is plausible but unverified.
- **Interleave page-diverse data in run 3** (e.g. cap variants per page and
  include more pages, or oversample rare first tokens) to test whether the
  val_unseen regression is inevitable or a data-mixing artifact.
- **Strip checkpoints at save time** behind a flag rather than post-hoc.
- **Track a calibration metric** (e.g. per-token entropy vs correctness) for
  the risk-warning story.

## 12.4 The project in three sentences

A 9.6M-parameter decoder, trained from scratch on one CPU-day of compute,
reaches 53–56% exact match on paraphrased requests for commands in its
training distribution and 0.5% on never-seen commands, while matching or
slightly beating a BM25 retriever on unseen-page token F1. The path there
was: fix a packing shortcut the loss curve happily hid (−1.04 nats),
diagnose label-space vs capacity limits with a controlled subset, then
scale the proven design 4.6×/10.9×. The remaining gap — composing commands
for tools it has never seen — is a data and knowledge-access problem, not a
decoding problem, and it defines the research agenda in Chapter 14.
