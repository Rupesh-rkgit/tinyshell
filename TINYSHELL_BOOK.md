# TinyShell: Building a Small Language Model from Scratch on a CPU

**A complete learning book — theory, implementation, experiments, and results**

*Version 1.0 · October 2026 · TinyShell contributors*

---

## What this book is

This book is a complete, self-contained account of the **TinyShell** project: a
2.1M- and 9.6M-parameter decoder-only language model, trained **from random
initialization on a CPU**, that converts natural-language requests into Linux
shell commands. It is written for a reader who wants to *learn by following a
real project end to end* — every design decision, every bug, every experiment,
and every number in this book comes from the repository you are reading it in.

Three things make this book different from a tutorial:

1. **It includes the failures.** The first training run had a subtle data
   packing bug that produced a model with 0% exact match. The diagnostic path
   that found the bug (prefix probes, mask ablations) is documented step by
   step, because the debugging process teaches more than the fix.
2. **It is honest about baselines.** A trivial BM25 retrieval baseline beats
   the neural model by a wide margin on seen commands. The book explains why
   that is by construction, why it is still the right baseline, and what the
   comparison does and does not say.
3. **It separates what scaling fixes from what it does not.** Going from
   2.1M parameters / 9M tokens to 9.6M parameters / 100M tokens moved
   seen-command exact match from ~0% to 53–56% — but left unseen-command
   generalization at 0.5%. The book treats that asymmetry as the central
   scientific finding, not a footnote.

## Who this is for

- Students and engineers who have read about transformers and want to see
  every piece — tokenization, packing, masking, training, evaluation,
  packaging — implemented in a real, runnable repository.
- Practitioners who want to train a small model on a single CPU or a borrowed
  GPU and need realistic expectations about data volume, throughput, and
  evaluation design.
- Researchers looking for a compact, reproducible testbed for questions about
  composition, memorization, and retrieval-versus-generation.

You do not need prior transformer experience; Chapters 2 and 3 build the theory
from scratch. You do need comfortable Python and enough Linux to run `make`.

## How the book is organized

| Part | Chapters | What it covers |
|---|---|---|
| **I — Foundations** | 1–3 | The project; language-model theory; transformer anatomy |
| **II — Building the system** | 4–6 | Task & data engineering; tokenization & packing; the training loop |
| **III — Experiments** | 7–11 | Evaluation design; run 1 (the bug); run 2 (the fix); the core diagnostic; run 3 (scaling) |
| **IV — Understanding** | 12–14 | Lessons; using the system; research frontiers |
| **Appendices** | 15–16 | Reproduction guide; glossary |

## Table of contents

- [Chapter 1 — Introduction: Why Build a Tiny Language Model?](01-introduction.md)
- [Chapter 2 — Language Models from Scratch: The Theory You Need](02-language-models.md)
- [Chapter 3 — Transformer Anatomy: Every Piece TinyShell Uses](03-transformer-anatomy.md)
- [Chapter 4 — The Task and the Data](04-data-engineering.md)
- [Chapter 5 — Tokenization and Packing](05-tokenization-packing.md)
- [Chapter 6 — The Training Loop](06-training.md)
- [Chapter 7 — Evaluation Design](07-evaluation.md)
- [Chapter 8 — Run 1: The Packing Bug](08-run1-packing-bug.md)
- [Chapter 9 — Run 2: Document-Aligned Packing](09-run2-docmask.md)
- [Chapter 10 — The Core Experiment](10-core-experiment.md)
- [Chapter 11 — Run 3: Scaling to 9.6M Parameters](11-run3-xl-scaling.md)
- [Chapter 12 — What We Learned](12-lessons.md)
- [Chapter 13 — Using TinyShell](13-using-tinyshell.md)
- [Chapter 14 — Research Frontiers](14-research-frontiers.md)
- [Chapter 15 — Reproduction Guide](15-reproduction-guide.md)
- [Chapter 16 — Glossary](16-glossary.md)

## The single-file edition

All chapters are combined, in reading order, into
**[`TINYSHELL_BOOK.md`](../../TINYSHELL_BOOK.md)** at the repository root.
Read that file if you want one continuous document; read the chapter files in
`docs/book/` if you want to navigate.

## The numbers in one glance

| | Run 1 | Run 2 | Core | Run 3 (shipped) |
|---|---|---|---|---|
| Parameters | 2,098,816 | 2,098,816 | 2,098,816 | 9,573,632 |
| Training tokens | 9.16M × 4 epochs | 9.16M × 4 epochs | 1.14M × 20 epochs | 99.99M × 1 pass |
| Packing | naive | document-aligned | document-aligned | document-aligned |
| Best val loss | 3.9749 | 2.9377 | 1.5472 | **0.6565** |
| Val loss (unseen pages) | 4.6025 | 3.7993 | — | 4.1451 |
| Exact (test_seen) | 0.000 | 0.000 | 0.028 | **0.525** |
| Command match (test_seen) | 0.084 | 0.081 | 0.737 | **0.692** |
| Token F1 (test_seen) | 0.117 | 0.173 | 0.155 | **0.644** |
| Exact (test_unseen) | 0.000 | 0.000 | — | 0.005 |
| Train time | ~3 h | 3 h 16 m | ~2.2 h | 27.7 h |
| Hardware | 4 vCPU ARM64 | 4 vCPU ARM64 | 4 vCPU ARM64 | 4 vCPU ARM64 |

## Companion documents

- [`README.md`](../../README.md) — project overview and quickstart
- [`docs/ARCHITECTURE.md`](../ARCHITECTURE.md) — a shorter design walkthrough
- [`docs/EXPERIMENTS.md`](../EXPERIMENTS.md) — the raw experiment log
- [`DATA_CARD.md`](../../DATA_CARD.md) / [`MODEL_CARD.md`](../../MODEL_CARD.md) — dataset and model documentation
- [`paper/`](../../paper/) — a research paper written from this project

## License and data attribution

Code and text: MIT. Training data derived from
[tldr-pages](https://github.com/tldr-pages/tldr) (CC-BY-4.0). See `LICENSE`
and `DATA_CARD.md` for details and attribution requirements.


---

# Part I — Foundations


---

# Chapter 1 — Introduction: Why Build a Tiny Language Model?

## 1.1 The project in one paragraph

TinyShell is an end-to-end language-model project with one narrow goal: given an
English request such as *"list all files including hidden ones, with
human-readable sizes"*, produce the Linux command `ls -lah` — and nothing else.
The model is a Llama-style decoder-only transformer trained **from random
initialization** on a CPU-only ARM64 machine (4 vCPU cores, no GPU). Everything
around the model is part of the project: cloning and parsing the tldr-pages
corpus, filtering unsafe commands, filling placeholder slots with realistic
values, rendering 16 prompt styles, training a 4,096-token byte-level BPE
tokenizer, packing examples into fixed-length chunks with document-aligned
attention, training, evaluating against a BM25 retrieval baseline, packaging the
model, and documenting all of it.

The final shipped model (`tinyshell-xl-10m`, run 3) has **9,573,632
parameters**, was trained on **99,992,950 packed tokens** in **27.7 hours**, and
reaches **52–56% exact match** on held-out test splits of commands it saw during
training — while scoring **0.5%** on commands whose pages were never seen. The
smaller 2.1M model (run 2) reaches essentially 0% exact match everywhere. That
contrast, and the diagnostic experiments that explain it, is the intellectual
core of this book.

## 1.2 Why natural language → shell command?

Any end-to-end LM project needs a task. The choice is the first real design
decision, and most choices are bad ones: tasks that are too big make small
models fail for uninteresting reasons, and tasks that are too easy make the
project boring. Natural-language-to-shell-command sits in a useful middle:

1. **The output is short.** The median command in our corpus is 26 characters,
   roughly 8–12 tokens. A 256-token context window is generous; generation
   stops after at most 64 new tokens. Long contexts are where small models and
   CPUs suffer most, so the task keeps the compute bounded.

2. **Correctness is measurable.** Unlike "write a poem," a command can be
   scored: exact string match after normalization, first-token (command) match,
   multiset token F1 over arguments, and a `bash -n` syntax check. None of
   these require a judge model or human rating, so evaluation is reproducible
   and cheap.

3. **Generalization can be *separated* into axes.** This is the subtle reason
   and the most important one. Because we generate the evaluation data
   ourselves, we can hold out different things independently:
   - hold out **slot values** (new filenames, hosts, packages), or not;
   - hold out **prompt phrasings** (styles 8–11 never appear in training), or not;
   - hold out **entire command pages** (so the model must compose a command it
     has literally never seen), or not.
   
   Standard benchmarks mix these axes together. In TinyShell they are
   orthogonal by construction, which is what made it possible to say precisely
   what a 10× scaling changed and what it did not.

4. **It is genuinely useful.** Every Linux user has stood in front of a terminal
   thinking *"I know this is possible, what was that flag?"* An offline,
   200–300 ms, 40 MB suggestion tool that never executes anything is a
   realistic product shape — provided the user stays in the loop, which the CLI
   enforces by design (it never runs the generated command).

5. **The data exists and is clean.** tldr-pages is a curated, human-written
   corpus of exactly this format: a short description plus a command template.
   It is small enough to process on a laptop and licensed CC-BY-4.0.

The closest prior work is NL2Bash (Lin et al., 2018), a 9,000-pair corpus of
web-scraped commands with expert descriptions, and Tellina (Lin et al., 2017),
an seq2seq semantic parser built on that idea. TinyShell differs in source
(tldr-pages rather than web scraping), scale (tiny, from scratch, CPU), and
purpose (an educational instrumented pipeline with an honest generalization
audit rather than a state-of-the-art system).

## 1.3 What was actually built

The repository contains the full pipeline as runnable code:

| Stage | Entry point | What it does |
|---|---|---|
| Data collection | `make data` → `tinyshell-prepare` | Sparse-clones tldr-pages at a pinned commit, parses Markdown pages, filters examples, fills slots, renders styles, writes JSONL splits + manifest |
| Tokenizer | `make tokenizer` → `tinyshell-tokenizer` | Trains a 4,096-token byte-level BPE on rendered training text; verifies special tokens and byte round-trips |
| Packing | `python -m tinyshell.dataset` | Tokenizes every split and packs complete examples into 256-token chunks with `doc_ids` and per-example `position_ids` |
| Model | `src/tinyshell/model.py` | RMSNorm, RoPE, fused causal SDPA attention, SwiGLU, tied embeddings, block-diagonal document mask |
| Training | `make train` → `tinyshell-train` | AdamW with decoupled weight decay, warmup + cosine schedule, gradient clipping, checkpoint/resume, periodic generation samples |
| Evaluation | `make eval` → `tinyshell-eval` | Greedy decoding on six splits; exact / command / token-F1 / `bash -n` / latency; BM25 baseline on the same examples; per-command report |
| Baseline | `src/tinyshell/baseline.py` | BM25 retriever with a NumPy inverted index over training requests |
| Serving | `tinyshell` CLI | Neural and `--retrieve` modes, `--json`, interactive mode, risk warnings |
| Packaging | `make package` | Self-contained `artifacts/model/` with checkpoint, config, tokenizer, cards, eval report |
| Observability | `make dashboard` | Stdlib-HTTP dashboard: runs, loss curves, eval tables, ETA |
| Diagnostics | `scripts/*.py` | Model-vs-BM25 summaries, error census, run comparison, doc updates |
| Tests | `make test` | Fast unit tests covering data, tokenizer math, model shapes, packing invariants, CLI |

Four training runs anchor the story:

- **Run 1** (`tinyshell-2m-run1`) — naive sequence packing; 0% exact match; the
  bug chapter.
- **Run 2** (`tinyshell-2m-run2-docmask`) — document-aligned packing; fixed the
  shortcut; still ~0% exact but mechanically sound.
- **Core** (`tinyshell-core-2m`) — 2.1M model trained on a 111-command-family
  subset; isolates label-space limits from capacity limits.
- **Run 3** (`tinyshell-xl-10m`) — 9.6M parameters, 100M tokens, stricter
  filters; the shipped model.

## 1.4 The four headline findings

1. **Naive packing teaches a shortcut, and validation loss hides it.** Run 1
   trained to a plausible-looking loss (3.97) while learning to copy from
   unrelated packed examples — `P(command | request, previous example)`
   instead of `P(command | request)`. Blocking cross-example attention at
   *evaluation time only* improved its validation loss by 0.56 nats with zero
   retraining, proving the shortcut existed.

2. **A retrieval baseline is the best sanity check available.** BM25 over
   training requests reaches 0.84–0.88 exact match on our seen-command test
   splits (they are paraphrases of training data, so retrieval is nearly an
   oracle there) and ~0.8% on unseen commands. Any generative model must be
   read against both numbers.

3. **Label-space size, not task difficulty, set the 2.1M model's ceiling.**
   With 3,651 distinct first command tokens, run 2 chose the right command
   family 6–9% of the time. Restricting training to 111 families with a 5×
   first-token loss weight raised family selection to 74–92% without touching
   the architecture.

4. **Scale fixes composition on seen commands, not generalization to new
   ones.** 4.6× parameters and 10.9× tokens moved test_seen exact match from
   0.000 to 0.525 and paraphrase exact match to 0.559 — but unseen-page exact
   match stayed at 0.005, and unseen-page validation *loss* actually worsened
   (3.80 → 4.15) because 150 variants per example hyper-fit the known pages.
   On unseen pages the tiny generative model now matches or slightly beats
   BM25 token F1 — the first evidence of composition rather than retrieval.

## 1.5 How to read this book

- If you want the **theory**, read Chapters 2–3 carefully; they assume only
  linear algebra and basic probability.
- If you want to **reproduce or extend the system**, read Chapters 4–8 with
  the source open, then follow the reproduction guide in Chapter 17.
- If you want the **research story**, read Chapters 9–13 and 16.
- If you want to **use the tool**, Chapter 15 is self-contained.

Every number in this book is generated by code in this repository and recorded
in `artifacts/` (evaluation reports, `metrics.jsonl` training logs, run
metadata). Where a claim depends on a run artifact, the chapter names the file.

## 1.6 A note on scale

It is worth being explicit about how small this model is, because the lessons
change with scale. GPT-3 has 175 billion parameters; Llama 3 has 8–405 billion;
even "small" open models are 1–8 billion. TinyShell's shipped model is
**9.6 million parameters — roughly 1/18,000 of Llama-3-8B** — and its training
budget was one CPU-day. At this scale:

- Every architectural choice matters measurably; there is no slack.
- The vocabulary (4,096 vs 32k–128k) is a first-order design decision because
  embeddings dominate the parameter count.
- Generalization behaves differently: the model can memorize its training
  distribution almost perfectly (train loss ≈ 0.13) while remaining helpless
  one distribution shift away (unseen pages, exact match 0.5%).
- The gap between "the loss went down" and "the model works" is enormous, which
  is exactly why the evaluation chapter insists on generation-based metrics.

Small models are not a scaled-down curiosity: they are the regime where
education, experimentation, and honest science are cheapest — and this book
tries to make that regime legible.


---

# Chapter 2 — Language Models from Scratch: The Theory You Need

This chapter builds the mathematical machinery TinyShell uses, from conditional
probability to cross-entropy loss, byte-level BPE, and why a tiny 4,096-token
vocabulary is a feature at this scale. Chapter 3 then covers the transformer
architecture itself. Everything here is standard; the value is in connecting
each abstraction to a concrete decision made in the repository.

## 2.1 Language modeling as conditional probability

A language model assigns probabilities to sequences of tokens. For a sequence
$x_1, x_2, \dots, x_T$, the chain rule factors the joint probability into
next-token predictions:

$$
P(x_1, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \dots, x_{t-1}).
$$

Training a neural language model means fitting a function
$f_\theta(\text{prefix}) \to \mathbb{R}^{|V|}$ (a distribution over the
vocabulary $V$) so that each factor of that product is predicted well.
*Inference* then means sampling or argmaxing from
$f_\theta(\cdot)$ one token at a time, appending each chosen token to the
prefix.

TinyShell is a **conditional** LM: the prefix always begins with a fixed
prompt that contains the user's request, and the distribution we care about is

$$
P(\text{command} \mid \text{request}).
$$

Two details of our setup follow immediately from this framing:

- **Loss masking.** We compute the training loss only on the tokens we want to
  model (the command and its end-of-sequence marker). The system prompt and the
  user request are conditions, not targets. In `dataset.py` their label
  positions are filled with `-100`, PyTorch's `ignore_index` for cross-entropy,
  so they contribute neither loss nor gradient.
- **The EOS token is part of the task.** The model must learn *when the command
  ends*; `<|eos|>` is a supervised target like any other token. Greedy decoding
  stops when the model emits it. An LM that cannot end its output is useless in
  a CLI.

## 2.2 Cross-entropy loss and perplexity

Given logits $z \in \mathbb{R}^{|V|}$ at one position, the model's distribution
is $\text{softmax}(z)$. The training loss is cross-entropy against the true
next token $y$:

$$
\mathcal{L} = -\log \frac{e^{z_y}}{\sum_{v} e^{z_v}}.
$$

Cross-entropy has a useful interpretation: it is the negative log-likelihood the
model assigns to the correct token, averaged over positions. Lower is better,
and the scale is intuitive:

- $\mathcal{L} = \ln |V|$ is the loss of a uniform guess (for
  $|V|=4{,}096$: **8.32 nats** — almost exactly where run 3 started, 8.24 at
  step 10).
- $\mathcal{L} = \log_2 |V|$ is the "choose the right word out of the
  vocabulary" entropy in bits.
- $e^{\mathcal{L}}$ is **perplexity**: the effective number of equally likely
  choices the model is choosing among. Run 3 finished at val loss 0.6565, i.e.
  perplexity ≈ 1.93 — on average fewer than two live options per supervised
  position on commands it has seen.

Two normalization subtleties matter for honest comparisons:

1. **Token-weighted averaging.** Because many positions are masked (`-100`),
   the loss must be averaged over *active* tokens only. `TinyLM.forward` uses
   `F.cross_entropy(..., ignore_index=-100)`, which does exactly this.
2. **Length effects.** A loss computed over a corpus of short commands is not
   comparable to one computed over long documents. All loss numbers in this
   book are over the same packed validation set, so run-to-run comparisons are
   valid; absolute comparisons to other projects' losses are not.

## 2.3 Tokens and byte-level BPE

Neural models consume integers, so text must be segmented into vocabulary
items. The three classic options:

- **Word-level**: every word is a token. Simple, but the vocabulary explodes
  and any unseen word becomes `<unk>` — fatal for a shell task where filenames
  are arbitrary.
- **Character-level**: tiny vocabulary, no unseen items, but sequences get
  long (context cost) and each token carries very little meaning.
- **Subword-level (BPE)**: start from characters and iteratively merge the
  most frequent adjacent pairs. Common words become single tokens; rare words
  decompose into pieces. This is the standard.

TinyShell uses **byte-level BPE** (`tokenizers` library, `models.BPE` with
`ByteLevel` pre-tokenization and decoding):

1. The base alphabet is the **256 byte values**, so *every* string — any
   Unicode, any filename — is representable. There is no `<unk>` and no
   round-trip failure. The tokenizer training code literally verifies
   `decode(encode(text)) == text` on a sample of the corpus and records
   `roundtrip_failures` (it was 0).
2. The `ByteLevel` pre-tokenizer splits text so that whitespace attaches to the
   following word (` al` for " al"), matching GPT-2 conventions.
3. Merges are learned on the rendered training text (`render_training_text` =
   full prompt + command + EOS), so token boundaries align with the exact
   strings the model will see.

**Why 4,096 and not 32,000?** The vocabulary size sets two parameter blocks:
the embedding matrix and the output projection. With tied weights (Section
3.6) that is $|V| \times d$ parameters. For the 2.1M model:
$4{,}096 \times 128 = 524{,}288$ parameters — **25% of the entire model**.
At 32k tokens the embedding alone would be 4M parameters, dwarfing everything
else. The cost of a small vocabulary is longer sequences and more tokens per
rare string; the measured corpus statistics make the trade concrete:

| Statistic (v1 corpus) | Value |
|---|---|
| Median example length | 67 tokens |
| p99 example length | 119 tokens |
| Max example length | 137 tokens (v1) / 229 (XL corpus) |
| Chars per token (sample mean) | ≈ 3.4 |

Everything the model needs fits comfortably in 256 positions. The XL corpus
(rare longer examples) drops the handful of examples exceeding the window
(`skipped_too_long` in the packing stats).

**Special tokens.** Five ids are reserved at the bottom of the vocabulary:
`<|pad|>` (0), `<|eos|>` (1), `<|system|>` (2), `<|user|>` (3),
`<|assistant|>` (4). They are inserted as `special_tokens` during BPE training
so their ids are stable, and they are never produced by merges.

## 2.4 The chat template

Every example is rendered into a fixed three-role template before
tokenization:

```text
<|system|>
You are a Linux shell command assistant. Convert the request into exactly one command. Reply with the command only, no explanation.
<|user|>
{request}
<|assistant|>
{command}<|eos|>
```

The template is *string-identical* at training and inference time
(`text.render_prompt`). This matters more than it looks: a template drift
between training and serving silently degrades the model. Keeping the template
in one function used by training, evaluation, and the CLI removes that failure
mode. The system prompt is fixed — this project has no multi-turn dialogue, no
few-shot examples, and no tool results; the entire interaction is
request-in, command-out.

## 2.5 Optimization in one section

Training minimizes the average masked cross-entropy over packed batches with
**AdamW** (Loshchilov & Hutter, 2019). The pieces worth understanding:

- **AdamW vs SGD+momentum.** Adam keeps per-parameter first and second moment
  estimates, giving each parameter an effectively normalized step. For
  transformers this is not optional; vanilla SGD is unstable or very
  slow to converge at these depths.
- **Decoupled weight decay.** Classic Adam applies L2 through the adaptive
  denominator, which regularizes large-gradient parameters *less* — the
  opposite of intent. AdamW decouples the decay step from the adaptive step:
  `p ← p − lr·λ·p` then the Adam update. TinyShell applies decay **only to
  matrices** (2-D parameters, excluding the tied embedding); norms, biases, and
  embeddings decay 0 (`configure_optimizer` in `model.py`).
- **β = (0.9, 0.95).** The second-moment discount is lower than sklearn's
  default 0.999 because transformer gradients are spiky; 0.95 forgets spikes
  faster. This follows GPT-2/Llama practice.
- **Warmup + cosine decay.** The learning rate ramps linearly from ~0 over
  `warmup_steps` (200 for 2.1M runs, 300 for run 3), then follows
  $$
  \text{lr}(t) = \text{lr}_{\min} + \tfrac{1}{2}(\text{lr}_{\max} -
  \text{lr}_{\min})(1 + \cos(\pi \cdot \text{progress}))
  $$
  from 3e-4 down to 3e-5. Warmup avoids the early-training blow-up when
  Adam's moment estimates are still noisy; the cosine tail anneals into a
  sharp minimum. Run 3's log shows the schedule working: lr 1e-5 at step 10,
  3.30e-5 at step 3430, 3.00e-5 flat at the end.
- **Gradient clipping** at global norm 1.0 (`grad_clip`) bounds any single
  batch's damage. Run 3's logged grad norms sit at 0.8–1.5 for most of
  training — the clip is active at the edge, which is normal and harmless.
- **Precision.** Everything is float32. On a 4-vCPU ARM64 machine, bf16/fp16
  support is patchy and the model is small enough that fp32 matmuls keep the
  cores busy; half precision would save memory the machine has plenty of.

## 2.6 What "learning" looks like at this scale

Run 3's validation trajectory (from `artifacts/runs/tinyshell-xl-10m/metrics.jsonl`)
is worth internalizing, because it is what real training curves look like:

| Step | Val loss | Perplexity |
|---:|---:|---:|
| 250 | 4.522 | 92.0 |
| 500 | 3.396 | 29.8 |
| 1,000 | 2.199 | 9.0 |
| 2,000 | 1.199 | 3.32 |
| 3,000 | 0.788 | 2.20 |
| 3,500 | 0.6565 | 1.93 |

Three observations that recur throughout this book:

1. The curve is **steep early, long flat late**: most of the learning happens
   in the first quarter; the last 500 steps buy ~0.06 nats. Early stopping on
   a noisy plateau would have been wrong here — the end-of-schedule anneal is
   where the final gains came from.
2. Loss ≈ 0.13 on *training* data by the end vs 0.66 on held-out validation:
   the model is memorizing its training distribution to a substantial degree.
   Whether that is "overfitting" depends entirely on which held-out set you
   care about — the central theme of Chapter 13.
3. **Loss is not skill.** Run 1 reached val loss 3.97 with 0% exact match;
   run 2 reached 2.94 with 0% exact match; run 3 reached 0.66 with 52.5%. Loss
   measures next-token fit; only generation measures the task. This is why
   Chapter 9 insists on generation-based evaluation with multiple partial
   credit metrics.

## 2.7 The mathematics of generalization (informally)

The seen/unseen gap that dominates the later chapters deserves a theoretical
frame. A conditional LM trained on pairs $D = \{(r_i, c_i)\}$ minimizes
empirical risk on its training distribution. Its behavior on a new request
distribution is determined by what function class it can express and what the
data taught it to rely on:

- **Interpolation** (new slot values for a known command) is easy: copying an
  entity from the context into a template is a local, learnable operation.
- **Re-derivation with held-out phrasings** (paraphrases) is easy *if* the
  training data shows many phrasings of the same intent — which our 12–16
  prompt styles deliberately provide.
- **Composition from unseen parts** (a command never seen in training) requires
  the model to map a description to a *novel* tool's flag vocabulary. Nothing
  in the training distribution supplies those flags, so the Bayes-optimal
  behavior given the data is to fall back on a prior over command tokens —
  which is exactly what the model does (`sudo ...` for obscure admin pages).

This is why "more variants of the same pages" (run 3's data strategy) improves
interpolation dramatically while *worsening* held-out-page loss: every
additional phrasing sharpens the fit to known pages and makes the prior over
unknown ones relatively less useful. Chapter 13 quantifies this with the
seen/unseen loss gap (0.87 nats for run 2 → 3.49 nats for run 3).


---

# Chapter 3 — Transformer Anatomy: Every Piece TinyShell Uses

TinyShell's model (`src/tinyshell/model.py`) is deliberately small and
readable: 272 lines, PyTorch-only, no external transformer library. This
chapter explains each component — why it exists, what the math is, and how the
implementation reflects it. We use the shipped 9.6M configuration
(`d_model=256`, 10 layers, 8 heads) for formulas and the 2.1M one
(`d_model=128`, 6 layers, 4 heads) where the numbers differ.

## 3.1 The block diagram

```text
input_ids ─► Embedding(4096 × 256) ─► Dropout(0.05)
    │
    ▼  ×10 identical blocks
┌─────────────────────────────────────┐
│ x = x + Attn( RMSNorm(x) )          │   RoPE on q,k; causal + block-diag mask
│ x = x + SwiGLU( RMSNorm(x) )        │   256 → 768 → 256
└─────────────────────────────────────┘
    │
    ▼
RMSNorm ─► Linear(256 → 4096)   [weights tied to the embedding]
    │
    ▼
logits → softmax / cross-entropy
```

Two properties make this a *pre-norm* decoder: normalization happens *inside*
the residual branch (before attention/MLP, not after the residual add), and
each sublayer's output is added back into a persistent residual stream. Pre-norm
gradients flow to the input unattenuated through every layer, which is why
transformers train stably without exotic initialization; the final RMSNorm
before the head compensates for the unnormalized stream.

## 3.2 Scaled dot-product attention

Given hidden states $X \in \mathbb{R}^{T \times d}$, attention projects them to
queries, keys, and values:

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V,
$$

then computes, for each head:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}} + M\right)V .
$$

Implementation notes that matter:

- **One fused QKV matrix.** `Attention.qkv` is a single
  `Linear(d_model, 3*d_model, bias=False)`; the output is chunked into three.
  One GEMM instead of three is meaningfully faster on CPU.
- **`scaled_dot_product_attention`** (PyTorch fused SDPA) handles the softmax
  and mask application in optimized kernels. With `attn_mask` provided, the
  causal pattern is expressed *by the mask* (`is_causal=False`); without one,
  `is_causal=True` uses the built-in triangular path at inference.
- **No attention biases.** Following Llama, QKV and output projections omit
  bias terms. Biases in attention rarely help and cost parameters.
- **The scaling factor** $\sqrt{d_h}$ keeps logit variance independent of head
  dimension so the softmax does not saturate. Here $d_h = 32$ in both model
  sizes (256/8 and 128/4) — a deliberate choice to keep per-head capacity
  equal across the scaling experiment (Chapter 13 relies on this).

**The mask $M$ is the heart of the method** (Section 3.4). During training it
is causal *and* block-diagonal over packed documents; at inference with a
single example it reduces to plain causal masking.

## 3.3 Multi-head attention

Instead of one $d$-dimensional attention, the hidden dimension is split into
$n_{\text{heads}}$ subspaces of $d_h$ each. Each head computes attention
independently; outputs are concatenated and re-projected by
`o_proj` (`Linear(d_model, d_model, bias=False)`).

Why multiple heads? Because a single softmax is a *single* weighted average —
it can only look at one thing at a time in a soft sense. Multiple heads let
different subspaces track different dependencies simultaneously. For this task
the natural specializations are: lexical copying of slot values
("`alice`" from the request into the command), command-family selection,
flag-pattern continuation, and EOS prediction. With 8 heads × 10 layers there
are 80 attention "roles" available; interpretability of small models makes it
plausible to verify such specializations (we did not run head attribution — a
listed future-work item).

## 3.4 The block-diagonal causal mask

This is the single most consequential implementation detail in the project —
the difference between run 1 (0% exact match) and run 2 (mechanically sound).
Recall from Chapter 7 that training examples are **packed**: several complete
prompt+command examples share one 256-token row to avoid padding waste.

Naive packing (run 1) lets token $t$ of example 3 attend to everything before
it — including example 2's *command*. The model then learns to continue the
*previous* example, a pattern that is unavailable at inference (where the row
contains exactly one example). The fix, in `TinyLM.document_mask`:

```python
same_document = doc_ids[:, :, None] == doc_ids[:, None, :]   # (B, T, T)
causal = torch.ones(T, T, dtype=torch.bool).tril()
mask = (same_document & causal).unsqueeze(1)                  # (B, 1, T, T)
```

A position may attend to $t' \le t$ **only if both positions belong to the
same document id**. Padding carries `doc_id = -1` and can never match. The
mask costs one $T \times T$ boolean broadcast per layer batch — unmeasurable
next to the attention itself — and it composes with the fused SDPA path as an
additive boolean mask.

Two companion details complete the design:

- **Per-example positions.** `position_ids` restart at 0 for every packed
  example (stored at packing time in `dataset.py`). RoPE consumes these
  positions (next section), so the model's positional world matches inference
  exactly.
- **Evaluation-side shenanigans are detectable.** Because `doc_ids` is passed
  explicitly, the *same checkpoint* can be evaluated with either mask. That is
  precisely how the run-1 shortcut was proven (Chapter 10): evaluate the old
  model with the new mask, no retraining, and watch validation loss drop 0.56
  nats.

This design is a small special case of a known technique: block-diagonal
attention masks for sequence packing with position reset (Krell et al., 2021,
"Efficient Sequence Packing without Cross-Contamination"), and
document-boundary masking with position shift as used in training multimodal
conditioned LMs (Copet et al., 2023). Our contribution is the *diagnostic use*
— quantifying a learned cross-document shortcut by mask ablation at eval time.

## 3.5 Rotary position embeddings (RoPE)

Transformers need position information. The classic choice is a learned or
sinusoidal position *vector added to the embedding*. RoPE (Su et al., 2021)
instead **rotates** each query/key pair by an angle proportional to position,
per frequency:

$$
\begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix}
= \begin{pmatrix} \cos \theta_i p & -\sin \theta_i p \\ \sin \theta_i p & \cos \theta_i p \end{pmatrix}
\begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix},
\qquad \theta_i = \text{base}^{-2i/d_h},
$$

with $p$ the token position and the same rotation applied to $k$. The dot
product of two rotated vectors depends only on their **relative** offset
$p_q - p_k$, so attention scores become translation-invariant. Advantages:

- relative position information with **zero learned parameters**;
- a mild decay of attention scores with distance for high-frequency pairs;
- exact extension properties (the context could be stretched with
  position-interpolation tricks if ever needed — not required here; positions
  are precomputed for the fixed 256-token context in `RotaryEmbedding` and
  indexed by `position_ids`).

Implementation: `inv_freq = 1/θ^(2i/d_h)`; the cos/sin tables are
`register_buffer`s (saved with the checkpoint but not trained); the split-halves
rotation `_rotate_half` is the standard contiguous implementation.

`θ = 10,000` (the default). Because our per-example positions never exceed ~230
(max example length 229 in the XL corpus), no long-context adaptation is
needed.

## 3.6 RMSNorm and weight tying

**RMSNorm** (Zhang & Sennrich, 2019) normalizes by root-mean-square only:

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\tfrac{1}{d}\sum_i x_i^2 + \epsilon}} \cdot \gamma .
$$

It drops LayerNorm's mean-centering and bias — one fewer reduction, a few
fewer parameters, and empirically equivalent quality in Llama-family models.
Used pre-attention and pre-MLP in every block plus a final one before the
head.

**Weight tying** shares the embedding matrix with the output projection:
`self.lm_head.weight = self.embed.weight` (Press & Wolf, 2017). The output
head must map $d \to |V|$; the embedding maps $|V| \to d$. Tying forces both
to live in the same vector space and saves
$4{,}096 \times 256 = 1{,}048{,}576$ parameters on the shipped model — 11% of
it. At small scale this is not an optimization; it is the difference between
having a few hundred thousand parameters for *everything else* or not.

## 3.7 SwiGLU MLP

Each block's feed-forward network is a gated two-layer MLP:

$$
\text{SwiGLU}(x) = W_{\text{down}}\big(\text{SiLU}(W_{\text{gate}} x) \odot
W_{\text{up}} x\big),
$$

where SiLU$(x) = x \cdot \sigma(x)$ is the swish activation and
$\odot$ is elementwise multiplication (Shazeer, 2020). The gate branch
modulates the up branch per dimension, giving each hidden unit a data-dependent
on/off profile. Per-parameter, gated variants consistently beat plain GELU MLPs
at equal width, which is why Llama uses them.

The intermediate size is 768 (3× `d_model`) on the 9.6M model and 512 (4×) on
the 2.1M one. Llama's ⅔-rounding convention for matching parameter counts is
not applied here — we simply chose the widths, then reported the exact counts
(see the parameter tables below).

## 3.8 Initialization

- All weights: normal(0, 0.02) (`_init_weights`).
- **Residual output projections** (`o_proj`, `down`) get
  normal(0, 0.02/√(2·n_layers)) — the GPT-2 trick. The residual stream grows
  by the sum of ~2·L sublayer outputs; scaling those outputs down by the
  square-root of the count keeps the stream's variance near identity at init,
  which preserves early-training signal depth.

## 3.9 Parameter counts, exactly

**2.1M model** (`configs/tinyshell.yaml`; `d_model=128`, 6 layers, 4 heads,
SwiGLU 512, vocab 4,096, tied embeddings):

| Component | Count |
|---|---:|
| Tied embedding + head | 524,288 |
| Per layer: QKV (128 → 384) | 49,152 |
| Per layer: attention out (128 → 128) | 16,384 |
| Per layer: SwiGLU gate+up+down (128→512 ×2, 512→128) | 196,608 |
| Per layer: 2 × RMSNorm gains | 256 |
| Per layer total | 262,400 |
| 6 layers | 1,574,400 |
| Final RMSNorm | 128 |
| **Total** | **2,098,816** |
| Non-embedding | 1,574,528 |

**9.6M model** (`configs/tinyshell-xl.yaml`; `d_model=256`, 10 layers, 8 heads,
SwiGLU 768):

| Component | Count |
|---|---:|
| Tied embedding + head (4096 × 256) | 1,048,576 |
| Per layer: QKV (256 → 768) | 196,608 |
| Per layer: attention out | 65,536 |
| Per layer: SwiGLU (256→768 ×2, 768→256) | 589,824 |
| Per layer: 2 × RMSNorm | 512 |
| Per layer total | 852,480 |
| 10 layers | 8,524,800 |
| Final RMSNorm | 256 |
| **Total** | **9,573,632** |
| Non-embedding | 8,525,056 |

Both totals match the values logged in each run's `model_summary.json` to the
parameter — counts in this project are computed by `num_parameters`, not
asserted from formulas.

## 3.10 Generation

`TinyLM.generate` is a plain autoregressive loop:

1. Re-run the full model on the last `max_seq_len` tokens (no KV cache — at
   256 tokens context this is fast enough on CPU and keeps the code trivial).
2. Take the last position's logits.
3. `temperature <= 0` → argmax (greedy; used by all evaluation and the CLI
   default). Otherwise scale, optionally top-k filter, softmax, and
   `multinomial` sample.
4. Append; stop on `<|eos|>` (batch-consistent `all()`) or after
   `max_new_tokens` (64 in evaluation).

Greedy decoding is the right choice for a benchmarked system: deterministic,
reproducible, and free of sampling variance across runs. Sampling knobs exist
in the CLI for exploration only.

## 3.11 What the architecture buys — and costs

| Design choice | Benefit in this project |
|---|---|
| Fused QKV, no biases | Fewer parameters, CPU-friendly GEMMs |
| Pre-norm + RMSNorm | Stable training without warmup games; 200–300-step warmup suffices |
| RoPE, zero params | Position budget spent on capacity; per-example positions integrate cleanly with packing |
| Tied embeddings | Saves 1.05M params (11%); standard at small scale |
| SwiGLU | More per-parameter expressivity; standard in modern decoders |
| Head dim 32 fixed across scales | Clean capacity comparison 2.1M → 9.6M |
| Block-diagonal mask | Removes the train/inference mismatch (the central finding of the book) |

Costs and omissions, stated honestly: no KV cache (latency would drop with
one), no flash-attention-style tiling (fused SDPA is close at this size), no
grouped-query attention (irrelevant at 8 heads), no learned positional
embeddings to ablate (RoPE only). Each omission was accepted because the
2.1M experiments showed no instability that they would fix.


---

# Part II — Building the System


---

# Chapter 4 — The Task and the Data: From tldr Markdown to a Training Corpus

Data is the largest and most opinionated part of TinyShell. This chapter walks
the full pipeline in `src/tinyshell/data.py` (1,019 lines): source, parsing,
filtering, slot filling, prompt styles, splitting, and the two dataset
versions (v1 and the XL corpus of run 3). Everything is deterministic and
re-runnable from a pinned source commit.

## 4.1 The source: tldr-pages

[tldr-pages](https://github.com/tldr-pages/tldr) is a community-maintained
collection of simplified man pages. A page (one Markdown file per command)
contains a set of examples:

```markdown
# ls

> List directory contents.

- List files including hidden ones, with human-readable sizes:

`ls -lah`

- List files sorted by modification time, newest first:

`ls -lt`
```

Each example is exactly the pair we need: a one-line description and a command
template with `{{placeholders}}` (`ls {{path/to/directory}}`). The pipeline
performs a **sparse, shallow, blob-filtered clone** of only `pages/common` and
`pages/linux`:

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/tldr-pages/tldr.git data/raw/tldr
git -C data/raw/tldr sparse-checkout set pages/common linux
```

The source commit (`95062b79e30fc3539246399b1dbfcfabbccc79cf`) is recorded in
the dataset manifest, which is what makes the corpus reproducible. tldr is
CC-BY-4.0; attribution is carried in `README.md`, `LICENSE`, both cards, and
the packaged model.

**Corpus snapshot (v1):** 6,803 pages / 30,308 raw examples → **6,621 usable
pages / 28,659 usable base examples** after filtering.

## 4.2 Parsing

Markdown parsing is regex-based and deliberately simple
(`EXAMPLE_RE`, `CODE_RE`): an example is a `- description:` line followed by a
backtick-quoted command line. This is more robust than a Markdown parser for
this corpus and produces no dependencies. Pages whose examples all fail
filtering are dropped.

## 4.3 Filtering: deciding what the model may learn

The filter (`example_is_valid` and friends) removes anything that is not a
single, self-contained Linux command. The v1 rules:

| Rule | Threshold / pattern | Reason |
|---|---|---|
| Description length | 8–160 chars | Descriptions must be informative but bounded |
| Command length | ≤ 120 chars | Fits the context budget with the prompt |
| Multiline commands | reject | Not one-liners |
| Multiple statements | reject `;` | Sequence is out of scope |
| Command substitution | reject `$(...)`, `${...}` | Nested semantics the model can't verify |
| History expansion | reject `!!`, `!$`, leading `!` | Depends on shell state |
| Interactive keybindings / editor commands | reject | Not shell commands |
| Shell keywords / snippets | reject `case`, `for`, `while`, … | Control flow, not commands |
| Comments / malformed placeholders | reject `{{` without `}}`, etc. | Template errors |
| Exact duplicate (desc, command) | reject | Deduplication |

XL (run 3) adds three stricter rejections, removing 139 further examples:

- first token shaped like a **cron schedule** (`0 10 * * * git status`);
- commands **starting with `!`** (history expansion, e.g. `!git ...`);
- **process substitution** (`source <(kubectl completion zsh)`).

The measured filter-reason counts (v1 manifest) show where examples went:
command too long 731, interactive keybinding 359, duplicate 236, shell
substitution/variable 108, multiple statements 96. The point of the filter is
not just safety — it defines the **task boundary**: TinyShell maps one English
sentence to one command, and the data must not contain examples that violate
the contract it can actually learn.

## 4.4 Splitting by page, not by example

Splits are decided by a **deterministic hash of the page title**: 80% train /
10% val / 10% test. All examples of a command land in exactly one split. This
single decision enables the paper's central measurement, because it creates a
hard generalization axis:

- `test_seen`, `test_paraphrase` draw from **train pages** (new slot values
  and/or new prompt styles — interpolation);
- `test_unseen`, `test_unseen_paraphrase` draw from pages **never seen in
  training** (composition).

Determinism matters for a second reason: run 3 reused the same page split, so
eval requests overlap heavily across runs even though the corpus grew
(`variants_per_example` 6 → 150). Directional cross-run comparison stays
honest.

## 4.5 Slot filling

tldr placeholders like `{{path/to/file}}` are filled with values from
hand-written realistic pools:

| Slot category | Example values |
|---|---|
| file | `notes.txt`, `data.csv`, `archive.tar.gz` |
| path | `/home/alice/projects`, `/var/log`, `./build` |
| directory | `src`, `~/.config`, `/etc/nginx` |
| user | `alice`, `bob`, `www-data`, `root` |
| host | `server01.example.com`, `10.0.0.5` |
| package | `nginx`, `python3-pip` |
| port | `8080`, `443` |
| pattern | `*.py`, `TODO` |
| version | `1.2.3` |

The filler (`SlotFiller`) also resolves tldr's alternative syntax:
`{{[-a|--all]}}` becomes either `-a` or `--all` (drawn per variant), and
`{{[clou|cloudinit]}}` picks one alternative. Consequences worth knowing:

- **Every variant of the same example has different concrete values** — this is
  what forces the model to *copy entities from the context* rather than
  memorize a fixed string.
- **The pools are small.** `file` appears 38,833 times, `name` 13,611,
  `path` 7,998 in v1's labels. A model can partially memorize the pool; the
  slot-robustness number should be read with that in mind (see the bias
  discussion in `DATA_CARD.md` and Chapter 14).

## 4.6 Prompt styles: 16 phrasings, 4 held out

Each example's description is rendered into a request using a style template
(`REQUEST_STYLES`, `build_request`):

| Index | Template | Used by |
|---:|---|---|
| 0 | `How do I {desc}?` | train |
| 1 | `I want to {desc}.` | train |
| 2 | `Can you give me a command to {desc}?` | train |
| 3 | `Please show me how to {desc}.` | train |
| 4 | `Write a shell command to {desc}.` | train |
| 5 | `Which Linux command should I use to {desc}?` | train |
| 6 | `I need to {desc}.` | train |
| 7 | `Command to {desc}:` | train |
| 8 | `Give me a one-liner for this: {desc}.` | **val only** |
| 9 | `What's the best way to {desc}?` | **test_seen only** |
| 10 | `Help me {desc}.` | **paraphrase splits only** |
| 11 | `Show me the command for this: {desc}.` | **paraphrase splits only** |
| 12–15 | run-3 additions (`What would you run to…`, `Do you know how to…`, `Type a shell command that will…`, `I'm looking for a command to…`) | train |

Slots mentioned generically in the description ("the file") are inlined with
their concrete value, and a **context sentence** appends the rest:
*"Assume the file is `/tmp/notes.txt`."* Roughly 25% of the time the context
sentence is prepended instead, so the model cannot rely on word order alone.
Negative descriptions ("don't overwrite the file") get a neutral
`Command for this task: …` framing to avoid ungrammatical templates like
"How do I don't…?".

Held-out styles are a *second* generalization axis: `val` selects checkpoints
on a phrasing the model never saw, `test_seen` adds new slot values on top,
and the paraphrase splits test both unseen phrasings and (for the unseen-page
splits) unseen commands.

## 4.7 Two dataset versions

| | v1 (runs 1–2, core) | XL (run 3) |
|---|---|---|
| `variants_per_example` | 6 | **150** |
| Train styles | 8 | 12 |
| Train requests | 135,860 | **1,303,083** |
| Training tokens (packed, real) | 9,164,892 / epoch | **99,992,950** total |
| Chunks (256 tokens) | 41,743 | **468,212** |
| Supervised label tokens | ≈1.49M positions | **17,411,259** |
| Max example length | 137 tokens | 229 tokens |
| Filter | v1 rules | v1 + 3 stricter rules (−139 examples) |
| Eval splits | 6 × 1,500 | same, regenerated, same logic |
| Page split | identical | identical |

The XL design has one subtlety worth understanding: with 150 variants per
example but only 8→12 distinct phrasings, **slot-free examples saturate at 12
unique strings** (each repeated many times) while **slot-bearing examples
scale near-linearly** (every variant draws a fresh slot-fill seed). The corpus
therefore spends its new volume mostly where variety is possible. Every
variant cycles the train-style pool deterministically (`TRAIN_STYLES[k %
len]`), and a requirement-level dedup pass removes identical (request,
command) pairs.

## 4.8 Corpus statistics and risk labels

v1 statistics recorded in the manifest and data card:

- **Command length:** mean 30.0 chars; p50 26; p90 54; p99 82.
- **Request length:** mean 94.9 chars; p50 90; p90 151; p99 214.
- **Risk labels** (heuristic pattern classes, not ground truth):
  129,177 safe / 14,822 caution / 861 destructive across all splits.
- **`bash -n` validity** on a 2,000-command sample: **99.85%** — the residual
  invalids come from templates whose placeholder substitution breaks syntax.

Risk labeling reuses the same pattern lists the CLI uses for warnings
(Chapter 15): `rm -rf` → destructive; `sudo`, `rm`, redirection → caution. It
is metadata for analysis, not a supervised target — no split trains on risk.

## 4.9 The core subset

`scripts/build_core.py` derives a focused subset from the same splits and
pools: keep pages whose **title** starts with a command in a curated list of
~100 common Linux/DevOps tools (`ls`, `git`, `docker`, `kubectl`, `systemctl`,
`aws`, `cargo`, `ffmpeg`, …). Result: 655 pages, 17,317 train requests, 5,186
chunks, 1,143,753 real tokens per epoch, **111 distinct first command tokens**
(91 from the curated list; the rest are shell snippets or aliases that the
title-based filter let through — e.g. a crontab schedule line). The core
unseen splits alias the seen ones (every retained page is in the train set),
so core evaluation measures slot/style robustness only. This subset exists to
answer one question (Chapter 12): *is the 2.1M model's failure label-space
limited or capacity limited?*

## 4.10 Known biases, stated plainly

From `DATA_CARD.md`, each of which bounds what conclusions the data supports:

1. **Synthetic requests.** Real users do not write 16 templates. Robustness
   across our styles overstates robustness across real phrasings.
2. **Fixed slot pools.** Common values (`notes.txt`, `alice`) recur; the model
   can memorize values instead of learning to copy arbitrary entities.
3. **tldr coverage bias.** Only `common` + `linux`; contributor taste
   determines which tools exist at all.
4. **Heuristic risk labels.** Regexes, not annotation.
5. **No execution.** `bash -n` checks syntax; semantics are never verified.
6. **Snapshot.** One tldr commit; commands and flags age.
7. **Names.** Synthetic users (`alice`, `bob`, `www-data`) are conventional
   examples, not claims about real people.

## 4.11 The manifest

Every build writes `data/processed*/dataset_manifest.json` with counts, filter
reasons, slot-category frequencies, risk distribution, and the source commit.
The dataset is *documented like a model*: version, statistics, known biases,
and a maintenance section warning that changing any rule invalidates
cross-run comparisons. This discipline — data cards, manifests, pinned
sources — is a large part of what makes the experiments in Part III
interpretable months later.


---

# Chapter 5 — Tokenization and Packing: Preparing the Fixed-Length Training Corpus

Between the JSONL splits and the training loop sits a stage that looks like
bookkeeping but decides both the *efficiency* and the *correctness* of
training: tokenizing every example and packing the results into fixed-length
chunks. This chapter follows `src/tinyshell/dataset.py` line by line.

## 5.1 Why pack at all?

Training batches must be rectangular: every row the same length. Examples
here are 20–230 tokens; padding every example to 256 independently would waste
most of the compute on padding. The classic fix is **packing** — concatenating
examples until the 256-token chunk is full. The classic *bug* is what run 1
did: attention flows across example boundaries inside a chunk (Chapter 10).
The TinyShell design keeps the efficiency and removes the bug:

1. pack **complete examples only** (never split one across chunks);
2. record `doc_ids` (which example each token belongs to; `-1` for padding);
3. record per-example `position_ids` (position within the example, from 0);
4. let the model build a **block-diagonal causal mask** from (2) and use (3)
   for RoPE.

## 5.2 Encoding one example

`encode_example` renders the chat template (Chapter 2), tokenizes the prompt
and the completion separately, and builds labels:

```text
prompt_ids  = encode(render_prompt(request))          # condition
comp_ids    = encode(command.strip()) + [eos_id]      # target
input_ids   = prompt_ids + comp_ids
labels      = [-100] * len(prompt_ids) + comp_ids
```

`-100` is `F.cross_entropy`'s `ignore_index`: masked positions neither
contribute loss nor receive gradient. About **14%** of packed positions are
supervised (17.41M of ~120M positions in the XL corpus, including padding) —
the loss is concentrated exactly on what the model must produce.

## 5.3 The packing loop

`pack_rows` is a first-fit sequential packer:

- Keep a current row. For each example: if adding it would exceed 256 tokens,
  flush the row (pad the tail), start a new one.
- **Skip overlong examples** (`len(ids) > seq_len`) and count them
  (`skipped_too_long`) rather than truncating — truncation would put partial
  commands in the data.
- Padding is filled with `pad_id` tokens, `-100` labels, `doc_id = -1`,
  `position 0`. Because `doc_id = -1` never matches any other id, padding can
  never be attended to, even by other padding.

The measured costs of complete-example packing: the v1 corpus grew from
35,800 (naive split-anywhere) to **41,743 chunks** (+16.6%) because chunk tails
are padded; padding is ~14% of positions. Run 3's 468,212 chunks carry
99,992,950 real tokens in 468,212 × 256 = 119.9M positions (**83.4%
utilization**). This is the explicit trade: 14% wasted positions in exchange
for document-aligned attention — and Chapter 10 shows the 14% is cheap at a
factor of ~1,000 relative to the shortcut it prevents.

## 5.4 The arrays on disk

Each split becomes four `numpy` arrays, memory-mapped at train time:

| File | dtype | Shape (train) | Content |
|---|---|---|---|
| `train_ids.npy` | uint16 | (468212, 256) | token ids (4,096 < 65,536 → uint16 saves 4×) |
| `train_labels.npy` | int32 | (468212, 256) | targets, −100-masked |
| `train_doc_ids.npy` | int16 | (468212, 256) | per-token document id |
| `train_position_ids.npy` | int16 | (468212, 256) | per-token within-example position |

`PackedDataset.__getitem__` casts each row to int64 tensors on the fly. The
mmap means the 1.9 GB XL corpus never has to fit in RAM and the OS page cache
does the job of a data loader; with `num_workers=0` there is no worker
process copying pages around — the right choice for a 4-vCPU box where
fork/copy overhead exceeds the trivial indexing cost.

A manifest (`packed/manifest.json`) records per-split stats: examples, real
tokens, padding tokens, chunks, active label tokens, max example length.
`PackedDataset` also tolerates older packing formats (missing doc/position
arrays fall back to one-document-per-chunk), which is how run-1 checkpoints
remained loadable for the ablation in Chapter 10.

## 5.5 Invariants worth testing

`tests/test_dataset.py` pins the properties that the training story depends
on:

- no example is ever split across chunks;
- `doc_ids` are contiguous per example, restart-safe across examples, and −1
  exactly on padding;
- `position_ids` restart at 0 at every document boundary;
- labels are −100 exactly on prompt and padding positions;
- decoding the ids of any example reproduces the source text (byte-level
  round trip through the tokenizer);
- every split's chunk count and token count match the manifest.

When packing behavior is the difference between 0% and working models, these
invariants are not pedantry; they are the safety net that makes later runs
comparable to earlier ones.

## 5.6 Tokenizer training, in detail

`tinyshell-tokenizer` (Chapter 4's corpus in, `artifacts/tokenizer/` out):

1. Writes `tokenizer_corpus.txt`: every train example as
   `render_training_text` (prompt + command + EOS). BPE therefore learns
   merges on *exactly the strings the model will see*, including the fixed
   system prompt (which collapses to a handful of tokens).
2. Trains `models.BPE` with `vocab_size=4096`, `min_frequency=2`,
   `initial_alphabet` = all 256 bytes, and the 5 special tokens reserved
   first (ids 0–4).
3. Verifies every special token exists and byte-level round-trips on a 5,000
   example sample; writes `stats.json` (length percentiles, chars/token,
   round-trip failures = 0) and a human-readable `vocab_preview.txt`.

The tokenizer is trained **once** and reused by every run (v1, core, XL),
with byte fallback guaranteeing new slot values in the XL corpus remain
representable. Changing the tokenizer would change every downstream number;
the pinned `tokenizer.json` inside each run directory is the record of which
one was actually used.

## 5.7 A worked example

Take the page snippet from Chapter 4 with description *"List files including
hidden ones, with human-readable sizes"*, template `ls -lah`, style 0, no
slots:

```text
request:  How do I list files including hidden ones, with human-readable sizes?
```

Rendered and tokenized (illustrative ids):

```text
[2, 351, 512, ...  ]   ◄── system + user prompt      labels: -100 ...
[... 4, 97, 15, 90, 330, 1]  ◄── "ls -lah" + <|eos|>  labels: kept
```

Packed with a second small example in the same chunk:

```text
chunk (256):  [ex1 prompt][ex1 cmd+eos][ex2 prompt][ex2 cmd+eos][pad…]
doc_ids    :  [ 0 0 0 0 0 0 0 0 0 | 1 1 1 1 1 1 1 1 1 | -1 -1 …]
position_ids: [ 0 1 2 3 …      | 0 1 2 3 …            | 0  0 …]
labels     :  [-100 … -100, cmd, 1 | -100 …, cmd, 1 | -100 …]
```

The attention mask lets example 1's tokens see only example 1; example 2's
tokens see only example 2; padding sees nothing. RoPE positions restart per
example. At inference, the row *is* one example — identical geometry. That
identity is the whole point.


---

# Chapter 6 — The Training Loop: Turning the Corpus into a Model

`src/tinyshell/train.py` (455 lines) is a from-scratch training loop. Nothing
in it is exotic; everything in it matters. This chapter explains each moving
piece as implemented, using run 3's 27.7-hour run as the running example.

## 6.1 What the Trainer holds

Constructed from a `Config` and a run directory, the `Trainer` owns:

- the tokenizer (loaded from `artifacts/tokenizer/`) and its `eos_id`;
- the `TinyLM` model (built from `cfg.model`, fp32, moved to `cpu`);
- the AdamW optimizer from `model.configure_optimizer` (decay on matrices
  only);
- three packed datasets: `train`, `val`, `val_unseen` (memory-mapped arrays
  from Chapter 5);
- a `LambdaLR` implementing warmup + cosine decay;
- run bookkeeping: `run_dir`, `checkpoints/`, `train.log`,
  `metrics.jsonl`, `samples.jsonl`, config/env snapshots.

Step arithmetic: `micro_per_epoch = len(train_ds) // batch_size`; with
`grad_accum_steps=1`, `total_steps = micro_per_epoch × epochs`. Run 3:
468,212 chunks / batch 128 = 3,657 steps for the single pass — exactly the
final logged step count. The 2.1M runs: 41,743 / 64 = 652 steps per epoch ×
4 = 2,608 steps.

## 6.2 The LR schedule

`_lr_lambda` computes the multiplier directly:

```python
warmup = max(1, min(cfg.warmup_steps, total_steps // 5))
if step < warmup:
    return step / warmup                       # linear ramp
ratio = min_lr / lr
progress = (step - warmup) / (total_steps - warmup)
return ratio + (1 - ratio) * 0.5 * (1 + cos(pi * progress))   # cosine
```

Two defensive details: warmup is capped at 20% of training (a smoke run with
200 configured warmup steps and 30 total steps would otherwise never leave
warmup), and progress is clamped to [0, 1] so any step overshoot saturates at
`min_lr`. Run 3 used warmup 300 of 3,657 steps and ended exactly at
lr = 3.0e-5 — visible in the final log lines.

## 6.3 The step loop

Per optimizer step:

1. Pull a batch of `(input_ids, labels, doc_ids, position_ids)` rows from the
   DataLoader (`shuffle=True` for train).
2. Shift: inputs are `x[:, :-1]`, targets `y[:, 1:]` — next-token prediction
   with no separate "target" corpus. (Doc ids and positions are sliced
   identically so masks/positions stay aligned with the shifted stream.)
3. `logits, loss = model(x, y, doc_ids=doc_ids, position_ids=position_ids)`
   — the block-diagonal mask is built inside forward from `doc_ids`.
4. `loss.backward()` → `clip_grad_norm_(model.parameters(), grad_clip)` →
   `optimizer.step()` → `scheduler.step()` → `optimizer.zero_grad()`.

Run 3's step is 128 × 256 = **32,768 positions**, ~27,300 of them real tokens;
the loop held a steady **~1,190 tokens/s** on 4 threads for 27.7 hours
(throughput sagged to ~1,080 tok/s during the step-3500 eval window, then
recovered — the log shows it).

## 6.4 Periodic work: eval, samples, checkpoints

Every `eval_interval` steps (250 for run 3) the loop:

- **Evaluates loss** on the packed `val` split (`evaluate_loss`): forward-only
  batches, loss weighted by each batch's active token count, aggregated
  globally — the same masked, token-weighted cross-entropy as training, so
  numbers are comparable across steps and runs.
- **Evaluates `val_unseen` loss** (logged at best-checkpoint time and at the
  end; this is the number that *rose* from 3.80 to 4.15 between run 2 and
  run 3 — the headline generalization datum).
- **Generates samples**: the first 5 rows of `samples.jsonl` go through the
  full inference path (prompt → generate → clean → risk check) and are
  appended to `samples.jsonl`. Reading these every few hundred steps was the
  fastest way to notice run 1's shortcut (`sudo aa-cli --list`) long before
  the formal evaluation.
- **Logs and checkpoints**: `best.pt` whenever val improves; `last.pt` every
  `save_interval` (500); `final.pt` at the end.

## 6.5 Checkpoints: what is inside, and why it matters

`save_checkpoint` writes one torch file containing:

```python
{
  "model":      model.state_dict(),
  "optimizer":  optimizer.state_dict(),
  "scheduler":  scheduler.state_dict(),
  "step": ..., "epoch": ..., "batch_in_epoch": ...,
  "best_val": ..., "config": cfg.to_dict(),
}
```

This is a **resumable** checkpoint: `load_checkpoint` restores weights, Adam
moments, the schedule position, and the progress counters, so a killed run
continues seamlessly (run 3 was resumed at least once mid-training; the log's
elapsed clock continues across the gap). It is also *heavy*: optimizer state
doubles the parameter count (two fp32 moment tensors), which is why run 3's
`best.pt` is 115 MB for a 38 MB model. (How that was handled for distribution
is in Chapter 17.)

`best.pt` selection uses **`val` — held-out prompt style 8 on seen pages** —
not a random val sample. That choice encodes an opinion: the checkpoint we
want is the one that generalizes across phrasings of known commands. For run
3 that was step 3,500 (val 0.6565), just before the final anneal bottomed out
— the final model (val 0.6571) is statistically identical, and `best` remains
the shipped checkpoint.

## 6.6 Run metadata: the reproducibility record

At trainer construction (`_write_run_metadata`), the run directory receives:

| File | Content |
|---|---|
| `config.yaml` | the full, resolved config (never overwritten on resume; a `config_resume.yaml` records changes) |
| `env.json` | Python 3.12.3, torch 2.14.1+cpu, numpy 2.5.4, aarch64, 4 threads, git commit, dirty flag, UTC timestamp |
| `model_summary.json` | 9,573,632 total / 8,525,056 non-embedding parameters + the model config |
| `git_commit.txt` | `81bdf02…` — the code state the run was launched from |
| `tokenizer/` | a copy of the exact tokenizer used |
| `README.md` | parameter/chunk/step summary |

With the config, the env, the commit, and the tokenizer pinned per run, every
result in this book can be tied to the exact code and data that produced it.
(One caveat recorded in `docs/EXPERIMENTS.md`: run 3 launched at commit
`81bdf02`, which was later rewritten to `2b88264` during a git-hygiene fix
that removed accidentally-committed large files; code content was unchanged.)

## 6.7 First-token loss weighting (the diagnostic knob)

`TinyLM.forward(first_token_weight=...)` implements a targeted loss reweight:
the *first command token* of every packed example gets its per-token
cross-entropy multiplied by the weight. It is off (1.0) everywhere except the
core experiment (5.0). The motivation: with 3,651 possible first tokens,
uniform token weighting under-represents the single decision that matters
most — which tool to call. The implementation identifies "first command
token" as an active-label position whose predecessor is inactive
(`active & ~previous_active`), which correctly handles multiple examples per
chunk (Chapter 3's weighted-loss code) — and, as `docs/EXPERIMENTS.md`
records honestly, the shipped core run predates this fix and tagged only the
first active token *per chunk* (~1/3 of examples), so the core result is a
conservative estimate of the weighting's effect.

## 6.8 What the logs say about run 3

A compact narrative from `train.log` / `metrics.jsonl`:

- **Steps 1–300 (warmup):** loss 8.24 → 7.6 — the uniform-guess regime
  (ln 4096 = 8.32) collapsing as the model learns unconditional command-token
  frequencies.
- **Steps 300–2,000:** the steep descent — 7.6 → 1.2 val (perplexity 2,000× →
  3.3×) — the model learns command families, flag syntax, and slot copying.
- **Steps 2,000–3,500:** the long grind, 1.20 → 0.656 val (ppl 3.3 → 1.93);
  train loss reaches ~0.13, i.e. the model can near-perfectly reproduce
  training commands — memorization territory, as Chapter 13 discusses.
- **Grad norms** 0.8–1.5 throughout; token/s steady at ~1,190; no spikes, no
  instabilities, no restarts (beyond resume).

The full six-split generation evaluation of the resulting checkpoint is
Chapter 13's subject.


---

# Part III — The Experiments


---

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


---

# Chapter 8 — Run 1: The Packing Bug That Taught the Wrong Task

> Run ID: `tinyshell-2m-run1` · 2,098,816 params · 4 vCPU ARM64 · ~3 h
> Artifacts: `artifacts/eval/tinyshell-2m-run1/`, `artifacts/runs/tinyshell-2m-run1/`

Run 1 is the most instructive artifact in the project: a model that trained
cleanly, reached a plausible validation loss, and was **completely useless** —
0% exact match — because of one design decision made before training started.

## 8.1 What run 1 did

The first packing implementation followed the standard LM recipe: concatenate
all prompt+command examples into one continuous token stream (separated by
EOS), slice into 256-token chunks, train with ordinary causal attention and
one continuous position sequence per chunk. Efficient, simple, standard — and
wrong for this task.

## 8.2 Why it is wrong

Under naive packing, position *t* in example *N* attends to everything before
it in the chunk — including example *N−1*'s **command**. Three consequences:

1. **A shortcut exists in the training objective.** `P(command_t | request_t,
   command_{t-1})` is much easier to fit than `P(command_t | request_t)`: the
   previous command is a *hint* about command family, style, and length.
   Gradient descent finds shortcuts; that is what it does.
2. **The shortcut vanishes at inference.** A serving request contains exactly
   one example starting at position 0. There is no previous command to copy.
   The model's learned conditional is undefined in this regime, so it falls
   back to marginal habits — high-frequency prefixes.
3. **Position ids also ran continuously** across the chunk while inference
   restarts at 0 — a second, milder mismatch stacked on the first.

## 8.3 The numbers

Run 1 trained for 4 epochs (2,608 steps) in ~3 hours. Validation loss:
**3.9749 best / 3.9771 final**; unseen-page val loss **4.6025**. These look
"fine" — run 2's honest numbers (2.94) are only ~1 nat better, and 1 nat in a
loss curve does not *look* like the difference between broken and working.
Generation told the real story (full six-split eval on 1,500 examples each):

| Split | Exact | Command | Token F1 | Valid bash |
|---|---:|---:|---:|---:|
| `val` | 0.000 | 0.065 | 0.108 | 0.995 |
| `val_unseen` | 0.001 | 0.078 | 0.126 | 0.998 |
| `test_seen` | 0.000 | 0.084 | 0.117 | 0.992 |
| `test_paraphrase` | 0.000 | 0.055 | 0.128 | 0.995 |
| `test_unseen` | 0.000 | 0.067 | 0.115 | 0.997 |
| `test_unseen_paraphrase` | 0.000 | 0.047 | 0.113 | 0.996 |

The model produced *syntactically perfect* commands (99.2–99.8% valid bash)
that were almost always the wrong ones. **72% of all predictions began with
`sudo`** versus 6% of references. The canonical failure:

```text
request:    list all files including hidden ones, with human-readable sizes
prediction: sudo aa-cli --list
```

(`aa-cli` is an obscure tldr page — i.e. the model was pattern-matching its
way around the command vocabulary it had memorized, prefixed by its favorite
high-frequency token.)

## 8.4 The diagnostics: proving the shortcut

Two independent probes, both cheap, both documented in `docs/EXPERIMENTS.md`:

**1. Prefix test (behavioral).** Prepend a fake previous example to the
prompt and regenerate:

```text
prompt with prior command `ls -la`  →  prediction copies `ls -la`
prompt with no prior example        →  prediction: sudo aa-cli --list
```

The output flipping based on *irrelevant* context is direct behavioral
evidence that the model's conditional includes the previous example.

**2. Mask ablation (mechanistic, no retraining).** Evaluate the *same
run-1 checkpoint* on the packed validation set twice — once with the ordinary
causal mask, once with the block-diagonal mask that blocks cross-example
attention:

| Attention at eval time | Validation loss |
|---|---:|
| Causal over the whole packed chunk | 3.866 |
| Causal + block-diagonal by example | **3.306** |

If the model were using preceding examples *legitimately* (e.g. as
distributional context), blocking that access would *hurt*. Instead the loss
**improved by 0.56 nats**: the model had learned a harmful copying habit, and
the packed data contained enough signal to confirm it quantitatively. This
ablation is only possible because `doc_ids` is a runtime input to the model
(Chapter 3) — design for diagnosability, not just for training.

## 8.5 What run 1 teaches

1. **Validation loss can hide a shortcut.** A model optimizing
   `P(command | request, previous command)` reaches a *lower* training loss
   than the true task requires, and nothing in the loss curve tells you the
   conditioning you trained is not the conditioning you will serve.
2. **Generation tests are non-negotiable.** The periodic `samples.jsonl`
   generations (`sudo aa-cli --list` at every eval window) were the first
   alarm. Loss-only monitoring would have shipped the model.
3. **Distribution shift can be *internal*.** The train/inference mismatch
   here is not about user demographics or drifting data — it is a mismatch
   the pipeline itself created. Self-inflicted shifts are the commonest kind
   and the least monitored.
4. **Sanity metrics floor out before they help.** Valid bash was 99.5% in
   run 1. A fluent-looking syntax distribution is achievable while learning
   none of the task.
5. **The fix must change the objective, not the model.** Nothing about
   capacity, depth, width, or the optimizer was wrong. The same architecture
   with corrected packing/attention/positions is run 2.


---

# Chapter 9 — Run 2: Document-Aligned Packing and What a Fix Buys

> Run ID: `tinyshell-2m-run2-docmask` · 2,098,816 params · 4 epochs · 2,608
> steps · 3 h 16 m · best val **2.9377**
> Artifacts: `artifacts/runs/tinyshell-2m-run2-docmask/`, `artifacts/eval/tinyshell-2m-run2-docmask/`

Run 2 changes **one thing**: how examples are packed and attended. Same
architecture, same tokenizer, same data, same optimizer, same seed. This is
the cleanest experiment in the project — a controlled ablation at training
time.

## 9.1 The fix, precisely

Four coordinated changes (Chapters 5 and 3 describe the implementation):

1. **Pack complete examples only.** A chunk contains one or more whole
   examples; the tail is padded. No example is ever split.
2. **Store `doc_ids` and `position_ids`** per token; padding gets `doc_id =
   −1`; positions restart at 0 for every example.
3. **Block-diagonal causal mask.** A token attends only to earlier tokens of
   its own example. At inference (one example per row) this *is* the ordinary
   causal mask — the training and serving conditionals become identical.
4. **Per-example RoPE.** Position ids fed to RoPE are within-example
   positions.

Cost: packed corpus grows from 35,800 to 41,743 chunks (+16.6% of positions
as padding); measured mask overhead at this scale: none observable.

## 9.2 Loss: the fix measured directly

| Metric | Run 1 (naive) | Run 2 (doc-aligned) | Δ |
|---|---:|---:|---:|
| Best validation loss | 3.9749 | **2.9377** | −1.04 |
| Final validation loss | 3.9771 | **2.9214** | −1.06 |
| Final `val_unseen` loss | 4.6025 | **3.7993** | −0.80 |
| Training chunks | 35,800 | 41,743 | +16.6% |

Over a nat of validation loss from removing a shortcut — with the *same*
number of gradient steps. Note also that the unseen-page loss improves too:
removing the copying habit helps everywhere, not just on seen pages.

## 9.3 Generation: better, but not good

Full six-split evaluation, greedy, 1,500 examples per split:

| Split | Exact | Command | Token F1 | Valid bash |
|---|---:|---:|---:|---:|
| `val` | 0.001 | 0.073 | 0.174 | 0.985 |
| `val_unseen` | 0.001 | 0.080 | 0.174 | 0.991 |
| `test_seen` | 0.000 | 0.081 | 0.173 | 0.987 |
| `test_paraphrase` | 0.002 | 0.089 | 0.187 | 0.995 |
| `test_unseen` | 0.000 | 0.069 | 0.170 | 0.995 |
| `test_unseen_paraphrase` | 0.000 | 0.062 | 0.172 | 0.991 |

Against run 1 (command match 0.047–0.084, token F1 0.108–0.128), run 2
improves on every split — argument copying roughly +60% (F1 ≈ 0.17 vs 0.11).
Exact match is still **zero**. The model now reliably produces
well-formed commands in a plausible family; it just chooses the wrong family
far more often than not.

**The frequency prior, attenuated but alive.** On `test_seen`, 42% of run-2
predictions (626/1,500) begin with `sudo` versus 6.8% of references (run 1:
72%, then 54% on test_seen). The shortcut that fed the prior is gone; the
*data imbalance* that motivates it is not. `sudo` appears in the corpus as
the universal wrapper for admin tasks, and a capacity-starved model defaults
to it exactly where its command knowledge is thinnest.

**The signature failure.** Structurally correct, semantically wrong:

```text
request:    stream pod logs. Assume the pod name is `alice`.
reference:  kubectl logs --follow alice
prediction: kubectl set --name alice --user
```

Right family (`kubectl`), right entity (`alice` copied from context), valid
syntax, wrong subcommand and flags. The model has learned *forms*: family
selection, entity copying, flag shapes. It has not learned *meaning*: which
subcommand expresses "stream logs".

## 9.4 The honest conclusion of run 2

Fixing packing removed a bug; it did not buy the task. A 2.1M-parameter model
trained on 9.2M real tokens cannot cover **3,651 distinct first command
tokens**. Two roads lead onward, and the project took both:

1. **Narrow the label space** → the core experiment (Chapter 10): if family
   selection is the bottleneck, shrink the label space and watch family
   selection become solvable.
2. **Grow capacity and data** → run 3 (Chapter 11): 4.6× parameters, 10.9×
   tokens, stricter filters, same page splits.

There is also a quiet methodological lesson: run 2's marginal gains over
run 1 (command match +0.0–0.005 on most splits) look tiny in a table, but
the *loss* gap is over a nat and the behavioral gap is "wrong conditional"
vs "right conditional." Headline generation metrics alone would have
undervalued the fix; the loss comparison plus the ablation made its value
unambiguous.


---

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


---

# Chapter 11 — Run 3: Scaling to 9.6M Parameters and 100M Tokens

> Run ID: `tinyshell-xl-10m` · 9,573,632 params · 1 pass over 99,992,950
> packed tokens · 468,212 chunks · 3,657 steps · 27.7 h (99,690 s) ·
> ~1,190 tokens/s · best val **0.6565** (step 3,500)
> Config: `configs/tinyshell-xl.yaml` · Artifacts:
> `artifacts/runs/tinyshell-xl-10m/`, `artifacts/eval/tinyshell-xl-10m/`

Run 3 is the shipped model and the project's main scaling experiment. It
changes capacity, data volume, and data quality *simultaneously* — a choice
made deliberately: after run 2 and the core experiment, every diagnostic
pointed at "more of the same proven design," so the interesting question was
not which knob matters (that is what ablations are for) but how far the
proven design scales.

## 11.1 What changed

| Axis | Run 2 | Run 3 |
|---|---|---|
| `d_model` × layers × heads | 128 × 6 × 4 | **256 × 10 × 8** |
| Head dim (per head) | 32 | 32 (deliberately unchanged) |
| SwiGLU intermediate | 512 | 768 |
| Parameters | 2,098,816 | **9,573,632** (4.6×) |
| Dropout | 0.05 | 0.05 (more data → same light regularization) |
| Variants per example | 6 | **150** |
| Train styles | 8 | 12 (adds indices 12–15; eval styles 8–11 untouched) |
| Corpus filter | v1 rules | + cron-schedule first token, leading `!`, process substitution (−139 examples) |
| Training tokens | 9.16M × 4 epochs | **99.99M × 1 pass** |
| Requests | 135,860 | **1,303,083** |
| Chunks | 41,743 | **468,212** |
| Batch | 64 × 256 | 128 × 256 (32,768 positions/step) |
| Warmup / steps | 200 / 2,608 | 300 / 3,657 |

Unchanged: tokenizer (vocab 4,096; byte fallback covers new slot values),
page-level splits, document-aligned packing, `first_token_weight` = 1.0,
AdamW settings, cosine schedule shape, seed 42, fp32 on 4 vCPU.

Not directly comparable to run 2 as a whole (the corpus changed), but the
page split is identical, so eval requests overlap heavily and directional
comparisons are meaningful — a caveat repeated in every results table.

## 11.2 Training as it actually happened

From `train.log` and `metrics.jsonl`:

- **Throughput:** steady ~1,190 tokens/s (32,768 positions/step), 4 threads,
  no restarts beyond one planned resume; 27 h 41 m wall clock.
- **Loss trajectory (val on held-out style 8, seen pages):**

| Step | Val loss | PPL |
|---:|---:|---:|
| 250 | 4.522 | 92.0 |
| 500 | 3.396 | 29.8 |
| 1,000 | 2.199 | 9.0 |
| 2,000 | 1.199 | 3.32 |
| 3,000 | 0.788 | 2.20 |
| **3,500 (best)** | **0.6565** | **1.93** |

- **Final:** val 0.6571 at step 3,657; train loss ~0.13–0.15; grad norms
  0.8–1.5 throughout; lr annealed to 3.00e-5 exactly on schedule.
- **Best checkpoint:** step 3,500 — the cosine tail's last big eval gain.
  `final.pt` is statistically identical on val (0.6571).

**The number that matters most: `val_unseen` loss 4.1451 — *worse* than run
2's 3.7993.** The seen/unseen loss gap widened from **0.87 nats (run 2)** to
**3.49 nats (run 3)**. Mechanism: 150 variants per page means each *page*
appears hundreds of times with fresh phrasings and slot fills; the model
hyper-fits its known command families while the marginal information about
*unseen* pages goes down. More data made interpolation better and
extrapolation *relatively worse*. This single number is why the eval tables
below must be read split-by-split, never averaged.

## 11.3 Generation evaluation

Checkpoint `best`, greedy decoding, 1,500 per split
(`artifacts/eval/tinyshell-xl-10m/`):

| Split | N | Exact | Command | Token F1 | Valid bash | p50 |
|---|---:|---:|---:|---:|---:|---:|
| `val` | 1499 | 0.460 | 0.618 | 0.604 | 0.997 | 290 ms |
| `val_unseen` | 1500 | 0.011 | 0.138 | 0.315 | 0.995 | 227 ms |
| `test_seen` | 1500 | **0.525** | **0.692** | 0.644 | 0.997 | 245 ms |
| `test_paraphrase` | 1500 | **0.559** | **0.722** | 0.681 | 0.997 | 242 ms |
| `test_unseen` | 1500 | 0.005 | 0.111 | 0.289 | 0.989 | 229 ms |
| `test_unseen_paraphrase` | 1500 | 0.005 | 0.098 | 0.292 | 0.991 | 242 ms |

BM25 on the identical examples:

| Split | BM25 exact | BM25 command | BM25 token F1 | Model exact | Δ exact |
|---|---:|---:|---:|---:|---:|
| `val` | 0.851 | 0.932 | 0.947 | 0.460 | −0.392 |
| `val_unseen` | 0.009 | 0.135 | 0.286 | 0.011 | +0.001 |
| `test_seen` | 0.843 | 0.918 | 0.918 | 0.525 | −0.317 |
| `test_paraphrase` | 0.877 | 0.942 | 0.964 | 0.559 | −0.317 |
| `test_unseen` | 0.007 | 0.124 | 0.277 | 0.005 | −0.003 |
| `test_unseen_paraphrase` | 0.008 | 0.114 | 0.291 | 0.005 | −0.003 |

Averaged over seen splits: BM25 exact **0.857** vs model **0.515**. On
unseen splits: exact ~0.008 vs ~0.005 — both fail exact match, but the
**token F1 comparison flips in the model's favor for the first time**
(0.289/0.292/0.315 vs 0.277/0.291/0.286), and command match is ahead on
`val_unseen` (0.138 vs 0.135). The tiny generative model now *composes*
roughly at parity with the retriever on novel pages, instead of trailing it
on every metric as run 2 did (F1 0.170–0.174 vs 0.248–0.286 there).

## 11.4 The frontier moved in three ways

1. **Seen-command composition works.** Exact 0.46–0.56 (run 2: ≤0.002),
   token F1 0.60–0.68 (run 2: ~0.18). Paraphrase robustness held: the
   *strongest* split is `test_paraphrase` (0.559) — 16 training phrasings
   made unseen phrasings a non-event.
2. **Unseen-page F1 ≥ retrieval.** The compositional advantage a generative
   model is supposed to have is finally measurable, if small.
3. **Within-tool confusions dominate the residue.** The seen splits' errors
   are mostly sibling-tool near-misses (`phpcs` ↔ `phpcbf`, `7zr` ↔ `7za`,
   `bun` ↔ `npm`).

## 11.5 The error census on `test_seen`

From the predictions file (1,500 examples):

| Outcome | Count | Share |
|---|---:|---:|
| Exact match | 788 | 52.5% |
| Right command, wrong arguments | 250 | 16.7% |
| Invalid bash | 4 | 0.3% |
| Empty generation | 0 | 0.0% |

**The `sudo` prior persists in attenuated form:** 14.7% of `test_seen`
predictions begin with `sudo` vs 5.5% of references (run 2: 42%). The
over-prediction concentrates on obscure admin pages (`zeroclaw`, `pveum`,
`koji`, `pg_config`) where root access is *plausible* — a rational prior
applied at the wrong time, and a warning sign that must be surfaced to users
(the CLI's caution patterns do flag `sudo`).

Representative failures, quoted from the failure log:

```text
reference: git changelog --start-tag 0.5.0 --final-tag 1.0.0
prediction: git changelog --filter 5:1.0.0 --list          # right family, invented flags

reference: zeroclaw agent
prediction: hangman                                        # unseen page → random command

reference: mysql_secure_installation --help
prediction: sudo rkhhunter --help                          # obscure page → sudo + wrong tool
```

## 11.6 What run 3 establishes

- **Scale (4.6× params, 10.9× tokens) converted the broad model from
  "usually the wrong tool" (≤2% exact) to a competitive suggestion engine on
  seen commands (53–56% exact with unseen phrasings).**
- **The unresolved axis is page-level generalization**, not decoding, not
  family selection, not packing: 0.5% exact on unseen pages, with the
  fallback prior (`sudo` + wrong tool) exactly where theory says a
  data-limited conditional LM should land.
- **More variants ≠ more knowledge.** The val_unseen loss regression
  (3.80 → 4.15) shows that phrasing diversity sharpens known-page fit; only
  *page diversity* (new tools/flags in training) or external knowledge
  (retrieval) moves unseen pages.
- The practical split stands: `tinyshell --retrieve` for commands known to
  be in the corpus (BM25: 0.84–0.88 exact, 0 ms); the neural path for
  composing *modifications* of known commands and for unseen-page behavior
  at parity with retrieval — plus a real path forward (RAG, Chapter 14).


---

# Part IV — Understanding and Going Forward


---

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


---

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


---

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


---

# Appendices


---

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


---

# Chapter 16 — Glossary

Quick definitions of every term of art used in this book, in the specific
sense TinyShell uses them.

**Active (supervised) tokens** — positions whose label ≠ `-100`; they
contribute loss and receive gradients. ≈14% of packed positions in the XL
corpus (17,411,259 tokens).

**AdamW** — Adam with *decoupled* weight decay (Loshchilov & Hutter, 2019).
TinyShell applies decay 0.1 to matrix weights only; norms/biases/embeddings
decay 0.

**Base example** — one parsed tldr description + command template pair
(28,659 usable in v1).

**BM25 baseline** — retriever over training requests (k1 1.5, b 0.75) that
returns the nearest training request's command. Near-oracle on seen splits,
~0.8% on unseen pages.

**Block-diagonal causal mask** — attention mask restricting each token to
earlier tokens *of its own document* inside a packed row
(`TinyLM.document_mask`). Identical to plain causal masking at inference.

**Byte-level BPE** — byte-pair encoding whose base alphabet is all 256 byte
values; no unknown tokens, exact round trips. TinyShell vocab: 4,096.

**Chunks** — fixed-length (256-token) training rows containing one or more
complete examples plus padding. XL train: 468,212.

**Command match** — metric: first executable token of prediction equals
reference's, skipping wrappers like `sudo`/`env`.

**Core experiment** — 2.1M model trained on the 655-page / 111-first-token
subset with `first_token_weight` 5.0; separated label-space limits from
capacity limits.

**Cross-entropy (nats)** — training loss; uniform guess = ln 4096 ≈ 8.32;
run 3 best val = 0.6565 (perplexity 1.93).

**doc_ids** — per-token id of the packed example a token belongs to; −1 on
padding. Runtime input enabling the block-diagonal mask and mask ablations.

**Document-aligned packing** — packing complete examples only, with per-token
doc ids and within-example positions (run 2's fix; ~14% padding cost).

**Dropout** — residual + attention dropout (0.05 broad/XL; 0.1 core),
disabled at eval.

**EOS (`<|eos|>`)** — supervised end-of-command token; generation stops on
it. Special ids 0–4: pad, eos, system, user, assistant.

**Exact match** — normalized prediction string equals reference. Strict: no
semantic equivalence. Seen-split run 3: 0.525–0.559; unseen: 0.005.

**First-token weighting** — multiply the first command token's loss by
`first_token_weight` (5.0 in core run) to emphasize command-family selection.

**Greedy decoding** — argmax next token; deterministic; temperature 0;
stops at EOS or 64 new tokens.

**GQA** — grouped-query attention; *not* used here (8 full heads).

**Packing shortcut** — run 1's learned dependence on previous packed
examples' commands; proven by prefix probe and eval-time mask ablation
(3.866 → 3.306).

**Paraphrase splits** — `test_paraphrase`, `test_unseen_paraphrase`: prompt
styles 10–11, never seen in training.

**Perplexity (PPL)** — exp(cross-entropy); effective choice count per
position.

**Position ids (per-example)** — within-example positions (restart at 0 per
document), fed to RoPE; stored at packing time.

**Risk assessment** — regex classification of generated commands into
safe / caution / destructive for CLI warnings; explicitly not a security
boundary.

**RMSNorm** — LayerNorm without mean/bias; pre-norm placement.

**RoPE** — rotary position embeddings; rotates q/k pairs by
position-dependent angles so attention scores depend on relative offsets;
θ = 10,000; zero learned parameters.

**SDPA** — PyTorch fused `scaled_dot_product_attention` used for the
attention core.

**Slot filling** — replacing tldr `{{placeholders}}` with pool values (file,
path, user, host, port, …) and alternative syntax `{{[-a|--all]}}`; fresh
values per variant.

**Styles (prompt styles)** — 16 request templates; train pool 0–7 (+12–15 in
XL), val 8, test_seen 9, paraphrase 10–11. Held-out styles are a generalization
axis.

**SwiGLU** — gated MLP: `down(silu(gate(x)) * up(x))`; intermediate 512
(2.1M) / 768 (9.6M).

**tldr-pages** — CC-BY-4.0 community corpus of simplified man pages; the
project's sole data source.

**Token F1** — multiset F1 over `shlex` arguments (command name excluded);
partial credit metric.

**Tied embeddings** — `lm_head.weight = embed.weight`; saves |V|×d
parameters (1.05M on the 9.6M model).

**Unseen pages** — evaluation on command pages excluded from training by the
page-level hash split; the composition axis of generalization.

**val / val_unseen** — packed-loss validation sets: seen pages + held-out
style 8; unseen pages + seen styles. Checkpoint selection uses `val`.

**Variants per example** — number of rendered (request, command) pairs per
base example (6 in v1; 150 in XL; every variant has fresh slot values and a
cycled style).

**Weight tying** — see tied embeddings.

**`bash -n` validity** — syntax-only parse check of a generated command;
never executed. 99.7–99.9% across all runs.
