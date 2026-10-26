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
