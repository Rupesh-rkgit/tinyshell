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
