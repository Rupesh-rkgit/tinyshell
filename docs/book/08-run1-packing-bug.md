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
