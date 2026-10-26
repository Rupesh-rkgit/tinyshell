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
