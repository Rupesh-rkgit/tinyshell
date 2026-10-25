# Experiments and failure analysis

This note records the main engineering finding from the first TinyShell run and
the change made in the second run. It is intentionally blunt: the first model
did not learn the task well, and the evaluation exposed why.

## Run 1: naive sequence packing

The first run packed all prompt+completion examples into one continuous token
stream, sliced it into 256-token chunks, and used ordinary causal attention.
This is a common and efficient way to train a small language model, but it has a
subtle train/inference mismatch:

- During training, a token in example *N* could attend to the command in example
  *N-1*. The model learned to copy or continue from that unrelated command.
- At inference, the prompt is a single example with no preceding example. The
  shortcut is unavailable, so the model fell back to high-frequency patterns.
- Position ids also ran continuously across a packed chunk, while inference
  starts at position 0.

The result was a model that produced syntactically valid but mostly wrong
commands. On the 1,500-example `val` split it had **0.000 exact match**, **0.065
command match**, and **0.108 token F1**. 72% of its predictions began with
`sudo`, although only 6% of references did. A typical output was:

```text
request:    list all files including hidden ones, with human-readable sizes
prediction: sudo aa-cli --list
```

Full run-1 evaluation:

| Split | Exact | Command | Token F1 | Valid bash |
|---|---:|---:|---:|---:|
| `val` | 0.000 | 0.065 | 0.108 | 0.995 |
| `val_unseen` | 0.001 | 0.078 | 0.126 | 0.998 |
| `test_seen` | 0.000 | 0.084 | 0.117 | 0.992 |
| `test_paraphrase` | 0.000 | 0.055 | 0.128 | 0.995 |
| `test_unseen` | 0.000 | 0.067 | 0.115 | 0.997 |
| `test_unseen_paraphrase` | 0.000 | 0.047 | 0.113 | 0.996 |

### Diagnostic

The leakage was confirmed two ways.

1. **Prefix test.** Prepending a fake previous example to the prompt changed the
   output dramatically. With `ls -la` as the previous command, the model copied
   `ls -la`; with no prefix it produced `sudo aa-cli --list`.
2. **Mask ablation.** The same run-1 checkpoint was evaluated on the packed
   validation set twice: once with ordinary causal attention, and once with a
   block-diagonal mask that blocks attention across examples. No retraining was
   involved.

| Attention at eval time | Validation loss |
|---|---:|
| Causal over the whole packed chunk | 3.866 |
| Causal + block-diagonal by example | 3.306 |

Blocking the shortcut *improved* loss by 0.56 nats. The model was not using the
previous examples as useful context; it had learned a harmful copying habit.

## Run 2: document-aligned packing

The fix touches packing, attention, and RoPE positions:

- **Pack complete examples only.** A chunk is filled with one or more complete
  examples and padded at the end. An example is never split across chunks.
- **Store `doc_ids` and `position_ids`.** Padding gets document id `-1`.
  Position ids restart at 0 for every example.
- **Block-diagonal causal mask.** A token attends only to earlier tokens in its
  own example. At inference (one document) this is exactly the usual causal mask.
- **Per-example RoPE.** Positions are the within-example positions, matching
  inference.

The packed corpus grows from 35,800 to 41,743 chunks because of the padding
(about 14% of positions), but the mask itself adds no measurable CPU cost.

### Run 2 results

Run 2 (`tinyshell-2m-run2-docmask`) used the same model, tokenizer, data, and
optimizer as run 1, changing only the packing/attention/position handling. It
trained for 4 epochs (2,608 steps) and took about 3h 16m on the 4-vCPU host.

| Metric | Run 1 (naive packing) | Run 2 (document-aligned) |
|---|---:|---:|
| Best validation loss | 3.9749 | **2.9377** |
| Final validation loss | 3.9771 | **2.9214** |
| Final `val_unseen` loss | 4.6025 | **3.7993** |
| Parameters | 2,098,816 | 2,098,816 |
| Training chunks | 35,800 | 41,743 |

Removing the cross-example shortcut lowered validation loss by about **1.04
nats** and unseen-page loss by **0.80 nats**. The model also learned to copy
arguments from the request much more reliably: token F1 on the quick 200-example
`val` check rose from roughly 0.11 to 0.20. It still usually chooses the wrong
command family, so exact match remains near zero. The full six-split evaluation
is in `artifacts/eval/tinyshell-2m-run2-docmask/`.

<!-- RUN2:START -->
Full six-split evaluation (1,500 examples per split, greedy decoding):

| Split | Exact | Command | Token F1 | Valid bash |
|---|---:|---:|---:|---:|
| `val` | 0.001 | 0.073 | 0.174 | 0.985 |
| `val_unseen` | 0.001 | 0.080 | 0.174 | 0.991 |
| `test_seen` | 0.000 | 0.081 | 0.173 | 0.987 |
| `test_paraphrase` | 0.002 | 0.089 | 0.187 | 0.995 |
| `test_unseen` | 0.000 | 0.069 | 0.170 | 0.995 |
| `test_unseen_paraphrase` | 0.000 | 0.062 | 0.172 | 0.991 |

Compare with run 1's command match of 0.047–0.084 and token F1 of
0.108–0.128. Run 2 is better on every split, especially on argument copying,
but exact match is still essentially zero.

The frequency prior also weakened but did not disappear. On `test_seen`, 42% of
run-2 predictions begin with `sudo` (626/1,500), down from 54% in run 1, while
only 6.8% of references do. The model still overuses a high-frequency prefix.

A typical failure is:

```text
request:    stream pod logs. Assume the pod name is `alice`.
reference:  kubectl logs --follow alice
prediction: kubectl set --name alice --user
```

The model has learned the command family (`kubectl`), the entity (`alice`), and
valid flag syntax, but not which subcommand and flags express "stream logs".
BM25 returns the correct command for this request because it retrieves a
near-duplicate training request.
<!-- RUN2:END -->

The remaining gap is a capacity/data problem, not a masking bug. A 2.1M model
trained on 9.2M real tokens cannot cover 3,651 command names. The companion
**TinyShell Core** experiment narrows the task to ~100 common command families
and ~111 first tokens; it is a more realistic target for this model size. See
`configs/core.yaml` and `scripts/build_core.py`.

## Core-command experiment

The broad dataset contains 3,651 distinct first tokens. That is a hard label
space for a 2.1M model trained on 9.2M real tokens. To separate "the model
cannot learn the task" from "the model cannot cover thousands of commands", we
built a focused subset:

- keep tldr pages whose title starts with a command in a curated list of ~100
  common Linux/DevOps commands (`ls`, `git`, `docker`, `systemctl`, `kubectl`,
  `aws`, `cargo`, `ffmpeg`, ...);
- 655 pages remain, 17,317 training requests, and 111 distinct first command
  tokens (91 are in the curated set; the rest are snippets or aliases);
- train the same 2.1M architecture for 20 epochs with dropout 0.1 and a
  `first_token_weight` of 5.0, which upweights the command-family token in the
  cross-entropy loss;
- evaluate on seen pages with new slot values (`test_seen`) and unseen prompt
  styles (`test_paraphrase`). BM25 is evaluated on the same splits.

Build and train with:

```bash
python scripts/build_core.py
python -m tinyshell.train --config configs/core.yaml
```

A data-quality caveat: the builder filters by page title, not by command. A few
retained examples are shell snippets (`!git`, a crontab schedule, an `xargs`
pipeline), so the 111 first-token labels include about 20 non-command tokens.
Run 3 adopts a stricter example-level filter that rejects these (see below).

<!-- CORE:START -->
### Core results

Trained 1,620 steps (20 epochs) in about 2.2 hours; best validation loss
1.5472 (final 1.5404). Reference: `artifacts/runs/tinyshell-core-2m/`.

| Split | N | Exact | Command | Token F1 | Valid bash | p50 |
|---|---:|---:|---:|---:|---:|---:|
| `val` | 187 | 0.048 | 0.866 | 0.227 | 0.952 | 161 ms |
| `test_seen` | 179 | 0.028 | 0.737 | 0.155 | 0.983 | 161 ms |
| `test_paraphrase` | 211 | 0.057 | 0.924 | 0.260 | 0.981 | 156 ms |

BM25 on the same splits (from `artifacts/eval/tinyshell-core-2m/report.md`):
exact **0.674 / 0.642 / 0.640**, command match **1.000 / 0.983 / 0.976**,
token F1 **0.878 / 0.857 / 0.864**.

**Command-family selection improved dramatically.** On the same 187 `val`
examples, the broad run-2 model reaches 0.358 command match and 0.144 token
F1 (evaluated with `--data-dir data/core`; predictions kept in
`artifacts/eval/tinyshell-2m-run2-coreval/`), versus 0.866 and 0.227 for the
core run. The comparison is not a controlled ablation — the core training set
is a 17,317-request slice of the broad corpus — but it demonstrates that
narrowing the first-command label space from 3,651 to 111 tokens converts
family selection from a failure into a nearly solved subproblem.

Two specific confirmations:

- The `sudo` frequency prior collapsed: only 12 of 179 `test_seen` core
  predictions (6.7%) begin with `sudo`, versus 626 of 1,500 (42%) for run 2.
  With a small label space and the 5× first-token weight, the model learns
  `sudo` as a conditional rather than a constant.
- Single-command-per-page filtering removed the multi-command distraction; the
  invalid-bash rate is now 1.7% on `test_seen` (vs 1.3% in run 2, essentially
  unchanged, so packing was not the source of malformed output).

**Arguments remain the weak link.** On `test_seen`, 127 of 179 predictions
(71%) choose the right command family but get the arguments wrong, and only
5 of 179 are exact. A representative failure:

```text
request:    stream pod logs. Assume the pod name is `alice`.
reference:  kubectl logs --follow alice
prediction: kubectl logs --image alpine:3.12-project
```

BM25 still triples the model's exact match (0.642 vs 0.028) because retrieval
copies a complete training command the model must instead compose.

**Conclusion.** Family selection is label-space limited; argument composition
is model/data limited. That is the direct motivation for Run 3 (10M
parameters, ~90M training tokens, same evaluation), below.
<!-- CORE:END -->

The shipped core run used the first version of the weighted loss, which tagged
only the first active token in each packed chunk (about one example in three).
The implementation now tags the first command token of every packed example;
the shipped core run predates the fix, so a re-trained core run would apply
the full 5× weight to every example while the reported numbers reflect the
partial version. Treat the core numbers as a slightly conservative estimate
of the weighting effect.

## Run 3: 10M parameters, ~100M training tokens (completed)

Run 3 scales the broad setup along the three axes the diagnostics above point
at: capacity, data volume, and data quality. It is **not directly comparable**
to run 2 because the corpus and filters change, but the page-level split is
unchanged, so the eval splits overlap heavily and directional comparisons
remain meaningful.

- **Architecture.** Same Llama-style decoder family, upsized from 128d × 6L
  to **256d × 10L, 8 heads, SwiGLU intermediate 768**: 9,573,632 parameters
  (~4.6× run 2). Head dimension stays 32, context stays 256, embeddings stay
  tied, dropout 0.05 (more data → lighter regularization). The deliberate
  choice is plain scaling of a proven design — the 2.1M experiments showed
  no instability that an architecture change would fix, while capacity was
  the binding constraint.
- **Data volume.** `variants_per_example` goes from 6 to **150** (styles cycle
  deterministically over the training pool and every variant draws fresh slot
  fills), growing the corpus from 9.2M to **99,992,950 training tokens**
  (1,303,083 unique requests across 468,212 document-aligned chunks — one
  pass). Slot-free examples saturate at their 12 distinct phrasings;
  slot-bearing examples scale near-linearly.
- **Data quality.** The example filter now additionally rejects: a leading `!`
  (history expansion like `!git ...`), a first token shaped like a cron
  schedule (`0 10 * * * git status`), and process substitution
  (`source <(kubectl completion zsh)`). Four additional prompt templates
  (style indices 12–15) join the training pool; the held-out eval styles
  (8, 9, 10–11) are unchanged.
- **Unchanged:** tokenizer (byte-level BPE, vocab 4096 — byte fallback covers
  new slot values), page-level splits, doc-aligned packing,
  `first_token_weight` 1.0.

Training ran to completion: 3,657 steps over 27.7 hours (99,690 s) on all 4
vCPU threads at a steady ~1,190 tokens/s (32,768 positions/step), one pass
over the packed corpus, cosine schedule ending at lr 3e-5, no early stop.

### Run 3 training outcome

- **Validation loss (seen pages, held-out style 8): 0.6565** vs run 2's
  2.9377/2.9214 (best/final) — the enlarged corpus plus 4.6× capacity cut
  seen-command fit dramatically.
- **Unseen-page validation loss: 4.1451 — run 3 is *worse* here than run 2**
  (3.7993). That inversion is the headline: the 150×-variant corpus repeats
  each *page* many times with fresh phrasings and slot values, so the model
  hyper-fits its known command families while gaining little on pages it
  has never seen. In loss terms the run-2/run-3 seen/unseen gap widened
  from 0.87 to 3.49.
- Training loss at step 3,650 was ~0.13 (checkpoint: `best.pt` at val 0.6565).
  Run 2 trained 4 epochs (~37.7M token passes); run 3 is a single pass over
  a 100M-token corpus — more distinct data, fewer repeats per string.
- Snapshots every 500 steps are in `artifacts/runs/tinyshell-xl-10m/`;
  cumulative throughput and the loss curves are in `plots/` and the
  dashboard.

<!-- RUN3-EVAL:START -->
### Run 3 generation evaluation

Checkpoint `best`, greedy decoding; the regenerated eval requests use
identical generation logic and page splits as run 2's (only the 139 pages
removed by the stricter filter changed), so cross-run comparison is
reasonably direct. Full data: `artifacts/eval/tinyshell-xl-10m/`.

| Split | N | Exact | Command | Token F1 | Valid bash | p50 |
|---|---:|---:|---:|---:|---:|---:|
| `val` | 1499 | 0.460 | 0.618 | 0.604 | 0.997 | 290 ms |
| `val_unseen` | 1500 | 0.011 | 0.138 | 0.315 | 0.995 | 227 ms |
| `test_seen` | 1500 | 0.525 | 0.692 | 0.644 | 0.997 | 245 ms |
| `test_paraphrase` | 1500 | 0.559 | 0.722 | 0.681 | 0.997 | 242 ms |
| `test_unseen` | 1500 | 0.005 | 0.111 | 0.289 | 0.989 | 229 ms |
| `test_unseen_paraphrase` | 1500 | 0.005 | 0.098 | 0.292 | 0.991 | 242 ms |

**And the frontier moved in three ways.**

1. **Seen-command composition works now.** Exact match on seen splits is
   0.46–0.56 (run 2: ≤0.002), token F1 0.60–0.68 (run 2: ~0.18). Paraphrase
   robustness held: `test_paraphrase` (0.559) is the *strongest* split —
   unseen phrasings do not hurt the model that trained on 12 phrasings.
2. **On unseen commands the model now edges out BM25 on F1.** Run 2 lost the
   unseen splits to retrieval on both metric families (e.g. F1 0.170–0.174
   vs 0.248–0.286). Run 3's unseen token F1 is 0.289/0.292/0.315 vs
   retrieval's 0.277/0.291/0.286 — ahead on all three (margin +0.001 to
   +0.029) — and command match is ahead only on `val_unseen`
   (0.138 vs 0.135). The tiny generative model now composes roughly
   at-par with the retriever on novel pages instead of trailing it.
3. **Within-tool confusions persist.** The seen splits' remaining errors
   are mostly near-misses (`phpcs`↔`phpcbf`, `7zr`↔`7za`, `bun`↔`npm`).

**Seen/unseen remains the run's core caveat.** Exactly as the training-loss
table above showed, the model hyper-fits pages it trained on: 0.525 exact on
`test_seen` vs 0.005 on `test_unseen`. For *known* pages the model is now a
competent suggestion engine; for *unknown* pages it is a combinatorial
`sudo`-biased prior. Error census (`test_seen`): 788/1500 exact (52.5%),
250 right-command-wrong-arguments (16.7%), 4 invalid bash (0.3%), zero empty.

The `sudo` prior persists in attenuated form: 14.7% of `test_seen`
predictions begin with `sudo` vs 5.5% of references (run 2: 42%). The
over-prediction concentrates on obscure admin commands (`zeroclaw`, `pveum`,
`koji`, `pg_config`) where root access in the training data is plausible —
rational, but defaults to privilege escalation and must be surfaced.

Representative failures (`test_seen`):

```text
reference: git changelog --start-tag 0.5.0 --final-tag 1.0.0
prediction: git changelog --filter 5:1.0.0 --list          # right family, invented flags
reference: zeroclaw agent
prediction: hangman                                        # unseen page -> random command
reference: mysql_secure_installation --help
prediction: sudo rkhhunter --help                          # obscure page -> sudo + wrong tool
```

**Comparison against BM25 (seen splits):** retrieval still wins exact match
on every seen split, 0.857 mean vs 0.515 — the requests are generated
paraphrases of training requests, so retrieval is close to an oracle there.
The practical split stands: `--retrieve` for commands known to be in the
corpus; the neural path only matters for composing modifications of known
commands (already more than run 2 could offer).

**Bottom line.** 4.6× parameters and 10.9× data moved the broad model from
"usually the wrong tool" (≤2% exact, 6–9% command match) to competitive
suggestion quality on seen commands (53–56% exact with unseen phrasings),
with model ≳ retrieval on unseen pages for the first time. The unresolved
axis is page-level generalization, not decoding or family selection: on
unseen pages the correct mechanism would need to be composition from a
stronger command vocabulary (or retrieval-augmented generation), which is
future work.
<!-- RUN3-EVAL:END -->

Repo-history note: the run was launched at commit `81bdf02`. That commit was
later rewritten during a git hygiene fix (the initial commit had accidentally
tracked large data/checkpoint files; history was filtered to code, docs, and
small eval artifacts only with identical content, and the `.gitignore` was
completed). Code-level content is unchanged; the new pre-launch hash is
`2b88264`.

## What this says about small-model evaluation

- **Loss can hide a shortcut.** Run 1's validation loss looked reasonable, but
  the model was optimizing the wrong conditional distribution: it was learning
  `P(command | request, previous command)` instead of `P(command | request)`.
- **Generation tests are necessary.** The prefix test and the mask ablation were
  more informative than loss alone.
- **A retrieval baseline is a strong sanity check.** BM25 reaches 0.60 exact
  match on seen-command splits because the evaluation requests are paraphrases
  of training requests. A generative model that cannot beat a simple retriever
  on seen commands is not yet useful, regardless of how fluent its output looks.
- **A 2.1M-parameter model has a capacity limit.** Fixing the packing removes a
  bug; it does not automatically give the model enough capacity to cover 5,000+
  commands. The honest framing is an educational prototype and a reproducible
  pipeline, not a replacement for `tldr` or a shell assistant.

## Reproduce

```bash
# Rebuild packed data with document-aligned chunks
.venv/bin/python -m tinyshell.dataset --config configs/tinyshell.yaml --force

# Train the fixed model
.venv/bin/python -m tinyshell.train --config configs/tinyshell.yaml \
  --run-id tinyshell-2m-run2-docmask

# Evaluate and compare
.venv/bin/python -m tinyshell.evaluate --run artifacts/runs/tinyshell-2m-run2-docmask
.venv/bin/python scripts/summarize_eval.py latest

# Focused core-command experiment
.venv/bin/python scripts/build_core.py
.venv/bin/python -m tinyshell.train --config configs/core.yaml
.venv/bin/python -m tinyshell.evaluate --run artifacts/runs/tinyshell-core-2m
```

A side-by-side generation comparison of run 1 and run 2 on the first 20 `val`
examples is in `artifacts/eval/run1-vs-run2.md`.
