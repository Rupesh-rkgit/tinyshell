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
