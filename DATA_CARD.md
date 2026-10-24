# TinyShell data card

## Dataset summary

TinyShell uses **tldr-pages** as its only source of command knowledge. Each tldr
example is a short description plus a command template with `{{placeholders}}`.
The pipeline parses those examples, filters out commands that are unsafe or hard
to represent, fills placeholders with realistic values, and renders each example
as a natural-language request paired with one command.

| | |
|---|---|
| Source | [tldr-pages/tldr](https://github.com/tldr-pages/tldr) |
| Source commit | `95062b79e30fc3539246399b1dbfcfabbccc79cf` |
| License | CC-BY-4.0 |
| Platforms | `pages/common`, `pages/linux` |
| Raw pages / examples | 6,803 / 30,308 |
| Usable pages / base examples | 6,621 / 28,659 |
| Training requests | 135,860 (5,258 pages) |
| Evaluation requests | 1,500 per split × 6 splits |
| Language | English requests, Linux shell commands |
| Task | single command generation |
| Tokenizer | 4,096-token byte-level BPE |
| Context | 256 tokens |

## Collection

`tinyshell-prepare` performs a sparse Git clone of tldr-pages:

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/tldr-pages/tldr.git data/raw/tldr
git -C data/raw/tldr sparse-checkout set pages/common pages/linux
```

The commit hash of the clone is recorded in
`data/processed/dataset_manifest.json`. The clone is not redistributed in this
repository; run `make data` to fetch it. Command examples are manually written
by tldr contributors, not scraped from users or executed on the machine.

## Processing

1. **Parse** Markdown pages: `- description:` followed by a backtick-quoted
   command.
2. **Filter** examples that are not a single, self-contained Linux command:
   - descriptions shorter than 8 or longer than 160 characters;
   - commands longer than 120 characters;
   - multiline commands, multiple statements (`;`), shell substitutions
     (`$(...)`, `${...}`, `{{$...}}`), history expansion (`!!`, `!$`);
   - interactive keybindings, editor commands, shell snippets/keywords;
   - comments and malformed placeholders;
   - exact duplicate description/template pairs.
3. **Split by page title** with a deterministic hash (80/10/10), so all examples
   of a command go to exactly one split. `test_unseen` and
   `test_unseen_paraphrase` therefore contain commands never seen in training.
4. **Fill placeholders** with values from realistic pools (paths, files, users,
   hosts, packages, ports, services, patterns, versions, etc.) and resolve
   tldr alternative syntax such as `{{[-a|--all]}}` or `{{[clou|cloudinit]}}`.
5. **Render requests** in one of 12 prompt styles and append a context sentence
   describing the filled values (e.g. "Assume the file is `/tmp/notes.txt`.").
6. **Deduplicate** identical (request, command) pairs and write JSONL files plus
   a manifest with counts, filter reasons, risk distribution, and `bash -n`
   validity.

Overall `bash -n` validity of a 2,000-command sample is **99.85%**. The
remaining invalid examples come from tldr templates whose placeholder substitution
does not preserve syntax.

## Splits

| Split | Size | Commands | Prompt style | Purpose |
|---|---:|---|---|---|
| `train` | 135,860 | seen | 0–7 | training |
| `val` | 1,500 | seen | 8 (held out) | checkpoint selection |
| `val_unseen` | 1,500 | unseen | 0–7 | validation on unseen commands |
| `test_seen` | 1,500 | seen | 9 (held out), new slots | slot filling + style robustness |
| `test_paraphrase` | 1,500 | seen | 10–11 (unseen) | paraphrase robustness |
| `test_unseen` | 1,500 | unseen | 0–7 | command generalization |
| `test_unseen_paraphrase` | 1,500 | unseen | 10–11 (unseen) | both held out |

`test_seen` and `test_paraphrase` use the same command pages as training but new
slot values and held-out request styles. `test_unseen` and
`test_unseen_paraphrase` use entirely held-out pages.

## Label distribution and statistics

- **Risk:** 129,177 safe, 14,822 caution, 861 destructive across all splits
  (heuristic pattern-based labels, not ground truth).
- **Command length:** mean 30.0 characters; p50 26, p90 54, p99 82.
- **Request length:** mean 94.9 characters; p50 90, p90 151, p99 214.
- **Tokenized example length:** median 67 tokens; p99 119; max 137.
- **Top slot categories:** file (38,833), name (13,611), path (7,998),
  number (7,145), directory (6,258), text (5,581), host (5,135), user (4,496).
- **Filter reasons:** command too long (731), interactive keybinding (359),
  duplicate (236), shell substitution/variable (108), multiple statements (96).

The full breakdown is in `data/processed/dataset_manifest.json`.

## Focused core subset

`data/core/` is a derived subset used by the TinyShell Core experiment. It is
not a second data source: `scripts/build_core.py` filters the broad splits by
the first command token, keeping pages whose tool is in a curated list of ~100
common Linux/DevOps commands (`ls`, `find`, `git`, `docker`, `systemctl`,
`kubectl`, `aws`, `ffmpeg`, ...). The same page-level split, slot pools, and
prompt styles are preserved.

| | |
|---|---|
| Pages | 655 |
| Training requests | 17,317 |
| Training chunks (256 tokens) | 5,186 |
| Real tokens per epoch | 1,143,753 |
| Active command/EOS label tokens per epoch | 176,359 |
| `val` / `test_seen` / `test_paraphrase` | 187 / 179 / 211 requests |
| Distinct first command tokens | 111 (91 in the curated command set; the rest are shell snippets or aliases from pages such as `crontab` and `xargs`) |

The core subset is smaller but much narrower: 111 command-family labels instead
of 3,651 in the broad training split. This makes it a diagnostic for whether
the 2.1M model can learn command selection when the label space is small, not a
deployment dataset. The unseen core splits are aliases of the seen splits
because every retained page is in the seen set; core evaluation therefore
measures slot filling and paraphrase robustness, not unseen-command
generalization.

## Known biases and limitations

- **Synthetic requests.** Real users write differently from the 12 templates.
  Training on templated paraphrases can overstate robustness and makes the
  evaluation cleaner than real deployment.
- **Fixed slot pools.** Paths, users, hosts, and packages are drawn from small
  hand-written lists. The model may memorize common values instead of learning
  to copy arbitrary entities.
- **tldr coverage bias.** Only `common` and `linux` pages are used. Commands
  popular with tldr contributors are overrepresented; niche tools, distro
  specifics, and non-English interfaces are missing.
- **Safety labels are heuristics.** The risk distribution comes from regex
  patterns, not expert annotation. `rm` is caution, `rm -rf` is destructive,
  but context and paths determine actual risk.
- **No execution feedback.** Commands are never run during dataset creation;
  syntax is checked with `bash -n` only. Semantic correctness and
  version-specific flags are not verified.
- **Snapshot.** The dataset reflects one tldr commit. Commands and flags change
  over time.
- **Names and stereotypes.** Synthetic users include `alice`, `bob`,
  `www-data`, and `root`; paths include `/home/alice`. These are conventional
  examples, not claims about real people.

## Ethical and legal considerations

tldr-pages is CC-BY-4.0. Attribution is included in `README.md`, `LICENSE`,
this data card, and the packaged model. Redistribution of derived datasets
should preserve the attribution and license notice.

The data contains command examples that can be destructive if executed (e.g.
`rm -rf`, disk tools). The model is trained to suggest commands, but the CLI
never executes them and displays risk warnings for review.

## Maintenance

The processed JSONL files and packed arrays are generated artifacts and are
gitignored. To rebuild:

```bash
make data        # clone + parse + split + fill
make tokenizer   # train BPE on the new train split
make train       # retrain the model
make eval        # evaluate the new checkpoint
```

Pin the tldr commit in the manifest if exact reproducibility matters. Changing
the source commit, filtering rules, prompt styles, slot pools, or split ratios
changes the dataset and invalidates comparisons with the shipped metrics.
