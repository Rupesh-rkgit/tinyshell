# TinyShell model card

## Model details

- **Name:** TinyShell
- **Version:** 0.1.0
- **Type:** decoder-only causal language model (Llama-style)
- **Parameters:** 2,098,816 total; 1,574,528 non-embedding
- **Architecture:** 6 layers, `d_model=128`, 4 attention heads (head dim 32),
  SwiGLU MLP (`intermediate_size=512`), RMSNorm, RoPE (θ = 10,000), causal
  scaled dot-product attention, tied input/output embeddings, dropout 0.05
- **Context length:** 256 tokens
- **Vocabulary:** 4,096 byte-level BPE tokens, including five special tokens
  (`<|pad|>`, `<|eos|>`, `<|system|>`, `<|user|>`, `<|assistant|>`)
- **Input:** a natural-language request, wrapped in a fixed chat template
- **Output:** one Linux shell command (no explanation)
- **Training hardware:** 4 vCPU ARM64 (Neoverse-N1), CPU-only PyTorch
- **License:** MIT for code and weights; training data is CC-BY-4.0
- **Contact:** TinyShell contributors

### Prompt template

```text
<|system|>
You are a Linux shell command assistant. Convert the request into exactly one command. Reply with the command only, no explanation.
<|user|>
{request}
<|assistant|>
{command}<|eos|>
```

During training, the loss is computed only on `{command}<|eos|>`; prompt tokens
are masked with `-100`. During inference, generation is greedy by default and
stops at `<|eos|>` or after 64 new tokens.

## Intended use

TinyShell is intended for:

- education and experimentation with small language models;
- offline, local suggestions for common Linux command-line tasks;
- a reproducible baseline for prompt/data/evaluation research;
- demonstrating an end-to-end training, evaluation, and packaging pipeline.

It is designed for a human in the loop: the user reads the suggestion and
decides whether to run it.

## Out-of-scope use

Do not use TinyShell for:

- autonomous command execution, agents, or unattended system administration;
- security-sensitive operations, production incident response, or destructive
  filesystem/device changes;
- macOS/Windows/BSD-specific commands;
- multi-step tasks, pipelines requiring state, or multi-turn conversations;
- interpreting natural language outside English;
- safety-critical decisions. The risk classifier is a warning heuristic, not a
  security control.

The model has no access to the filesystem, environment, installed packages, or
command history. It cannot verify that paths exist or that flags are supported
by the local version of a tool.

## Training data

TinyShell is trained on examples derived from
[tldr-pages](https://github.com/tldr-pages/tldr), CC-BY-4.0, restricted to
`pages/common` and `pages/linux`.

- 6,803 raw pages, 30,308 raw examples
- 6,621 usable pages, 28,659 usable base examples after filtering
- 135,860 training requests from 5,258 pages (6 variants per base example)
- 1,500 examples in each of six held-out evaluation splits
- Page-level splits: 80% train / 10% validation / 10% test, so unseen-command
  splits contain commands and pages never seen during training

Requests are synthetic: each tldr example's placeholders are filled with
realistic values, and the description is rendered in one of 12 prompt styles.
Training uses styles 0–7. Style 8 is used for validation, style 9 for
`test_seen`, and styles 10–11 for paraphrase tests. See `DATA_CARD.md` for
filtering rules, slot categories, and known biases.

## Training procedure

- **Objective:** next-token cross-entropy on command and EOS tokens only
- **Epochs:** 4
- **Optimizer steps:** 2,608
- **Batch:** 64 packed 256-token sequences
- **Tokens per epoch:** 9,164,892 real tokens (10,686,208 positions including
  padding; 36.7M real-token passes total)
- **Packing:** complete examples only, with `doc_ids` and per-example
  `position_ids`; block-diagonal causal attention prevents cross-example
  leakage
- **Optimizer:** AdamW, β = (0.9, 0.95), weight decay 0.1 on matrix weights
- **Learning rate:** 3e-4, cosine decay to 3e-5, 200-step linear warmup
- **Gradient clipping:** 1.0
- **Checkpoint selection:** lowest validation loss (`val`, held-out prompt style)
- **Seed:** 42
- **Throughput:** roughly 3–4k tokens/second on the target 4-vCPU machine

The full configuration is in `configs/tinyshell.yaml`; the exact run metadata is
in `artifacts/runs/tinyshell-2m-run2-docmask/`.

## Evaluation

The primary evaluation is exact-match accuracy on six held-out splits, with
command-match and argument-token F1 as partial-credit metrics and `bash -n`
syntax validity as a sanity check. A BM25 retrieval baseline is evaluated on
the same examples.

<!-- EVAL:START -->
Run `tinyshell-2m-run2-docmask`, checkpoint `best`, 2,098,816 parameters.

| Split | N | Exact | Command | Token F1 | Valid bash | p50 latency |
|---|---:|---:|---:|---:|---:|---:|
| `val` | 1500 | 0.001 | 0.073 | 0.174 | 0.985 | 155 ms |
| `val_unseen` | 1500 | 0.001 | 0.080 | 0.174 | 0.991 | 133 ms |
| `test_seen` | 1500 | 0.000 | 0.081 | 0.173 | 0.987 | 151 ms |
| `test_paraphrase` | 1500 | 0.002 | 0.089 | 0.187 | 0.995 | 140 ms |
| `test_unseen` | 1500 | 0.000 | 0.069 | 0.170 | 0.995 | 139 ms |
| `test_unseen_paraphrase` | 1500 | 0.000 | 0.062 | 0.172 | 0.991 | 141 ms |

### BM25 baseline

| Split | BM25 exact | BM25 command | BM25 token F1 | Model exact | Δ exact |
|---|---:|---:|---:|---:|---:|
| `val` | 0.601 | 0.916 | 0.811 | 0.001 | -0.601 |
| `val_unseen` | 0.006 | 0.131 | 0.259 | 0.001 | -0.005 |
| `test_seen` | 0.601 | 0.915 | 0.815 | 0.000 | -0.601 |
| `test_paraphrase` | 0.599 | 0.921 | 0.825 | 0.002 | -0.597 |
| `test_unseen` | 0.006 | 0.113 | 0.248 | 0.000 | -0.006 |
| `test_unseen_paraphrase` | 0.006 | 0.111 | 0.254 | 0.000 | -0.006 |
<!-- EVAL:END -->

### Focused core variant (diagnostic)

A side experiment narrows training to five command families (17,317 requests,
111 distinct first-command labels) and trains the same 2.1M architecture for
20 epochs with a 5× first-command-token loss weight:

| Split | N | Exact | Command | Token F1 | Valid bash |
|---|---:|---:|---:|---:|---:|
| `val` | 187 | 0.048 | 0.866 | 0.227 | 0.952 |
| `test_seen` | 179 | 0.028 | 0.737 | 0.155 | 0.983 |
| `test_paraphrase` | 211 | 0.057 | 0.924 | 0.260 | 0.981 |

Command-family match rises to 74–92% and `sudo` over-prediction drops from 42%
to 6.7% of predictions, while exact match stays at 3–6%: family selection is
label-space limited, argument composition is capacity limited. This variant
lives in `artifacts/runs/tinyshell-core-2m/` and is a diagnostic, not the
packaged default.

### Metric caveats

- Exact match is strict: a different-but-valid command is counted as wrong.
- `bash -n` checks syntax only. It does not execute the command and cannot
  detect a dangerous or semantically incorrect command.
- The test prompts are synthetic paraphrases. Performance on real user phrasing
  is likely lower.
- The BM25 baseline retrieves from training requests. It is strong on seen
  commands by construction and cannot compose unseen commands.

## Limitations and risks

- **Hallucination:** generated flags, paths, and subcommands may not exist or
  may not match the installed tool version.
- **Stale knowledge:** the model learns only what was in the tldr snapshot at
  training time.
- **Distribution shift:** synthetic slot values and prompt templates differ from
  real usage.
- **Safety:** pattern-based risk warnings can produce false negatives and false
  positives. The model may suggest destructive commands; it must not be given
  execution authority.
- **Coverage:** common/Linux commands only; no shell state, environment
  variables, aliases, or user-specific tooling.
- **Size:** 2.1M parameters limit world knowledge and compositional reasoning.
  Run 2 gets the command family right on 6–9% of held-out requests, and 42% of
  `test_seen` predictions still begin with `sudo` versus 6.8% of references.
  It often produces a plausible tool with the wrong subcommand, or copies the
  right arguments under the wrong command. The focused core variant (see
  above) shows family selection is tractable in a small label space while
  exact-match argument composition remains the hard part. See
  `docs/EXPERIMENTS.md` for the packing bug found in run 1, the fix evaluated
  in run 2, the core-command experiment, and the 10M-parameter run.
- **Bias:** tldr-pages reflects its contributors' choices of commands, examples,
  and naming; the slot pools use synthetic names and could reinforce stereotypes
  (e.g. `alice`/`bob`, `/home/alice`). No demographic data is used.

## Ethical considerations

TinyShell is a learning project. It should be presented as a suggestion tool,
not an authority. Users should review commands, especially anything involving
deletion, permissions, disks, services, or privileged access. The CLI never
executes commands and always displays the generated command for review.

## Citation

If you use TinyShell in teaching or research, cite this repository and
tldr-pages:

```text
TinyShell: a 2.1M-parameter natural-language-to-shell-command model.
https://github.com/tldr-pages/tldr (CC-BY-4.0).
```

## Maintenance

This is a static educational artifact. The tokenizer, model weights, config,
evaluation report, and environment metadata are packaged together so the run can
be inspected after the fact. Retraining requires the data pipeline and the
source snapshot described in `DATA_CARD.md`.
