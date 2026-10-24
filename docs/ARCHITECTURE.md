# Architecture notes

This document explains the design choices behind TinyShell for readers who want
to learn how a small decoder-only language model is built. The code is the
source of truth; this is a guided tour.

## 1. From tldr page to training sequence

A tldr example looks like this:

```markdown
- List files including hidden ones, with human-readable sizes:

`ls -lah`
```

The pipeline turns it into a request/command pair by filling placeholders and
choosing a prompt style:

```text
request:  "What's the best way to list files including hidden ones, with human-readable sizes?"
command:  "ls -lah"
```

It then renders the chat template:

```text
<|system|>
You are a Linux shell command assistant. ...
<|user|>
What's the best way to list files including hidden ones, with human-readable sizes?
<|assistant|>
ls -lah<|eos|>
```

The tokenizer is byte-level BPE with a 4,096-token vocabulary. All whitespace and
Unicode characters are representable because the base alphabet is the 256 byte
values. The five special tokens are reserved at ids 0–4.

## 2. Prompt masking and packing

We want the model to learn to produce the **command**, not to reproduce the
system prompt. `dataset.encode_example` builds:

```text
input_ids:  [system/user/assistant prompt tokens] [command tokens] [EOS]
labels:     [-100 ... -100]                        [command tokens] [EOS]
```

`-100` is PyTorch's `ignore_index` for cross-entropy, so prompt tokens contribute
nothing to the loss. About 14% of the 256-token positions in the packed corpus
are supervised (1.49M of 10.7M positions, including padding).

Examples are packed into 256-token chunks without splitting an example across
chunks. The unused tail of a chunk is padded; padding is ~14% of positions. Each
token also stores:

- `doc_ids`: which example inside the chunk it belongs to (`-1` for padding);
- `position_ids`: its position **within that example**, starting at 0.

The attention mask is causal **and** block-diagonal: a token may attend only to
earlier tokens in its own example. RoPE uses the within-example position. This
matters because at inference the prompt is a single example starting at position
0, with no preceding examples. A naive packed stream would train the model to
copy from unrelated preceding examples, a shortcut that disappears at inference
time. On the first run's checkpoint, simply blocking cross-example attention
lowered validation loss from 3.866 to 3.306 without any retraining — direct
evidence that the shortcut was harmful. See `docs/EXPERIMENTS.md`.

## 3. The model

TinyShell is a pre-norm decoder-only transformer with six identical blocks.

### RMSNorm

Layer normalization without mean subtraction or bias:

```text
RMSNorm(x) = x / sqrt(mean(x²) + ε) * γ
```

It is cheaper than LayerNorm and works well in transformer language models.

### Rotary position embeddings (RoPE)

Instead of adding a position vector, RoPE rotates pairs of query/key dimensions
by an angle proportional to the token position:

```text
q_rot = q cos(θ_pos) + rotate_half(q) sin(θ_pos)
k_rot = k cos(θ_pos) + rotate_half(k) sin(θ_pos)
```

The dot product of rotated vectors depends on their relative distance, which
gives the model relative position information without learned position
embeddings. Frequencies are precomputed up to the 256-token context.

### Causal self-attention

Queries, keys, and values are linear projections of the same hidden state. The
model uses PyTorch's fused `scaled_dot_product_attention`:

```text
Attention(Q, K, V) = softmax(Q Kᵀ / sqrt(d_head) + mask) V
```

During training the mask is lower-triangular **and** block-diagonal over packed
documents: position `t` can attend only to positions `≤ t` in the same example.
At inference (one document) this reduces to the usual causal mask. This is what
makes next-token prediction a valid objective without leaking information from
unrelated examples.

With 4 heads of dimension 32, each head can specialize: lexical copying of slot
values, command-name selection, flag patterns, etc.

### SwiGLU MLP

Each block's feed-forward network is:

```text
SwiGLU(x) = down(silu(gate(x)) * up(x))
```

The gated formulation gives more expressive power per parameter than a plain
two-layer MLP at this scale. The intermediate size is 512, four times `d_model`.

### Residual stream

Each block adds its attention and MLP outputs back into the residual stream:

```text
x = x + Attention(RMSNorm(x))
x = x + SwiGLU(RMSNorm(x))
```

Pre-norm keeps gradients stable without warmup-heavy tricks. The final RMSNorm is
followed by a linear head. Input embeddings and the output head share weights
("weight tying"), which saves 524k parameters and is standard for small LMs.

### Parameter count

| Component | Parameters |
|---|---:|
| Token embedding (tied with output head) | 524,288 |
| Per layer: QKV projection | 49,152 |
| Per layer: attention output | 16,384 |
| Per layer: SwiGLU gate/up/down | 196,608 |
| Per layer: two RMSNorms | 256 |
| Per layer total | 262,400 |
| 6 layers | 1,574,400 |
| Final RMSNorm | 128 |
| **Total** | **2,098,816** |
| Non-embedding | 1,574,528 |

## 4. Training

- **Loss:** cross-entropy on supervised tokens only.
- **Optional first-token weighting:** `first_token_weight` multiplies the loss
  on the first command token of every packed example. The focused core run uses
  5.0 to emphasize command-family selection; the broad run uses the default 1.0.
  This is a diagnostic knob, not a replacement for better data or capacity.
- **Optimizer:** AdamW with decoupled weight decay. Matrices get weight decay
  0.1; norms, biases, and embeddings do not.
- **Schedule:** linear warmup for 200 steps, then cosine decay from 3e-4 to 3e-5.
- **Batch:** 64 chunks × 256 tokens = 16,384 tokens per optimizer step.
- **Gradient clipping:** global norm 1.0.
- **Precision:** float32 on CPU. The model is small enough that fp32 is fast and
  numerically simple; bf16 is not universally fast on ARM CPUs.
- **Checkpointing:** best validation loss, last periodic snapshot, and final
  snapshot. Validation uses a held-out prompt style, not just held-out slots.
- **Reproducibility:** seed 42, deterministic page splits and variant seeds, and
  environment metadata saved in `env.json`.

The measured throughput on the target 4-vCPU Neoverse-N1 machine is roughly
3–4k tokens/second, so one 9.16M-token epoch takes about 40–50 minutes.

## 5. Decoding

Evaluation and the CLI use greedy decoding (`temperature=0`): pick the most
likely next token, stop at `<|eos|>` or after 64 tokens. Greedy decoding is
deterministic and avoids sampling noise in comparisons. The CLI also exposes
temperature and top-k for experimentation.

Generated text is cleaned conservatively: strip special tokens, remove wrapping
backticks or a leading `$`, keep the first non-empty line, and normalize
whitespace. Scoring never grants semantic equivalence; a different but valid
command counts as an exact-match error.

## 6. Why this fits on a CPU

- 2.1M parameters × 4 bytes ≈ 8.4 MB of weights.
- One 256-token forward/backward pass is small enough for cache-resident matrix
  multiplies.
- Document-aligned packing keeps padding to about 14%, and batch 64 keeps the
  math libraries busy.
- The vocabulary is 4k instead of 32k+, shrinking the embedding and output
  matrices, which dominate parameter counts at this scale.
- The task's outputs are short, so 256 tokens of context and 64 new tokens are
  sufficient.

The tradeoff is capacity: 2.1M parameters can learn command patterns and slot
copying, but not general shell expertise. That is why evaluation includes
unseen-command splits and why the model card emphasizes human review.

## 7. Where to look in the code

| Concept | File |
|---|---|
| tldr parsing, filtering, slot filling, prompt styles | `src/tinyshell/data.py` |
| BPE training and special tokens | `src/tinyshell/tokenizer.py` |
| Prompt masking and packing | `src/tinyshell/dataset.py` |
| RMSNorm, RoPE, attention, SwiGLU, generation | `src/tinyshell/model.py` |
| Training loop and scheduler | `src/tinyshell/train.py` |
| Metrics, report, baseline | `src/tinyshell/evaluate.py`, `baseline.py` |
| CLI and packaging | `src/tinyshell/cli.py`, `generate.py`, `package.py` |
