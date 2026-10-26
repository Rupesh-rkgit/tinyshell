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
