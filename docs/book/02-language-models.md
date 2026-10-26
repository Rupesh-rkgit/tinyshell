# Chapter 2 — Language Models from Scratch: The Theory You Need

This chapter builds the mathematical machinery TinyShell uses, from conditional
probability to cross-entropy loss, byte-level BPE, and why a tiny 4,096-token
vocabulary is a feature at this scale. Chapter 3 then covers the transformer
architecture itself. Everything here is standard; the value is in connecting
each abstraction to a concrete decision made in the repository.

## 2.1 Language modeling as conditional probability

A language model assigns probabilities to sequences of tokens. For a sequence
$x_1, x_2, \dots, x_T$, the chain rule factors the joint probability into
next-token predictions:

$$
P(x_1, \dots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \dots, x_{t-1}).
$$

Training a neural language model means fitting a function
$f_\theta(\text{prefix}) \to \mathbb{R}^{|V|}$ (a distribution over the
vocabulary $V$) so that each factor of that product is predicted well.
*Inference* then means sampling or argmaxing from
$f_\theta(\cdot)$ one token at a time, appending each chosen token to the
prefix.

TinyShell is a **conditional** LM: the prefix always begins with a fixed
prompt that contains the user's request, and the distribution we care about is

$$
P(\text{command} \mid \text{request}).
$$

Two details of our setup follow immediately from this framing:

- **Loss masking.** We compute the training loss only on the tokens we want to
  model (the command and its end-of-sequence marker). The system prompt and the
  user request are conditions, not targets. In `dataset.py` their label
  positions are filled with `-100`, PyTorch's `ignore_index` for cross-entropy,
  so they contribute neither loss nor gradient.
- **The EOS token is part of the task.** The model must learn *when the command
  ends*; `<|eos|>` is a supervised target like any other token. Greedy decoding
  stops when the model emits it. An LM that cannot end its output is useless in
  a CLI.

## 2.2 Cross-entropy loss and perplexity

Given logits $z \in \mathbb{R}^{|V|}$ at one position, the model's distribution
is $\text{softmax}(z)$. The training loss is cross-entropy against the true
next token $y$:

$$
\mathcal{L} = -\log \frac{e^{z_y}}{\sum_{v} e^{z_v}}.
$$

Cross-entropy has a useful interpretation: it is the negative log-likelihood the
model assigns to the correct token, averaged over positions. Lower is better,
and the scale is intuitive:

- $\mathcal{L} = \ln |V|$ is the loss of a uniform guess (for
  $|V|=4{,}096$: **8.32 nats** — almost exactly where run 3 started, 8.24 at
  step 10).
- $\mathcal{L} = \log_2 |V|$ is the "choose the right word out of the
  vocabulary" entropy in bits.
- $e^{\mathcal{L}}$ is **perplexity**: the effective number of equally likely
  choices the model is choosing among. Run 3 finished at val loss 0.6565, i.e.
  perplexity ≈ 1.93 — on average fewer than two live options per supervised
  position on commands it has seen.

Two normalization subtleties matter for honest comparisons:

1. **Token-weighted averaging.** Because many positions are masked (`-100`),
   the loss must be averaged over *active* tokens only. `TinyLM.forward` uses
   `F.cross_entropy(..., ignore_index=-100)`, which does exactly this.
2. **Length effects.** A loss computed over a corpus of short commands is not
   comparable to one computed over long documents. All loss numbers in this
   book are over the same packed validation set, so run-to-run comparisons are
   valid; absolute comparisons to other projects' losses are not.

## 2.3 Tokens and byte-level BPE

Neural models consume integers, so text must be segmented into vocabulary
items. The three classic options:

- **Word-level**: every word is a token. Simple, but the vocabulary explodes
  and any unseen word becomes `<unk>` — fatal for a shell task where filenames
  are arbitrary.
- **Character-level**: tiny vocabulary, no unseen items, but sequences get
  long (context cost) and each token carries very little meaning.
- **Subword-level (BPE)**: start from characters and iteratively merge the
  most frequent adjacent pairs. Common words become single tokens; rare words
  decompose into pieces. This is the standard.

TinyShell uses **byte-level BPE** (`tokenizers` library, `models.BPE` with
`ByteLevel` pre-tokenization and decoding):

1. The base alphabet is the **256 byte values**, so *every* string — any
   Unicode, any filename — is representable. There is no `<unk>` and no
   round-trip failure. The tokenizer training code literally verifies
   `decode(encode(text)) == text` on a sample of the corpus and records
   `roundtrip_failures` (it was 0).
2. The `ByteLevel` pre-tokenizer splits text so that whitespace attaches to the
   following word (` al` for " al"), matching GPT-2 conventions.
3. Merges are learned on the rendered training text (`render_training_text` =
   full prompt + command + EOS), so token boundaries align with the exact
   strings the model will see.

**Why 4,096 and not 32,000?** The vocabulary size sets two parameter blocks:
the embedding matrix and the output projection. With tied weights (Section
3.6) that is $|V| \times d$ parameters. For the 2.1M model:
$4{,}096 \times 128 = 524{,}288$ parameters — **25% of the entire model**.
At 32k tokens the embedding alone would be 4M parameters, dwarfing everything
else. The cost of a small vocabulary is longer sequences and more tokens per
rare string; the measured corpus statistics make the trade concrete:

| Statistic (v1 corpus) | Value |
|---|---|
| Median example length | 67 tokens |
| p99 example length | 119 tokens |
| Max example length | 137 tokens (v1) / 229 (XL corpus) |
| Chars per token (sample mean) | ≈ 3.4 |

Everything the model needs fits comfortably in 256 positions. The XL corpus
(rare longer examples) drops the handful of examples exceeding the window
(`skipped_too_long` in the packing stats).

**Special tokens.** Five ids are reserved at the bottom of the vocabulary:
`<|pad|>` (0), `<|eos|>` (1), `<|system|>` (2), `<|user|>` (3),
`<|assistant|>` (4). They are inserted as `special_tokens` during BPE training
so their ids are stable, and they are never produced by merges.

## 2.4 The chat template

Every example is rendered into a fixed three-role template before
tokenization:

```text
<|system|>
You are a Linux shell command assistant. Convert the request into exactly one command. Reply with the command only, no explanation.
<|user|>
{request}
<|assistant|>
{command}<|eos|>
```

The template is *string-identical* at training and inference time
(`text.render_prompt`). This matters more than it looks: a template drift
between training and serving silently degrades the model. Keeping the template
in one function used by training, evaluation, and the CLI removes that failure
mode. The system prompt is fixed — this project has no multi-turn dialogue, no
few-shot examples, and no tool results; the entire interaction is
request-in, command-out.

## 2.5 Optimization in one section

Training minimizes the average masked cross-entropy over packed batches with
**AdamW** (Loshchilov & Hutter, 2019). The pieces worth understanding:

- **AdamW vs SGD+momentum.** Adam keeps per-parameter first and second moment
  estimates, giving each parameter an effectively normalized step. For
  transformers this is not optional; vanilla SGD is unstable or very
  slow to converge at these depths.
- **Decoupled weight decay.** Classic Adam applies L2 through the adaptive
  denominator, which regularizes large-gradient parameters *less* — the
  opposite of intent. AdamW decouples the decay step from the adaptive step:
  `p ← p − lr·λ·p` then the Adam update. TinyShell applies decay **only to
  matrices** (2-D parameters, excluding the tied embedding); norms, biases, and
  embeddings decay 0 (`configure_optimizer` in `model.py`).
- **β = (0.9, 0.95).** The second-moment discount is lower than sklearn's
  default 0.999 because transformer gradients are spiky; 0.95 forgets spikes
  faster. This follows GPT-2/Llama practice.
- **Warmup + cosine decay.** The learning rate ramps linearly from ~0 over
  `warmup_steps` (200 for 2.1M runs, 300 for run 3), then follows
  $$
  \text{lr}(t) = \text{lr}_{\min} + \tfrac{1}{2}(\text{lr}_{\max} -
  \text{lr}_{\min})(1 + \cos(\pi \cdot \text{progress}))
  $$
  from 3e-4 down to 3e-5. Warmup avoids the early-training blow-up when
  Adam's moment estimates are still noisy; the cosine tail anneals into a
  sharp minimum. Run 3's log shows the schedule working: lr 1e-5 at step 10,
  3.30e-5 at step 3430, 3.00e-5 flat at the end.
- **Gradient clipping** at global norm 1.0 (`grad_clip`) bounds any single
  batch's damage. Run 3's logged grad norms sit at 0.8–1.5 for most of
  training — the clip is active at the edge, which is normal and harmless.
- **Precision.** Everything is float32. On a 4-vCPU ARM64 machine, bf16/fp16
  support is patchy and the model is small enough that fp32 matmuls keep the
  cores busy; half precision would save memory the machine has plenty of.

## 2.6 What "learning" looks like at this scale

Run 3's validation trajectory (from `artifacts/runs/tinyshell-xl-10m/metrics.jsonl`)
is worth internalizing, because it is what real training curves look like:

| Step | Val loss | Perplexity |
|---:|---:|---:|
| 250 | 4.522 | 92.0 |
| 500 | 3.396 | 29.8 |
| 1,000 | 2.199 | 9.0 |
| 2,000 | 1.199 | 3.32 |
| 3,000 | 0.788 | 2.20 |
| 3,500 | 0.6565 | 1.93 |

Three observations that recur throughout this book:

1. The curve is **steep early, long flat late**: most of the learning happens
   in the first quarter; the last 500 steps buy ~0.06 nats. Early stopping on
   a noisy plateau would have been wrong here — the end-of-schedule anneal is
   where the final gains came from.
2. Loss ≈ 0.13 on *training* data by the end vs 0.66 on held-out validation:
   the model is memorizing its training distribution to a substantial degree.
   Whether that is "overfitting" depends entirely on which held-out set you
   care about — the central theme of Chapter 13.
3. **Loss is not skill.** Run 1 reached val loss 3.97 with 0% exact match;
   run 2 reached 2.94 with 0% exact match; run 3 reached 0.66 with 52.5%. Loss
   measures next-token fit; only generation measures the task. This is why
   Chapter 9 insists on generation-based evaluation with multiple partial
   credit metrics.

## 2.7 The mathematics of generalization (informally)

The seen/unseen gap that dominates the later chapters deserves a theoretical
frame. A conditional LM trained on pairs $D = \{(r_i, c_i)\}$ minimizes
empirical risk on its training distribution. Its behavior on a new request
distribution is determined by what function class it can express and what the
data taught it to rely on:

- **Interpolation** (new slot values for a known command) is easy: copying an
  entity from the context into a template is a local, learnable operation.
- **Re-derivation with held-out phrasings** (paraphrases) is easy *if* the
  training data shows many phrasings of the same intent — which our 12–16
  prompt styles deliberately provide.
- **Composition from unseen parts** (a command never seen in training) requires
  the model to map a description to a *novel* tool's flag vocabulary. Nothing
  in the training distribution supplies those flags, so the Bayes-optimal
  behavior given the data is to fall back on a prior over command tokens —
  which is exactly what the model does (`sudo ...` for obscure admin pages).

This is why "more variants of the same pages" (run 3's data strategy) improves
interpolation dramatically while *worsening* held-out-page loss: every
additional phrasing sharpens the fit to known pages and makes the prior over
unknown ones relatively less useful. Chapter 13 quantifies this with the
seen/unseen loss gap (0.87 nats for run 2 → 3.49 nats for run 3).
