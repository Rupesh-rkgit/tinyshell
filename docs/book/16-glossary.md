# Chapter 16 — Glossary

Quick definitions of every term of art used in this book, in the specific
sense TinyShell uses them.

**Active (supervised) tokens** — positions whose label ≠ `-100`; they
contribute loss and receive gradients. ≈14% of packed positions in the XL
corpus (17,411,259 tokens).

**AdamW** — Adam with *decoupled* weight decay (Loshchilov & Hutter, 2019).
TinyShell applies decay 0.1 to matrix weights only; norms/biases/embeddings
decay 0.

**Base example** — one parsed tldr description + command template pair
(28,659 usable in v1).

**BM25 baseline** — retriever over training requests (k1 1.5, b 0.75) that
returns the nearest training request's command. Near-oracle on seen splits,
~0.8% on unseen pages.

**Block-diagonal causal mask** — attention mask restricting each token to
earlier tokens *of its own document* inside a packed row
(`TinyLM.document_mask`). Identical to plain causal masking at inference.

**Byte-level BPE** — byte-pair encoding whose base alphabet is all 256 byte
values; no unknown tokens, exact round trips. TinyShell vocab: 4,096.

**Chunks** — fixed-length (256-token) training rows containing one or more
complete examples plus padding. XL train: 468,212.

**Command match** — metric: first executable token of prediction equals
reference's, skipping wrappers like `sudo`/`env`.

**Core experiment** — 2.1M model trained on the 655-page / 111-first-token
subset with `first_token_weight` 5.0; separated label-space limits from
capacity limits.

**Cross-entropy (nats)** — training loss; uniform guess = ln 4096 ≈ 8.32;
run 3 best val = 0.6565 (perplexity 1.93).

**doc_ids** — per-token id of the packed example a token belongs to; −1 on
padding. Runtime input enabling the block-diagonal mask and mask ablations.

**Document-aligned packing** — packing complete examples only, with per-token
doc ids and within-example positions (run 2's fix; ~14% padding cost).

**Dropout** — residual + attention dropout (0.05 broad/XL; 0.1 core),
disabled at eval.

**EOS (`<|eos|>`)** — supervised end-of-command token; generation stops on
it. Special ids 0–4: pad, eos, system, user, assistant.

**Exact match** — normalized prediction string equals reference. Strict: no
semantic equivalence. Seen-split run 3: 0.525–0.559; unseen: 0.005.

**First-token weighting** — multiply the first command token's loss by
`first_token_weight` (5.0 in core run) to emphasize command-family selection.

**Greedy decoding** — argmax next token; deterministic; temperature 0;
stops at EOS or 64 new tokens.

**GQA** — grouped-query attention; *not* used here (8 full heads).

**Packing shortcut** — run 1's learned dependence on previous packed
examples' commands; proven by prefix probe and eval-time mask ablation
(3.866 → 3.306).

**Paraphrase splits** — `test_paraphrase`, `test_unseen_paraphrase`: prompt
styles 10–11, never seen in training.

**Perplexity (PPL)** — exp(cross-entropy); effective choice count per
position.

**Position ids (per-example)** — within-example positions (restart at 0 per
document), fed to RoPE; stored at packing time.

**Risk assessment** — regex classification of generated commands into
safe / caution / destructive for CLI warnings; explicitly not a security
boundary.

**RMSNorm** — LayerNorm without mean/bias; pre-norm placement.

**RoPE** — rotary position embeddings; rotates q/k pairs by
position-dependent angles so attention scores depend on relative offsets;
θ = 10,000; zero learned parameters.

**SDPA** — PyTorch fused `scaled_dot_product_attention` used for the
attention core.

**Slot filling** — replacing tldr `{{placeholders}}` with pool values (file,
path, user, host, port, …) and alternative syntax `{{[-a|--all]}}`; fresh
values per variant.

**Styles (prompt styles)** — 16 request templates; train pool 0–7 (+12–15 in
XL), val 8, test_seen 9, paraphrase 10–11. Held-out styles are a generalization
axis.

**SwiGLU** — gated MLP: `down(silu(gate(x)) * up(x))`; intermediate 512
(2.1M) / 768 (9.6M).

**tldr-pages** — CC-BY-4.0 community corpus of simplified man pages; the
project's sole data source.

**Token F1** — multiset F1 over `shlex` arguments (command name excluded);
partial credit metric.

**Tied embeddings** — `lm_head.weight = embed.weight`; saves |V|×d
parameters (1.05M on the 9.6M model).

**Unseen pages** — evaluation on command pages excluded from training by the
page-level hash split; the composition axis of generalization.

**val / val_unseen** — packed-loss validation sets: seen pages + held-out
style 8; unseen pages + seen styles. Checkpoint selection uses `val`.

**Variants per example** — number of rendered (request, command) pairs per
base example (6 in v1; 150 in XL; every variant has fresh slot values and a
cycled style).

**Weight tying** — see tied embeddings.

**`bash -n` validity** — syntax-only parse check of a generated command;
never executed. 99.7–99.9% across all runs.
