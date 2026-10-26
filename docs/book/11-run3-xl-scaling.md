# Chapter 11 — Run 3: Scaling to 9.6M Parameters and 100M Tokens

> Run ID: `tinyshell-xl-10m` · 9,573,632 params · 1 pass over 99,992,950
> packed tokens · 468,212 chunks · 3,657 steps · 27.7 h (99,690 s) ·
> ~1,190 tokens/s · best val **0.6565** (step 3,500)
> Config: `configs/tinyshell-xl.yaml` · Artifacts:
> `artifacts/runs/tinyshell-xl-10m/`, `artifacts/eval/tinyshell-xl-10m/`

Run 3 is the shipped model and the project's main scaling experiment. It
changes capacity, data volume, and data quality *simultaneously* — a choice
made deliberately: after run 2 and the core experiment, every diagnostic
pointed at "more of the same proven design," so the interesting question was
not which knob matters (that is what ablations are for) but how far the
proven design scales.

## 11.1 What changed

| Axis | Run 2 | Run 3 |
|---|---|---|
| `d_model` × layers × heads | 128 × 6 × 4 | **256 × 10 × 8** |
| Head dim (per head) | 32 | 32 (deliberately unchanged) |
| SwiGLU intermediate | 512 | 768 |
| Parameters | 2,098,816 | **9,573,632** (4.6×) |
| Dropout | 0.05 | 0.05 (more data → same light regularization) |
| Variants per example | 6 | **150** |
| Train styles | 8 | 12 (adds indices 12–15; eval styles 8–11 untouched) |
| Corpus filter | v1 rules | + cron-schedule first token, leading `!`, process substitution (−139 examples) |
| Training tokens | 9.16M × 4 epochs | **99.99M × 1 pass** |
| Requests | 135,860 | **1,303,083** |
| Chunks | 41,743 | **468,212** |
| Batch | 64 × 256 | 128 × 256 (32,768 positions/step) |
| Warmup / steps | 200 / 2,608 | 300 / 3,657 |

Unchanged: tokenizer (vocab 4,096; byte fallback covers new slot values),
page-level splits, document-aligned packing, `first_token_weight` = 1.0,
AdamW settings, cosine schedule shape, seed 42, fp32 on 4 vCPU.

Not directly comparable to run 2 as a whole (the corpus changed), but the
page split is identical, so eval requests overlap heavily and directional
comparisons are meaningful — a caveat repeated in every results table.

## 11.2 Training as it actually happened

From `train.log` and `metrics.jsonl`:

- **Throughput:** steady ~1,190 tokens/s (32,768 positions/step), 4 threads,
  no restarts beyond one planned resume; 27 h 41 m wall clock.
- **Loss trajectory (val on held-out style 8, seen pages):**

| Step | Val loss | PPL |
|---:|---:|---:|
| 250 | 4.522 | 92.0 |
| 500 | 3.396 | 29.8 |
| 1,000 | 2.199 | 9.0 |
| 2,000 | 1.199 | 3.32 |
| 3,000 | 0.788 | 2.20 |
| **3,500 (best)** | **0.6565** | **1.93** |

- **Final:** val 0.6571 at step 3,657; train loss ~0.13–0.15; grad norms
  0.8–1.5 throughout; lr annealed to 3.00e-5 exactly on schedule.
- **Best checkpoint:** step 3,500 — the cosine tail's last big eval gain.
  `final.pt` is statistically identical on val (0.6571).

**The number that matters most: `val_unseen` loss 4.1451 — *worse* than run
2's 3.7993.** The seen/unseen loss gap widened from **0.87 nats (run 2)** to
**3.49 nats (run 3)**. Mechanism: 150 variants per page means each *page*
appears hundreds of times with fresh phrasings and slot fills; the model
hyper-fits its known command families while the marginal information about
*unseen* pages goes down. More data made interpolation better and
extrapolation *relatively worse*. This single number is why the eval tables
below must be read split-by-split, never averaged.

## 11.3 Generation evaluation

Checkpoint `best`, greedy decoding, 1,500 per split
(`artifacts/eval/tinyshell-xl-10m/`):

| Split | N | Exact | Command | Token F1 | Valid bash | p50 |
|---|---:|---:|---:|---:|---:|---:|
| `val` | 1499 | 0.460 | 0.618 | 0.604 | 0.997 | 290 ms |
| `val_unseen` | 1500 | 0.011 | 0.138 | 0.315 | 0.995 | 227 ms |
| `test_seen` | 1500 | **0.525** | **0.692** | 0.644 | 0.997 | 245 ms |
| `test_paraphrase` | 1500 | **0.559** | **0.722** | 0.681 | 0.997 | 242 ms |
| `test_unseen` | 1500 | 0.005 | 0.111 | 0.289 | 0.989 | 229 ms |
| `test_unseen_paraphrase` | 1500 | 0.005 | 0.098 | 0.292 | 0.991 | 242 ms |

BM25 on the identical examples:

| Split | BM25 exact | BM25 command | BM25 token F1 | Model exact | Δ exact |
|---|---:|---:|---:|---:|---:|
| `val` | 0.851 | 0.932 | 0.947 | 0.460 | −0.392 |
| `val_unseen` | 0.009 | 0.135 | 0.286 | 0.011 | +0.001 |
| `test_seen` | 0.843 | 0.918 | 0.918 | 0.525 | −0.317 |
| `test_paraphrase` | 0.877 | 0.942 | 0.964 | 0.559 | −0.317 |
| `test_unseen` | 0.007 | 0.124 | 0.277 | 0.005 | −0.003 |
| `test_unseen_paraphrase` | 0.008 | 0.114 | 0.291 | 0.005 | −0.003 |

Averaged over seen splits: BM25 exact **0.857** vs model **0.515**. On
unseen splits: exact ~0.008 vs ~0.005 — both fail exact match, but the
**token F1 comparison flips in the model's favor for the first time**
(0.289/0.292/0.315 vs 0.277/0.291/0.286), and command match is ahead on
`val_unseen` (0.138 vs 0.135). The tiny generative model now *composes*
roughly at parity with the retriever on novel pages, instead of trailing it
on every metric as run 2 did (F1 0.170–0.174 vs 0.248–0.286 there).

## 11.4 The frontier moved in three ways

1. **Seen-command composition works.** Exact 0.46–0.56 (run 2: ≤0.002),
   token F1 0.60–0.68 (run 2: ~0.18). Paraphrase robustness held: the
   *strongest* split is `test_paraphrase` (0.559) — 16 training phrasings
   made unseen phrasings a non-event.
2. **Unseen-page F1 ≥ retrieval.** The compositional advantage a generative
   model is supposed to have is finally measurable, if small.
3. **Within-tool confusions dominate the residue.** The seen splits' errors
   are mostly sibling-tool near-misses (`phpcs` ↔ `phpcbf`, `7zr` ↔ `7za`,
   `bun` ↔ `npm`).

## 11.5 The error census on `test_seen`

From the predictions file (1,500 examples):

| Outcome | Count | Share |
|---|---:|---:|
| Exact match | 788 | 52.5% |
| Right command, wrong arguments | 250 | 16.7% |
| Invalid bash | 4 | 0.3% |
| Empty generation | 0 | 0.0% |

**The `sudo` prior persists in attenuated form:** 14.7% of `test_seen`
predictions begin with `sudo` vs 5.5% of references (run 2: 42%). The
over-prediction concentrates on obscure admin pages (`zeroclaw`, `pveum`,
`koji`, `pg_config`) where root access is *plausible* — a rational prior
applied at the wrong time, and a warning sign that must be surfaced to users
(the CLI's caution patterns do flag `sudo`).

Representative failures, quoted from the failure log:

```text
reference: git changelog --start-tag 0.5.0 --final-tag 1.0.0
prediction: git changelog --filter 5:1.0.0 --list          # right family, invented flags

reference: zeroclaw agent
prediction: hangman                                        # unseen page → random command

reference: mysql_secure_installation --help
prediction: sudo rkhhunter --help                          # obscure page → sudo + wrong tool
```

## 11.6 What run 3 establishes

- **Scale (4.6× params, 10.9× tokens) converted the broad model from
  "usually the wrong tool" (≤2% exact) to a competitive suggestion engine on
  seen commands (53–56% exact with unseen phrasings).**
- **The unresolved axis is page-level generalization**, not decoding, not
  family selection, not packing: 0.5% exact on unseen pages, with the
  fallback prior (`sudo` + wrong tool) exactly where theory says a
  data-limited conditional LM should land.
- **More variants ≠ more knowledge.** The val_unseen loss regression
  (3.80 → 4.15) shows that phrasing diversity sharpens known-page fit; only
  *page diversity* (new tools/flags in training) or external knowledge
  (retrieval) moves unseen pages.
- The practical split stands: `tinyshell --retrieve` for commands known to
  be in the corpus (BM25: 0.84–0.88 exact, 0 ms); the neural path for
  composing *modifications* of known commands and for unseen-page behavior
  at parity with retrieval — plus a real path forward (RAG, Chapter 14).
