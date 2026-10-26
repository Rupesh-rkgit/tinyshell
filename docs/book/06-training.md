# Chapter 6 — The Training Loop: Turning the Corpus into a Model

`src/tinyshell/train.py` (455 lines) is a from-scratch training loop. Nothing
in it is exotic; everything in it matters. This chapter explains each moving
piece as implemented, using run 3's 27.7-hour run as the running example.

## 6.1 What the Trainer holds

Constructed from a `Config` and a run directory, the `Trainer` owns:

- the tokenizer (loaded from `artifacts/tokenizer/`) and its `eos_id`;
- the `TinyLM` model (built from `cfg.model`, fp32, moved to `cpu`);
- the AdamW optimizer from `model.configure_optimizer` (decay on matrices
  only);
- three packed datasets: `train`, `val`, `val_unseen` (memory-mapped arrays
  from Chapter 5);
- a `LambdaLR` implementing warmup + cosine decay;
- run bookkeeping: `run_dir`, `checkpoints/`, `train.log`,
  `metrics.jsonl`, `samples.jsonl`, config/env snapshots.

Step arithmetic: `micro_per_epoch = len(train_ds) // batch_size`; with
`grad_accum_steps=1`, `total_steps = micro_per_epoch × epochs`. Run 3:
468,212 chunks / batch 128 = 3,657 steps for the single pass — exactly the
final logged step count. The 2.1M runs: 41,743 / 64 = 652 steps per epoch ×
4 = 2,608 steps.

## 6.2 The LR schedule

`_lr_lambda` computes the multiplier directly:

```python
warmup = max(1, min(cfg.warmup_steps, total_steps // 5))
if step < warmup:
    return step / warmup                       # linear ramp
ratio = min_lr / lr
progress = (step - warmup) / (total_steps - warmup)
return ratio + (1 - ratio) * 0.5 * (1 + cos(pi * progress))   # cosine
```

Two defensive details: warmup is capped at 20% of training (a smoke run with
200 configured warmup steps and 30 total steps would otherwise never leave
warmup), and progress is clamped to [0, 1] so any step overshoot saturates at
`min_lr`. Run 3 used warmup 300 of 3,657 steps and ended exactly at
lr = 3.0e-5 — visible in the final log lines.

## 6.3 The step loop

Per optimizer step:

1. Pull a batch of `(input_ids, labels, doc_ids, position_ids)` rows from the
   DataLoader (`shuffle=True` for train).
2. Shift: inputs are `x[:, :-1]`, targets `y[:, 1:]` — next-token prediction
   with no separate "target" corpus. (Doc ids and positions are sliced
   identically so masks/positions stay aligned with the shifted stream.)
3. `logits, loss = model(x, y, doc_ids=doc_ids, position_ids=position_ids)`
   — the block-diagonal mask is built inside forward from `doc_ids`.
4. `loss.backward()` → `clip_grad_norm_(model.parameters(), grad_clip)` →
   `optimizer.step()` → `scheduler.step()` → `optimizer.zero_grad()`.

Run 3's step is 128 × 256 = **32,768 positions**, ~27,300 of them real tokens;
the loop held a steady **~1,190 tokens/s** on 4 threads for 27.7 hours
(throughput sagged to ~1,080 tok/s during the step-3500 eval window, then
recovered — the log shows it).

## 6.4 Periodic work: eval, samples, checkpoints

Every `eval_interval` steps (250 for run 3) the loop:

- **Evaluates loss** on the packed `val` split (`evaluate_loss`): forward-only
  batches, loss weighted by each batch's active token count, aggregated
  globally — the same masked, token-weighted cross-entropy as training, so
  numbers are comparable across steps and runs.
- **Evaluates `val_unseen` loss** (logged at best-checkpoint time and at the
  end; this is the number that *rose* from 3.80 to 4.15 between run 2 and
  run 3 — the headline generalization datum).
- **Generates samples**: the first 5 rows of `samples.jsonl` go through the
  full inference path (prompt → generate → clean → risk check) and are
  appended to `samples.jsonl`. Reading these every few hundred steps was the
  fastest way to notice run 1's shortcut (`sudo aa-cli --list`) long before
  the formal evaluation.
- **Logs and checkpoints**: `best.pt` whenever val improves; `last.pt` every
  `save_interval` (500); `final.pt` at the end.

## 6.5 Checkpoints: what is inside, and why it matters

`save_checkpoint` writes one torch file containing:

```python
{
  "model":      model.state_dict(),
  "optimizer":  optimizer.state_dict(),
  "scheduler":  scheduler.state_dict(),
  "step": ..., "epoch": ..., "batch_in_epoch": ...,
  "best_val": ..., "config": cfg.to_dict(),
}
```

This is a **resumable** checkpoint: `load_checkpoint` restores weights, Adam
moments, the schedule position, and the progress counters, so a killed run
continues seamlessly (run 3 was resumed at least once mid-training; the log's
elapsed clock continues across the gap). It is also *heavy*: optimizer state
doubles the parameter count (two fp32 moment tensors), which is why run 3's
`best.pt` is 115 MB for a 38 MB model. (How that was handled for distribution
is in Chapter 17.)

`best.pt` selection uses **`val` — held-out prompt style 8 on seen pages** —
not a random val sample. That choice encodes an opinion: the checkpoint we
want is the one that generalizes across phrasings of known commands. For run
3 that was step 3,500 (val 0.6565), just before the final anneal bottomed out
— the final model (val 0.6571) is statistically identical, and `best` remains
the shipped checkpoint.

## 6.6 Run metadata: the reproducibility record

At trainer construction (`_write_run_metadata`), the run directory receives:

| File | Content |
|---|---|
| `config.yaml` | the full, resolved config (never overwritten on resume; a `config_resume.yaml` records changes) |
| `env.json` | Python 3.12.3, torch 2.14.1+cpu, numpy 2.5.4, aarch64, 4 threads, git commit, dirty flag, UTC timestamp |
| `model_summary.json` | 9,573,632 total / 8,525,056 non-embedding parameters + the model config |
| `git_commit.txt` | `81bdf02…` — the code state the run was launched from |
| `tokenizer/` | a copy of the exact tokenizer used |
| `README.md` | parameter/chunk/step summary |

With the config, the env, the commit, and the tokenizer pinned per run, every
result in this book can be tied to the exact code and data that produced it.
(One caveat recorded in `docs/EXPERIMENTS.md`: run 3 launched at commit
`81bdf02`, which was later rewritten to `2b88264` during a git-hygiene fix
that removed accidentally-committed large files; code content was unchanged.)

## 6.7 First-token loss weighting (the diagnostic knob)

`TinyLM.forward(first_token_weight=...)` implements a targeted loss reweight:
the *first command token* of every packed example gets its per-token
cross-entropy multiplied by the weight. It is off (1.0) everywhere except the
core experiment (5.0). The motivation: with 3,651 possible first tokens,
uniform token weighting under-represents the single decision that matters
most — which tool to call. The implementation identifies "first command
token" as an active-label position whose predecessor is inactive
(`active & ~previous_active`), which correctly handles multiple examples per
chunk (Chapter 3's weighted-loss code) — and, as `docs/EXPERIMENTS.md`
records honestly, the shipped core run predates this fix and tagged only the
first active token *per chunk* (~1/3 of examples), so the core result is a
conservative estimate of the weighting's effect.

## 6.8 What the logs say about run 3

A compact narrative from `train.log` / `metrics.jsonl`:

- **Steps 1–300 (warmup):** loss 8.24 → 7.6 — the uniform-guess regime
  (ln 4096 = 8.32) collapsing as the model learns unconditional command-token
  frequencies.
- **Steps 300–2,000:** the steep descent — 7.6 → 1.2 val (perplexity 2,000× →
  3.3×) — the model learns command families, flag syntax, and slot copying.
- **Steps 2,000–3,500:** the long grind, 1.20 → 0.656 val (ppl 3.3 → 1.93);
  train loss reaches ~0.13, i.e. the model can near-perfectly reproduce
  training commands — memorization territory, as Chapter 13 discusses.
- **Grad norms** 0.8–1.5 throughout; token/s steady at ~1,190; no spikes, no
  instabilities, no restarts (beyond resume).

The full six-split generation evaluation of the resulting checkpoint is
Chapter 13's subject.
