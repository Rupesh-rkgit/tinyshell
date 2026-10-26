# Chapter 5 — Tokenization and Packing: Preparing the Fixed-Length Training Corpus

Between the JSONL splits and the training loop sits a stage that looks like
bookkeeping but decides both the *efficiency* and the *correctness* of
training: tokenizing every example and packing the results into fixed-length
chunks. This chapter follows `src/tinyshell/dataset.py` line by line.

## 5.1 Why pack at all?

Training batches must be rectangular: every row the same length. Examples
here are 20–230 tokens; padding every example to 256 independently would waste
most of the compute on padding. The classic fix is **packing** — concatenating
examples until the 256-token chunk is full. The classic *bug* is what run 1
did: attention flows across example boundaries inside a chunk (Chapter 10).
The TinyShell design keeps the efficiency and removes the bug:

1. pack **complete examples only** (never split one across chunks);
2. record `doc_ids` (which example each token belongs to; `-1` for padding);
3. record per-example `position_ids` (position within the example, from 0);
4. let the model build a **block-diagonal causal mask** from (2) and use (3)
   for RoPE.

## 5.2 Encoding one example

`encode_example` renders the chat template (Chapter 2), tokenizes the prompt
and the completion separately, and builds labels:

```text
prompt_ids  = encode(render_prompt(request))          # condition
comp_ids    = encode(command.strip()) + [eos_id]      # target
input_ids   = prompt_ids + comp_ids
labels      = [-100] * len(prompt_ids) + comp_ids
```

`-100` is `F.cross_entropy`'s `ignore_index`: masked positions neither
contribute loss nor receive gradient. About **14%** of packed positions are
supervised (17.41M of ~120M positions in the XL corpus, including padding) —
the loss is concentrated exactly on what the model must produce.

## 5.3 The packing loop

`pack_rows` is a first-fit sequential packer:

- Keep a current row. For each example: if adding it would exceed 256 tokens,
  flush the row (pad the tail), start a new one.
- **Skip overlong examples** (`len(ids) > seq_len`) and count them
  (`skipped_too_long`) rather than truncating — truncation would put partial
  commands in the data.
- Padding is filled with `pad_id` tokens, `-100` labels, `doc_id = -1`,
  `position 0`. Because `doc_id = -1` never matches any other id, padding can
  never be attended to, even by other padding.

The measured costs of complete-example packing: the v1 corpus grew from
35,800 (naive split-anywhere) to **41,743 chunks** (+16.6%) because chunk tails
are padded; padding is ~14% of positions. Run 3's 468,212 chunks carry
99,992,950 real tokens in 468,212 × 256 = 119.9M positions (**83.4%
utilization**). This is the explicit trade: 14% wasted positions in exchange
for document-aligned attention — and Chapter 10 shows the 14% is cheap at a
factor of ~1,000 relative to the shortcut it prevents.

## 5.4 The arrays on disk

Each split becomes four `numpy` arrays, memory-mapped at train time:

| File | dtype | Shape (train) | Content |
|---|---|---|---|
| `train_ids.npy` | uint16 | (468212, 256) | token ids (4,096 < 65,536 → uint16 saves 4×) |
| `train_labels.npy` | int32 | (468212, 256) | targets, −100-masked |
| `train_doc_ids.npy` | int16 | (468212, 256) | per-token document id |
| `train_position_ids.npy` | int16 | (468212, 256) | per-token within-example position |

`PackedDataset.__getitem__` casts each row to int64 tensors on the fly. The
mmap means the 1.9 GB XL corpus never has to fit in RAM and the OS page cache
does the job of a data loader; with `num_workers=0` there is no worker
process copying pages around — the right choice for a 4-vCPU box where
fork/copy overhead exceeds the trivial indexing cost.

A manifest (`packed/manifest.json`) records per-split stats: examples, real
tokens, padding tokens, chunks, active label tokens, max example length.
`PackedDataset` also tolerates older packing formats (missing doc/position
arrays fall back to one-document-per-chunk), which is how run-1 checkpoints
remained loadable for the ablation in Chapter 10.

## 5.5 Invariants worth testing

`tests/test_dataset.py` pins the properties that the training story depends
on:

- no example is ever split across chunks;
- `doc_ids` are contiguous per example, restart-safe across examples, and −1
  exactly on padding;
- `position_ids` restart at 0 at every document boundary;
- labels are −100 exactly on prompt and padding positions;
- decoding the ids of any example reproduces the source text (byte-level
  round trip through the tokenizer);
- every split's chunk count and token count match the manifest.

When packing behavior is the difference between 0% and working models, these
invariants are not pedantry; they are the safety net that makes later runs
comparable to earlier ones.

## 5.6 Tokenizer training, in detail

`tinyshell-tokenizer` (Chapter 4's corpus in, `artifacts/tokenizer/` out):

1. Writes `tokenizer_corpus.txt`: every train example as
   `render_training_text` (prompt + command + EOS). BPE therefore learns
   merges on *exactly the strings the model will see*, including the fixed
   system prompt (which collapses to a handful of tokens).
2. Trains `models.BPE` with `vocab_size=4096`, `min_frequency=2`,
   `initial_alphabet` = all 256 bytes, and the 5 special tokens reserved
   first (ids 0–4).
3. Verifies every special token exists and byte-level round-trips on a 5,000
   example sample; writes `stats.json` (length percentiles, chars/token,
   round-trip failures = 0) and a human-readable `vocab_preview.txt`.

The tokenizer is trained **once** and reused by every run (v1, core, XL),
with byte fallback guaranteeing new slot values in the XL corpus remain
representable. Changing the tokenizer would change every downstream number;
the pinned `tokenizer.json` inside each run directory is the record of which
one was actually used.

## 5.7 A worked example

Take the page snippet from Chapter 4 with description *"List files including
hidden ones, with human-readable sizes"*, template `ls -lah`, style 0, no
slots:

```text
request:  How do I list files including hidden ones, with human-readable sizes?
```

Rendered and tokenized (illustrative ids):

```text
[2, 351, 512, ...  ]   ◄── system + user prompt      labels: -100 ...
[... 4, 97, 15, 90, 330, 1]  ◄── "ls -lah" + <|eos|>  labels: kept
```

Packed with a second small example in the same chunk:

```text
chunk (256):  [ex1 prompt][ex1 cmd+eos][ex2 prompt][ex2 cmd+eos][pad…]
doc_ids    :  [ 0 0 0 0 0 0 0 0 0 | 1 1 1 1 1 1 1 1 1 | -1 -1 …]
position_ids: [ 0 1 2 3 …      | 0 1 2 3 …            | 0  0 …]
labels     :  [-100 … -100, cmd, 1 | -100 …, cmd, 1 | -100 …]
```

The attention mask lets example 1's tokens see only example 1; example 2's
tokens see only example 2; padding sees nothing. RoPE positions restart per
example. At inference, the row *is* one example — identical geometry. That
identity is the whole point.
