# TinyShell evaluation: `tinyshell-2m-run2-docmask`

- Checkpoint: `best`
- Model: 128d x 6L, 4096 vocab, 256 context
- Decoding: greedy (temperature=0.0, top_k=1)

## Results

| Split | N | Exact | Command | Token F1 | Valid bash | Latency p50 |
|---|---:|---:|---:|---:|---:|---:|
| val | 187 | 0.000 | 0.358 | 0.144 | 0.984 | 178 ms |

`test_seen` = seen commands + new slots; `test_paraphrase` = seen commands + unseen prompt styles; `test_unseen` = unseen commands; `test_unseen_paraphrase` = both unseen.

## Notes

- Exact match is strict: flags, quoting, and paths must match the reference.
- `valid_bash` only checks syntax with `bash -n`; it does not check semantics.
- Predictions are never executed.
- Failure examples are in `<split>_failures.jsonl`; all predictions are in `<split>_predictions.jsonl`.
