# TinyShell evaluation: `tinyshell-2m-run1`

- Checkpoint: `best`
- Model: 128d x 6L, 4096 vocab, 256 context
- Decoding: greedy (temperature=0.0, top_k=1)

## Results

| Split | N | Exact | Command | Token F1 | Valid bash | Latency p50 |
|---|---:|---:|---:|---:|---:|---:|
| val | 1500 | 0.000 | 0.065 | 0.108 | 0.995 | 181 ms |
| val_unseen | 1500 | 0.001 | 0.078 | 0.126 | 0.998 | 169 ms |
| test_seen | 1500 | 0.000 | 0.084 | 0.117 | 0.992 | 158 ms |
| test_paraphrase | 1500 | 0.000 | 0.055 | 0.128 | 0.995 | 133 ms |
| test_unseen | 1500 | 0.000 | 0.067 | 0.115 | 0.997 | 126 ms |
| test_unseen_paraphrase | 1500 | 0.000 | 0.047 | 0.113 | 0.996 | 135 ms |

`test_seen` = seen commands + new slots; `test_paraphrase` = seen commands + unseen prompt styles; `test_unseen` = unseen commands; `test_unseen_paraphrase` = both unseen.

## Retrieval baseline (BM25)

| Split | N | Exact | Command | Token F1 | Valid bash | Latency p50 |
|---|---:|---:|---:|---:|---:|---:|
| val | 1500 | 0.601 | 0.916 | 0.811 | 0.999 | 0 ms |
| val_unseen | 1500 | 0.006 | 0.131 | 0.259 | 1.000 | 0 ms |
| test_seen | 1500 | 0.601 | 0.915 | 0.815 | 1.000 | 0 ms |
| test_paraphrase | 1500 | 0.599 | 0.921 | 0.825 | 0.999 | 0 ms |
| test_unseen | 1500 | 0.006 | 0.113 | 0.248 | 1.000 | 0 ms |
| test_unseen_paraphrase | 1500 | 0.006 | 0.111 | 0.254 | 0.999 | 0 ms |

The baseline returns the command of the most similar training request. It is strong on seen commands and weak on unseen commands.

## Notes

- Exact match is strict: flags, quoting, and paths must match the reference.
- `valid_bash` only checks syntax with `bash -n`; it does not check semantics.
- Predictions are never executed.
- Failure examples are in `<split>_failures.jsonl`; all predictions are in `<split>_predictions.jsonl`.
