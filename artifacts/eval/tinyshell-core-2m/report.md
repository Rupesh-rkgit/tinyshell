# TinyShell evaluation: `tinyshell-core-2m`

- Checkpoint: `best`
- Model: 128d x 6L, 4096 vocab, 256 context
- Decoding: greedy (temperature=0.0, top_k=1)

## Results

| Split | N | Exact | Command | Token F1 | Valid bash | Latency p50 |
|---|---:|---:|---:|---:|---:|---:|
| val | 187 | 0.048 | 0.866 | 0.227 | 0.952 | 161 ms |
| test_seen | 179 | 0.028 | 0.737 | 0.155 | 0.983 | 161 ms |
| test_paraphrase | 211 | 0.057 | 0.924 | 0.260 | 0.981 | 156 ms |

`test_seen` = seen commands + new slots; `test_paraphrase` = seen commands + unseen prompt styles; `test_unseen` = unseen commands; `test_unseen_paraphrase` = both unseen.

## Retrieval baseline (BM25)

| Split | N | Exact | Command | Token F1 | Valid bash | Latency p50 |
|---|---:|---:|---:|---:|---:|---:|
| val | 187 | 0.674 | 1.000 | 0.878 | 1.000 | 0 ms |
| test_seen | 179 | 0.642 | 0.983 | 0.857 | 1.000 | 0 ms |
| test_paraphrase | 211 | 0.640 | 0.976 | 0.864 | 1.000 | 0 ms |

The baseline returns the command of the most similar training request. It is strong on seen commands and weak on unseen commands.

### Per-command: test_seen

| Command | N | Exact | Command match |
|---|---:|---:|---:|
| `git gui` | 3 | 0.000 | 0.667 |

## Notes

- Exact match is strict: flags, quoting, and paths must match the reference.
- `valid_bash` only checks syntax with `bash -n`; it does not check semantics.
- Predictions are never executed.
- Failure examples are in `<split>_failures.jsonl`; all predictions are in `<split>_predictions.jsonl`.
