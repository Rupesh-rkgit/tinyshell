# TinyShell evaluation: `tinyshell-xl-10m`

- Checkpoint: `best`
- Model: 256d x 10L, 4096 vocab, 256 context
- Decoding: greedy (temperature=0.0, top_k=1)

## Results

| Split | N | Exact | Command | Token F1 | Valid bash | Latency p50 |
|---|---:|---:|---:|---:|---:|---:|
| val | 1499 | 0.460 | 0.618 | 0.604 | 0.997 | 290 ms |
| val_unseen | 1500 | 0.011 | 0.138 | 0.315 | 0.995 | 227 ms |
| test_seen | 1500 | 0.525 | 0.692 | 0.644 | 0.997 | 245 ms |
| test_paraphrase | 1500 | 0.559 | 0.722 | 0.681 | 0.997 | 242 ms |
| test_unseen | 1500 | 0.005 | 0.111 | 0.289 | 0.989 | 229 ms |
| test_unseen_paraphrase | 1500 | 0.005 | 0.098 | 0.292 | 0.991 | 242 ms |

`test_seen` = seen commands + new slots; `test_paraphrase` = seen commands + unseen prompt styles; `test_unseen` = unseen commands; `test_unseen_paraphrase` = both unseen.

## Retrieval baseline (BM25)

| Split | N | Exact | Command | Token F1 | Valid bash | Latency p50 |
|---|---:|---:|---:|---:|---:|---:|
| val | 1499 | 0.851 | 0.932 | 0.947 | 1.000 | 0 ms |
| val_unseen | 1500 | 0.009 | 0.135 | 0.286 | 0.999 | 0 ms |
| test_seen | 1500 | 0.843 | 0.918 | 0.918 | 1.000 | 0 ms |
| test_paraphrase | 1500 | 0.877 | 0.942 | 0.964 | 0.999 | 0 ms |
| test_unseen | 1500 | 0.007 | 0.124 | 0.277 | 1.000 | 0 ms |
| test_unseen_paraphrase | 1500 | 0.008 | 0.114 | 0.291 | 1.000 | 0 ms |

The baseline returns the command of the most similar training request. It is strong on seen commands and weak on unseen commands.

### Per-command: test_seen

| Command | N | Exact | Command match |
|---|---:|---:|---:|
| `proctl` | 3 | 0.000 | 0.333 |
| `eselect news` | 3 | 0.000 | 0.333 |
| `asnmap` | 3 | 0.333 | 0.667 |
| `openai` | 3 | 0.333 | 1.000 |
| `mumble` | 3 | 0.333 | 1.000 |
| `pdftotext` | 3 | 0.333 | 1.000 |
| `obsidian` | 3 | 0.667 | 0.667 |
| `npm pkg` | 3 | 0.667 | 0.667 |
| `nest` | 3 | 0.667 | 0.667 |
| `lid` | 3 | 0.667 | 1.000 |

### Per-command: test_unseen

| Command | N | Exact | Command match |
|---|---:|---:|---:|
| `hatch` | 3 | 0.000 | 0.000 |
| `install-tl` | 3 | 0.000 | 0.000 |
| `krunvm` | 3 | 0.000 | 0.000 |
| `rustscan` | 3 | 0.000 | 0.000 |
| `wait4x` | 3 | 0.000 | 0.000 |
| `gh attestation` | 3 | 0.000 | 0.000 |
| `wayland-info` | 3 | 0.000 | 0.000 |
| `d2` | 3 | 0.000 | 0.000 |
| `conda init` | 3 | 0.000 | 0.000 |
| `rdesktop` | 3 | 0.000 | 0.000 |

## Notes

- Exact match is strict: flags, quoting, and paths must match the reference.
- `valid_bash` only checks syntax with `bash -n`; it does not check semantics.
- Predictions are never executed.
- Failure examples are in `<split>_failures.jsonl`; all predictions are in `<split>_predictions.jsonl`.
