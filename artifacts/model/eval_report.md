# TinyShell evaluation: `tinyshell-2m-run2-docmask`

- Checkpoint: `best`
- Model: 128d x 6L, 4096 vocab, 256 context
- Decoding: greedy (temperature=0.0, top_k=1)

## Results

| Split | N | Exact | Command | Token F1 | Valid bash | Latency p50 |
|---|---:|---:|---:|---:|---:|---:|
| val | 1500 | 0.001 | 0.073 | 0.174 | 0.985 | 155 ms |
| val_unseen | 1500 | 0.001 | 0.080 | 0.174 | 0.991 | 133 ms |
| test_seen | 1500 | 0.000 | 0.081 | 0.173 | 0.987 | 151 ms |
| test_paraphrase | 1500 | 0.002 | 0.089 | 0.187 | 0.995 | 140 ms |
| test_unseen | 1500 | 0.000 | 0.069 | 0.170 | 0.995 | 139 ms |
| test_unseen_paraphrase | 1500 | 0.000 | 0.062 | 0.172 | 0.991 | 141 ms |

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

### Per-command: test_seen

| Command | N | Exact | Command match |
|---|---:|---:|---:|
| `pkl` | 3 | 0.000 | 0.000 |
| `dart` | 3 | 0.000 | 0.000 |
| `git gui` | 3 | 0.000 | 0.667 |
| `kdialog` | 3 | 0.000 | 0.000 |
| `asnmap` | 3 | 0.000 | 0.000 |
| `pydocstyle` | 3 | 0.000 | 0.000 |
| `wajig` | 3 | 0.000 | 0.000 |
| `pulumi stack history` | 3 | 0.000 | 0.000 |
| `opam` | 3 | 0.000 | 0.000 |
| `multipass` | 3 | 0.000 | 0.000 |

### Per-command: test_unseen

| Command | N | Exact | Command match |
|---|---:|---:|---:|
| `hatch` | 3 | 0.000 | 0.000 |
| `install-tl` | 3 | 0.000 | 0.000 |
| `netplan` | 3 | 0.000 | 0.000 |
| `krunvm` | 3 | 0.000 | 0.000 |
| `gh attestation` | 3 | 0.000 | 0.000 |
| `wayland-info` | 3 | 0.000 | 0.000 |
| `d2` | 3 | 0.000 | 0.000 |
| `conda init` | 3 | 0.000 | 0.000 |
| `smbmap` | 3 | 0.000 | 0.000 |
| `virt-sparsify` | 3 | 0.000 | 0.000 |

## Notes

- Exact match is strict: flags, quoting, and paths must match the reference.
- `valid_bash` only checks syntax with `bash -n`; it does not check semantics.
- Predictions are never executed.
- Failure examples are in `<split>_failures.jsonl`; all predictions are in `<split>_predictions.jsonl`.
