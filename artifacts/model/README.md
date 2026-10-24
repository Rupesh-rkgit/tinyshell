# TinyShell packaged model

Run: `tinyshell-2m-run2-docmask`

Usage:

```bash
tinyshell --model artifacts/model "list all files including hidden ones"
tinyshell --retrieve "stream pod logs. Assume the pod name is alice"
```

The model only generates a command; it never executes it. Review commands before running them. See `MODEL_CARD.md`, `eval_report.md`, and `EXPERIMENTS.md`.
