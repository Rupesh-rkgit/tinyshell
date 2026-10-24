#!/usr/bin/env python3
"""Compare two TinyShell checkpoints on the same held-out requests."""

from __future__ import annotations

import argparse
from pathlib import Path

from tinyshell.evaluate import score_row
from tinyshell.generate import generate_command, load_model
from tinyshell.utils import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", default=None, help="Optional markdown output path")
    args = parser.parse_args()

    path = Path("data/processed") / f"{args.split}.jsonl"
    rows = list(read_jsonl(path))[: args.limit]
    models = []
    for name in (args.run_a, args.run_b):
        model, tokenizer, cfg, run_dir = load_model(name)
        models.append((run_dir.name, model, tokenizer, cfg))

    lines = [
        f"# Run comparison: {models[0][0]} vs {models[1][0]}",
        "",
        f"Split: `{args.split}`, first {len(rows)} examples. Greedy decoding.",
        "",
        "| # | Request | Reference | " + models[0][0] + " | " + models[1][0] + " |",
        "|---:|---|---|---|---|",
    ]
    scores = {name: {"exact": 0, "command_match": 0, "token_f1": 0.0} for name, *_ in models}
    for index, row in enumerate(rows, start=1):
        outputs = []
        for name, model, tokenizer, cfg in models:
            result = generate_command(model, tokenizer, cfg, row["request"])
            scored = score_row(result["command"], row["command"])
            outputs.append(result["command"])
            scores[name]["exact"] += int(scored["exact"])
            scores[name]["command_match"] += int(scored["command_match"])
            scores[name]["token_f1"] += float(scored["token_f1"])
        request = row["request"].replace("|", "\\|")
        reference = row["command"].replace("|", "\\|")
        lines.append(
            f"| {index} | {request} | `{reference}` | `{outputs[0]}` | `{outputs[1]}` |"
        )

    lines += ["", "## Subset scores", "", "| Run | Exact | Command | Token F1 |", "|---|---:|---:|---:|"]
    for name, values in scores.items():
        n = max(1, len(rows))
        lines.append(
            f"| `{name}` | {values['exact']/n:.3f} | {values['command_match']/n:.3f} | "
            f"{values['token_f1']/n:.3f} |"
        )
    text = "\n".join(lines) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
