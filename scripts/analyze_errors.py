#!/usr/bin/env python3
"""Summarize common failure modes from TinyShell prediction files."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_dir", nargs="?", default="latest")
    parser.add_argument("--split", default="test_seen")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    if args.eval_dir == "latest":
        candidates = sorted(
            Path("artifacts/eval").glob("*/metrics.json"),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            raise SystemExit("No evaluation metrics found under artifacts/eval")
        eval_dir = candidates[-1].parent

    path = eval_dir / f"{args.split}_predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    def name(command: str) -> str:
        tokens = command.split()
        return tokens[0] if tokens else "(empty)"

    confusions = collections.Counter(
        (name(row["reference"]), name(row["prediction"])) for row in rows if not row["command_match"]
    )
    empty = sum(row["empty"] for row in rows)
    invalid = sum(not row["valid_bash"] for row in rows)
    risk = collections.Counter(row["risk"] for row in rows)
    exact = sum(row["exact"] for row in rows)
    right_command = sum(row["command_match"] for row in rows) - exact
    predicted = collections.Counter(name(row["prediction"]) for row in rows)

    print(f"# Failure analysis: {eval_dir.name} / {args.split}\n")
    print(f"- Examples: {len(rows)}")
    print(f"- Exact: {exact} ({exact / len(rows):.1%})")
    print(f"- Right command, wrong arguments: {right_command} ({right_command / len(rows):.1%})")
    print(f"- Empty: {empty} ({empty / len(rows):.1%})")
    print(f"- Invalid bash: {invalid} ({invalid / len(rows):.1%})")
    print(f"- Risk: {dict(risk)}\n")
    print("## Most common predicted commands\n")
    print("| Command | Count |")
    print("|---|---:|")
    for command, count in predicted.most_common(10):
        print(f"| `{command}` | {count} |")
    print("\n## Most common command confusions (reference -> prediction)\n")
    print("| Reference | Predicted | Count |")
    print("|---|---|---:|")
    for (ref, pred), count in confusions.most_common(args.top):
        print(f"| `{ref}` | `{pred}` | {count} |")
    print("\n## Sample errors\n")
    shown = 0
    for row in rows:
        if row["exact"] or shown >= args.top:
            continue
        print(f"- **Request:** {row['request']}")
        print(f"  **Reference:** `{row['reference']}`")
        print(f"  **Prediction:** `{row['prediction'] or '(empty)'}`")
        print()
        shown += 1


if __name__ == "__main__":
    main()
