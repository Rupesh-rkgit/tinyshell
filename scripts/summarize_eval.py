#!/usr/bin/env python3
"""Print a compact markdown comparison from an evaluation metrics.json file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="?", default="latest")
    args = parser.parse_args()

    if args.metrics == "latest":
        candidates = sorted(
            Path("artifacts/eval").glob("*/metrics.json"),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            raise SystemExit("No evaluation metrics found under artifacts/eval")
        path = candidates[-1]
    else:
        path = Path(args.metrics)
    data = json.loads(path.read_text(encoding="utf-8"))
    baseline = data.get("baseline") or {}

    print(f"# {data['run']} ({data['parameters']:,} parameters)\n")
    print("| Split | N | Model exact | BM25 exact | Model cmd | BM25 cmd | Model F1 | BM25 F1 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for split, m in data["splits"].items():
        b = baseline.get(split, {})
        print(
            f"| {split} | {m.get('n', 0)} | {m.get('exact_match', 0):.3f} | "
            f"{b.get('exact_match', 0):.3f} | {m.get('command_match', 0):.3f} | "
            f"{b.get('command_match', 0):.3f} | {m.get('token_f1', 0):.3f} | "
            f"{b.get('token_f1', 0):.3f} |"
        )
    print()
    print("| Split | Valid bash | Empty | Risk (safe/caution/destructive) | p50 ms |")
    print("|---|---:|---:|---|---:|")
    for split, m in data["splits"].items():
        risk = m.get("risk", {})
        print(
            f"| {split} | {m.get('valid_bash', 0):.3f} | {m.get('empty_rate', 0):.3f} | "
            f"{risk.get('safe', 0)}/{risk.get('caution', 0)}/{risk.get('destructive', 0)} | "
            f"{m.get('latency_ms', {}).get('p50', 0):.0f} |"
        )


if __name__ == "__main__":
    main()
