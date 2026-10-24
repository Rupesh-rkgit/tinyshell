#!/usr/bin/env python3
"""Insert the latest evaluation tables into README.md and MODEL_CARD.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def model_table(splits: dict) -> str:
    lines = [
        "| Split | N | Exact | Command | Token F1 | Valid bash | p50 latency |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, m in splits.items():
        lines.append(
            f"| `{split}` | {m.get('n', 0)} | {fmt(m.get('exact_match', 0))} | "
            f"{fmt(m.get('command_match', 0))} | {fmt(m.get('token_f1', 0))} | "
            f"{fmt(m.get('valid_bash', 0))} | "
            f"{m.get('latency_ms', {}).get('p50', 0):.0f} ms |"
        )
    return "\n".join(lines)


def baseline_table(splits: dict, baseline: dict) -> str:
    lines = [
        "| Split | BM25 exact | BM25 command | BM25 token F1 | Model exact | Δ exact |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, m in splits.items():
        b = baseline.get(split, {})
        delta = m.get("exact_match", 0) - b.get("exact_match", 0)
        lines.append(
            f"| `{split}` | {fmt(b.get('exact_match', 0))} | "
            f"{fmt(b.get('command_match', 0))} | {fmt(b.get('token_f1', 0))} | "
            f"{fmt(m.get('exact_match', 0))} | {delta:+.3f} |"
        )
    return "\n".join(lines)


def replace_block(text: str, start: str, end: str, body: str) -> str:
    before, sep, rest = text.partition(start)
    if not sep:
        raise SystemExit(f"Missing marker {start}")
    _, sep, after = rest.partition(end)
    if not sep:
        raise SystemExit(f"Missing marker {end}")
    return f"{before}{start}\n{body}\n{end}{after}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        default="latest",
        help="Run id under artifacts/eval, or a path to a metrics.json file",
    )
    args = parser.parse_args()

    if args.run == "latest":
        candidates = sorted(
            Path("artifacts/eval").glob("*/metrics.json"),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            raise SystemExit("No evaluation metrics found under artifacts/eval")
        path = candidates[-1]
    else:
        path = Path(args.run)
        if path.is_dir():
            path = path / "metrics.json"
        elif not path.exists():
            path = Path("artifacts/eval") / args.run / "metrics.json"
    if not path.exists():
        raise SystemExit(f"Metrics not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    splits = data["splits"]
    baseline = data.get("baseline") or {}
    params = data.get("parameters", 0)

    summary = (
        f"Trained model: **{params:,} parameters**, checkpoint `{data['checkpoint']}`. "
        f"Run: `{data['run']}`.\n\n"
        + model_table(splits)
    )
    if baseline:
        seen = [s for s in splits if "unseen" not in s]
        unseen = [s for s in splits if "unseen" in s]

        def mean(keys: list[str], source: dict, field: str) -> float:
            values = [source[k].get(field, 0.0) for k in keys if k in source]
            return sum(values) / len(values) if values else 0.0

        seen_model = mean(seen, splits, "exact_match")
        seen_bm25 = mean(seen, baseline, "exact_match")
        unseen_model = mean(unseen, splits, "exact_match")
        unseen_bm25 = mean(unseen, baseline, "exact_match")
        summary += (
            "\n\n### Model vs BM25 retrieval baseline\n\n"
            + baseline_table(splits, baseline)
            + f"\n\nAveraged over seen-command splits, BM25 exact match is "
            f"**{seen_bm25:.3f}** vs **{seen_model:.3f}** for the model. On "
            f"unseen-command splits both are weak: **{unseen_bm25:.3f}** vs "
            f"**{unseen_model:.3f}**. BM25 returns a memorized training command; "
            "the model can in principle compose a new one, but this 2.1M-parameter "
            "prototype has not learned to do so reliably. Treat the model as an "
            "educational baseline, not a command oracle."
        )

    readme = Path("README.md")
    readme.write_text(
        replace_block(
            readme.read_text(encoding="utf-8"),
            "<!-- RESULTS:START -->",
            "<!-- RESULTS:END -->",
            summary,
        ),
        encoding="utf-8",
    )

    card = Path("MODEL_CARD.md")
    card_body = (
        f"Run `{data['run']}`, checkpoint `{data['checkpoint']}`, {params:,} parameters.\n\n"
        + model_table(splits)
    )
    if baseline:
        card_body += "\n\n### BM25 baseline\n\n" + baseline_table(splits, baseline)
    card.write_text(
        replace_block(
            card.read_text(encoding="utf-8"),
            "<!-- EVAL:START -->",
            "<!-- EVAL:END -->",
            card_body,
        ),
        encoding="utf-8",
    )
    print(f"Updated README.md and MODEL_CARD.md from {path}")


if __name__ == "__main__":
    main()
