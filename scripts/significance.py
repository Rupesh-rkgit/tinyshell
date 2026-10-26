"""Confidence intervals for model and BM25 metrics, and for their paired difference.

Reports, per split:

- Wilson 95% intervals for exact and command match (model and BM25);
- paired bootstrap 95% intervals for model - BM25 differences in exact match,
  command match, and token F1, resampling either examples or whole tldr pages.
  Page-clustered resampling is the honest one: examples of one page share a
  command and are not independent draws.

Model predictions are read from ``<eval-dir>/<split>_predictions.jsonl``
(written by ``tinyshell.evaluate``); BM25 is recomputed per example.

Usage:
    python scripts/significance.py --eval-dir artifacts/eval/tinyshell-xl-10m \
        --data-dir data/processed-xl
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from tinyshell.baseline import build_baseline
from tinyshell.evaluate import score_row
from tinyshell.utils import read_jsonl, write_json

SPLITS = ["val", "val_unseen", "test_seen", "test_paraphrase", "test_unseen", "test_unseen_paraphrase"]
METRICS = ["exact", "command_match", "token_f1"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(centre - half, 4), round(centre + half, 4))


def bootstrap(diff: np.ndarray, groups: np.ndarray | None, rng, n_boot: int) -> tuple[float, float]:
    """Percentile interval of the mean of ``diff``, resampling items or groups."""
    if groups is None:
        idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
        means = diff[idx].mean(1)
    else:
        uniq, inverse = np.unique(groups, return_inverse=True)
        sums = np.bincount(inverse, weights=diff)
        counts = np.bincount(inverse).astype(float)
        pick = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
        means = sums[pick].sum(1) / counts[pick].sum(1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (round(float(lo), 4), round(float(hi), 4))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", default="artifacts/eval/tinyshell-xl-10m")
    parser.add_argument("--data-dir", default="data/processed-xl")
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    eval_dir = Path(args.eval_dir)
    retriever = build_baseline(Path(args.data_dir) / "train.jsonl")
    rng = np.random.default_rng(args.seed)
    result: dict[str, dict] = {}
    lines = [
        "| Split | Model exact [95% CI] | BM25 exact [95% CI] | Δ token F1 [item CI] | Δ token F1 [page CI] |",
        "|---|---|---|---|---|",
    ]
    for split in SPLITS:
        path = eval_dir / f"{split}_predictions.jsonl"
        if not path.exists():
            continue
        rows = list(read_jsonl(path))
        model = {m: np.array([float(r[m]) for r in rows]) for m in METRICS}
        bm25_scored = []
        for r in rows:
            hits = retriever.search(r["request"], top_k=1)
            bm25_scored.append(score_row(hits[0]["command"] if hits else "", r["reference"]))
        bm25 = {m: np.array([float(s[m]) for s in bm25_scored]) for m in METRICS}
        pages = np.array([r.get("page") or "?" for r in rows])
        n = len(rows)
        entry: dict[str, dict] = {"n": n, "pages": int(len(set(pages)))}
        for m in METRICS:
            diff = model[m] - bm25[m]
            entry[m] = {
                "model": round(float(model[m].mean()), 4),
                "bm25": round(float(bm25[m].mean()), 4),
                "delta": round(float(diff.mean()), 4),
                "delta_ci_items": bootstrap(diff, None, rng, args.n_boot),
                "delta_ci_pages": bootstrap(diff, pages, rng, args.n_boot),
            }
            if m != "token_f1":
                entry[m]["model_wilson"] = wilson(int(model[m].sum()), n)
                entry[m]["bm25_wilson"] = wilson(int(bm25[m].sum()), n)
        # Discordant pairs for exact match (McNemar-style counts).
        entry["exact_discordant"] = {
            "model_only": int(((model["exact"] == 1) & (bm25["exact"] == 0)).sum()),
            "bm25_only": int(((model["exact"] == 0) & (bm25["exact"] == 1)).sum()),
        }
        result[split] = entry
        e, f = entry["exact"], entry["token_f1"]
        lines.append(
            f"| `{split}` | {e['model']:.3f} [{e['model_wilson'][0]:.3f}, {e['model_wilson'][1]:.3f}] | "
            f"{e['bm25']:.3f} [{e['bm25_wilson'][0]:.3f}, {e['bm25_wilson'][1]:.3f}] | "
            f"{f['delta']:+.3f} [{f['delta_ci_items'][0]:+.3f}, {f['delta_ci_items'][1]:+.3f}] | "
            f"{f['delta']:+.3f} [{f['delta_ci_pages'][0]:+.3f}, {f['delta_ci_pages'][1]:+.3f}] |"
        )
        print(lines[-1], flush=True)
    write_json(eval_dir / "significance.json", result)
    (eval_dir / "significance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {eval_dir / 'significance.json'}")


if __name__ == "__main__":
    main()
