"""Training and evaluation plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .utils import ensure_dir, read_jsonl, resolve_run_dir  # noqa: E402


def plot_training(run: str | Path = "latest") -> Path:
    run_dir = resolve_run_dir(run)
    rows = list(read_jsonl(run_dir / "metrics.jsonl"))
    train = [r for r in rows if r.get("split") == "train"]
    val = [r for r in rows if r.get("split") == "val"]
    out = ensure_dir(run_dir / "plots")

    if train:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot([r["step"] for r in train], [r["loss"] for r in train], label="train", alpha=0.7)
        if val:
            ax.plot([r["step"] for r in val], [r["loss"] for r in val], "o-", label="val")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("cross-entropy loss")
        ax.set_yscale("log")
        ax.set_title("TinyShell training loss")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "loss.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot([r["step"] for r in train], [r["lr"] for r in train])
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("learning rate")
        ax.set_title("Learning rate schedule")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "lr.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot([r["step"] for r in train], [r["tokens_per_sec"] for r in train])
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("tokens / second")
        ax.set_title("CPU training throughput")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "throughput.png", dpi=150)
        plt.close(fig)
    return out


def resolve_eval_dir(path: str | Path) -> Path:
    if str(path) == "latest":
        candidates = sorted(
            (p.parent for p in Path("artifacts/eval").glob("*/metrics.json")),
            key=lambda p: p.stat().st_mtime,
        )
        if not candidates:
            raise FileNotFoundError("No evaluation metrics found under artifacts/eval")
        return candidates[-1]
    return Path(path)


def plot_eval(eval_dir: str | Path) -> Path:
    eval_dir = resolve_eval_dir(eval_dir)
    data = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))
    splits = list(data["splits"])
    metrics = data["splits"]
    out = ensure_dir(eval_dir / "plots")

    labels = ["exact_match", "command_match", "valid_bash"]
    width = 0.25
    x = range(len(splits))
    fig, ax = plt.subplots(figsize=(max(9, len(splits) * 1.5), 4.5))
    for offset, label in enumerate(labels):
        values = [metrics[s].get(label, 0) for s in splits]
        positions = [i + (offset - 1) * width for i in x]
        ax.bar(positions, values, width, label=label)
    ax.set_xticks(list(x))
    ax.set_xticklabels(splits, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("TinyShell evaluation by split")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "eval_metrics.png", dpi=150)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate TinyShell plots")
    parser.add_argument("--run", default="latest")
    parser.add_argument(
        "--eval-dir",
        default="latest",
        help="Evaluation directory, or 'latest' (default). Use 'none' to skip.",
    )
    args = parser.parse_args(argv)
    print(f"Training plots: {plot_training(args.run)}")
    if args.eval_dir.lower() != "none":
        try:
            print(f"Eval plots: {plot_eval(args.eval_dir)}")
        except FileNotFoundError as exc:
            print(f"Skipping eval plots: {exc}")


if __name__ == "__main__":
    main()
