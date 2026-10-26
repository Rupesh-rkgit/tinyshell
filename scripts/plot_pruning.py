"""Paper figures for the pruning / interpretability study.

Reads ``artifacts/analysis/<run>/pruning.json`` (from ``scripts/prune_analysis.py``)
and writes vector PDFs to ``paper/figures/``.

Usage:
    python scripts/plot_pruning.py --analysis artifacts/analysis/tinyshell-xl-10m/pruning.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

# Validated categorical order (fixed slots, never cycled) + secondary encoding.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
DIV = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#1c5cab", "#86b6ef", "#f0efec", "#f0a3a2", "#c22f2e"]
)
METHOD_ORDER = ["random", "magnitude", "activation", "fisher", "fisher_global"]
LABELS = {
    "random": "Random (uniform)",
    "magnitude": "Magnitude (uniform)",
    "activation": "Activation×weight (uniform)",
    "fisher": "Fisher (uniform)",
    "fisher_global": "Fisher (global)",
    "seen_specific": "Seen-specific units",
    "unseen_specific": "Unseen-specific units",
    "most_important_seen": "Most important (seen)",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.edgecolor": INK_2,
            "axes.labelcolor": INK,
            "axes.linewidth": 0.6,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "lines.linewidth": 1.4,
            "lines.markersize": 4,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def curve(rows, method, key):
    pts = [(r["sparsity"], key(r)) for r in rows if r["method"] == method]
    pts = [(x, y) for x, y in pts if y is not None]
    return [p[0] for p in pts], [p[1] for p in pts]


def gen_exact(split):
    return lambda r: r.get(split, {}).get("exact")


def pruning_curves(d, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.4), sharex="col")
    dense_loss = d["dense"]["loss_seen"]["loss"]
    dense_exact = d["dense"]["test_seen"]["exact"]
    seen_loss = lambda r: r["loss_seen"]["loss"]  # noqa: E731
    panels = [
        (axes[0, 0], "mlp", seen_loss, dense_loss, "Seen-page loss (nats)", "(a) MLP neurons"),
        (axes[1, 0], "mlp", gen_exact("test_seen"), dense_exact, "test_seen exact match", None),
        (axes[0, 1], "heads", seen_loss, dense_loss, None, "(b) Attention heads"),
        (axes[1, 1], "heads", gen_exact("test_seen"), dense_exact, None, None),
    ]
    for ax, kind, key, dense, ylabel, title in panels:
        for i, m in enumerate(METHOD_ORDER):
            xs, ys = curve(d[kind], m, key)
            if xs:
                ax.plot([0.0] + xs, [dense] + ys, color=SERIES[i], marker=MARKERS[i], label=LABELS[m])
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title, loc="left", fontsize=9, color=INK)
    for ax in axes[0]:
        ax.set_yscale("log")
        ax.axhline(dense_loss, color=INK_2, lw=0.6, ls=":")
    for ax in axes[1]:
        ax.set_ylim(-0.02, max(0.6, dense_exact + 0.05))
        ax.axhline(dense_exact, color=INK_2, lw=0.6, ls=":")
        ax.set_xlabel("Fraction of units removed")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.04), fontsize=7)
    fig.savefig(out / "pruning_curves.pdf")
    plt.close(fig)


def logit_lens(d, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.3, 2.4))
    for split, ls in (("seen", "-"), ("unseen", "--")):
        rows = d["logit_lens"][split]
        layers = [r["layer"] for r in rows]
        for j, (key, name) in enumerate((("first_acc", "first token"), ("arg_acc", "argument tokens"))):
            ys = [r[key] for r in rows]
            ax.plot(layers, ys, color=SERIES[j], ls=ls, marker=MARKERS[j], label=f"{name}, {split}")
            ax.annotate(
                f"{ys[-1]:.2f}", (layers[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                fontsize=6.5, color=INK_2, va="center",
            )
    ax.set_xlabel("Residual stream after layer (0 = embeddings)")
    ax.set_ylabel("Top-1 accuracy of decoded token")
    ax.set_xticks(range(0, len(d["logit_lens"]["seen"])))
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6.5, loc="upper left")
    fig.savefig(out / "logit_lens.pdf")
    plt.close(fig)


def head_maps(d, out: Path) -> None:
    heads = d["ablations"]["heads"]
    n_layers = max(h["layer"] for h in heads) + 1
    n_heads = max(h["head"] for h in heads) + 1
    first = np.zeros((n_layers, n_heads))
    arg = np.zeros((n_layers, n_heads))
    for h in heads:
        first[h["layer"], h["head"]] = h["first_loss"]
        arg[h["layer"], h["head"]] = h["arg_loss"]
    copy = np.array(d["copy_scores"])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    for ax, mat, title, cmap, fmt in (
        (axes[0], first, "(a) Δ first-token loss", DIV, "{:.2f}"),
        (axes[1], arg, "(b) Δ argument loss", DIV, "{:.2f}"),
        (axes[2], copy, "(c) Copy score", SEQ, "{:.2f}"),
    ):
        if cmap is DIV:
            lim = max(abs(mat.min()), abs(mat.max()), 1e-3)
            im = ax.imshow(mat, cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
        else:
            im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=max(mat.max(), 1e-3), aspect="auto")
        # Direct-label only the three strongest cells; the colorbar carries the rest.
        top = np.argsort(np.abs(mat).flatten())[-3:]
        for idx in top:
            r, c = divmod(int(idx), n_heads)
            ax.text(c, r, fmt.format(mat[r, c]).replace("0.", "."), ha="center", va="center", fontsize=5,
                    color="#ffffff" if abs(mat[r, c]) > 0.6 * np.abs(mat).max() else INK)
        ax.set_title(title, loc="left", fontsize=8.5, color=INK)
        ax.set_xlabel("Head")
        ax.set_xticks(range(n_heads))
        ax.set_yticks(range(n_layers))
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=6)
        cb.outline.set_linewidth(0.4)
    axes[0].set_ylabel("Layer")
    fig.tight_layout()
    fig.savefig(out / "head_maps.pdf")
    plt.close(fig)


def sublayers(d, out: Path) -> None:
    rows = d["ablations"]["sublayers"]
    n_layers = max(r["layer"] for r in rows) + 1
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.2), sharey=True)
    width = 0.38
    for ax, key, title in ((axes[0], "first_loss", "(a) Δ first-token loss"), (axes[1], "arg_loss", "(b) Δ argument loss")):
        for j, kind in enumerate(("attn", "mlp")):
            ys = [next(r[key] for r in rows if r["layer"] == layer and r["kind"] == kind) for layer in range(n_layers)]
            ax.bar(np.arange(n_layers) + (j - 0.5) * width, ys, width=width - 0.04,
                   color=SERIES[j], label="attention sublayer" if kind == "attn" else "MLP sublayer")
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_title(title, loc="left", fontsize=8.5, color=INK)
        ax.set_xlabel("Layer ablated")
        ax.set_xticks(range(n_layers))
    axes[0].set_ylabel("Loss increase (nats)")
    axes[0].legend(fontsize=6.5)
    fig.savefig(out / "sublayer_ablation.pdf")
    plt.close(fig)


def differential(d, out: Path) -> None:
    rows = d["differential"]
    methods = ["random", "most_important_seen", "seen_specific", "unseen_specific"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))
    for ax, key, title in (
        (axes[0], "loss_seen", "(a) Seen pages (val)"),
        (axes[1], "loss_unseen", "(b) Unseen pages (test_unseen)"),
    ):
        base = d["dense"][key]["loss"]
        for i, m in enumerate(methods):
            xs, ys = curve(rows, m, lambda r, key=key: r[key]["loss"])
            ax.plot([0.0] + xs, [base] + ys, color=SERIES[i], marker=MARKERS[i], label=LABELS[m])
        ax.axhline(base, color=INK_2, lw=0.6, ls=":")
        ax.set_title(title, loc="left", fontsize=8.5, color=INK)
        ax.set_xlabel("Fraction of MLP neurons removed")
    axes[0].set_ylabel("Loss (nats)")
    axes[0].set_yscale("log")
    axes[1].legend(fontsize=6.5)
    fig.savefig(out / "differential.pdf")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", default="artifacts/analysis/tinyshell-xl-10m/pruning.json")
    parser.add_argument("--out", default="paper/figures")
    args = parser.parse_args(argv)
    d = json.loads(Path(args.analysis).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    style()
    pruning_curves(d, out)
    logit_lens(d, out)
    head_maps(d, out)
    sublayers(d, out)
    differential(d, out)
    print(f"Wrote figures to {out}")


if __name__ == "__main__":
    main()
