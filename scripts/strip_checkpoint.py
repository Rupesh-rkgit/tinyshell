"""Strip optimizer/scheduler state from a TinyShell checkpoint.

Full checkpoints store model + optimizer + scheduler + progress so that long
runs can be resumed (see Chapter 6 of docs/book).  For distribution, the
model weights are all that is needed: the CLI, evaluator, and packaging code
read ``ckpt["model"]``.  This script writes a small ``*-model.pt`` file with
the state dict plus enough metadata to identify the run.

Usage:
    python scripts/strip_checkpoint.py artifacts/runs/tinyshell-xl-10m/checkpoints/best.pt \
        [--out best-model.pt]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch


def strip_checkpoint(src: str | Path, out: str | Path | None = None) -> Path:
    src = Path(src)
    if out is None:
        out = src.with_name(f"{src.stem}-model.pt")
    out = Path(out)
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    if "model" not in ckpt:
        raise ValueError(f"{src} does not look like a TinyShell checkpoint")
    payload = {
        "model": ckpt["model"],
        "step": ckpt.get("step"),
        "best_val": ckpt.get("best_val"),
        "config": ckpt.get("config"),
        "stripped_from": src.name,
        "stripped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    torch.save(payload, out)
    src_mb = src.stat().st_size / 1e6
    out_mb = out.stat().st_size / 1e6
    print(f"{src} ({src_mb:.1f} MB) -> {out} ({out_mb:.1f} MB)")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    strip_checkpoint(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
