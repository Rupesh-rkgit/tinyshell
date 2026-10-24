"""Package a trained run into a self-contained artifacts/model directory."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Any

from .config import Config
from .utils import ensure_dir, read_json, resolve_run_dir, write_json


def package_run(run: str | Path = "latest", output: str | Path = "artifacts/model") -> Path:
    run_dir = resolve_run_dir(run)
    cfg = Config.from_yaml(run_dir / "config.yaml")
    out = ensure_dir(output)

    ckpt = run_dir / "checkpoints" / "best.pt"
    if not ckpt.exists():
        ckpt = run_dir / "checkpoints" / "final.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No checkpoint in {run_dir}")
    shutil.copy2(ckpt, out / "model.pt")

    for name in ("config.yaml", "model_summary.json", "final_metrics.json"):
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, out / name)

    tokenizer_src = run_dir / "tokenizer"
    tokenizer_out = ensure_dir(out / "tokenizer")
    for name in ("tokenizer.json", "tokenizer_config.json", "stats.json", "vocab_preview.txt"):
        src = tokenizer_src / name
        if src.exists():
            shutil.copy2(src, tokenizer_out / name)

    eval_dir = Path("artifacts/eval") / run_dir.name
    if eval_dir.exists():
        for src_name, dst_name in (
            ("report.md", "eval_report.md"),
            ("metrics.json", "eval_metrics.json"),
        ):
            src = eval_dir / src_name
            if src.exists():
                shutil.copy2(src, out / dst_name)

    model_card = Path("MODEL_CARD.md")
    if model_card.exists():
        shutil.copy2(model_card, out / "MODEL_CARD.md")
    data_card = Path("DATA_CARD.md")
    if data_card.exists():
        shutil.copy2(data_card, out / "DATA_CARD.md")
    experiments = Path("docs/EXPERIMENTS.md")
    if experiments.exists():
        shutil.copy2(experiments, out / "EXPERIMENTS.md")

    model_path = out.as_posix()
    (out / "README.md").write_text(
        "# TinyShell packaged model\n\n"
        f"Run: `{run_dir.name}`\n\n"
        "Usage:\n\n"
        "```bash\n"
        f'tinyshell --model {model_path} "list all files including hidden ones"\n'
        'tinyshell --retrieve "stream pod logs. Assume the pod name is alice"\n'
        "```\n\n"
        "The model only generates a command; it never executes it. Review commands "
        "before running them. See `MODEL_CARD.md`, `eval_report.md`, and "
        "`EXPERIMENTS.md`.\n",
        encoding="utf-8",
    )

    summary_path = run_dir / "model_summary.json"
    parameters = read_json(summary_path)["parameters"] if summary_path.exists() else None
    manifest: dict[str, Any] = {
        "run": run_dir.name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": str(ckpt),
        "parameters": parameters,
        "model_config": cfg.model.__dict__,
        "training_data": "tldr-pages (CC-BY-4.0)",
        "files": sorted(p.name for p in out.iterdir() if p.is_file()),
    }
    write_json(out / "manifest.json", manifest)
    print(f"Packaged {run_dir.name} -> {out}")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Package a TinyShell run")
    parser.add_argument("--run", default="latest")
    parser.add_argument("--output", default="artifacts/model")
    args = parser.parse_args(argv)
    package_run(args.run, args.output)


if __name__ == "__main__":
    main()
