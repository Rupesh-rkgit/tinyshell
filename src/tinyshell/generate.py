"""Load a trained checkpoint and generate a command."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from .config import Config
from .evaluate import load_run
from .text import (
    EOS_TOKEN,
    assess_risk,
    clean_generation,
    is_valid_bash,
    render_prompt,
    strip_special_tokens,
)
from .tokenizer import decode, encode, special_id


def load_model(
    run: str | Path = "latest", checkpoint: str = "best"
) -> tuple[Any, Any, Config, Path]:
    """Load a run directory, a packaged model directory, or a checkpoint path."""
    path = Path(run)
    if path.is_file():
        if path.name == "model.pt" and (path.parent / "config.yaml").exists():
            run_dir = path.parent
        else:
            run_dir = path.parent.parent
        return load_run(run_dir, str(path))[:3] + (run_dir,)
    if (path / "model.pt").exists():
        run_dir = path
        return load_run(run_dir, str(path / "model.pt"))[:3] + (run_dir,)
    loaded = load_run(path, checkpoint)
    return loaded[:3] + (loaded[4],)


@torch.inference_mode()
def generate_command(
    model: Any,
    tokenizer: Any,
    cfg: Config,
    request: str,
    max_new_tokens: int | None = None,
    temperature: float = 0.0,
    top_k: int = 1,
) -> dict[str, Any]:
    prompt = render_prompt(request)
    ids = encode(tokenizer, prompt)
    if len(ids) >= cfg.model.max_seq_len:
        ids = ids[-(cfg.model.max_seq_len - 1) :]
    input_ids = torch.tensor([ids], dtype=torch.long)
    eos_id = special_id(tokenizer, EOS_TOKEN)
    started = time.perf_counter()
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens or cfg.eval.max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=eos_id,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    new_ids = output[0, len(ids) :].tolist()
    raw = decode(tokenizer, new_ids)
    command = clean_generation(strip_special_tokens(raw))
    risk = assess_risk(command)
    return {
        "request": request,
        "command": command,
        "raw": raw,
        "valid_bash": is_valid_bash(command),
        "risk": risk.level,
        "risk_reasons": list(risk.reasons),
        "latency_ms": round(latency_ms, 2),
        "generated_tokens": len(new_ids),
    }
