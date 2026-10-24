"""Typed configuration objects loaded from YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


def _build(cls: type, data: dict[str, Any]) -> Any:
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


@dataclass
class DataConfig:
    source_dir: str = "data/raw/tldr"
    processed_dir: str = "data/processed"
    tokenizer_dir: str = "artifacts/tokenizer"
    seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    variants_per_example: int = 6
    eval_examples: int = 1500
    max_command_chars: int = 120
    min_description_chars: int = 8
    max_description_chars: int = 160

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        if self.variants_per_example < 1:
            raise ValueError("variants_per_example must be >= 1")
        if self.max_command_chars < 20:
            raise ValueError("max_command_chars is too small")


@dataclass
class ModelConfig:
    vocab_size: int = 4096
    d_model: int = 128
    n_layers: int = 6
    n_heads: int = 4
    intermediate_size: int = 512
    max_seq_len: int = 256
    rope_theta: float = 10_000.0
    dropout: float = 0.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.d_model < 16:
            raise ValueError("d_model must be >= 16")
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if self.max_seq_len < 16:
            raise ValueError("max_seq_len must be >= 16")


@dataclass
class TrainConfig:
    batch_size: int = 32
    grad_accum_steps: int = 1
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    first_token_weight: float = 1.0
    warmup_steps: int = 200
    epochs: int = 4
    eval_interval: int = 250
    sample_interval: int = 250
    save_interval: int = 1000
    log_interval: int = 10
    num_workers: int = 0
    num_threads: int = 4
    device: str = "cpu"
    output_dir: str = "artifacts/runs"
    max_steps: int | None = None
    early_stopping_patience: int = 5

    def __post_init__(self) -> None:
        if self.batch_size < 1 or self.grad_accum_steps < 1:
            raise ValueError("batch_size and grad_accum_steps must be >= 1")
        if self.lr <= 0 or self.min_lr < 0:
            raise ValueError("Learning rates must be positive")
        if self.min_lr > self.lr:
            raise ValueError("min_lr must be <= lr")
        if self.first_token_weight < 1.0:
            raise ValueError("first_token_weight must be >= 1.0")
        if self.epochs < 1 and self.max_steps is None:
            raise ValueError("epochs must be >= 1 when max_steps is not set")


@dataclass
class EvalConfig:
    batch_size: int = 1
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_k: int = 1
    splits: list[str] | None = None
    num_failures: int = 50
    max_examples: int | None = None

    def __post_init__(self) -> None:
        if self.splits is None:
            self.splits = [
                "val",
                "val_unseen",
                "test_seen",
                "test_paraphrase",
                "test_unseen",
                "test_unseen_paraphrase",
            ]
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")


@dataclass
class Config:
    run_name: str
    seed: int
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    eval: EvalConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        known = {"run_name", "seed", "data", "model", "train", "eval"}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}")
        return cls(
            run_name=raw.get("run_name", "tinyshell"),
            seed=int(raw.get("seed", 42)),
            data=_build(DataConfig, raw.get("data", {})),
            model=_build(ModelConfig, raw.get("model", {})),
            train=_build(TrainConfig, raw.get("train", {})),
            eval=_build(EvalConfig, raw.get("eval", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
