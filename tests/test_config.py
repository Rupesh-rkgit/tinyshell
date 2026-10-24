from pathlib import Path

import pytest

from tinyshell.config import Config, DataConfig, ModelConfig, TrainConfig


def test_round_trip(tmp_path: Path) -> None:
    cfg = Config.from_yaml(Path("configs/smoke.yaml"))
    path = tmp_path / "config.yaml"
    cfg.save(path)
    loaded = Config.from_yaml(path)
    assert loaded.to_dict() == cfg.to_dict()


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("model:\n  d_model: 32\n  made_up: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown keys"):
        Config.from_yaml(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_ratio": 0.9, "val_ratio": 0.1, "test_ratio": 0.1},
        {"variants_per_example": 0},
    ],
)
def test_invalid_data_config(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        DataConfig(**kwargs)


def test_invalid_model_config() -> None:
    with pytest.raises(ValueError):
        ModelConfig(d_model=30, n_heads=4)


def test_invalid_train_config() -> None:
    with pytest.raises(ValueError):
        TrainConfig(lr=1e-3, min_lr=2e-3)
    with pytest.raises(ValueError):
        TrainConfig(first_token_weight=0.5)
