from pathlib import Path

import pytest
import torch

from tinyshell.config import Config
from tinyshell.generate import generate_command, load_model
from tinyshell.model import TinyLM

TOKENIZER_DIR = Path("artifacts/tokenizer")


@pytest.fixture()
def tiny_run(tmp_path: Path) -> Path:
    if not (TOKENIZER_DIR / "tokenizer.json").exists():
        pytest.skip("trained tokenizer artifact is not available")
    cfg = Config.from_yaml(Path("configs/smoke.yaml"))
    cfg.model.d_model = 32
    cfg.model.n_layers = 1
    cfg.model.n_heads = 4
    cfg.model.intermediate_size = 64
    cfg.model.max_seq_len = 32
    cfg.model.dropout = 0.0
    cfg.data.tokenizer_dir = str(TOKENIZER_DIR)
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    cfg.save(run / "config.yaml")
    torch.manual_seed(0)
    model = TinyLM(cfg.model)
    torch.save({"model": model.state_dict()}, run / "checkpoints" / "best.pt")
    return run


def test_load_model_from_run_directory(tiny_run: Path) -> None:
    model, tokenizer, cfg, run_dir = load_model(tiny_run)
    assert run_dir == tiny_run
    assert model.num_parameters == TinyLM(cfg.model).num_parameters
    assert tokenizer.get_vocab_size() == 4096
    assert not model.training


def test_load_model_from_checkpoint_file(tiny_run: Path) -> None:
    _model, _tokenizer, _cfg, run_dir = load_model(tiny_run / "checkpoints" / "best.pt")
    assert run_dir == tiny_run


def test_generate_command_returns_structured_result(tiny_run: Path) -> None:
    model, tokenizer, cfg, _run_dir = load_model(tiny_run)
    result = generate_command(model, tokenizer, cfg, "list files", max_new_tokens=4)
    assert set(result) == {
        "request",
        "command",
        "raw",
        "valid_bash",
        "risk",
        "risk_reasons",
        "latency_ms",
        "generated_tokens",
    }
    assert result["request"] == "list files"
    assert 1 <= result["generated_tokens"] <= 4
    assert result["risk"] in {"safe", "caution", "destructive"}
