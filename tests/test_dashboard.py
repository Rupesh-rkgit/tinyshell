"""Tests for the training/evaluation dashboard renderer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import dashboard  # noqa: E402


def _make_run_dir(root: Path, name: str = "r-test", complete: bool = False) -> Path:
    run = root / "artifacts" / "runs" / name
    run.mkdir(parents=True, exist_ok=True)
    summary = {
        "parameters": 123456,
        "config": {
            "d_model": 256,
            "n_layers": 10,
            "n_heads": 8,
            "intermediate_size": 768,
            "vocab_size": 4096,
            "max_seq_len": 256,
        },
    }
    (run / "model_summary.json").write_text(json.dumps(summary))
    (run / "env.json").write_text(
        json.dumps(
            {
                "timestamp": "2026-10-24T10:06:06Z",
                "git_commit": "0123456789abcdef",
                "torch": "2.14.1+cpu",
            }
        )
    )
    rows = [
        {"step": 10, "split": "train", "loss": 8.0, "tokens_per_sec": 100.0, "elapsed": 5.0},
        {"step": 20, "split": "train", "loss": 7.0, "tokens_per_sec": 100.0, "elapsed": 10.0},
        {"step": 250, "split": "val", "loss": 4.0},
    ]
    (run / "metrics.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n" + "not json at all\n"
    )
    (run / "train.log").write_text(
        "data: 10 train chunks, 2 val chunks, 5 optimizer steps/epoch, 1200 total steps\n"
    )
    if complete:
        (run / "final_metrics.json").write_text(
            json.dumps({"step": 1200, "best_val": 3.5, "parameters": 123456})
        )
    return run


def _make_eval_dir(root: Path, name: str = "e-test") -> Path:
    eval_dir = root / "artifacts" / "eval" / name
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "run": "r-test",
        "checkpoint": "best",
        "parameters": 123456,
        "generated_at": "2026-10-24T09:00:00",
        "splits": {
            "val": {
                "n": 10,
                "exact_match": 0.5,
                "command_match": 0.866,
                "token_f1": 0.25,
                "valid_bash": 1.0,
            },
            "test_seen": {
                "n": 5,
                "exact_match": 0.25,
                "command_match": 0.75,
                "token_f1": 0.2,
                "valid_bash": 1.0,
            },
        },
        "baseline": {
            "val": {"exact_match": 0.7, "command_match": 1.0, "token_f1": 0.9},
        },
    }
    (eval_dir / "metrics.json").write_text(json.dumps(metrics))
    return eval_dir


@pytest.fixture()
def dash_root(tmp_path: Path) -> Path:
    _make_run_dir(tmp_path)
    _make_eval_dir(tmp_path)
    return tmp_path


def test_collect_reads_runs_and_metrics(dash_root: Path) -> None:
    state = dashboard.collect(dash_root)
    assert len(state["runs"]) == 1
    run = state["runs"][0]
    assert run["name"] == "r-test"
    assert run["status"] in {"complete", "stopped"}
    assert run["total_steps"] == 1200
    assert run["latest_step"] == 20
    assert run["latest_loss"] == 7.0
    assert run["best_val"] == 4.0
    assert run["sec_per_step"] == pytest.approx(0.5)
    # Not running: no ETA/progress computed.
    assert run["eta_secs"] is None
    assert run["progress_pct"] is None
    assert len(state["evals"]) == 1
    record = state["evals"][0]
    assert record["splits"]["val"]["exact"] == 0.5
    assert record["baseline"]["val"]["exact_match"] == 0.7


def test_collect_status_complete_via_final_metrics(tmp_path: Path) -> None:
    _make_run_dir(tmp_path, name="done-test", complete=True)
    state = dashboard.collect(tmp_path)
    run = state["runs"][0]
    assert run["status"] == "complete"
    assert run["best_val"] == 4.0


def test_render_page_contains_run_and_eval_sections(dash_root: Path) -> None:
    page = dashboard.render_page(dashboard.collect(dash_root))
    assert "<svg" in page  # loss curve rendered from metrics rows
    assert "r-test" in page
    assert "1200" in page
    assert "01234567" in page  # shortened git commit
    assert "Evaluations" in page
    assert "BM25 exact" in page  # baseline columns present
    assert "badge" in page


def test_render_page_handles_empty_root(tmp_path: Path) -> None:
    page = dashboard.render_page(dashboard.collect(tmp_path))
    assert "No training in progress." in page
    assert "No runs found" in page
    assert "No evaluations found" in page


def test_escape_of_log_lines(dash_root: Path) -> None:
    run_dir = dash_root / "artifacts" / "runs" / "r-test"
    (run_dir / "train.log").write_text("<script>alert(1)< /script> ok\n")
    page = dashboard.render_page(dashboard.collect(dash_root))
    assert "<script>alert(1)" not in page
    assert "&lt;script&gt;" in page
