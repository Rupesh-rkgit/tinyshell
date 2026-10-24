import json
import os

from tinyshell.evaluate import compute_metrics, score_row
from tinyshell.utils import read_jsonl, resolve_run_dir, write_jsonl


def test_score_row_exact_match() -> None:
    row = score_row("ls   -la", "ls -la")
    assert row["exact"]
    assert row["command_match"]
    assert row["token_f1"] == 1.0
    assert row["valid_bash"]
    assert row["risk"] == "safe"


def test_score_row_flags_destructive() -> None:
    row = score_row("rm -rf /tmp/x", "ls")
    assert not row["exact"]
    assert not row["command_match"]
    assert row["risk"] == "destructive"


def test_compute_metrics() -> None:
    rows = [
        {"exact": True, "command_match": True, "token_f1": 1.0, "valid_bash": True,
         "empty": False, "risk": "safe", "latency_ms": 10.0},
        {"exact": False, "command_match": True, "token_f1": 0.0, "valid_bash": True,
         "empty": False, "risk": "caution", "latency_ms": 20.0},
    ]
    metrics = compute_metrics(rows)
    assert metrics["n"] == 2
    assert metrics["exact_match"] == 0.5
    assert metrics["command_match"] == 1.0
    assert metrics["token_f1"] == 0.5
    assert metrics["risk"] == {"safe": 1, "caution": 1}
    assert metrics["latency_ms"]["p50"] == 15.0


def test_compute_metrics_empty() -> None:
    assert compute_metrics([]) == {"n": 0}


def test_jsonl_round_trip(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    rows = [{"a": 1}, {"b": "héllo"}]
    assert write_jsonl(path, rows) == 2
    assert list(read_jsonl(path)) == rows


def test_resolve_run_dir_latest(tmp_path) -> None:
    runs = tmp_path / "runs"
    for name, mtime in (("old", 1), ("new", 2)):
        ckpt = runs / name / "checkpoints"
        ckpt.mkdir(parents=True)
        (ckpt / "best.pt").write_bytes(b"x")
        (runs / name).touch()
        os.utime(runs / name, (mtime, mtime))
    assert resolve_run_dir("latest", runs_root=runs).name == "new"


def test_metrics_are_json_serializable() -> None:
    metrics = compute_metrics([score_row("ls", "ls") | {"latency_ms": 1.0}])
    json.dumps(metrics)
