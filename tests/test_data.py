import random
from pathlib import Path

from tinyshell.config import Config
from tinyshell.data import (
    Example,
    SlotFiller,
    _filter_reason,
    parse_page,
    split_pages,
)


def _cfg() -> Config:
    return Config.from_yaml(Path("configs/smoke.yaml"))


def test_parse_page(tmp_path: Path) -> None:
    page = tmp_path / "linux" / "git-status.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "# git status\n\n"
        "> Show the working tree status.\n\n"
        "- Show the working tree status:\n\n"
        "`git status`\n\n"
        "- Show a short status:\n\n"
        "`git status --short`\n",
        encoding="utf-8",
    )
    examples = parse_page(page)
    assert [e.command_template for e in examples] == ["git status", "git status --short"]
    assert examples[0].page == "git status"
    assert examples[0].platform == "linux"


def test_parse_malformed_page_returns_empty(tmp_path: Path) -> None:
    page = tmp_path / "common" / "empty.md"
    page.parent.mkdir(parents=True)
    page.write_text("", encoding="utf-8")
    assert parse_page(page) == []


def _example(command: str = "ls -la {{path}}", description: str = "List files") -> Example:
    return Example(
        page="ls",
        platform="common",
        description=description,
        command_template=command,
        source="test.md",
    )


def test_filter_accepts_simple_command() -> None:
    assert _filter_reason(_example(), _cfg()) is None


def test_filter_rejects_complex_commands() -> None:
    cfg = _cfg()
    assert _filter_reason(_example("ls $(pwd)"), cfg) == "shell_substitution_or_variable"
    assert _filter_reason(_example("cd /tmp; ls"), cfg) == "multiple_statements"
    assert _filter_reason(_example("echo # comment"), cfg) == "comment"
    assert _filter_reason(_example("ls " + "a" * 200), cfg) == "command_too_long"


def test_slot_filling_is_self_consistent() -> None:
    filler = SlotFiller(random.Random(7))
    filled = filler.fill("tar -czf {{archive.tar.gz}} {{path/to/directory}}")
    assert "{{" not in filled.command
    assert filled.command.startswith("tar -czf ")
    assert filled.values
    values = [value.value for value in filled.values]
    assert values[0].endswith(".tar.gz")
    assert any("/" in value for value in values)


def test_split_pages_is_deterministic_and_disjoint() -> None:
    cfg = _cfg()
    grouped = {
        f"command-{i}": [
            Example(str(i), "common", "Do something useful", f"cmd{i} {{path}}", "x")
        ]
        for i in range(200)
    }
    first = split_pages(grouped, cfg)
    second = split_pages(grouped, cfg)
    assert first == second
    sets = [set(first[name]) for name in ("train", "val", "test")]
    assert sets[0] & sets[1] == set()
    assert sets[0] & sets[2] == set()
    assert sets[1] & sets[2] == set()
    assert sum(len(s) for s in sets) == len(grouped)
    assert all(sets)
