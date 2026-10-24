from tinyshell.baseline import BM25Retriever, tokenize


def test_tokenize_lowercases_and_keeps_paths() -> None:
    assert tokenize("List /tmp/Files with *.PY") == ["list", "/tmp/files", "with", "*.py"]


def test_bm25_returns_best_match() -> None:
    rows = [
        {"request": "list files in a directory", "command": "ls"},
        {"request": "show disk usage", "command": "du -sh"},
        {"request": "show disk usage for a directory", "command": "du -sh /tmp"},
    ]
    retriever = BM25Retriever(rows)
    assert retriever.search("list files")[0]["command"] == "ls"
    assert retriever.search("disk usage")[0]["command"].startswith("du")
    assert retriever.search("unknown words")[0]["command"] == "ls"


def test_bm25_empty_index() -> None:
    assert BM25Retriever([]).search("anything") == []
