from tinyshell.cli import _retrieve_command


class DummyRetriever:
    def search(self, query: str, top_k: int = 1):
        assert query == "list files"
        assert top_k == 1
        return [{"command": "ls -la", "request": "list all files"}]


def test_retrieve_command_returns_structured_result() -> None:
    result = _retrieve_command("list files", DummyRetriever())
    assert result["mode"] == "bm25"
    assert result["command"] == "ls -la"
    assert result["valid_bash"] is True
    assert result["risk"] == "safe"
    assert result["retrieved_request"] == "list all files"
