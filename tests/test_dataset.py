from pathlib import Path

import numpy as np
import pytest

from tinyshell.dataset import PackedDataset, encode_example, pack_rows
from tinyshell.text import EOS_TOKEN, render_prompt
from tinyshell.tokenizer import decode, encode, load_tokenizer, special_id

TOKENIZER_DIR = Path("artifacts/tokenizer")


@pytest.fixture(scope="module")
def tokenizer():
    if not (TOKENIZER_DIR / "tokenizer.json").exists():
        pytest.skip("trained tokenizer artifact is not available")
    return load_tokenizer(TOKENIZER_DIR)


def test_tokenizer_unsets_padding(tokenizer) -> None:
    assert tokenizer.padding is None
    assert tokenizer.truncation is None


def test_special_tokens_exist(tokenizer) -> None:
    assert special_id(tokenizer, EOS_TOKEN) >= 0


def test_round_trip(tokenizer) -> None:
    text = "list all files, including hidden ones → ls -la"
    assert decode(tokenizer, encode(tokenizer, text)) == text


def test_encode_example_masks_prompt(tokenizer) -> None:
    eos_id = special_id(tokenizer, EOS_TOKEN)
    ids, labels = encode_example(tokenizer, "list files", "ls -la", eos_id)
    prompt_len = len(encode(tokenizer, render_prompt("list files")))
    assert len(ids) == len(labels)
    assert labels[:prompt_len] == [-100] * prompt_len
    assert labels[prompt_len:] == encode(tokenizer, "ls -la") + [eos_id]
    assert labels[-1] == eos_id


def test_pack_rows_marks_document_boundaries(tokenizer) -> None:
    eos_id = special_id(tokenizer, EOS_TOKEN)
    rows = [
        {"request": f"list files in folder {i}", "command": f"ls /tmp/folder{i}"}
        for i in range(20)
    ]
    ids, labels, docs, positions, stats = pack_rows(
        rows, tokenizer, seq_len=128, eos_id=eos_id
    )
    assert ids.dtype == np.uint16
    assert labels.dtype == np.int32
    assert docs.dtype == np.int16
    assert positions.dtype == np.int16
    assert ids.shape == labels.shape == docs.shape == positions.shape
    assert ids.shape[1] == 128
    assert stats["examples"] == 20
    assert stats["tokens"] > 0
    assert stats["padding_tokens"] > 0
    assert (labels[:, 0] == -100).all()
    assert (labels >= -100).all()
    assert stats["chunks"] == len(ids)
    # Every document starts at position zero, and padding has no document.
    assert docs[0, 0] == 0
    assert positions[0, 0] == 0
    assert positions[docs == 0][0] == 0
    if (docs == 1).any():
        assert positions[docs == 1][0] == 0
    assert (docs == -1).any()


def test_pack_rows_rejects_empty_corpus(tokenizer) -> None:
    eos_id = special_id(tokenizer, EOS_TOKEN)
    with pytest.raises(ValueError, match="Not enough tokens"):
        pack_rows([], tokenizer, seq_len=128, eos_id=eos_id)


def test_packed_dataset_returns_document_metadata(tmp_path: Path) -> None:
    ids = np.asarray([[1, 2, 3, 0]], dtype=np.uint16)
    labels = np.asarray([[-100, 4, 5, -100]], dtype=np.int32)
    docs = np.asarray([[0, 0, 0, -1]], dtype=np.int16)
    positions = np.asarray([[0, 1, 2, 0]], dtype=np.int16)
    np.save(tmp_path / "ids.npy", ids)
    np.save(tmp_path / "labels.npy", labels)
    np.save(tmp_path / "docs.npy", docs)
    np.save(tmp_path / "positions.npy", positions)

    dataset = PackedDataset(
        tmp_path / "ids.npy",
        tmp_path / "labels.npy",
        tmp_path / "docs.npy",
        tmp_path / "positions.npy",
    )
    x, y, d, p = dataset[0]
    assert x.tolist() == ids[0].tolist()
    assert y.tolist() == labels[0].tolist()
    assert d.tolist() == docs[0].tolist()
    assert p.tolist() == positions[0].tolist()
