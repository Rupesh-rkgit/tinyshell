"""Tokenize and pack JSONL examples into fixed-length training chunks.

Examples are packed only up to document boundaries: a chunk contains one or
more complete prompt+completion examples, and the remainder is padded.  Each
token also carries a document id and a within-document position so the model
can use block-diagonal attention and reset RoPE at every example boundary.
Labels are -100 on prompt/system/padding tokens, so the loss is computed only
on the assistant's command and EOS.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import Config
from .text import EOS_TOKEN, PAD_TOKEN, render_prompt
from .tokenizer import encode, load_tokenizer, special_id
from .utils import ensure_dir, read_jsonl, write_json

SPLITS = [
    "train",
    "val",
    "val_unseen",
    "test_seen",
    "test_paraphrase",
    "test_unseen",
    "test_unseen_paraphrase",
]


def encode_example(
    tokenizer, request: str, command: str, eos_id: int
) -> tuple[list[int], list[int]]:
    prompt_ids = encode(tokenizer, render_prompt(request))
    completion_ids = encode(tokenizer, command.strip()) + [eos_id]
    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    return input_ids, labels


def pack_rows(
    rows: Iterable[dict[str, Any]],
    tokenizer,
    seq_len: int,
    eos_id: int,
    pad_id: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Pack complete examples into fixed-length chunks.

    A chunk is never split inside an example.  The unused tail is filled with
    ``pad_id``, labels -100, document id -1, and position id 0.
    """
    chunks: list[tuple[list[int], list[int], list[int], list[int]]] = []
    cur_ids: list[int] = []
    cur_labels: list[int] = []
    cur_docs: list[int] = []
    cur_positions: list[int] = []
    doc_index = 0

    n_examples = 0
    n_tokens = 0
    n_padding = 0
    skipped_too_long = 0
    max_example_len = 0

    def flush() -> None:
        nonlocal cur_ids, cur_labels, cur_docs, cur_positions, doc_index, n_padding
        if not cur_ids:
            return
        padding = seq_len - len(cur_ids)
        if padding:
            cur_ids.extend([pad_id] * padding)
            cur_labels.extend([-100] * padding)
            cur_docs.extend([-1] * padding)
            cur_positions.extend([0] * padding)
            n_padding += padding
        chunks.append((cur_ids, cur_labels, cur_docs, cur_positions))
        cur_ids, cur_labels, cur_docs, cur_positions = [], [], [], []
        doc_index = 0

    for row in rows:
        ids, labels = encode_example(tokenizer, row["request"], row["command"], eos_id)
        if len(ids) < 2:
            continue
        n_examples += 1
        n_tokens += len(ids)
        max_example_len = max(max_example_len, len(ids))
        if len(ids) > seq_len:
            skipped_too_long += 1
            continue
        if len(cur_ids) + len(ids) > seq_len:
            flush()
        cur_ids.extend(ids)
        cur_labels.extend(labels)
        cur_docs.extend([doc_index] * len(ids))
        cur_positions.extend(range(len(ids)))
        doc_index += 1

    flush()
    if not chunks:
        raise ValueError(f"Not enough tokens to make one {seq_len}-token chunk")

    ids_arr = np.asarray([c[0] for c in chunks], dtype=np.uint16)
    labels_arr = np.asarray([c[1] for c in chunks], dtype=np.int32)
    docs_arr = np.asarray([c[2] for c in chunks], dtype=np.int16)
    positions_arr = np.asarray([c[3] for c in chunks], dtype=np.int16)
    stats = {
        "examples": n_examples,
        "tokens": n_tokens,
        "padding_tokens": n_padding,
        "chunks": int(len(ids_arr)),
        "dropped_chunks": 0,
        "skipped_too_long": skipped_too_long,
        "active_label_tokens": int((labels_arr != -100).sum()),
        "seq_len": seq_len,
        "max_example_tokens": max_example_len,
    }
    return ids_arr, labels_arr, docs_arr, positions_arr, stats


class PackedDataset(Dataset):
    def __init__(
        self,
        ids_path: str | Path,
        labels_path: str | Path,
        docs_path: str | Path | None = None,
        positions_path: str | Path | None = None,
    ):
        self.ids = np.load(ids_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")
        if self.ids.shape != self.labels.shape:
            raise ValueError("ids and labels arrays must have the same shape")
        if docs_path is not None and Path(docs_path).exists():
            self.docs = np.load(docs_path, mmap_mode="r")
        else:
            # Backward compatibility with old packed data: one document per chunk.
            self.docs = np.zeros_like(self.ids, dtype=np.int16)
        if positions_path is not None and Path(positions_path).exists():
            self.positions = np.load(positions_path, mmap_mode="r")
        else:
            positions = np.arange(self.ids.shape[1], dtype=np.int16)
            self.positions = np.tile(positions, (len(self.ids), 1))
        for name, array in (("docs", self.docs), ("positions", self.positions)):
            if array.shape != self.ids.shape:
                raise ValueError(f"{name} and ids arrays must have the same shape")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.ids[index].astype(np.int64)),
            torch.from_numpy(self.labels[index].astype(np.int64)),
            torch.from_numpy(self.docs[index].astype(np.int64)),
            torch.from_numpy(self.positions[index].astype(np.int64)),
        )


def pack_split(
    jsonl_path: str | Path,
    out_dir: str | Path,
    tokenizer,
    seq_len: int,
    eos_id: int,
    pad_id: int,
    max_examples: int | None = None,
) -> dict[str, Any]:
    rows = read_jsonl(jsonl_path)
    if max_examples is not None:
        rows = (row for i, row in enumerate(rows) if i < max_examples)
    ids, labels, docs, positions, stats = pack_rows(
        rows, tokenizer, seq_len, eos_id, pad_id=pad_id
    )
    name = Path(jsonl_path).stem
    np.save(Path(out_dir) / f"{name}_ids.npy", ids)
    np.save(Path(out_dir) / f"{name}_labels.npy", labels)
    np.save(Path(out_dir) / f"{name}_doc_ids.npy", docs)
    np.save(Path(out_dir) / f"{name}_position_ids.npy", positions)
    stats["name"] = name
    return stats


def pack_all_splits(cfg: Config, force: bool = False) -> dict[str, Any]:
    processed = Path(cfg.data.processed_dir)
    packed = ensure_dir(processed / "packed")
    tokenizer = load_tokenizer(cfg.data.tokenizer_dir)
    eos_id = special_id(tokenizer, EOS_TOKEN)
    pad_id = special_id(tokenizer, PAD_TOKEN)
    manifest_path = packed / "manifest.json"
    if manifest_path.exists() and not force:
        return json.loads(manifest_path.read_text())

    stats: dict[str, Any] = {
        "seq_len": cfg.model.max_seq_len,
        "packing": "document_aligned",
        "splits": {},
    }
    for split in SPLITS:
        path = processed / f"{split}.jsonl"
        if not path.exists():
            continue
        split_stats = pack_split(
            path,
            packed,
            tokenizer,
            cfg.model.max_seq_len,
            eos_id,
            pad_id,
            max_examples=cfg.eval.max_examples if split != "train" else None,
        )
        stats["splits"][split] = split_stats
        print(
            f"  packed {split:24s} examples={split_stats['examples']:,} "
            f"chunks={split_stats['chunks']:,} tokens={split_stats['tokens']:,}"
        )
    write_json(manifest_path, stats)
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tokenize and pack TinyShell data")
    parser.add_argument("--config", default="configs/tinyshell.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    cfg = Config.from_yaml(args.config)
    stats = pack_all_splits(cfg, force=args.force)
    total = sum(s["tokens"] for s in stats["splits"].values())
    print(f"Packed {total:,} tokens across {len(stats['splits'])} splits.")


if __name__ == "__main__":
    main()
