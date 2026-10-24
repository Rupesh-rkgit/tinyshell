"""Train and load the byte-level BPE tokenizer."""

from __future__ import annotations

import argparse
import statistics
import time
from collections.abc import Iterator
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tokenizers.processors import ByteLevel as ByteLevelProcessor

from .config import Config
from .text import (
    EOS_TOKEN,
    SPECIAL_TOKENS,
    SYSTEM_PROMPT,
    render_training_text,
)
from .utils import ensure_dir, human_time, read_jsonl, write_json


def iter_corpus(path: str | Path) -> Iterator[str]:
    for row in read_jsonl(path):
        yield render_training_text(row["request"], row["command"])


def train_tokenizer(
    train_path: str | Path,
    output_dir: str | Path,
    vocab_size: int = 4096,
    min_frequency: int = 2,
    sample_size: int = 5000,
) -> dict:
    output_dir = ensure_dir(output_dir)
    started = time.perf_counter()

    corpus_path = Path(train_path).parent / "tokenizer_corpus.txt"
    n_rows = 0
    n_chars = 0
    with open(corpus_path, "w", encoding="utf-8") as f:
        for text in iter_corpus(train_path):
            f.write(text + "\n")
            n_rows += 1
            n_chars += len(text)

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=True)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tokenizer.train_from_iterator(iter_corpus(train_path), trainer=trainer, length=n_rows)
    tokenizer.save(str(output_dir / "tokenizer.json"))

    # Verify special tokens and byte-level round trips.
    special_ids = {token: tokenizer.token_to_id(token) for token in SPECIAL_TOKENS}
    missing = [token for token, idx in special_ids.items() if idx is None]
    if missing:
        raise RuntimeError(f"Tokenizer is missing special tokens: {missing}")

    lengths: list[int] = []
    sample_chars = 0
    roundtrip_failures = 0
    for i, row in enumerate(read_jsonl(train_path)):
        if i >= sample_size:
            break
        text = render_training_text(row["request"], row["command"])
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        lengths.append(len(ids))
        sample_chars += len(text)
        if tokenizer.decode(ids, skip_special_tokens=False) != text:
            roundtrip_failures += 1

    def pct(p: float) -> float:
        values = sorted(lengths)
        idx = min(len(values) - 1, max(0, int(round(p * (len(values) - 1)))))
        return float(values[idx])

    stats = {
        "vocab_size": tokenizer.get_vocab_size(),
        "requested_vocab_size": vocab_size,
        "special_tokens": special_ids,
        "corpus": {
            "examples": n_rows,
            "chars": n_chars,
            "chars_per_token_sample": round(sample_chars / max(1, sum(lengths)), 3),
        },
        "token_lengths": {
            "n": len(lengths),
            "mean": round(statistics.mean(lengths), 2) if lengths else 0,
            "p50": pct(0.50),
            "p90": pct(0.90),
            "p95": pct(0.95),
            "p99": pct(0.99),
            "max": max(lengths) if lengths else 0,
        },
        "roundtrip_failures": roundtrip_failures,
        "train_seconds": round(time.perf_counter() - started, 2),
    }

    write_json(output_dir / "stats.json", stats)
    write_json(
        output_dir / "tokenizer_config.json",
        {
            "tokenizer_class": "ByteLevelBPE",
            "vocab_size": tokenizer.get_vocab_size(),
            "pad_token": "<|pad|>",
            "eos_token": EOS_TOKEN,
            "system_token": "<|system|>",
            "user_token": "<|user|>",
            "assistant_token": "<|assistant|>",
            "system_prompt": SYSTEM_PROMPT,
            "chat_template": (
                "<|system|>\\n{system}\\n<|user|>\\n{user}\\n<|assistant|>\\n"
            ),
        },
    )

    vocab = sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])
    with open(output_dir / "vocab_preview.txt", "w", encoding="utf-8") as f:
        for token, idx in vocab[:200]:
            f.write(f"{idx:5d}  {token!r}\n")

    print(
        f"Trained tokenizer: vocab={tokenizer.get_vocab_size()} "
        f"examples={n_rows:,} tokens/example p50={stats['token_lengths']['p50']} "
        f"p99={stats['token_lengths']['p99']} in {human_time(stats['train_seconds'])}"
    )
    if roundtrip_failures:
        print(f"WARNING: {roundtrip_failures}/{len(lengths)} round trips failed")
    return stats


def load_tokenizer(path: str | Path) -> Tokenizer:
    path = Path(path)
    if path.is_dir():
        path = path / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {path}")
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return tokenizer


def encode(tokenizer: Tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False).ids


def decode(tokenizer: Tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False)


def special_id(tokenizer: Tokenizer, token: str) -> int:
    idx = tokenizer.token_to_id(token)
    if idx is None:
        raise ValueError(f"Tokenizer has no special token {token!r}")
    return idx


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the TinyShell tokenizer")
    parser.add_argument("--config", default="configs/tinyshell.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    train_path = Path(cfg.data.processed_dir) / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} missing; run `make data` first")
    out = Path(cfg.data.tokenizer_dir)
    if (out / "tokenizer.json").exists() and not args.force:
        print(f"{out / 'tokenizer.json'} exists; use --force to retrain.")
        return
    train_tokenizer(train_path, out, vocab_size=cfg.model.vocab_size)


if __name__ == "__main__":
    main()
