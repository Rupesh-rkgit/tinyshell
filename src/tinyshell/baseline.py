"""A BM25 retrieval baseline: return the command of the nearest training request."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .utils import read_jsonl

TOKEN_RE = re.compile(r"[a-z0-9_./*+-]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """BM25 with a NumPy-backed inverted index.

    Posting weights are precomputed once, so a query only needs one vector
    addition per distinct token.  This keeps full evaluation practical on a CPU.
    """

    def __init__(self, rows: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.rows = rows
        self.k1 = k1
        self.b = b
        self.n = len(rows)
        doc_tokens = [tokenize(row["request"]) for row in rows]
        doc_len = np.asarray([len(tokens) for tokens in doc_tokens], dtype=np.float64)
        avgdl = float(doc_len.mean()) if self.n else 0.0
        df: Counter[str] = Counter()
        raw_postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for idx, tokens in enumerate(doc_tokens):
            for token, freq in Counter(tokens).items():
                raw_postings[token].append((idx, freq))
                df[token] += 1
        del doc_tokens

        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for token, pairs in raw_postings.items():
            indices = np.fromiter((p[0] for p in pairs), dtype=np.int32, count=len(pairs))
            freqs = np.fromiter((p[1] for p in pairs), dtype=np.float64, count=len(pairs))
            idf = math.log(1.0 + (self.n - df[token] + 0.5) / (df[token] + 0.5))
            denom = freqs + k1 * (
                1.0 - b + b * doc_len[indices] / max(1e-9, avgdl)
            )
            weights = idf * (k1 + 1.0) * freqs / denom
            self.postings[token] = (indices, weights)

    def search(self, query: str, top_k: int = 1) -> list[dict[str, Any]]:
        if not self.rows:
            return []
        scores = np.zeros(self.n, dtype=np.float64)
        for token, query_freq in Counter(tokenize(query)).items():
            entry = self.postings.get(token)
            if entry is None:
                continue
            indices, weights = entry
            scores[indices] += query_freq * weights
        if not scores.any():
            return [self.rows[0]]
        top_k = min(top_k, self.n)
        if top_k < self.n:
            candidates = np.argpartition(-scores, top_k - 1)[:top_k]
            candidates = candidates[np.argsort(-scores[candidates])]
        else:
            candidates = np.argsort(-scores)
        return [self.rows[int(idx)] for idx in candidates if scores[idx] > 0]


def build_baseline(train_path: str | Path) -> BM25Retriever:
    rows = list(read_jsonl(train_path))
    if not rows:
        raise ValueError(f"No training rows in {train_path}")
    return BM25Retriever(rows)
