"""Evaluate a TinyShell checkpoint on seen/unseen command and prompt splits."""

from __future__ import annotations

import argparse
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from .baseline import BM25Retriever
from .config import Config
from .model import TinyLM
from .text import (
    EOS_TOKEN,
    assess_risk,
    clean_generation,
    command_name,
    is_valid_bash,
    normalize_command,
    render_prompt,
    strip_special_tokens,
    token_f1,
)
from .tokenizer import decode, encode, load_tokenizer, special_id
from .utils import (
    ensure_dir,
    read_jsonl,
    resolve_run_dir,
    write_json,
    write_jsonl,
)


def load_run(
    run_dir: str | Path, checkpoint: str = "best"
) -> tuple[TinyLM, Any, Config, dict[str, Any], Path]:
    run_dir = resolve_run_dir(run_dir)
    cfg = Config.from_yaml(run_dir / "config.yaml")
    tokenizer_dir = run_dir / "tokenizer"
    if not (tokenizer_dir / "tokenizer.json").exists():
        tokenizer_dir = Path(cfg.data.tokenizer_dir)
    tokenizer = load_tokenizer(tokenizer_dir)
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoints" / f"{checkpoint}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = TinyLM(cfg.model)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tokenizer, cfg, ckpt, run_dir


@torch.inference_mode()
def predict(
    model: TinyLM,
    tokenizer: Any,
    request: str,
    cfg: Config,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    prompt = render_prompt(request)
    ids = encode(tokenizer, prompt)
    max_len = cfg.model.max_seq_len
    if len(ids) >= max_len:
        ids = ids[-(max_len - 1) :]
    input_ids = torch.tensor([ids], dtype=torch.long)
    eos_id = special_id(tokenizer, EOS_TOKEN)
    started = time.perf_counter()
    output = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens or cfg.eval.max_new_tokens,
        temperature=cfg.eval.temperature if temperature is None else temperature,
        top_k=cfg.eval.top_k if top_k is None else top_k,
        eos_id=eos_id,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    new_ids = output[0, len(ids) :].tolist()
    raw = decode(tokenizer, new_ids)
    command = clean_generation(strip_special_tokens(raw))
    return {
        "command": command,
        "raw": raw,
        "latency_ms": round(latency_ms, 2),
        "generated_tokens": len(new_ids),
    }


def score_row(prediction: str, reference: str) -> dict[str, Any]:
    pred_norm = normalize_command(prediction)
    ref_norm = normalize_command(reference)
    risk = assess_risk(pred_norm)
    return {
        "prediction": pred_norm,
        "reference": ref_norm,
        "exact": pred_norm == ref_norm,
        "command_match": command_name(pred_norm) == command_name(ref_norm),
        "token_f1": round(token_f1(pred_norm, ref_norm), 4),
        "valid_bash": is_valid_bash(pred_norm),
        "risk": risk.level,
        "empty": not pred_norm,
    }


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    latencies = [float(r["latency_ms"]) for r in rows]
    metrics = {
        "n": len(rows),
        "exact_match": round(sum(bool(r["exact"]) for r in rows) / len(rows), 4),
        "command_match": round(sum(bool(r["command_match"]) for r in rows) / len(rows), 4),
        "token_f1": round(statistics.mean(float(r["token_f1"]) for r in rows), 4),
        "valid_bash": round(sum(bool(r["valid_bash"]) for r in rows) / len(rows), 4),
        "empty_rate": round(sum(bool(r["empty"]) for r in rows) / len(rows), 4),
        "risk": dict(Counter(r["risk"] for r in rows)),
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 2),
            "p50": round(statistics.median(latencies), 2),
            "p95": round(sorted(latencies)[min(len(latencies) - 1, int(0.95 * len(latencies)))], 2),
        },
    }
    return metrics


def per_page(rows: list[dict[str, Any]], min_examples: int = 3) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row.get("page", "?")].append(row)
    items = []
    for page, page_rows in groups.items():
        if len(page_rows) < min_examples:
            continue
        items.append(
            {
                "page": page,
                "n": len(page_rows),
                "exact_match": round(sum(bool(r["exact"]) for r in page_rows) / len(page_rows), 4),
                "command_match": round(
                    sum(bool(r["command_match"]) for r in page_rows) / len(page_rows), 4
                ),
            }
        )
    items.sort(key=lambda item: (item["exact_match"], item["n"]), reverse=True)
    return {"best": items[:10], "worst": list(reversed(items[-10:]))}


def evaluate_split(
    model: TinyLM,
    tokenizer: Any,
    cfg: Config,
    split: str,
    limit: int | None,
    max_new_tokens: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = Path(cfg.data.processed_dir) / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing split: {path}")
    source = list(read_jsonl(path))
    if limit is not None:
        source = source[:limit]
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(source):
        result = predict(model, tokenizer, row["request"], cfg, max_new_tokens=max_new_tokens)
        scored = score_row(result["command"], row["command"])
        rows.append(
            {
                "id": row.get("id"),
                "split": split,
                "page": row.get("page"),
                "request": row["request"],
                "reference": row["command"],
                **scored,
                "raw": result["raw"],
                "latency_ms": result["latency_ms"],
                "generated_tokens": result["generated_tokens"],
            }
        )
        if (i + 1) % 100 == 0:
            print(f"  {split}: {i + 1}/{len(source)}", flush=True)
    metrics = compute_metrics(rows)
    metrics["per_page"] = per_page(rows)
    return rows, metrics


def evaluate_baseline(
    splits: list[str],
    cfg: Config,
    limit: int | None,
    train_path: Path,
) -> dict[str, Any]:
    print("Building BM25 baseline index...", flush=True)
    retriever = BM25Retriever(list(read_jsonl(train_path)))
    results: dict[str, Any] = {}
    for split in splits:
        path = Path(cfg.data.processed_dir) / f"{split}.jsonl"
        source = list(read_jsonl(path))
        if limit is not None:
            source = source[:limit]
        rows = []
        for row in source:
            hits = retriever.search(row["request"], top_k=1)
            prediction = hits[0]["command"] if hits else ""
            scored = score_row(prediction, row["command"])
            rows.append({"page": row.get("page"), **scored, "latency_ms": 0.0})
        metrics = compute_metrics(rows)
        results[split] = metrics
        print(
            f"  baseline {split:24s} exact={metrics['exact_match']:.3f} "
            f"cmd={metrics['command_match']:.3f} f1={metrics['token_f1']:.3f}",
            flush=True,
        )
    return results


def _fmt_table(metrics: dict[str, Any]) -> str:
    header = "| Split | N | Exact | Command | Token F1 | Valid bash | Latency p50 |"
    sep = "|---|---:|---:|---:|---:|---:|---:|"
    lines = [header, sep]
    for split, m in metrics.items():
        lines.append(
            f"| {split} | {m.get('n', 0)} | {m.get('exact_match', 0):.3f} | "
            f"{m.get('command_match', 0):.3f} | {m.get('token_f1', 0):.3f} | "
            f"{m.get('valid_bash', 0):.3f} | {m.get('latency_ms', {}).get('p50', 0):.0f} ms |"
        )
    return "\n".join(lines)


def write_report(
    out_dir: Path,
    run_dir: Path,
    checkpoint: str,
    cfg: Config,
    metrics: dict[str, Any],
    baseline: dict[str, Any] | None,
) -> None:
    lines = [
        f"# TinyShell evaluation: `{run_dir.name}`",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Model: {cfg.model.d_model}d x {cfg.model.n_layers}L, "
        f"{cfg.model.vocab_size} vocab, {cfg.model.max_seq_len} context",
        f"- Decoding: greedy (temperature={cfg.eval.temperature}, top_k={cfg.eval.top_k})",
        "",
        "## Results",
        "",
        _fmt_table(metrics),
        "",
        "`test_seen` = seen commands + new slots; `test_paraphrase` = seen commands + "
        "unseen prompt styles; `test_unseen` = unseen commands; "
        "`test_unseen_paraphrase` = both unseen.",
        "",
    ]
    if baseline:
        lines += ["## Retrieval baseline (BM25)", "", _fmt_table(baseline), ""]
        lines += [
            "The baseline returns the command of the most similar training request. "
            "It is strong on seen commands and weak on unseen commands.",
            "",
        ]
    for split in ("test_seen", "test_unseen"):
        m = metrics.get(split)
        if not m or not m.get("per_page"):
            continue
        lines += [f"### Per-command: {split}", ""]
        lines += ["| Command | N | Exact | Command match |", "|---|---:|---:|---:|"]
        for item in m["per_page"]["worst"]:
            lines.append(
                f"| `{item['page']}` | {item['n']} | {item['exact_match']:.3f} | "
                f"{item['command_match']:.3f} |"
            )
        lines.append("")
    lines += [
        "## Notes",
        "",
        "- Exact match is strict: flags, quoting, and paths must match the reference.",
        "- `valid_bash` only checks syntax with `bash -n`; it does not check semantics.",
        "- Predictions are never executed.",
        "- Failure examples are in `<split>_failures.jsonl`; all predictions are in "
        "`<split>_predictions.jsonl`.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate TinyShell")
    parser.add_argument("--run", default="latest", help="Run directory or 'latest'")
    parser.add_argument("--checkpoint", default="best", help="best/final/last or a .pt path")
    parser.add_argument("--splits", default=None, help="Comma-separated split names")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--num-failures", type=int, default=50)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override the processed data directory (e.g. data/core)",
    )
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument(
        "--reuse-predictions",
        action="store_true",
        help="Recompute metrics from existing <split>_predictions.jsonl files",
    )
    args = parser.parse_args(argv)

    model, tokenizer, cfg, _ckpt, run_dir = load_run(args.run, args.checkpoint)
    if args.data_dir:
        cfg.data.processed_dir = args.data_dir
    splits = (
        [s.strip() for s in args.splits.split(",") if s.strip()]
        if args.splits
        else list(cfg.eval.splits or [])
    )
    limit = args.limit if args.limit is not None else cfg.eval.max_examples
    out_dir = ensure_dir(args.output_dir or (Path("artifacts/eval") / run_dir.name))

    print(
        f"Evaluating {run_dir.name} ({model.num_parameters/1e6:.2f}M params) "
        f"on {len(splits)} splits",
        flush=True,
    )
    metrics: dict[str, Any] = {}
    for split in splits:
        if args.reuse_predictions:
            path = out_dir / f"{split}_predictions.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"Missing reused predictions: {path}")
            rows = list(read_jsonl(path))
            split_metrics = compute_metrics(rows)
            failures = [r for r in rows if not r["exact"]][: args.num_failures]
            write_jsonl(out_dir / f"{split}_failures.jsonl", failures)
        else:
            rows, split_metrics = evaluate_split(
                model,
                tokenizer,
                cfg,
                split,
                limit,
                args.max_new_tokens,
            )
            write_jsonl(out_dir / f"{split}_predictions.jsonl", rows)
            failures = [r for r in rows if not r["exact"]][: args.num_failures]
            write_jsonl(out_dir / f"{split}_failures.jsonl", failures)
        metrics[split] = split_metrics
        print(
            f"{split:24s} n={split_metrics['n']:4d} exact={split_metrics['exact_match']:.3f} "
            f"cmd={split_metrics['command_match']:.3f} f1={split_metrics['token_f1']:.3f} "
            f"valid={split_metrics['valid_bash']:.3f}",
            flush=True,
        )

    baseline = None
    if not args.no_baseline:
        baseline = evaluate_baseline(
            splits,
            cfg,
            limit,
            Path(cfg.data.processed_dir) / "train.jsonl",
        )

    result = {
        "run": run_dir.name,
        "checkpoint": args.checkpoint,
        "parameters": model.num_parameters,
        "config": cfg.eval.__dict__,
        "splits": metrics,
        "baseline": baseline,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(out_dir / "metrics.json", result)
    write_report(out_dir, run_dir, args.checkpoint, cfg, metrics, baseline)
    print(f"Wrote {out_dir/'metrics.json'} and {out_dir/'report.md'}", flush=True)


if __name__ == "__main__":
    main()
