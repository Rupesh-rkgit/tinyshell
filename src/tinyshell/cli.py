"""Command-line interface for TinyShell."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

from .baseline import build_baseline
from .generate import generate_command, load_model
from .text import assess_risk, is_valid_bash


def _default_model() -> str:
    packaged = Path("artifacts/model/model.pt")
    if packaged.exists():
        return "artifacts/model"
    return "latest"


def _load_retriever():
    train_path = Path("data/processed/train.jsonl")
    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} not found; run `make data` before using --retrieve"
        )
    return build_baseline(train_path)


def _retrieve_command(request: str, retriever) -> dict:
    """Return the nearest training command with BM25 (no neural model)."""
    started = time.perf_counter()
    matches = retriever.search(request, top_k=1)
    if not matches:
        command = ""
        source_request = ""
    else:
        command = matches[0]["command"]
        source_request = matches[0]["request"]
    risk = assess_risk(command)
    return {
        "request": request,
        "command": command,
        "raw": command,
        "valid_bash": is_valid_bash(command),
        "risk": risk.level,
        "risk_reasons": list(risk.reasons),
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
        "generated_tokens": 0,
        "mode": "bm25",
        "retrieved_request": source_request,
    }


def _print_result(result: dict, verbose: bool, model_info: dict | None) -> None:
    print()
    print(f"$ {result['command']}" if result["command"] else "(no command generated)")
    print()
    status = "valid bash" if result["valid_bash"] else "INVALID bash syntax"
    print(f"  {status} | risk: {result['risk']} | {result['latency_ms']:.0f} ms")
    if result["risk"] == "destructive":
        print("  WARNING: this command looks destructive. Review it before running.")
        print(f"  Reasons: {', '.join(result['risk_reasons'])}")
    elif result["risk"] == "caution":
        print("  Caution: review this command before running it.")
    if verbose:
        if result.get("mode") == "bm25":
            print("  mode: BM25 retrieval baseline (nearest training request)")
            if result.get("retrieved_request"):
                print(f"  nearest request: {result['retrieved_request']}")
        elif model_info:
            print(f"  model: {model_info['run']} ({model_info['parameters']:,} params)")
            print(f"  tokens: {result['generated_tokens']}")


def _json_result(result: dict, model_info: dict | None) -> dict:
    return {**result, "model": model_info}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="tinyshell",
        description="TinyShell: natural language -> Linux shell command",
    )
    parser.add_argument("request", nargs="?", help="Natural-language request")
    parser.add_argument("--model", default=None, help="Run dir, packaged model, or checkpoint")
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--retrieve",
        action="store_true",
        help="Use the BM25 retrieval baseline instead of the neural model",
    )
    args = parser.parse_args(argv)

    if not args.request and not args.interactive:
        parser.error("provide a request or use --interactive")

    model_info = None
    if args.retrieve:
        if args.model:
            parser.error("--model cannot be combined with --retrieve")
        try:
            retriever = _load_retriever()
        except FileNotFoundError as exc:
            print(f"Could not build retriever: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc

        def run(request: str) -> dict:
            return _retrieve_command(request, retriever)

    else:
        model_path = args.model or _default_model()
        try:
            model, tokenizer, cfg, run_dir = load_model(model_path, args.checkpoint)
        except FileNotFoundError as exc:
            print(f"Could not load model: {exc}", file=sys.stderr)
            print("Train one with `make train` (or `make smoke` first).", file=sys.stderr)
            raise SystemExit(1) from exc

        run_name = run_dir.name
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            with contextlib.suppress(OSError, json.JSONDecodeError):
                run_name = json.loads(manifest_path.read_text(encoding="utf-8")).get(
                    "run", run_name
                )
        model_info = {
            "run": run_name,
            "parameters": model.num_parameters,
            "path": str(run_dir),
        }

        def run(request: str) -> dict:
            return generate_command(
                model,
                tokenizer,
                cfg,
                request,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )

    if args.interactive:
        print("TinyShell interactive mode. Type 'quit' to exit.")
        while True:
            try:
                request = input("\ntinyshell> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if request.lower() in {"quit", "exit"}:
                break
            if not request:
                continue
            result = run(request)
            if args.as_json:
                print(json.dumps(_json_result(result, model_info), indent=2))
            else:
                _print_result(result, args.verbose, model_info)
        return

    result = run(args.request)
    if args.as_json:
        print(json.dumps(_json_result(result, model_info), indent=2))
    else:
        _print_result(result, args.verbose, model_info)


if __name__ == "__main__":
    main()
