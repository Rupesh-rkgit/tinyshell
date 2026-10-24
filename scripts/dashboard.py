#!/usr/bin/env python3
"""Live dashboard for TinyShell training runs and evaluations.

Renders a self-contained HTML page from the repository's artifacts:

- run status (running / complete / stopped) with progress and ETA
- train/val loss curves as inline SVG
- evaluation tables (exact match, command match, token F1, BM25 baseline)
- log tails for runs that are currently training

Usage:
    python scripts/dashboard.py --serve --port 8585   # HTTP server, refreshes each request
    python scripts/dashboard.py --render > dash.html  # one-shot static render

Stdlib only (http.server, json, html); no third-party dependencies.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "artifacts" / "runs"
EVALS_DIR = REPO_ROOT / "artifacts" / "eval"
REFRESH_SECONDS = 60
TAIL_LINES = 60
EVAL_SPLIT_ORDER = [
    "val",
    "val_unseen",
    "test_seen",
    "test_paraphrase",
    "test_unseen",
    "test_unseen_paraphrase",
]
CSS = """
:root { color-scheme: dark; }
body { font-family: ui-sans-serif, system-ui, sans-serif; background: #0d1117;
       color: #c9d1d9; margin: 0; padding: 24px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 17px; margin: 28px 0 10px; border-bottom: 1px solid #21262d;
     padding-bottom: 6px; }
p.sub { color: #8b949e; margin: 0 0 20px; font-size: 13px; }
table { border-collapse: collapse; font-size: 13px; width: 100%; }
th, td { padding: 6px 10px; text-align: left; border-bottom: 1px solid #21262d; }
th { color: #8b949e; font-weight: 600; white-space: nowrap; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.card { background: #161b22; border: 1px solid #21262d; border-radius: 8px;
        padding: 16px 18px; margin: 14px 0; }
.card h3 { margin: 0 0 8px; font-size: 15px; }
.badge { display: inline-block; border-radius: 10px; padding: 1px 10px;
         font-size: 11px; font-weight: 600; vertical-align: middle; }
.badge.running { background: #12331f; color: #3fb950; }
.badge.complete { background: #0d2b55; color: #58a6ff; }
.badge.stopped { background: #42700e33; color: #d29922; }
.bar { height: 10px; background: #21262d; border-radius: 6px; overflow: hidden;
       margin: 10px 0 4px; }
.bar > span { display: block; height: 100%; background: #3fb950; }
pre.log { background: #010409; border: 1px solid #21262d; border-radius: 6px;
          padding: 10px 12px; font-size: 12px; line-height: 1.45; max-height: 340px;
          overflow: auto; font-family: ui-monospace, monospace; color: #8dd3a7; }
svg { background: #010409; border: 1px solid #21262d; border-radius: 6px; }
a { color: #58a6ff; text-decoration: none; }
a:hover { text-decoration: underline; }
.muted { color: #8b949e; }
.legend { font-size: 12px; color: #8b949e; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 50%;
            margin: 0 4px 0 10px; }
footer { margin-top: 30px; color: #8b949e; font-size: 12px; }
"""


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _fmt_secs(secs: float) -> str:
    if not math.isfinite(secs) or secs <= 0:
        return "-"
    h, rem = divmod(int(secs), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"


def _short_commit(commit: str | None) -> str:
    return (commit or "-")[:8]


def find_running() -> dict[str, list[int]]:
    """Map running `tinyshell.train --run-id NAME` processes to their PIDs."""
    procs: dict[str, list[int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        parts = raw.split(b"\x00")
        if not parts or not parts[0].decode(errors="replace").endswith("python"):
            continue
        cmd = [p.decode(errors="replace") for p in parts if p]
        if "tinyshell.train" not in " ".join(cmd):
            continue
        try:
            idx = cmd.index("--run-id")
            run_id = cmd[idx + 1]
        except (ValueError, IndexError):
            run_id = "<unknown>"
        procs.setdefault(run_id, []).append(int(entry.name))
    return procs


def read_metrics_jsonl(path: Path) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Split metrics.jsonl rows into (train, val) series."""
    train: list[dict[str, float]] = []
    val: list[dict[str, float]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return train, val
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not all(k in row for k in ("step", "loss", "split")):
            continue
        (train if row.get("split") == "train" else val).append(row)
    return train, val


def read_run(run_dir: Path, running: dict[str, list[int]]) -> dict[str, Any]:
    env = _load_json(run_dir / "env.json") or {}
    summary = _load_json(run_dir / "model_summary.json") or {}
    final = _load_json(run_dir / "final_metrics.json") or {}
    train_rows, val_rows = read_metrics_jsonl(run_dir / "metrics.jsonl")

    pids = sorted(running.get(run_dir.name, []))
    if pids:
        status = "running"
    elif final or val_rows:
        status = "complete"
    else:
        status = "stopped"

    latest = train_rows[-1] if train_rows else None
    val_losses = [r["loss"] for r in val_rows if "loss" in r]
    best_val = min(val_losses) if val_losses else final.get("best_val")

    log_text = ""
    log_path = run_dir / "train.log"
    if log_path.exists():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"(\d+) total steps", log_text)
    total_steps = int(match.group(1)) if match else None

    info: dict[str, Any] = {
        "name": run_dir.name,
        "dir": str(run_dir),
        "status": status,
        "pids": pids,
        "parameters": summary.get("parameters") or final.get("parameters"),
        "architecture": describe_architecture(summary),
        "started": env.get("timestamp"),
        "git_commit": env.get("git_commit"),
        "torch": env.get("torch"),
        "train_rows": train_rows,
        "val_rows": val_rows,
        "latest_step": int(latest["step"]) if latest else None,
        "latest_loss": latest.get("loss") if latest else None,
        "latest_tokps": latest.get("tokens_per_sec") if latest else None,
        "latest_elapsed": latest.get("elapsed") if latest else None,
        "best_val": best_val,
        "total_steps": total_steps,
        "log_tail": log_text.splitlines()[-TAIL_LINES:],
    }

    latest_step = info["latest_step"]
    elapsed = info["latest_elapsed"]
    if latest_step and elapsed and elapsed > 0:
        info["sec_per_step"] = elapsed / latest_step
    else:
        info["sec_per_step"] = None
    if status == "running" and total_steps and latest_step and info["sec_per_step"]:
        remaining = max(0, total_steps - latest_step) * info["sec_per_step"]
        info["eta_secs"] = remaining
        info["progress_pct"] = 100.0 * latest_step / total_steps
    else:
        info["eta_secs"] = None
        info["progress_pct"] = None
    return info


def describe_architecture(summary: dict[str, Any]) -> str | None:
    cfg = summary.get("config")
    if not isinstance(cfg, dict):
        return None
    return (
        f"{cfg.get('d_model')}d x {cfg.get('n_layers')}L x {cfg.get('n_heads')}H, "
        f"mlp {cfg.get('intermediate_size')}, vocab {cfg.get('vocab_size')}, "
        f"ctx {cfg.get('max_seq_len')}"
    )


def collect(root: Path = REPO_ROOT) -> dict[str, Any]:
    runs_root = root / "artifacts" / "runs"
    evals_root = root / "artifacts" / "eval"
    running = find_running()
    runs: list[dict[str, Any]] = []
    if runs_root.is_dir():
        for run_dir in sorted(runs_root.iterdir()):
            if not run_dir.is_dir():
                continue
            if (run_dir / "env.json").exists() or (
                run_dir / "metrics.jsonl"
            ).exists():
                runs.append(read_run(run_dir, running))
    runs.sort(key=lambda r: (r["status"] != "running", r["name"]))

    evals: list[dict[str, Any]] = []
    if evals_root.is_dir():
        for eval_dir in sorted(evals_root.iterdir()):
            metrics = _load_json(eval_dir / "metrics.json")
            if metrics is None:
                continue
            evals.append(summarize_eval(eval_dir, metrics))
    return {"generated": iso_now(), "runs": runs, "evals": evals}


def summarize_eval(eval_dir: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    splits: dict[str, dict[str, Any]] = {}
    for name, raw in (metrics.get("splits") or {}).items():
        splits[name] = {
            "n": raw.get("n"),
            "exact": raw.get("exact_match"),
            "cmd": raw.get("command_match"),
            "f1": raw.get("token_f1"),
            "valid": raw.get("valid_bash"),
        }
    baseline = metrics.get("baseline") or {}
    return {
        "dir": eval_dir.name,
        "run": metrics.get("run"),
        "checkpoint": metrics.get("checkpoint"),
        "parameters": metrics.get("parameters"),
        "generated_at": metrics.get("generated_at"),
        "splits": splits,
        "baseline": baseline,
    }


def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def loss_svg(run: dict[str, Any], width: int = 620, height: int = 160) -> str:
    """Render train/val loss curves as a line chart inside an SVG."""
    series = [
        ("#8dd3a7", [(r["step"], r["loss"]) for r in run["train_rows"]]),
        ("#58a6ff", [(r["step"], r["loss"]) for r in run["val_rows"]]),
    ]
    points = [(x, y) for _, s in series for x, y in s]
    if len(points) < 2:
        return ""
    pad, w, h = 34, width, height
    max_x = max(x for x, _ in points)
    ys = [y for _, y in points]
    y_lo, y_hi = min(ys), max(ys)
    span = max(1e-6, y_hi - y_lo)
    y_lo -= 0.06 * span
    y_hi += 0.06 * span
    yspan = max(1e-6, y_hi - y_lo)

    def tx(x: float) -> float:
        return pad + (x - points[0][0]) / max(1, max_x - points[0][0]) * (w - pad - 8)

    def ty(y: float) -> float:
        return 12 + (y_hi - y) / yspan * (h - 30)

    parts = [
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" '
        f'aria-label="loss curve">',
        f'<text x="6" y="22" class="legend">{y_hi:.2f}</text>',
        f'<text x="6" y="{h - 6}" class="legend">{y_lo:.2f}</text>',
        f'<text x="{pad}" y="12" class="legend">loss (train {len(run["train_rows"])} pts, '
        f'val {len(run["val_rows"])} pts)</text>',
        f'<text x="{w - 8}" y="{h - 6}" text-anchor="end" class="legend">step {max_x:.0f}</text>',
    ]
    grid = [y_lo + i * span / 4 for i in range(5)]
    for gy in grid:
        parts.append(
            f'<line x1="{pad}" y1="{ty(gy):.1f}" x2="{w - 8}" y2="{ty(gy):.1f}" '
            f'stroke="#21262d" stroke-width="1" />'
        )
    for color, data in series:
        if len(data) < 2:
            continue
        # Cap points for very long histories to keep the SVG small.
        stride = max(1, len(data) // 240)
        sample = data[::stride]
        if sample[-1] != data[-1]:
            sample.append(data[-1])
        pts = " ".join(f"{tx(x):.1f},{ty(y):.1f}" for x, y in sample)
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="1.6" points="{pts}" />'
        )
    parts.append("</svg>")
    return "".join(parts)


def fmt_num(value: Any, digits: int = 3) -> str:
    if value is None or isinstance(value, bool):
        return "-"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    if math.isinf(value) or math.isnan(value):
        return "-"
    return f"{value:.{digits}f}"


def render_run_card(run: dict[str, Any]) -> str:
    e = html.escape
    params = f"{run['parameters']:,} params" if run["parameters"] else "params: -"
    arch = e(run["architecture"] or "architecture unknown")
    progress = ""
    if run["progress_pct"] is not None:
        progress = (
            f'<div class="bar"><span style="width:{run["progress_pct"]:.1f}%"></span></div>'
            f'<div class="muted">step {run["latest_step"]:,} / {run["total_steps"]:,} '
            f'({run["progress_pct"]:.1f}%) - ETA {_fmt_secs(run["eta_secs"] or 0)}</div>'
        )
    stats = (
        f'<div>loss now: <b>{fmt_num(run["latest_loss"], 4)}</b>'
        f' &nbsp;|&nbsp; tok/s: {fmt_num(run["latest_tokps"], 0)}'
        f' &nbsp;|&nbsp; best val loss: <b>{fmt_num(run["best_val"], 4)}</b>'
        f' &nbsp;|&nbsp; started: {e(run["started"] or "-")}'
        f' &nbsp;|&nbsp; commit: {e(_short_commit(run["git_commit"]))}'
        f' &nbsp;|&nbsp; {e(run["torch"] or "-")}</div>'
    )
    tail = ""
    if run["log_tail"]:
        tail = (
            '<pre class="log">'
            + e("\n".join(run["log_tail"][-40:]))
            + "</pre>"
        )
    return (
        f'<div class="card"><h3>{e(run["name"])}'
        f' <span class="badge {run["status"]}">{run["status"]}</span></h3>'
        f'<div class="muted">{arch} - {params}</div>'
        f"{progress}{stats}{tail}"
        f"{loss_svg(run)}</div>"
    )


def render_runs_table(runs: list[dict[str, Any]]) -> str:
    rows = []
    for run in runs:
        params = run["parameters"]
        params_text = f"{params:,}" if params else "-"
        eta_secs = run["eta_secs"]
        eta_text = f" ({_fmt_secs(eta_secs)} left)" if eta_secs else ""
        status = f'<span class="badge {run["status"]}">{run["status"]}</span>'
        rows.append(
            "<tr>"
            f"<td><b>{html.escape(run['name'])}</b></td>"
            f"<td>{status}</td>"
            f"<td>{html.escape(run['architecture'] or '-')}</td>"
            f"<td class='num'>{params_text}</td>"
            f"<td class='num'>{fmt_num(run['best_val'], 4)}</td>"
            f"<td class='num'>{fmt_num(run['latest_loss'], 4)}</td>"
            f"<td class='num'>{fmt_num(run['latest_tokps'], 0)}</td>"
            f"<td class='num'>{run['latest_step'] or 0:,}</td>"
            f"<td class='num'>{run['total_steps'] or '-'}{eta_text}</td>"
            f"<td>{html.escape(run['started'] or '-')}</td>"
            f"<td>{html.escape(_short_commit(run['git_commit']))}</td>"
            "</tr>"
        )
    header = (
        "<tr><th>run</th><th>status</th><th>architecture</th>"
        "<th class='num'>params</th><th class='num'>best val loss</th>"
        "<th class='num'>last loss</th><th class='num'>tok/s</th>"
        "<th class='num'>step</th><th class='num'>of total (ETA)</th>"
        "<th>started</th><th>commit</th></tr>"
    )
    return (
        f'<table>{header}{"".join(rows)}</table>'
        if rows
        else '<p class="muted">No runs found under artifacts/runs.</p>'
    )


def render_eval_table(evals: list[dict[str, Any]]) -> str:
    parts = []
    for record in evals:
        params = record["parameters"]
        param_text = f" - {params:,} params" if params else ""
        title = html.escape(
            f"{record['run']} / ckpt {record['checkpoint']}{param_text}"
        )
        link = f"/static/{html.escape(record['dir'])}/report.md"
        rows = []
        splits = record["splits"]
        names = [s for s in EVAL_SPLIT_ORDER if s in splits] + sorted(
            set(splits) - set(EVAL_SPLIT_ORDER)
        )
        base = record["baseline"]
        base_cols = ""
        if base:
            base_cols = (
                "<th class='num'>BM25 exact</th><th class='num'>BM25 cmd</th>"
                "<th class='num'>BM25 F1</th>"
            )
        for name in names:
            row = splits[name]
            n_val = row["n"]
            row_cells = (
                f"<td>{name}<span class='muted'> ({n_val})</span></td>"
                + "".join(
                    f"<td class='num'>{fmt_num(row[k])}</td>"
                    for k in ("exact", "cmd", "f1", "valid")
                )
            )
            if base:
                bm = base.get(name) or {}
                row_cells += "".join(
                    f"<td class='num'>{fmt_num(bm.get(k))}</td>"
                    for k in ("exact_match", "command_match", "token_f1")
                )
            rows.append(f"<tr>{row_cells}</tr>")
        header = (
            "<tr><th>split</th><th class='num'>exact</th>"
            "<th class='num'>command</th><th class='num'>token F1</th>"
            f"<th class='num'>valid bash</th>{base_cols}</tr>"
        )
        parts.append(
            f'<div class="card"><h3>{title} <a href="{link}">report.md</a></h3>'
            f"<table>{header}{''.join(rows)}</table></div>"
        )
    if not parts:
        return '<p class="muted">No evaluations found under artifacts/eval.</p>'
    return "".join(parts)


def render_page(state: dict[str, Any]) -> str:
    e = html.escape
    running = [r for r in state["runs"] if r["status"] == "running"]
    n_running = len(running)
    n_runs = len(state["runs"])
    sub = (
        f"generated {e(state['generated'])} - {n_running} run(s) currently training"
        f" out of {n_runs} known - auto-refresh {REFRESH_SECONDS}s"
    )
    parts = [
        '<!doctype html><html><head><meta charset="utf-8">',
        f'<meta http-equiv="refresh" content="{REFRESH_SECONDS}">',
        "<title>TinyShell dashboard</title>",
        f"<style>{CSS}</style></head><body>",
        "<h1>TinyShell dashboard</h1>",
        f"<p class='sub'>{sub}</p>",
    ]
    parts.append("<h2>Currently running</h2>")
    parts.append(
        "".join(render_run_card(r) for r in running)
        or '<p class="muted">No training in progress.</p>'
    )
    parts.append("<h2>All runs</h2>")
    parts.append(render_runs_table(state["runs"]))
    idle = [r for r in state["runs"] if r["status"] != "running"]
    parts.append("<h2>Finished runs: curves and logs</h2>")
    parts.append(
        "".join(render_run_card(r) for r in idle)
        or '<p class="muted">No finished runs yet.</p>'
    )
    parts.append("<h2>Evaluations</h2>")
    parts.append(render_eval_table(state["evals"]))
    parts.append(
        "<footer>TinyShell dashboard - data read from artifacts/ (runs, eval) - "
        '<i style="background:#8dd3a7"></i>train loss '
        '<i style="background:#58a6ff"></i>validation loss - '
        f"refresh interval {REFRESH_SECONDS}s</footer></body></html>"
    )
    return "".join(parts)


class Handler(BaseHTTPRequestHandler):
    """Serve the dashboard at '/' and evaluation artifacts under /static/."""

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            body = render_page(collect()).encode("utf-8")
            self._respond(200, body, "text/html; charset=utf-8")
        elif path.startswith("/static/"):
            self._serve_static(path[len("/static/") :])
        else:
            self._respond(404, b"not found", "text/plain")

    def _serve_static(self, rel: str) -> None:
        target = (EVALS_DIR / rel).resolve()
        allowed_root = EVALS_DIR.resolve()
        if not str(target).startswith(str(allowed_root) + "/") or not target.is_file():
            self._respond(404, b"not found", "text/plain")
            return
        ctype = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".md": "text/markdown; charset=utf-8",
            ".json": "application/json",
            ".csv": "text/csv",
        }.get(target.suffix, "text/plain; charset=utf-8")
        self._respond(200, target.read_bytes(), ctype)

    def _respond(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        return  # keep server output quiet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", help="start the HTTP server")
    parser.add_argument("--render", action="store_true", help="render HTML to stdout")
    parser.add_argument("--port", type=int, default=8585)
    args = parser.parse_args()
    if not args.serve and not args.render:
        args.render = True
    if args.render and not args.serve:
        print(render_page(collect()))
        return
    server = ThreadingHTTPServer(("", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"TinyShell dashboard serving at {url} (Ctrl-C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
