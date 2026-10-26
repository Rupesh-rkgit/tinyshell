"""Post-training structured pruning and mechanistic probes for a TinyShell run.

Everything operates on a frozen checkpoint with multiplicative unit masks:

- MLP neuron masks scale each SwiGLU hidden unit before ``down``.
- Head masks scale each attention head's output before ``o_proj``.

Unit scores come from weight magnitude, activation x weight (Wanda-style),
the empirical Fisher of mask gradients (Michel et al. 2019; Kwon et al.
2022), or random order.  Pruned models are scored with teacher-forced loss,
split into the first command token (the family decision) and the remaining
argument tokens, and with greedy generation metrics from ``tinyshell.evaluate``.

Probes:

- logit lens: at which layer is the next command token already decodable?
- single-head and per-sublayer ablations: family selection vs argument copying.
- copy scores: attention mass each head puts on prompt tokens equal to the
  token being generated.
- differential pruning: remove MLP units that are important on seen pages but
  not on unseen pages, and check whether unseen-page loss improves.
- recovery: short masked distillation from the dense model, then physical
  slicing of pruned MLPs to measure real parameter and latency savings.

Usage:
    python scripts/prune_analysis.py --run artifacts/runs/tinyshell-xl-10m
    python scripts/prune_analysis.py --quick   # smoke test, a few minutes
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from tinyshell.dataset import encode_example
from tinyshell.evaluate import compute_metrics, load_run, predict, score_row
from tinyshell.model import TinyLM
from tinyshell.text import EOS_TOKEN, clean_generation, render_prompt, strip_special_tokens
from tinyshell.tokenizer import decode, encode, special_id
from tinyshell.utils import ensure_dir, read_jsonl, write_json

# (name, score key, per-layer allocation).  Magnitude and activation scores are
# not comparable across layers, so they are only used with uniform allocation;
# Fisher scores are in loss units and also support a global ranking.
METHODS = [
    ("random", "random", True),
    ("magnitude", "magnitude", True),
    ("activation", "activation", True),
    ("fisher", "fisher", True),
    ("fisher_global", "fisher", False),
]

# Command families for the specialization experiment (subnetwork extraction).
FAMILIES = {
    "git", "docker", "kubectl", "systemctl", "cargo", "npm", "pip", "terraform", "aws", "helm",
}


def in_family(row: dict[str, Any]) -> bool:
    return row["page"].split()[0] in FAMILIES


# ---------------------------------------------------------------------------
# Masks


class Masks:
    """Multiplicative unit masks installed as forward pre-hooks."""

    def __init__(self, model: TinyLM):
        cfg = model.config
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.neuron = torch.ones(cfg.n_layers, cfg.intermediate_size)
        self.head = torch.ones(cfg.n_layers, cfg.n_heads)
        for i, layer in enumerate(model.layers):
            layer.mlp.down.register_forward_pre_hook(self._mlp_hook(i))
            layer.attn.o_proj.register_forward_pre_hook(self._head_hook(i))

    def _mlp_hook(self, i: int):
        def hook(_module, args):
            return (args[0] * self.neuron[i],)

        return hook

    def _head_hook(self, i: int):
        def hook(_module, args):
            x = args[0]
            b, s, d = x.shape
            x = x.view(b, s, self.n_heads, self.head_dim) * self.head[i][:, None]
            return (x.view(b, s, d),)

        return hook

    def reset(self, requires_grad: bool = False) -> None:
        self.neuron = torch.ones_like(self.neuron).requires_grad_(requires_grad)
        self.head = torch.ones_like(self.head).requires_grad_(requires_grad)

    def set(self, neuron: torch.Tensor | None = None, head: torch.Tensor | None = None) -> None:
        self.reset()
        if neuron is not None:
            self.neuron = neuron.clone().float()
        if head is not None:
            self.head = head.clone().float()


def mask_from_scores(
    scores: torch.Tensor, sparsity: float, per_layer: bool = False
) -> torch.Tensor:
    """Zero the lowest-scoring ``sparsity`` fraction of units.

    ``per_layer`` ranks within each layer (uniform allocation, needed when
    scores are not comparable across layers); otherwise one global ranking.
    """
    if per_layer:
        return torch.stack([mask_from_scores(row, sparsity) for row in scores])
    flat = scores.flatten()
    k = int(round(sparsity * flat.numel()))
    mask = torch.ones_like(flat)
    if k:
        mask[torch.argsort(flat)[:k]] = 0.0
    return mask.view_as(scores)


# ---------------------------------------------------------------------------
# Data


def sample_jsonl(path: Path, n: int, seed: int) -> list[dict[str, Any]]:
    """Uniform sample of ``n`` rows without parsing the whole (large) file."""
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    rng = random.Random(seed)
    return [json.loads(line) for line in rng.sample(lines, min(n, len(lines)))]


def filter_jsonl(path: Path, keep, n: int, seed: int) -> list[dict[str, Any]]:
    """Uniform sample of ``n`` rows satisfying ``keep(row)``."""
    with open(path, encoding="utf-8") as f:
        rows = [row for row in map(json.loads, f) if keep(row)]
    return random.Random(seed).sample(rows, min(n, len(rows)))


def make_batches(rows: list[dict[str, Any]], tokenizer, batch_size: int) -> list[tuple]:
    """Right-padded batches of (inputs, targets, first, valid) for one example per row.

    ``first`` marks the position whose target is the first command token.
    Right padding is safe with causal attention: real tokens never see it.
    """
    eos_id = special_id(tokenizer, EOS_TOKEN)
    encoded = []
    for row in rows:
        ids, labels = encode_example(tokenizer, row["request"], row["command"], eos_id)
        start = next(i for i, label in enumerate(labels) if label != -100)
        encoded.append((ids[:-1], labels[1:], start - 1))
    encoded.sort(key=lambda e: len(e[0]))
    batches = []
    for k in range(0, len(encoded), batch_size):
        chunk = encoded[k : k + batch_size]
        width = max(len(e[0]) for e in chunk)
        inp = torch.zeros(len(chunk), width, dtype=torch.long)
        tgt = torch.full((len(chunk), width), -100, dtype=torch.long)
        first = torch.zeros(len(chunk), width, dtype=torch.bool)
        valid = torch.zeros(len(chunk), width, dtype=torch.bool)
        for r, (x, y, f) in enumerate(chunk):
            inp[r, : len(x)] = torch.tensor(x)
            tgt[r, : len(y)] = torch.tensor(y)
            first[r, f] = True
            valid[r, : len(x)] = True
        batches.append((inp, tgt, first, valid))
    return batches


# ---------------------------------------------------------------------------
# Metrics


def token_losses(model: TinyLM, batch: tuple) -> tuple[torch.Tensor, torch.Tensor]:
    inp, tgt, _first, _valid = batch
    logits, _ = model(inp)
    tok = F.cross_entropy(
        logits.flatten(0, 1), tgt.flatten(), ignore_index=-100, reduction="none"
    ).view_as(tgt)
    return tok, logits


@torch.no_grad()
def loss_metrics(model: TinyLM, batches: list[tuple]) -> dict[str, float]:
    s = dict(all=0.0, first=0.0, arg=0.0, n_all=0, n_first=0, n_arg=0, first_ok=0, arg_ok=0)
    for batch in batches:
        _inp, tgt, first, _valid = batch
        tok, logits = token_losses(model, batch)
        active = tgt != -100
        arg = active & ~first
        pred = logits.argmax(-1)
        s["all"] += tok[active].sum().item()
        s["first"] += tok[first].sum().item()
        s["arg"] += tok[arg].sum().item()
        s["n_all"] += int(active.sum())
        s["n_first"] += int(first.sum())
        s["n_arg"] += int(arg.sum())
        s["first_ok"] += int((pred[first] == tgt[first]).sum())
        s["arg_ok"] += int((pred[arg] == tgt[arg]).sum())
    return {
        "loss": round(s["all"] / s["n_all"], 4),
        "first_loss": round(s["first"] / s["n_first"], 4),
        "arg_loss": round(s["arg"] / s["n_arg"], 4),
        "first_acc": round(s["first_ok"] / s["n_first"], 4),
        "arg_acc": round(s["arg_ok"] / s["n_arg"], 4),
    }


def _cached_step(
    model: TinyLM,
    ids: torch.Tensor,
    positions: torch.Tensor,
    real: torch.Tensor,
    cache: list[tuple[torch.Tensor, torch.Tensor]],
    prefill: bool = False,
) -> torch.Tensor:
    """One KV-cached forward over new tokens; returns last-position logits.

    Uses the model's own submodules, so pruning-mask hooks on ``o_proj`` and
    ``down`` still apply.  ``real`` marks non-pad key positions so far.
    """
    cfg = model.config
    hd = cfg.d_model // cfg.n_heads
    b, t = ids.shape
    total = real.shape[1]
    # Keys allowed: real tokens, causally; every position may also see itself so
    # that pad query rows are never fully masked (their outputs are unused).
    allowed = real[:, None, :].expand(b, t, total).clone()
    if prefill:
        causal = torch.ones(t, t, dtype=torch.bool).tril()
        allowed &= causal[None]
        allowed |= torch.eye(t, dtype=torch.bool)[None]
    mask = allowed[:, None]  # (b, 1, t, total)
    x = model.embed(ids)
    for i, layer in enumerate(model.layers):
        h = layer.norm1(x)
        q, k, v = layer.attn.qkv(h).chunk(3, dim=-1)
        q = q.view(b, t, cfg.n_heads, hd).transpose(1, 2)
        k = k.view(b, t, cfg.n_heads, hd).transpose(1, 2)
        v = v.view(b, t, cfg.n_heads, hd).transpose(1, 2)
        q, k = model.rope(q, k, positions)
        if prefill:
            cache.append((k, v))
        else:
            k = torch.cat([cache[i][0], k], dim=2)
            v = torch.cat([cache[i][1], v], dim=2)
            cache[i] = (k, v)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + layer.attn.o_proj(out.transpose(1, 2).reshape(b, t, cfg.d_model))
        x = x + layer.mlp(layer.norm2(x))
    return model.lm_head(model.norm(x[:, -1]))


@torch.no_grad()
def batched_greedy(
    model: TinyLM, tokenizer, requests: list[str], max_new_tokens: int, batch_size: int = 50
) -> list[str]:
    """Greedy decoding for many prompts at once; equivalent to batch-size-1 decoding.

    Prompts are left-padded.  Pads get document id -1 and real tokens id 0, so
    the block-diagonal document mask keeps real tokens from attending to pads,
    and per-row position ids restart at 0 on the first real token (RoPE sees
    exactly the positions of an unpadded prompt).
    """
    eos_id = special_id(tokenizer, EOS_TOKEN)
    max_len = model.config.max_seq_len
    prompts = [encode(tokenizer, render_prompt(r))[-(max_len - 1) :] for r in requests]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outputs: list[str] = [""] * len(prompts)
    for k in range(0, len(order), batch_size):
        idx = order[k : k + batch_size]
        width = max(len(prompts[i]) for i in idx)
        ids = torch.zeros(len(idx), width, dtype=torch.long)
        docs = torch.full((len(idx), width), -1, dtype=torch.long)
        for r, i in enumerate(idx):
            p = prompts[i]
            ids[r, width - len(p) :] = torch.tensor(p)
            docs[r, width - len(p) :] = 0
        real = docs == 0
        positions = (real.cumsum(1) - 1).clamp(min=0)
        cache: list[tuple[torch.Tensor, torch.Tensor]] = []
        logits = _cached_step(model, ids, positions, real, cache, prefill=True)
        done = torch.zeros(len(idx), dtype=torch.bool)
        new_tokens: list[torch.Tensor] = []
        next_pos = positions[:, -1] + 1
        for _ in range(max_new_tokens):
            nxt = logits.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, eos_id), nxt)
            new_tokens.append(nxt)
            done |= nxt == eos_id
            if bool(done.all()) or int(next_pos.max()) >= max_len:
                break
            real = torch.cat([real, torch.ones(len(idx), 1, dtype=torch.bool)], dim=1)
            logits = _cached_step(model, nxt[:, None], next_pos[:, None], real, cache)
            next_pos = next_pos + 1
        gen = torch.stack(new_tokens, dim=1).tolist() if new_tokens else [[] for _ in idx]
        for r, i in enumerate(idx):
            row = gen[r]
            if eos_id in row:
                row = row[: row.index(eos_id) + 1]
            outputs[i] = clean_generation(strip_special_tokens(decode(tokenizer, row)))
    return outputs


def gen_metrics(
    model: TinyLM, tokenizer, cfg, rows: list[dict[str, Any]], latency: bool = False
) -> dict[str, Any]:
    """Generation metrics.  ``latency=True`` decodes one request at a time and times it."""
    scored = []
    if latency:
        for row in rows:
            result = predict(model, tokenizer, row["request"], cfg)
            scored.append(
                {**score_row(result["command"], row["command"]), "latency_ms": result["latency_ms"]}
            )
    else:
        commands = batched_greedy(
            model, tokenizer, [r["request"] for r in rows], cfg.eval.max_new_tokens
        )
        scored = [
            {**score_row(cmd, row["command"]), "latency_ms": 0.0}
            for cmd, row in zip(commands, rows, strict=True)
        ]
    m = compute_metrics(scored)
    sudo = sum(r["prediction"].startswith("sudo ") for r in scored) / len(scored)
    return {
        "exact": m["exact_match"],
        "command": m["command_match"],
        "token_f1": m["token_f1"],
        "valid_bash": m["valid_bash"],
        "sudo_rate": round(sudo, 4),
        "p50_ms": m["latency_ms"]["p50"],
        "predictions": [r["prediction"] for r in scored],
    }


# ---------------------------------------------------------------------------
# Importance scores (higher = more important)


def magnitude_scores(model: TinyLM) -> tuple[torch.Tensor, torch.Tensor]:
    neuron, head = [], []
    d = model.config.d_model
    hd = d // model.config.n_heads
    for layer in model.layers:
        mlp = layer.mlp
        neuron.append(
            mlp.gate.weight.norm(dim=1) * mlp.up.weight.norm(dim=1) * mlp.down.weight.norm(dim=0)
        )
        w_v = layer.attn.qkv.weight[2 * d :].view(model.config.n_heads, hd, d)
        w_o = layer.attn.o_proj.weight.view(d, model.config.n_heads, hd)
        head.append(w_v.flatten(1).norm(dim=1) * w_o.permute(1, 0, 2).flatten(1).norm(dim=1))
    return torch.stack(neuron).detach(), torch.stack(head).detach()


@torch.no_grad()
def activation_scores(
    model: TinyLM, masks: Masks, batches: list[tuple]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean |unit output| over real tokens times the norm of its outgoing weights."""
    cfg = model.config
    hd = cfg.d_model // cfg.n_heads
    act_n = torch.zeros(cfg.n_layers, cfg.intermediate_size)
    act_h = torch.zeros(cfg.n_layers, cfg.n_heads)
    state: dict[str, Any] = {}
    handles = []

    def neuron_hook(i):
        def hook(_m, args):
            act_n[i] += args[0][state["valid"]].abs().sum(0)

        return hook

    def head_hook(i):
        def hook(_m, args):
            x = args[0][state["valid"]].view(-1, cfg.n_heads, hd)
            act_h[i] += x.norm(dim=-1).sum(0)

        return hook

    for i, layer in enumerate(model.layers):
        handles.append(layer.mlp.down.register_forward_pre_hook(neuron_hook(i)))
        handles.append(layer.attn.o_proj.register_forward_pre_hook(head_hook(i)))
    n_tokens = 0
    for batch in batches:
        state["valid"] = batch[3]
        n_tokens += int(batch[3].sum())
        model(batch[0])
    for h in handles:
        h.remove()
    w_down = torch.stack([layer.mlp.down.weight.norm(dim=0) for layer in model.layers])
    w_o = torch.stack(
        [
            layer.attn.o_proj.weight.view(cfg.d_model, cfg.n_heads, hd)
            .permute(1, 0, 2)
            .flatten(1)
            .norm(dim=1)
            for layer in model.layers
        ]
    )
    return act_n / n_tokens * w_down, act_h / n_tokens * w_o


def fisher_scores(
    model: TinyLM, masks: Masks, batches: list[tuple]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Empirical Fisher of the unit masks, E_x[(dL(x)/dm)^2] at m=1.

    Expects one example per batch.  Squaring per example avoids the sign
    cancellation of |E_x dL/dm|; 0.5 * F_ii approximates the loss increase
    from removing unit i (Kwon et al. 2022).
    """
    neuron = torch.zeros_like(masks.neuron)
    head = torch.zeros_like(masks.head)
    masks.reset(requires_grad=True)
    for batch in batches:
        tok, _ = token_losses(model, batch)
        loss = tok[batch[1] != -100].mean()
        g_n, g_h = torch.autograd.grad(loss, [masks.neuron, masks.head])
        neuron += g_n.pow(2)
        head += g_h.pow(2)
    masks.reset()
    return neuron / len(batches), head / len(batches)


# ---------------------------------------------------------------------------
# Probes


@torch.no_grad()
def logit_lens(model: TinyLM, batches: list[tuple]) -> list[dict[str, float]]:
    """Top-1 accuracy of decoding the residual stream after each layer."""
    states: list[torch.Tensor] = []
    handles = [model.drop.register_forward_hook(lambda _m, _a, out: states.append(out))]
    for layer in model.layers:
        handles.append(layer.register_forward_hook(lambda _m, _a, out: states.append(out)))
    n_layers = len(model.layers) + 1
    ok_first = [0] * n_layers
    ok_arg = [0] * n_layers
    n_first = n_arg = 0
    for batch in batches:
        _inp, tgt, first, _valid = batch
        arg = (tgt != -100) & ~first
        states.clear()
        model(batch[0])
        for i, h in enumerate(states):
            pred = model.lm_head(model.norm(h)).argmax(-1)
            ok_first[i] += int((pred[first] == tgt[first]).sum())
            ok_arg[i] += int((pred[arg] == tgt[arg]).sum())
        n_first += int(first.sum())
        n_arg += int(arg.sum())
    for h in handles:
        h.remove()
    return [
        {"layer": i, "first_acc": round(ok_first[i] / n_first, 4), "arg_acc": round(ok_arg[i] / n_arg, 4)}
        for i in range(n_layers)
    ]


@torch.no_grad()
def copy_scores(model: TinyLM, batches: list[tuple]) -> torch.Tensor:
    """Per head: mean attention mass on prompt tokens identical to the target token.

    Averaged over supervised positions whose target token occurs in the prompt
    (e.g. a file name or user name being copied into the command).
    """
    cfg = model.config
    hd = cfg.d_model // cfg.n_heads
    captured: dict[int, torch.Tensor] = {}
    handles = []
    for i, layer in enumerate(model.layers):

        def hook(module, args, i=i):
            x, rope, position_ids = args[0], args[1], args[2]
            b, s, d = x.shape
            q, k, _v = module.qkv(x).chunk(3, dim=-1)
            q = q.view(b, s, cfg.n_heads, hd).transpose(1, 2)
            k = k.view(b, s, cfg.n_heads, hd).transpose(1, 2)
            q, k = rope(q, k, position_ids)
            att = (q @ k.transpose(-1, -2)) / hd**0.5
            causal = torch.ones(s, s, dtype=torch.bool).tril()
            captured[i] = att.masked_fill(~causal, float("-inf")).softmax(-1)

        handles.append(layer.attn.register_forward_pre_hook(hook))
    total = torch.zeros(cfg.n_layers, cfg.n_heads)
    count = 0
    for batch in batches:
        inp, tgt, first, _valid = batch
        b, s = inp.shape
        prompt_end = first.float().argmax(-1)  # last prompt position per row
        in_prompt = torch.arange(s)[None, :] <= prompt_end[:, None]
        match = (inp[:, None, :] == tgt[:, :, None]) & in_prompt[:, None, :]  # (b, t, j)
        rows = (tgt != -100) & match.any(-1)
        if not rows.any():
            continue
        captured.clear()
        model(inp)
        for i in range(cfg.n_layers):
            mass = (captured[i] * match[:, None].float()).sum(-1)  # (b, H, t)
            total[i] += mass.permute(0, 2, 1)[rows].sum(0)
        count += int(rows.sum())
    for h in handles:
        h.remove()
    return total / max(1, count)


def ablations(model: TinyLM, masks: Masks, batches: list[tuple]) -> dict[str, Any]:
    base = loss_metrics(model, batches)
    cfg = model.config

    def delta(m):
        return {k: round(m[k] - base[k], 4) for k in ("first_loss", "arg_loss", "first_acc", "arg_acc")}

    heads = []
    for layer in range(cfg.n_layers):
        for h in range(cfg.n_heads):
            masks.reset()
            masks.head[layer, h] = 0.0
            heads.append({"layer": layer, "head": h, **delta(loss_metrics(model, batches))})
    sublayers = []
    for layer in range(cfg.n_layers):
        for kind in ("attn", "mlp"):
            masks.reset()
            if kind == "attn":
                masks.head[layer] = 0.0
            else:
                masks.neuron[layer] = 0.0
            sublayers.append({"layer": layer, "kind": kind, **delta(loss_metrics(model, batches))})
    masks.reset()
    return {"base": base, "heads": heads, "sublayers": sublayers}


# ---------------------------------------------------------------------------
# Recovery and physical slicing


def recover(
    model: TinyLM,
    teacher: TinyLM,
    batches: list[tuple],
    steps: int,
    lr: float,
    alpha: float = 0.5,
) -> list[float]:
    """Masked fine-tuning: alpha * CE(labels) + (1 - alpha) * KL(dense teacher || student)."""
    model.requires_grad_(True)
    model.train()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0
    )
    losses = []
    for step in range(steps):
        batch = batches[step % len(batches)]
        active = batch[1] != -100
        with torch.no_grad():
            t_logits, _ = teacher(batch[0])
        tok, s_logits = token_losses(model, batch)
        ce = tok[active].mean()
        kl = F.kl_div(
            F.log_softmax(s_logits[active], -1),
            F.log_softmax(t_logits[active], -1),
            log_target=True,
            reduction="batchmean",
        )
        loss = alpha * ce + (1 - alpha) * kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(round(loss.item(), 4))
    model.eval()
    model.requires_grad_(False)
    return losses


def slice_mlp(model: TinyLM, neuron_mask: torch.Tensor) -> TinyLM:
    """Return a hook-free copy whose MLPs physically keep only unmasked units."""
    sliced = TinyLM(model.config)
    sliced.load_state_dict(model.state_dict())
    d = model.config.d_model
    for i, layer in enumerate(sliced.layers):
        keep = neuron_mask[i].nonzero().flatten()
        mlp = layer.mlp
        gate = nn.Linear(d, len(keep), bias=False)
        up = nn.Linear(d, len(keep), bias=False)
        down = nn.Linear(len(keep), d, bias=False)
        gate.weight.data = mlp.gate.weight.data[keep].clone()
        up.weight.data = mlp.up.weight.data[keep].clone()
        down.weight.data = mlp.down.weight.data[:, keep].clone()
        mlp.gate, mlp.up, mlp.down = gate, up, down
    sliced.eval()
    return sliced


# ---------------------------------------------------------------------------
# Driver


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="TinyShell pruning and interpretability study")
    parser.add_argument("--run", default="artifacts/runs/tinyshell-xl-10m")
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--out", default=None)
    parser.add_argument("--calib", type=int, default=512, help="calibration examples per set")
    parser.add_argument("--loss-n", type=int, default=500, help="examples for loss metrics")
    parser.add_argument("--gen-n", type=int, default=200, help="examples for generation metrics")
    parser.add_argument("--recover-steps", type=int, default=400)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    if args.quick:
        args.calib, args.loss_n, args.gen_n, args.recover_steps = 64, 96, 12, 10

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    model, tokenizer, cfg, _ckpt, run_dir = load_run(args.run, args.checkpoint)
    model.requires_grad_(False)
    teacher = TinyLM(cfg.model)
    teacher.load_state_dict(model.state_dict())
    teacher.eval()
    masks = Masks(model)
    out_dir = ensure_dir(args.out or Path("artifacts/analysis") / run_dir.name)
    data = Path(cfg.data.processed_dir)
    started = time.perf_counter()

    def log(msg: str) -> None:
        print(f"[{time.perf_counter() - started:7.0f}s] {msg}", flush=True)

    def head_rows(split: str, n: int) -> list[dict[str, Any]]:
        return list(read_jsonl(data / f"{split}.jsonl"))[:n]

    # Calibration (scoring) sets are disjoint from all reported sets:
    # seen = training rows, unseen = val_unseen pages.  Reported: val + test_*.
    calib_seen = sample_jsonl(data / "train.jsonl", args.calib, args.seed)
    calib_unseen = head_rows("val_unseen", args.calib)
    sets = {
        "calib_seen": make_batches(calib_seen, tokenizer, 1),
        "calib_unseen": make_batches(calib_unseen, tokenizer, 1),
        "seen": make_batches(head_rows("val", args.loss_n), tokenizer, 64),
        "unseen": make_batches(head_rows("test_unseen", args.loss_n), tokenizer, 64),
    }
    gen_sets = {
        "test_seen": head_rows("test_seen", args.gen_n),
        "test_unseen": head_rows("test_unseen", args.gen_n),
    }
    result: dict[str, Any] = {
        "run": run_dir.name,
        "checkpoint": args.checkpoint,
        "parameters": model.num_parameters,
        "mlp_parameters": sum(p.numel() for layer in model.layers for p in layer.mlp.parameters()),
        "attn_parameters": sum(p.numel() for layer in model.layers for p in layer.attn.parameters()),
        "args": vars(args),
    }

    def evaluate(neuron=None, head=None, generate=True) -> dict[str, Any]:
        masks.set(neuron, head)
        row = {
            "loss_seen": loss_metrics(model, sets["seen"]),
            "loss_unseen": loss_metrics(model, sets["unseen"]),
        }
        if generate:
            for name, rows in gen_sets.items():
                g = gen_metrics(model, tokenizer, cfg, rows)
                g.pop("predictions")
                row[name] = g
        masks.reset()
        return row

    def save() -> None:
        write_json(out_dir / "pruning.json", result)

    log("dense baseline")
    result["dense"] = evaluate()
    # Batched decoding must match the reference one-request-at-a-time decoder.
    check = gen_sets["test_seen"][:100]
    batched = gen_metrics(model, tokenizer, cfg, check)["predictions"]
    single = gen_metrics(model, tokenizer, cfg, check, latency=True)["predictions"]
    agree = sum(x == y for x, y in zip(batched, single, strict=True)) / len(check)
    result["batched_decode_agreement"] = round(agree, 4)
    log(f"  batched vs single-request decoding agreement: {agree:.3f}")
    save()

    log("scoring units")
    rand_gen = torch.Generator().manual_seed(args.seed)
    scores = {
        "random": (
            torch.rand(masks.neuron.shape, generator=rand_gen),
            torch.rand(masks.head.shape, generator=rand_gen),
        ),
        "magnitude": magnitude_scores(model),
        "activation": activation_scores(model, masks, sets["calib_seen"]),
        "fisher": fisher_scores(model, masks, sets["calib_seen"]),
    }
    fisher_unseen = fisher_scores(model, masks, sets["calib_unseen"])
    torch.save(scores | {"fisher_unseen": fisher_unseen}, out_dir / "scores.pt")

    gen_sparsities = {0.1, 0.25, 0.5, 0.75} if not args.quick else {0.5}
    log("MLP neuron pruning curves")
    result["mlp"] = []
    for method, key, per_layer in METHODS:
        for sp in [0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9]:
            mask = mask_from_scores(scores[key][0], sp, per_layer)
            row = evaluate(neuron=mask, generate=sp in gen_sparsities)
            result["mlp"].append({"method": method, "sparsity": sp, **row})
            log(f"  mlp {method:13s} {sp:.2f} seen_loss={row['loss_seen']['loss']:.3f}")
        save()

    log("attention head pruning curves")
    result["heads"] = []
    for method, key, per_layer in METHODS:
        for sp in [0.125, 0.25, 0.375, 0.5, 0.625, 0.75]:
            mask = mask_from_scores(scores[key][1], sp, per_layer)
            row = evaluate(head=mask, generate=sp in {0.25, 0.5})
            result["heads"].append({"method": method, "sparsity": sp, **row})
            log(f"  heads {method:13s} {sp:.3f} seen_loss={row['loss_seen']['loss']:.3f}")
        save()

    log("differential (seen-specific) pruning")
    seen_n = scores["fisher"][0] / scores["fisher"][0].sum()
    unseen_n = fisher_unseen[0] / fisher_unseen[0].sum()
    # Lowest score is pruned first: units with high seen-only importance go first.
    differential = unseen_n - seen_n
    result["differential"] = []
    for sp in [0.01, 0.02, 0.05, 0.1, 0.2]:
        k = int(round(sp * differential.numel()))
        for name, sc in (
            ("seen_specific", differential),
            ("unseen_specific", -differential),
            ("most_important_seen", -seen_n),
            ("random", scores["random"][0]),
        ):
            mask = mask_from_scores(sc, sp)
            row = evaluate(neuron=mask, generate=sp in {0.05, 0.1} and not args.quick)
            result["differential"].append({"method": name, "sparsity": sp, "units": k, **row})
            log(
                f"  diff {name:20s} {sp:.2f} seen={row['loss_seen']['loss']:.3f} "
                f"unseen={row['loss_unseen']['loss']:.3f}"
            )
        save()

    log("probes: logit lens, copy scores, ablations")
    result["logit_lens"] = {
        "seen": logit_lens(model, sets["seen"]),
        "unseen": logit_lens(model, sets["unseen"]),
    }
    result["copy_scores"] = copy_scores(model, sets["seen"]).tolist()
    result["ablations"] = ablations(model, masks, sets["seen"])
    # Overlap of the pruned (lowest-scoring) half between scoring rules; 0.5 = chance.
    result["score_overlap"] = {}
    keys = ["random", "magnitude", "activation", "fisher"]
    n_units = scores["fisher"][0].numel()
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            low_a = set(torch.argsort(scores[a][0].flatten())[: n_units // 2].tolist())
            low_b = set(torch.argsort(scores[b][0].flatten())[: n_units // 2].tolist())
            result["score_overlap"][f"{a}~{b}"] = round(len(low_a & low_b) / len(low_a), 4)
    ranks = torch.stack(
        [
            scores["fisher"][0].flatten().argsort().argsort().float(),
            fisher_unseen[0].flatten().argsort().argsort().float(),
        ]
    )
    result["fisher_seen_unseen_spearman"] = round(float(torch.corrcoef(ranks)[0, 1]), 4)
    result["fisher_global_layer_sparsity"] = {
        str(sp): (1 - mask_from_scores(scores["fisher"][0], sp).mean(1)).round(decimals=4).tolist()
        for sp in (0.25, 0.5, 0.75)
    }
    result["fisher_global_head_sparsity"] = {
        str(sp): (1 - mask_from_scores(scores["fisher"][1], sp).mean(1)).round(decimals=4).tolist()
        for sp in (0.25, 0.5)
    }
    save()

    log("recovery (masked distillation) + physical slicing")
    state = {k: v.clone() for k, v in model.state_dict().items()}
    recovery_rows = sample_jsonl(
        data / "train.jsonl", max(2048, 32 * args.recover_steps), args.seed + 1
    )
    recovery_batches = make_batches(recovery_rows, tokenizer, 32)
    random.Random(args.seed).shuffle(recovery_batches)
    result["recovery"] = []
    plan = (
        [("fisher_global", 0.5), ("random", 0.5), ("fisher_global", 0.75)]
        if not args.quick
        else [("fisher_global", 0.5)]
    )
    dense_lat = gen_metrics(teacher, tokenizer, cfg, gen_sets["test_seen"][:100], latency=True)
    for method, sp in plan:
        if method == "random":
            mask = mask_from_scores(scores["random"][0], sp, per_layer=True)
        else:
            mask = mask_from_scores(scores["fisher"][0], sp)
        masks.set(neuron=mask)
        curve = recover(model, teacher, recovery_batches, args.recover_steps, lr=1e-4)
        row = evaluate(neuron=mask)
        sliced = slice_mlp(model, mask)
        # Physical slicing must reproduce the masked model exactly.
        masks.set(neuron=mask)
        probe = sets["seen"][0][0][:4]
        max_diff = (model(probe)[0] - sliced(probe)[0]).abs().max().item()
        masks.reset()
        lat = gen_metrics(sliced, tokenizer, cfg, gen_sets["test_seen"][:100], latency=True)
        result["recovery"].append(
            {
                "method": method,
                "sparsity": sp,
                "steps": args.recover_steps,
                "train_loss_curve": curve[:: max(1, len(curve) // 40)],
                **row,
                "sliced_parameters": sliced.num_parameters,
                "sliced_max_logit_diff": round(max_diff, 6),
                "sliced_p50_ms": lat["p50_ms"],
                "dense_p50_ms": dense_lat["p50_ms"],
            }
        )
        log(
            f"  recovered {method} {sp:.2f}: seen exact={row['test_seen']['exact']:.3f} "
            f"params={sliced.num_parameters:,} p50 {lat['p50_ms']:.0f} vs {dense_lat['p50_ms']:.0f} ms"
        )
        model.load_state_dict(state)
        save()

    log("family specialization: extract a git/docker/kubectl/... subnetwork")
    held_out = [
        row
        for split in ("val", "test_seen", "test_paraphrase")
        for row in read_jsonl(data / f"{split}.jsonl")
    ]
    fam_eval = [r for r in held_out if in_family(r)][: args.gen_n]
    other_eval = [r for r in gen_sets["test_seen"] if not in_family(r)]
    fam_train = filter_jsonl(
        data / "train.jsonl", in_family, max(args.calib, 32 * args.recover_steps), args.seed + 2
    )
    fam_calib = make_batches(fam_train[: args.calib], tokenizer, 1)
    fam_recovery = make_batches(fam_train, tokenizer, 32)
    random.Random(args.seed).shuffle(fam_recovery)
    family_scores = fisher_scores(model, masks, fam_calib)
    result["family"] = {
        "families": sorted(FAMILIES),
        "n_family_eval": len(fam_eval),
        "n_other_eval": len(other_eval),
        "rows": [],
    }

    def family_eval(neuron=None, head=None) -> dict[str, Any]:
        masks.set(neuron, head)
        row = {}
        for name, rows in (("family", fam_eval), ("other", other_eval)):
            g = gen_metrics(model, tokenizer, cfg, rows)
            g.pop("predictions")
            row[name] = g
        masks.reset()
        return row

    result["family"]["rows"].append({"method": "dense", "sparsity": 0.0, **family_eval()})
    for sp in (0.5, 0.75, 0.9) if not args.quick else (0.75,):
        for name, sc in (("general_fisher", scores["fisher"][0]), ("family_fisher", family_scores[0])):
            mask = mask_from_scores(sc, sp)
            row = family_eval(neuron=mask)
            result["family"]["rows"].append({"method": name, "sparsity": sp, **row})
            log(
                f"  {name:15s} {sp:.2f} family exact={row['family']['exact']:.3f} "
                f"other exact={row['other']['exact']:.3f}"
            )
        save()
    # Specialist: prune MLP neurons and heads by family scores, then distill on family data.
    head_dim = cfg.model.d_model // cfg.model.n_heads
    for sp, head_sp in ((0.75, 0.25), (0.9, 0.5)) if not args.quick else ((0.75, 0.25),):
        neuron_mask = mask_from_scores(family_scores[0], sp)
        head_mask = mask_from_scores(family_scores[1], head_sp)
        masks.set(neuron=neuron_mask, head=head_mask)
        recover(model, teacher, fam_recovery, args.recover_steps, lr=1e-4)
        row = family_eval(neuron=neuron_mask, head=head_mask)
        sliced = slice_mlp(model, neuron_mask)
        removed_heads = int((1 - head_mask).sum().item())
        result["family"]["rows"].append(
            {
                "method": "family_fisher+recovery",
                "sparsity": sp,
                "head_sparsity": head_sp,
                "steps": args.recover_steps,
                # Heads are masked, not sliced: count q/k/v/o slices as removed.
                "effective_parameters": sliced.num_parameters
                - removed_heads * 4 * head_dim * cfg.model.d_model,
                **row,
            }
        )
        log(
            f"  specialist {sp:.2f}/{head_sp:.2f} + recovery: family exact="
            f"{row['family']['exact']:.3f} other exact={row['other']['exact']:.3f}"
        )
        model.load_state_dict(state)
        save()

    result["elapsed_s"] = round(time.perf_counter() - started, 1)
    save()
    log(f"wrote {out_dir / 'pruning.json'}")


if __name__ == "__main__":
    main()
