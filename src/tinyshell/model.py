"""A small Llama-style decoder-only transformer.

Deliberately dependency-light (PyTorch only) and readable: RMSNorm, RoPE,
causal self-attention via scaled_dot_product_attention, and SwiGLU MLPs.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    """Precomputed rotary position embeddings for the maximum context length."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len).float()
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", torch.cat([freqs.cos(), freqs.cos()], dim=-1))
        self.register_buffer("sin", torch.cat([freqs.sin(), freqs.sin()], dim=-1))

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        cos = self.cos[position_ids].unsqueeze(1)  # (batch, 1, seq, head_dim)
        sin = self.sin[position_ids].unsqueeze(1)
        return _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (_rotate_half(x) * sin)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        position_ids: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = rope(q, k, position_ids)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=attn_mask is None,
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.intermediate_size, bias=False)
        self.up = nn.Linear(config.d_model, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = Attention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        rope: RotaryEmbedding,
        position_ids: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope, position_ids, attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.rope = RotaryEmbedding(
            config.d_model // config.n_heads, config.max_seq_len, config.rope_theta
        )

        self.apply(self._init_weights)
        # GPT-2 style scaled init for residual output projections.
        for name, module in self.named_modules():
            if name.endswith(("o_proj", "down")):
                nn.init.normal_(module.weight, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))
        if config.tie_embeddings:
            self.lm_head.weight = self.embed.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_parameters_non_embedding(self) -> int:
        return self.num_parameters - self.embed.weight.numel()

    @staticmethod
    def document_mask(doc_ids: torch.Tensor) -> torch.Tensor:
        """Causal mask that also blocks attention across packed documents."""
        seq_len = doc_ids.shape[1]
        same_document = doc_ids[:, :, None] == doc_ids[:, None, :]
        causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=doc_ids.device).tril()
        # Shape (batch, 1, seq, seq); broadcast over heads.
        return (same_document & causal).unsqueeze(1)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        doc_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        first_token_weight: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}"
            )
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        attn_mask = self.document_mask(doc_ids) if doc_ids is not None else None
        x = self.drop(self.embed(input_ids))
        for layer in self.layers:
            x = layer(x, self.rope, position_ids, attn_mask)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            flat_logits = logits.reshape(-1, self.config.vocab_size)
            flat_targets = targets.reshape(-1)
            if first_token_weight == 1.0:
                loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-100)
            else:
                # Upweight the first command token.  This is a cheap way to
                # push the model to choose the right command family instead of
                # only learning to copy arguments from the prompt.
                token_loss = F.cross_entropy(
                    flat_logits, flat_targets, ignore_index=-100, reduction="none"
                )
                active = targets != -100
                # A packed chunk can contain several examples.  The first
                # command token of every example is the token immediately
                # after an inactive prompt/padding run, not just the first
                # active token in the whole row.
                previous_active = torch.zeros_like(active)
                previous_active[:, 1:] = active[:, :-1]
                first = active & ~previous_active
                weights = torch.ones_like(targets, dtype=token_loss.dtype).reshape(-1)
                weights[first.reshape(-1)] = first_token_weight
                loss = (token_loss * weights).sum() / weights[active.reshape(-1)].sum()
        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int | None = None,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive decoding. Batch size 1 is the intended use."""
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.max_seq_len :]
            logits, _ = self(context)
            logits = logits[:, -1, :]
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = torch.where(
                        logits < values[:, [-1]], torch.full_like(logits, float("-inf")), logits
                    )
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if eos_id is not None and bool((next_token == eos_id).all()):
                break
        return input_ids

    def configure_optimizer(
        self,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
    ) -> torch.optim.Optimizer:
        """AdamW with weight decay only on matrices (not norms/embeddings)."""
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2 and "embed" not in name:
                decay.append(param)
            else:
                no_decay.append(param)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
