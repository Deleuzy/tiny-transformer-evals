"""A small decoder-only transformer: token embedding, sinusoidal positions, pre-LayerNorm blocks."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from tte.attention import KV, MultiHeadSelfAttention


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    max_len: int = 128
    dropout: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sinusoidal_positions(max_len: int, d_model: int) -> Tensor:
    """PE[pos, 2i] = sin(pos / 10000^(2i/d)), PE[pos, 2i+1] = cos(same). Returns [max_len, d_model]."""
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    freq = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * freq)
    pe[:, 1::2] = torch.cos(pos * freq)
    return pe


class Block(nn.Module):
    """x + Attn(LN(x)), then x + MLP(LN(x)). Pre-LN keeps gradients well behaved without warm tricks."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor, past: KV | None = None) -> tuple[Tensor, KV]:
        a, present = self.attn(self.ln1(x), past)
        x = x + self.drop(a)
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x, present


class TinyTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.register_buffer("pos_emb", sinusoidal_positions(cfg.max_len, cfg.d_model), persistent=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: Tensor, past: list[KV] | None = None) -> tuple[Tensor, list[KV]]:
        """idx [B, T] -> logits [B, T, V] and the new key/value cache (one entry per layer)."""
        t = idx.size(1)
        offset = 0 if past is None else past[0][0].size(2)
        if offset + t > self.cfg.max_len:
            raise ValueError(f"sequence length {offset + t} exceeds max_len {self.cfg.max_len}")
        pos = self.get_buffer("pos_emb")
        x = self.drop(self.tok_emb(idx) * math.sqrt(self.cfg.d_model) + pos[offset : offset + t])
        presents: list[KV] = []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, None if past is None else past[i])
            presents.append(kv)
        return self.head(self.ln_f(x)), presents


@torch.no_grad()
def generate_greedy(
    model: TinyTransformer,
    prompt: list[int],
    max_new: int,
    stop_ids: frozenset[int] = frozenset(),
    use_cache: bool = True,
) -> list[int]:
    """Greedy decoding. Returns only the new tokens (the stop token, if hit, is included).

    With use_cache the prompt is encoded once and each step feeds one token. When the context would
    pass max_len, it falls back to re-encoding the last max_len tokens (no cache) for that step.
    """
    device = next(model.parameters()).device
    max_len = model.cfg.max_len
    seq = list(prompt)
    out: list[int] = []
    past: list[KV] | None = None
    for _ in range(max_new):
        if use_cache and past is not None and len(seq) <= max_len:
            logits, past = model(torch.tensor([[seq[-1]]], device=device), past)
        else:
            window = seq[-max_len:]
            logits, past = model(torch.tensor([window], device=device))
            if len(seq) > max_len or not use_cache:
                past = None
        nxt = int(logits[0, -1].argmax())
        seq.append(nxt)
        out.append(nxt)
        if nxt in stop_ids:
            break
    return out
