"""Scaled dot-product attention and multi-head self-attention, written by hand.

Shapes use these letters: B batch, H heads, Tq query length, Tk key length, D head size, C model width.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

# A layer's cached keys and values, each [B, H, T_past, D].
KV = tuple[Tensor, Tensor]


def causal_mask(t_q: int, t_k: int, device: torch.device | None = None) -> Tensor:
    """Boolean mask [Tq, Tk]; True where query i may look at key j.

    The queries are the last Tq positions of a sequence of length Tk (Tq == Tk without a cache,
    Tq == 1 when decoding with a cache). Query i sits at absolute position Tk - Tq + i and may see
    every key at that position or earlier, never a later one.
    """
    if t_q > t_k:
        raise ValueError(f"more queries ({t_q}) than keys ({t_k})")
    offset = t_k - t_q
    q_pos = torch.arange(t_q, device=device).unsqueeze(1) + offset
    k_pos = torch.arange(t_k, device=device).unsqueeze(0)
    return k_pos <= q_pos


def attention_weights(q: Tensor, k: Tensor, mask: Tensor | None = None) -> Tensor:
    """softmax(q k^T / sqrt(D)) with masked positions set to probability 0. Returns [..., Tq, Tk]."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1)


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(D)) V.

    q: [..., Tq, D], k: [..., Tk, D], v: [..., Tk, Dv], mask: broadcastable to [..., Tq, Tk] (True = keep).
    """
    return attention_weights(q, k, mask) @ v


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention with an optional key/value cache for decoding."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        return x.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D]

    def forward(self, x: Tensor, past: KV | None = None) -> tuple[Tensor, KV]:
        b, t, c = x.shape
        q, k, v = self.qkv(x).split(c, dim=-1)
        q, k, v = self._split(q), self._split(k), self._split(v)
        if past is not None:
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        mask = causal_mask(t, k.size(2), device=x.device)
        weights = self.dropout(attention_weights(q, k, mask))
        out = (weights @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(out), (k, v)
