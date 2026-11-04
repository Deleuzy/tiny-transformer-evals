"""The baseline to beat: a character bigram model, P(next | previous char), from counts."""

from __future__ import annotations

import torch
from torch import Tensor


class BigramModel:
    """Add-alpha smoothed counts. alpha is fixed in the config before any evaluation."""

    def __init__(self, vocab_size: int, alpha: float) -> None:
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.log_probs = torch.full((vocab_size, vocab_size), -float(torch.log(torch.tensor(vocab_size))))

    def fit(self, ids: Tensor) -> BigramModel:
        counts = torch.zeros(self.vocab_size, self.vocab_size, dtype=torch.float64)
        pairs = ids[:-1] * self.vocab_size + ids[1:]
        counts.view(-1).index_add_(0, pairs, torch.ones_like(pairs, dtype=torch.float64))
        counts += self.alpha
        self.log_probs = (counts / counts.sum(dim=1, keepdim=True)).log().float()
        return self

    def logits(self, x: Tensor) -> Tensor:
        """[B, T] -> [B, T, V] log-probabilities of the next char (valid logits for softmax/CE)."""
        return self.log_probs.to(x.device)[x]
