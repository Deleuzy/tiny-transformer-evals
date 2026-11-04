"""Inference: int8 dynamic quantization and latency at p50, p95 and p99, with and without a KV cache.

Dynamic quantization stores every nn.Linear weight as int8 and quantizes activations on the fly.
Embeddings and LayerNorm stay float32. It runs on CPU only.
"""

from __future__ import annotations

import io
import time
import warnings
from typing import Any

import numpy as np
import torch
from torch import nn

from tte.model import TinyTransformer, generate_greedy


def quantize_int8(model: TinyTransformer) -> TinyTransformer:
    # x86 / fbgemm on Intel and AMD CPUs (e.g. GitHub runners), qnnpack on Apple silicon and other ARM.
    engines = torch.backends.quantized.supported_engines
    torch.backends.quantized.engine = next(e for e in ("x86", "fbgemm", "qnnpack", *engines) if e in engines)
    warnings.filterwarnings("ignore", message=".*qnnpack incorrectly ignores reduce_range.*")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # torch.ao.quantization prints a deprecation notice in recent versions
        q = torch.ao.quantization.quantize_dynamic(model.cpu().eval(), {nn.Linear}, dtype=torch.qint8)
    assert isinstance(q, TinyTransformer)
    return q


def state_bytes(model: nn.Module) -> int:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell()


def percentiles(ms: list[float]) -> dict[str, float]:
    a = np.asarray(ms)
    return {f"p{p}_ms": round(float(np.percentile(a, p)), 3) for p in (50, 95, 99)} | {"mean_ms": round(float(a.mean()), 3)}


@torch.no_grad()
def latency(model: TinyTransformer, prompt: list[int], new_tokens: int, runs: int, warmup: int) -> dict[str, Any]:
    """Two measurements: one full forward pass over the prompt (prefill), and generating new_tokens
    greedily with and without the KV cache."""
    x = torch.tensor([prompt])
    res: dict[str, Any] = {}
    for name, fn in (
        ("prefill", lambda: model(x)),
        ("generate_kv_cache", lambda: generate_greedy(model, prompt, new_tokens, use_cache=True)),
        ("generate_no_cache", lambda: generate_greedy(model, prompt, new_tokens, use_cache=False)),
    ):
        for _ in range(warmup):
            fn()
        ms = []
        for _ in range(runs):
            t = time.perf_counter()
            fn()
            ms.append((time.perf_counter() - t) * 1000)
        res[name] = percentiles(ms)
    return res
