"""Inference: int8 dynamic quantization and latency at p50, p95 and p99, with and without a KV cache.

Dynamic quantization stores every nn.Linear weight as int8 and quantizes activations on the fly.
Embeddings and LayerNorm stay float32. It runs on CPU only.
"""

from __future__ import annotations

import io
import time
import warnings
from collections.abc import Callable
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
def latency(
    models: dict[str, TinyTransformer], prompt: list[int], new_tokens: int, runs: int, warmup: int
) -> dict[str, dict[str, Any]]:
    """Per model, three measurements: one full forward pass over the prompt (prefill), and generating
    new_tokens greedily with and without the KV cache.

    Runs are interleaved across models, and the model order flips on every run, so slow periods on a
    busy laptop (thermal throttling, background work) fall on all models alike rather than on one.
    """
    x = torch.tensor([prompt])

    def tasks(m: TinyTransformer) -> dict[str, Callable[[], object]]:
        return {
            "prefill": lambda: m(x),
            "generate_kv_cache": lambda: generate_greedy(m, prompt, new_tokens, use_cache=True),
            "generate_no_cache": lambda: generate_greedy(m, prompt, new_tokens, use_cache=False),
        }

    fns = {name: tasks(m) for name, m in models.items()}
    times: dict[str, dict[str, list[float]]] = {name: {k: [] for k in fns[name]} for name in models}
    names = list(models)
    for r in range(warmup + runs):
        for name in names if r % 2 == 0 else names[::-1]:
            for kind, fn in fns[name].items():
                t = time.perf_counter()
                fn()
                if r >= warmup:
                    times[name][kind].append((time.perf_counter() - t) * 1000)
    return {name: {kind: percentiles(ms) for kind, ms in kinds.items()} for name, kinds in times.items()}
