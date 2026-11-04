"""Run configuration. Every setting lives in configs/*.json, fixed before the run."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tte.data import ROOT


@dataclass(frozen=True)
class TrainCfg:
    steps: int
    batch: int
    lr: float
    warmup: int
    weight_decay: float
    eval_every: int
    eval_windows: int


@dataclass(frozen=True)
class SftCfg:
    steps: int
    task_batch: int
    lm_batch: int
    lr: float
    n_train_words: int


@dataclass(frozen=True)
class DpoCfg:
    steps: int
    batch: int
    lr: float
    beta: float
    n_pairs: int
    nll_weight: float = 0.0  # 0 = plain DPO; > 0 adds NLL on the chosen answer (RPO, Pang et al. 2024)


@dataclass(frozen=True)
class EvalCfg:
    max_items: int  # cap per capability eval file (0 = all)
    max_chars: int  # cap per robustness text (0 = all)


@dataclass(frozen=True)
class QuantCfg:
    runs: int
    warmup: int
    prompt_len: int
    new_tokens: int


@dataclass(frozen=True)
class RunConfig:
    name: str
    seed: int
    device: str
    threads: int
    model: dict[str, Any]
    train: TrainCfg
    bigram_alpha: float
    sft: SftCfg
    dpo: DpoCfg
    evals: EvalCfg
    quant: QuantCfg
    # Extra seeds for SFT and DPO (same pretrained base model) to show how much the shaping results vary.
    shaping_seeds: tuple[int, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @property
    def results_dir(self) -> Path:
        return ROOT / "results" / self.name

    @property
    def runs_dir(self) -> Path:
        return ROOT / "runs" / self.name


def load_config(path: str | Path) -> RunConfig:
    raw = json.loads(Path(path).read_text())
    return RunConfig(
        name=raw["name"],
        seed=raw["seed"],
        device=raw.get("device", "cpu"),
        threads=raw.get("threads", 4),
        model=raw["model"],
        train=TrainCfg(**raw["train"]),
        bigram_alpha=raw["bigram"]["alpha"],
        sft=SftCfg(**raw["sft"]),
        dpo=DpoCfg(**raw["dpo"]),
        evals=EvalCfg(**raw["evals"]),
        quant=QuantCfg(**raw["quant"]),
        shaping_seeds=tuple(raw.get("shaping_seeds", [])),
        raw=raw,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)
