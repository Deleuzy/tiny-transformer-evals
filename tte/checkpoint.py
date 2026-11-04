"""Save and load a model together with its tokenizer and a little metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from tte.data import CharTokenizer
from tte.model import ModelConfig, TinyTransformer


def save(path: Path, model: TinyTransformer, tok: CharTokenizer, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"config": model.cfg.to_dict(), "state": state, "tokenizer": tok.to_json(), "meta": meta}, path)


def load(path: Path, device: torch.device | str = "cpu") -> tuple[TinyTransformer, CharTokenizer, dict[str, Any]]:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    model = TinyTransformer(ModelConfig(**blob["config"]))
    model.load_state_dict(blob["state"])
    model.to(device).eval()
    return model, CharTokenizer.from_json(blob["tokenizer"]), dict(blob["meta"])
