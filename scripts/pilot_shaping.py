"""Pilot for the DPO settings, scored ONLY on validation-split words, never on the test eval sets.

Run once after pretraining:  uv run python scripts/pilot_shaping.py
It runs SFT with the settings in configs/full.json, then DPO with each candidate below, and writes
results/full/pilot_shaping.json. The rule, fixed before looking: pick the candidate with the highest
in-distribution exact match on validation words; break ties on long words. The chosen candidate is
then copied into configs/full.json and evaluated once on the test eval sets by `tte all`.
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tte import checkpoint
from tte.config import load_config
from tte.data import ROOT, load_splits
from tte.model import generate_greedy
from tte.shaping import dpo, sft
from tte.tasks import ID_LEN, LONG_LEN, lower_words, reversal_answer, reversal_prompt, task_train_words

CANDIDATES: list[dict[str, float]] = [
    {"lr": 1e-4, "nll_weight": 0.0},
    {"lr": 1e-5, "nll_weight": 0.0},
    {"lr": 1e-4, "nll_weight": 1.0},
    {"lr": 3e-5, "nll_weight": 1.0},
]


def score(path: Path, words_id: list[str], words_long: list[str]) -> dict[str, Any]:
    model, tok, _ = checkpoint.load(path)
    stop = frozenset({tok.stoi["\n"]})
    out: dict[str, Any] = {}
    for name, words in (("in_dist", words_id), ("long", words_long)):
        hits, len_err = 0, 0
        for w in words:
            o = tok.decode(generate_greedy(model, tok.encode(reversal_prompt(w)), len(w) + 2, stop))
            hits += o == reversal_answer(w)
            len_err += len(o.rstrip("\n")) - len(w)
        out[name] = {"n": len(words), "exact_match": round(hits / len(words), 4),
                     "mean_length_error": round(len_err / len(words), 3)}
    return out


def main() -> None:
    torch.use_deterministic_algorithms(True)
    torch.set_flush_denormal(True)
    full = load_config(ROOT / "configs" / "full.json")
    torch.set_num_threads(full.threads)
    cfg = replace(full, name="pilot")
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(full.runs_dir / "base.pt", cfg.runs_dir / "base.pt")
    s = load_splits()
    train_words = set(task_train_words(s))
    words_id = [w for w in lower_words(s.val, *ID_LEN) if w not in train_words][:200]
    words_long = [w for w in lower_words(s.val, *LONG_LEN) if w not in train_words][:150]
    sft(cfg)
    res: dict[str, Any] = {"split": "validation", "rule": __doc__.split("The rule, fixed before looking: ")[1].split(".")[0],
                           "sft": score(cfg.runs_dir / "sft.pt", words_id, words_long), "dpo": []}
    for cand in CANDIDATES:
        c = replace(cfg, dpo=replace(cfg.dpo, lr=cand["lr"], nll_weight=cand["nll_weight"]))
        log = dpo(c)
        res["dpo"].append({**cand, "pairs": log["pairs"], **score(cfg.runs_dir / "dpo.pt", words_id, words_long)})
        print(res["dpo"][-1], flush=True)
    best = max(res["dpo"], key=lambda r: (r["in_dist"]["exact_match"], r["long"]["exact_match"]))
    res["chosen"] = {"lr": best["lr"], "nll_weight": best["nll_weight"]}
    (full.results_dir / "pilot_shaping.json").write_text(json.dumps(res, indent=2) + "\n")
    shutil.rmtree(cfg.runs_dir)
    shutil.rmtree(cfg.results_dir)
    print("chosen", res["chosen"])


if __name__ == "__main__":
    main()
