"""Behaviour shaping at small scale: supervised fine-tuning (SFT), then DPO on preference pairs.

SFT teaches the reversal task ('\\nvend:hest=' -> 'tseh\\n'). Each SFT step mixes a task batch (loss on
the answer characters only) with an ordinary language-modelling batch, so the model is not trained
to forget Danish.

DPO (Rafailov et al. 2023) then starts from the SFT model. Pairs are built on-policy from words the
SFT model has not been trained on: chosen = the correct reversal, rejected = the SFT model's own
wrong answer. The loss is

    -log sigmoid(beta * [(log pi(c|x) - log ref(c|x)) - (log pi(r|x) - log ref(r|x))])

with the frozen SFT model as the reference. With nll_weight > 0 the loss also adds the mean
per-character negative log-likelihood of the chosen answer (the RPO variant), which keeps DPO from
lowering the probability of the correct answer along with the wrong one.
"""

from __future__ import annotations

import copy
import json
import random
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from tte import checkpoint
from tte.config import RunConfig, seed_everything
from tte.data import CharTokenizer, load_splits, random_batch
from tte.model import TinyTransformer, generate_greedy
from tte.tasks import reversal_answer, reversal_prompt, task_train_words

PAD = 0


def pack(tok: CharTokenizer, prompts: list[str], answers: list[str]) -> tuple[Tensor, Tensor, Tensor]:
    """Right-padded inputs [B, T], targets [B, T], and a mask [B, T] that is 1 on answer targets only."""
    seqs = [tok.encode(p + a) for p, a in zip(prompts, answers, strict=True)]
    t = max(len(s) for s in seqs) - 1
    x = torch.full((len(seqs), t), PAD)
    y = torch.full((len(seqs), t), PAD)
    m = torch.zeros((len(seqs), t))
    for i, (s, p) in enumerate(zip(seqs, prompts, strict=True)):
        n = len(s) - 1
        x[i, :n] = torch.tensor(s[:-1])
        y[i, :n] = torch.tensor(s[1:])
        m[i, len(p) - 1 : n] = 1.0  # targets from the first answer character on
    return x, y, m


def answer_logprob(model: TinyTransformer, x: Tensor, y: Tensor, m: Tensor) -> Tensor:
    """Sum of log-probabilities of the answer characters, one value per row."""
    lp = torch.log_softmax(model(x)[0], dim=-1).gather(2, y.unsqueeze(2)).squeeze(2)
    return (lp * m).sum(dim=1)


def split_words(cfg: RunConfig) -> tuple[list[str], list[str]]:
    words = task_train_words(load_splits())
    rng = random.Random(cfg.seed)
    rng.shuffle(words)
    n = cfg.sft.n_train_words
    return words[:n], words[n:]  # SFT words, DPO words (never seen in SFT)


def sft(cfg: RunConfig) -> dict[str, Any]:
    seed_everything(cfg.seed + 1)
    model, tok, _ = checkpoint.load(cfg.runs_dir / "base.pt")
    train_ids = torch.tensor(tok.encode(load_splits().train))
    sft_words, _ = split_words(cfg)
    gen = torch.Generator().manual_seed(cfg.seed + 1)
    rng = random.Random(cfg.seed + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.sft.lr, weight_decay=0.0)
    log: list[dict[str, float]] = []
    t0 = time.perf_counter()
    for step in range(cfg.sft.steps):
        model.train()
        ws = rng.sample(sft_words, cfg.sft.task_batch)
        x, y, m = pack(tok, [reversal_prompt(w) for w in ws], [reversal_answer(w) for w in ws])
        logits, _ = model(x)
        ce = F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="none").view_as(m)
        task_loss = (ce * m).sum() / m.sum()
        lx, ly = random_batch(train_ids, cfg.sft.lm_batch, model.cfg.max_len, gen)
        lm_logits, _ = model(lx)
        lm_loss = F.cross_entropy(lm_logits.flatten(0, 1), ly.flatten())
        loss = task_loss + lm_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % max(1, cfg.sft.steps // 20) == 0 or step == cfg.sft.steps - 1:
            log.append({"step": step + 1, "task_loss": task_loss.item(), "lm_loss": lm_loss.item()})
            print(f"sft {step + 1:4d} task {task_loss.item():.4f} lm {lm_loss.item():.4f}", flush=True)
    checkpoint.save(cfg.runs_dir / "sft.pt", model, tok, {"stage": "sft", "config": cfg.name})
    out = {"train_words": len(sft_words), "seconds": round(time.perf_counter() - t0, 1), "log": log}
    (cfg.results_dir / "sft_log.json").write_text(json.dumps(out, indent=2))
    return out


@torch.no_grad()
def build_pairs(model: TinyTransformer, tok: CharTokenizer, words: list[str], n: int) -> tuple[list[dict[str, str]], int]:
    """Chosen = correct reversal, rejected = the SFT model's greedy answer when it is wrong."""
    model.eval()
    stop = frozenset({tok.stoi["\n"]})
    pairs: list[dict[str, str]] = []
    tried = 0
    for w in words:
        if len(pairs) >= n:
            break
        tried += 1
        out = tok.decode(generate_greedy(model, tok.encode(reversal_prompt(w)), len(w) + 2, stop))
        if not out.endswith("\n"):
            out = out + "\n"
        if out != reversal_answer(w):
            pairs.append({"word": w, "prompt": reversal_prompt(w), "chosen": reversal_answer(w), "rejected": out})
    return pairs, tried


def dpo(cfg: RunConfig) -> dict[str, Any]:
    seed_everything(cfg.seed + 2)
    policy, tok, _ = checkpoint.load(cfg.runs_dir / "sft.pt")
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    _, dpo_words = split_words(cfg)
    pairs, tried = build_pairs(policy, tok, dpo_words, cfg.dpo.n_pairs)
    (cfg.results_dir / "dpo_pairs.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False, sort_keys=True) for p in pairs) + "\n", encoding="utf-8"
    )
    if not pairs:
        raise RuntimeError("the SFT model answered every DPO word correctly; no preference pairs to train on")
    rng = random.Random(cfg.seed + 2)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.dpo.lr, weight_decay=0.0)
    log: list[dict[str, float]] = []
    t0 = time.perf_counter()
    for step in range(cfg.dpo.steps):
        policy.train()
        batch = [pairs[rng.randrange(len(pairs))] for _ in range(cfg.dpo.batch)]
        prompts = [b["prompt"] for b in batch]
        cx, cy, cm = pack(tok, prompts, [b["chosen"] for b in batch])
        rx, ry, rm = pack(tok, prompts, [b["rejected"] for b in batch])
        with torch.no_grad():
            ref_c, ref_r = answer_logprob(ref, cx, cy, cm), answer_logprob(ref, rx, ry, rm)
        pol_c, pol_r = answer_logprob(policy, cx, cy, cm), answer_logprob(policy, rx, ry, rm)
        margin = cfg.dpo.beta * ((pol_c - ref_c) - (pol_r - ref_r))
        loss = -F.logsigmoid(margin).mean()
        if cfg.dpo.nll_weight > 0:
            loss = loss + cfg.dpo.nll_weight * (-pol_c / cm.sum(dim=1)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        if (step + 1) % max(1, cfg.dpo.steps // 20) == 0 or step == cfg.dpo.steps - 1:
            rec = {
                "step": step + 1,
                "loss": loss.item(),
                "reward_accuracy": (margin > 0).float().mean().item(),
                "mean_margin": margin.mean().item(),
            }
            log.append(rec)
            print(f"dpo {step + 1:4d} loss {rec['loss']:.4f} acc {rec['reward_accuracy']:.3f}", flush=True)
    checkpoint.save(cfg.runs_dir / "dpo.pt", policy, tok, {"stage": "dpo", "config": cfg.name})
    out = {
        "pairs": len(pairs),
        "words_tried_to_find_pairs": tried,
        "sft_error_rate_on_dpo_words": round(len(pairs) / tried, 4),
        "seconds": round(time.perf_counter() - t0, 1),
        "log": log,
    }
    (cfg.results_dir / "dpo_log.json").write_text(json.dumps(out, indent=2))
    return out
