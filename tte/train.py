"""Pretraining from scratch with next-character prediction, plus the bigram baseline."""

from __future__ import annotations

import json
import math
import time
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from tte import checkpoint
from tte.bigram import BigramModel
from tte.config import RunConfig, pick_device, seed_everything
from tte.data import CharTokenizer, eval_windows, load_splits, random_batch
from tte.model import ModelConfig, TinyTransformer


def lr_at(step: int, base: float, warmup: int, total: int) -> float:
    """Linear warm-up, then cosine decay to 10% of the base rate."""
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def mean_nll(model: TinyTransformer, x: Tensor, y: Tensor, batch: int = 256) -> float:
    model.eval()
    dev = next(model.parameters()).device
    total, n = 0.0, 0
    for i in range(0, x.size(0), batch):
        xb, yb = x[i : i + batch].to(dev), y[i : i + batch].to(dev)
        logits, _ = model(xb)
        total += float(F.cross_entropy(logits.flatten(0, 1), yb.flatten(), reduction="sum"))
        n += yb.numel()
    return total / n


def pretrain(cfg: RunConfig) -> dict[str, Any]:
    seed_everything(cfg.seed)
    device = pick_device(cfg.device)
    splits = load_splits()
    tok = CharTokenizer.from_text(splits.train)
    train_ids = torch.tensor(tok.encode(splits.train))
    val_ids = torch.tensor(tok.encode(splits.val))
    mcfg = ModelConfig(vocab_size=tok.vocab_size, **cfg.model)
    model = TinyTransformer(mcfg).to(device)

    # The bigram baseline sees exactly the same training text.
    bigram = BigramModel(tok.vocab_size, cfg.bigram_alpha).fit(train_ids)
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"log_probs": bigram.log_probs, "alpha": cfg.bigram_alpha}, cfg.runs_dir / "bigram.pt")

    vx, vy = eval_windows(val_ids, mcfg.max_len)
    pick = torch.linspace(0, vx.size(0) - 1, min(cfg.train.eval_windows, vx.size(0))).long()
    vx, vy = vx[pick], vy[pick]
    bigram_val = float(F.cross_entropy(bigram.logits(vx).flatten(0, 1), vy.flatten()))

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.train.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.train.lr,
        betas=(0.9, 0.95),
    )
    gen = torch.Generator().manual_seed(cfg.seed)
    curve: list[dict[str, float]] = []
    running, count = 0.0, 0
    t0 = time.perf_counter()
    for step in range(cfg.train.steps):
        model.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg.train.lr, cfg.train.warmup, cfg.train.steps)
        x, y = random_batch(train_ids, cfg.train.batch, mcfg.max_len, gen)
        logits, _ = model(x.to(device))
        loss = F.cross_entropy(logits.flatten(0, 1), y.to(device).flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running += loss.item()
        count += 1
        last = step == cfg.train.steps - 1
        if (step + 1) % cfg.train.eval_every == 0 or last:
            curve.append({"step": step + 1, "train_loss": running / count, "val_loss": mean_nll(model, vx, vy)})
            running, count = 0.0, 0
            el = time.perf_counter() - t0
            print(f"step {step + 1:5d}  train {curve[-1]['train_loss']:.4f}  val {curve[-1]['val_loss']:.4f}  {el:.0f}s", flush=True)
    seconds = time.perf_counter() - t0

    checkpoint.save(cfg.runs_dir / "base.pt", model, tok, {"stage": "base", "config": cfg.name})
    log = {
        "config": cfg.raw,
        "device": str(device),
        "torch": torch.__version__,
        "params": model.num_params(),
        "vocab_size": tok.vocab_size,
        "train_chars": len(splits.train),
        "val_chars": len(splits.val),
        "test_chars": len(splits.test),
        "tokens_seen": cfg.train.steps * cfg.train.batch * mcfg.max_len,
        "train_seconds": round(seconds, 1),
        "bigram_val_loss_on_curve_windows": bigram_val,
        "curve": curve,
    }
    out = cfg.results_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_log.json").write_text(json.dumps(log, indent=2))
    plot_curve(log, out / "loss_curve.png")
    return log


def plot_curve(log: dict[str, Any], path: Any) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [c["step"] for c in log["curve"]]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=120)
    ax.plot(steps, [c["train_loss"] for c in log["curve"]], label="train (mean over interval)")
    ax.plot(steps, [c["val_loss"] for c in log["curve"]], label="validation")
    ax.axhline(log["bigram_val_loss_on_curve_windows"], color="grey", ls="--", label="bigram baseline (validation)")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy (nats per char)")
    ax.set_title(f"Pretraining, {log['params']:,} parameters")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def load_bigram(cfg: RunConfig) -> BigramModel:
    blob = torch.load(cfg.runs_dir / "bigram.pt", weights_only=True)
    lp: Tensor = blob["log_probs"]
    bg = BigramModel(lp.size(0), float(blob["alpha"]))
    bg.log_probs = lp
    return bg
