"""Fixed eval sets with answer keys, built deterministically from the held-out test split.

Written to evals/*.jsonl so a reviewer can read every item and its answer. CI rebuilds them and
fails on any byte of difference.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from tte.data import ROOT, Splits

EVALS = ROOT / "evals"
LETTERS = "a-zæøåé"
_WORD = re.compile(r"[A-Za-zÆØÅæøåéÉ]+")
_LOWER_WORD = re.compile(rf"^[{LETTERS}]+$")

# Reversal task: prompt "vend:<word>=" and the answer is the word reversed, ended by a newline.
ID_LEN = (3, 8)
LONG_LEN = (9, 12)


def reversal_prompt(word: str) -> str:
    return f"\nvend:{word}="


def reversal_answer(word: str) -> str:
    return word[::-1] + "\n"


def lower_words(text: str, lo: int, hi: int) -> list[str]:
    """Unique lowercase words with lo <= len <= hi, in order of first appearance."""
    seen: dict[str, None] = {}
    for w in _WORD.findall(text):
        if lo <= len(w) <= hi and _LOWER_WORD.match(w):
            seen.setdefault(w, None)
    return list(seen)


def task_train_words(splits: Splits) -> list[str]:
    """Words for SFT and DPO come only from the training split, minus any word used in the eval sets."""
    held = set(lower_words(splits.test, *ID_LEN)) | set(lower_words(splits.test, *LONG_LEN))
    return [w for w in lower_words(splits.train, *ID_LEN) if w not in held]


def build_word_completion(test: str, n: int, rng: random.Random) -> list[dict[str, Any]]:
    """Context of 96 chars, then the first 3 letters of a word of 6+ letters. Answer: the rest of the word.

    The context must lie inside one paragraph: the test split joins blocks that are not neighbours in
    the books, so a context that crossed a line break could join unrelated text."""
    cands = [
        m
        for m in _WORD.finditer(test)
        if len(m.group()) >= 6 and m.start() >= 96 and "\n" not in test[m.start() - 96 : m.start()]
    ]
    items = []
    for m in sorted(rng.sample(cands, min(n, len(cands))), key=lambda m: m.start()):
        w = m.group()
        items.append({"prompt": test[m.start() - 96 : m.start()] + w[:3], "answer": w[3:], "word": w})
    return items


def _sentences(test: str) -> list[str]:
    out = []
    for para in test.split("\n"):
        for s in re.split(r"(?<=[.!?])\s+", para):
            if 30 <= len(s) <= 150 and len(s.split()) >= 5:
                out.append(s)
    return out


def build_minimal_pairs(test: str, n: int, rng: random.Random) -> list[dict[str, Any]]:
    """Each item: a real held-out sentence and a corrupted copy with the same characters.
    word_swap swaps two adjacent words; char_swap swaps two adjacent letters inside one word."""
    sents = _sentences(test)
    items: list[dict[str, Any]] = []
    for kind in ("word_swap", "char_swap"):
        pool = rng.sample(sents, len(sents))
        made = 0
        for s in pool:
            if made >= n:
                break
            words = s.split(" ")
            if kind == "word_swap":
                idx = [i for i in range(len(words) - 1) if words[i] != words[i + 1]]
                if not idx:
                    continue
                i = rng.choice(idx)
                words[i], words[i + 1] = words[i + 1], words[i]
            else:
                idx = [i for i, w in enumerate(words) if _WORD.fullmatch(w) and len(w) >= 4]
                if not idx:
                    continue
                i = rng.choice(idx)
                w = words[i]
                js = [j for j in range(1, len(w) - 2) if w[j] != w[j + 1]]
                if not js:
                    continue
                j = rng.choice(js)
                words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2 :]
            bad = " ".join(words)
            if bad != s:
                items.append({"kind": kind, "good": s, "bad": bad})
                made += 1
    return items


def build_reversal(test: str, lo: int, hi: int, n: int, rng: random.Random) -> list[dict[str, Any]]:
    words = lower_words(test, lo, hi)
    picked = rng.sample(words, min(n, len(words)))
    return [{"word": w, "prompt": reversal_prompt(w), "answer": reversal_answer(w)} for w in picked]


def build_all(splits: Splits, seed: int = 0) -> dict[str, list[dict[str, Any]]]:
    """Each set has its own generator (seeded from its name), so changing one set never reshuffles another."""

    def rng(name: str) -> random.Random:
        return random.Random(f"{seed}:{name}")  # str seeds are hashed with SHA-512: stable across runs

    return {
        "word_completion": build_word_completion(splits.test, 400, rng("word_completion")),
        "minimal_pairs": build_minimal_pairs(splits.test, 300, rng("minimal_pairs")),
        "reversal_in_dist": build_reversal(splits.test, *ID_LEN, 300, rng("reversal_in_dist")),
        "reversal_long": build_reversal(splits.test, *LONG_LEN, 200, rng("reversal_long")),
    }


def serialize(items: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(it, ensure_ascii=False, sort_keys=True) for it in items) + "\n"


def write_all(sets: dict[str, list[dict[str, Any]]], out: Path = EVALS) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, items in sets.items():
        (out / f"{name}.jsonl").write_text(serialize(items), encoding="utf-8")


def read_set(name: str, limit: int = 0) -> list[dict[str, Any]]:
    lines = (EVALS / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
    items = [json.loads(line) for line in lines if line]
    return items[:limit] if limit else items
