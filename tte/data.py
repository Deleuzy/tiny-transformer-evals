"""Corpus loading, cleaning, the train/validation/test split, and a character tokenizer.

All texts are public domain Project Gutenberg books, committed under data/raw/ with fixed SHA-256 sums.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


@dataclass(frozen=True)
class Book:
    file: str
    title: str
    author: str
    language: str
    sha256: str


TRAIN_BOOKS = (
    Book("pg46278.txt", "Fra Mindebo: Jyske Folkeæventyr", "Evald Tang Kristensen", "da",
         "638d20de9742018fcabe05fc3bea5fc20b9a0db0762e8537236844638d76d998"),
    Book("pg13175.txt", "Ved Vejen", "Herman Bang", "da",
         "c7051312723dda3929e41d95b13926f50e1a8606c97671b9b6a41d4d8f38d7f8"),
    Book("pg10686.txt", "Tine", "Herman Bang", "da",
         "cddc3e56dc4b822cc75166ccff9fb8a90f020c455d6d6fdb26e929619762dea3"),
    Book("pg51384.txt", "Lykke-Per, Første Del", "Henrik Pontoppidan", "da",
         "546fefb78fe0077c8ebbebc1f28105b9a025aeb2748737ca61e9338248541b55"),
)
OOD_BOOKS = {
    "danish_other_author": Book("pg44275.txt", "Fru Marie Grubbe", "J. P. Jacobsen", "da",
                                "d60cfc0494503931c75c3a64043e9d94683d8e282f748cf98f97ceb74a07942c"),
    "english": Book("pg1597.txt", "Andersen's Fairy Tales (English translation)", "H. C. Andersen", "en",
                    "f886b6c92c0c7884cb64059d84363f10977cfb12d0bc0b120c2207ce2f57b2f2"),
}

_START = re.compile(r"\*\*\* ?START OF TH(E|IS) PROJECT GUTENBERG EBOOK[^\n]*\n")
_END = re.compile(r"\*\*\* ?END OF TH(E|IS) PROJECT GUTENBERG EBOOK")


def read_book(book: Book) -> str:
    raw = (RAW / book.file).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != book.sha256:
        raise ValueError(f"{book.file}: sha256 {digest} does not match the pinned {book.sha256}")
    return clean_gutenberg(raw.decode("utf-8"))


def clean_gutenberg(text: str) -> str:
    """Body between the Gutenberg markers, one paragraph per line, whitespace collapsed."""
    text = text.replace("\r\n", "\n").replace("﻿", "")
    start, end = _START.search(text), _END.search(text)
    if start is None or end is None:
        raise ValueError("Gutenberg start or end marker not found")
    body = text[start.end() : end.start()]
    paragraphs = []
    for block in re.split(r"\n\s*\n", body):
        para = " ".join(block.split())
        # Drop page numbers, rules and transcriber notes; keep real prose and dialogue.
        if len(para) < 2 or set(para) <= set("*-_=0123456789. "):
            continue
        para = para.replace("_", "")
        paragraphs.append(para)
    return "\n".join(paragraphs) + "\n"


@dataclass(frozen=True)
class Splits:
    train: str
    val: str
    test: str


def split_paragraphs(texts: list[str], block: int = 20) -> Splits:
    """Deterministic split. Paragraphs are grouped in consecutive blocks of `block`;
    block index mod 10 == 8 goes to validation, == 9 to test, the rest to training.
    Whole blocks keep neighbouring sentences of the held-out text out of training."""
    parts: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for text in texts:
        paras = text.strip("\n").split("\n")
        for b in range(0, len(paras), block):
            r = (b // block) % 10
            key = "val" if r == 8 else "test" if r == 9 else "train"
            parts[key].extend(paras[b : b + block])
    return Splits(*("\n".join(parts[k]) + "\n" for k in ("train", "val", "test")))


def load_splits() -> Splits:
    return split_paragraphs([read_book(b) for b in TRAIN_BOOKS])


# Characters the reversal task needs even if the training text never contains them.
TASK_CHARS = "vend:=\n"
UNK = "\x00"


class CharTokenizer:
    """One id per character seen in training text; id 0 is the unknown character."""

    def __init__(self, chars: list[str]) -> None:
        if chars[0] != UNK:
            raise ValueError("first character must be the unknown marker")
        self.chars = chars
        self.stoi = {c: i for i, c in enumerate(chars)}

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        return cls([UNK, *sorted(set(text + TASK_CHARS) - {UNK})])

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.chars[i] for i in ids)

    def unknown_rate(self, text: str) -> float:
        return sum(c not in self.stoi for c in text) / max(1, len(text))

    def to_json(self) -> str:
        return json.dumps(self.chars, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> CharTokenizer:
        chars = json.loads(s)
        if not isinstance(chars, list) or not all(isinstance(c, str) for c in chars):
            raise ValueError("tokenizer JSON must be a list of strings")
        return cls(chars)


def random_batch(data: Tensor, batch: int, ctx: int, gen: torch.Generator) -> tuple[Tensor, Tensor]:
    """batch random windows of ctx+1 tokens: inputs [B, ctx] and targets shifted by one."""
    starts = torch.randint(0, data.numel() - ctx - 1, (batch,), generator=gen)
    x = torch.stack([data[s : s + ctx] for s in starts.tolist()])
    y = torch.stack([data[s + 1 : s + ctx + 1] for s in starts.tolist()])
    return x, y


def eval_windows(data: Tensor, ctx: int) -> tuple[Tensor, Tensor]:
    """Non-overlapping windows covering the text once: every target char is scored exactly once,
    given up to ctx previous chars (fewer at the start of each window)."""
    n = (data.numel() - 1) // ctx
    x = data[: n * ctx].view(n, ctx)
    y = data[1 : n * ctx + 1].view(n, ctx)
    return x, y
