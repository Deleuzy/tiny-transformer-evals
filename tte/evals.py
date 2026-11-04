"""The eval harness. One call runs every eval on one model and returns a JSON-ready dict.

Both the transformer and the bigram baseline go through the same `Scorer` interface, so every
number for one is computed exactly the same way as for the other.
"""

from __future__ import annotations

import math
import random
from functools import cache
from typing import Any, Protocol

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tte.data import OOD_BOOKS, CharTokenizer, eval_windows, load_splits, read_book
from tte.tasks import _WORD, read_set

# What each eval does and does not measure. Copied into every results file next to the numbers.
DESCRIPTIONS: dict[str, dict[str, str]] = {
    "heldout": {
        "measures": "Average next-character cross-entropy on the held-out test split (paragraph blocks never "
        "trained on), reported as perplexity and bits per character with a bootstrap 95% interval over windows.",
        "does_not_measure": "Whether generated text is coherent or useful. Same books and spelling as training, "
        "so it says nothing about other domains. Windows are non-overlapping, so early characters in each "
        "window see little context. The interval covers only the choice of test windows (which are "
        "contiguous, so it is somewhat too narrow), not variation between training runs.",
    },
    "word_completion": {
        "measures": "Exact-match rate when the model, given 96 characters of held-out text from the same "
        "paragraph and the first 3 letters of a word of 6 or more letters, completes the word by greedy "
        "decoding. Also the share of misses that are another word found in the training text.",
        "does_not_measure": "Understanding. Frequent words and names repeated within a story are easy; "
        "a single gold answer marks plausible alternatives wrong.",
    },
    "minimal_pairs": {
        "measures": "How often the model gives a real held-out sentence a higher likelihood than a copy with "
        "two adjacent words swapped (word order) or two adjacent letters swapped (spelling). Chance is 50%.",
        "does_not_measure": "Grammar in general; only these two local corruptions. A model can score well "
        "from character statistics alone, which is why the bigram baseline is reported too.",
    },
    "reversal": {
        "measures": "Exact-match rate on the SFT task: given '\\nvend:<word>=' write the word reversed and a "
        "newline. In-distribution words have 3 to 8 letters, as in training; 'long' words have 9 to 12 letters, "
        "a length never seen in training. mean_length_error is output length minus word length (positive = too "
        "long). char_accuracy compares characters position by position, so one dropped letter makes every "
        "later position wrong. All eval words come from the test split and are excluded from the training "
        "words.",
        "does_not_measure": "Instruction following in general. It is one narrow synthetic task.",
    },
    "robustness": {
        "measures": "Bits per character on perturbed or out-of-distribution text, and the change from clean "
        "test text. Typos: each letter of the context is replaced, with probability p, by a different letter "
        "of the same case, while the targets stay the clean text (how well the model still predicts the right "
        "text from a noisy context); the measured share of changed letters is recorded. Case changes and "
        "other books are scored on their own text.",
        "does_not_measure": "Adversarial robustness: the noise is random, not searched for. The unknown-"
        "character rate is reported because unknown characters cannot be predicted at all.",
    },
}


class Scorer(Protocol):
    max_len: int

    def logits(self, x: Tensor) -> Tensor:
        """[B, T] token ids -> [B, T, V] unnormalised next-token scores."""
        ...


class TransformerScorer:
    def __init__(self, model: nn.Module, max_len: int) -> None:
        self.model = model.eval()
        self.max_len = max_len

    @torch.no_grad()
    def logits(self, x: Tensor) -> Tensor:
        out: Tensor = self.model(x)[0]
        return out


class BigramScorer:
    def __init__(self, bigram: Any, max_len: int) -> None:
        self.bigram = bigram
        self.max_len = max_len

    def logits(self, x: Tensor) -> Tensor:
        out: Tensor = self.bigram.logits(x)
        return out


def token_nll(scorer: Scorer, x: Tensor, y: Tensor, batch: int = 128) -> Tensor:
    """Per-window mean NLL [N] for windows x -> targets y."""
    out = []
    for i in range(0, x.size(0), batch):
        lg = scorer.logits(x[i : i + batch])
        nll = F.cross_entropy(lg.flatten(0, 1), y[i : i + batch].flatten(), reduction="none")
        out.append(nll.view(lg.size(0), -1).mean(dim=1))
    return torch.cat(out)


def bootstrap_ci(values: Tensor, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, values.numel(), (n_boot, values.numel()), generator=g)
    means = values[idx].mean(dim=1).sort().values
    return float(means[int(0.025 * n_boot)]), float(means[int(0.975 * n_boot) - 1])


def nll_summary(per_window: Tensor) -> dict[str, Any]:
    nll = float(per_window.mean())
    lo, hi = bootstrap_ci(per_window)
    ln2 = math.log(2)
    return {
        "nll_nats": round(nll, 4),
        "perplexity": round(math.exp(nll), 3),
        "bits_per_char": round(nll / ln2, 4),
        "bits_per_char_ci95": [round(lo / ln2, 4), round(hi / ln2, 4)],
        "windows": per_window.numel(),
    }


def sequence_logprob(scorer: Scorer, context: list[int], cont: list[int]) -> float:
    """log P(cont | context). Each continuation character is scored given up to max_len previous
    characters; long sequences are scored in chunks of max_len // 2 targets with a sliding window."""
    if not context:
        raise ValueError("context must hold at least one token")
    seq = context + cont
    total, pos, step = 0.0, len(context), max(1, scorer.max_len // 2)
    while pos < len(seq):
        end = min(len(seq), pos + step)
        a = max(0, end - 1 - scorer.max_len)  # inputs seq[a:end-1] predict targets seq[a+1:end]
        lp = torch.log_softmax(scorer.logits(torch.tensor([seq[a : end - 1]]))[0], dim=-1)
        tgt = torch.tensor(seq[pos:end])
        total += float(lp[pos - 1 - a : end - 1 - a].gather(1, tgt.unsqueeze(1)).sum())
        pos = end
    return total


def greedy(scorer: Scorer, context: list[int], max_new: int, stop: frozenset[int]) -> list[int]:
    seq, out = list(context), []
    for _ in range(max_new):
        nxt = int(scorer.logits(torch.tensor([seq[-scorer.max_len :]]))[0, -1].argmax())
        out.append(nxt)
        seq.append(nxt)
        if nxt in stop:
            break
    return out


def eval_heldout(scorer: Scorer, tok: CharTokenizer, text: str) -> dict[str, Any]:
    x, y = eval_windows(torch.tensor(tok.encode(text)), scorer.max_len)
    return {"chars_scored": y.numel(), **nll_summary(token_nll(scorer, x, y))}


@cache
def training_words() -> frozenset[str]:
    """Every word (lowercased) in the training split: used to tell a real-word miss from a non-word."""
    return frozenset(w.lower() for w in _WORD.findall(load_splits().train))


def eval_word_completion(scorer: Scorer, tok: CharTokenizer, limit: int) -> dict[str, Any]:
    items = read_set("word_completion", limit)
    stop = frozenset(i for c, i in tok.stoi.items() if not c.isalpha())
    vocab = training_words()
    hits, real_misses = 0, 0
    preds: list[dict[str, Any]] = []
    for idx, it in enumerate(items):
        out = tok.decode(greedy(scorer, tok.encode(it["prompt"]), len(it["answer"]) + 1, stop))
        guess = "".join(ch for ch in out if ch.isalpha())
        ok = guess == it["answer"]
        hits += ok
        word = it["word"][:3] + guess
        real = not ok and word.lower() in vocab
        real_misses += real
        preds.append({"set": "word_completion", "id": idx, "gold": it["word"], "output": word, "correct": ok,
                      "miss_is_training_word": real})
    misses = len(items) - hits
    return {
        "n": len(items),
        "exact_match": round(hits / len(items), 4),
        "misses": misses,
        "misses_that_are_training_words": round(real_misses / misses, 4) if misses else 0.0,
        "examples": [{"word": p["gold"], "model": p["output"], "correct": p["correct"]} for p in preds[:5]],
        "predictions": preds,
    }


def eval_minimal_pairs(scorer: Scorer, tok: CharTokenizer, limit: int) -> dict[str, Any]:
    items = read_set("minimal_pairs")
    res: dict[str, Any] = {}
    for kind in ("word_swap", "char_swap"):
        sub = [it for it in items if it["kind"] == kind]
        sub = sub[:limit] if limit else sub
        wins = 0
        preds: list[dict[str, Any]] = []
        ctx = tok.encode("\n")
        for idx, it in enumerate(sub):
            good = sequence_logprob(scorer, ctx, tok.encode(it["good"]))
            bad = sequence_logprob(scorer, ctx, tok.encode(it["bad"]))
            wins += good > bad
            preds.append({"set": f"minimal_pairs_{kind}", "id": idx, "logprob_good": round(good, 4),
                          "logprob_bad": round(bad, 4), "correct": good > bad})
        res[kind] = {"n": len(sub), "accuracy": round(wins / len(sub), 4), "predictions": preds}
    return res


def eval_reversal(scorer: Scorer, tok: CharTokenizer, limit: int) -> dict[str, Any]:
    stop = frozenset({tok.stoi["\n"]})
    res: dict[str, Any] = {}
    for name in ("reversal_in_dist", "reversal_long"):
        items = read_set(name, limit)
        hits, char_acc, len_err = 0, 0.0, 0
        preds: list[dict[str, Any]] = []
        for idx, it in enumerate(items):
            out = tok.decode(greedy(scorer, tok.encode(it["prompt"]), len(it["answer"]) + 2, stop))
            gold = it["answer"]
            hits += out == gold
            char_acc += sum(a == b for a, b in zip(out, gold, strict=False)) / len(gold)
            len_err += len(out.rstrip("\n")) - len(it["word"])
            preds.append({"set": name, "id": idx, "gold": it["word"], "output": out.rstrip("\n"), "correct": out == gold})
        res[name.removeprefix("reversal_")] = {
            "n": len(items),
            "exact_match": round(hits / len(items), 4),
            "char_accuracy": round(char_acc / len(items), 4),
            "mean_length_error": round(len_err / len(items), 3),
            "examples": [{"word": p["gold"], "model": p["output"], "correct": p["correct"]} for p in preds[:4]],
            "predictions": preds,
        }
    return res


TYPO_LETTERS = "abcdefghijklmnopqrstuvwxyzæøå"


def add_typos(text: str, p: float, rng: random.Random) -> str:
    """Replace each letter, with probability p, by a different letter of the same case."""
    out = []
    for c in text:
        low = c.lower()
        if low in TYPO_LETTERS and rng.random() < p:
            new = rng.choice([x for x in TYPO_LETTERS if x != low])
            c = new.upper() if c.isupper() else new
        out.append(c)
    return "".join(out)


def eval_robustness(scorer: Scorer, tok: CharTokenizer, test: str, max_chars: int) -> dict[str, Any]:
    clean = test[:max_chars] if max_chars else test
    ids = torch.tensor(tok.encode(clean))
    x, y = eval_windows(ids, scorer.max_len)
    base = float(token_nll(scorer, x, y).mean()) / math.log(2)
    res: dict[str, Any] = {"clean": {"bits_per_char": round(base, 4)}}

    def record(name: str, bpc: float, unk: float) -> None:
        res[name] = {"bits_per_char": round(bpc, 4), "delta_vs_clean": round(bpc - base, 4), "unknown_char_rate": round(unk, 5)}

    letters = sum(c.lower() in TYPO_LETTERS for c in clean)
    for p in (0.05, 0.15):
        noisy = add_typos(clean, p, random.Random(1000 + int(p * 100)))
        assert len(noisy) == len(clean)  # same length, so noisy inputs line up with the clean targets y
        nx, _ = eval_windows(torch.tensor(tok.encode(noisy)), scorer.max_len)
        name = f"typos_{int(p * 100)}pct_context"
        record(name, float(token_nll(scorer, nx, y).mean()) / math.log(2), tok.unknown_rate(noisy))
        res[name]["changed_letter_rate"] = round(sum(a != b for a, b in zip(clean, noisy, strict=True)) / letters, 4)
    for name, fn in (("lowercase", str.lower), ("uppercase", str.upper)):
        t = fn(clean)
        tx, ty = eval_windows(torch.tensor(tok.encode(t)), scorer.max_len)
        record(name, float(token_nll(scorer, tx, ty).mean()) / math.log(2), tok.unknown_rate(t))
    for name, book in OOD_BOOKS.items():
        t = read_book(book)[2000:]  # skip the title pages
        t = t[: max_chars or len(clean)]
        tx, ty = eval_windows(torch.tensor(tok.encode(t)), scorer.max_len)
        record(f"ood_{name}", float(token_nll(scorer, tx, ty).mean()) / math.log(2), tok.unknown_rate(t))
    return res


def pop_predictions(res: dict[str, Any]) -> list[dict[str, Any]]:
    """Remove every per-item prediction list from a run_all result (in place) and return them flat."""
    out: list[dict[str, Any]] = []
    for key in ("word_completion", "minimal_pairs", "reversal"):
        sec = res[key]
        for part in [sec] if "predictions" in sec else list(sec.values()):
            out.extend(part.pop("predictions"))
    return out


def run_all(scorer: Scorer, tok: CharTokenizer, max_items: int, max_chars: int, label: str) -> dict[str, Any]:
    test = load_splits().test
    heldout_text = test[:max_chars] if max_chars else test
    torch.manual_seed(0)
    return {
        "model": label,
        "descriptions": DESCRIPTIONS,
        "heldout": eval_heldout(scorer, tok, heldout_text),
        "word_completion": eval_word_completion(scorer, tok, max_items),
        "minimal_pairs": eval_minimal_pairs(scorer, tok, max_items),
        "reversal": eval_reversal(scorer, tok, max_items),
        "robustness": eval_robustness(scorer, tok, test, max_chars),
    }
