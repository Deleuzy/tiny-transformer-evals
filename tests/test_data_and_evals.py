import math
import random

import torch

from tte.bigram import BigramModel
from tte.data import CharTokenizer, clean_gutenberg, eval_windows, load_splits, split_paragraphs
from tte.evals import TYPO_LETTERS, BigramScorer, add_typos, greedy, sequence_logprob, token_nll
from tte.shaping import pack
from tte.stats import mcnemar_exact
from tte.tasks import EVALS, build_all, reversal_answer, reversal_prompt, serialize, task_train_words


def test_clean_gutenberg_keeps_body_only() -> None:
    raw = "Header\r\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\r\nOne\r\nline.\r\n\r\n12\r\n\r\nTwo.\r\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\r\nLicense"
    assert clean_gutenberg(raw) == "One line.\nTwo.\n"


def test_split_is_disjoint_and_complete() -> None:
    paras = [f"p{i}" for i in range(400)]
    s = split_paragraphs(["\n".join(paras)])
    got = [set(x.strip().split("\n")) for x in (s.train, s.val, s.test)]
    assert not (got[0] & got[1]) and not (got[0] & got[2]) and not (got[1] & got[2])
    assert set().union(*got) == set(paras)
    assert len(got[1]) == len(got[2]) == 40


def test_real_corpus_checksums_and_sizes() -> None:
    s = load_splits()
    assert len(s.train) > 900_000 and len(s.test) > 100_000
    assert s.test.split("\n")[0] not in s.train


def test_tokenizer_roundtrip_and_unknown() -> None:
    tok = CharTokenizer.from_text("hej med dig")
    assert tok.decode(tok.encode("hej dig")) == "hej dig"
    assert tok.encode("q") == [0]
    assert CharTokenizer.from_json(tok.to_json()).chars == tok.chars
    for c in "vend:=\n":
        assert c in tok.stoi


def test_eval_windows_score_each_target_once() -> None:
    ids = torch.arange(25)
    x, y = eval_windows(ids, 8)
    assert x.shape == y.shape == (3, 8)
    assert torch.equal(y.flatten(), torch.arange(1, 25))


def test_bigram_learns_counts_and_is_normalised() -> None:
    ids = torch.tensor([1, 2, 1, 2, 1, 2, 1, 3])
    bg = BigramModel(4, alpha=0.0).fit(ids)
    p = bg.log_probs.exp()
    assert math.isclose(float(p[1, 2]), 0.75, rel_tol=1e-5)
    assert math.isclose(float(p[1, 3]), 0.25, rel_tol=1e-5)
    smoothed = BigramModel(4, alpha=0.1).fit(ids).log_probs.exp()
    torch.testing.assert_close(smoothed.sum(1), torch.ones(4))


def test_sequence_logprob_matches_window_nll_for_long_inputs() -> None:
    torch.manual_seed(0)
    ids = torch.randint(0, 5, (200,))
    bg = BigramModel(5, alpha=0.5).fit(ids)
    sc = BigramScorer(bg, max_len=16)
    seq = ids[:50].tolist()
    lp = sequence_logprob(sc, seq[:1], seq[1:])
    direct = float(bg.log_probs[ids[:49], ids[1:50]].sum())
    assert math.isclose(lp, direct, rel_tol=1e-5)
    x, y = eval_windows(ids[:49], 16)
    assert token_nll(sc, x, y).shape == (3,)


def test_greedy_stops_on_stop_token() -> None:
    bg = BigramModel(3, alpha=0.0).fit(torch.tensor([0, 1, 2, 0, 1, 2]))
    out = greedy(BigramScorer(bg, 8), [0], 10, frozenset({2}))
    assert out == [1, 2]


def test_pack_masks_only_answer_targets() -> None:
    tok = CharTokenizer.from_text("abc")
    _, y, m = pack(tok, [reversal_prompt("abc")], [reversal_answer("abc")])
    answer_targets = tok.decode([int(t) for t, keep in zip(y[0], m[0], strict=True) if keep])
    assert answer_targets == "cba\n"


def test_eval_sets_are_reproducible_and_disjoint_from_training_words() -> None:
    s = load_splits()
    sets = build_all(s)
    for name, items in sets.items():
        assert (EVALS / f"{name}.jsonl").read_text(encoding="utf-8") == serialize(items), name
    train_words = set(task_train_words(s))
    for name in ("reversal_in_dist", "reversal_long"):
        assert not {it["word"] for it in sets[name]} & train_words
    for it in sets["minimal_pairs"]:
        assert sorted(it["good"]) == sorted(it["bad"]) and it["good"] != it["bad"]
    for it in sets["word_completion"]:
        assert it["prompt"][-3:] + it["answer"] == it["word"]
        assert "\n" not in it["prompt"]  # the context stays inside one paragraph


def test_typos_keep_length_case_and_always_change_the_letter() -> None:
    text = "Hej Med Dig, sagde Per. " * 50
    noisy = add_typos(text, 1.0, random.Random(0))
    assert len(noisy) == len(text)
    for a, b in zip(text, noisy, strict=True):
        if a.lower() in TYPO_LETTERS:
            assert a != b and a.isupper() == b.isupper()
        else:
            assert a == b
    assert add_typos(text, 0.0, random.Random(0)) == text


def test_mcnemar_exact() -> None:
    before = [False] * 17 + [True] * 4 + [True] * 279
    after = [True] * 17 + [False] * 4 + [True] * 279
    t = mcnemar_exact(before, after)
    assert (t["fixed"], t["broken"], t["n"]) == (17, 4, 300)
    assert math.isclose(t["p_value"], 2 * sum(math.comb(21, i) for i in range(5)) / 2**21)
    assert mcnemar_exact([True, False], [True, False])["p_value"] == 1.0
