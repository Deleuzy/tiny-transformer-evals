import math

import pytest
import torch
import torch.nn.functional as F

from tte.model import ModelConfig, TinyTransformer, generate_greedy, sinusoidal_positions


def small(vocab: int = 11, max_len: int = 16) -> TinyTransformer:
    torch.manual_seed(0)
    return TinyTransformer(ModelConfig(vocab, d_model=16, n_heads=2, n_layers=2, d_ff=32, max_len=max_len, dropout=0.0)).eval()


def test_logit_shapes_and_cache_shapes() -> None:
    m = small()
    logits, past = m(torch.randint(0, 11, (3, 9)))
    assert logits.shape == (3, 9, 11)
    assert len(past) == 2 and past[0][0].shape == (3, 2, 9, 8)


def test_initial_loss_is_near_uniform() -> None:
    m = small()
    x = torch.randint(0, 11, (8, 16))
    with torch.no_grad():
        loss = F.cross_entropy(m(x)[0].flatten(0, 1), x.flatten())
    assert abs(loss.item() - math.log(11)) < 0.3


def test_model_is_causal() -> None:
    m = small()
    x = torch.randint(0, 11, (1, 12))
    x2 = x.clone()
    x2[0, 8:] = (x2[0, 8:] + 1) % 11
    torch.testing.assert_close(m(x)[0][:, :8], m(x2)[0][:, :8])


def test_too_long_input_is_refused() -> None:
    with pytest.raises(ValueError):
        small(max_len=8)(torch.zeros(1, 9, dtype=torch.long))


def test_sinusoidal_positions() -> None:
    pe = sinusoidal_positions(10, 8)
    assert pe.shape == (10, 8)
    torch.testing.assert_close(pe[0, 0::2], torch.zeros(4))
    torch.testing.assert_close(pe[0, 1::2], torch.ones(4))
    assert float(pe[3, 0]) == pytest.approx(math.sin(3.0), abs=1e-6)


def test_generation_with_and_without_cache_agree_past_max_len() -> None:
    m = small(max_len=10)
    prompt = [1, 2, 3, 4, 5]
    a = generate_greedy(m, prompt, 12, use_cache=True)
    b = generate_greedy(m, prompt, 12, use_cache=False)
    assert a == b and len(a) == 12


def test_overfits_a_tiny_sequence() -> None:
    """Sanity check that the whole stack can learn: memorise one 12-token sequence."""
    m = small()
    m.train()
    x = torch.tensor([[1, 3, 5, 7, 9, 2, 4, 6, 8, 10, 1, 3, 5]])
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(150):
        loss = F.cross_entropy(m(x[:, :-1])[0].flatten(0, 1), x[:, 1:].flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
