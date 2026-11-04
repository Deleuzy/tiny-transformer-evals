import pytest
import torch
import torch.nn.functional as F

from tte.attention import MultiHeadSelfAttention, attention_weights, causal_mask, scaled_dot_product_attention


def test_causal_mask_is_lower_triangular() -> None:
    m = causal_mask(4, 4)
    assert torch.equal(m, torch.tril(torch.ones(4, 4, dtype=torch.bool)))


def test_causal_mask_with_cache_offset() -> None:
    # One new query at position 5 of 6 may see all six keys; two new queries see 5 and 6 keys.
    assert causal_mask(1, 6).all()
    assert causal_mask(2, 6).tolist() == [[True] * 5 + [False], [True] * 6]
    with pytest.raises(ValueError):
        causal_mask(3, 2)


@pytest.mark.parametrize("shape", [(2, 3, 5, 8), (1, 1, 1, 4), (3, 4, 17, 16)])
def test_matches_torch_sdpa_causal(shape: tuple[int, int, int, int]) -> None:
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(*shape, generator=g) for _ in range(3))
    t = shape[2]
    ours = scaled_dot_product_attention(q, k, v, causal_mask(t, t))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert ours.shape == ref.shape == shape
    torch.testing.assert_close(ours, ref, atol=1e-5, rtol=1e-5)


def test_matches_torch_sdpa_arbitrary_mask_and_lengths() -> None:
    g = torch.Generator().manual_seed(1)
    q = torch.randn(2, 3, 4, 8, generator=g)
    k = torch.randn(2, 3, 7, 8, generator=g)
    v = torch.randn(2, 3, 7, 6, generator=g)
    mask = torch.rand(4, 7, generator=g) > 0.3
    mask[:, 0] = True  # every query keeps at least one key
    torch.testing.assert_close(
        scaled_dot_product_attention(q, k, v, mask), F.scaled_dot_product_attention(q, k, v, attn_mask=mask),
        atol=1e-5, rtol=1e-5,
    )


def test_weights_are_distributions_and_zero_on_future() -> None:
    q = torch.randn(1, 2, 6, 4)
    k = torch.randn(1, 2, 6, 4)
    w = attention_weights(q, k, causal_mask(6, 6))
    torch.testing.assert_close(w.sum(-1), torch.ones(1, 2, 6))
    assert (w.triu(diagonal=1) == 0).all()


def test_future_tokens_cannot_change_the_past() -> None:
    torch.manual_seed(0)
    attn = MultiHeadSelfAttention(16, 4).eval()
    x = torch.randn(1, 10, 16)
    y1, _ = attn(x)
    x2 = x.clone()
    x2[:, 6:] = torch.randn(1, 4, 16) * 10  # change only positions 6..9
    y2, _ = attn(x2)
    torch.testing.assert_close(y1[:, :6], y2[:, :6])
    assert not torch.allclose(y1[:, 6:], y2[:, 6:])


def test_kv_cache_matches_full_pass() -> None:
    torch.manual_seed(0)
    attn = MultiHeadSelfAttention(16, 2).eval()
    x = torch.randn(2, 7, 16)
    full, _ = attn(x)
    out, past = attn(x[:, :4])
    steps = [out]
    for t in range(4, 7):
        o, past = attn(x[:, t : t + 1], past)
        steps.append(o)
    torch.testing.assert_close(torch.cat(steps, dim=1), full, atol=1e-6, rtol=1e-5)


def test_heads_must_divide_width() -> None:
    with pytest.raises(ValueError):
        MultiHeadSelfAttention(10, 3)
