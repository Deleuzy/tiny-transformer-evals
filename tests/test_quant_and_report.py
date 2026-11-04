import torch

from tte.config import load_config
from tte.data import ROOT
from tte.model import ModelConfig, TinyTransformer
from tte.quant import latency, percentiles, quantize_int8, state_bytes
from tte.report import update_readme


def test_int8_model_is_smaller_and_close() -> None:
    torch.manual_seed(0)
    m = TinyTransformer(ModelConfig(20, d_model=64, n_heads=4, n_layers=2, d_ff=128, max_len=32, dropout=0.0)).eval()
    x = torch.randint(0, 20, (2, 32))
    ref = m(x)[0]
    size = state_bytes(m)
    q = quantize_int8(m)
    out = q(x)[0]
    assert state_bytes(q) < 0.6 * size
    assert torch.allclose(torch.softmax(out, -1), torch.softmax(ref, -1), atol=0.02)


def test_latency_reports_every_model_and_measurement() -> None:
    torch.manual_seed(0)
    m = TinyTransformer(ModelConfig(20, d_model=16, n_heads=2, n_layers=1, d_ff=32, max_len=16, dropout=0.0)).eval()
    res = latency({"a": m, "b": quantize_int8(m)}, [1, 2, 3], 4, runs=3, warmup=1)
    assert set(res) == {"a", "b"}
    for r in res.values():
        assert set(r) == {"prefill", "generate_kv_cache", "generate_no_cache"}
        assert all(v["p50_ms"] > 0 for v in r.values())


def test_percentiles() -> None:
    p = percentiles([float(i) for i in range(1, 101)])
    assert p["p50_ms"] == 50.5 and p["p99_ms"] > p["p95_ms"] > p["p50_ms"]


def test_readme_tables_match_saved_results() -> None:
    """Every number in the README results tables is rendered from results/full/*.json."""
    assert update_readme(load_config(ROOT / "configs" / "full.json"), check=True) == 0
