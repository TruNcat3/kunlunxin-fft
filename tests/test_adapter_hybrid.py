"""Processing-unit adapter (issue #3 §4-§5 four-cell ablation support)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.xpu.adapter_hybrid import (  # noqa: E402
    HybridPlan,
    _bitrev_index,
    plan_hybrid,
    run_hybrid,
)


def test_plan_validation():
    p = plan_hybrid(1024, 16, tile=64, core="vendor")
    assert p.stage2_len == 16 and p.direction == "dit"
    with pytest.raises(ValueError, match="must divide"):
        plan_hybrid(1024, 16, tile=48)
    with pytest.raises(ValueError, match="powers of two"):
        plan_hybrid(960, 16, tile=64)  # divides, but N is not pow2
    with pytest.raises(ValueError, match="DIT only"):
        HybridPlan(n=1024, batch=16, tile=64, direction="dif")
    with pytest.raises(ValueError, match="unknown core"):
        plan_hybrid(1024, 16, tile=64, core="external")


def test_bitrev_index_matches_numpy():

    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("needs importable vendor torch (host-only CI lacks it)")
    torch = pytest.importorskip("torch")  # host CI has no device stack
    import numpy as np

    for n in (8, 16, 64, 256):
        got = _bitrev_index(n, "cpu").tolist()
        bits = n.bit_length() - 1
        want = [int(f"{i:0{bits}b}"[::-1], 2) for i in range(n)]
        assert got == want, n
        assert torch.equal(_bitrev_index(n, "cpu"),
                           torch.tensor(want, dtype=torch.int32))


pytestmark_hardware = pytest.mark.hardware


@pytest.mark.hardware
def test_small_triton_stage_matches_vendor():
    import torch
    from src.xpu.adapter_hybrid import _row_fft_triton_small
    from src.xpu.error_metric import max_rel_error

    for length in (8, 16, 32):
        re = torch.randn(64, length)
        im = torch.randn(64, length)
        x = torch.complex(re, im).cuda()
        y = _row_fft_triton_small(x)
        ref = torch.fft.fft(x, dim=-1)
        assert max_rel_error(y.detach().cpu().numpy(),
                             ref.detach().cpu().numpy()) < 1e-5


@pytest.mark.hardware
def test_m2_local_stage_matches_vendor():
    import torch
    from src.xpu.adapter_hybrid import _local_stage_triton_1d
    from src.xpu.error_metric import max_rel_error

    plan = plan_hybrid(1024, 16, tile=64, core="triton")
    x = torch.complex(torch.randn(256, 64), torch.randn(256, 64)).cuda()
    y = _local_stage_triton_1d(x, 64, plan)
    ref = torch.fft.fft(x, dim=-1)
    assert max_rel_error(y.detach().cpu().numpy(),
                         ref.detach().cpu().numpy()) < 1e-5


@pytest.mark.hardware
@pytest.mark.parametrize("tile", (32, 64, 128))
@pytest.mark.parametrize("core", ("vendor", "triton"))
def test_hybrid_end_to_end_correctness(tile, core):
    import numpy as np
    import torch
    from src.xpu.error_metric import max_rel_error

    n, batch = 1024, 4
    rng = np.random.default_rng(7)
    re = rng.standard_normal((batch, n)).astype(np.float32)
    im = rng.standard_normal((batch, n)).astype(np.float32)
    x = torch.complex(torch.from_numpy(re), torch.from_numpy(im)).cuda()
    ref = np.fft.fft(re + 1j * im, axis=-1)
    y = run_hybrid(x, plan_hybrid(n, batch, tile=tile, core=core))
    assert max_rel_error(y.detach().cpu().numpy(), ref) < 1e-4


@pytest.mark.hardware
def test_a0p1_abstain_is_structural():
    # A0P1 is a documented abstention, not a runtime path: there is no
    # function to call, by construction (fused core boundary).
    from src.xpu.adapter_hybrid import HybridPlan
    p = HybridPlan(n=1024, batch=16, tile=64, core="vendor")
    assert p.core == "vendor"  # plan carries the core; the fused flow
    # cannot: runner APIs take no external-core parameter at all.
    import inspect
    from src.xpu import runner
    sig = inspect.signature(runner.fft_dit_matrix_batch)
    assert "core" not in sig.parameters


def test_summarize_samples_stats():
    from src.xpu.adapter_hybrid import summarize_samples
    # Even length: p50 = mean of the two middle values.
    s = summarize_samples([1.0, 2.0, 3.0, 4.0])
    assert s["n"] == 4
    assert abs(s["p50"] - 2.5) < 1e-12
    assert s["max"] == 4.0
    # MAD: deviations from 2.5 are 1.5, 0.5, 0.5, 1.5 -> sorted
    # 0.5, 0.5, 1.5, 1.5 -> median = (0.5+1.5)/2 = 1.0
    assert abs(s["mad"] - 1.0) < 1e-12
    # No outliers: threshold = 2.5 + 3*1.0 = 5.5
    assert s["outlier_count"] == 0
    # Outlier disclosure: a huge sample is flagged and listed.
    s2 = summarize_samples([1.0] * 9 + [100.0])
    assert s2["outlier_count"] == 1
    assert s2["outlier_values"] == [100.0]
    assert s2["max"] == 100.0
    # MAD == 0 fallback: constant samples, threshold = 1.5 * p50.
    s3 = summarize_samples([2.0] * 8 + [4.0])
    assert s3["mad"] == 0.0
    assert s3["outlier_count"] == 1
    assert s3["outlier_values"] == [4.0]
    # Empty input.
    assert summarize_samples([]) == {"n": 0}


@pytest.mark.hardware
def test_profile_hybrid_matches_run_hybrid():
    import numpy as np
    import torch
    from src.xpu.adapter_hybrid import profile_hybrid, run_hybrid
    from src.xpu.error_metric import max_rel_error
    rng = np.random.default_rng(7)
    n, batch = 1024, 16
    re = rng.standard_normal((batch, n)).astype(np.float32)
    im = rng.standard_normal((batch, n)).astype(np.float32)
    x = torch.complex(torch.from_numpy(re), torch.from_numpy(im)).cuda()
    ref = np.fft.fft(re + 1j * im, axis=-1)
    plan = plan_hybrid(n, batch, tile=64, core="vendor")
    y1 = run_hybrid(x, plan)
    y2, phases = profile_hybrid(x, plan)
    assert max_rel_error(y1.detach().cpu().numpy(), ref) < 1e-4
    assert max_rel_error(y2.detach().cpu().numpy(), ref) < 1e-4
    # Phase accounting: every phase recorded, stage2 input contiguous
    # (grid reshape view, no hidden copy).
    for key in ("gather_ms", "stage1_core_ms", "twiddle_transpose_ms",
                "stage2_core_ms", "scatter_ms"):
        assert phases[key] >= 0.0
    assert phases["stage2_in_contiguous"] is True
    assert phases["hidden_copies"] == []


@pytest.mark.hardware
def test_run_hybrid_no_flat2_buffer():
    # The fused pipeline must not allocate a flat2 staging buffer:
    # stage2 reads a reshape view of the grid buffer directly.
    import numpy as np
    import torch
    from src.xpu.adapter_hybrid import run_hybrid
    rng = np.random.default_rng(11)
    n, batch = 1024, 16
    re = rng.standard_normal((batch, n)).astype(np.float32)
    im = rng.standard_normal((batch, n)).astype(np.float32)
    x = torch.complex(torch.from_numpy(re), torch.from_numpy(im)).cuda()
    plan = plan_hybrid(n, batch, tile=64, core="vendor")
    run_hybrid(x, plan)
    names = {key[0] for key in plan._buffers}
    assert "flat2" not in names
    assert "blocks" in names and "grid" in names
