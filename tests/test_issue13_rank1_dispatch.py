"""issue #13 WP0: real rank-1 plan_fft -> execute_fft_plan coverage.

The review required production-contract tests on the REAL runners —
not the Fake1D stub used by the holdout meta unit tests. This file
exercises every dispatch branch plan_fft can resolve with a true
rank-1 ``(N,)`` complex64 tensor:

* M0-fallback branch (``profile_hash=None`` abstain -> ``fft_dit``);
* butterfly-table branch (semantic entry -> recorded ``executed``
  runner, here ``fft_dit_matrix``);
* one-shot ``run_fft`` rank-1 entry;
* the inverse contract: a batch=1 plan REQUIRES rank-1 input and a
  batch>1 plan REQUIRES rank-2 ``(B, N)`` (no silent squeeze at the
  plan-reuse boundary; ``run_fft`` keeps its documented ``(1, N)``
  convenience squeeze).

Hardware-marked: needs the vendor xpytorch stack + reachable P800.
"""

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

try:
    import torch
    import torch_xmlir  # noqa: F401
except Exception as e:  # pragma: no cover - absent in ordinary CI
    pytest.skip(f"vendor xpytorch stack unavailable: {e}",
                allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.hardware

from src.xpu.entry import execute_fft_plan, plan_fft, run_fft  # noqa: E402
from src.xpu.selector import load_table  # noqa: E402

TABLE = load_table()


def _x(n: int, seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    re = torch.randn(n, generator=g)
    im = torch.randn(n, generator=g)
    return torch.complex(re, im).to("cuda")


def _err(y, ref):
    a = y.detach().cpu().numpy()
    return float(np.abs(a - ref).max() / max(np.abs(ref).max(), 1e-30))


def test_rank1_m0_fallback_plan_execute():
    plan = plan_fft("dit", 64, batch=1, profile_hash=None, table=TABLE)
    assert plan.fallback is True and plan.selection_source == "abstain"
    assert plan.executed == "fft_dit"
    x = _x(64)
    assert x.dim() == 1
    y = execute_fft_plan(plan, x)
    assert y.dim() == 1 and y.shape == (64,)
    ref = np.fft.fft(x.cpu().numpy().astype(np.complex128))
    assert _err(y, ref) <= 1e-4
    # strict contract: a batch=1 plan never accepts (1, N) silently
    with pytest.raises(ValueError, match="rank-1"):
        execute_fft_plan(plan, x.unsqueeze(0))


def test_rank1_table_plan_execute_real_runner():
    plan = plan_fft("dit", 64, batch=1,
                    profile_hash=TABLE["profile_hash"], table=TABLE,
                    env_hash="auto")
    assert plan.selection_source == "table" and plan.fallback is False
    assert plan.executed == "fft_dit_matrix"  # semantic entry's runner
    x = _x(64, seed=8)
    assert x.dim() == 1
    y = execute_fft_plan(plan, x)
    assert y.dim() == 1 and y.shape == (64,)
    ref = np.fft.fft(x.cpu().numpy().astype(np.complex128))
    assert _err(y, ref) <= 1e-4


def test_rank1_run_fft_one_shot():
    x = _x(64, seed=9)
    y, report = run_fft(x, "dit")
    assert y.dim() == 1 and y.shape == (64,)
    ref = np.fft.fft(x.cpu().numpy().astype(np.complex128))
    assert _err(y, ref) <= 1e-4
    assert report.n == 64 and report.batch == 1


def test_batch_plan_execute_requires_rank2():
    plan = plan_fft("dit", 1024, batch=16,
                    profile_hash=TABLE["profile_hash"], table=TABLE,
                    env_hash="auto")
    assert plan.batch == 16
    g = torch.Generator().manual_seed(11)
    re = torch.randn(16, 1024, generator=g)
    im = torch.randn(16, 1024, generator=g)
    x = torch.complex(re, im).to("cuda")
    y = execute_fft_plan(plan, x)
    assert y.shape == (16, 1024)
    ref = np.fft.fft(x.cpu().numpy().astype(np.complex128), axis=-1)
    assert _err(y, ref) <= 1e-4
    # B>1 plan: rank-1 input must be rejected, never silently promoted
    with pytest.raises(ValueError, match=r"\(16, 1024\)"):
        execute_fft_plan(plan, x[0])


def test_run_fft_one_squeeze_restores_layout():
    # documented convenience: (1, N) runs the B=1 winner, returns (1, N)
    g = torch.Generator().manual_seed(13)
    re = torch.randn(64, generator=g)
    im = torch.randn(64, generator=g)
    x2 = torch.complex(re, im).unsqueeze(0).to("cuda")
    y, report = run_fft(x2, "dit")
    assert y.shape == (1, 64) and report.batch == 1
