"""Inverse C2C + normalization correctness on device (issue #6 PR-B).

Hardware-marked (see conftest.py): skipped cleanly without the vendor
xpytorch stack or a reachable XPU. Reference is numpy complex128
(the vendor torch.fft caps at N <= 2048 and is not the contract under
test — torch.fft norm semantics are reproduced exactly by
:func:`src.xpu.entry.norm_scale`).
"""

import math
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

try:
    import torch
    import torch_xmlir  # noqa: F401
except Exception as e:  # pragma: no cover - absent in ordinary CI
    pytest.skip(f"vendor xpytorch stack unavailable: {e}", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.hardware

from src.xpu import DeviceProfile  # noqa: E402
from src.xpu import load_table  # noqa: E402
from src.xpu.entry import (  # noqa: E402
    SUPPORTED_NORMS,
    execute_fft_plan,
    plan_fft,
    run_fft,
)
from src.xpu.error_metric import max_rel_error  # noqa: E402

TABLE_PATH = Path(__file__).resolve().parents[1] / "config" / "selector_table.json"


def _table():
    if not TABLE_PATH.exists():
        pytest.skip("selector table not built")
    return load_table(str(TABLE_PATH))


def _profile_hash():
    return DeviceProfile.load().hash()


def _to_np(x):
    return x.detach().cpu().numpy().astype(np.complex128)


def _np_ref(x_np, transform_direction, norm, decomposition):
    # numpy has no norm="none": map our unscaled variants explicitly.
    n = x_np.shape[-1]
    if transform_direction == "forward":
        if norm == "none":
            norm = "backward"  # numpy backward == unscaled forward
        return np.fft.fft(x_np, axis=-1, norm=norm)
    if norm in ("none", "forward"):
        return np.fft.ifft(x_np, axis=-1, norm="backward") * n
    return np.fft.ifft(x_np, axis=-1, norm=norm)


def _make_input(kind, n, b, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "impulse":
        a = np.zeros((b, n), dtype=np.complex128)
        a[:, 0] = 1.0
    elif kind == "constant":
        a = np.full((b, n), 3.0 + 4.0j, dtype=np.complex128)
    elif kind == "hdr":
        mag = np.exp(rng.uniform(-3, 3, size=(b, n)))
        phase = rng.uniform(0, 2 * np.pi, size=(b, n))
        a = mag * np.exp(1j * phase)
    else:  # random
        a = (rng.standard_normal((b, n))
             + 1j * rng.standard_normal((b, n)))
    a = a.astype(np.complex64)
    if b == 1:
        a = a[0]
    return torch.from_numpy(a).to("cuda")


@pytest.mark.parametrize("decomposition", ["dit", "dif"])
@pytest.mark.parametrize("n,b", [(8, 1), (64, 4), (256, 1), (1024, 16)])
@pytest.mark.parametrize("transform_direction", ["forward", "inverse"])
@pytest.mark.parametrize("norm", SUPPORTED_NORMS)
def test_c2c_matches_numpy_all_norms(decomposition, n, b,
                                     transform_direction, norm):
    table = _table()
    prof = _profile_hash()
    x = _make_input("random", n, b, seed=n + b)
    plan = plan_fft(decomposition, n, batch=b, profile_hash=prof,
                    table=table, transform_direction=transform_direction,
                    norm=norm)
    y = execute_fft_plan(plan, x)
    ref = _np_ref(_to_np(x), transform_direction, norm, decomposition)
    rel = max_rel_error(_to_np(y), ref)
    assert rel < 1e-4, (
        f"{decomposition} {transform_direction} norm={norm} "
        f"N={n} B={b}: rel={rel:.3e} (fallback={plan.fallback})")


@pytest.mark.parametrize("kind", ["impulse", "constant", "hdr", "random"])
def test_inverse_special_inputs(kind):
    table = _table()
    prof = _profile_hash()
    n, b = 512, 2
    x = _make_input(kind, n, b, seed=7)
    plan = plan_fft("dit", n, batch=b, profile_hash=prof, table=table,
                    transform_direction="inverse", norm="backward")
    y = execute_fft_plan(plan, x)
    ref = np.fft.ifft(_to_np(x), axis=-1, norm="backward")
    rel = max_rel_error(_to_np(y), ref)
    assert rel < 1e-4, f"{kind} inverse: rel={rel:.3e}"


@pytest.mark.parametrize("decomposition", ["dit", "dif"])
@pytest.mark.parametrize("n,b", [(256, 1), (1024, 16)])
def test_round_trip_identity(decomposition, n, b):
    table = _table()
    prof = _profile_hash()
    x = _make_input("random", n, b, seed=11)
    fwd = plan_fft(decomposition, n, batch=b, profile_hash=prof,
                   table=table, transform_direction="forward",
                   norm="backward")
    inv = plan_fft(decomposition, n, batch=b, profile_hash=prof,
                   table=table, transform_direction="inverse",
                   norm="backward")
    y = execute_fft_plan(inv, execute_fft_plan(fwd, x))
    rel = max_rel_error(_to_np(y), _to_np(x))
    assert rel < 1e-4, (
        f"round trip {decomposition} N={n} B={b}: rel={rel:.3e}")


@pytest.mark.parametrize("n,b", [(128, 1), (1024, 16)])
def test_one_shot_matches_plan_reuse(n, b):
    table = _table()
    prof = _profile_hash()
    x = _make_input("random", n, b, seed=13)
    plan = plan_fft("dit", n, batch=b, profile_hash=prof, table=table,
                    transform_direction="inverse", norm="ortho")
    reused = execute_fft_plan(plan, x)
    once, report = run_fft(x, "dit", profile_hash=prof, table=table,
                           transform_direction="inverse", norm="ortho")
    assert report.executed == plan.executed
    assert report.transform_direction == "inverse"
    assert max_rel_error(_to_np(once), _to_np(reused)) == 0.0


def test_repeat_execution_bitwise_stable():
    table = _table()
    prof = _profile_hash()
    n, b = 1024, 16
    x = _make_input("random", n, b, seed=17)
    plan = plan_fft("dif", n, batch=b, profile_hash=prof, table=table,
                    transform_direction="inverse", norm="backward")
    y1 = execute_fft_plan(plan, x)
    y2 = execute_fft_plan(plan, x)
    # torch.equal is unimplemented on this vendor stack; compare the
    # split components bitwise on the host instead.
    a1, a2 = _to_np(y1), _to_np(y2)
    assert np.array_equal(a1, a2), "inverse plan re-execution is not stable"


def test_forward_backward_norm_is_torch_fft_backward():
    """Default inverse (norm='backward') equals torch.fft.ifft default."""
    table = _table()
    prof = _profile_hash()
    n, b = 256, 2
    x = _make_input("random", n, b, seed=19)
    plan = plan_fft("dit", n, batch=b, profile_hash=prof, table=table,
                    transform_direction="inverse", norm="backward")
    y = execute_fft_plan(plan, x)
    ref = torch.fft.ifft(x)
    rel = max_rel_error(_to_np(y), _to_np(ref))
    assert rel < 1e-4, f"default inverse vs torch.fft.ifft: rel={rel:.3e}"
