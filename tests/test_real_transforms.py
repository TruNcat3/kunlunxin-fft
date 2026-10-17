"""R2C/C2R compatibility-path correctness (issue #6 PR-C).

Hardware-marked: skipped without the vendor xpytorch stack or a
reachable XPU. The compatibility path packs real input into a complex
buffer, runs the selected C2C core, and crops/reconstructs the half
spectrum inside the timed execution — verified here against numpy
complex128 ``rfft``/``irfft`` with the same norm.
"""

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
    REAL_COMPAT_MARK,
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
    a = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    return a.astype(np.complex128) if np.iscomplexobj(a) else a.astype(np.float64)


def _make_real(kind, n, b, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "impulse":
        a = np.zeros((b, n), dtype=np.float64)
        a[:, 0] = 1.0
    elif kind == "constant":
        a = np.full((b, n), 3.0, dtype=np.float64)
    elif kind == "hdr":
        a = rng.uniform(-1e3, 1e3, size=(b, n))
    else:
        a = rng.standard_normal((b, n))
    a = a.astype(np.float32)
    if b == 1:
        a = a[0]
    return torch.from_numpy(a).to("cuda")


def _plan(tt, decomposition, n, b, norm):
    return plan_fft(decomposition, n, batch=b,
                    profile_hash=_profile_hash(), table=_table(),
                    transform_type=tt, norm=norm)


@pytest.mark.parametrize("decomposition", ["dit", "dif"])
@pytest.mark.parametrize("n,b", [(64, 1), (256, 16), (1024, 1), (2048, 64)])
@pytest.mark.parametrize("norm", SUPPORTED_NORMS)
def test_r2c_matches_numpy_rfft(decomposition, n, b, norm):
    x = _make_real("random", n, b, seed=n + b)
    plan = _plan("r2c", decomposition, n, b, norm)
    assert plan.transform_type == "r2c"
    assert plan.transform_direction == "forward"
    assert plan.params.get("real_compat") == REAL_COMPAT_MARK
    y = execute_fft_plan(plan, x)
    assert y.dtype == torch.complex64
    assert tuple(y.shape) == ((b, n // 2 + 1) if b > 1
                              else (n // 2 + 1,))
    ref = np.fft.rfft(_to_np(x), axis=-1,
                      norm=("backward" if norm == "none" else norm))
    rel = max_rel_error(_to_np(y), ref)
    assert rel < 1e-4, (
        f"r2c {decomposition} norm={norm} N={n} B={b}: rel={rel:.3e}")


@pytest.mark.parametrize("decomposition", ["dit", "dif"])
@pytest.mark.parametrize("n,b", [(64, 1), (256, 16), (1024, 1), (2048, 64)])
@pytest.mark.parametrize("norm", SUPPORTED_NORMS)
def test_c2r_matches_numpy_irfft(decomposition, n, b, norm):
    x = _make_real("random", n, b, seed=n * 3 + b)
    half = np.fft.rfft(_to_np(x), axis=-1).astype(np.complex64)
    if b == 1:
        half_t = torch.from_numpy(half).to("cuda")
    else:
        half_t = torch.from_numpy(half).to("cuda")
    plan = _plan("c2r", decomposition, n, b, norm)
    assert plan.transform_type == "c2r"
    assert plan.transform_direction == "inverse"
    assert plan.params.get("real_compat") == REAL_COMPAT_MARK
    y = execute_fft_plan(plan, half_t)
    assert y.dtype == torch.float32
    assert tuple(y.shape) == ((b, n) if b > 1 else (n,))
    # numpy inverse norms: backward=1/N, ortho=1/sqrt(N), forward=1;
    # ours: none/forward unscaled, backward 1/N, ortho 1/sqrt(N).
    if norm == "ortho":
        ref = np.fft.irfft(_to_np(half), n=n, axis=-1, norm="ortho")
    else:
        ref = np.fft.irfft(_to_np(half), n=n, axis=-1, norm="backward")
        if norm in ("none", "forward"):
            ref = ref * n

    rel = max_rel_error(_to_np(y), ref)
    assert rel < 1e-4, (
        f"c2r {decomposition} norm={norm} N={n} B={b}: rel={rel:.3e}")


@pytest.mark.parametrize("decomposition", ["dit", "dif"])
@pytest.mark.parametrize("n,b", [(128, 1), (1024, 16)])
def test_rfft_irfft_round_trip(decomposition, n, b):
    x = _make_real("random", n, b, seed=23)
    r2c = _plan("r2c", decomposition, n, b, "backward")
    c2r = _plan("c2r", decomposition, n, b, "backward")
    y = execute_fft_plan(c2r, execute_fft_plan(r2c, x))
    rel = max_rel_error(_to_np(y), _to_np(x))
    assert rel < 1e-4, f"round trip {decomposition} N={n} B={b}: rel={rel:.3e}"


@pytest.mark.parametrize("kind", ["impulse", "constant", "hdr", "random"])
def test_r2c_special_inputs(kind):
    n, b = 512, 2
    x = _make_real(kind, n, b, seed=29)
    plan = _plan("r2c", "dit", n, b, "backward")
    y = execute_fft_plan(plan, x)
    ref = np.fft.rfft(_to_np(x), axis=-1, norm="backward")
    rel = max_rel_error(_to_np(y), ref)
    assert rel < 1e-4, f"{kind} r2c: rel={rel:.3e}"


def test_c2r_dc_nyquist_imaginary_is_ignored():
    """torch.fft contract: imag at DC/Nyquist is dropped, not an error."""
    n = 256
    m = n // 2 + 1
    rng = np.random.default_rng(31)
    half = (rng.standard_normal(m) + 1j * rng.standard_normal(m)
            ).astype(np.complex64)
    x = torch.from_numpy(half).to("cuda")
    plan = _plan("c2r", "dit", n, 1, "backward")
    y = execute_fft_plan(plan, x)
    ref = np.fft.irfft(half, n=n, norm="backward")
    rel = max_rel_error(_to_np(y), ref)
    assert rel < 1e-4, f"DC/Nyquist ownership: rel={rel:.3e}"


def test_r2c_rejects_complex_input_before_launch():
    n = 64
    x = torch.zeros(n, dtype=torch.complex64, device="cuda")
    with pytest.raises(TypeError, match="float32"):
        run_fft(x, "dit", transform_type="r2c",
                profile_hash=_profile_hash(), table=_table())


def test_c2r_one_shot_matches_plan_reuse():
    n, b = 512, 2
    x = _make_real("random", n, b, seed=37)
    half = np.fft.rfft(_to_np(x), axis=-1).astype(np.complex64)
    half_t = torch.from_numpy(half).to("cuda")
    plan = _plan("c2r", "dit", n, b, "ortho")
    reused = execute_fft_plan(plan, half_t)
    once, report = run_fft(half_t, "dit", transform_type="c2r",
                           profile_hash=_profile_hash(), table=_table(),
                           norm="ortho")
    assert report.executed == plan.executed
    assert report.transform_type == "c2r"
    assert max_rel_error(_to_np(once), _to_np(reused)) == 0.0


def test_r2c_repeat_execution_bitwise_stable():
    n, b = 1024, 16
    x = _make_real("random", n, b, seed=41)
    plan = _plan("r2c", "dif", n, b, "backward")
    y1 = execute_fft_plan(plan, x)
    y2 = execute_fft_plan(plan, x)
    assert np.array_equal(_to_np(y1), _to_np(y2))
