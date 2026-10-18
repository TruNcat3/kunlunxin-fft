"""Dedicated real-transform flow parity on the XPU (issue #8 PR-H).

Hardware-marked: skipped without the vendor xpytorch stack or a
reachable XPU. Executes ``plan.params["real_flow"]`` plans (half-length
core + pack/stitch/pre/unpack epilogues) against numpy complex128
``rfft``/``irfft`` with the same norm (rel < 1e-4, the shared gate),
and the launched epilogue kernels against their torch references.
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

from src.xpu.entry import (  # noqa: E402
    REAL_COMPAT_MARK,
    REAL_FLOW_MARK,
    SUPPORTED_NORMS,
    ExecutionPlan,
    execute_fft_plan,
)
from src.xpu.error_metric import max_rel_error  # noqa: E402
from src.xpu.runner import (  # noqa: E402
    c2r_pre,
    c2r_pre_ref,
    c2r_unpack,
    c2r_unpack_ref,
    r2c_stitch,
    r2c_stitch_ref,
)

_R2C_FLOW = "dedicated-r2c-pack-v1"
_C2R_FLOW = "dedicated-c2r-hermitian-v1"


def _to_np(x):
    a = x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
    return (a.astype(np.complex128) if np.iscomplexobj(a)
            else a.astype(np.float64))


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


def _dedicated_plan(tt, n, b, norm, executed=None):
    """Directly constructed dedicated-flow plan (no table yet)."""
    flow = _R2C_FLOW if tt == "r2c" else _C2R_FLOW
    if executed is None:
        executed = "fft_dif_matrix" if b == 1 else "fft_dif_matrix_batch"
    params = {"tile": 64, "rows": 8, "real_flow": flow,
              "real_compat": REAL_FLOW_MARK}
    if b > 1:
        params["variant"] = "spread"
    return ExecutionPlan(
        n=n, batch=b, direction="dif", precision="fp32",
        selection_source="oracle-candidate", reason="pr-h device parity",
        fallback=False, executed=executed, block=1024,
        flow_id=flow, flow_realization="matrix-resident",
        processing_unit="dense-dot-fft64", params=params,
        transform_type=tt,
        transform_direction=("forward" if tt == "r2c" else "inverse"),
        norm=norm)


def _compat_plan(plan):
    """Same plan on the full-length compat pipeline (PR-H A/B reference)."""
    params = {k: v for k, v in plan.params.items()
              if k not in ("real_flow", "real_compat")}
    params["real_compat"] = REAL_COMPAT_MARK
    return ExecutionPlan(**{**plan.__dict__, "params": params,
                            "flow_id": "local-fused-tile"})


@pytest.mark.parametrize("n,b", [(64, 1), (256, 16), (1024, 16)])
@pytest.mark.parametrize("norm", SUPPORTED_NORMS)
def test_dedicated_r2c_matches_numpy(n, b, norm):
    x = _make_real("random", n, b, seed=n + b + 7)
    plan = _dedicated_plan("r2c", n, b, norm)
    y = execute_fft_plan(plan, x)
    assert y.dtype == torch.complex64
    assert tuple(y.shape) == ((b, n // 2 + 1) if b > 1
                              else (n // 2 + 1,))
    ref = np.fft.rfft(_to_np(x), axis=-1,
                      norm=("backward" if norm == "none" else norm))
    rel = max_rel_error(_to_np(y), ref)
    assert rel < 1e-4, (
        f"dedicated r2c norm={norm} N={n} B={b}: rel={rel:.3e}")


@pytest.mark.parametrize("n,b", [(64, 1), (256, 16), (1024, 16)])
@pytest.mark.parametrize("norm", SUPPORTED_NORMS)
def test_dedicated_c2r_matches_numpy(n, b, norm):
    x = _make_real("random", n, b, seed=n * 3 + b)
    half = np.fft.rfft(_to_np(x), axis=-1).astype(np.complex64)
    half_t = torch.from_numpy(half).to("cuda")
    plan = _dedicated_plan("c2r", n, b, norm)
    y = execute_fft_plan(plan, half_t)
    assert y.dtype == torch.float32
    assert tuple(y.shape) == ((b, n) if b > 1 else (n,))
    if norm == "ortho":
        ref = np.fft.irfft(_to_np(half), n=n, axis=-1, norm="ortho")
    else:
        ref = np.fft.irfft(_to_np(half), n=n, axis=-1, norm="backward")
        if norm in ("none", "forward"):
            ref = ref * n
    rel = max_rel_error(_to_np(y), ref)
    assert rel < 1e-4, (
        f"dedicated c2r norm={norm} N={n} B={b}: rel={rel:.3e}")


@pytest.mark.parametrize("kind", ["impulse", "constant", "hdr", "random"])
def test_dedicated_r2c_special_inputs(kind):
    n, b = 256, 16
    x = _make_real(kind, n, b, seed=91)
    plan = _dedicated_plan("r2c", n, b, "backward")
    y = execute_fft_plan(plan, x)
    ref = np.fft.rfft(_to_np(x), axis=-1)
    rel = max_rel_error(_to_np(y), ref)
    assert rel < 1e-4, f"r2c {kind}: rel={rel:.3e}"


def test_dedicated_c2r_dc_nyquist_imaginary_is_ignored():
    n, b = 1024, 16
    x = _make_real("random", n, b, seed=7)
    half = np.fft.rfft(_to_np(x), axis=-1).astype(np.complex64)
    # corrupt ONLY the imaginary parts torch.fft ignores
    half[..., 0] = half[..., 0] + 0.125j
    half[..., -1] = half[..., -1] + 0.75j
    half_t = torch.from_numpy(half).to("cuda")
    plan = _dedicated_plan("c2r", n, b, "backward")
    y = execute_fft_plan(plan, half_t)
    clean = np.fft.rfft(_to_np(x), axis=-1)
    ref = np.fft.irfft(clean, n=n, axis=-1, norm="backward")
    rel = max_rel_error(_to_np(y), ref)
    assert rel < 1e-4, f"c2r garbage-imag: rel={rel:.3e}"


@pytest.mark.parametrize("n,b", [(64, 1), (256, 16)])
def test_dedicated_epilogues_match_refs_on_device(n, b):
    a = np.random.default_rng(n + b).standard_normal((b, n)).astype(np.float32)
    x = torch.from_numpy(a).to("cuda")
    z = torch.fft.fft(torch.complex(x[..., 0::2], x[..., 1::2]))
    got = r2c_stitch(z, n, 0.5, block=256)
    ref = r2c_stitch_ref(z, n, 0.5)
    assert got.dtype == torch.complex64
    assert tuple(got.shape) == tuple(ref.shape)
    rel = max_rel_error(_to_np(got), _to_np(ref))
    assert rel < 1e-6, f"r2c_stitch kernel vs ref: rel={rel:.3e}"

    X = torch.fft.rfft(x).contiguous()
    zpre = c2r_pre(X, n, block=256)
    zref = c2r_pre_ref(X, n)
    rel = max_rel_error(_to_np(zpre), _to_np(zref))
    assert rel < 1e-6, f"c2r_pre kernel vs ref: rel={rel:.3e}"

    u = torch.complex(torch.randn(b, n // 2),
                      torch.randn(b, n // 2)).to("cuda")
    out = c2r_unpack(u, n, 0.125, block=256)
    oref = c2r_unpack_ref(u, n, 0.125)
    assert out.dtype == torch.float32
    rel = max_rel_error(_to_np(out), _to_np(oref))
    assert rel < 1e-6, f"c2r_unpack kernel vs ref: rel={rel:.3e}"


@pytest.mark.parametrize("tt", ["r2c", "c2r"])
def test_dedicated_matches_compat_path(tt):
    """Half-length pipeline must be numerically identical to compat."""
    n, b = 1024, 16
    x = _make_real("random", n, b, seed=41)
    if tt == "c2r":
        half = np.fft.rfft(_to_np(x), axis=-1).astype(np.complex64)
        inp = torch.from_numpy(half).to("cuda")
    else:
        inp = x
    plan = _dedicated_plan(tt, n, b, "backward")
    got = execute_fft_plan(plan, inp)
    want = execute_fft_plan(_compat_plan(plan), inp)
    rel = max_rel_error(_to_np(got), _to_np(want))
    assert rel < 1e-5, f"dedicated vs compat {tt}: rel={rel:.3e}"


def test_dedicated_c2r_one_shot_stable():
    n, b = 256, 16
    x = _make_real("random", n, b, seed=5)
    half = np.fft.rfft(_to_np(x), axis=-1).astype(np.complex64)
    half_t = torch.from_numpy(half).to("cuda")
    plan = _dedicated_plan("c2r", n, b, "backward")
    y0 = execute_fft_plan(plan, half_t)
    y1 = execute_fft_plan(plan, half_t)
    assert torch.equal(y0, y1)
