"""Hardware correctness tests: Triton XPU FFT vs numpy (CPU, complex128).

Marked ``hardware`` (see conftest.py): skipped cleanly when the vendor
xpytorch stack or a reachable XPU is absent. The reference is numpy because
the vendor torch.fft.fft caps at N <= 2048 on this device.
"""

import math
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

# The vendor stack must be imported torch-first (the xpytorch import hook
# resolves libtorch_python.so only after `import torch`); importorskip races
# that order, so use a plain import and skip cleanly on failure.
try:
    import torch
    import torch_xmlir  # noqa: F401
except Exception as e:  # pragma: no cover - absent in ordinary CI
    pytest.skip(f"vendor xpytorch stack unavailable: {e}", allow_module_level=True)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.hardware

from src.xpu import (  # noqa: E402
    device_ready,
    fft_dif,
    fft_dit,
    fft_dit_fused,
    fft_dif_matrix,
    fft_dit_matrix,
    fft_dif_grouped,
    fft_dit_grouped,
)

if not device_ready():  # pragma: no cover - guarded again by the fixture
    pytest.skip("XPU device not ready", allow_module_level=True)


def max_err(a, np_ref) -> float:
    """Max abs error against a numpy complex128 reference."""
    ar = a.real.cpu().numpy().astype(np.float64)
    ai = a.imag.cpu().numpy().astype(np.float64)
    return float(np.maximum(np.abs(ar - np_ref.real), np.abs(ai - np_ref.imag)).max())


def max_mag(np_ref) -> float:
    return float(np.abs(np_ref).max())


def make_input(n: int, seed: int):
    torch.manual_seed(seed)
    x = torch.complex(
        torch.randn(n, dtype=torch.float32, device="cuda"),
        torch.randn(n, dtype=torch.float32, device="cuda"),
    )
    host = x.real.cpu().numpy() + 1j * x.imag.cpu().numpy()
    return x, np.fft.fft(host)


def test_bitrev_only():
    from src.xpu import bit_reverse_permute

    n = 64
    x = torch.complex(
        torch.arange(n, dtype=torch.float32, device="cuda"),
        torch.arange(n, dtype=torch.float32, device="cuda") * 0.5,
    )
    y = bit_reverse_permute(x)
    bits = n.bit_length() - 1
    expect = torch.tensor(
        [int(format(i, f"0{bits}b")[::-1], 2) for i in range(n)],
        dtype=torch.int64,
        device="cuda",
    )
    xr = x.real.to(torch.int64)
    yr = y.real.to(torch.int64)
    assert torch.equal(yr, xr[expect]), "bit-reversal mismatch"


@pytest.mark.parametrize("n", [8, 16, 64, 256, 1024, 4096, 65536, 1048576])
def test_dit_sizes(n):
    x, ref = make_input(n, 0)
    rel = max_err(fft_dit(x), ref) / max_mag(ref)
    assert rel < 1e-4, f"DIT failed at N={n}: rel={rel:.3e}"


@pytest.mark.parametrize("n", [8, 16, 64, 256, 1024, 4096, 65536])
def test_dif_sizes(n):
    x, ref = make_input(n, 1)
    rel = max_err(fft_dif(x), ref) / max_mag(ref)
    assert rel < 1e-4, f"DIF failed at N={n}: rel={rel:.3e}"


@pytest.mark.parametrize("n", [8, 64, 1024, 4096, 65536, 1048576])
def test_fused_sizes(n):
    """M1 local-fused-tile candidate must match numpy at every size."""
    x, ref = make_input(n, 0)
    rel = max_err(fft_dit_fused(x, tile=64), ref) / max_mag(ref)
    assert rel < 1e-4, f"fused failed at N={n}: rel={rel:.3e}"


@pytest.mark.parametrize("n", [8, 64, 1024, 4096, 65536, 1048576])
def test_matrix_dit_sizes(n):
    """M2 register-resident DIT (matrix local group) vs numpy at every size."""
    x, ref = make_input(n, 0)
    rel = max_err(fft_dit_matrix(x, tile=64), ref) / max_mag(ref)
    assert rel < 1e-4, f"matrix DIT failed at N={n}: rel={rel:.3e}"


@pytest.mark.parametrize("n", [8, 64, 1024, 4096, 65536])
def test_matrix_dif_sizes(n):
    """M2 register-resident DIF (matrix local group) vs numpy at every size."""
    x, ref = make_input(n, 1)
    rel = max_err(fft_dif_matrix(x, tile=64), ref) / max_mag(ref)
    assert rel < 1e-4, f"matrix DIF failed at N={n}: rel={rel:.3e}"


@pytest.mark.parametrize("tile", [8, 16, 64])
def test_matrix_tile_axis_is_neutral(tile):
    """tile changes the group size, never the result (axes are semantics neutral)."""
    n = 4096
    x, ref = make_input(n, 2)
    rel = max_err(fft_dit_matrix(x, tile=tile), ref) / max_mag(ref)
    assert rel < 1e-4, f"tile={tile} failed: rel={rel:.3e}"


@pytest.mark.parametrize("rows", [1, 8])
def test_matrix_rows_is_neutral(rows):
    """rows only batches tile-blocks per program; results must be identical."""
    n = 4096
    x, ref = make_input(n, 3)
    rel = max_err(fft_dif_matrix(x, rows=rows), ref) / max_mag(ref)
    assert rel < 1e-4, f"rows={rows} failed: rel={rel:.3e}"


@pytest.mark.parametrize("n", [8, 64, 1024, 4096, 65536, 1048576])
def test_grouped_dit_sizes(n):
    """M3 execution-group DIT (grouped tail) vs numpy at every size."""
    x, ref = make_input(n, 0)
    rel = max_err(fft_dit_grouped(x, tile=64), ref) / max_mag(ref)
    assert rel < 1e-4, f"grouped DIT failed at N={n}: rel={rel:.3e}"


@pytest.mark.parametrize("n", [8, 64, 1024, 4096, 65536])
def test_grouped_dif_sizes(n):
    """M3 execution-group DIF (grouped tail) vs numpy at every size."""
    x, ref = make_input(n, 1)
    rel = max_err(fft_dif_grouped(x, tile=64), ref) / max_mag(ref)
    assert rel < 1e-4, f"grouped DIF failed at N={n}: rel={rel:.3e}"


@pytest.mark.parametrize("glog", [1, 2, 3])
def test_grouped_glog_axis_is_neutral(glog):
    """glog changes group depth, never the result (glog=1 disables grouping)."""
    n = 4096
    x, ref = make_input(n, 4)
    rel = max_err(fft_dit_grouped(x, tile=8, glog=glog), ref) / max_mag(ref)
    assert rel < 1e-4, f"glog={glog} failed: rel={rel:.3e}"


@pytest.mark.parametrize("maxp", [128, 4096, 8192])
def test_grouped_maxp_axis_is_neutral(maxp):
    """maxp only caps the group period (maxp=128 forces all singles)."""
    n = 4096
    x, ref = make_input(n, 5)
    rel = max_err(fft_dif_grouped(x, tile=8, glog=2, maxp=maxp), ref) / max_mag(ref)
    assert rel < 1e-4, f"maxp={maxp} failed: rel={rel:.3e}"


@pytest.mark.parametrize("cells", [1, 4])
def test_grouped_cells_axis_is_neutral(cells):
    """cells only packs BLOCK windows per program; results must be identical."""
    n = 65536
    x, ref = make_input(n, 6)
    rel = max_err(fft_dit_grouped(x, tile=8, glog=2, cells=cells), ref) / max_mag(ref)
    assert rel < 1e-4, f"cells={cells} failed: rel={rel:.3e}"


def test_known_signals():
    """Impulse -> flat spectrum; single tone -> one bin."""
    n = 256
    impulse = torch.zeros(n, dtype=torch.complex64, device="cuda")
    impulse[0] = 1.0
    got = fft_dit(impulse)
    ones = np.ones(n, dtype=np.complex128)
    assert max_err(got, ones) < 1e-5, "impulse failed"

    k0 = 5
    ang = 2.0 * math.pi * k0 * torch.arange(n, device="cuda", dtype=torch.float32) / n
    tone = torch.complex(torch.cos(ang), torch.sin(ang))
    got = fft_dit(tone)
    host = tone.real.cpu().numpy() + 1j * tone.imag.cpu().numpy()
    ref = np.fft.fft(host)
    assert max_err(got, ref) < 1e-4, "tone failed"


def test_sentinel_output_changes():
    """Launch-path gate: a non-zero input must produce a changed, finite output."""
    n = 512
    base = torch.arange(1.0, n + 1.0, device="cuda")
    x = torch.complex(base, -base)
    y = fft_dit(x)
    assert bool(torch.isfinite(y.real).all())
    assert not torch.equal(y.real, x.real), "output identical to input: kernel did not run"
    assert float(y.real.abs().sum()) > 0.0


def test_negative_inputs_rejected():
    """Wrong semantic inputs must raise before any launch."""
    good = torch.complex(torch.ones(16, device="cuda"), torch.zeros(16, device="cuda"))
    with pytest.raises(ValueError):
        fft_dit(good.reshape(4, 4))          # rank
    with pytest.raises(TypeError):
        fft_dit(good.real)                   # dtype
    with pytest.raises(ValueError):
        fft_dit(torch.complex(torch.ones(12, device="cuda"),
                              torch.zeros(12, device="cuda")))  # non-pow2
    with pytest.raises(ValueError):
        fft_dit(torch.complex(torch.ones(16), torch.zeros(16)))  # device


@triton.jit
def _pointer_gather_kernel(X, Y, IDX, N: tl.constexpr):
    cols = tl.arange(0, N)
    idx = tl.load(IDX + cols)
    tl.store(Y + cols, tl.load(X + idx))


@triton.jit
def _tensor_subscript_kernel(X, Y, N: tl.constexpr):
    cols = tl.arange(0, N)
    Xt = tl.load(X + cols)
    tl.store(Y + cols, Xt[cols % N])


def test_pointer_form_gather_lowers():
    """Runtime pointer arithmetic offsets lower on this backend.

    This is the form that keeps register/UB residence (M2) on the ladder of
    legal alternatives; if this ever starts failing, the ladder itself is
    blocked and docs/device-stack.md must say so.
    """
    x = torch.arange(64, dtype=torch.float32, device="cuda")
    idx = torch.arange(63, -1, -1, dtype=torch.int32, device="cuda")
    y = torch.empty_like(x)
    _pointer_gather_kernel[(1,)](x, y, idx, 64)
    torch.cuda.synchronize()
    assert torch.equal(y, x.flip(0)), "pointer-form gather produced wrong values"


def test_tensor_subscript_gather_blocked():
    """Canary for the one indexing form the tested lowering rejects.

    `Xt[idx_tensor]` (gather from an SSA tensor via a tensor index) raises
    CompilationError/ValueError on the Triton 3.0.0 XPU build. When a future
    toolchain accepts it, this test FAILS — which is the signal that the
    register/UB-residence (M2) SSA form has become expressible and the
    limitation recorded in docs/device-stack.md no longer applies.
    """
    x = torch.arange(64, dtype=torch.float32, device="cuda")
    y = torch.empty_like(x)
    with pytest.raises(Exception) as exc:
        _tensor_subscript_kernel[(1,)](x, y, 64)
    assert "triton.jit" in str(exc.value) or "CompilationError" in str(exc.value)


if __name__ == "__main__":
    if not device_ready():
        print("no XPU device available, skipping")
        sys.exit(0)
    test_bitrev_only()
    for n in (8, 16, 64, 256, 1024, 4096, 65536, 1048576):
        test_dit_sizes(n)
    for n in (8, 16, 64, 256, 1024, 4096, 65536):
        test_dif_sizes(n)
    for n in (8, 64, 1024, 4096, 65536, 1048576):
        test_fused_sizes(n)
    for n in (8, 64, 1024, 4096, 65536, 1048576):
        test_matrix_dit_sizes(n)
    for n in (8, 64, 1024, 4096, 65536):
        test_matrix_dif_sizes(n)
    for tile in (8, 16, 64):
        test_matrix_tile_axis_is_neutral(tile)
    for rows in (1, 8):
        test_matrix_rows_is_neutral(rows)
    for n in (8, 64, 1024, 4096, 65536, 1048576):
        test_grouped_dit_sizes(n)
    for n in (8, 64, 1024, 4096, 65536):
        test_grouped_dif_sizes(n)
    for glog in (1, 2, 3):
        test_grouped_glog_axis_is_neutral(glog)
    for maxp in (128, 4096, 8192):
        test_grouped_maxp_axis_is_neutral(maxp)
    for cells in (1, 4):
        test_grouped_cells_axis_is_neutral(cells)
    test_known_signals()
    test_sentinel_output_changes()
    test_negative_inputs_rejected()
    test_pointer_form_gather_lowers()
    test_tensor_subscript_gather_blocked()
    print("all correctness tests passed")
