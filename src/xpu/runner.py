"""Host-side driver for the P800 Triton FFT.

Every device launch runs inside :func:`xpu_context.xpu_context`, the single
central device-context manager (Phase 0). Public entry points validate their
inputs via :mod:`.validate` before any launch.

Complex values are carried split-real/imaginary (planar SoA) internally because
the Triton XPU backend has no ``complex64`` argument type. The **public** API
accepts and returns ``complex64``; planar is an internal ABI, not the public
layout.

Flow identifiers (migration convergence plan):

* ``baseline-global-stage`` (M0) — one global-memory round trip and one kernel
  launch per FFT stage. The current compatibility baseline.
* ``local-fused-tile`` (M1) — the first ``log2(TILE)`` block-local DIT stages
  are fused into one kernel (global-pointer form, no SSA gather); the remaining
  stages use the M0 path.
"""

import math
import random
import time

import torch

try:  # the vendor runtime that turns 'cuda' into the XPU device
    import torch_xmlir  # noqa: F401
except ImportError:  # pragma: no cover - only present in the P800 container
    torch_xmlir = None

import triton

from .context import device_ready, xpu_context
from .kernels import (
    bitrev_kernel,
    dif_stage_kernel,
    dit_fused_kernel,
    dit_stage_kernel,
)
from .validate import log2_exact, validate_input, validate_twiddle

_XPU_MODEL = {3: "P800"}


def device_name() -> str:
    """Human-readable device string.

    ``torch.cuda.get_device_properties`` is stubbed by the XPU stack (it reports
    a generic "GPU" with 8 multiprocessors), so the real topology comes from the
    Triton XPU driver, which reads the XMLIR device model.
    """
    if not device_ready():
        return "cpu"
    try:
        from triton.runtime.driver import driver

        props = driver.active.utils.get_device_properties(0)
        model = _XPU_MODEL.get(props.get("device_model"), "xpu")
        return (
            f"kunlunxin-{model} (xpu arch={props.get('device_model')}, "
            f"{props.get('multiprocessor_count')} clusters, "
            f"{props.get('max_shared_mem') // 1024}KiB smem)"
        )
    except Exception:
        return "kunlunxin-xpu"


def twiddle_tables(n: int, device):
    """Cosine/sine tables of exp(-2j*pi*idx/n) for idx in [0, n/2)."""
    idx = torch.arange(n >> 1, device=device, dtype=torch.float32)
    ang = (-2.0 * math.pi * idx / n).to(torch.float32)
    return torch.cos(ang), torch.sin(ang)


def _as_so(x: torch.Tensor):
    """Split a complex tensor into (real, imag) float32 views."""
    return x.real.contiguous(), x.imag.contiguous()


def _from_so(re: torch.Tensor, im: torch.Tensor) -> torch.Tensor:
    return torch.complex(re, im)


def _launch(kernel, block: int, grid: int, *args, **kw):
    kernel[(grid,)](*args, BLOCK=block, **kw)


def fft_dit(
    x: torch.Tensor,
    block: int = 1024,
    twiddle=None,
) -> torch.Tensor:
    """M0 baseline: complex FFT via DIT stages + folded bit-reversal.

    ``x`` must be a contiguous 1-D complex64 tensor on the XPU device. Returns a
    new complex64 tensor in natural order. One kernel launch and one global
    memory round trip per stage (flow ``baseline-global-stage``).
    """
    validate_input(x, "x", block=block)
    n = x.numel()
    logn = log2_exact(n)
    if twiddle is None:
        twiddle = twiddle_tables(n, x.device)
    validate_twiddle(twiddle, n, x.device)
    wr, wi = twiddle
    half = n >> 1
    grid = triton.cdiv(half, block)

    with xpu_context(x.device.index or 0):
        cr, ci = _as_so(x)
        for s in range(logn):
            stage_n = 1 << s
            M = n // (stage_n << 1)
            nr = torch.empty_like(cr)
            ni = torch.empty_like(ci)
            _launch(
                dit_stage_kernel,
                block,
                grid,
                cr,
                ci,
                nr,
                ni,
                wr,
                wi,
                stage_n,
                M,
                n,
                logn,
                BITREV_INPUT=(s == 0),
            )
            cr, ci = nr, ni
    return _from_so(cr, ci)


def fft_dif(
    x: torch.Tensor,
    block: int = 1024,
    twiddle=None,
) -> torch.Tensor:
    """M0 baseline: complex FFT via DIF stages + folded bit-reversal."""
    validate_input(x, "x", block=block)
    n = x.numel()
    logn = log2_exact(n)
    if twiddle is None:
        twiddle = twiddle_tables(n, x.device)
    validate_twiddle(twiddle, n, x.device)
    wr, wi = twiddle
    half = n >> 1
    grid = triton.cdiv(half, block)

    with xpu_context(x.device.index or 0):
        cr, ci = _as_so(x)
        for s in range(logn - 1, -1, -1):
            stage_n = 1 << s
            M = n // (stage_n << 1)
            nr = torch.empty_like(cr)
            ni = torch.empty_like(ci)
            _launch(
                dif_stage_kernel,
                block,
                grid,
                cr,
                ci,
                nr,
                ni,
                wr,
                wi,
                stage_n,
                M,
                n,
                logn,
                BITREV_OUTPUT=(s == 0),
            )
            cr, ci = nr, ni
    return _from_so(cr, ci)


def fft_dit_fused(
    x: torch.Tensor,
    block: int = 1024,
    tile: int = 64,
    twiddle=None,
) -> torch.Tensor:
    """M1: DIT FFT fusing the first log2(tile) block-local stages (``local-fused-tile``).

    ``tile`` must be a power of two dividing ``N``. The first ``log2(tile)`` DIT
    stages are block-local, so they are fused into one kernel using global-pointer
    loads/stores (no SSA gather — the form the tested Triton XPU lowering
    accepts). An explicit bit-reversal precedes the fused stages; the remaining
    ``log2(N) - log2(tile)`` stages use the M0 global-stage path.

    Launch count: ``1 (bitrev) + 1 (fused) + log2(N) - log2(tile)`` vs M0's
    ``log2(N)``. The fused stages' intermediate values stay inside a tile-sized
    block (L1/L2-resident) instead of round-tripping per stage.
    """
    validate_input(x, "x", block=block)
    n = x.numel()
    logn = log2_exact(n)
    if tile <= 1:
        return fft_dit(x, block=block, twiddle=twiddle)
    log2_exact(tile)  # raises unless tile is a power of two
    # Both tile and n are powers of two, so min(tile, n) divides n. For sizes
    # smaller than the requested tile the candidate degenerates to a fully
    # fused single-block transform (tile == n).
    eff_tile = min(tile, n)
    fused_log = log2_exact(eff_tile)
    if twiddle is None:
        twiddle = twiddle_tables(n, x.device)
    validate_twiddle(twiddle, n, x.device)
    wr, wi = twiddle
    half = n >> 1

    with xpu_context(x.device.index or 0):
        # 1. explicit bit-reversal (global, out of place)
        xr, xi = _as_so(x)
        cur_r = torch.empty_like(xr)
        cur_i = torch.empty_like(xi)
        br_grid = triton.cdiv(n, block)
        _launch(bitrev_kernel, block, br_grid, xr, xi, cur_r, cur_i, n, logn)

        # 2. fused block-local stages, in place on cur (grid = N/tile programs).
        #    Stages n < eff_tile never cross a tile boundary, so no cross-program
        #    barrier is required; only an intra-program ordering barrier.
        fuse_grid = n // eff_tile
        _launch(
            dit_fused_kernel,
            eff_tile,
            fuse_grid,
            cur_r,
            cur_i,
            wr,
            wi,
            n,
            logn,
            eff_tile,
            fused_log,
        )

        # 3. remaining stages (n = eff_tile .. N/2) on the M0 global-stage path
        if fused_log < logn:
            grid = triton.cdiv(half, block)
            for s in range(fused_log, logn):
                stage_n = 1 << s
                M = n // (stage_n << 1)
                nr = torch.empty_like(cur_r)
                ni = torch.empty_like(cur_i)
                _launch(
                    dit_stage_kernel,
                    block,
                    grid,
                    cur_r,
                    cur_i,
                    nr,
                    ni,
                    wr,
                    wi,
                    stage_n,
                    M,
                    n,
                    logn,
                    BITREV_INPUT=False,
                )
                cur_r, cur_i = nr, ni
    return _from_so(cur_r, cur_i)


def bit_reverse_permute(x: torch.Tensor, block: int = 1024) -> torch.Tensor:
    validate_input(x, "x", block=block, require_complex=True)
    n = x.numel()
    bits = log2_exact(n)
    with xpu_context(x.device.index or 0):
        xr, xi = _as_so(x)
        yr = torch.empty_like(xr)
        yi = torch.empty_like(xi)
        grid = triton.cdiv(n, block)
        _launch(bitrev_kernel, block, grid, xr, xi, yr, yi, n, bits)
    return _from_so(yr, yi)


def _sync():
    if device_ready():
        torch.cuda.synchronize()


def time_call(fn, warmup: int = 5, iters: int = 20) -> float:
    """Return median **API wall time** in milliseconds (includes Python dispatch,
    planar copies, allocations, output assembly, and synchronization)."""
    for _ in range(warmup):
        fn()
    _sync()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        _sync()
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return samples[len(samples) // 2]


def time_call_detailed(fn, warmup: int = 10, trials: int = 50, seed: int = 0):
    """Phase-2 timing protocol: interleaved, randomized trials.

    Returns a dict with API wall-time stats (median/min/p10/p90) and, when the
    runtime exposes device events, a separate kernel-time estimate. The two
    metrics are reported separately because API wall time includes Python
    dispatch, allocation, and layout conversion that kernel time does not.
    """
    for _ in range(warmup):
        fn()
    _sync()

    use_events = device_ready()
    api_samples = []
    kernel_samples = []
    order = list(range(trials))
    rng = random.Random(seed)
    rng.shuffle(order)

    for _ in order:
        if use_events:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter()
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            api_samples.append((time.perf_counter() - t0) * 1e3)
            # elapsed_time is in ms; XPU events may return 0 if unsupported
            kt = start.elapsed_time(end)
            if kt > 0:
                kernel_samples.append(kt)
        else:
            t0 = time.perf_counter()
            fn()
            _sync()
            api_samples.append((time.perf_counter() - t0) * 1e3)

    def _stats(xs):
        if not xs:
            return None
        xs = sorted(xs)
        return {
            "median": xs[len(xs) // 2],
            "min": xs[0],
            "p10": xs[max(0, int(0.10 * len(xs)))],
            "p90": xs[min(len(xs) - 1, int(0.90 * len(xs)))],
            "trials": len(xs),
        }

    return {
        "api_ms": _stats(api_samples),
        "kernel_ms": _stats(kernel_samples),
        "metric_note": (
            "api_ms = synchronized wall time (dispatch+alloc+layout+sync); "
            "kernel_ms = CUDA/XPU event elapsed (device time) when events report > 0, "
            "else null"
        ),
    }
