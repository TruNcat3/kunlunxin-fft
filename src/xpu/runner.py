"""Host-side driver for the P800 Triton FFT.

The XPU device context is established by importing ``torch`` + ``torch_xmlir``
(see docs/device-stack.md). After that ``device='cuda'`` addresses the
Kunlunxin P800 and Triton compiles/launches device code.

Complex values are carried split-real/imaginary because the Triton XPU backend
has no ``complex64`` argument type.
"""

import math
import time

import torch

try:  # the vendor runtime that turns 'cuda' into the XPU device
    import torch_xmlir  # noqa: F401
except ImportError:  # pragma: no cover - only present in the P800 container
    torch_xmlir = None

import triton

from .kernels import bitrev_kernel, dif_stage_kernel, dit_stage_kernel


def device_ready() -> bool:
    """True when an XPU device is addressable through the torch 'cuda' API."""
    return torch_xmlir is not None and torch.cuda.is_available()


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


def log2_exact(n: int) -> int:
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"size must be a power of two, got {n}")
    return int(math.log2(n))


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


def _launch(kernel, half: int, block: int, grid: int, *args, **kw):
    kernel[(grid,)](*args, BLOCK=block, **kw)


def fft_dit(
    x: torch.Tensor,
    block: int = 1024,
    twiddle=None,
) -> torch.Tensor:
    """Complex FFT of a 1-D power-of-two tensor, DIT stages + folded bit-reversal.

    ``x`` must be contiguous complex64 on the XPU device. Returns a new tensor
    in natural order.
    """
    n = x.numel()
    logn = log2_exact(n)
    if twiddle is None:
        twiddle = twiddle_tables(n, x.device)
    wr, wi = twiddle
    half = n >> 1
    grid = triton.cdiv(half, block)

    cr, ci = _as_so(x)
    for s in range(logn):
        stage_n = 1 << s
        M = n // (stage_n << 1)
        nr = torch.empty_like(cr)
        ni = torch.empty_like(ci)
        _launch(
            dit_stage_kernel,
            half,
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
    """Complex FFT via DIF stages with bit-reversal folded into the last stage."""
    n = x.numel()
    logn = log2_exact(n)
    if twiddle is None:
        twiddle = twiddle_tables(n, x.device)
    wr, wi = twiddle
    half = n >> 1
    grid = triton.cdiv(half, block)

    cr, ci = _as_so(x)
    for s in range(logn - 1, -1, -1):
        stage_n = 1 << s
        M = n // (stage_n << 1)
        nr = torch.empty_like(cr)
        ni = torch.empty_like(ci)
        _launch(
            dif_stage_kernel,
            half,
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


def bit_reverse_permute(x: torch.Tensor, block: int = 1024) -> torch.Tensor:
    n = x.numel()
    bits = log2_exact(n)
    xr, xi = _as_so(x)
    yr = torch.empty_like(xr)
    yi = torch.empty_like(xi)
    grid = triton.cdiv(n, block)
    bitrev_kernel[(grid,)](xr, xi, yr, yi, n, bits, BLOCK=block)
    return _from_so(yr, yi)


def _sync():
    if device_ready():
        torch.cuda.synchronize()


def time_call(fn, warmup: int = 5, iters: int = 20) -> float:
    """Return median wall time in milliseconds."""
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
