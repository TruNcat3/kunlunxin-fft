"""Input validation for the public FFT API (migration convergence plan Phase 0).

The audit found the public API inferred shape from ``x.numel()`` with no
rank/dtype/device/contiguity checks, so a wrong semantic input could silently
enter an incorrect path. These checks reject bad inputs before any launch.
"""

import torch

from .context import device_ready


def log2_exact(n: int) -> int:
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"size must be a positive power of two, got {n}")
    return int(n.bit_length() - 1)


def validate_input(x, name: str = "x", *, allow_n: int = None, block: int = None,
                   require_complex: bool = True) -> None:
    """Validate a public-API input tensor.

    Checks (Phase 0 deliverable 3): rank-1, complex64 (when required),
    contiguous, on the XPU device, and a power-of-two length. Optionally
    cross-checks an expected length and a power-of-two block.

    Contract boundary (issue #10 review): this helper is the public
    single-length (1-D) contract. Batched execution is a separate
    entry/runner-level contract — ``plan_fft(batch=b)`` /
    ``execute_fft_plan`` on rank-2 ``(B, N)`` inputs dispatch to the
    ``fft_*_batch`` runners, which validate via
    ``runner._validate_batch_input`` (rank-2, contiguous, complex64,
    power-of-two axes); ``run_fft`` accepts rank-2 ``(B, N)`` as well.
    The batched rows in ``results/p800/issue10_capability/`` are
    labelled as that entry-level evidence; extending this 1-D helper to
    batched inputs is a separate contract change.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x).__name__}")

    if x.dim() != 1:
        raise ValueError(f"{name} must be 1-D, got {x.dim()}-D (shape {tuple(x.shape)}); "
                         "batched APIs are not implemented yet")

    if require_complex and x.dtype != torch.complex64:
        raise TypeError(f"{name} must be complex64, got {x.dtype}; "
                        "the public layout is complex64 (planar is an internal ABI)")

    if not x.is_contiguous():
        raise ValueError(f"{name} must be contiguous; got a non-contiguous view. "
                         "Clone or contiguous() it before calling")

    if not device_ready():
        raise RuntimeError("XPU device stack not available; cannot validate device")

    if x.device.type != "cuda":
        raise ValueError(f"{name} must be on the XPU device (device='cuda'), got device={x.device}")

    n = x.numel()
    log2_exact(n)  # raises on non-power-of-two

    if allow_n is not None and n != allow_n:
        raise ValueError(f"{name} length {n} != expected {allow_n}")

    if block is not None:
        # `block` is the Triton vector width / grid stride; cdiv() handles
        # N < block and non-divisible N, so it only needs to be a power of two.
        log2_exact(block)


def validate_twiddle(tw, n: int, device) -> None:
    """Validate a precomputed twiddle table against the FFT length."""
    if not isinstance(tw, tuple) or len(tw) != 2:
        raise TypeError("twiddle must be a (cos, sin) tuple from twiddle_tables()")
    wr, wi = tw
    for t, nm in ((wr, "cos"), (wi, "sin")):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"twiddle {nm} must be a torch.Tensor")
        if t.dtype != torch.float32:
            raise TypeError(f"twiddle {nm} must be float32, got {t.dtype}")
        if t.device != device:
            raise ValueError(f"twiddle {nm} device {t.device} != input device {device}")
        if t.numel() != n >> 1:
            raise ValueError(f"twiddle {nm} length {t.numel()} != N/2 ({n >> 1})")
