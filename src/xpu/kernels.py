"""Triton device kernels for the Kunlunxin P800 FFT.

Device code path on P800 is Triton -> XMLIR -> XPU instructions. nvcc/PTX
binaries cannot execute on this device (see docs/device-stack.md).

The Triton XPU backend has no ``complex64`` argument type, so complex data is
carried split-real/imaginary (SoA): two ``float32`` buffers, plus a twiddle
table stored as separate cosine/sine tables.

Building blocks:

* ``bitrev_kernel``     -- out-of-place bit-reversal permutation.
* ``dit_stage_kernel``  -- one radix-2 decimation-in-time butterfly stage.
* ``dif_stage_kernel``  -- one radix-2 decimation-in-frequency butterfly stage.

The first DIT stage reads its input in bit-reversed order (``BITREV_INPUT``),
so a full FFT is ``log2(N)`` stage launches with no separate permutation pass.
"""

import triton
import triton.language as tl


@triton.jit
def _bit_reverse(x, BITS: tl.constexpr):
    """Reverse the low ``BITS`` bits of the integer tensor ``x``.

    ``BITS`` is a compile-time constant so the loop is fully unrolled.
    """
    rev = tl.zeros_like(x)
    v = x
    for _ in range(BITS):
        rev = (rev << 1) | (v & 1)
        v = v >> 1
    return rev


@triton.jit
def bitrev_kernel(
    Xr,
    Xi,
    Yr,
    Yi,
    N,
    BITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Permute a split-complex array into bit-reversed order."""
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < N
    src = _bit_reverse(off, BITS)
    tl.store(Yr + off, tl.load(Xr + src, mask=mask), mask=mask)
    tl.store(Yi + off, tl.load(Xi + src, mask=mask), mask=mask)


@triton.jit
def dit_stage_kernel(
    Xr,
    Xi,
    Yr,
    Yi,
    Wr,
    Wi,
    n,
    M,
    N,
    BITS: tl.constexpr,
    BLOCK: tl.constexpr,
    BITREV_INPUT: tl.constexpr,
):
    """One radix-2 DIT butterfly stage on split-complex data.

    Reads ``k`` in [0, N/2); ``j = k % n``, ``i = (k // n) * 2n + j``::

        a = X[i]
        b = X[i + n] * W[j * M]
        Y[i]     = a + b
        Y[i + n] = a - b

    ``n`` is the butterfly half-span (1, 2, 4, ..., N/2) and ``M = N / (2n)``
    is the twiddle stride. ``Wr``/``Wi`` hold ``exp(-2j*pi*idx/N)`` split into
    cosine/sine for idx in [0, N/2); only ``W[j * M]`` with ``j < n`` is read,
    i.e. indices below N/2.

    When ``BITREV_INPUT`` is set the first stage loads ``X[bit_reverse(i)]``
    instead of ``X[i]``, folding the input permutation into this stage.
    """
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    half = N >> 1
    mask = off < half

    j = off % n
    i = ((off // n) << 1) * n + j

    if BITREV_INPUT:
        src = _bit_reverse(i, BITS)
        src2 = _bit_reverse(i + n, BITS)
    else:
        src = i
        src2 = i + n

    w = j * M
    wr = tl.load(Wr + w, mask=mask)
    wi = tl.load(Wi + w, mask=mask)

    ar = tl.load(Xr + src, mask=mask)
    ai = tl.load(Xi + src, mask=mask)
    br = tl.load(Xr + src2, mask=mask)
    bi = tl.load(Xi + src2, mask=mask)

    # b = X[i+n] * W
    brw = br * wr - bi * wi
    biw = br * wi + bi * wr

    tl.store(Yr + i, ar + brw, mask=mask)
    tl.store(Yi + i, ai + biw, mask=mask)
    tl.store(Yr + i + n, ar - brw, mask=mask)
    tl.store(Yi + i + n, ai - biw, mask=mask)


@triton.jit
def dif_stage_kernel(
    Xr,
    Xi,
    Yr,
    Yi,
    Wr,
    Wi,
    n,
    M,
    N,
    BITS: tl.constexpr,
    BLOCK: tl.constexpr,
    BITREV_OUTPUT: tl.constexpr,
):
    """One radix-2 DIF butterfly stage on split-complex data.

    ``n`` runs N/2, N/4, ..., 1. Same index mapping as the DIT stage, but the
    twiddle is applied to the difference::

        a = X[i]; b = X[i + n]
        Y[i]     = a + b
        Y[i + n] = (a - b) * W[j * M]

    When ``BITREV_OUTPUT`` is set the final stage writes through a
    bit-reversal permutation, so the result is in natural order.
    """
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    half = N >> 1
    mask = off < half

    j = off % n
    i = ((off // n) << 1) * n + j

    w = j * M
    wr = tl.load(Wr + w, mask=mask)
    wi = tl.load(Wi + w, mask=mask)

    ar = tl.load(Xr + i, mask=mask)
    ai = tl.load(Xi + i, mask=mask)
    br = tl.load(Xr + i + n, mask=mask)
    bi = tl.load(Xi + i + n, mask=mask)

    sr = ar + br
    si = ai + bi
    dr = ar - br
    di = ai - bi
    drw = dr * wr - di * wi
    diw = dr * wi + di * wr

    if BITREV_OUTPUT:
        d0 = _bit_reverse(i, BITS)
        d1 = _bit_reverse(i + n, BITS)
        tl.store(Yr + d0, sr, mask=mask)
        tl.store(Yi + d0, si, mask=mask)
        tl.store(Yr + d1, drw, mask=mask)
        tl.store(Yi + d1, diw, mask=mask)
    else:
        tl.store(Yr + i, sr, mask=mask)
        tl.store(Yi + i, si, mask=mask)
        tl.store(Yr + i + n, drw, mask=mask)
        tl.store(Yi + i + n, diw, mask=mask)
