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


@triton.jit
def dit_fused_kernel(
    Xr,
    Xi,
    Wr,
    Wi,
    N,
    BITS: tl.constexpr,
    TILE: tl.constexpr,
    FUSED_LOG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """M1: fuse the first FUSED_LOG = log2(TILE) block-local DIT stages.

    Stages with butterfly half-span n < TILE are **block-local**: a pair
    (i, i+n) with i in a TILE-aligned block stays inside that block, so there is
    no cross-program edge and the stages need no inter-program barrier. Each
    program therefore owns one TILE-block and applies the first log2(TILE)
    stages in place using only global pointer tl.load/tl.store (the form the
    tested Triton XPU lowering accepts — no SSA gather).

    Reads the bit-reversed input (the caller runs bitrev_kernel first) and
    leaves the block in place for the remaining stages. ``BLOCK`` == ``TILE``.
    """
    pid = tl.program_id(0)
    base = pid * TILE

    half_tile = TILE >> 1
    off = tl.arange(0, BLOCK)
    # butterfly index t in [0, TILE/2); only these lanes do work
    kmask = off < half_tile

    for s in range(FUSED_LOG):
        n = 1 << s
        M = N // (n << 1)
        # within the TILE-block: subblock m = t // n, lane j = t % n
        m = off // n
        j = off % n
        i = base + (m << 1) * n + j
        hi = i + n
        w = j * M

        wr = tl.load(Wr + w, mask=kmask)
        wi = tl.load(Wi + w, mask=kmask)
        ar = tl.load(Xr + i, mask=kmask)
        ai = tl.load(Xi + i, mask=kmask)
        br = tl.load(Xr + hi, mask=kmask)
        bi = tl.load(Xi + hi, mask=kmask)

        brw = br * wr - bi * wi
        biw = br * wi + bi * wr

        tl.store(Xr + i, ar + brw, mask=kmask)
        tl.store(Xi + i, ai + biw, mask=kmask)
        tl.store(Xr + hi, ar - brw, mask=kmask)
        tl.store(Xi + hi, ai - biw, mask=kmask)
        # compiler + intra-program ordering between stages (blocks are disjoint,
        # so this does not synchronize across programs)
        tl.debug_barrier()


@triton.jit
def local_matrix_kernel(
    Xr,
    Xi,
    Yr,
    Yi,
    Mr,
    Mi,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
):
    """M2: register-resident local micro-FFT via a dense fp32 (BLOCK x BLOCK) matrix.

    One program owns ``ROWS`` consecutive tile-blocks. The tile-local stage
    group (first ``log2(BLOCK)`` DIT stages, or last ``log2(BLOCK)`` DIF
    stages) is a linear map ``out = M @ in`` whose coefficients depend only on
    ``BLOCK`` -- the butterfly twiddle ``exp(-2j*pi*j/(2n))`` cancels ``N``
    exactly -- so the host precomputes it once and the group crosses device
    memory exactly once (one load, one store) with all stage intermediates
    held in registers.

    ``tl.trans``/``tl.split``/exact ``tl.reshape`` crash the XPU backend, so
    the matrix ships pre-transposed (planar ``Mr``/``Mi`` with
    ``out_row = in_row @ (Mr + 1j*Mi)``) and the complex product is four
    ``tl.dot(..., input_precision="ieee")`` calls. The default dot precision
    is bf16-like on this stack and would corrupt the FFT; ``"ieee"`` is fp32
    end-to-end.

    Loads/stores are strictly natural-order: combining a bit-reversal gather
    with ``tl.dot`` in one kernel trips an MLIR assertion in the XPU backend
    (``getOpResultImpl`` result-number-out-of-range), so callers run the
    proven ``bitrev_kernel`` as its own pass around this kernel.
    """
    pid = tl.program_id(0)
    row = tl.arange(0, ROWS)[:, None]
    col = tl.arange(0, BLOCK)[None, :]
    off = (pid * ROWS + row) * BLOCK + col
    ar = tl.load(Xr + off)
    ai = tl.load(Xi + off)
    moff = tl.arange(0, BLOCK)[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
    mr = tl.load(Mr + moff)
    mi = tl.load(Mi + moff)
    orr = tl.dot(ar, mr, input_precision="ieee") - tl.dot(ai, mi, input_precision="ieee")
    oi = tl.dot(ar, mi, input_precision="ieee") + tl.dot(ai, mr, input_precision="ieee")
    tl.store(Yr + off, orr)
    tl.store(Yi + off, oi)


@triton.jit
def tail_group_kernel(
    Xr,
    Xi,
    Yr,
    Yi,
    Cr,
    Ci,
    N,
    N0: tl.constexpr,
    NDEP: tl.constexpr,
    BLOCK: tl.constexpr,
    CELLS: tl.constexpr,
):
    """M3: one execution group of ``log2(NDEP)`` consecutive tail stages.

    A group with half-spans ``n0, 2*n0, ..., 2^(g-1)*n0`` is a single
    block-closed linear map over the ``P = N0 << log2(NDEP)`` period:
    ``out[i] = sum_k T[k, i mod P] * in[i ^ (k*N0)]`` (DIT applies the stages
    ascending, DIF the same set descending -- already baked into ``T``). Only
    group boundaries cross global memory, so the tail's ``G`` and GM
    transactions collapse to one pass per group instead of one per stage.

    The host precomputes ``T`` once per ``(N0, g, direction)`` -- like the M2
    local matrix, the group twiddle ``exp(-2j*pi*j/(2n))`` does not depend on
    ``N`` -- and ships it planar ``[k, i_local]`` so each unrolled iteration
    reads one contiguous coalesced row. ``BLOCK`` may span several periods
    (``BLOCK % P == 0``, enforced by the caller); rows are indexed by
    ``i mod P`` and the gather is pointer arithmetic only (``k*N0 < P`` keeps
    ``i ^ (k*N0)`` inside the program's window). Combining a gather with
    the dense matrix-multiply builtin trips an XPU MLIR assertion, so this
    kernel stays on the pointer form and the bit-reversal stays its own
    pass, exactly like :func:`local_matrix_kernel`.

    The ``k`` loop must be ``tl.static_range`` (compile-time unrolled): the
    vendor ``TritonXPUUnrollControl`` pass exits the process with status 1
    (null ``op2`` in ``isAncestorOf``, UnrollControl.cpp:79) when a
    ``range()`` loop coexists with ``BLOCK >= 256``, so a plain loop would
    hard-crash the interpreter at compile time on wide blocks.

    Program ``pid`` owns ``CELLS`` consecutive ``BLOCK`` windows; ``i < N``
    masks the tail of the array.
    """
    pid = tl.program_id(0)
    period = N0 * NDEP
    for c in range(CELLS):
        loc = tl.arange(0, BLOCK)
        i = (pid * CELLS + c) * BLOCK + loc
        mask = i < N
        lop = i & (period - 1)
        cr = tl.load(Cr + lop)
        ci = tl.load(Ci + lop)
        xr = tl.load(Xr + i, mask=mask)
        xi = tl.load(Xi + i, mask=mask)
        acc_r = xr * cr - xi * ci
        acc_i = xr * ci + xi * cr
        for k in tl.static_range(1, NDEP):
            cr = tl.load(Cr + k * period + lop)
            ci = tl.load(Ci + k * period + lop)
            src = i ^ (k * N0)
            xr = tl.load(Xr + src, mask=mask)
            xi = tl.load(Xi + src, mask=mask)
            acc_r = acc_r + (xr * cr - xi * ci)
            acc_i = acc_i + (xr * ci + xi * cr)
        tl.store(Yr + i, acc_r, mask=mask)
        tl.store(Yi + i, acc_i, mask=mask)


@triton.jit
def dit_stage_batch_kernel(
    Xr,
    Xi,
    Yr,
    Yi,
    Wr,
    Wi,
    n,
    M,
    N,
    B,
    LOGN: tl.constexpr,
    BLOCK: tl.constexpr,
    BITREV_INPUT: tl.constexpr,
    SPREAD: tl.constexpr,
):
    """M4: one radix-2 DIT stage over a (B, N) batch, two data-space layouts.

    Stage semantics are exactly ``dit_stage_kernel`` (reads ``k`` in
    [0, N/2), butterfly half-span ``n``, twiddle ``W[j*M]``, optional
    bit-reversed input on the first stage); every address is rebased to
    ``base + per-batch index``, so batch items never mix.

    ``SPREAD=True`` (data-space expansion, ``Ud=B``): grid is
    ``(cdiv(N/2, BLOCK), B)`` -- batch rides grid axis 1, ``base = pid1 * N``,
    and each program sees one contiguous partition of a single transform.

    ``SPREAD=False`` (data-time expansion, ``Td=B``): one flat 1D grid over the
    pair space ``[0, B*N/2)``; ``b = off >> (LOGN-1)`` splits the batch out of
    the linear index and ``base = b*N`` re-offsets every address. Batch items
    are folded into the same program stream instead of a new grid axis, which
    keeps the launch geometry identical to the unbatched flow up to a factor.
    """
    half = N >> 1
    if SPREAD:
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        base = tl.program_id(1) * N
        mask = off < half
    else:
        off_all = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        base = (off_all >> (LOGN - 1)) * N
        off = off_all & (half - 1)
        mask = off_all < (half * B)

    j = off % n
    i = ((off // n) << 1) * n + j

    if BITREV_INPUT:
        src = base + _bit_reverse(i, LOGN)
        src2 = base + _bit_reverse(i + n, LOGN)
    else:
        src = base + i
        src2 = base + i + n

    w = j * M
    wr = tl.load(Wr + w, mask=mask)
    wi = tl.load(Wi + w, mask=mask)
    ar = tl.load(Xr + src, mask=mask)
    ai = tl.load(Xi + src, mask=mask)
    br = tl.load(Xr + src2, mask=mask)
    bi = tl.load(Xi + src2, mask=mask)
    brw = br * wr - bi * wi
    biw = br * wi + bi * wr
    tl.store(Yr + base + i, ar + brw, mask=mask)
    tl.store(Yi + base + i, ai + biw, mask=mask)
    tl.store(Yr + base + i + n, ar - brw, mask=mask)
    tl.store(Yi + base + i + n, ai - biw, mask=mask)


@triton.jit
def dif_stage_batch_kernel(
    Xr,
    Xi,
    Yr,
    Yi,
    Wr,
    Wi,
    n,
    M,
    N,
    B,
    LOGN: tl.constexpr,
    BLOCK: tl.constexpr,
    BITREV_OUTPUT: tl.constexpr,
    SPREAD: tl.constexpr,
):
    """M4: one radix-2 DIF stage over a (B, N) batch (see
    :func:`dit_stage_batch_kernel` for the SPREAD layout contract).

    Same index mapping as :func:`dif_stage_kernel`; the final stage optionally
    writes through the per-batch bit-reversal so the result is natural order.
    """
    half = N >> 1
    if SPREAD:
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        base = tl.program_id(1) * N
        mask = off < half
    else:
        off_all = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        base = (off_all >> (LOGN - 1)) * N
        off = off_all & (half - 1)
        mask = off_all < (half * B)

    j = off % n
    i = ((off // n) << 1) * n + j
    w = j * M
    wr = tl.load(Wr + w, mask=mask)
    wi = tl.load(Wi + w, mask=mask)
    ar = tl.load(Xr + base + i, mask=mask)
    ai = tl.load(Xi + base + i, mask=mask)
    br = tl.load(Xr + base + i + n, mask=mask)
    bi = tl.load(Xi + base + i + n, mask=mask)
    sr = ar + br
    si = ai + bi
    dr = ar - br
    di = ai - bi
    drw = dr * wr - di * wi
    diw = dr * wi + di * wr

    if BITREV_OUTPUT:
        d0 = base + _bit_reverse(i, LOGN)
        d1 = base + _bit_reverse(i + n, LOGN)
        tl.store(Yr + d0, sr, mask=mask)
        tl.store(Yi + d0, si, mask=mask)
        tl.store(Yr + d1, drw, mask=mask)
        tl.store(Yi + d1, diw, mask=mask)
    else:
        tl.store(Yr + base + i, sr, mask=mask)
        tl.store(Yi + base + i, si, mask=mask)
        tl.store(Yr + base + i + n, drw, mask=mask)
        tl.store(Yi + base + i + n, diw, mask=mask)
