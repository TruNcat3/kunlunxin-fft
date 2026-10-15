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
def _dit_digit_reverse(x, BITS: tl.constexpr):
    """DIT input permutation: mixed-radix digit reversal (M5).

    Radix-4 digits low-to-high (2-bit each); the top digit is 1-bit when
    ``BITS`` is odd (radices ``[4]*m`` or ``[4]*m + [2]``). Digit ``q`` keeps
    weight ``2**(BITS - 2q - 2)``, i.e. the product of the radices above it;
    an odd top bit lands in bit 0.
    """
    rev = tl.zeros_like(x)
    for q in range(BITS // 2):
        d = (x >> (2 * q)) & 3
        rev = rev | (d << (BITS - 2 * q - 2))
    if BITS % 2 == 1:
        rev = rev | ((x >> (BITS - 1)) & 1)
    return rev


@triton.jit
def _dif_digit_reverse(x, BITS: tl.constexpr):
    """DIF output permutation: mixed-radix digit reversal (M5).

    Same reversal with radices ``[2] + [4]*m`` when ``BITS`` is odd (the
    1-bit digit sits at the bottom of the input index and carries the highest
    output weight); even ``BITS`` matches :func:`_dit_digit_reverse`.
    """
    if BITS % 2 == 1:
        rev = (x & 1) << (BITS - 1)
        for q in range(1, BITS // 2 + 1):
            d = (x >> (2 * q - 1)) & 3
            rev = rev | (d << (BITS - 1 - 2 * q))
        return rev
    else:
        rev = tl.zeros_like(x)
        for q in range(BITS // 2):
            d = (x >> (2 * q)) & 3
            rev = rev | (d << (BITS - 2 * q - 2))
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
    REV_MODE: tl.constexpr,
):
    """One radix-2 DIF butterfly stage on split-complex data.

    ``n`` runs N/2, N/4, ..., 1. Same index mapping as the DIT stage, but the
    twiddle is applied to the difference::

        a = X[i]; b = X[i + n]
        Y[i]     = a + b
        Y[i + n] = (a - b) * W[j * M]

    ``REV_MODE`` selects the output permutation folded into this (possibly
    final) stage: ``0`` none, ``1`` bit-reversal (M0), ``2`` the mixed-radix
    digit reversal the M5 radix-4 cascade leaves behind (odd ``log2(N)``).
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

    if REV_MODE == 1:
        d0 = _bit_reverse(i, BITS)
        d1 = _bit_reverse(i + n, BITS)
        tl.store(Yr + d0, sr, mask=mask)
        tl.store(Yi + d0, si, mask=mask)
        tl.store(Yr + d1, drw, mask=mask)
        tl.store(Yi + d1, diw, mask=mask)
    elif REV_MODE == 2:
        d0 = _dif_digit_reverse(i, BITS)
        d1 = _dif_digit_reverse(i + n, BITS)
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
def dit_stage4_kernel(
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
    DIGREV_INPUT: tl.constexpr,
):
    """One radix-4 DIT stage (M5), span ``4n``, ``n`` in 1, 4, 16, ...

    Index mapping mirrors the radix-2 DIT stage with groups of ``4n``::

        j = off % n
        i = (off // n) * 4n + j
        b_d = X[i + d*n] * W**(d*j*M)          d = 0..3
        Y[i + d*n] = column_d(b_0..b_3)        analysis matrix

    ``M = N / (4n)`` so ``d*j*M < 3N/4``; the ``3*j*M`` twiddle folds above
    ``N/2`` through ``W[N-w] = conj(W[w])`` (``w == N/2`` never occurs for
    power-of-four ``n``). When ``DIGREV_INPUT`` is set the first stage reads
    through the mixed digit-reversed permutation, so output is natural order
    with no trailing pass.
    """
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < (N >> 2)

    j = off % n
    i = ((off // n) * 4) * n + j

    if DIGREV_INPUT:
        src0 = _dit_digit_reverse(i, BITS)
        src1 = _dit_digit_reverse(i + n, BITS)
        src2 = _dit_digit_reverse(i + 2 * n, BITS)
        src3 = _dit_digit_reverse(i + 3 * n, BITS)
    else:
        src0 = i
        src1 = i + n
        src2 = i + 2 * n
        src3 = i + 3 * n

    ar = tl.load(Xr + src0, mask=mask)
    ai = tl.load(Xi + src0, mask=mask)
    br = tl.load(Xr + src1, mask=mask)
    bi = tl.load(Xi + src1, mask=mask)
    cr = tl.load(Xr + src2, mask=mask)
    ci = tl.load(Xi + src2, mask=mask)
    dr = tl.load(Xr + src3, mask=mask)
    di = tl.load(Xi + src3, mask=mask)

    # twiddle b_d = a_d * W**(d*j*M) before the butterfly (DIT)
    jm = j * M
    w2 = jm << 1
    w3 = jm + w2
    big = w3 > (N >> 1)
    k3 = tl.where(big, N - w3, w3)
    wi3s = tl.where(big, -1.0, 1.0)
    w1r = tl.load(Wr + jm, mask=mask)
    w1i = tl.load(Wi + jm, mask=mask)
    w2r = tl.load(Wr + w2, mask=mask)
    w2i = tl.load(Wi + w2, mask=mask)
    w3r = tl.load(Wr + k3, mask=mask)
    w3i = tl.load(Wi + k3, mask=mask) * wi3s

    b1r = br * w1r - bi * w1i
    b1i = br * w1i + bi * w1r
    b2r = cr * w2r - ci * w2i
    b2i = cr * w2i + ci * w2r
    b3r = dr * w3r - di * w3i
    b3i = dr * w3i + di * w3r

    # analysis column DFT (same butterfly as DIF, applied to twiddled inputs)
    y0r = ar + b1r + b2r + b3r
    y0i = ai + b1i + b2i + b3i
    y1r = ar + b1i - b2r - b3i
    y1i = ai - b1r - b2i + b3r
    y2r = ar - b1r + b2r - b3r
    y2i = ai - b1i + b2i - b3i
    y3r = ar - b1i - b2r + b3i
    y3i = ai + b1r - b2i - b3r

    tl.store(Yr + i, y0r, mask=mask)
    tl.store(Yi + i, y0i, mask=mask)
    tl.store(Yr + i + n, y1r, mask=mask)
    tl.store(Yi + i + n, y1i, mask=mask)
    tl.store(Yr + i + 2 * n, y2r, mask=mask)
    tl.store(Yi + i + 2 * n, y2i, mask=mask)
    tl.store(Yr + i + 3 * n, y3r, mask=mask)
    tl.store(Yi + i + 3 * n, y3i, mask=mask)


@triton.jit
def dif_stage4_kernel(
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
    DIGREV_OUTPUT: tl.constexpr,
):
    """One radix-4 DIF stage (M5), span ``4n``.

    Same index mapping as :func:`dit_stage4_kernel`, but the column DFT runs
    first and the twiddle ``W**(c*j*M)`` is applied to output quarter ``c``
    (fold above ``N/2`` via conjugation, as in the DIT stage). With
    ``DIGREV_OUTPUT`` the final stage stores through the mixed digit-reversed
    permutation, leaving the result in natural order.
    """
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    mask = off < (N >> 2)

    j = off % n
    i = ((off // n) * 4) * n + j

    ar = tl.load(Xr + i, mask=mask)
    ai = tl.load(Xi + i, mask=mask)
    br = tl.load(Xr + i + n, mask=mask)
    bi = tl.load(Xi + i + n, mask=mask)
    cr = tl.load(Xr + i + 2 * n, mask=mask)
    ci = tl.load(Xi + i + 2 * n, mask=mask)
    dr = tl.load(Xr + i + 3 * n, mask=mask)
    di = tl.load(Xi + i + 3 * n, mask=mask)

    y0r = ar + br + cr + dr
    y0i = ai + bi + ci + di
    y1r = ar + bi - cr - di
    y1i = ai - br - ci + dr
    y2r = ar - br + cr - dr
    y2i = ai - bi + ci - di
    y3r = ar - bi - cr + di
    y3i = ai + br - ci - dr

    jm = j * M
    w2 = jm << 1
    w3 = jm + w2
    big = w3 > (N >> 1)
    k3 = tl.where(big, N - w3, w3)
    wi3s = tl.where(big, -1.0, 1.0)
    w1r = tl.load(Wr + jm, mask=mask)
    w1i = tl.load(Wi + jm, mask=mask)
    w2r = tl.load(Wr + w2, mask=mask)
    w2i = tl.load(Wi + w2, mask=mask)
    w3r = tl.load(Wr + k3, mask=mask)
    w3i = tl.load(Wi + k3, mask=mask) * wi3s

    t1r = y1r * w1r - y1i * w1i
    t1i = y1r * w1i + y1i * w1r
    t2r = y2r * w2r - y2i * w2i
    t2i = y2r * w2i + y2i * w2r
    t3r = y3r * w3r - y3i * w3i
    t3i = y3r * w3i + y3i * w3r

    if DIGREV_OUTPUT:
        d0 = _dif_digit_reverse(i, BITS)
        d1 = _dif_digit_reverse(i + n, BITS)
        d2 = _dif_digit_reverse(i + 2 * n, BITS)
        d3 = _dif_digit_reverse(i + 3 * n, BITS)
    else:
        d0 = i
        d1 = i + n
        d2 = i + 2 * n
        d3 = i + 3 * n

    tl.store(Yr + d0, y0r, mask=mask)
    tl.store(Yi + d0, y0i, mask=mask)
    tl.store(Yr + d1, t1r, mask=mask)
    tl.store(Yi + d1, t1i, mask=mask)
    tl.store(Yr + d2, t2r, mask=mask)
    tl.store(Yi + d2, t2i, mask=mask)
    tl.store(Yr + d3, t3r, mask=mask)
    tl.store(Yi + d3, t3i, mask=mask)


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
def bitrev_batch_kernel(
    Xr,
    Xi,
    Yr,
    Yi,
    N,
    B,
    BITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """M2/M5 batch: out-of-place bit-reversal over ``(B, N)`` (spread, Ud=B).

    Batch rides grid axis 1 (``base = pid1 * N``); every load/store is rebased
    by the row base while the per-row reversal uses only the unbased index, so
    batch items never mix. The separate pass mirrors the rank-1 M2 constraint:
    a gather cannot share a kernel with the dense dot builtin (XPU MLIR
    assertion; the literal builtin name is kept out of this docstring because
    the vendor ASTSource scan substring-matches kernel source for it).
    """
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    base = tl.program_id(1) * N
    mask = off < N
    src = base + _bit_reverse(off, BITS)
    tl.store(Yr + base + off, tl.load(Xr + src, mask=mask), mask=mask)
    tl.store(Yi + base + off, tl.load(Xi + src, mask=mask), mask=mask)


@triton.jit
def bitrev_interleaved_batch_kernel(
    Xr,
    Xi,
    Y,
    N,
    B,
    BITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Bit-reversal writing the re/im-INTERLEAVED ``(B, 2N)`` packet layout.

    Same gather as :func:`bitrev_batch_kernel`, but the output writes each
    reversed element's real/imag parts as adjacent floats (``Y[2*off]``,
    ``Y[2*off+1]`` per row), which the packet-loop group consumes as one
    contiguous ``2*BLOCK``-wide span. Fusing the layout conversion into the
    reversal keeps the packet flow's pass count identical to the spread
    flow (zero extra passes). Feeds the experimental packet probe only.
    """
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    base = tl.program_id(1) * N
    mask = off < N
    src = base + _bit_reverse(off, BITS)
    re = tl.load(Xr + src, mask=mask)
    im = tl.load(Xi + src, mask=mask)
    ioff = tl.program_id(1) * (2 * N) + off * 2
    tl.store(Y + ioff, re, mask=mask)
    tl.store(Y + ioff + 1, im, mask=mask)


@triton.jit
def local_matrix_batch_kernel(
    Xr,
    Xi,
    Yr,
    Yi,
    Mr,
    Mi,
    N,
    B,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """M2 batch (spread, Ud=B): register-resident local micro-FFT per row.

    Batch-aware version of :func:`local_matrix_kernel`: grid is
    ``(tiles // ROWS, B)``, program ``(p0, p1)`` owns ``ROWS`` consecutive
    tile-blocks of batch row ``p1`` (``off = p1*N + (p0*ROWS+row)*BLOCK+col``).
    The tile matrix is N- and B-independent, so it is loaded unbased and shared
    across batch rows; each row still crosses device memory exactly once (one
    load, one store) with stage intermediates held in registers. ``tl.dot``
    shapes and the four-dot complex product are identical to the rank-1 kernel.
    """
    pid = tl.program_id(0)
    bidx = tl.program_id(1)
    row = tl.arange(0, ROWS)[:, None]
    col = tl.arange(0, BLOCK)[None, :]
    off = bidx * N + (pid * ROWS + row) * BLOCK + col
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
def dit_stage4_batch_kernel(
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
    BITS: tl.constexpr,
    BLOCK: tl.constexpr,
    DIGREV_INPUT: tl.constexpr,
):
    """M5 batch (spread, Ud=B): one radix-4 DIT stage over ``(B, N)``.

    Same butterfly and twiddle fold as :func:`dit_stage4_kernel`; batch rides
    grid axis 1 and every X/Y address is rebased by ``base = pid1 * N``.
    Twiddle loads stay unbased (the table is N-scoped, shared across rows).
    ``DIGREV_INPUT`` folds the mixed digit-reversal into the first stage using
    the unbased per-row index, exactly as the rank-1 kernel does.
    """
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    base = tl.program_id(1) * N
    mask = off < (N >> 2)

    j = off % n
    i = ((off // n) * 4) * n + j

    if DIGREV_INPUT:
        src0 = base + _dit_digit_reverse(i, BITS)
        src1 = base + _dit_digit_reverse(i + n, BITS)
        src2 = base + _dit_digit_reverse(i + 2 * n, BITS)
        src3 = base + _dit_digit_reverse(i + 3 * n, BITS)
    else:
        src0 = base + i
        src1 = base + i + n
        src2 = base + i + 2 * n
        src3 = base + i + 3 * n

    ar = tl.load(Xr + src0, mask=mask)
    ai = tl.load(Xi + src0, mask=mask)
    br = tl.load(Xr + src1, mask=mask)
    bi = tl.load(Xi + src1, mask=mask)
    cr = tl.load(Xr + src2, mask=mask)
    ci = tl.load(Xi + src2, mask=mask)
    dr = tl.load(Xr + src3, mask=mask)
    di = tl.load(Xi + src3, mask=mask)

    jm = j * M
    w2 = jm << 1
    w3 = jm + w2
    big = w3 > (N >> 1)
    k3 = tl.where(big, N - w3, w3)
    wi3s = tl.where(big, -1.0, 1.0)
    w1r = tl.load(Wr + jm, mask=mask)
    w1i = tl.load(Wi + jm, mask=mask)
    w2r = tl.load(Wr + w2, mask=mask)
    w2i = tl.load(Wi + w2, mask=mask)
    w3r = tl.load(Wr + k3, mask=mask)
    w3i = tl.load(Wi + k3, mask=mask) * wi3s

    b1r = br * w1r - bi * w1i
    b1i = br * w1i + bi * w1r
    b2r = cr * w2r - ci * w2i
    b2i = cr * w2i + ci * w2r
    b3r = dr * w3r - di * w3i
    b3i = dr * w3i + di * w3r

    y0r = ar + b1r + b2r + b3r
    y0i = ai + b1i + b2i + b3i
    y1r = ar + b1i - b2r - b3i
    y1i = ai - b1r - b2i + b3r
    y2r = ar - b1r + b2r - b3r
    y2i = ai - b1i + b2i - b3i
    y3r = ar - b1i - b2r + b3i
    y3i = ai + b1r - b2i - b3r

    tl.store(Yr + base + i, y0r, mask=mask)
    tl.store(Yi + base + i, y0i, mask=mask)
    tl.store(Yr + base + i + n, y1r, mask=mask)
    tl.store(Yi + base + i + n, y1i, mask=mask)
    tl.store(Yr + base + i + 2 * n, y2r, mask=mask)
    tl.store(Yi + base + i + 2 * n, y2i, mask=mask)
    tl.store(Yr + base + i + 3 * n, y3r, mask=mask)
    tl.store(Yi + base + i + 3 * n, y3i, mask=mask)


@triton.jit
def dif_stage4_batch_kernel(
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
    BITS: tl.constexpr,
    BLOCK: tl.constexpr,
    DIGREV_OUTPUT: tl.constexpr,
):
    """M5 batch (spread, Ud=B): one radix-4 DIF stage over ``(B, N)``.

    The DIF mirror of :func:`dit_stage4_batch_kernel`: column DFT first, then
    the twiddle ``W**(c*j*M)`` on output quarter ``c``; ``DIGREV_OUTPUT``
    stores the final stage through the mixed digit-reversed permutation
    (unbased per-row index, rebased store).
    """
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    base = tl.program_id(1) * N
    mask = off < (N >> 2)

    j = off % n
    i = ((off // n) * 4) * n + j

    ar = tl.load(Xr + base + i, mask=mask)
    ai = tl.load(Xi + base + i, mask=mask)
    br = tl.load(Xr + base + i + n, mask=mask)
    bi = tl.load(Xi + base + i + n, mask=mask)
    cr = tl.load(Xr + base + i + 2 * n, mask=mask)
    ci = tl.load(Xi + base + i + 2 * n, mask=mask)
    dr = tl.load(Xr + base + i + 3 * n, mask=mask)
    di = tl.load(Xi + base + i + 3 * n, mask=mask)

    y0r = ar + br + cr + dr
    y0i = ai + bi + ci + di
    y1r = ar + bi - cr - di
    y1i = ai - br - ci + dr
    y2r = ar - br + cr - dr
    y2i = ai - bi + ci - di
    y3r = ar - bi - cr + di
    y3i = ai + br - ci - dr

    jm = j * M
    w2 = jm << 1
    w3 = jm + w2
    big = w3 > (N >> 1)
    k3 = tl.where(big, N - w3, w3)
    wi3s = tl.where(big, -1.0, 1.0)
    w1r = tl.load(Wr + jm, mask=mask)
    w1i = tl.load(Wi + jm, mask=mask)
    w2r = tl.load(Wr + w2, mask=mask)
    w2i = tl.load(Wi + w2, mask=mask)
    w3r = tl.load(Wr + k3, mask=mask)
    w3i = tl.load(Wi + k3, mask=mask) * wi3s

    t1r = y1r * w1r - y1i * w1i
    t1i = y1r * w1i + y1i * w1r
    t2r = y2r * w2r - y2i * w2i
    t2i = y2r * w2i + y2i * w2r
    t3r = y3r * w3r - y3i * w3i
    t3i = y3r * w3i + y3i * w3r

    if DIGREV_OUTPUT:
        d0 = base + _dif_digit_reverse(i, BITS)
        d1 = base + _dif_digit_reverse(i + n, BITS)
        d2 = base + _dif_digit_reverse(i + 2 * n, BITS)
        d3 = base + _dif_digit_reverse(i + 3 * n, BITS)
    else:
        d0 = base + i
        d1 = base + i + n
        d2 = base + i + 2 * n
        d3 = base + i + 3 * n

    tl.store(Yr + d0, y0r, mask=mask)
    tl.store(Yi + d0, y0i, mask=mask)
    tl.store(Yr + d1, t1r, mask=mask)
    tl.store(Yi + d1, t1i, mask=mask)
    tl.store(Yr + d2, t2r, mask=mask)
    tl.store(Yi + d2, t2i, mask=mask)
    tl.store(Yr + d3, t3r, mask=mask)
    tl.store(Yi + d3, t3i, mask=mask)



@triton.jit
def dif_stage2_batch_kernel(
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
    BITS: tl.constexpr,
    BLOCK: tl.constexpr,
    DIGREV_OUTPUT: tl.constexpr,
):
    """M5 batch (spread, Ud=B): radix-2 DIF tail stage over ``(B, N)``.

    Odd ``log2(N)`` M5 DIF cascades end on a radix-2 stage that must store
    through the mixed digit-reversed permutation (the ``REV_MODE=2`` of the
    rank-1 :func:`dif_stage_kernel`); the M4 batch kernel only offers plain
    bit-reversal, so this tail is its own kernel. Same butterfly as
    :func:`dif_stage_batch_kernel` (spread addressing only).
    """
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    base = tl.program_id(1) * N
    half = N >> 1
    mask = off < half

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

    if DIGREV_OUTPUT:
        d0 = base + _dif_digit_reverse(i, BITS)
        d1 = base + _dif_digit_reverse(i + n, BITS)
    else:
        d0 = base + i
        d1 = base + i + n
    tl.store(Yr + d0, sr, mask=mask)
    tl.store(Yi + d0, si, mask=mask)
    tl.store(Yr + d1, drw, mask=mask)
    tl.store(Yi + d1, diw, mask=mask)



@triton.jit
def local_matrix_packet_kernel(
    X,
    Yr,
    Yi,
    WR,
    WI,
    N,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    PACKETS: tl.constexpr,
):
    """M2 packet loop (experimental, P2): sequential Td with resident tables.

    Grid is ``(tiles // ROWS, B // PACKETS)``; program ``(p0, p1)`` loops
    over its ``PACKETS`` consecutive batch rows with the packet index as a
    compile-time scalar (``k`` from ``static_range``), so every per-iteration
    offset stays in the affine row form the backend lowers. The fused-pair
    tables ``(2*BLOCK, BLOCK)`` are loaded **once before the loop** and
    reused for each packet: one worker consumes ``PACKETS`` packets while
    the tables stay register-resident -- matrix global-memory traffic drops
    ``PACKETS x`` versus the spread form (one program per packet row), which
    is the data-time expansion (``Td=PACKETS``) claim under test.

    Input rows are re/im interleaved (from
    :func:`bitrev_interleaved_batch_kernel`), so a contiguous
    ``(ROWS, 2*BLOCK)`` load feeds the fused pair-product as **two** dot
    operations per packet. Ceilings probed directly on this backend and
    enforced by the runner: ``PACKETS <= 2`` (programs beyond four dot
    operations silently miscompile -- no error, wrong results), and
    non-affine row offsets in pointer arithmetic (stacked/div/shift/select/
    load-table forms all crash the offset pass), hence the scalar-unroll
    loop. Bit-reversal and the global tail run outside as the proven spread
    kernels (mixed dataflow by design; ``WR``/``WI`` pair-interleave the
    local tile matrices ``[re; -im]``/``[im; re]`` along the K axis).
    """
    pid = tl.program_id(0)
    p1 = tl.program_id(1)
    row = tl.arange(0, ROWS)[:, None]
    col1 = tl.arange(0, BLOCK)[None, :]
    col2 = tl.arange(0, 2 * BLOCK)[None, :]
    moff = tl.arange(0, 2 * BLOCK)[:, None] * BLOCK + tl.arange(0, BLOCK)[None, :]
    wr = tl.load(WR + moff)
    wi = tl.load(WI + moff)
    for k in tl.static_range(PACKETS):
        off_i = (p1 * PACKETS + k) * (2 * N) + (pid * ROWS + row) * (2 * BLOCK) + col2
        in_r = tl.load(X + off_i)
        orr = tl.dot(in_r, wr, input_precision="ieee")
        oi = tl.dot(in_r, wi, input_precision="ieee")
        off_o = (p1 * PACKETS + k) * N + (pid * ROWS + row) * BLOCK + col1
        tl.store(Yr + off_o, orr)
        tl.store(Yi + off_o, oi)


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
