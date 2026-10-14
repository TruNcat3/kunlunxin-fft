"""Host-side construction of the register-resident local FFT matrices (M2).

The first ``log2(TILE)`` block-local DIT stages (or the last ``log2(TILE)``
block-local DIF stages) form a linear map ``out = M @ in`` on a single
TILE-block whose coefficients depend **only on TILE**: the butterfly twiddle
``W_N[j*N/(2n)] = exp(-2j*pi*j/(2n))`` cancels ``N`` exactly.  ``M`` is
therefore precomputed once per ``(tile, direction)`` on the host and shipped to
the device, where the local group becomes one dense ``tl.dot`` (fp32) instead
of ``log2(TILE)`` global-memory round trips.

Row-vector convention used by the kernels: ``out_row = in_row @ MT`` where
``MT = M.T``.  The real/imag parts are split planar ``float32`` (the Triton XPU
backend has no ``complex64`` argument type).
"""

import numpy as np

__all__ = [
    "local_matrix",
    "local_matrix_planar",
    "verify_local_matrix",
    "verify_decomposition",
]


def _log2_exact(n: int) -> int:
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"expected a power of two, got {n}")
    return n.bit_length() - 1


def _twiddle(n_half_span: int, j) -> np.ndarray:
    """exp(-2j*pi*j/(2n)) with n = butterfly half-span; N-independent."""
    return np.exp(-2j * np.pi * np.asarray(j, dtype=np.float64) / (2 * n_half_span))


def _apply_dit_stages(v: np.ndarray, tile: int) -> np.ndarray:
    """Apply the block-local DIT stages n = 1, 2, ..., tile/2 to ``v``."""
    logt = _log2_exact(tile)
    for s in range(logt):
        n = 1 << s
        w = _twiddle(n, np.arange(n))
        # pairs (i, i+n) with i = m*2n + j, j in [0, n)
        blocks = v.reshape(-1, 2 * n)
        lo = blocks[:, :n]
        hi = blocks[:, n:] * w
        v = np.concatenate((lo + hi, lo - hi), axis=1).reshape(tile)
    return v


def _apply_dif_stages(v: np.ndarray, tile: int) -> np.ndarray:
    """Apply the block-local DIF stages n = tile/2, ..., 1 to ``v``."""
    logt = _log2_exact(tile)
    for s in range(logt - 1, -1, -1):
        n = 1 << s
        w = _twiddle(n, np.arange(n))
        blocks = v.reshape(-1, 2 * n)
        lo = blocks[:, :n]
        hi = blocks[:, n:]
        v = np.concatenate((lo + hi, (lo - hi) * w), axis=1).reshape(tile)
    return v


def local_matrix(tile: int, direction: str = "dit") -> np.ndarray:
    """Return complex128 ``M`` with ``out = M @ in`` for the local group.

    ``direction`` is ``"dit"`` (first ``log2(tile)`` DIT stages, applied to the
    bit-reversed input) or ``"dif"`` (last ``log2(tile)`` DIF stages, applied
    before the bit-reversed output).
    """
    _log2_exact(tile)
    apply = _apply_dit_stages if direction == "dit" else _apply_dif_stages
    if direction not in ("dit", "dif"):
        raise ValueError(f"unknown direction {direction!r}")
    m = np.empty((tile, tile), dtype=np.complex128)
    for t in range(tile):
        e = np.zeros(tile, dtype=np.complex128)
        e[t] = 1.0
        m[:, t] = apply(e, tile)
    return m


def local_matrix_planar(tile: int, direction: str = "dit"):
    """``(MT_r, MT_i)`` float32 contiguous with ``out_row = in_row @ MT``.

    The kernel computes ``in_row @ (MT_r + 1j*MT_i)`` via four ``tl.dot``s, so
    the host transposes ``M`` up front (``tl.trans`` crashes the XPU backend).
    """
    mt = local_matrix(tile, direction).T.copy()  # row-vector convention, C-contig
    mr = np.ascontiguousarray(mt.real, dtype=np.float32)
    mi = np.ascontiguousarray(mt.imag, dtype=np.float32)
    return mr, mi


def verify_local_matrix(tile: int, direction: str = "dit", rtol=1e-10, atol=1e-12) -> float:
    """Independent scalar-loop check of ``M``; returns the max abs error.

    Uses explicit per-pair index loops (not the vectorized builder), so a bug
    in the reshape construction cannot cancel itself.
    """
    _log2_exact(tile)
    m = local_matrix(tile, direction)
    err = 0.0
    for t in range(tile):
        v = np.zeros(tile, dtype=np.complex128)
        v[t] = 1.0
        ref = _scalar_stages(v.copy(), tile, direction)
        err = max(err, float(np.abs(m[:, t] - ref).max()))
    if err > atol + rtol:
        raise AssertionError(f"local matrix {direction} tile={tile}: err={err:.3e}")
    return err


def _scalar_stages(v: np.ndarray, tile: int, direction: str) -> np.ndarray:
    logt = _log2_exact(tile)
    span = range(logt) if direction == "dit" else range(logt - 1, -1, -1)
    for s in span:
        n = 1 << s
        out = np.empty_like(v)
        w = np.exp(-2j * np.pi * np.arange(n) / (2 * n))
        for i in range(0, tile, 2 * n):
            for j in range(n):
                a = v[i + j]
                b = v[i + j + n]
                if direction == "dit":
                    out[i + j] = a + b * w[j]
                    out[i + j + n] = a - b * w[j]
                else:
                    out[i + j] = a + b
                    out[i + j + n] = (a - b) * w[j]
        v = out
    return v


def verify_decomposition(n: int, tile: int, direction: str = "dit", rtol=1e-10) -> float:
    """Check ``M`` against a full scalar-loop ``np.fft.fft`` reference.

    For ``tile == n`` the local group is the whole transform:

    * DIT: ``M @ bitrev(x) == fft(x)``
    * DIF: ``bitrev(M @ x) == fft(x)``

    For ``tile < n`` the local group is composed with the remaining global
    stages (also scalar-looped) and must still reproduce ``fft(x)``.  This
    pins the ``N``-independence of the local twiddles against a transform
    whose twiddles do depend on ``N``.
    """
    logn = _log2_exact(n)
    # mirror the runner: sizes smaller than the requested tile degenerate to a
    # fully fused single-block transform (tile == n)
    tile = min(tile, n)
    logt = _log2_exact(tile)
    if n % tile:
        raise ValueError(f"tile={tile} must divide n={n}")
    rng = np.random.default_rng(0xC0FFEE ^ n ^ tile)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    x = x.astype(np.complex128)

    br_idx = np.array([int(f"{i:0{logn}b}"[::-1], 2) for i in range(n)])
    m = local_matrix(tile, direction)

    if direction == "dit":
        # flow: bitrev input -> local group -> global tail; ref: np.fft.fft
        state = _group_apply(x[br_idx], m, tile)
        state = _global_stages_scalar(state, logt, logn, "dit")
        ref = np.fft.fft(x)
    else:
        # flow: global tail (n = N/2 .. tile) -> local group -> bitrev output;
        # scalar reference: all DIF stages (N-dependent twiddles) in one loop.
        state = _group_apply(
            _global_stages_scalar(x, logt, logn, "dif"), m, tile
        )
        ref = _global_stages_scalar(x, 0, logn, "dif")

    err = float(np.abs(state - ref).max())
    scale = float(np.abs(ref).max()) + 1.0
    if err / scale > rtol:
        raise AssertionError(
            f"decomposition {direction} n={n} tile={tile}: err={err:.3e}"
        )
    return err / scale


def _group_apply(state: np.ndarray, m: np.ndarray, tile: int) -> np.ndarray:
    """Apply the local group matrix block-wise over ``state``."""
    out = np.empty_like(state)
    for b in range(0, state.size, tile):
        out[b : b + tile] = m @ state[b : b + tile]
    return out


def _global_stages_scalar(v: np.ndarray, from_log: int, to_log: int, direction: str) -> np.ndarray:
    """Explicit full-array DIT/DIF stages with N-dependent twiddles."""
    n = v.size
    logn = _log2_exact(n)
    span = range(from_log, to_log) if direction == "dit" else range(to_log - 1, from_log - 1, -1)
    for s in span:
        half_span = 1 << s
        stride = n // (half_span << 1)
        out = np.empty_like(v)
        for k in range(n >> 1):
            j = k % half_span
            i = ((k // half_span) << 1) * half_span + j
            w = np.exp(-2j * np.pi * (j * stride) / n)
            a, b = v[i], v[i + half_span]
            if direction == "dit":
                out[i] = a + b * w
                out[i + half_span] = a - b * w
            else:
                out[i] = a + b
                out[i + half_span] = (a - b) * w
        v = out
    return v


def _scalar_full_dif(x: np.ndarray) -> np.ndarray:
    n = x.size
    logn = _log2_exact(n)
    v = _global_stages_scalar(x.copy(), 0, logn, "dif")  # all global stages n=1..N/2
    return v


# --- M3: execution groups over the global (tail) stages ---


def _bitrev_idx(n: int, logn: int) -> np.ndarray:
    return np.array([int(f"{i:0{logn}b}"[::-1], 2) for i in range(n)])


def _apply_stage_matrix(cols: np.ndarray, n: int, direction: str) -> np.ndarray:
    """Apply one full-array stage to every column of ``cols`` (positions on axis 0)."""
    p = cols.shape[0]
    w = _twiddle(n, np.arange(n))[:, None]
    blocks = cols.reshape(p // (2 * n), 2 * n, p)
    lo = blocks[:, :n, :]
    hi = blocks[:, n:, :]
    if direction == "dit":
        out = np.concatenate((lo + hi * w, lo - hi * w), axis=1)
    else:
        out = np.concatenate((lo + hi, (lo - hi) * w), axis=1)
    return out.reshape(p, p)


def _scalar_stage_vec(v: np.ndarray, n: int, direction: str) -> np.ndarray:
    """One stage via explicit per-pair index loops (independent of the builder)."""
    p = v.size
    out = np.empty_like(v)
    for i in range(0, p, 2 * n):
        for j in range(n):
            a = v[i + j]
            b = v[i + j + n]
            w = np.exp(-2j * np.pi * j / (2 * n))
            if direction == "dit":
                out[i + j] = a + b * w
                out[i + j + n] = a - b * w
            else:
                out[i + j] = a + b
                out[i + j + n] = (a - b) * w
    return out


def _group_stage_seq(n0: int, glog: int, direction: str):
    """Stage half-spans of one execution group, in application order."""
    span = [n0 << s for s in range(glog)]
    return span if direction == "dit" else list(reversed(span))


def tail_group_table(n0: int, glog: int, direction: str) -> np.ndarray:
    """Coefficient table for one M3 execution group, layout ``[k, i_local]``.

    A group of ``glog`` consecutive tail stages starting at half-span ``n0``
    (DIT: n0, 2n0, ...,; DIF: reversed) is a linear map on each period-``P``
    block with ``P = n0 << glog`` whose dependency offsets are XOR multiples
    of ``n0``::

        out[i] = sum_k table[k, i mod P] * in[i ^ (k * n0)]

    Like the M2 local matrix the coefficients cancel ``N`` exactly (twiddle
    ``exp(-2j*pi*j/(2n))``), so the table depends only on ``(n0, glog,
    direction)``. The kernel loads one contiguous ``[k]`` row per unrolled
    iteration, which keeps the coefficient stream coalesced.
    """
    if glog < 1:
        raise ValueError(f"glog must be >= 1, got {glog}")
    p = n0 << glog
    ndep = 1 << glog
    cols = np.eye(p, dtype=np.complex128)
    for n in _group_stage_seq(n0, glog, direction):
        cols = _apply_stage_matrix(cols, n, direction)
    idx = np.arange(p)
    tab = np.empty((ndep, p), dtype=np.complex128)
    for k in range(ndep):
        tab[k] = cols[idx, idx ^ (k * n0)]
    return tab


def tail_group_planar(n0: int, glog: int, direction: str):
    """``(Cr, Ci)`` float32 contiguous with the ``[k, i_local]`` layout."""
    tab = tail_group_table(n0, glog, direction)
    return (
        np.ascontiguousarray(tab.real, dtype=np.float32),
        np.ascontiguousarray(tab.imag, dtype=np.float32),
    )


def apply_tail_group_table(v: np.ndarray, n0: int, glog: int, direction: str) -> np.ndarray:
    """Host reference: apply the group table block-wise over ``v``."""
    p = n0 << glog
    tab = tail_group_table(n0, glog, direction)
    ndep = tab.shape[0]
    out = np.zeros_like(v)
    for b in range(0, v.size, p):
        blk = v[b : b + p]
        idx = np.arange(p)
        for k in range(ndep):
            out[b : b + p] += tab[k] * blk[idx ^ (k * n0)]
    return out


def verify_tail_group(n0: int, glog: int, direction: str, rtol: float = 1e-10) -> float:
    """Table vs independent scalar stage loops on random period vectors."""
    p = n0 << glog
    rng = np.random.default_rng(0x5EED ^ n0 ^ (glog << 8) ^ (direction == "dif"))
    err = 0.0
    for _ in range(4):
        v = rng.standard_normal(p) + 1j * rng.standard_normal(p)
        ref = v.copy()
        for n in _group_stage_seq(n0, glog, direction):
            ref = _scalar_stage_vec(ref, n, direction)
        got = apply_tail_group_table(v, n0, glog, direction)
        err = max(err, float(np.abs(got - ref).max()))
    scale = float(np.abs(v).max()) + 1.0
    if err / scale > rtol:
        raise AssertionError(f"tail group {direction} n0={n0} glog={glog}: err={err:.3e}")
    return err / scale


def verify_grouped_decomposition(n: int, tile: int, glog: int, direction: str,
                                 rtol: float = 1e-10) -> float:
    """End-to-end host check for ``n = tile << glog`` (no leftover stages).

    The flow (group + local + bit-reversal edges) must reproduce
    ``np.fft.fft``: this pins the group against a reference whose twiddles are
    N-dependent, exactly like :func:`verify_decomposition` does for M2.
    """
    if n != (tile << glog):
        raise ValueError(f"this check needs n == tile << glog, got n={n} tile={tile} glog={glog}")
    logn = _log2_exact(n)
    rng = np.random.default_rng(0xC0DE ^ n ^ glog)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    br_idx = _bitrev_idx(n, logn)
    m = local_matrix(tile, direction)

    if direction == "dit":
        state = _group_apply(x[br_idx], m, tile)
        state = apply_tail_group_table(state, tile, glog, direction)
        ref = np.fft.fft(x)
    else:
        state = apply_tail_group_table(x, tile, glog, direction)
        state = _group_apply(state, m, tile)
        state = state[br_idx]
        ref = np.fft.fft(x)

    err = float(np.abs(state - ref).max())
    scale = float(np.abs(ref).max()) + 1.0
    if err / scale > rtol:
        raise AssertionError(
            f"grouped decomposition {direction} n={n} tile={tile} glog={glog}: err={err:.3e}"
        )
    return err / scale


def tail_groups(tile: int, logn: int, glog: int, maxp: int):
    """Partition tail stages ``n = tile .. N/2`` into execution groups + singles.

    Returns ``(groups, singles)``: ``groups`` is ``[(n0, g), ...]`` ascending
    (each covers stages ``n0, 2n0, ..., 2^(g-1)*n0`` with period
    ``n0 << g <= maxp``) and ``singles`` is the remaining stage half-spans
    ascending, run one per M0 global-stage kernel. Groups smaller than two
    stages are never worth their period-sized vector shape, so a stage that
    cannot pair up falls through to ``singles``.

    ``maxp`` bounds the period-shaped vector block: the group kernel's block
    is ``P = n0 << g`` lanes, which is only viable for small ``P``.
    """
    if glog < 1:
        raise ValueError(f"glog must be >= 1, got {glog}")
    stages = [1 << s for s in range(_log2_exact(tile), logn)]
    groups = []
    singles = []
    i = 0
    while i < len(stages):
        n0 = stages[i]
        take = 0
        for s in range(min(glog, len(stages) - i), 1, -1):
            if (n0 << s) <= maxp:
                take = s
                break
        if take:
            groups.append((n0, take))
            i += take
        else:
            singles.append(n0)
            i += 1
    return groups, singles
