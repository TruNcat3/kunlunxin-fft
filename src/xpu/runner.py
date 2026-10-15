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
* ``register-resident-matrix`` (M2) — the tile-local stage group is one dense
  fp32 ``tl.dot`` against a precomputed TILE-only matrix, so its stage
  intermediates never leave registers; the remaining stages use the M0 path.
* ``grouped-boundary`` (M3) — the tail stages are partitioned into execution
  groups of consecutive stages; each group is one kernel launch over an
  N-independent coefficient table, so only group boundaries cross global
  memory. Leftover large stages use the M0 path.
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
    bitrev_batch_kernel,
    bitrev_interleaved_batch_kernel,
    bitrev_kernel,
    dif_stage2_batch_kernel,
    dif_stage4_batch_kernel,
    dif_stage4_kernel,
    dif_stage_batch_kernel,
    dif_stage_kernel,
    dit_fused_kernel,
    dit_stage4_batch_kernel,
    dit_stage4_kernel,
    dit_stage_batch_kernel,
    dit_stage_kernel,
    local_matrix_batch_kernel,
    local_matrix_kernel,
    local_matrix_packet_kernel,
    tail_group_kernel,
)
from .matrices import local_matrix_planar, tail_group_planar, tail_groups
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


def _launch(kernel, block: int, grid, *args, **kw):
    """Launch ``kernel`` with ``BLOCK=block``; ``grid`` is an int or a tuple.

    Tuple grids are passed through for the M4 spread variant (batch on grid
    axis 1); int grids stay the 1-D ``(n,)`` form every other flow uses.
    """
    kernel[grid if isinstance(grid, tuple) else (grid,)](*args, BLOCK=block, **kw)


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
                REV_MODE=(1 if s == 0 else 0),
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


_MATRIX_CACHE = {}


def _local_matrices(tile: int, direction: str, device):
    """Planar float32 (MT_r, MT_i) for the tile-local group, cached per device."""
    key = (device, tile, direction)
    m = _MATRIX_CACHE.get(key)
    if m is None:
        mr, mi = local_matrix_planar(tile, direction)
        m = (torch.from_numpy(mr).to(device), torch.from_numpy(mi).to(device))
        _MATRIX_CACHE[key] = m
    return m


def _matrix_group_params(n: int, tile: int, rows: int):
    """Validate the M2 axes and return (eff_tile, rows_eff, grid)."""
    if tile <= 1:
        return None
    log2_exact(tile)  # raises unless tile is a power of two
    if rows <= 1:
        rows = 1
    else:
        log2_exact(rows)  # raises unless rows is a power of two
    eff_tile = min(tile, n)
    tiles = n // eff_tile
    rows_eff = min(rows, tiles)
    grid = tiles // rows_eff
    if grid < 1:
        raise ValueError(f"n={n} tile={tile} rows={rows} leaves no programs")
    return eff_tile, rows_eff, grid


def _fft_matrix_tail(
    cur_r,
    cur_i,
    direction: str,
    fused_log: int,
    n: int,
    logn: int,
    block: int,
    wr,
    wi,
):
    """Remaining stages (``n = eff_tile .. N/2``) on the M0 global-stage path."""
    half = n >> 1
    grid = triton.cdiv(half, block)
    stage = dit_stage_kernel if direction == "dit" else dif_stage_kernel
    if direction == "dit":
        span = range(fused_log, logn)
    else:
        span = range(logn - 1, fused_log - 1, -1)
    for s in span:
        stage_n = 1 << s
        m = n // (stage_n << 1)
        nr = torch.empty_like(cur_r)
        ni = torch.empty_like(cur_i)
        kw = (
            {"BITREV_INPUT": False}
            if direction == "dit"
            else {"REV_MODE": 0}
        )
        _launch(
            stage,
            block,
            grid,
            cur_r,
            cur_i,
            nr,
            ni,
            wr,
            wi,
            stage_n,
            m,
            n,
            logn,
            **kw,
        )
        cur_r, cur_i = nr, ni
    return cur_r, cur_i


def fft_dit_matrix(
    x: torch.Tensor,
    block: int = 1024,
    tile: int = 64,
    rows: int = 8,
    twiddle=None,
) -> torch.Tensor:
    """M2: DIT FFT with the tile-local group resident in registers.

    The ``log2(tile)`` first stages become one dense fp32 ``tl.dot`` against a
    precomputed TILE-only matrix (flow ``register-resident-matrix``): their
    intermediates never touch device memory. Flow: explicit bit-reversal pass,
    register-resident matrix group, then the M0 tail. The bit-reversal cannot
    be folded into the matrix kernel: combining a bit-reversal gather with
    ``tl.dot`` trips an MLIR assertion in the XPU backend, so the proven
    ``bitrev_kernel`` runs as its own pass. Launches:
    ``2 + log2(N) - log2(tile)`` vs M0's ``log2(N)`` and M1's
    ``log2(N) + 1`` (smaller for ``tile >= 4``). ``tile``/``rows`` must be
    powers of two; ``rows`` is how many tile-blocks one program owns
    (matrix-load amortization, semantics neutral).
    """
    validate_input(x, "x", block=block)
    n = x.numel()
    logn = log2_exact(n)
    params = _matrix_group_params(n, tile, rows)
    if params is None:
        return fft_dit(x, block=block, twiddle=twiddle)
    eff_tile, rows_eff, grid = params
    fused_log = log2_exact(eff_tile)

    with xpu_context(x.device.index or 0):
        xr, xi = _as_so(x)
        # 1. explicit bit-reversal: the gather cannot share a kernel with dot
        #    (XPU MLIR assertion), so it stays its own proven pass
        br_r = torch.empty_like(xr)
        br_i = torch.empty_like(xi)
        br_grid = triton.cdiv(n, block)
        _launch(bitrev_kernel, block, br_grid, xr, xi, br_r, br_i, n, logn)

        # 2. register-resident local group (natural in, natural out)
        cur_r = torch.empty_like(xr)
        cur_i = torch.empty_like(xi)
        mr, mi = _local_matrices(eff_tile, "dit", x.device)
        _launch(
            local_matrix_kernel,
            eff_tile,
            grid,
            br_r,
            br_i,
            cur_r,
            cur_i,
            mr,
            mi,
            ROWS=rows_eff,
        )
        if fused_log < logn:
            if twiddle is None:
                twiddle = twiddle_tables(n, x.device)
            validate_twiddle(twiddle, n, x.device)
            wr, wi = twiddle
            cur_r, cur_i = _fft_matrix_tail(
                cur_r, cur_i, "dit", fused_log, n, logn, block, wr, wi
            )
    return _from_so(cur_r, cur_i)


def fft_dif_matrix(
    x: torch.Tensor,
    block: int = 1024,
    tile: int = 64,
    rows: int = 8,
    twiddle=None,
) -> torch.Tensor:
    """M2: DIF FFT with the tile-local group resident in registers.

    The global tail stages run first on the M0 path; then the last
    ``log2(tile)`` stages become the register-resident matrix group; finally
    the bit-reversed output is materialized by its own pass (the gather and
    ``dot`` cannot share a kernel on this backend -- see
    :func:`fft_dit_matrix`). Launches: ``2 + log2(N) - log2(tile)``, same as
    :func:`fft_dit_matrix`.
    """
    validate_input(x, "x", block=block)
    n = x.numel()
    logn = log2_exact(n)
    params = _matrix_group_params(n, tile, rows)
    if params is None:
        return fft_dif(x, block=block, twiddle=twiddle)
    eff_tile, rows_eff, grid = params
    fused_log = log2_exact(eff_tile)

    with xpu_context(x.device.index or 0):
        cur_r, cur_i = _as_so(x)
        if fused_log < logn:
            if twiddle is None:
                twiddle = twiddle_tables(n, x.device)
            validate_twiddle(twiddle, n, x.device)
            wr, wi = twiddle
            cur_r, cur_i = _fft_matrix_tail(
                cur_r, cur_i, "dif", fused_log, n, logn, block, wr, wi
            )
        mr, mi = _local_matrices(eff_tile, "dif", x.device)
        # register-resident local group (natural in, natural out)
        loc_r = torch.empty_like(cur_r)
        loc_i = torch.empty_like(cur_i)
        _launch(
            local_matrix_kernel,
            eff_tile,
            grid,
            cur_r,
            cur_i,
            loc_r,
            loc_i,
            mr,
            mi,
            ROWS=rows_eff,
        )
        # output bit-reversal as its own pass (cannot fuse with dot)
        out_r = torch.empty_like(loc_r)
        out_i = torch.empty_like(loc_i)
        br_grid = triton.cdiv(n, block)
        _launch(bitrev_kernel, block, br_grid, loc_r, loc_i, out_r, out_i, n, logn)
        cur_r, cur_i = out_r, out_i
    return _from_so(cur_r, cur_i)




_GROUP_CACHE = {}


def _tail_group_tables(n0: int, glog: int, direction: str, device):
    """Planar float32 ``(Cr, Ci)`` group coefficient table, cached per device."""
    key = (device, n0, glog, direction)
    t = _GROUP_CACHE.get(key)
    if t is None:
        cr, ci = tail_group_planar(n0, glog, direction)
        t = (torch.from_numpy(cr).to(device), torch.from_numpy(ci).to(device))
        _GROUP_CACHE[key] = t
    return t


def _fft_group_tail(
    cur_r,
    cur_i,
    direction: str,
    groups,
    singles,
    n: int,
    logn: int,
    block: int,
    wr,
    wi,
    cells: int,
    device,
):
    """M3 tail: execution groups on ``tail_group_kernel`` + leftover singles.

    ``groups``/``singles`` come from :func:`xpu.matrices.tail_groups` and are
    ascending in half-span; the units therefore cover every tail stage exactly
    once in M0 order, and DIF reverses the list so the largest half-span runs
    first (same stage order as :func:`fft_dif`). The group kernel's ``BLOCK``
    spans at least one full period so its coefficient rows and XOR-gathered
    inputs stay inside one program window.
    """
    units = [("g", n0, g) for n0, g in groups] + [("s", s, None) for s in singles]
    if direction == "dif":
        units.reverse()
    half = n >> 1
    stage_grid = triton.cdiv(half, block)
    stage = dit_stage_kernel if direction == "dit" else dif_stage_kernel
    for kind, a, b in units:
        nr = torch.empty_like(cur_r)
        ni = torch.empty_like(cur_i)
        if kind == "g":
            period = a << b
            cr, ci = _tail_group_tables(a, b, direction, device)
            blk = block if period <= block else period
            if blk > n:
                blk = n
            windows = triton.cdiv(n, blk)
            grid = triton.cdiv(windows, cells)
            _launch(
                tail_group_kernel,
                blk,
                grid,
                cur_r,
                cur_i,
                nr,
                ni,
                cr,
                ci,
                n,
                N0=a,
                NDEP=1 << b,
                CELLS=cells,
            )
        else:
            m = n // (a << 1)
            kw = (
                {"BITREV_INPUT": False}
                if direction == "dit"
                else {"REV_MODE": 0}
            )
            _launch(
                stage,
                block,
                stage_grid,
                cur_r,
                cur_i,
                nr,
                ni,
                wr,
                wi,
                a,
                m,
                n,
                logn,
                **kw,
            )
        cur_r, cur_i = nr, ni
    return cur_r, cur_i


def fft_dit_grouped(
    x: torch.Tensor,
    block: int = 1024,
    tile: int = 64,
    rows: int = 8,
    glog: int = 2,
    maxp: int = 4096,
    cells: int = 1,
    twiddle=None,
) -> torch.Tensor:
    """M3: DIT FFT with tail execution groups (flow ``grouped-boundary``).

    Like :func:`fft_dit_matrix`, the first ``log2(tile)`` stages are the
    register-resident matrix group and the input bit-reversal is its own pass.
    The tail (stages ``tile .. N/2``) is then partitioned by
    :func:`xpu.matrices.tail_groups` into execution groups of ``glog``
    consecutive stages with period ``n0 << glog <= maxp``; each group is one
    :func:`tail_group_kernel` launch whose intermediates stay in registers, so
    only group boundaries cross global memory. Leftover large stages run one
    per launch on the proven M0 path.

    Launches: ``2 + G + S`` (bit-reversal, local matrix, ``G`` groups,
    ``S`` singles) vs M0's ``log2(N)`` and M2's
    ``2 + log2(N) - log2(tile)``. ``tile``/``rows`` are powers of two as in
    :func:`fft_dit_matrix`; ``glog >= 1`` caps stages per group (``glog=1``
    disables grouping), ``maxp`` caps the period-shaped register block, and
    ``cells`` is how many ``BLOCK`` windows one program owns.
    """
    validate_input(x, "x", block=block)
    if glog < 1:
        raise ValueError(f"glog must be >= 1, got {glog}")
    if cells < 1:
        raise ValueError(f"cells must be >= 1, got {cells}")
    n = x.numel()
    logn = log2_exact(n)
    params = _matrix_group_params(n, tile, rows)
    if params is None:
        return fft_dit(x, block=block, twiddle=twiddle)
    eff_tile, rows_eff, grid = params
    fused_log = log2_exact(eff_tile)
    groups, singles = tail_groups(eff_tile, logn, glog, maxp)

    with xpu_context(x.device.index or 0):
        xr, xi = _as_so(x)
        # 1. explicit bit-reversal pass (cannot fuse with the matrix dot)
        br_r = torch.empty_like(xr)
        br_i = torch.empty_like(xi)
        br_grid = triton.cdiv(n, block)
        _launch(bitrev_kernel, block, br_grid, xr, xi, br_r, br_i, n, logn)

        # 2. register-resident local group (natural in, natural out)
        cur_r = torch.empty_like(xr)
        cur_i = torch.empty_like(xi)
        mr, mi = _local_matrices(eff_tile, "dit", x.device)
        _launch(
            local_matrix_kernel,
            eff_tile,
            grid,
            br_r,
            br_i,
            cur_r,
            cur_i,
            mr,
            mi,
            ROWS=rows_eff,
        )

        # 3. execution groups over the tail, singles on the M0 path
        if fused_log < logn:
            wr = wi = None
            if singles:
                if twiddle is None:
                    twiddle = twiddle_tables(n, x.device)
                validate_twiddle(twiddle, n, x.device)
                wr, wi = twiddle
            cur_r, cur_i = _fft_group_tail(
                cur_r, cur_i, "dit", groups, singles, n, logn, block, wr, wi,
                cells, x.device,
            )
    return _from_so(cur_r, cur_i)


def fft_dif_grouped(
    x: torch.Tensor,
    block: int = 1024,
    tile: int = 64,
    rows: int = 8,
    glog: int = 2,
    maxp: int = 4096,
    cells: int = 1,
    twiddle=None,
) -> torch.Tensor:
    """M3: DIF FFT with tail execution groups (flow ``grouped-boundary``).

    The DIF mirror of :func:`fft_dit_grouped`: tail execution groups run
    first (largest half-span down), then the register-resident local matrix
    group, then the output bit-reversal as its own pass. Launches:
    ``2 + G + S``, same accounting as the DIT entry point.
    """
    validate_input(x, "x", block=block)
    if glog < 1:
        raise ValueError(f"glog must be >= 1, got {glog}")
    if cells < 1:
        raise ValueError(f"cells must be >= 1, got {cells}")
    n = x.numel()
    logn = log2_exact(n)
    params = _matrix_group_params(n, tile, rows)
    if params is None:
        return fft_dif(x, block=block, twiddle=twiddle)
    eff_tile, rows_eff, grid = params
    fused_log = log2_exact(eff_tile)
    groups, singles = tail_groups(eff_tile, logn, glog, maxp)

    with xpu_context(x.device.index or 0):
        cur_r, cur_i = _as_so(x)
        # 1. execution groups over the tail (DIF: descending), singles on M0
        if fused_log < logn:
            wr = wi = None
            if singles:
                if twiddle is None:
                    twiddle = twiddle_tables(n, x.device)
                validate_twiddle(twiddle, n, x.device)
                wr, wi = twiddle
            cur_r, cur_i = _fft_group_tail(
                cur_r, cur_i, "dif", groups, singles, n, logn, block, wr, wi,
                cells, x.device,
            )
        # 2. register-resident local group
        mr, mi = _local_matrices(eff_tile, "dif", x.device)
        loc_r = torch.empty_like(cur_r)
        loc_i = torch.empty_like(cur_i)
        _launch(
            local_matrix_kernel,
            eff_tile,
            grid,
            cur_r,
            cur_i,
            loc_r,
            loc_i,
            mr,
            mi,
            ROWS=rows_eff,
        )
        # 3. output bit-reversal as its own pass
        out_r = torch.empty_like(loc_r)
        out_i = torch.empty_like(loc_i)
        br_grid = triton.cdiv(n, block)
        _launch(bitrev_kernel, block, br_grid, loc_r, loc_i, out_r, out_i, n, logn)
        cur_r, cur_i = out_r, out_i
    return _from_so(cur_r, cur_i)




def _validate_batch_input(x: torch.Tensor, block: int, variant: str):
    """M4 input contract: contiguous (B, N) complex64 on the XPU, pow2 axes."""
    if variant not in ("spread", "folded"):
        raise ValueError(f"variant must be 'spread' or 'folded', got {variant!r}")
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
    if x.dtype != torch.complex64:
        raise TypeError(f"x must be complex64, got {x.dtype}")
    if x.device.type != "cuda":
        raise ValueError(f"x must be on the XPU device, got {x.device}")
    if x.dim() != 2:
        raise ValueError(f"x must be rank-2 (batch, length), got dim={x.dim()}")
    if not x.is_contiguous():
        raise ValueError("x must be contiguous")
    if block <= 0 or (block & (block - 1)) != 0:
        raise ValueError(f"block must be a power of two, got {block}")
    b, n = x.shape
    if b < 1:
        raise ValueError(f"batch must be >= 1, got {b}")
    log2_exact(b)
    log2_exact(n)
    return b, n


def fft_dit_batch(
    x: torch.Tensor,
    block: int = 1024,
    variant: str = "spread",
    twiddle=None,
) -> torch.Tensor:
    """M4: batched DIT FFT over ``x`` of shape ``(B, N)`` (flow
    ``batch-spread`` or ``batch-folded``).

    One launch per stage -- the same ``log2(N)`` launches as the M0 baseline --
    with batch items mapped by ``variant``:

    * ``spread`` (data-space expansion, ``Ud=B``): grid
      ``(cdiv(N/2, block), B)``, batch on grid axis 1.
    * ``folded`` (data-time expansion, ``Td=B``): one flat grid over the pair
      space ``B*N/2``, batch split out of the linear program index, so the
      launch geometry is baseline-shaped up to a factor of ``B``.

    ``B=1`` degenerates to the M0 baseline in both variants. The batch axis is
    a workload axis (plan section 6/7), not an assumed speedup.
    """
    b, n = _validate_batch_input(x, block, variant)
    logn = log2_exact(n)
    if twiddle is None:
        twiddle = twiddle_tables(n, x.device)
    validate_twiddle(twiddle, n, x.device)
    wr, wi = twiddle
    half = n >> 1
    if variant == "spread":
        grid = (triton.cdiv(half, block), b)
    else:
        grid = triton.cdiv(half * b, block)
    spread = variant == "spread"

    with xpu_context(x.device.index or 0):
        cr, ci = _as_so(x)
        for s in range(logn):
            stage_n = 1 << s
            M = n // (stage_n << 1)
            nr = torch.empty_like(cr)
            ni = torch.empty_like(ci)
            _launch(
                dit_stage_batch_kernel,
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
                b,
                LOGN=logn,
                BITREV_INPUT=(s == 0),
                SPREAD=spread,
            )
            cr, ci = nr, ni
    return _from_so(cr, ci)


def fft_dif_batch(
    x: torch.Tensor,
    block: int = 1024,
    variant: str = "spread",
    twiddle=None,
) -> torch.Tensor:
    """M4: batched DIF FFT over ``x`` of shape ``(B, N)``.

    The DIF mirror of :func:`fft_dit_batch`: stages run N/2 down to 1 with the
    output bit-reversal folded into the final stage, so again exactly
    ``log2(N)`` launches. See :func:`fft_dit_batch` for the ``variant``
    contract (``spread`` = Ud data-space, ``folded`` = Td data-time).
    """
    b, n = _validate_batch_input(x, block, variant)
    logn = log2_exact(n)
    if twiddle is None:
        twiddle = twiddle_tables(n, x.device)
    validate_twiddle(twiddle, n, x.device)
    wr, wi = twiddle
    half = n >> 1
    if variant == "spread":
        grid = (triton.cdiv(half, block), b)
    else:
        grid = triton.cdiv(half * b, block)
    spread = variant == "spread"

    with xpu_context(x.device.index or 0):
        cr, ci = _as_so(x)
        for s in range(logn - 1, -1, -1):
            stage_n = 1 << s
            M = n // (stage_n << 1)
            nr = torch.empty_like(cr)
            ni = torch.empty_like(ci)
            _launch(
                dif_stage_batch_kernel,
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
                b,
                LOGN=logn,
                BITREV_OUTPUT=(s == 0),
                SPREAD=spread,
            )
            cr, ci = nr, ni
    return _from_so(cr, ci)


def _fft_matrix_batch_tail(
    cur_r,
    cur_i,
    direction: str,
    fused_log: int,
    n: int,
    logn: int,
    block: int,
    b: int,
    wr,
    wi,
):
    """M2 batch tail: remaining global stages on the M4 batch-stage path.

    The post-group stages are the same radix-2 global stages M0/M4 run; the
    batch-aware stage kernels consume them with ``SPREAD=True`` (batch on
    grid axis 1), one launch per stage, no fusion across the group boundary.
    """
    half = n >> 1
    grid = (triton.cdiv(half, block), b)
    stage = dit_stage_batch_kernel if direction == "dit" else dif_stage_batch_kernel
    if direction == "dit":
        span = range(fused_log, logn)
    else:
        span = range(logn - 1, fused_log - 1, -1)
    for s in span:
        stage_n = 1 << s
        m = n // (stage_n << 1)
        nr = torch.empty_like(cur_r)
        ni = torch.empty_like(cur_i)
        kw = (
            {"BITREV_INPUT": False}
            if direction == "dit"
            else {"BITREV_OUTPUT": False}
        )
        _launch(
            stage,
            block,
            grid,
            cur_r,
            cur_i,
            nr,
            ni,
            wr,
            wi,
            stage_n,
            m,
            n,
            b,
            LOGN=logn,
            SPREAD=True,
            **kw,
        )
        cur_r, cur_i = nr, ni
    return cur_r, cur_i


def fft_dit_matrix_batch(
    x: torch.Tensor,
    block: int = 1024,
    tile: int = 64,
    rows: int = 8,
    variant: str = "spread",
    twiddle=None,
) -> torch.Tensor:
    """M2 batch: DIT FFT over ``(B, N)`` with the tile-local group in registers.

    Batch-aware M2 (issue #3 P0.1): the batch axis enters the address
    computation of every launch, not a host loop over rows. ``variant`` is the
    data-space/data-time expansion knob; this round lowers ``spread`` only
    (``Ud=B``, batch rides grid axis 1 -- each program owns ``ROWS`` tile-blocks
    of one batch row, one matrix load amortized over those tiles, one GM
    crossing per row per stage group). ``folded`` (``Td=B``, a per-worker
    packet loop over multiple rows) is **not-lowered** here: index flattening
    without worker residency would misattribute Td; the bounded packet-loop
    probe lives in :func:`fft_dit_matrix_packet_batch`.

    Flow: batched bit-reversal pass (gather cannot share a kernel with
    ``tl.dot`` on this backend), batched register-resident matrix group, then
    the batched M0 tail. Launches: ``2 + log2(N) - log2(tile)`` regardless of
    ``B``. ``B=1`` degenerates to :func:`fft_dit_matrix`.
    """
    b, n = _validate_batch_input(x, block, variant)
    if variant != "spread":
        raise ValueError(
            f"fft_dit_matrix_batch lowers variant='spread' (Ud=B) only this "
            f"round; folded (Td) would flatten indices without worker "
            f"residency. Got variant={variant!r}")
    logn = log2_exact(n)
    params = _matrix_group_params(n, tile, rows)
    if params is None:
        return fft_dit_batch(x, block=block, variant=variant, twiddle=twiddle)
    eff_tile, rows_eff, grid = params
    fused_log = log2_exact(eff_tile)

    with xpu_context(x.device.index or 0):
        xr, xi = _as_so(x)
        br_r = torch.empty_like(xr)
        br_i = torch.empty_like(xi)
        _launch(
            bitrev_batch_kernel,
            block,
            (triton.cdiv(n, block), b),
            xr,
            xi,
            br_r,
            br_i,
            n,
            b,
            BITS=logn,
        )
        cur_r = torch.empty_like(xr)
        cur_i = torch.empty_like(xi)
        mr, mi = _local_matrices(eff_tile, "dit", x.device)
        _launch(
            local_matrix_batch_kernel,
            eff_tile,
            (grid, b),
            br_r,
            br_i,
            cur_r,
            cur_i,
            mr,
            mi,
            n,
            b,
            ROWS=rows_eff,
        )
        if fused_log < logn:
            if twiddle is None:
                twiddle = twiddle_tables(n, x.device)
            validate_twiddle(twiddle, n, x.device)
            wr, wi = twiddle
            cur_r, cur_i = _fft_matrix_batch_tail(
                cur_r, cur_i, "dit", fused_log, n, logn, block, b, wr, wi
            )
    return _from_so(cur_r, cur_i)


def fft_dif_matrix_batch(
    x: torch.Tensor,
    block: int = 1024,
    tile: int = 64,
    rows: int = 8,
    variant: str = "spread",
    twiddle=None,
) -> torch.Tensor:
    """M2 batch: DIF FFT over ``(B, N)`` with the tile-local group in registers.

    The DIF mirror of :func:`fft_dit_matrix_batch`: batched global tail first,
    then the batched register-resident matrix group, then the batched
    bit-reversal output pass. Same ``spread``-only lowering contract and
    ``2 + log2(N) - log2(tile)`` launches. ``B=1`` degenerates to
    :func:`fft_dif_matrix`.
    """
    b, n = _validate_batch_input(x, block, variant)
    if variant != "spread":
        raise ValueError(
            f"fft_dif_matrix_batch lowers variant='spread' (Ud=B) only this "
            f"round; folded (Td) would flatten indices without worker "
            f"residency. Got variant={variant!r}")
    logn = log2_exact(n)
    params = _matrix_group_params(n, tile, rows)
    if params is None:
        return fft_dif_batch(x, block=block, variant=variant, twiddle=twiddle)
    eff_tile, rows_eff, grid = params
    fused_log = log2_exact(eff_tile)

    with xpu_context(x.device.index or 0):
        cur_r, cur_i = _as_so(x)
        if fused_log < logn:
            if twiddle is None:
                twiddle = twiddle_tables(n, x.device)
            validate_twiddle(twiddle, n, x.device)
            wr, wi = twiddle
            cur_r, cur_i = _fft_matrix_batch_tail(
                cur_r, cur_i, "dif", fused_log, n, logn, block, b, wr, wi
            )
        mr, mi = _local_matrices(eff_tile, "dif", x.device)
        loc_r = torch.empty_like(cur_r)
        loc_i = torch.empty_like(cur_i)
        _launch(
            local_matrix_batch_kernel,
            eff_tile,
            (grid, b),
            cur_r,
            cur_i,
            loc_r,
            loc_i,
            mr,
            mi,
            n,
            b,
            ROWS=rows_eff,
        )
        out_r = torch.empty_like(loc_r)
        out_i = torch.empty_like(loc_i)
        _launch(
            bitrev_batch_kernel,
            block,
            (triton.cdiv(n, block), b),
            loc_r,
            loc_i,
            out_r,
            out_i,
            n,
            b,
            BITS=logn,
        )
        cur_r, cur_i = out_r, out_i
    return _from_so(cur_r, cur_i)


def fft_dit_radix4_batch(
    x: torch.Tensor,
    block: int = 1024,
    variant: str = "spread",
    twiddle=None,
) -> torch.Tensor:
    """M5 batch: radix-4 DIT FFT over ``(B, N)`` (flow ``batch-spread``).

    Batch-aware M5 (issue #3 P0.2): the rank-1 radix-4 cascade with every
    stage kernel addressing the batch axis (``Ud=B``, grid
    ``(cdiv(N/4 or N/2, block), B)``). The optional odd-``log2(N)`` radix-2
    tail reuses the M4 batch stage kernel (``BITREV_INPUT=False``); the mixed
    digit-reversal stays folded into the first radix-4 stage via
    ``DIGREV_INPUT``. ``spread`` only, same contract as
    :func:`fft_dit_matrix_batch`. Launches: ``ceil(log2(N)/2)`` plus the odd
    tail, regardless of ``B``.
    """
    b, n = _validate_batch_input(x, block, variant)
    if variant != "spread":
        raise ValueError(
            f"fft_dit_radix4_batch lowers variant='spread' (Ud=B) only this "
            f"round; folded (Td) is not-lowered for radix-4. "
            f"Got variant={variant!r}")
    logn = log2_exact(n)
    if n < 4:
        raise ValueError(f"radix-4 requires N >= 4, got N={n}")
    if twiddle is None:
        twiddle = twiddle_tables(n, x.device)
    validate_twiddle(twiddle, n, x.device)
    wr, wi = twiddle
    stages = _r4_dit_stages(n)
    grid4 = (triton.cdiv(n >> 2, block), b)
    grid2 = (triton.cdiv(n >> 1, block), b)

    with xpu_context(x.device.index or 0):
        cr, ci = _as_so(x)
        for kind, span in stages:
            nr = torch.empty_like(cr)
            ni = torch.empty_like(ci)
            if kind == "r4":
                stage_n = span >> 2
                _launch(
                    dit_stage4_batch_kernel,
                    block,
                    grid4,
                    cr,
                    ci,
                    nr,
                    ni,
                    wr,
                    wi,
                    stage_n,
                    n // span,
                    n,
                    b,
                    BITS=logn,
                    DIGREV_INPUT=(span == 4),
                )
            else:
                _launch(
                    dit_stage_batch_kernel,
                    block,
                    grid2,
                    cr,
                    ci,
                    nr,
                    ni,
                    wr,
                    wi,
                    n >> 1,
                    1,
                    n,
                    b,
                    LOGN=logn,
                    BITREV_INPUT=False,
                    SPREAD=True,
                )
            cr, ci = nr, ni
    return _from_so(cr, ci)


def fft_dif_radix4_batch(
    x: torch.Tensor,
    block: int = 1024,
    variant: str = "spread",
    twiddle=None,
) -> torch.Tensor:
    """M5 batch: radix-4 DIF FFT over ``(B, N)`` (flow ``batch-spread``).

    The DIF mirror of :func:`fft_dit_radix4_batch`: radix-4 groups
    ``N, N/4, ...`` with the mixed digit-reversal on whichever stage is last
    (radix-4 ``DIGREV_OUTPUT``, or the radix-2 tail
    :func:`dif_stage2_batch_kernel` with ``DIGREV_OUTPUT=True``). Same
    ``spread``-only contract and launch count as the DIT direction.
    """
    b, n = _validate_batch_input(x, block, variant)
    if variant != "spread":
        raise ValueError(
            f"fft_dif_radix4_batch lowers variant='spread' (Ud=B) only this "
            f"round; folded (Td) is not-lowered for radix-4. "
            f"Got variant={variant!r}")
    logn = log2_exact(n)
    if n < 4:
        raise ValueError(f"radix-4 requires N >= 4, got N={n}")
    if twiddle is None:
        twiddle = twiddle_tables(n, x.device)
    validate_twiddle(twiddle, n, x.device)
    wr, wi = twiddle
    stages = _r4_dif_stages(n)
    grid4 = (triton.cdiv(n >> 2, block), b)
    grid2 = (triton.cdiv(n >> 1, block), b)

    with xpu_context(x.device.index or 0):
        cr, ci = _as_so(x)
        for idx, (kind, span) in enumerate(stages):
            last = idx == len(stages) - 1
            nr = torch.empty_like(cr)
            ni = torch.empty_like(ci)
            if kind == "r4":
                stage_n = span >> 2
                _launch(
                    dif_stage4_batch_kernel,
                    block,
                    grid4,
                    cr,
                    ci,
                    nr,
                    ni,
                    wr,
                    wi,
                    stage_n,
                    n // span,
                    n,
                    b,
                    BITS=logn,
                    DIGREV_OUTPUT=last,
                )
            else:
                _launch(
                    dif_stage2_batch_kernel,
                    block,
                    grid2,
                    cr,
                    ci,
                    nr,
                    ni,
                    wr,
                    wi,
                    1,
                    n >> 1,
                    n,
                    b,
                    BITS=logn,
                    DIGREV_OUTPUT=True,
                )
            cr, ci = nr, ni
    return _from_so(cr, ci)


def fft_dit_matrix_packet_batch(
    x: torch.Tensor,
    block: int = 1024,
    tile: int = 64,
    rows: int = 8,
    packets: int = 2,
    twiddle=None,
) -> torch.Tensor:
    """M2 packet loop (experimental, P2 canary): sequential Td on the group.

    Flow: bit-reversal **into the interleaved re/im layout** (fused, zero
    extra passes), then one packet-loop program per
    ``(tile-block-group, packet-group)`` pair. Each program loops over its
    ``packets`` consecutive batch rows with the packet index as a
    compile-time scalar while the fused-pair local tables load **once**
    before the loop and stay register-resident across iterations: the
    worker consumes ``packets`` packets with ``packets x`` less matrix
    global-memory traffic than the spread form -- the data-time expansion
    (``Td=packets``) claim, not a flattened index. Output is plain SOA, so
    the proven spread tail finishes the FFT (mixed dataflow by design: the
    probe isolates the matrix-group residency claim).

    Constraints probed directly on this backend: ``packets <= 2`` (the loop
    spends two dot operations per packet; programs beyond four dot
    operations silently miscompile -- no error, wrong values -- so larger
    values are rejected), ``packets`` must divide ``B``, and only the DIT
    flow is offered (the DIF flow would need an extra interleave pass; not
    probed this round). Experimental: not a selector candidate; used by the
    bounded canary measurement only.
    """
    b, n = _validate_batch_input(x, block, "spread")
    if packets < 1 or (packets & (packets - 1)) != 0:
        raise ValueError(f"packets must be a power of two, got {packets}")
    if packets > 2:
        raise ValueError(
            f"packets={packets} exceeds this backend's ceiling of 2: the "
            "packet loop spends two dot operations per packet and programs "
            "beyond four dot operations silently miscompile (probed on "
            "device; no error raised, wrong values returned)"
        )
    if b % packets != 0:
        raise ValueError(f"packets={packets} must divide batch B={b}")
    logn = log2_exact(n)
    params = _matrix_group_params(n, tile, rows)
    if params is None:
        return fft_dit_batch(x, block=block, variant="spread", twiddle=twiddle)
    eff_tile, rows_eff, grid = params
    fused_log = log2_exact(eff_tile)

    with xpu_context(x.device.index or 0):
        xr, xi = _as_so(x)
        ilv = torch.empty(b, 2 * n, dtype=torch.float32, device=x.device)
        _launch(
            bitrev_interleaved_batch_kernel,
            block,
            (triton.cdiv(n, block), b),
            xr,
            xi,
            ilv,
            n,
            b,
            BITS=logn,
        )
        mr, mi = _local_matrices(eff_tile, "dit", x.device)
        wr = torch.empty(2 * eff_tile, eff_tile, dtype=torch.float32, device=x.device)
        wi = torch.empty(2 * eff_tile, eff_tile, dtype=torch.float32, device=x.device)
        wr[0::2] = mr
        wr[1::2] = -mi
        wi[0::2] = mi
        wi[1::2] = mr
        cur_r = torch.empty_like(xr)
        cur_i = torch.empty_like(xi)
        _launch(
            local_matrix_packet_kernel,
            eff_tile,
            (grid, b // packets),
            ilv,
            cur_r,
            cur_i,
            wr,
            wi,
            n,
            ROWS=rows_eff,
            PACKETS=packets,
        )
        if fused_log < logn:
            if twiddle is None:
                twiddle = twiddle_tables(n, x.device)
            validate_twiddle(twiddle, n, x.device)
            wr_t, wi_t = twiddle
            cur_r, cur_i = _fft_matrix_batch_tail(
                cur_r, cur_i, "dit", fused_log, n, logn, block, b, wr_t, wi_t
            )
    return _from_so(cur_r, cur_i)


def _r4_dit_stages(n: int) -> list:
    """M5 DIT stage sequence: radix-4 spans 4, 16, ... plus an optional
    radix-2 tail spanning ``N`` when ``log2(N)`` is odd."""
    stages = []
    s = 4
    while True:
        stages.append(("r4", s))
        if s == n:
            break
        if 4 * s <= n:
            s *= 4
        elif 2 * s == n:
            stages.append(("r2", n))
            break
        else:
            raise AssertionError(f"no next DIT span for N={n}, s={s}")
    return stages


def _r4_dif_stages(n: int) -> list:
    """M5 DIF stage sequence: radix-4 groups ``N, N/4, ...`` down to a
    group of 4, plus a radix-2 tail (group 2) when ``log2(N)`` is odd."""
    stages = []
    s = n
    while s % 4 == 0 and s >= 4:
        stages.append(("r4", s))
        s >>= 2
    if s == 2:
        stages.append(("r2", 2))
    return stages


def fft_dit_radix4(
    x: torch.Tensor,
    block: int = 1024,
    twiddle=None,
) -> torch.Tensor:
    """M5: radix-4 DIT FFT (flow ``baseline-global-stage``, core ``radix-4``).

    Same flow as the M0 baseline -- one global round trip and one launch per
    stage -- but the processing unit is a radix-4 butterfly: ``ceil(log4(N))``
    radix-4 stages (spans 4, 16, ...) plus one radix-2 tail when
    ``log2(N)`` is odd. The mixed digit-reversal input permutation is folded
    into the first stage, so there is no separate pass. Twiddle count per
    transform drops versus radix-2 (3 twiddles per 4 outputs per stage vs 1
    per 2 outputs); see ``src/xpu/candidates.py:arithmetic_cost`` for the
    analytic comparison.
    """
    validate_input(x, "x", block=block)
    n = x.numel()
    logn = log2_exact(n)
    if n < 4:
        raise ValueError(f"radix-4 requires N >= 4, got N={n}")
    if twiddle is None:
        twiddle = twiddle_tables(n, x.device)
    validate_twiddle(twiddle, n, x.device)
    wr, wi = twiddle
    stages = _r4_dit_stages(n)
    grid4 = triton.cdiv(n >> 2, block)
    grid2 = triton.cdiv(n >> 1, block)

    with xpu_context(x.device.index or 0):
        cr, ci = _as_so(x)
        for kind, span in stages:
            nr = torch.empty_like(cr)
            ni = torch.empty_like(ci)
            if kind == "r4":
                stage_n = span >> 2
                _launch(
                    dit_stage4_kernel,
                    block,
                    grid4,
                    cr,
                    ci,
                    nr,
                    ni,
                    wr,
                    wi,
                    stage_n,
                    n // span,
                    n,
                    logn,
                    DIGREV_INPUT=(span == 4),
                )
            else:
                _launch(
                    dit_stage_kernel,
                    block,
                    grid2,
                    cr,
                    ci,
                    nr,
                    ni,
                    wr,
                    wi,
                    n >> 1,
                    1,
                    n,
                    logn,
                    BITREV_INPUT=False,
                )
            cr, ci = nr, ni
    return _from_so(cr, ci)


def fft_dif_radix4(
    x: torch.Tensor,
    block: int = 1024,
    twiddle=None,
) -> torch.Tensor:
    """M5: radix-4 DIF FFT (flow ``baseline-global-stage``, core ``radix-4``).

    The DIF mirror of :func:`fft_dit_radix4`: radix-4 groups ``N, N/4, ...``
    with the mixed digit-reversal output permutation folded into whichever
    stage is last (radix-4 with ``DIGREV_OUTPUT``, or the radix-2 tail with
    ``REV_MODE=2``). Launch count matches the DIT sequence, so DIT and DIF
    differ only in stage direction under the same flow.
    """
    validate_input(x, "x", block=block)
    n = x.numel()
    logn = log2_exact(n)
    if n < 4:
        raise ValueError(f"radix-4 requires N >= 4, got N={n}")
    if twiddle is None:
        twiddle = twiddle_tables(n, x.device)
    validate_twiddle(twiddle, n, x.device)
    wr, wi = twiddle
    stages = _r4_dif_stages(n)
    grid4 = triton.cdiv(n >> 2, block)
    grid2 = triton.cdiv(n >> 1, block)

    with xpu_context(x.device.index or 0):
        cr, ci = _as_so(x)
        for idx, (kind, span) in enumerate(stages):
            last = idx == len(stages) - 1
            nr = torch.empty_like(cr)
            ni = torch.empty_like(ci)
            if kind == "r4":
                stage_n = span >> 2
                _launch(
                    dif_stage4_kernel,
                    block,
                    grid4,
                    cr,
                    ci,
                    nr,
                    ni,
                    wr,
                    wi,
                    stage_n,
                    n // span,
                    n,
                    logn,
                    DIGREV_OUTPUT=last,
                )
            else:
                _launch(
                    dif_stage_kernel,
                    block,
                    grid2,
                    cr,
                    ci,
                    nr,
                    ni,
                    wr,
                    wi,
                    1,
                    n >> 1,
                    n,
                    logn,
                    REV_MODE=2,
                )
            cr, ci = nr, ni
    return _from_so(cr, ci)

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
