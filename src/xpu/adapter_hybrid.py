"""Minimal processing-unit adapter for the four-cell ablation (issue #3 §4-§5).

The review asked whether the flow can be split so an *external* processing
unit computes the local transform while the framework owns decomposition,
twiddles, layout, and boundaries. On this machine the only legal,
host-callable external transform core is the vendor whole-tensor FFT
(``torch.fft.*`` → xdnn ``fft_fft``; see ``docs/benchmarks/external-cores.md``).
It is *full-transform-only* by API, so the adapter exercises it the only
legal way a local core can be: on **framework-owned sub-block tensors**
the framework gathers and scatters itself.

The four cells (A = flow, P = local core):

- ``A0P0`` — current fused lowering (:mod:`src.xpu.runner` m2, triton
  local core in one program family). The production path.
- ``A0P1`` — *abstain by construction*: a fused local program cannot host
  an external whole-tensor core; the core boundary is the whole kernel.
- ``A1P0`` — decomposition flow + triton local stage (same triton stack
  as P0, but unfused; isolates the IO cost of the decomposition).
- ``A1P1`` — decomposition flow + vendor ``torch.fft.fft`` on the same
  sub-block tensors (the external P1 core).

Decomposition (DIT, N = T*R, both powers of two): view x as (T, R),
row-FFT the R axis (the local core), twiddle by
ω_N^{k0·k1} on the (k0, m1) grid, column-FFT the T axis (the local core
again), transpose to natural order. That transpose *is* the mixed-radix
digit reversal (derived: X[k1 + T·k0] = C[k0, k1]); the framework owns
it, so every gather/scatter/transposed-materialize shows up in the
measured execution time — which is exactly what the ablation must
attribute.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

try:
    import torch
    import triton  # noqa: F401  (kernel decorators below)
    import triton.language as tl  # noqa: F401
    HAS_RUNTIME = True
except ImportError:  # host CI / offline inspection: no device stack here
    torch = None
    triton = None
    tl = None
    HAS_RUNTIME = False


def _require_runtime() -> None:
    """Entry-point guard: plan-level APIs are pure Python; execution
    needs torch+triton (issue #3 P0: adapter must import without them)."""
    if not HAS_RUNTIME:
        raise RuntimeError(
            "adapter_hybrid execution requires torch+triton "
            "(plan_hybrid/validation works offline)")


@dataclass
class HybridPlan:
    """Framework-owned decomposition plan; the core is a parameter, not
    the owner of the plan (issue #3 review: resource constraints and
    layout live in plan/lowering, not in the core)."""

    n: int
    batch: int
    tile: int
    direction: str = "dit"
    core: str = "vendor"  # vendor | triton
    stage2_len: int = field(init=False)
    # Plan-owned caches (issue #3 P2: reuse rather than rebuild): the
    # twiddle table, the staging buffers, and the triton small-stage
    # (bit-reversal, twiddle) tables live as long as the plan does.
    _tw: dict = field(default_factory=dict, repr=False)
    _buffers: dict = field(default_factory=dict, repr=False)
    _rows: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.direction != "dit":
            raise ValueError("adapter covers DIT only this round")
        if self.n % self.tile:
            raise ValueError(f"tile {self.tile} must divide N={self.n}")
        if (self.n & (self.n - 1)) or (self.tile & (self.tile - 1)):
            raise ValueError("N and tile must be powers of two")
        if self.core not in ("vendor", "triton"):
            raise ValueError(f"unknown core {self.core!r}")
        self.stage2_len = self.n // self.tile

    def twiddle(self, device, dtype):
        """Cached (t2, tile) twiddle table w_N^{k_t*m0} for this plan."""
        key = (str(device), str(dtype))
        if key not in self._tw:
            k_t = torch.arange(self.stage2_len, device=device,
                               dtype=torch.float64)
            m0 = torch.arange(self.tile, device=device, dtype=torch.float64)
            ang = -2.0 * math.pi * torch.outer(k_t, m0) / self.n
            self._tw[key] = torch.complex(
                torch.cos(ang), torch.sin(ang)).to(dtype).contiguous()
        return self._tw[key]

    def buffer(self, name: str, shape, dtype, device):
        """Cached pre-allocated staging tensor (one per name/shape/dtype)."""
        key = (name, tuple(shape), str(dtype), str(device))
        if key not in self._buffers:
            self._buffers[key] = torch.empty(shape, dtype=dtype,
                                             device=device)
        return self._buffers[key]


def plan_hybrid(n: int, batch: int, tile: int = 64,
                core: str = "vendor") -> HybridPlan:
    return HybridPlan(n=n, batch=batch, tile=tile, core=core)


_ROW_FFT_CACHE: dict = {}

if HAS_RUNTIME:
    @triton.jit
    def _row_fft_kernel(RE, IM, WRE, WIM, ORE, OIM, N,
                        STAGE: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < N
        re = tl.load(RE + row * N + offs, mask=mask, other=0.0)
        im = tl.load(IM + row * N + offs, mask=mask, other=0.0)
        half_s = STAGE // 2
        j = offs % STAGE
        upper = j >= half_s
        j2 = tl.where(upper, j - half_s, j)
        grp = offs // STAGE
        poffs = grp * STAGE + tl.where(upper, j2, j2 + half_s)
        bre = tl.load(RE + row * N + poffs, mask=mask, other=0.0)
        bim = tl.load(IM + row * N + poffs, mask=mask, other=0.0)
        w_idx = j2 * (N // STAGE)          # < N/2 always
        wr = tl.load(WRE + w_idx, mask=mask, other=1.0)
        wi = tl.load(WIM + w_idx, mask=mask, other=0.0)
        # butterfly (a, b) -> (a + w*b, a - w*b), a = lower, b = upper:
        # lower writes re + w*bre; upper writes bre - w*re.
        a_re = tl.where(upper, bre, re)
        a_im = tl.where(upper, bim, im)
        b_re = tl.where(upper, re, bre)
        b_im = tl.where(upper, im, bim)
        tre = b_re * wr - b_im * wi
        tim = b_re * wi + b_im * wr
        ore = tl.where(upper, a_re - tre, a_re + tre)
        oim = tl.where(upper, a_im - tim, a_im + tim)
        tl.store(ORE + row * N + offs, ore, mask=mask)
        tl.store(OIM + row * N + offs, oim, mask=mask)




def _bitrev_index(n: int, device) -> torch.Tensor:
    bits = n.bit_length() - 1
    idx = torch.arange(n, device=device, dtype=torch.int32)
    rev = torch.zeros_like(idx)
    for i in range(bits):
        rev = rev | (((idx >> i) & 1) << (bits - 1 - i))
    return rev


def _row_fft_triton_small(flat: torch.Tensor,
                          cache: dict | None = None) -> torch.Tensor:
    """Unfused P0 local stage for small lengths (R < 64): one triton
    program per row, iterative radix-2 DIT over float re/im planes.

    Bit-reversal is a framework-owned gather (measured as A1 IO) and the
    twiddle table is framework-owned; the kernel only does butterflies.
    Exists because the fused m2 kernel cannot go below tile=32 while the
    decomposition's second stage can be shorter — exactly where an
    unfused flow pays.
    """
    rows, n = flat.shape
    if n & (n - 1) or n > 512:
        raise ValueError(f"small triton stage supports pow2 R<=512, got {n}")

    block = 1 << (n - 1).bit_length()
    ckey = (n, str(flat.device))
    if cache is not None and ckey in cache:
        rev, w_re, w_im = cache[ckey]
    else:
        rev = _bitrev_index(n, flat.device)
        half = n // 2
        k = torch.arange(half, device=flat.device, dtype=torch.float64)
        ang = -2.0 * torch.pi * k / n
        w_re = torch.cos(ang).to(torch.float32).contiguous()
        w_im = torch.sin(ang).to(torch.float32).contiguous()
        if cache is not None:
            cache[ckey] = (rev, w_re, w_im)
    # complex index_select is unimplemented on this backend; permute the
    # float planes instead (still device-side, still framework IO).
    re = flat.real.to(torch.float32).index_select(1, rev).contiguous()
    im = flat.imag.to(torch.float32).index_select(1, rev).contiguous()

    re_b, im_b = re, im
    for stage in (2, 4, 8, 16, 32, 64, 128, 256, 512):
        if stage > n:
            break
        ore = torch.empty_like(re_b)
        oim = torch.empty_like(im_b)
        _row_fft_kernel[(rows,)](
            re_b, im_b, w_re, w_im, ore, oim, n,
            STAGE=stage, BLOCK=block, num_warps=4)
        re_b, im_b = ore, oim
    return torch.complex(re_b, im_b).to(flat.dtype)


def _local_stage_triton_1d(flat: torch.Tensor, length: int,
                           plan: HybridPlan) -> torch.Tensor:
    """P0 core on (rows, length) contiguous blocks: fused m2 kernel for
    length>=64, the unfused triton row kernel for shorter stages."""
    from src.xpu.runner import fft_dit_matrix_batch

    rows, n1 = flat.shape
    assert n1 == length, (n1, length)
    if length >= 64:
        return fft_dit_matrix_batch(flat, block=min(max(length, 256), 1024),
                                    tile=min(64, length), rows=4,
                                    variant="spread")
    return _row_fft_triton_small(flat, cache=plan._rows)


def run_hybrid(x: torch.Tensor, plan: HybridPlan) -> torch.Tensor:
    _require_runtime()

    """Framework decomposition with the plan's local core; DIT natural
    in, natural out. Verified against NumPy FFT to ~1e-16 for the tile
    sizes in the published space.

    Pipeline (N = T*R, m = m1*R + m0 digits):

    1. framework gather ``x.reshape(T, R).T`` -> (m0, m1)   [IO]
    2. core: row FFT length T (contiguous)                  [core]
    3. framework transpose -> (k_t, m0)                     [IO]
    4. framework twiddle ``ω_N^{k_t·m0}``                   [framework]
    5. core: row FFT length R (contiguous)                  [core]
    6. framework transpose -> (k_r, k_t) and flatten        [IO]
    """
    n, tile, t2 = plan.n, plan.tile, plan.stage2_len
    b = x.shape[0]

    # x is (B, N); digit view per batch row: (m0, m1) with m = m1*R + m0.
    # Staging tensors and the twiddle table come from the plan cache
    # (issue #3 P2: pre-allocate + reuse; no per-call arange/outer/cos).
    # The gather is ONE copy into the cached buffer: the (b, tile, t2)
    # source is a free view, and blocks.view(b, tile, t2) is the cached
    # memory itself (no intermediate reshape copy).
    src = x.reshape(b, t2, tile).permute(0, 2, 1)    # (b, tile, t2) view
    blocks = plan.buffer("blocks", (b * tile, t2), x.dtype, x.device)
    blocks.view(b, tile, t2).copy_(src)
    # (B*R, T): rows are (m0) sequences over m1 — length-T row FFTs
    stage1 = _core_fft_1d(blocks, t2, plan)          # (B*R, T): (m0, k_t)

    # Fused transpose + twiddle (issue #3 P2: one GM pass, not copy
    # then mul): read the permuted stage-1 view and the cached twiddle,
    # write the cached (b, t2, tile) grid once.
    grid = plan.buffer("grid", (b, t2, tile), x.dtype, x.device)
    torch.mul(stage1.reshape(b, tile, t2).permute(0, 2, 1),
              plan.twiddle(x.device, x.dtype), out=grid)

    # (B*T, R): rows are k_t-sequences over m0 — length-R row FFTs.
    # grid is contiguous, so the reshape is a free view — no flat2 copy.
    stage2 = _core_fft_1d(grid.reshape(b * t2, tile), tile, plan)

    # fresh output tensor every call (reshape of the permuted stage-2
    # view forces a copy): callers never alias the plan's buffers
    out = stage2.reshape(b, t2, tile).permute(0, 2, 1).reshape(b, n)
    # (B, R, T) -> (k_r, k_t) rows -> flatten: X[k_r*T + k_t]
    return out


def plan_ledger(plan: HybridPlan) -> dict:
    """Call/boundary accounting for one plan (issue #3 P2: call ledger).

    Analytical, not measured: per-stage sub-FFT shapes and call counts,
    the framework-owned materialized boundaries with GM-byte estimates
    (complex64 assumed, 8 B/element), and the sync points the timing
    protocol adds. Vendor-core launches inside the torch/XDNN stack are
    opaque; they are counted as one host API call per row batch.
    """
    n, b, tile, t2 = plan.n, plan.batch, plan.tile, plan.stage2_len
    elem = 8  # complex64

    def _launches_per_core_call(length: int) -> int:
        if plan.core == "vendor":
            return 1  # one host API call; backend-internal unknown
        if length >= 64:
            return 1  # fused m2 program
        return max(0, length.bit_length() - 2)  # radix-2 stages, small path

    boundaries = [
        {"op": "gather x.reshape(T,R).T -> blocks (single copy into "
               "plan buffer)",
         "bytes_est": b * n * elem},
        {"op": "fused stage1 transpose + twiddle -> grid (one GM "
               "pass; tw table plan-cached)",
         "bytes_est": b * t2 * tile * elem + t2 * tile * elem},
        {"op": "digit-reversal transpose + flatten -> out",
         "bytes_est": b * n * elem},
    ]
    return {
        "core": plan.core,
        "core_calls": {"stage1": b * tile, "stage1_len": t2,
                       "stage2": b * t2, "stage2_len": tile},
        "core_launches_est": (
            b * tile * _launches_per_core_call(t2)
            + b * t2 * _launches_per_core_call(tile)),
        "materialized_boundaries": boundaries,
        "gm_bytes_framework_est": sum(x["bytes_est"] for x in boundaries),
        "sync_points_per_timed_call": 1,
        "element_bytes_assumed": elem,
        "twiddle_and_row_tables": "plan-cached (HybridPlan fields)",
        "buffers": "plan-cached staging tensors; output is fresh per call",
    }


def _core_fft_1d(flat2d: torch.Tensor, length: int,
                 plan: HybridPlan) -> torch.Tensor:
    """Apply the plan's local core as contiguous row FFTs of `length`."""
    if plan.core == "vendor":
        return torch.fft.fft(flat2d, dim=-1)
    return _local_stage_triton_1d(flat2d, length, plan)


def summarize_samples(samples: list[float]) -> dict:
    """Robust steady-state stats (issue #3 review): p50, MAD, p95, max,
    outlier count. An outlier is a sample > p50 + 3*MAD (or, when
    MAD == 0, > 1.5 * p50). Pure Python, host-testable."""
    if not samples:
        return {"n": 0}
    s = sorted(samples)
    n = len(s)
    p50 = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    abs_dev = sorted(abs(v - p50) for v in s)
    mad = abs_dev[n // 2] if n % 2 else 0.5 * (abs_dev[n // 2 - 1]
                                               + abs_dev[n // 2])
    p95 = s[min(n - 1, int(round(0.95 * (n - 1))))]
    thr = p50 + 3.0 * mad if mad > 0 else 1.5 * p50
    outliers = [v for v in s if v > thr]
    return {
        "n": n,
        "p50": p50,
        "mad": mad,
        "p95": p95,
        "max": s[-1],
        "outlier_count": len(outliers),
        "outlier_values": outliers,
        "outlier_threshold": thr,
    }


def time_cell(fn, x: torch.Tensor, warmup: int, trials: int):
    """Split timing (issue #3 P1-5): (first_ms, stats). `first` is the
    single first call (JIT/compile + plan cache fill); `stats` is
    summarize_samples over `trials` calls after `warmup`, explicit
    synchronize around every timed call."""
    _require_runtime()
    dev = x.device
    t0 = time.perf_counter()
    fn(x)
    torch.cuda.synchronize(dev)
    first_ms = (time.perf_counter() - t0) * 1e3
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize(dev)
    samples = []
    for _ in range(trials):
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        fn(x)
        torch.cuda.synchronize(dev)
        samples.append((time.perf_counter() - t0) * 1e3)
    return first_ms, summarize_samples(samples)


def profile_hybrid(x: torch.Tensor, plan: HybridPlan):
    """Phase-level breakdown of run_hybrid (issue #3 review §3).

    Returns (out, phases) where phases has explicit-synchronize wall
    times for gather, stage1 core, fused transpose+twiddle, stage2
    core, and scatter, plus per-phase materialized-input contiguity
    (hidden-copy audit: a non-contiguous core input would force an
    opaque copy inside the vendor/local API). Diagnostic only — the
    sync between phases makes it slower than the steady path.
    """
    _require_runtime()
    dev = x.device
    n, tile, t2 = plan.n, plan.tile, plan.stage2_len
    b = x.shape[0]
    phases: dict = {"core": plan.core}

    def _lap(t0):
        torch.cuda.synchronize(dev)
        return (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    src = x.reshape(b, t2, tile).permute(0, 2, 1)
    blocks = plan.buffer("blocks", (b * tile, t2), x.dtype, x.device)
    blocks.view(b, tile, t2).copy_(src)
    phases["gather_ms"] = _lap(t0)

    t0 = time.perf_counter()
    stage1 = _core_fft_1d(blocks, t2, plan)
    phases["stage1_core_ms"] = _lap(t0)
    phases["stage1_out_contiguous"] = bool(stage1.is_contiguous())

    t0 = time.perf_counter()
    grid = plan.buffer("grid", (b, t2, tile), x.dtype, x.device)
    torch.mul(stage1.reshape(b, tile, t2).permute(0, 2, 1),
              plan.twiddle(dev, x.dtype), out=grid)
    phases["twiddle_transpose_ms"] = _lap(t0)

    t0 = time.perf_counter()
    flat2_view = grid.reshape(b * t2, tile)
    phases["stage2_in_contiguous"] = bool(flat2_view.is_contiguous())
    stage2 = _core_fft_1d(flat2_view, tile, plan)
    phases["stage2_core_ms"] = _lap(t0)
    phases["stage2_out_contiguous"] = bool(stage2.is_contiguous())

    t0 = time.perf_counter()
    out = stage2.reshape(b, t2, tile).permute(0, 2, 1).reshape(b, n)
    phases["scatter_ms"] = _lap(t0)
    phases["hidden_copies"] = [
        name for name, ok in (
            ("stage1_in", blocks.is_contiguous()),
            ("stage2_in", phases["stage2_in_contiguous"]),
        ) if not ok
    ]
    return out, phases
