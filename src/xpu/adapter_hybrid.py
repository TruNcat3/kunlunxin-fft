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

import torch
import triton  # noqa: F401  (kernel decorators below)
import triton.language as tl  # noqa: F401

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


def plan_hybrid(n: int, batch: int, tile: int = 64,
                core: str = "vendor") -> HybridPlan:
    return HybridPlan(n=n, batch=batch, tile=tile, core=core)


_ROW_FFT_CACHE: dict = {}

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


def _row_fft_triton_small(flat: torch.Tensor) -> torch.Tensor:
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
    rev = _bitrev_index(n, flat.device)
    # complex index_select is unimplemented on this backend; permute the
    # float planes instead (still device-side, still framework IO).
    re = flat.real.to(torch.float32).index_select(1, rev).contiguous()
    im = flat.imag.to(torch.float32).index_select(1, rev).contiguous()

    half = n // 2
    k = torch.arange(half, device=flat.device, dtype=torch.float64)
    ang = -2.0 * torch.pi * k / n
    w_re = torch.cos(ang).to(torch.float32).contiguous()
    w_im = torch.sin(ang).to(torch.float32).contiguous()

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
    return _row_fft_triton_small(flat)


def run_hybrid(x: torch.Tensor, plan: HybridPlan) -> torch.Tensor:
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

    # x is (B, N); digit view per batch row: (m0, m1) with m = m1*R + m0
    blocks = x.reshape(b, t2, tile).permute(0, 2, 1).reshape(
        b * tile, t2).contiguous()
    # (B*R, T): rows are (m0) sequences over m1 — length-T row FFTs
    stage1 = _core_fft_1d(blocks, t2, plan)          # (B*R, T): (m0, k_t)

    grid = stage1.reshape(b, tile, t2).permute(0, 2, 1).contiguous()
    # (B, T, R): (k_t, m0)
    k_t = torch.arange(t2, device=x.device, dtype=torch.float64)
    m0 = torch.arange(tile, device=x.device, dtype=torch.float64)
    ang = -2.0 * math.pi * torch.outer(k_t, m0) / n
    tw = torch.complex(torch.cos(ang), torch.sin(ang)).to(x.dtype)
    grid = grid * tw.unsqueeze(0)

    flat2 = grid.reshape(b * t2, tile).contiguous()  # (B*T, R): (k_t, k_r?)
    # rows are k_t-sequences over m0 — length-R row FFTs -> (k_t, k_r)
    stage2 = _core_fft_1d(flat2, tile, plan)         # (B*T, R)

    out = stage2.reshape(b, t2, tile).permute(0, 2, 1).reshape(b, n)
    # (B, R, T) -> (k_r, k_t) rows -> flatten: X[k_r*T + k_t]
    return out


def _core_fft_1d(flat2d: torch.Tensor, length: int,
                 plan: HybridPlan) -> torch.Tensor:
    """Apply the plan's local core as contiguous row FFTs of `length`."""
    if plan.core == "vendor":
        return torch.fft.fft(flat2d, dim=-1)
    return _local_stage_triton_1d(flat2d, length, plan)


def time_cell(fn, x: torch.Tensor, warmup: int, trials: int):
    """(setup_ms, exec_median_ms): setup = first call incl. JIT/compile;
    exec = median of `trials` timed calls with explicit synchronize."""
    dev = x.device
    t0 = time.perf_counter()
    fn(x)
    torch.cuda.synchronize(dev)
    setup_ms = (time.perf_counter() - t0) * 1e3
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
    samples.sort()
    return setup_ms, samples[len(samples) // 2]
