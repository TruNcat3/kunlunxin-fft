"""Production 4-step hybrid runners (issue #13 WP2, A+P compositions).

The WP1-measured promotion candidates (``hybrid_vendor_p`` /
``hybrid_fused_p`` of the issue #10 holdout) are the 4-step
Cooley-Tukey flow whose local stages run on the **vendor** host-API
FFT over materialized sub-blocks with a framework-owned Triton
twiddle — the ``a1p1_materialized_subtransform`` capability boundary
(``config/external_cores.json``): the vendor core stays
full-transform-only and only ever sees framework-owned sub-block
tensors, while decomposition, twiddle, layout and digit reversal stay
framework-owned (docs/benchmarks/external-cores.md).

This module is the production deployment of those *measured*
implementations: the kernels and orchestration are the same code the
holdout/ablation measured in ``scripts/issue10_wp3_perf.py``
(``k_twiddle`` / ``k_twiddle_fused``, ``_factors`` split, same stage
order); ``tests/test_issue13_hybrid4_hw.py`` asserts output parity
against the benchmark runners so the deployed flow can never silently
diverge from the measured one.

Contracts:

* DIT decomposition only (``direction="dit"``), power-of-two ``N``
  with a legal ``(t, r)`` split (``t >= 8, r >= 8``);
* :func:`run_hybrid4` / :func:`run_hybrid4_fused` take rank-2
  ``(B, N)`` complex64 and return ``(B, N)`` complex64 — the public
  rank-1 ``(N,)`` entry contract is applied by the plan/execute
  adapter (``src/xpu/entry._invoke_hybrid4``), the same zero-copy
  unsqueeze the adapter_hybrid composition path uses;
* ``inverse=True`` builds the IDFT natively (ifft stages +
  conjugated twiddle, exact 1/N backward scaling on complex input) —
  the same call the holdout measured for the inverse cells;
* plan-level caches (twiddle/staging buffer) make plan-reuse
  allocation-free, mirroring the holdout's setup-outside-timing
  structure.

Host-importable without torch/triton (execution raises through
:func:`_require_runtime`); the twiddle kernels lower through the
same ``xpu_context`` as every other production Triton launch.
"""
from __future__ import annotations

from dataclasses import dataclass, field

try:
    import torch
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401
    HAS_RUNTIME = True
except ImportError:  # host CI / offline inspection
    torch = None
    triton = None
    tl = None
    HAS_RUNTIME = False

VARIANTS = ("4step-transpose", "4step-fused")
# variant -> the executed runner name recorded on the plan/entry
VARIANT_EXECUTED = {
    "4step-transpose": "run_hybrid4",
    "4step-fused": "run_hybrid4_fused",
}
EXECUTED_VARIANT = {v: k for k, v in VARIANT_EXECUTED.items()}


def _require_runtime() -> None:
    if not HAS_RUNTIME:
        raise RuntimeError(
            "src.xpu.hybrid4 execution requires torch+triton "
            "(plan_hybrid4/validation works offline)")


if HAS_RUNTIME:
    @triton.jit
    def k_twiddle(X, O, N: tl.constexpr, R: tl.constexpr,
                  T: tl.constexpr, BLOCK: tl.constexpr,
                  CONJ: tl.constexpr):
        """4-step hybrid twiddle: O = X * W_N^{+-j*k} on interleaved
        complex64. Same kernel the issue #10 holdout measured."""
        pid = tl.program_id(0)
        j = pid % R
        b = pid // R
        base = ((b * R + j) * T) * 2
        k = tl.arange(0, BLOCK)
        mask = k < T
        re = tl.load(X + base + k * 2, mask=mask, other=0.0)
        im = tl.load(X + base + k * 2 + 1, mask=mask, other=0.0)
        jf = j.to(tl.float32) + (k * 0.0)
        ang = (6.283185307179586 / N) * jf * k.to(tl.float32)
        if CONJ == 1:
            ang = -ang
        wr = tl.cos(ang)
        wi = -tl.sin(ang)
        tl.store(O + base + k * 2, re * wr - im * wi, mask=mask)
        tl.store(O + base + k * 2 + 1, re * wi + im * wr, mask=mask)

    @triton.jit
    def k_twiddle_fused(X, O, N: tl.constexpr, R: tl.constexpr,
                        T: tl.constexpr, BLOCK: tl.constexpr,
                        CONJ: tl.constexpr):
        """Twiddle fused with the r-FFT -> stage2 layout handoff
        (zero-copy [b,k,j] read). Same kernel the holdout measured."""
        pid = tl.program_id(0)
        j = pid % R
        b = pid // R
        k = tl.arange(0, BLOCK)
        mask = k < T
        xoff = (b * T * R + k * R + j) * 2
        re = tl.load(X + xoff, mask=mask, other=0.0)
        im = tl.load(X + xoff + 1, mask=mask, other=0.0)
        jf = j.to(tl.float32) + (k * 0.0)
        ang = (6.283185307179586 / N) * jf * k.to(tl.float32)
        if CONJ == 1:
            ang = -ang
        wr = tl.cos(ang)
        wi = -tl.sin(ang)
        obase = ((b * R + j) * T) * 2
        tl.store(O + obase + k * 2, re * wr - im * wi, mask=mask)
        tl.store(O + obase + k * 2 + 1, re * wi + im * wr, mask=mask)


def factors(n: int) -> tuple[int, int]:
    """(t, r) split for N = t*r — closest factor pair with t, r >= 8.

    Identical to the measured ``scripts/issue10_wp3_perf._factors``
    (closest-to-square split), so the deployed geometry is the
    measured geometry for every promoted N.
    """
    cands = []
    t = 2
    while t * t <= n:
        if n % t == 0 and t >= 8 and n // t >= 8:
            cands.append((t, n // t))
        t += 1
    cands.sort(key=lambda c: abs(c[0] - c[1]))
    if not cands:
        raise ValueError(
            f"no legal 4-step split for N={n} (need pow2 factors "
            "t, r >= 8)")
    return cands[0]


@dataclass
class Hybrid4Plan:
    """Framework-owned 4-step plan; the core is the vendor host-API
    FFT (a1p1 capability), the twiddle/layout/digit-reversal are ours."""

    n: int
    batch: int
    variant: str
    direction: str = "dit"
    t: int = field(init=False)
    r: int = field(init=False)
    _tw: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.direction != "dit":
            raise ValueError("4-step hybrid4 flow is DIT-only this round")
        if self.variant not in VARIANTS:
            raise ValueError(
                f"variant {self.variant!r} not in {VARIANTS}")
        if self.batch < 1:
            raise ValueError(f"batch {self.batch} not positive")
        if self.n & (self.n - 1):
            raise ValueError(f"N={self.n} not a power of two")
        self.t, self.r = factors(self.n)

    def scratch(self, device) -> torch.Tensor:
        """Plan-cached (b, r, t) staging/twiddle buffer (reused
        in-place every call — the measured setup structure)."""
        key = str(device)
        if key not in self._tw:
            self._tw[key] = torch.empty(
                (self.batch, self.r, self.t), dtype=torch.complex64,
                device=device)
        return self._tw[key]


def plan_hybrid4(n: int, batch: int, variant: str = "4step-fused",
                 direction: str = "dit") -> Hybrid4Plan:
    return Hybrid4Plan(n=n, batch=batch, variant=variant,
                       direction=direction)


def _validate_x(x, plan: Hybrid4Plan) -> None:
    if not HAS_RUNTIME or not isinstance(x, torch.Tensor):
        raise TypeError("hybrid4 input must be a torch.Tensor "
                        "(torch+triton runtime required)")
    if x.dtype != torch.complex64:
        raise TypeError(f"hybrid4 input must be complex64, got {x.dtype}")
    if tuple(x.shape) != (plan.batch, plan.n):
        raise ValueError(
            f"hybrid4 plan expects ({plan.batch}, {plan.n}); "
            f"got {tuple(x.shape)}")


def _run4(x: torch.Tensor, plan: Hybrid4Plan, inverse: bool,
          fused: bool) -> torch.Tensor:
    """One 4-step execution (B, N) complex64 -> (B, N) complex64.

    Stage order and data movement are byte-for-byte the measured
    benchmark structure; plan caches replace per-call allocation. The
    one intentional deviation: the harness's mid-call
    ``torch.cuda.synchronize()`` after the twiddle launch is dropped —
    stream ordering gives identical outputs and identical steady-state
    work, and any effect can only make production no slower than the
    measured number (the caller's end-of-call sync still bounds the
    same total span).
    """
    from src.xpu.context import xpu_context

    _require_runtime()
    _validate_x(x, plan)
    b, n = plan.batch, plan.n
    t, r = plan.t, plan.r
    view = x.view(b, r, t)          # [j, k], n = j*t + k (contiguous x)
    tw = plan.scratch(x.device)
    tw_flat = torch.view_as_real(tw).reshape(-1)
    conj = 1 if inverse else 0
    op = torch.fft.ifft if inverse else torch.fft.fft
    if fused:
        # stage1 r-FFT over j with output kept [b, t, k-major j]; the
        # fused twiddle writes the stage-2 buffer directly (no
        # transpose-back / reshape GM copy between the stages).
        vt = view.transpose(1, 2)                       # [k, j]
        stage1 = op(vt, dim=-1)                         # [k, j]
        x_flat = torch.view_as_real(stage1).reshape(-1)  # view, no copy
        with xpu_context(0):
            k_twiddle_fused[(b * r,)](x_flat, tw_flat, n, r, t, t, conj)
        stage2 = op(tw, dim=2)
    else:
        # measured transpose variant: explicit [k, j] -> twiddle in
        # [j, k] layout -> t-FFT, digit-reversal permute out.
        vt = view.transpose(1, 2)                       # [k, j]
        stage1 = op(vt, dim=-1).transpose(1, 2)         # [j, k]
        x_flat = torch.view_as_real(stage1).reshape(-1)
        with xpu_context(0):
            k_twiddle[(b * r,)](x_flat, tw_flat, n, r, t, t, conj)
        stage2 = op(tw, dim=2)
    # digit-reversal transpose + flatten (framework-owned, measured)
    return stage2.permute(0, 2, 1).contiguous().reshape(b, n)


def run_hybrid4(x: torch.Tensor, plan: Hybrid4Plan,
                inverse: bool = False) -> torch.Tensor:
    """4-step hybrid, transpose-twiddle variant (measured
    ``hybrid_vendor_p`` implementation)."""
    return _run4(x, plan, inverse, fused=False)


def run_hybrid4_fused(x: torch.Tensor, plan: Hybrid4Plan,
                      inverse: bool = False) -> torch.Tensor:
    """4-step hybrid with the twiddle fused into the stage1->stage2
    handoff (measured ``hybrid_fused_p`` implementation)."""
    return _run4(x, plan, inverse, fused=True)


def plan_ledger(plan: Hybrid4Plan) -> dict:
    """Analytical call/boundary accounting (mirrors the adapter's
    plan_ledger; host-testable, never a measurement)."""
    b, n, t, r = plan.batch, plan.n, plan.t, plan.r
    elem = 8
    return {
        "variant": plan.variant,
        "executed": VARIANT_EXECUTED[plan.variant],
        "split": {"t": t, "r": r},
        "core": "vendor host-API FFT on materialized sub-blocks "
                "(a1p1_materialized_subtransform)",
        "twiddle": ("triton k_twiddle" if plan.variant == "4step-transpose"
                    else "triton k_twiddle_fused (zero-copy handoff)"),
        "materialized_boundaries": [
            {"op": "stage1 input transpose [j,k]->[k,j]",
             "bytes_est": b * n * elem},
            {"op": "twiddle write into stage-2 buffer",
             "bytes_est": b * n * elem + r * t * elem},
            {"op": "digit-reversal permute + flatten -> out",
             "bytes_est": b * n * elem},
        ],
        "buffers": "plan-cached twiddle/staging buffer; output fresh "
                   "per call",
    }
