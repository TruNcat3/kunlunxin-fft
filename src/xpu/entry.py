"""Unified selector -> execution entry (review closure P0/P1).

Contract (plan section 5/P2, review exit criteria):

1. validate input/profile — only the currently supported semantics
   (``direction in {dit,dif}``, ``transform_type=c2c``,
   ``transform_direction in {forward,inverse}``, ``norm in
   {none,forward,backward,ortho}``, ``precision=fp32``, complex64 XPU
   tensor) are accepted; anything else raises before any table query or
   launch;
2. query the frozen table (:mod:`.selector`);
3. validate the lowering contract (runner-side validation);
4. execute the selected M0/M1/M2/M3/M5/M4 implementation;
5. abstain / undispatchable => execute M0 and report the fallback reason.

Two execution modes (plan semantics, like an FFT library plan):

* :func:`plan_fft` resolves the table once (host-only, no torch);
  :func:`execute_fft_plan` runs the resolved plan without re-querying —
  the **plan-reuse** path the 2% guard is defined against.
* :func:`run_fft` is the one-shot convenience API (plan + execute per
  call); its per-call dispatch overhead is reported separately and is
  never conflated with kernel performance.

A rank-2 ``(1, N)`` input is squeezed to rank-1, executed as the B=1
winner, and the output is restored to ``(1, N)``. M0 closure: abstain,
profile mismatch, or an undispatchable combination executes ``fft_dit``
/ ``fft_dif`` (batch>1: the M4 stage flow, conservative ``folded``
variant) at the table entry's incumbent block when known, else the
library default.

Inverse C2C (issue #6 WP1) reuses the forward lowerings: the plan runs
the same selected flow on the conjugated input, conjugates the output,
and applies the norm's epilogue scale — no parallel architecture, same
selector table (conjugate-trick lowerings share the candidate
structure, so the forward incumbent ordering carries over). Norm
semantics match ``torch.fft``/``numpy.fft`` exactly (``backward`` is
the default: unscaled forward, ``1/N`` inverse).

R2C/C2R (issue #6 PR-C) are **compatibility paths**, explicitly marked
in ``plan.params["real_compat"]``: they reuse the C2C core with all
packing / Hermitian reconstruction / half-spectrum cropping work done
inside :func:`execute_fft_plan` (i.e. inside any timed plan-reuse or
one-shot call — never hidden outside the measurement). R2C input is
real ``float32`` ``(N,)``/``(B,N)``, output complex64 half spectrum
``(N//2+1,)``; C2R input is that half spectrum, output real
``float32`` full length. DC/Nyquist imaginary parts follow
torch.fft: ignored, not an error. These paths must not be presented as
a dedicated real-FFT core; a vendor/third-party real-FFT processing
unit can later replace the core behind the same contract.
"""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass, field

from .composition import composition_select
from .config import (
    NORMALIZATIONS,
    TRANSFORM_DIRECTIONS,
    TRANSFORM_TYPES,
)
from .selector import Selection, find_entry, load_table, select

DEFAULT_BLOCK = 1024
SUPPORTED_DIRECTIONS = ("dit", "dif")
SUPPORTED_PRECISIONS = ("fp32",)
# Single source: the semantic-key taxonomy lives with the table
# contract (config.py, issue #8 PR-G); these aliases keep the
# historical entry-module names.
SUPPORTED_TRANSFORMS = TRANSFORM_TYPES
REAL_COMPAT_MARK = "packed-c2c-compat"
SUPPORTED_TRANSFORM_DIRECTIONS = TRANSFORM_DIRECTIONS
SUPPORTED_NORMS = NORMALIZATIONS
_SUPPORTED_DTYPES = {"fp32": "complex64"}
REAL_FLOW_MARK = "dedicated-half-spectrum"


def norm_scale(n: int, transform_direction: str, norm: str) -> float:
    """Epilogue scale for one (direction, norm) pair — torch.fft semantics.

    ================== ============ ============
    norm               forward      inverse
    ================== ============ ============
    ``none``           1            1
    ``forward``        1/N          1
    ``backward``       1            1/N
    ``ortho``          1/sqrt(N)    1/sqrt(N)
    ================== ============ ============
    """
    if transform_direction not in SUPPORTED_TRANSFORM_DIRECTIONS:
        raise ValueError(f"bad transform_direction {transform_direction!r}")
    if norm not in SUPPORTED_NORMS:
        raise ValueError(f"bad norm {norm!r}")
    if norm == "ortho":
        return 1.0 / math.sqrt(n)
    if (norm == "forward" and transform_direction == "forward") or (
            norm == "backward" and transform_direction == "inverse"):
        return 1.0 / n
    return 1.0


@dataclass(frozen=True)
class ExecutionPlan:
    """What will actually run for one shape under the frozen table."""

    n: int
    batch: int
    direction: str
    precision: str
    selection_source: str  # "table" | "abstain" | "composition"
    reason: str
    fallback: bool               # True when M0 runs instead of a table winner
    executed: str                # runner function name actually used
    block: int | None = None
    flow_id: str | None = None
    flow_realization: str | None = None
    processing_unit: str | None = None
    params: dict = field(default_factory=dict)
    # WP1 semantic key: transform_type / direction / normalization.
    transform_type: str = "c2c"
    transform_direction: str = "forward"
    norm: str = "backward"

    def as_report(self) -> "ExecutionReport":
        return ExecutionReport(**self.__dict__)


@dataclass(frozen=True)
class ExecutionReport(ExecutionPlan):
    """ExecutionPlan plus the observation that the plan was carried out."""


def _validate_semantics(direction: str, precision: str,
                        transform_type: str = "c2c",
                        transform_direction: str = "forward",
                        norm: str = "backward") -> None:
    """Reject unsupported semantics before any table query or launch."""
    if direction not in SUPPORTED_DIRECTIONS:
        raise ValueError(
            f"unsupported direction {direction!r}; this round supports "
            f"{SUPPORTED_DIRECTIONS} only (forward/inverse C2C). r2c/c2r "
            "are roadmap and must not fall back to a different transform")
    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(
            f"unsupported precision {precision!r}; this round supports "
            f"{SUPPORTED_PRECISIONS} only")
    if transform_type not in SUPPORTED_TRANSFORMS:
        raise ValueError(
            f"unsupported transform_type {transform_type!r}; this round "
            f"supports {SUPPORTED_TRANSFORMS} only; "
            "unknown transforms must not fall back to a different one")
    if transform_direction not in SUPPORTED_TRANSFORM_DIRECTIONS:
        raise ValueError(
            f"unsupported transform_direction {transform_direction!r}; "
            f"supports {SUPPORTED_TRANSFORM_DIRECTIONS}")
    if norm not in SUPPORTED_NORMS:
        raise ValueError(
            f"unsupported norm {norm!r}; supports {SUPPORTED_NORMS}")
    # Real transforms are direction-locked: rfft is forward, irfft
    # inverse (torch.fft contract). A mismatched pair is an error, not
    # a silent reinterpretation.
    if transform_type == "r2c" and transform_direction != "forward":
        raise ValueError(
            "r2c is a forward transform; got "
            f"transform_direction={transform_direction!r}")
    if transform_type == "c2r" and transform_direction != "inverse":
        raise ValueError(
            "c2r is an inverse transform; got "
            f"transform_direction={transform_direction!r}")


def _m0_name(direction: str, batch: int) -> str:
    if batch > 1:
        return "fft_dit_batch" if direction == "dit" else "fft_dif_batch"
    return "fft_dit" if direction == "dit" else "fft_dif"


def _m0_plan(n, batch, direction, precision, reason, fallback,
             block=DEFAULT_BLOCK, transform_type: str = "c2c",
             transform_direction: str = "forward",
             norm: str = "backward") -> ExecutionPlan:
    return ExecutionPlan(
        n=n, batch=batch, direction=direction, precision=precision,
        selection_source="abstain" if fallback else "table",
        reason=reason, fallback=fallback, executed=_m0_name(direction, batch),
        block=block, flow_id="baseline-global-stage",
        flow_realization="global-stage", processing_unit="radix2-scalar",
        params=({"variant": "folded", "batch": batch} if batch > 1 else {}),
        transform_type=transform_type,
        transform_direction=transform_direction, norm=norm,
    )


def _dispatch(sel: Selection) -> str | None:
    """Map a table Selection onto the runner function name, or None when no
    lowering implements that (realization, unit, direction, batch)
    combination. Batched M2 (matrix-resident) and M5 (radix-4) lower through
    the batch-aware kernels; batched M1/M3 have no lowering."""
    key = (sel.flow_realization, sel.processing_unit, sel.direction)
    if key == ("global-stage", "radix2-scalar", "dit"):
        return "fft_dit" if sel.batch == 1 else "fft_dit_batch"
    if key == ("global-stage", "radix2-scalar", "dif"):
        return "fft_dif" if sel.batch == 1 else "fft_dif_batch"
    if key == ("global-stage", "radix4-scalar", "dit"):
        return "fft_dit_radix4" if sel.batch == 1 else "fft_dit_radix4_batch"
    if key == ("global-stage", "radix4-scalar", "dif"):
        return "fft_dif_radix4" if sel.batch == 1 else "fft_dif_radix4_batch"
    if key == ("matrix-resident", "dense-dot-fft64", "dit"):
        return "fft_dit_matrix" if sel.batch == 1 else "fft_dit_matrix_batch"
    if key == ("matrix-resident", "dense-dot-fft64", "dif"):
        return "fft_dif_matrix" if sel.batch == 1 else "fft_dif_matrix_batch"
    if sel.batch != 1:
        # batched M1/M3 have no lowering this round.
        return None
    if key == ("fused-inplace", "radix2-scalar", "dit"):
        # M1 is DIT-only in this tree; there is no DIF fused lowering.
        return "fft_dit_fused"
    if key == ("grouped-boundary", "composite-linear-map", "dit"):
        return "fft_dit_grouped"
    if key == ("grouped-boundary", "composite-linear-map", "dif"):
        return "fft_dif_grouped"
    return None


# HybridPlan instances are cached per ExecutionPlan (frozen plan ->
# dict cannot hold them and stay JSON-safe). The registry keys by
# id(plan) but holds a weakref to the plan: when the plan dies its
# entry is evicted, so a recycled object id can never observe a
# stale HybridPlan from a different shape.
_HYBRID_CACHE: dict = {}


def _composition_plan(comp: dict, n: int, batch: int, direction: str,
                      precision: str, transform_direction: str,
                      norm: str) -> ExecutionPlan:
    """Exact-shape promoted composition entry -> ExecutionPlan.

    Attribution stays honest: ``executed`` names the composition
    runner, ``flow_realization``/``processing_unit`` say composition
    (host API on materialized sub-blocks), never resident kernel.
    """
    ev = comp["evidence"]
    reason = (
        f"composition {comp['flow_id']} (exact shape, "
        f"{comp['guard_pct']:.1f}% >= {2.0}% guard vs "
        f"{comp.get('incumbent_flow', 'incumbent')}; "
        f"p50={ev['p50_ms']:.4f}ms over {ev['sessions']} sessions "
        f"+ confirmation, err_max={ev['err_max']:g}): {comp['reason']}")
    return ExecutionPlan(
        n=n, batch=batch, direction=direction, precision=precision,
        selection_source="composition", reason=reason, fallback=False,
        executed="run_hybrid", block=None,
        flow_id=comp["flow_id"],
        flow_realization=comp["flow_realization"],
        processing_unit=comp["processing_unit"],
        params={"flow_kind": "composition",
                "tile": comp["tile"], "core": comp["core"],
                "contract_id": comp.get("contract_id")
                or comp.get("_contract_id"),
                "composition_guard_pct": comp["guard_pct"],
                "evidence_artifact": ev["artifact"]},
        transform_type="c2c",
        transform_direction=transform_direction, norm=norm,
    )


def _with_compat(plan: ExecutionPlan, compat: dict) -> ExecutionPlan:
    """Return ``plan`` with the real-compat marker merged into params."""
    params = dict(plan.params or {})
    params.update(compat)
    return ExecutionPlan(**{**plan.__dict__, "params": params})


def _entry_block(table: dict, sel: Selection) -> int:
    """Incumbent block for the shape when the table records one.

    Resolves through the same semantic tier as ``select()`` (exact
    semantic entry first, legacy fallback), so the M0 block and the
    selected plan never come from different tiers of the table.
    """
    e = find_entry(table, sel.n, sel.batch, sel.direction,
                   sel.transform_type, sel.transform_direction,
                   sel.normalization)
    if e is None:
        return DEFAULT_BLOCK
    return int(e.get("incumbent_block") or e.get("block") or DEFAULT_BLOCK)


def plan_fft(direction: str, n: int, batch: int = 1, precision: str = "fp32",
             profile_hash: str | None = None,
             table: dict | None = None,
             env_hash: str | None = "auto",
             transform_type: str = "c2c",
             transform_direction: str | None = None,
             norm: str = "backward") -> ExecutionPlan:
    """Resolve the frozen table onto a concrete plan (host-only, no torch).

    The semantic key is (transform_type, direction, normalization):
    ``transform_type`` is ``"c2c"`` this round (r2c/c2r raise),
    ``direction`` is the decomposition (``dit``/``dif``),
    ``transform_direction`` is ``"forward"``/``"inverse"``, and ``norm``
    is one of :data:`SUPPORTED_NORMS` (``"backward"`` default, matching
    torch.fft). Defaults reproduce the historical forward, unscaled
    behaviour bit-for-bit.

    Unsupported direction/precision/transform/norm raise ``ValueError``
    before the table query — they must never fall back onto a different
    transform. ``profile_hash=None`` abstains to M0; an undispatchable
    table entry (e.g. a hypothetical batched M1) falls back to M0 with
    the reason. ``env_hash`` (default ``"auto"``) re-verifies that a
    table bound to an environment fingerprint is being used on that same
    device/runtime/compiler stack; unknown or stale environment falls
    back to M0.

    Semantic tiering (issue #8 PR-G): the table query carries the full
    semantic key. Entries built with exact ``(transform_type,
    transform_direction, normalization)`` answer only their own triple
    (inverse / R2C / C2R no longer inherit a forward ordering); legacy
    entries without the fields remain the old deployed path for any
    semantic until that semantic's own entry clears its own guard. For
    a legacy c2c hit the plan still records the semantics and the
    executor wraps the lowering (conjugate in/out + epilogue scale for
    inverse, pack/crop for real transforms) -- the wrapper is part of
    the measured path of that semantic's confirmation rows.

    Previous-deployed fallback (issue #8 Option A): a cell declared in
    the table's ``previous_fallbacks`` list outranks the semantic
    entry at that exact key; the plan deploys the old geometry and
    carries ``params["previous_fallback"] = True`` plus the fallback
    reason so raw rows are attributable to the fallback decision.

    Runner closure (issue #8 P1-A): a table entry carrying an
    ``executed`` field deploys exactly that runner — the measured
    winner recorded end-to-end, even when its orientation differs from
    the entry's locked decomposition direction; legacy entries without
    the field dispatch via the direction-implied runner table.
    """
    # None means "the type's locked direction" (r2c forward, c2r
    # inverse, c2c forward); an explicit value is validated against the
    # lock and raises on conflict.
    if transform_direction is None:
        transform_direction = ("inverse" if transform_type == "c2r"
                               else "forward")
    _validate_semantics(direction, precision, transform_type,
                        transform_direction, norm)
    # Exact-shape promoted composition (PR-E) outranks the butterfly
    # table for its one shape; c2c only — the real-compat path keeps
    # the butterfly lowering. Fail closed to the butterfly path.
    # composition_select also gates on the queried semantic (issue #8
    # P1-B: only c2c+forward+backward may match a composition entry).
    if transform_type == "c2c":
        comp = composition_select(direction, n, batch=batch,
                                  precision=precision,
                                  profile_hash=profile_hash,
                                  env_hash=env_hash,
                                  transform_type=transform_type,
                                  transform_direction=transform_direction,
                                  normalization=norm)
        if comp is not None:
            return _composition_plan(comp, n, batch, direction,
                                     precision, transform_direction,
                                     norm)
    table = table if table is not None else load_table()
    sel = select(direction, n, batch=batch, precision=precision,
                 profile_hash=profile_hash, table=table,
                 env_hash=env_hash, transform_type=transform_type,
                 transform_direction=transform_direction,
                 normalization=norm)
    compat = ({"real_compat": REAL_COMPAT_MARK}
              if transform_type in ("r2c", "c2r") else {})
    if not sel.selected:
        plan = _m0_plan(n, batch, direction, precision, sel.reason, True,
                        block=_entry_block(table, sel),
                        transform_type=transform_type,
                        transform_direction=transform_direction, norm=norm)
        if compat:
            plan = _with_compat(plan, compat)
        return plan
    # Dispatch honors the measured runner recorded end-to-end on the
    # entry (issue #8 P1-A); legacy entries without the field keep the
    # direction-implied runner table.
    name = sel.executed or _dispatch(sel)
    if name is None:
        plan = _m0_plan(
            n, batch, direction, precision,
            f"no lowering for {sel.flow_realization}/{sel.processing_unit} "
            f"direction={direction} batch={batch}; M0 fallback",
            True, block=_entry_block(table, sel),
            transform_type=transform_type,
            transform_direction=transform_direction, norm=norm)
        if compat:
            plan = _with_compat(plan, compat)
        return plan
    params = dict(sel.params or {})
    if getattr(sel, "entry_tier", "") == "previous-fallback":
        # Issue #8 Option A: the plan (and every row measured under it)
        # is attributable to the previous-deployed fallback cell
        # without diffing geometry. Selection.params stays clean --
        # only the deployed plan carries the marker.
        params["previous_fallback"] = True
    if params.get("real_flow"):
        # Dedicated real flow (PR-H): the entry carries the flow id;
        # validate it fail-closed here and mark the plan accordingly
        # (M0 fallbacks above keep the plain compat mark -- they drop
        # real_flow with the rest of the unselected params).
        from .semantic import family_allows

        if not family_allows(transform_type, params["real_flow"]):
            raise ValueError(
                f"real_flow {params['real_flow']!r} not in the "
                f"{transform_type} family")
        params["real_compat"] = REAL_FLOW_MARK
    else:
        params.update(compat)
    return ExecutionPlan(
        n=n, batch=batch, direction=direction, precision=precision,
        selection_source="table", reason=sel.reason, fallback=False,
        executed=name, block=sel.block, flow_id=sel.flow_id,
        flow_realization=sel.flow_realization,
        processing_unit=sel.processing_unit, params=params,
        transform_type=transform_type,
        transform_direction=transform_direction, norm=norm,
    )


def _invoke(fn, x, direction, block, params):
    """Call the runner function with the plan's parameters."""
    name = getattr(fn, "__name__", fn)
    if name in ("fft_dit", "fft_dif"):
        return fn(x, block=block)
    if name in ("fft_dit_radix4", "fft_dif_radix4"):
        return fn(x, block=block)
    if name == "fft_dit_fused":
        return fn(x, block=block, tile=int(params.get("tile", 64)))
    if name in ("fft_dit_matrix", "fft_dif_matrix"):
        return fn(x, block=block, tile=int(params.get("tile", 64)),
                  rows=int(params.get("rows", 8)))
    if name in ("fft_dit_matrix_batch", "fft_dif_matrix_batch"):
        return fn(x, block=block, tile=int(params.get("tile", 64)),
                  rows=int(params.get("rows", 8)),
                  variant=str(params.get("variant", "spread")))
    if name in ("fft_dit_radix4_batch", "fft_dif_radix4_batch"):
        return fn(x, block=block,
                  variant=str(params.get("variant", "spread")))
    if name in ("fft_dit_grouped", "fft_dif_grouped"):
        return fn(x, block=block, tile=int(params.get("tile", 64)),
                  rows=int(params.get("rows", 8)),
                  glog=int(params.get("glog", 2)),
                  maxp=int(params.get("maxp", 4096)),
                  cells=int(params.get("cells", 1)))
    if name in ("fft_dit_batch", "fft_dif_batch"):
        return fn(x, block=block, variant=str(params.get("variant", "folded")))
    raise AssertionError(f"unmapped runner function {fn}")


def _expected_input_n(plan: "ExecutionPlan") -> int:
    """Last-dim length the plan expects for its transform_type."""
    if plan.transform_type == "c2r":
        # half spectrum: N//2+1 complex bins reconstruct N real samples
        return plan.n // 2 + 1
    return plan.n


def _check_plan_input(plan: "ExecutionPlan", shape, dtype_name: str) -> None:
    """Host-testable shape/dtype contract for plan-reuse execution.

    A plan resolved for one shape must never silently launch against a
    different tensor: rank/N/batch and dtype are validated before any
    runner call. Per-transform contracts:

    * ``c2c``: complex64 ``(N,)``/``(B,N)``;
    * ``r2c``: float32 real ``(N,)``/``(B,N)``;
    * ``c2r``: complex64 half spectrum ``(N//2+1,)``/``(B,N//2+1)``.
    """
    if plan.transform_type == "r2c":
        want = "float32"
    else:
        want = _SUPPORTED_DTYPES.get(plan.precision, "complex64")
    if dtype_name != want:
        raise TypeError(
            f"x must be {want} for transform_type={plan.transform_type} "
            f"precision={plan.precision}, got {dtype_name}")
    shape = tuple(int(s) for s in shape)
    n_expect = _expected_input_n(plan)
    if plan.batch == 1:
        if len(shape) != 1 or shape[0] != n_expect:
            raise ValueError(
                f"plan expects rank-1 x with N={n_expect} "
                f"(batch=1, transform_type={plan.transform_type}); "
                f"got shape {shape}")
    else:
        if len(shape) != 2 or shape != (plan.batch, n_expect):
            raise ValueError(
                f"plan expects x shape ({plan.batch}, {n_expect}); "
                f"got shape {shape}")


def execute_fft_plan(plan: ExecutionPlan, x):
    """Execute a resolved plan without re-querying the table (plan-reuse).

    ``x`` must already sit on the XPU device with the plan's semantics
    (rank-1 for B=1, rank-2 for batched) and the plan's dtype; the shape
    and dtype are validated before launch. The one-time plan/dispatch cost
    of :func:`plan_fft` is **not** included here.
    """
    import torch  # local import: entry stays host-importable for planning

    from .ledger import phase  # local: keeps entry host-importable

    with phase("plan-validate"):
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"x must be a torch.Tensor, got {type(x).__name__}")
        _check_plan_input(plan, tuple(x.shape),
                          str(x.dtype).split(".")[-1])
        if plan.transform_type not in SUPPORTED_TRANSFORMS:
            raise ValueError(
                f"plan transform_type {plan.transform_type!r} "
                "not supported")
        if plan.transform_direction not in SUPPORTED_TRANSFORM_DIRECTIONS:
            raise ValueError(
                f"plan transform_direction {plan.transform_direction!r} "
                "not supported")
        if plan.norm not in SUPPORTED_NORMS:
            raise ValueError(f"plan norm {plan.norm!r} not supported")
    if plan.params.get("flow_kind") == "composition":
        # Composition path: framework decomposition owns the flow,
        # the vendor core runs only on materialized sub-blocks
        # (adapter_hybrid). c2c only; the conjugate/scale epilogue is
        # shared with the butterfly path so norm/inverse contracts
        # hold unchanged.
        if plan.transform_type != "c2c":
            raise ValueError(
                "composition plans support transform_type='c2c' only, "
                f"got {plan.transform_type!r}")
        z = x
        if plan.transform_direction == "inverse":
            with phase("inverse-conj"):
                z = torch.complex(z.real, -z.imag)
        with phase("composition-core"):
            y = _invoke_composition(plan, z)
        with phase("epilogue-scale"):
            return _scale_complex(
                y, norm_scale(plan.n, plan.transform_direction, plan.norm),
                conjugate=plan.transform_direction == "inverse")
    module = __import__("src.xpu.runner", fromlist=[plan.executed])
    fn = getattr(module, plan.executed)
    # Dedicated real flow (PR-H): pack / half-spectrum stitch / pre-
    # twiddle / unpack replace the full-length compat pipeline with a
    # half-length core; every timed op stays inside the execution, the
    # same measurement contract as real_compat (plan.params["real_flow"]).
    real_flow = plan.params.get("real_flow")
    if real_flow is not None:
        from .semantic import family_allows

        if plan.transform_type not in ("r2c", "c2r"):
            raise ValueError(
                "real_flow plans support transform_type r2c/c2r only, "
                f"got {plan.transform_type!r}")
        if not family_allows(plan.transform_type, real_flow):
            raise ValueError(
                f"real_flow {real_flow!r} not in the "
                f"{plan.transform_type} family")
        core_block = min(plan.block or DEFAULT_BLOCK, plan.n >> 1)
        if plan.transform_type == "r2c":
            with phase("real-input-pack"):
                z = torch.complex(x[..., 0::2], x[..., 1::2])
            with phase("core-invoke"):
                u = _invoke(fn, z, plan.direction, core_block, plan.params)
            with phase("half-spectrum-stitch"):
                return module.r2c_stitch(
                    u, plan.n,
                    scale=norm_scale(plan.n, "forward", plan.norm),
                    block=plan.block or DEFAULT_BLOCK)
        with phase("half-spectrum-pre"):
            zpre = module.c2r_pre(x, plan.n,
                                  block=plan.block or DEFAULT_BLOCK)
        with phase("core-invoke"):
            u = _invoke(fn, zpre, plan.direction, core_block, plan.params)
        with phase("real-output-epilogue"):
            return module.c2r_unpack(
                u, plan.n,
                scale=2.0 * norm_scale(plan.n, "inverse", plan.norm),
                block=plan.block or DEFAULT_BLOCK)
    # Real compatibility path: all packing / Hermitian reconstruction /
    # half-spectrum cropping happens HERE, inside the timed execution —
    # never hidden outside the measurement (plan.params["real_compat"]).
    if plan.transform_type == "r2c":
        with phase("real-input-pack"):
            z = torch.complex(x, torch.zeros_like(x))
        with phase("core-invoke"):
            y = _invoke(fn, z, plan.direction, plan.block or DEFAULT_BLOCK,
                        plan.params)
        with phase("epilogue-scale"):
            y = _scale_complex(
                y, norm_scale(plan.n, "forward", plan.norm),
                conjugate=False)
        with phase("half-spectrum-crop"):
            return y[..., : plan.n // 2 + 1].contiguous()
    if plan.transform_type == "c2r":
        with phase("hermitian-rebuild"):
            full = _hermitian_reconstruct(x, plan.n)
        with phase("inverse-conj"):
            zc = torch.complex(full.real, -full.imag)
        with phase("core-invoke"):
            y = _invoke(fn, zc, plan.direction, plan.block or DEFAULT_BLOCK,
                        plan.params)
        with phase("epilogue-scale"):
            y = _scale_complex(
                y, norm_scale(plan.n, "inverse", plan.norm),
                conjugate=True)
        with phase("real-output-cast"):
            return y.real.contiguous()
    # Explicit conjugation (not a conj-view): the XPU kernels split
    # real/imag views, so materialize the conjugate for the inverse.
    if plan.transform_direction == "inverse":
        with phase("inverse-conj"):
            x = torch.complex(x.real, -x.imag)
    with phase("core-invoke"):
        y = _invoke(fn, x, plan.direction, plan.block or DEFAULT_BLOCK,
                    plan.params)
    with phase("epilogue-scale"):
        return _scale_complex(
            y, norm_scale(plan.n, plan.transform_direction, plan.norm),
            conjugate=plan.transform_direction == "inverse")


def _invoke_composition(plan: ExecutionPlan, x):
    """Run the composition flow for a plan (plan-reuse caches the
    HybridPlan: twiddle/staging buffers live as long as the cache)."""
    from .adapter_hybrid import plan_hybrid, run_hybrid
    from .ledger import phase

    key = id(plan)
    cached = _HYBRID_CACHE.get(key)
    hp = cached[0] if cached is not None and cached[1]() is plan else None
    if hp is None:
        with phase("composition-prepare"):
            hp = plan_hybrid(plan.n, plan.batch,
                             tile=int(plan.params["tile"]),
                             core=plan.params["core"])
        _HYBRID_CACHE[key] = (
            hp, weakref.ref(plan,
                            lambda _r, k=key: _HYBRID_CACHE.pop(k, None)))
    squeeze = x.dim() == 1
    if squeeze:
        x = x.unsqueeze(0)
    with phase("composition-run"):
        y = run_hybrid(x, hp)
    return y[0] if squeeze else y


def _scale_complex(y, scale: float, conjugate: bool):
    """Epilogue scale on split components.

    ``complex64 * python float`` is NOT IMPLEMENTED in this vendor stack
    (torch_xmlir mul), while the float path is. Conjugation fuses with
    the scale when ``conjugate`` is set (inverse path).
    """
    import torch  # local: entry stays host-importable for planning
    if conjugate:
        return torch.complex(y.real * scale, -y.imag * scale)
    if scale != 1.0:
        return torch.complex(y.real * scale, y.imag * scale)
    return y


def _hermitian_reconstruct(x, n: int):
    """Full-length complex spectrum from a torch.fft half spectrum.

    ``x`` is complex64 ``(n//2+1,)`` or ``(B, n//2+1)``. Mirrors
    ``X[n-k] = conj(X[k])``; DC and (even ``n``) Nyquist imaginary parts
    are zeroed — torch.fft ignores them rather than erroring.
    """
    import torch  # local: entry stays host-importable for planning
    m = n // 2 + 1
    full = torch.empty(
        (*x.shape[:-1], n), dtype=x.dtype, device=x.device)
    full[..., :m] = x
    tail_src = x[..., 1:m - 1] if n % 2 == 0 else x[..., 1:m]
    # flip on the float components: complex64 flip is unimplemented in
    # this vendor stack (xdnn flip.cpp)
    mirrored = torch.complex(tail_src.real.flip(-1),
                             -tail_src.imag.flip(-1))
    full[..., m:] = mirrored
    # DC/Nyquist ownership: imaginary parts are not carried.
    full[..., 0] = torch.complex(full[..., 0].real,
                                torch.zeros_like(full[..., 0].real))
    if n % 2 == 0:
        full[..., m - 1] = torch.complex(
            full[..., m - 1].real,
            torch.zeros_like(full[..., m - 1].real))
    return full


def run_fft(x, direction: str, *, batch: int | None = None,
            precision: str = "fp32", profile_hash: str | None = None,
            table: dict | None = None,
            transform_type: str = "c2c",
            transform_direction: str | None = None,
            norm: str = "backward"):
    """One-shot convenience: plan + execute; return ``(y, ExecutionReport)``.

    Unsupported direction/precision/dtype raise before any query. A rank-2
    ``(1, N)`` input is squeezed to rank-1 for the B=1 winner and the
    output is restored to ``(1, N)``. Per-call plan/dispatch overhead is
    included here — use :func:`plan_fft` + :func:`execute_fft_plan` for
    steady-state (plan-reuse) timings.
    """
    import torch  # local import: entry stays host-importable for planning

    if transform_direction is None:
        transform_direction = ("inverse" if transform_type == "c2r"
                               else "forward")
    _validate_semantics(direction, precision, transform_type,
                        transform_direction, norm)
    if transform_type == "r2c":
        want_dtype = torch.float32
    else:
        want_dtype = getattr(torch, _SUPPORTED_DTYPES[precision])
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
    if x.dtype != want_dtype:
        raise TypeError(
            f"x must be {want_dtype} for transform_type={transform_type} "
            f"precision={precision}, got {x.dtype}")
    squeezed = False
    if x.dim() == 1:
        n, b = x.numel(), 1
    elif x.dim() == 2 and x.shape[0] == 1:
        x = x[0]
        n, b = x.numel(), 1
        squeezed = True
    elif x.dim() == 2:
        b, n = x.shape
    else:
        raise ValueError(f"x must be rank-1 or rank-2, got dim={x.dim()}")
    if batch is not None and batch != b:
        raise ValueError(f"batch={batch} does not match effective batch {b}")
    if transform_type == "c2r":
        # input last dim is the half spectrum; plan n is the inferred
        # even full length (torch.fft contract: odd N needs explicit n,
        # so the one-shot API infers even only).
        if n < 2:
            raise ValueError(f"c2r half-spectrum length {n} too short")
        n = (n - 1) * 2

    plan = plan_fft(direction, n, batch=b, precision=precision,
                    profile_hash=profile_hash, table=table,
                    transform_type=transform_type,
                    transform_direction=transform_direction, norm=norm)
    y = execute_fft_plan(plan, x)
    if squeezed:
        y = y.unsqueeze(0)
    return y, plan.as_report()
