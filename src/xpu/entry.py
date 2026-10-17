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
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .selector import Selection, load_table, select

DEFAULT_BLOCK = 1024
SUPPORTED_DIRECTIONS = ("dit", "dif")
SUPPORTED_PRECISIONS = ("fp32",)
SUPPORTED_TRANSFORMS = ("c2c",)
SUPPORTED_TRANSFORM_DIRECTIONS = ("forward", "inverse")
SUPPORTED_NORMS = ("none", "forward", "backward", "ortho")
_SUPPORTED_DTYPES = {"fp32": "complex64"}


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
    selection_source: str        # "table" | "abstain" | "baseline-control"
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
            f"supports {SUPPORTED_TRANSFORMS} only (C2C forward+inverse); "
            "r2c/c2r must not fall back to a different transform")
    if transform_direction not in SUPPORTED_TRANSFORM_DIRECTIONS:
        raise ValueError(
            f"unsupported transform_direction {transform_direction!r}; "
            f"supports {SUPPORTED_TRANSFORM_DIRECTIONS}")
    if norm not in SUPPORTED_NORMS:
        raise ValueError(
            f"unsupported norm {norm!r}; supports {SUPPORTED_NORMS}")


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


def _entry_block(table: dict, sel: Selection) -> int:
    """Incumbent block for the shape when the table records one."""
    for e in table.get("entries", ()):
        if (e["N"] == sel.n and e["batch"] == sel.batch
                and e["direction"] == sel.direction):
            return int(e.get("incumbent_block") or e.get("block")
                       or DEFAULT_BLOCK)
    return DEFAULT_BLOCK


def plan_fft(direction: str, n: int, batch: int = 1, precision: str = "fp32",
             profile_hash: str | None = None,
             table: dict | None = None,
             env_hash: str | None = "auto",
             transform_type: str = "c2c",
             transform_direction: str = "forward",
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

    Inverse C2C reuses the forward incumbent from the table: conjugate-
    trick lowerings share candidate structure, so the measured ordering
    carries over; the plan records the semantics and the executor wraps
    the forward lowering (conjugate in/out + epilogue scale).
    """
    _validate_semantics(direction, precision, transform_type,
                        transform_direction, norm)
    table = table if table is not None else load_table()
    sel = select(direction, n, batch=batch, precision=precision,
                 profile_hash=profile_hash, table=table,
                 env_hash=env_hash)
    if not sel.selected:
        return _m0_plan(n, batch, direction, precision, sel.reason, True,
                        block=_entry_block(table, sel),
                        transform_type=transform_type,
                        transform_direction=transform_direction, norm=norm)
    name = _dispatch(sel)
    if name is None:
        return _m0_plan(
            n, batch, direction, precision,
            f"no lowering for {sel.flow_realization}/{sel.processing_unit} "
            f"direction={direction} batch={batch}; M0 fallback",
            True, block=_entry_block(table, sel),
            transform_type=transform_type,
            transform_direction=transform_direction, norm=norm)
    return ExecutionPlan(
        n=n, batch=batch, direction=direction, precision=precision,
        selection_source="table", reason=sel.reason, fallback=False,
        executed=name, block=sel.block, flow_id=sel.flow_id,
        flow_realization=sel.flow_realization,
        processing_unit=sel.processing_unit, params=dict(sel.params or {}),
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


def _check_plan_input(plan: "ExecutionPlan", shape, dtype_name: str) -> None:
    """Host-testable shape/dtype contract for plan-reuse execution.

    A plan resolved for one shape must never silently launch against a
    different tensor: rank/N/batch and dtype are validated before any
    runner call.
    """
    want = _SUPPORTED_DTYPES.get(plan.precision, "complex64")
    if dtype_name != want:
        raise TypeError(
            f"x must be {want} for plan precision={plan.precision}, "
            f"got {dtype_name}")
    shape = tuple(int(s) for s in shape)
    if plan.batch == 1:
        if len(shape) != 1 or shape[0] != plan.n:
            raise ValueError(
                f"plan expects rank-1 x with N={plan.n} (batch=1); "
                f"got shape {shape}")
    else:
        if len(shape) != 2 or shape != (plan.batch, plan.n):
            raise ValueError(
                f"plan expects x shape ({plan.batch}, {plan.n}); "
                f"got shape {shape}")


def execute_fft_plan(plan: ExecutionPlan, x):
    """Execute a resolved plan without re-querying the table (plan-reuse).

    ``x`` must already sit on the XPU device with the plan's semantics
    (rank-1 for B=1, rank-2 for batched) and the plan's dtype; the shape
    and dtype are validated before launch. The one-time plan/dispatch cost
    of :func:`plan_fft` is **not** included here.
    """
    import torch  # local import: entry stays host-importable for planning

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
    _check_plan_input(plan, tuple(x.shape), str(x.dtype).split(".")[-1])
    if plan.transform_type not in SUPPORTED_TRANSFORMS:
        raise ValueError(
            f"plan transform_type {plan.transform_type!r} not supported")
    if plan.transform_direction not in SUPPORTED_TRANSFORM_DIRECTIONS:
        raise ValueError(
            f"plan transform_direction {plan.transform_direction!r} "
            "not supported")
    if plan.norm not in SUPPORTED_NORMS:
        raise ValueError(f"plan norm {plan.norm!r} not supported")
    module = __import__("src.xpu.runner", fromlist=[plan.executed])
    fn = getattr(module, plan.executed)
    # Explicit conjugation (not a conj-view): the XPU kernels split
    # real/imag views, so materialize the conjugate for the inverse.
    if plan.transform_direction == "inverse":
        x = torch.complex(x.real, -x.imag)
    y = _invoke(fn, x, plan.direction, plan.block or DEFAULT_BLOCK,
                plan.params)
    scale = norm_scale(plan.n, plan.transform_direction, plan.norm)
    # Scale the split components: complex64 * python float is NOT
    # IMPLEMENTED in this vendor stack (torch_xmlir mul), while the
    # float path is. Conjugation fuses with the inverse scale here.
    if plan.transform_direction == "inverse":
        return torch.complex(y.real * scale, -y.imag * scale)
    if scale != 1.0:
        return torch.complex(y.real * scale, y.imag * scale)
    return y


def run_fft(x, direction: str, *, batch: int | None = None,
            precision: str = "fp32", profile_hash: str | None = None,
            table: dict | None = None,
            transform_type: str = "c2c",
            transform_direction: str = "forward",
            norm: str = "backward"):
    """One-shot convenience: plan + execute; return ``(y, ExecutionReport)``.

    Unsupported direction/precision/dtype raise before any query. A rank-2
    ``(1, N)`` input is squeezed to rank-1 for the B=1 winner and the
    output is restored to ``(1, N)``. Per-call plan/dispatch overhead is
    included here — use :func:`plan_fft` + :func:`execute_fft_plan` for
    steady-state (plan-reuse) timings.
    """
    import torch  # local import: entry stays host-importable for planning

    _validate_semantics(direction, precision, transform_type,
                        transform_direction, norm)
    want_dtype = getattr(torch, _SUPPORTED_DTYPES[precision])
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
    if x.dtype != want_dtype:
        raise TypeError(
            f"x must be {want_dtype} for precision={precision}, "
            f"got {x.dtype}")
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

    plan = plan_fft(direction, n, batch=b, precision=precision,
                    profile_hash=profile_hash, table=table,
                    transform_type=transform_type,
                    transform_direction=transform_direction, norm=norm)
    y = execute_fft_plan(plan, x)
    if squeezed:
        y = y.unsqueeze(0)
    return y, plan.as_report()
