"""Unified selector -> execution entry (review closure P0/P1).

Contract (plan section 5/P2, review exit criteria):

1. validate input/profile;
2. query the frozen table (:mod:`.selector`);
3. validate the lowering contract (runner-side validation);
4. execute the selected M0/M1/M2/M3/M5 implementation;
5. abstain / unavailable => execute M0 and report the fallback reason.

``plan_fft`` is host-only (no torch): it resolves the Selection onto a concrete
runner function and records exactly what would run. ``run_fft`` performs the
same resolution and then launches, returning ``(y, ExecutionReport)`` so the
actual executed configuration is observable — never "selected metadata but
ran fixed M0".

M0 closure: the fallback path executes ``fft_dit``/``fft_dif`` (batch>1:
the M4 stage flow at the conservative ``folded`` variant) at the table
entry's incumbent block when known, else the library default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .selector import Selection, load_table, select

DEFAULT_BLOCK = 1024


@dataclass(frozen=True)
class ExecutionPlan:
    """What will actually run for one shape under the frozen table."""

    n: int
    batch: int
    direction: str
    precision: str
    selection_source: str        # "table" | "abstain"
    reason: str
    fallback: bool               # True when M0 runs instead of a table winner
    executed: str                # runner function name actually used
    block: int | None = None
    flow_id: str | None = None
    flow_realization: str | None = None
    processing_unit: str | None = None
    params: dict = field(default_factory=dict)

    def as_report(self) -> "ExecutionReport":
        return ExecutionReport(**self.__dict__)


@dataclass(frozen=True)
class ExecutionReport(ExecutionPlan):
    """ExecutionPlan plus the observation that the plan was carried out."""


def _m0_name(direction: str, batch: int) -> str:
    if batch > 1:
        return "fft_dit_batch" if direction == "dit" else "fft_dif_batch"
    return "fft_dit" if direction == "dit" else "fft_dif"


def _m0_plan(n, batch, direction, precision, reason, fallback,
             block=DEFAULT_BLOCK) -> ExecutionPlan:
    return ExecutionPlan(
        n=n, batch=batch, direction=direction, precision=precision,
        selection_source="abstain" if fallback else "table",
        reason=reason, fallback=fallback, executed=_m0_name(direction, batch),
        block=block, flow_id="baseline-global-stage",
        flow_realization="global-stage", processing_unit="radix2-scalar",
        params=({"variant": "folded", "batch": batch} if batch > 1 else {}),
    )


def _dispatch(sel: Selection) -> str | None:
    """Map a table Selection onto the runner function name, or None when no
    lowering implements that (realization, unit, direction) combination."""
    key = (sel.flow_realization, sel.processing_unit, sel.direction)
    if key == ("global-stage", "radix2-scalar", "dit"):
        return "fft_dit" if sel.batch == 1 else "fft_dit_batch"
    if key == ("global-stage", "radix2-scalar", "dif"):
        return "fft_dif" if sel.batch == 1 else "fft_dif_batch"
    if key == ("global-stage", "radix4-scalar", "dit"):
        return "fft_dit_radix4" if sel.batch == 1 else None
    if key == ("global-stage", "radix4-scalar", "dif"):
        return "fft_dif_radix4" if sel.batch == 1 else None
    if key == ("fused-inplace", "radix2-scalar", "dit"):
        # M1 is DIT-only in this tree; there is no DIF fused lowering.
        return "fft_dit_fused"
    if key == ("matrix-resident", "dense-dot-fft64", "dit"):
        return "fft_dit_matrix" if sel.batch == 1 else None
    if key == ("matrix-resident", "dense-dot-fft64", "dif"):
        return "fft_dif_matrix" if sel.batch == 1 else None
    if key == ("grouped-boundary", "composite-linear-map", "dit"):
        return "fft_dit_grouped" if sel.batch == 1 else None
    if key == ("grouped-boundary", "composite-linear-map", "dif"):
        return "fft_dif_grouped" if sel.batch == 1 else None
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
             table: dict | None = None) -> ExecutionPlan:
    """Resolve the frozen table onto a concrete plan (host-only, no torch).

    ``profile_hash=None`` abstains — a missing profile can never bypass the
    table. Anything the dispatch layer cannot lower (unknown realization,
    batched non-M4 flow, DIF fused) falls back to M0 with the reason.
    """
    table = table if table is not None else load_table()
    sel = select(direction, n, batch=batch, precision=precision,
                 profile_hash=profile_hash, table=table)
    if not sel.selected:
        return _m0_plan(n, batch, direction, precision, sel.reason, True,
                        block=_entry_block(table, sel))
    name = _dispatch(sel)
    if name is None:
        return _m0_plan(
            n, batch, direction, precision,
            f"no lowering for {sel.flow_realization}/{sel.processing_unit} "
            f"direction={direction} batch={batch}; M0 fallback",
            True, block=_entry_block(table, sel))
    return ExecutionPlan(
        n=n, batch=batch, direction=direction, precision=precision,
        selection_source="table", reason=sel.reason, fallback=False,
        executed=name, block=sel.block, flow_id=sel.flow_id,
        flow_realization=sel.flow_realization,
        processing_unit=sel.processing_unit, params=dict(sel.params or {}),
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
    if name in ("fft_dit_grouped", "fft_dif_grouped"):
        return fn(x, block=block, tile=int(params.get("tile", 64)),
                  rows=int(params.get("rows", 8)),
                  glog=int(params.get("glog", 2)),
                  maxp=int(params.get("maxp", 4096)),
                  cells=int(params.get("cells", 1)))
    if name in ("fft_dit_batch", "fft_dif_batch"):
        return fn(x, block=block, variant=str(params.get("variant", "folded")))
    raise AssertionError(f"unmapped runner function {fn}")


def run_fft(x, direction: str, *, batch: int | None = None,
            precision: str = "fp32", profile_hash: str | None = None,
            table: dict | None = None):
    """Execute the planned implementation; return ``(y, ExecutionReport)``.

    ``x`` must already sit on the XPU device. ``batch`` defaults to the
    tensor's leading dimension for rank-2 inputs and 1 for rank-1.
    """
    import torch  # local import: entry stays host-importable for planning

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
    if x.dim() == 1:
        n, b = x.numel(), 1
    elif x.dim() == 2:
        b, n = x.shape
    else:
        raise ValueError(f"x must be rank-1 or rank-2, got dim={x.dim()}")
    if batch is not None and batch != b:
        raise ValueError(f"batch={batch} does not match x.shape[0]={b}")

    plan = plan_fft(direction, n, batch=b, precision=precision,
                    profile_hash=profile_hash, table=table)
    module = __import__("src.xpu.runner", fromlist=[plan.executed])
    fn = getattr(module, plan.executed)
    y = _invoke(fn, x, direction, plan.block or DEFAULT_BLOCK, plan.params)
    return y, plan.as_report()
