"""Phase / launch ledger for issue #8 WP2 attribution.

Three evidence kinds, never mixed:

* **fresh measured** — :class:`PhaseLedger` segments an execution with
  ``torch.cuda.synchronize()`` around each phase. That is attribution
  mode only: the headline protocol timing (matrix, canary, selector
  gates) runs with the ledger OFF and carries no extra synchronization.
* **analytic estimate** — :func:`launch_ledger` derives launches / GM
  bytes / flops from the lowering families already declared in
  :mod:`src.xpu.cost_model` plus the real-compat op list; on this stack
  there is no NCU-equivalent counter, so these numbers are never
  reported as hardware counters.
* **summary-derived** — :func:`batch_fill_analysis` re-reads release
  summary medians to show per-point cost vs batch (observable
  fill/degredation), without claiming worker/wave counters.

Entry wires :func:`phase` around its segments; with no active ledger
each call collapses to ``nullcontext`` (no sync, no record).
"""
from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from contextvars import ContextVar

KIND_FRESH = "fresh measured (phase timers; extra synchronize added)"
KIND_ANALYTIC = "analytic estimate (cost_model.axes + real-compat op list)"
KIND_SUMMARY = "summary-derived (release summary medians)"

ACTIVE: ContextVar = ContextVar("xpu_phase_ledger", default=None)

# entry executed-runner -> cost_model family (honest absence elsewhere)
_FAMILIES = {
    "fft_dit": "baseline_m0_rank1_loop",
    "fft_dif": "baseline_m0_rank1_loop",
    "fft_dit_batch": "incumbent_m4_folded",
    "fft_dif_batch": "incumbent_m4_folded",
    "fft_dit_radix4": "m5_batch",
    "fft_dif_radix4": "m5_batch",
    "fft_dit_radix4_batch": "m5_batch",
    "fft_dif_radix4_batch": "m5_batch",
    "fft_dit_matrix": "m2_batch",
    "fft_dif_matrix": "m2_batch",
    "fft_dit_matrix_batch": "m2_batch",
    "fft_dif_matrix_batch": "m2_batch",
    "fft_dit_matrix_packet_batch": "m2_packet",
    "fft_dif_matrix_packet_batch": "m2_packet",
}
_UNMODELLED = {
    "fft_dit_fused": "m1 fused lowering has no family in cost_model.axes",
    "fft_dif_fused": "m1 fused lowering has no family in cost_model.axes",
    "fft_dit_grouped": "m3 grouped lowering has no family in cost_model.axes",
    "fft_dif_grouped": "m3 grouped lowering has no family in cost_model.axes",
}


def _sync() -> None:
    import torch
    torch.cuda.synchronize()


class PhaseLedger:
    """Accumulates per-phase wall times for one process/context.

    ``sync=True`` brackets every phase with a device synchronize —
    correct for attribution, wrong for headline timing (the reason the
    ledger is opt-in and off by default).
    """

    def __init__(self, sync: bool = True):
        self.sync = sync
        self.phases: list[dict] = []

    @contextmanager
    def phase(self, name: str):
        if self.sync:
            _sync()
        t0 = time.perf_counter()
        err = None
        try:
            yield
        except BaseException as e:  # record the failing segment too
            err = f"{type(e).__name__}: {e}"
            raise
        finally:
            if self.sync:
                try:
                    _sync()
                except Exception:  # noqa: BLE001 - keep the record
                    pass
            self.phases.append({
                "name": name,
                "ms": (time.perf_counter() - t0) * 1e3,
                "error": err,
            })

    def summary(self) -> dict:
        """Aggregate recorded phases (mean/p50/max, counts)."""
        by: dict[str, list[float]] = {}
        errors: list[dict] = []
        for p in self.phases:
            by.setdefault(p["name"], []).append(p["ms"])
            if p.get("error"):
                errors.append({"name": p["name"], "error": p["error"]})
        phases = {}
        for name, xs in by.items():
            phases[name] = {
                "calls": len(xs),
                "mean_ms": statistics.fmean(xs),
                "p50_ms": statistics.median(xs),
                "max_ms": max(xs),
            }
        return {
            "kind": KIND_FRESH,
            "sync_added": self.sync,
            "phases": phases,
            "phase_order": list(dict.fromkeys(
                p["name"] for p in self.phases)),
            "errors": errors,
            "note": ("attribution mode: each phase is bracketed by "
                     "synchronize(); NOT comparable to headline "
                     "protocol timings (ledger off there)"),
        }


class _NoopPhase:
    """Reusable no-allocation context manager (headline path: five of
    these per execute must stay sub-microsecond)."""

    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


_NOOP = _NoopPhase()


def phase(name: str):
    """Context manager: record ``name`` on the active ledger, or the
    shared no-op when no ledger is active (default/headline path)."""
    led = ACTIVE.get()
    if led is None:
        return _NOOP
    return led.phase(name)


def current_ledger() -> PhaseLedger | None:
    return ACTIVE.get()


def launch_ledger(plan) -> dict:
    """Analytic launch/GM/flop ledger for a resolved ExecutionPlan.

    Reuses :func:`src.xpu.cost_model.axes` (the same declared lowering
    families the model validation uses). Real-compat pack/crop/Hermitian
    work is listed as separate analytic ops. Vendor launches inside a
    composition core stay opaque — recorded as such, never invented.
    """
    executed = getattr(plan, "executed", None)
    n = int(plan.n)
    batch = int(plan.batch)
    params = dict(getattr(plan, "params", None) or {})
    out: dict = {"kind": KIND_ANALYTIC, "executed": executed,
                 "estimate": True}

    if executed == "run_hybrid":
        out.update({
            "family": None,
            "available": False,
            "why": ("composition core: vendor sub-transform calls are "
                    "opaque here; sub-transform shapes/counts live in "
                    "the HybridPlan plan_ledger (canary artifact)"),
            "real_compat_ops": [],
        })
        return out

    family = _FAMILIES.get(executed)
    if family is None:
        out.update({
            "family": None, "available": False,
            "why": _UNMODELLED.get(
                executed, f"no registered lowering family for {executed!r}"),
            "real_compat_ops": [],
        })
        return out

    from .cost_model import axes
    ax = axes(family, n, batch, params)
    out.update({"family": family, "available": True, **ax})

    ops = []
    ttype = getattr(plan, "transform_type", "c2c")
    cplx = n * batch * 8
    half = (n // 2 + 1) * batch * 4
    real_flow = params.get("real_flow")
    if real_flow and ttype == "r2c":
        # Dedicated flow (PR-H): stitch stores fused scale; no crop,
        # no full-length spectrum materialization.
        ops += [
            {"op": "real-input pack (strided torch.complex)",
             "in_bytes": n * batch * 4, "out_bytes": (n // 2) * batch * 8,
             "kind": "analytic estimate"},
            {"op": "half-spectrum stitch (triton, scale fused)",
             "in_bytes": (n // 2) * batch * 8, "out_bytes": half,
             "kind": "analytic estimate"},
        ]
    elif real_flow and ttype == "c2r":
        # Dedicated flow (PR-H): pre-twiddle kernel, then one unpack
        # store folding mirror + conjugate + scale (fused, no separate
        # normalization pass).
        ops += [
            {"op": "half-spectrum pre-twiddle (triton, DC/Nyquist owned)",
             "in_bytes": half, "out_bytes": (n // 2) * batch * 8,
             "kind": "analytic estimate"},
            {"op": "mirror+scale+interleave epilogue (triton, fused)",
             "in_bytes": (n // 2) * batch * 8, "out_bytes": n * batch * 4,
             "kind": "analytic estimate"},
        ]
    elif ttype == "r2c":
        ops += [
            {"op": "real-input-pack (torch.complex)",
             "in_bytes": n * batch * 8, "out_bytes": cplx,
             "kind": "analytic estimate"},
            {"op": "half-spectrum crop (slice+contiguous)",
             "in_bytes": cplx, "out_bytes": half,
             "kind": "analytic estimate"},
        ]
    elif ttype == "c2r":
        ops += [
            {"op": "hermitian-rebuild (empty+flip+write)",
             "in_bytes": half, "out_bytes": cplx,
             "kind": "analytic estimate"},
            {"op": "conjugate materialize",
             "in_bytes": cplx, "out_bytes": cplx,
             "kind": "analytic estimate"},
            {"op": "real-output cast (y.real)",
             "in_bytes": cplx, "out_bytes": n * batch * 4,
             "kind": "analytic estimate"},
        ]
    if (ttype == "c2c"
            and getattr(plan, "transform_direction", "forward")
            == "inverse"):
        # c2r already lists its conjugate above; only c2c inverts here
        ops.append({"op": "inverse conjugate materialize",
                    "in_bytes": cplx, "out_bytes": cplx,
                    "kind": "analytic estimate"})
    if ops and not real_flow:
        # Dedicated real flows fuse the normalization into their final
        # store (listed per-op above); never a separate pass.
        ops.append({"op": "normalization epilogue (scale)",
                    "in_bytes": cplx, "out_bytes": cplx,
                    "kind": "analytic estimate",
                    "note": "skipped at runtime when scale == 1.0"})
    out["real_compat_ops"] = ops
    return out


def measure_phases(plan, x, trials: int = 30, warmup: int = 5) -> dict:
    """Fresh-measured phase breakdown of ``execute_fft_plan`` (device).

    Sync-per-phase; returned dict is labelled fresh measured and must
    never be mixed with headline protocol medians.
    """
    import torch
    from .entry import execute_fft_plan

    led = PhaseLedger(sync=True)
    tok = ACTIVE.set(led)
    walls: list[float] = []
    try:
        for _ in range(max(1, warmup)):
            execute_fft_plan(plan, x)
            torch.cuda.synchronize()
        led.phases.clear()
        for _ in range(max(1, trials)):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            execute_fft_plan(plan, x)
            torch.cuda.synchronize()
            walls.append((time.perf_counter() - t0) * 1e3)
    finally:
        ACTIVE.reset(tok)
    out = led.summary()
    n_trials = max(1, trials)
    out["trials"] = n_trials
    out["trial_wall_p50_ms"] = statistics.median(walls)
    out["trial_wall_mean_ms"] = statistics.fmean(walls)
    out["phase_sum_mean_ms"] = sum(
        v["mean_ms"] for v in out["phases"].values())
    out["inter_phase_host_overhead_ms"] = (
        out["trial_wall_mean_ms"] - out["phase_sum_mean_ms"])
    return out


def batch_fill_analysis(cases: list[dict],
                        n_values: list[int] | None = None) -> list[dict]:
    """Per-point cost vs batch for same (N, semantic) — observable
    fill/degradation from release summary medians (no counters).

    ``poor_batch_fill`` fires when the selector's per-point cost at the
    largest B grows >1.5x vs the smallest B (parallel fill getting
    worse with batch), with >=3 batch points.
    """
    groups: dict[tuple, dict] = {}
    for c in cases:
        if c.get("selector_ms") is None or c.get("vendor_ms") is None:
            continue
        if n_values and c["N"] not in n_values:
            continue
        groups.setdefault((c["N"], c["semantic"]), {})[c["batch"]] = c
    out = []
    for (n, semantic), by_b in sorted(groups.items()):
        if len(by_b) < 3:
            continue
        pts = []
        for b in sorted(by_b):
            c = by_b[b]
            pts.append({
                "B": b,
                "selector_ms": c["selector_ms"],
                "vendor_ms": c["vendor_ms"],
                "selector_us_per_point": c["selector_ms"] * 1e3 / (n * b),
                "vendor_us_per_point": c["vendor_ms"] * 1e3 / (n * b),
            })
        base = pts[0]["selector_us_per_point"]
        growth = (pts[-1]["selector_us_per_point"] / base
                  if base else None)
        vbase = pts[0]["vendor_us_per_point"]
        vgrowth = (pts[-1]["vendor_us_per_point"] / vbase
                   if vbase else None)
        out.append({
            "N": n, "semantic": semantic, "kind": KIND_SUMMARY,
            "points": pts,
            "selector_per_point_growth_smallest_to_largest_B": growth,
            "vendor_per_point_growth_smallest_to_largest_B": vgrowth,
            "poor_batch_fill": bool(growth is not None and growth > 1.5),
            "conclusion_scope": ("per-point wall time from summary "
                                 "medians; fill proxy only — no "
                                 "worker/wave hardware counters exist "
                                 "on this stack"),
        })
    return out
