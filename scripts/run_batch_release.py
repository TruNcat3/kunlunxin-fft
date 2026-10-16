#!/usr/bin/env python3
"""Batch-release focus harness (issue #3 P0 acceptance + P1.1 metadata).

Four focus shapes -- N in {1024, 2048} x B in {16, 64}, DIT fp32 forward --
timed under one protocol across candidates:

    baseline_m0_rank1_loop   literal M0: B rank-1 calls in host sequence
    incumbent_m4_folded      batch radix-2 folded control (release-matrix
                             control semantics: executed metadata says so)
    m4_spread                batch radix-2 spread (Ud=B on grid axis 1)
    m2_batch                 new batch-aware M2 (spread)
    m5_batch                 new batch-aware M5 radix-4 (spread)
    selected                 frozen selector, plan-reuse (execute_fft_plan)
    vendor                   torch.fft.fft, re-probed this round
    m2_packet_td1/td2        P2 canary extras, N=1024 B=16 only

Protocol (mirrors run_release_matrix, kept protocol-compatible):

* 3 independent process sessions (parent spawns one child per session);
* correctness-first: every candidate must pass max_rel_error < 1e-4 before
  any timing is recorded;
* randomized interleaving of candidate order inside each trial round;
* plan-reuse timing for ``selected`` (plan_fft once, execute_fft_plan per
  call); one-shot plan cost recorded separately per shape;
* api wall time (dispatch+alloc+layout+sync), kernel/device time not
  collected on this stack;
* all trials kept in raw_session{S}.jsonl; medians rebuilt offline.

P1.1 execution metadata on every raw row: profile hash, candidate id /
processing unit / realization, N / batch / direction / dtype / layout,
effective Us/Ts/Ud/Td (``not-lowered`` or a real fixed value -- never an
aspirational name), logical groups vs physical execution groups, launch
count vs materialized GM boundaries (analytical estimate, flagged),
tile / block / rows / worker packet count, error and timing protocol.

Output (results/p800/batch_release/, immutable once written):

    manifest.json  raw_session{S}.jsonl  summary.json  summary.csv
    server_report.md

Usage::

    CUDA_VISIBLE_DEVICES=6 python scripts/run_batch_release.py \
        --sessions 3 --trials 30 --out results/p800/batch_release
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPTS)

import run_release_matrix as relm  # noqa: E402  (import-safe module)

def _parse_shapes(spec):
    """\"1024x16,2048x64\" -> [(1024, 16), ...]; None/'' -> None."""
    if not spec:
        return None
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        n_s, b_s = part.lower().split("x")
        n, b = int(n_s), int(b_s)
        if n <= 0 or (n & (n - 1)) or b <= 0:
            raise SystemExit(f"bad --shapes entry {part!r} "
                             "(N must be a power of two, B > 0)")
        out.append((n, b))
    if not out:
        raise SystemExit("--shapes given but empty")
    return out


PROTOCOL_ID = (
    "batch-release-v1:focus:plan-reuse:interleave:correctness-first:3sess:p11"
)

SHAPES = [
    (1024, 16),
    (1024, 64),
    (2048, 16),
    (2048, 64),
]
CANARY_SHAPE = (1024, 16)
DIRECTION = "dit"
from src.xpu.config import DesignSpace  # noqa: E402  (single design space)

_space = DesignSpace.load()
TILE, ROWS = _space.matrix_default or (64, 8)

CANDIDATES = [
    "baseline_m0_rank1_loop",
    "incumbent_m4_folded",
    "m4_spread",
    "m2_batch",
    "m5_batch",
    "selected",
    "vendor",
]
CANARY_CANDIDATES = ["m2_packet_td1", "m2_packet_td2"]

METRIC_COLS = ["err", "correct", "status"]


def _finite_positive(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v) and v > 0


def _log2(n: int) -> int:
    return n.bit_length() - 1


# --------------------------------------------------------------------------
# P1.1 execution metadata (effective axes, groups, launches, GM estimates)
# --------------------------------------------------------------------------
def _m2_launches(n: int) -> int:
    # bitrev pass + matrix group + M0-style tail stages
    return 2 + _log2(n) - _log2(TILE)


def _m5_launches(n: int) -> int:
    # radix-4 spans with digit reversal folded into the first span; the odd
    # radix-2 tail (odd log2(N)) is counted inside ceil(log2(N)/2)
    return (_log2(n) + 1) // 2


def _gm_bytes(launches, n, batch):
    if launches is None or isinstance(launches, str):
        return "unknown", "unknown"
    # complex64: read + write of (B, N) per launch -- analytical estimate,
    # no hardware counters on this stack (issue: mark as estimate).
    return launches * 2 * n * batch * 8, "analytical-estimate"


def _exec_meta(name: str, n: int, batch: int, block: int):
    """P1.1 axes + group/launch/GM metadata for one focus candidate."""
    logn = _log2(n)
    base = {
        "tile": None, "block": block, "rows": None, "worker_packets": 1,
        "logical_groups": logn,
    }
    est_table = {
        "baseline_m0_rank1_loop": dict(
            us="not-lowered", ts="not-lowered", ud="1", td="not-lowered",
            launches=batch * logn,
            note="literal M0: host sequence of B rank-1 calls; no "
                 "batch lowering on this path",
        ),
        "incumbent_m4_folded": dict(
            us="not-lowered", ts="not-lowered", ud=f"B={batch}",
            td="not-lowered", launches=logn,
            note="flattened batch grid; no resident-worker packet loop "
                 "(folded index != true Td, issue #3 P1.1)",
        ),
        "m4_spread": dict(
            us="not-lowered", ts="not-lowered", ud=f"B={batch}", td="1",
            launches=logn, note="batch on grid axis 1 (data-space)",
        ),
        "m2_batch": dict(
            us=f"fixed:local-matrix(tile={TILE},rows={ROWS})",
            ts="fixed:fused-in-register", ud=f"B={batch}", td="1",
            launches=_m2_launches(n), tile=TILE, rows=ROWS,
            worker_packets=1,
            note="local stage group resident in registers; whole FFT not "
                 "on-chip; Us/Ts fixed, not searched",
        ),
        "m5_batch": dict(
            us="not-lowered", ts="not-lowered", ud=f"B={batch}", td="1",
            launches=_m5_launches(n),
            note="global-stage radix-4 spans; digit reversal folded into "
                 "first span",
        ),
        "m2_packet_td1": dict(
            us=f"fixed:local-matrix(tile={TILE},rows={ROWS})",
            ts="fixed:fused-in-register", ud=f"B={batch}", td="1",
            launches=_m2_launches(n), tile=TILE, rows=ROWS,
            worker_packets=1,
            note="packet probe degenerate (packets=1); interleaved layout, "
                 "2-dot fused pair-product",
        ),
        "m2_packet_td2": dict(
            us=f"fixed:local-matrix(tile={TILE},rows={ROWS})",
            ts="fixed:fused-in-register", ud=f"B={batch // 2}",
            td="2", launches=_m2_launches(n), tile=TILE, rows=ROWS,
            worker_packets=2,
            note="true worker packet loop (static_range; tables loaded "
                 "once before loop, resident across iterations); DIT only",
        ),
        "vendor": dict(
            us="external", ts="external", ud="external", td="external",
            launches=None, note="vendor implementation; internals unknown",
        ),
    }
    if name in est_table:
        est = est_table[name]
    elif name.startswith("m2_t"):
        est = dict(est_table["m2_batch"])
        tile = int(name.split("_t", 1)[1].split("_", 1)[0])
        rows = int(name.split("_rows", 1)[1])
        est.update(
            tile=tile, rows=rows,
            us=f"fixed:local-matrix(tile={tile},rows={rows})",
            launches=2 + logn - _log2(tile),
            note="cost-model bounded-search top-K variant (P1.3); "
                 "measured under the same guard as m2_batch")
    else:
        raise KeyError(f"no execution metadata for candidate {name!r}")
    out = dict(base)
    out.update(est)
    launches = out.pop("launches")
    out["physical_exec_groups"] = (
        launches if launches is not None else "unknown")
    out["launch_count"] = launches if launches is not None else "unknown"
    out["gm_boundaries_est"] = (
        launches if launches is not None else "unknown")
    # Each m0 launch moves one rank-1 row (N points), not the full (B, N):
    # the host loop's launches are already per-row, so do not scale by B
    # again (corrects an earlier overcount by factor B).
    gm_batch = 1 if name == "baseline_m0_rank1_loop" else batch
    gb, kind = _gm_bytes(launches, n, gm_batch)
    out["gm_bytes_est"] = gb
    out["gm_bytes_kind"] = kind
    out["notes"] = out.pop("note")
    return out


def _axes_for_plan(flow_realization, params, batch):
    """Conservative Us/Ts/Ud/Td for whatever the frozen table selected.

    Defaults are ``not-lowered``: an axis is only reported with a real
    value when this repo demonstrably lowers it (never aspirational).
    """
    p = params or {}
    variant = p.get("variant")
    us = "not-lowered"
    ts = "not-lowered"
    note = "from frozen table entry"
    if flow_realization == "matrix-resident":
        us = f"fixed:local-matrix(tile={p.get('tile', TILE)}," \
             f"rows={p.get('rows', ROWS)})"
        ts = "fixed:fused-in-register"
        note = "local group resident; whole FFT not on-chip"
    elif flow_realization == "grouped":
        us = "fixed:execution-group"
        note = "group boundaries materialized to GM"
    ud = f"B={batch}" if batch > 1 else "1"
    if batch > 1 and variant == "folded":
        td = "not-lowered"
        note = "flattened batch index; no resident-worker packet loop"
    else:
        td = "1"
    return {"us": us, "ts": ts, "ud": ud, "td": td, "notes": note}


def _row_meta(name, n, batch, block, direction, profile_hash, protocol):
    """P1.1 metadata block shared by every raw row."""
    dtype = "complex64"
    layout = "(B,N) C-contiguous complex; kernels split re/im (SOA)"
    if name == "selected":
        return {}  # filled by caller with plan fields
    if name == "vendor":
        meta = _exec_meta(name, n, batch, block)
    else:
        meta = _exec_meta(name, n, batch, block)
    meta.update({
        "dtype": dtype, "layout": layout,
        "profile_hash": profile_hash, "protocol_id": protocol,
        "timed_scope": "api wall time (dispatch+alloc+layout+sync), ms",
    })
    return meta


# --------------------------------------------------------------------------
# candidate construction
# --------------------------------------------------------------------------
def _m0_rank1_loop(x, direction, block):
    import torch
    from src.xpu import runner
    fn = runner.fft_dit if direction == "dit" else runner.fft_dif
    rows = [fn(x[i].contiguous(), block=block) for i in range(x.shape[0])]
    return torch.stack(rows, dim=0)


def _build_candidates(x, n, batch, direction, block, table, ph,
                      protocol=PROTOCOL_ID, full_space=False):
    """name -> (callable, attribution dict incl. P1.1 axes)."""
    from src.xpu.entry import execute_fft_plan, plan_fft
    from src.xpu import runner

    def _meta(name, extra=None):
        m = _row_meta(name, n, batch, block, direction, ph, protocol)
        m.update(extra or {})
        return m

    plan = plan_fft(direction, n, batch=batch, profile_hash=ph, table=table)
    _batch_fn = runner.fft_dit_batch if direction == "dit" \
        else runner.fft_dif_batch
    _m2_fn = runner.fft_dit_matrix_batch if direction == "dit" \
        else runner.fft_dif_matrix_batch
    _m5_fn = runner.fft_dit_radix4_batch if direction == "dit" \
        else runner.fft_dif_radix4_batch
    inc = {
        "selection_source": "baseline-control",
        "executed": _batch_fn.__name__, "block": block,
        "flow_id": "batch-folded",
        "flow_realization": "global-stage", "processing_unit": "radix2-scalar",
        "params": {"variant": "folded", "batch": batch},
        "fallback": False, "reason": "batch radix-2 folded control",
    }
    cands = {
        "baseline_m0_rank1_loop": (
            (lambda: _m0_rank1_loop(x, direction, block)),
            dict(_meta("baseline_m0_rank1_loop"),
                 selection_source="baseline-literal",
                 executed="fft_dit(loop)", block=block,
                 flow_id="baseline-global-stage",
                 flow_realization="global-stage",
                 processing_unit="radix2-scalar",
                 params={"batch": batch, "loop": "host"},
                 fallback=False,
                 reason="literal M0 at batch shapes (host row loop)")),
        "incumbent_m4_folded": (
            (lambda: _batch_fn(x, block=block, variant="folded")),
            dict(_meta("incumbent_m4_folded"), **inc)),
        "m4_spread": (
            (lambda: _batch_fn(x, block=block, variant="spread")),
            dict(_meta("m4_spread"),
                 selection_source="focus-candidate",
                 executed=_batch_fn.__name__, block=block,
                 flow_id="batch-spread",
                 flow_realization="global-stage",
                 processing_unit="radix2-scalar",
                 params={"variant": "spread", "batch": batch},
                 fallback=False, reason="M4 spread (Ud=B)")),
        "m2_batch": (
            (lambda: _m2_fn(x, block=block, tile=TILE, rows=ROWS,
                            variant="spread")),
            dict(_meta("m2_batch"),
                 selection_source="focus-candidate",
                 executed=_m2_fn.__name__, block=block,
                 flow_id="local-fused-tile",
                 flow_realization="matrix-resident",
                 processing_unit="dense-dot-fft64",
                 params={"variant": "spread", "tile": TILE, "rows": ROWS,
                         "batch": batch},
                 fallback=False, reason="issue #3 P0.1 batch-aware M2")),
        "m5_batch": (
            (lambda: _m5_fn(x, block=block, variant="spread")),
            dict(_meta("m5_batch"),
                 selection_source="focus-candidate",
                 executed=_m5_fn.__name__, block=block,
                 flow_id="batch-spread",
                 flow_realization="global-stage",
                 processing_unit="radix4-scalar",
                 params={"variant": "spread", "batch": batch},
                 fallback=False, reason="issue #3 P0.2 batch-aware M5")),
        "selected": (
            (lambda: execute_fft_plan(plan, x)),
            dict(_axes_for_plan(plan.flow_realization, plan.params, batch),
                 dtype="complex64",
                 layout="(B,N) C-contiguous complex; kernels split re/im "
                        "(SOA)",
                 profile_hash=ph, protocol_id=protocol,
                 timed_scope="api wall time (dispatch+alloc+layout+sync), "
                             "ms",
                 tile=plan.params.get("tile") if isinstance(plan.params, dict)
                 else None,
                 block=plan.block,
                 rows=plan.params.get("rows") if isinstance(plan.params, dict)
                 else None,
                 worker_packets=2 if (isinstance(plan.params, dict)
                                      and plan.params.get("packet") == 2)
                 else 1,
                 logical_groups=_log2(n),
                 physical_exec_groups="from-selected",
                 launch_count="from-selected",
                 gm_boundaries_est="from-selected",
                 gm_bytes_est="unknown", gm_bytes_kind="unknown",
                 selection_source=plan.selection_source,
                 executed=plan.executed, flow_id=plan.flow_id,
                 flow_realization=plan.flow_realization,
                 processing_unit=plan.processing_unit, params=plan.params,
                 fallback=plan.fallback, reason=plan.reason)),
        "vendor": (
            (lambda: _vendor_call(x)),
            dict(_meta("vendor"),
                 selection_source="external-baseline",
                 executed="torch.fft.fft", block=None,
                 flow_realization=None, processing_unit=None,
                 params={"dim": -1}, fallback=False,
                 reason="matched (B,N) vendor call, probed this round")),
    }

    # P1.3 bounded search: model screens tile x rows, measurement decides.
    # Any name starting with m2_t is soft-gated in run_shape (a failing
    # experimental variant is recorded and dropped, never fatal).
    screen_failure = None
    try:
        from src.xpu import cost_model as _cm
        if full_space:
            tops = [dict(s) for s in _cm.m2_space()]
        else:
            tops = _cm.bounded_search("m2_batch", n, batch, k=3)
    except Exception as e:  # noqa: BLE001  (screening must not kill run)
        tops = []
        screen_failure = f"{type(e).__name__}: {e}"
        if full_space:
            raise SystemExit(
                f"[{n}x{batch}] full-space enumeration failed: "
                f"{screen_failure} (holdout run requires the published "
                "space; aborting instead of silently shrinking it)")
    for prm in tops:
        pt, pr = int(prm["tile"]), int(prm["rows"])
        if (pt, pr) == (TILE, ROWS):
            continue  # default m2_batch already covers this configuration
        name = f"m2_t{pt}_rows{pr}"
        cands[name] = (
            (lambda _p=(pt, pr): _m2_fn(x, block=block, tile=_p[0],
                                        rows=_p[1], variant="spread")),
            dict(_meta(name),
                 selection_source="cost-model-topk",
                 executed=_m2_fn.__name__, block=block,
                 flow_id="local-fused-tile",
                 flow_realization="matrix-resident",
                 processing_unit="dense-dot-fft64",
                 params={"variant": "spread", "tile": pt, "rows": pr,
                         "batch": batch},
                 fallback=False,
                 reason=("full published m2 space (holdout, issue #3 fix)"
                         if full_space else
                         "cost-model bounded-search top-K "
                         "(issue #3 P1.3)")))
    return cands, plan, screen_failure


def _vendor_call(x):
    import torch
    return torch.fft.fft(x, dim=-1)


def _canary_candidates(x, n, batch, direction, block):
    from src.xpu import runner
    out = {}
    for p in (1, 2):
        name = f"m2_packet_td{p}"
        out[name] = (
            (lambda _p=p: runner.fft_dit_matrix_packet_batch(
                x, block=block, tile=TILE, rows=ROWS, packets=_p)),
            dict(_row_meta(name, n, batch, block, direction, "local",
                           PROTOCOL_ID),
                 selection_source="p2-canary",
                 executed="fft_dit_matrix_packet_batch", block=block,
                 flow_id="local-fused-tile",
                 flow_realization="matrix-resident",
                 processing_unit="dense-dot-fft64",
                 params={"packets": p, "tile": TILE, "rows": ROWS,
                         "batch": batch},
                 fallback=False,
                 reason=f"P2 Td probe (packets={p})"))
    return out


# --------------------------------------------------------------------------
# per-shape run
# --------------------------------------------------------------------------
def run_shape(session, n, batch, direction, trials, seed, table, raw,
              warmup, protocol=PROTOCOL_ID, full_space=False):
    import torch
    from src.xpu.config import DeviceProfile

    tag = f"N{n}B{batch}_{direction}"
    x = relm._make_input(n, batch, seed)
    ref = relm._ref_np(n, batch, seed)
    ph = DeviceProfile.load().hash()
    entry = relm._entry_for(table, n, batch, direction)
    table_entry_present = entry is not None
    if entry is None:
        # Focus shapes (1024/2048 x 16/64) may be absent from the frozen
        # table until their batch entries clear the 2% guard. plan_fft
        # abstains to M0 there (honestly recorded); we only borrow the
        # same-N B=1 incumbent block as the baseline candidate.
        entry = relm._entry_for(table, n, 1, direction)
        if entry is not None:
            print(f"  [{tag}] no table entry for batch={batch}; "
                  f"using B=1 incumbent_block baseline", flush=True)
        else:
            print(f"  [{tag}] no table entry at all; selected -> M0 "
                  f"fallback", flush=True)
    inc_block = int(entry.get("incumbent_block") or entry.get("block") or 1024)

    cands, plan, screen_failure = _build_candidates(
        x, n, batch, direction, inc_block, table, ph, protocol=protocol,
        full_space=full_space)
    # The Td=2 packet canary is DIT-only this round (no DIF interleave);
    # holdout runs enumerate the m2 space and skip the packet canary.
    if not full_space and (n, batch) == CANARY_SHAPE and direction == "dit":
        cands.update(_canary_candidates(x, n, batch, direction, inc_block))

    # plan creation cost (diagnostic; not part of interleaved timing)
    from src.xpu.entry import plan_fft
    t0 = time.perf_counter()
    plan_again = plan_fft(direction, n, batch=batch, profile_hash=ph,
                          table=table)
    plan_create_ms = (time.perf_counter() - t0) * 1e3
    del plan_again  # keep the first plan for execute_fft_plan timing

    # --- correctness gates (before any timing) --------------------------
    # Core candidates are hard gates; m2_t* bounded-search variants are
    # experimental and soft-gate (recorded failure, dropped from timing).
    errs = {}
    extras_failed = []
    if screen_failure:
        extras_failed.append({
            "candidate": "m2_t* (space screening)", "status": "screen-failed",
            "detail": screen_failure})
    for name in list(cands):
        fn, _d = cands[name]
        is_extra = name.startswith("m2_t")
        try:
            y = fn()
            err = relm._err_vs_np(y, ref)
        except Exception as e:  # noqa: BLE001
            if is_extra:
                extras_failed.append(
                    {"candidate": name, "status": "error",
                     "detail": f"{type(e).__name__}: {e}"})
                del cands[name]
                continue
            raise SystemExit(f"[{tag}] {name} raised: {type(e).__name__}: {e}")
        if not relm._err_gate(err):
            if is_extra:
                extras_failed.append(
                    {"candidate": name, "status": "wrong-value",
                     "max_rel_error": err})
                del cands[name]
                continue
            raise SystemExit(f"[{tag}] {name} correctness failed: {err}")
        errs[name] = err

    # --- interleaved plan-reuse timing ----------------------------------
    rng = random.Random(seed * 1000003 + session)
    samples = {k: [] for k in cands}
    relm._interleave_warmup([fn for fn, _ in cands.values()], warmup)
    torch.cuda.synchronize()
    rows = []
    order_names = list(cands)
    meta_by_name = {name: dict(desc) for name, (_f, desc) in cands.items()}
    for t in range(trials):
        order = list(order_names)
        rng.shuffle(order)
        for pos, name in enumerate(order):
            fn, desc = cands[name]
            api = relm._timed_call(fn, warmup=0)
            samples[name].append(api)
            row = {
                "session": session, "trial": t, "order": pos,
                "candidate": name, "N": n, "batch": batch,
                "direction": direction, "label": tag,
                "api_ms": api, "correct": True, "status": "ok",
                "err": errs[name],
            }
            row.update(meta_by_name[name])
            rows.append(row)
    with open(raw, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    summary = {
        "label": tag, "N": n, "batch": batch, "direction": direction,
        "incumbent_block": inc_block,
        "table_entry_present": table_entry_present,
        "extras_failed": extras_failed,
        "plan_create_ms": plan_create_ms,
        "plan_executed": plan.executed,
        "plan_block": plan.block,
        "plan_flow_realization": plan.flow_realization,
        "plan_processing_unit": plan.processing_unit,
        "plan_params": plan.params,
        "plan_fallback": plan.fallback,
        "plan_selection_source": plan.selection_source,
        "plan_reason": plan.reason,
        "candidates": {
            name: {
                "median_ms": relm._median(samples[name]),
                "err": errs[name],
                "executed": meta_by_name[name].get("executed"),
                "block": meta_by_name[name].get("block"),
                "flow_realization": meta_by_name[name].get("flow_realization"),
                "processing_unit": meta_by_name[name].get("processing_unit"),
                "us": meta_by_name[name].get("us"),
                "ts": meta_by_name[name].get("ts"),
                "ud": meta_by_name[name].get("ud"),
                "td": meta_by_name[name].get("td"),
                "launch_count": meta_by_name[name].get("launch_count"),
                "gm_bytes_est": meta_by_name[name].get("gm_bytes_est"),
                "gm_bytes_kind": meta_by_name[name].get("gm_bytes_kind"),
                "worker_packets": meta_by_name[name].get("worker_packets"),
                "stdev_ms": (statistics.pstdev(samples[name])
                             if len(samples[name]) > 1 else 0.0),
                "trials": len(samples[name]),
            }
            for name in cands
        },
    }
    for name, c in summary["candidates"].items():
        src_desc = meta_by_name[name]
        for k in ("flow_id", "params", "selection_source", "fallback",
                  "reason"):
            c.setdefault(k, src_desc.get(k))
    inc_ms = summary["candidates"]["incumbent_m4_folded"]["median_ms"]
    for name, c in summary["candidates"].items():
        c["speedup_over_incumbent"] = (
            inc_ms / c["median_ms"] if c["median_ms"] else None)
    print(f"  [{tag}] block={inc_block} "
          + " ".join(
              f"{n_}={summary['candidates'][n_]['median_ms']:.4f}ms"
              f"(x{summary['candidates'][n_]['speedup_over_incumbent']:.2f})"
              for n_ in CANDIDATES
              if summary["candidates"][n_]["median_ms"])
          + " plan={:.4f}ms".format(plan_create_ms), flush=True)
    return summary


# --------------------------------------------------------------------------
# child / parent / aggregate
# --------------------------------------------------------------------------
def run_child(args):
    from src.xpu.config import DeviceProfile
    from src.xpu.selector import load_table

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    raw = os.path.join(out_dir, f"raw_session{args.session}.jsonl")
    open(raw, "w").close()
    table = load_table()
    seed = args.seed + args.session * 7919
    results = []
    shapes = _parse_shapes(args.shapes) or SHAPES
    protocol = PROTOCOL_ID
    if args.full_space:
        protocol += ":m2-fullspace-holdout"
    print(f"session {args.session}: seed={seed} protocol={protocol}",
          flush=True)
    for n, b in shapes:
        for direction in ("dit", "dif"):
            results.append(run_shape(args.session, n, b, direction,
                                     args.trials, seed, table, raw,
                                     args.warmup, protocol=protocol,
                                     full_space=args.full_space))
    with open(os.path.join(out_dir,
                           f"summary_session{args.session}.json"), "w") as f:
        json.dump({"session": args.session, "shapes": results}, f, indent=2)
        f.write("\n")
    print(f"session {args.session}: wrote {raw}", flush=True)


def _vendor_probe():
    """Vendor support re-probed in this environment (no hardcoded limits)."""
    import torch
    out = []
    for n, b in ((16, 1), (1024, 16), (2048, 64), (4096, 16), (16384, 1),
                 (65536, 1), (1048576, 1)):
        try:
            x = torch.complex(
                torch.randn(b, n, dtype=torch.float32, device="cuda"),
                torch.randn(b, n, dtype=torch.float32, device="cuda"))
            y = torch.fft.fft(x, dim=-1)
            torch.cuda.synchronize()
            del y
            out.append({"N": n, "batch": b, "status": "ok"})
        except Exception as e:  # noqa: BLE001
            out.append({"N": n, "batch": b,
                        "status": f"unsupported: {type(e).__name__}: {e}"[:180]})
    return out


def _manifest(seed, warmup, trials, sessions, shapes=None,
              full_space=False):
    import torch
    from src.xpu.config import DeviceProfile
    from src.xpu.envcheck import collect as _collect_env
    from src.xpu.envcheck import env_hash as _current_env_hash
    p = DeviceProfile.load()
    table_path = os.path.join(ROOT, "config", "selector_table.json")
    toolchain = f"torch {torch.__version__} + torch_xmlir"
    try:
        import torch_xmlir
        vd = torch_xmlir.get_version_dict()
        toolchain += (f" (XPYTORCH {vd.get('XPYTORCH', '?')}, "
                      f"XRE {vd.get('XRE', '?')})")
    except Exception:
        pass
    shapes = shapes or SHAPES
    protocol = PROTOCOL_ID
    if full_space:
        protocol += ":m2-fullspace-holdout"
    shape_arg = ("" if (not full_space and shapes == SHAPES) else
                 "--shapes "
                 + ",".join(f"{n}x{b}" for n, b in shapes)
                 + (" --full-space " if full_space else " "))
    repro = (
        "CUDA_VISIBLE_DEVICES=<healthy card> timeout -s KILL 3600 "
        "python scripts/run_batch_release.py --sessions "
        f"{sessions} --trials {trials} --warmup {warmup} "
        f"{shape_arg}--out results/p800/batch_release"
    )
    return {
        "focus": ("issue #3 P1.3 full-space holdout validation"
                  if full_space else
                  "issue #3 P0 acceptance + P1.1 execution metadata"),
        "protocol_id": protocol,
        "full_space": bool(full_space),
        "commit": relm._git("rev-parse", "HEAD"),
        "dirty": bool(relm._git("status", "--porcelain")),
        "device": p.device,
        "driver": p.driver,
        "device_index": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "toolchain": toolchain,
        "profile_hash": p.hash(),
        "config_hash": p.hash(),
        "env_hash": _current_env_hash(),
        "env": _collect_env(),
        "selector_table_sha256": relm._sha256(table_path),
        "sessions": sessions,
        "warmup": warmup,
        "trials": trials,
        "seed_base": seed,
        "shapes": [f"N{n}B{b}:{d}" for n, b in shapes
                   for d in ("dit", "dif")],
        "candidates": CANDIDATES + (CANARY_CANDIDATES
                                    if not full_space else [])
        + [("m2_t*_rows* (full published space, holdout: 9 points)"
            if full_space else
            "m2_t*_rows* (cost-model bounded search, per-shape, "
            "soft-gated)")],
        "canary_shape": f"N{CANARY_SHAPE[0]}B{CANARY_SHAPE[1]}",
        "timing_scope": "api wall time (dispatch+alloc+layout+sync), ms",
        "kernel_time_status": (
            "not collected: this stack exposes reliable sync API wall time; "
            "device/kernel time not gathered this round"),
        "error_threshold": 1e-4,
        "error_definition": (
            "max_rel_error = max(max(|Re(got-ref)|, |Im(got-ref)|)) "
            "/ max(|ref|); numpy complex128 reference"),
        "gm_bytes_kind": (
            "analytical estimate from launch counts x read+write of (B,N) "
            "complex64; no hardware counters -- not a measured bandwidth"),
        "selection_semantics": (
            "frozen table; plan-reuse timings use execute_fft_plan after "
            "one plan_fft; 2% promotion gate applies to candidates vs the "
            "incumbent control in a separate gate step"),
        "vendor_probe": _vendor_probe(),
        "repro_command": repro,
    }


def _load_raw_trials(out_dir):
    """All trial timings per (label, candidate) from raw session logs."""
    trials = {}
    s = 0
    while True:
        path = os.path.join(out_dir, f"raw_session{s}.jsonl")
        if not os.path.exists(path):
            break
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                trials.setdefault((row["label"], row["candidate"]), []).append(
                    float(row["api_ms"]))
        s += 1
    return trials


def _aggregate(out_dir, args):
    with open(os.path.join(out_dir, "manifest.json")) as f:
        man = json.load(f)
    expected_sessions = int(man.get("sessions") or 0)
    trials_per_session = int(man.get("trials") or 0)
    per_session = []
    s = 0
    while True:
        p = os.path.join(out_dir, f"summary_session{s}.json")
        if not os.path.exists(p):
            break
        with open(p) as f:
            per_session.append(json.load(f))
        s += 1
    if not per_session:
        raise SystemExit(f"no summary_session*.json under {out_dir}")
    raw_trials = _load_raw_trials(out_dir)
    shapes = {}
    for ss in per_session:
        for r in ss["shapes"]:
            shapes.setdefault(r["label"], []).append(r)
    agg = []
    for label, rs in sorted(shapes.items()):
        # union across sessions: a candidate present only in a later
        # session must still surface (with incomplete evidence flags)
        names = list(dict.fromkeys(
            name for r in rs for name in r["candidates"]))
        entry = {
            "label": label, "N": rs[0]["N"], "batch": rs[0]["batch"],
            "direction": rs[0]["direction"],
            "incumbent_block": rs[0]["incumbent_block"],
            "sessions": len(rs),
            "plan_create_ms": relm._median([r["plan_create_ms"] for r in rs]),
            "selected_executed": rs[0]["plan_executed"],
            "selected_block": rs[0]["plan_block"],
            "flow_realization": rs[0]["plan_flow_realization"],
            "processing_unit": rs[0]["plan_processing_unit"],
            "fallback": rs[0]["plan_fallback"],
            "selection_source": rs[0]["plan_selection_source"],
            "table_entry_present": rs[0].get("table_entry_present"),
            "extras_failed": [f for r in rs
                              for f in (r.get("extras_failed") or [])],
            "candidates": {},
        }
        all_correct = True
        for name in names:
            meds = [r["candidates"][name]["median_ms"] for r in rs
                    if name in r["candidates"]]
            errs = [r["candidates"][name]["err"] for r in rs
                    if name in r["candidates"]]
            meta = next((r["candidates"][name] for r in rs
                         if name in r["candidates"]), {})
            med = relm._median(meds)
            inc_meds = [r["candidates"]["incumbent_m4_folded"]["median_ms"]
                        for r in rs
                        if "incumbent_m4_folded" in r["candidates"]]
            inc = relm._median(inc_meds) if inc_meds else None
            ratio = (inc / med) if inc and med else None
            if errs and not all(relm._err_gate(e) for e in errs):
                all_correct = False
            verdict = None
            if ratio is not None:
                verdict = ("speedup" if ratio >= 1.02
                           else "regression" if ratio <= 0.98 else "parity")
            all_trials = raw_trials.get((label, name), [])
            # Promotion-integrity metadata (issue #3 review fix): the
            # builder refuses anything whose session/trial evidence is not
            # complete; `complete` makes that checkable in the artifact.
            n_sess = len(meds)
            want_trials = expected_sessions * trials_per_session
            complete = bool(
                expected_sessions and trials_per_session
                and n_sess == expected_sessions
                and len(all_trials) == want_trials
                and all(_finite_positive(v) for v in meds)
                and all(isinstance(e, (int, float)) and math.isfinite(e)
                        for e in errs)
                and all(relm._err_gate(e) for e in errs))
            entry["candidates"][name] = dict(
                meta, median_ms=med, sessions=n_sess,
                sessions_expected=expected_sessions,
                n_trials_want=want_trials, complete=complete,
                stdev_ms=(statistics.pstdev(meds) if len(meds) > 1
                          else 0.0),
                session_medians_ms=meds,
                n_trials=len(all_trials),
                stdev_ms_all_trials=(
                    statistics.pstdev(all_trials)
                    if len(all_trials) > 1 else None),
                err_max=(max(errs) if errs else None),
                speedup_over_incumbent=ratio, verdict=verdict)
        entry["all_correct"] = all_correct
        entry["complete"] = all(c.get("complete")
                                for c in entry["candidates"].values())
        agg.append(entry)
    missing_sessions = expected_sessions - len(per_session)
    if missing_sessions > 0:
        print(f"WARNING: {out_dir}: {len(per_session)}/{expected_sessions} "
              "sessions aggregated -- incomplete run, the selector table "
              "builder will refuse it", flush=True)
    summary = {"manifest": man, "sessions": len(per_session),
               "sessions_expected": expected_sessions,
               "sessions_missing": max(0, missing_sessions),
               "shapes": agg}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    _write_csv(out_dir, agg)
    _write_report(out_dir, summary)
    print(f"aggregated {len(agg)} shapes from {len(per_session)} sessions "
          f"-> {out_dir}/summary.json")


def _write_csv(out_dir, agg):
    path = os.path.join(out_dir, "summary.csv")
    names = CANDIDATES + CANARY_CANDIDATES
    cols = (["label", "N", "batch", "direction", "incumbent_block",
             "sessions", "plan_create_ms", "selected_executed",
             "selected_block", "fallback", "selection_source", "all_correct"]
            + [f"{n}_ms" for n in names]
            + [f"{n}_speedup_vs_inc" for n in names]
            + [f"{n}_verdict" for n in names])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in agg:
            row = {k: r.get(k) for k in cols if k in r}
            for n in names:
                c = r["candidates"].get(n, {})
                row[f"{n}_ms"] = c.get("median_ms")
                row[f"{n}_speedup_vs_inc"] = c.get("speedup_over_incumbent")
                row[f"{n}_verdict"] = c.get("verdict")
            w.writerow(row)


def _ratio(cs, ctrl, var):
    """control_ms / variant_ms (>1: variant faster); None-safe."""
    a = (cs.get(ctrl) or {}).get("median_ms")
    b = (cs.get(var) or {}).get("median_ms")
    if not a or not b:
        return None
    return a / b


def _ablation_lines(summary):
    """Issue #3 ablation contrasts from the same interleaved schedule.

    Ratios are control/variant (>1 means the variant is faster); every
    pair varies one axis while holding the others fixed.
    """
    out = [
        "",
        "## Ablation contrasts (median ms ratios, >1 = variant faster)",
        "",
        "- **flow / batch lowering** (radix-2 held): m0 host loop vs "
        "m4 spread (Ud=B on grid axis) and m4 folded (flattened index); "
        "the spread/folded pair isolates the Td-indexing form.",
        "- **core radix** (global-stage flow held): m4 spread radix-2 vs "
        "m5 radix-4 spans.",
        "- **local-matrix core** (batch spread held): m4 spread scalar "
        "stages vs m2 register-resident dot group.",
        "- **bounded-search delta** (same m2 flow): default tile=64 vs the "
        "fastest cost-model top-K variant measured this round.",
        "- **P2 Td canary** (dit canary shape only): m2 spread vs packet "
        "Td=1 vs packet Td=2.",
        "",
        "| label | m0/m4spr | m4spr/m5 | m4spr/m2 | m2/t2best | "
        "m2/m2pkt_td2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    def _fmt(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "-"

    for r in summary["shapes"]:
        cs = r["candidates"]
        best = None
        for name, c in cs.items():
            if name.startswith("m2_t") and c.get("median_ms"):
                if best is None or c["median_ms"] < best[1]:
                    best = (name, c["median_ms"])
        m2_default = (cs.get("m2_batch") or {}).get("median_ms")
        delta = (m2_default / best[1] if best and m2_default else None)
        canary_pair = None
        if (r["N"], r["batch"], r["direction"]) == (
                CANARY_SHAPE[0], CANARY_SHAPE[1], "dit"):
            canary_pair = _ratio(cs, "m2_batch", "m2_packet_td2")
        out.append(
            f"| {r['label']} "
            f"| {_fmt(_ratio(cs, 'baseline_m0_rank1_loop', 'm4_spread'))} "
            f"| {_fmt(_ratio(cs, 'm4_spread', 'm5_batch'))} "
            f"| {_fmt(_ratio(cs, 'm4_spread', 'm2_batch'))} "
            f"| {_fmt(delta)} "
            f"| {_fmt(canary_pair)} |")
    return out


def _write_report(out_dir, summary):
    m = summary["manifest"]
    lines = [
        "# Batch release focus report (issue #3 P0 acceptance)",
        "",
        f"commit `{m['commit'][:12]}` dirty={m['dirty']} device="
        f"{m['device']} driver={m['driver']} protocol=`{m['protocol_id']}`",
        f"profile_hash=`{m['profile_hash']}` table_sha256="
        f"`{m['selector_table_sha256'][:16]}...` sessions="
        f"{summary['sessions']} warmup={m['warmup']} trials={m['trials']} "
        f"seed={m['seed_base']}",
        "",
        "Timing = api wall time (dispatch+alloc+layout+sync), ms; median of "
        "session medians over independent process sessions. Verdict vs the "
        "incumbent radix-2 folded control: >=1.02 speedup, <=0.98 "
        "regression, else parity (matches the 2% promotion gate). GM bytes "
        "in raw rows are analytical estimates (no hardware counters).",
        "",
        "| label | N | B | blk | m0 loop | inc fold | m4 spr | m2 b | m5 b | "
        "selected | vendor | m2>=1.02? | m5>=1.02? | sel | correct |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"
        "---|---|",
    ]
    for r in summary["shapes"]:
        cs = r["candidates"]

        def _ms(n):
            c = cs.get(n)
            return f"{c['median_ms']:.4f}" if c and c.get("median_ms") else "-"

        def _v(n):
            c = cs.get(n)
            if not c or not c.get("verdict"):
                return "-"
            return f"{c['verdict']} ({c['speedup_over_incumbent']:.3f}x)"

        lines.append(
            f"| {r['label']} | {r['N']} | {r['batch']} "
            f"| {r['incumbent_block']} "
            f"| {_ms('baseline_m0_rank1_loop')} "
            f"| {_ms('incumbent_m4_folded')} {_v('incumbent_m4_folded')} "
            f"| {_ms('m4_spread')} {_v('m4_spread')} "
            f"| {_ms('m2_batch')} {_v('m2_batch')} "
            f"| {_ms('m5_batch')} {_v('m5_batch')} "
            f"| {_ms('selected')} {_v('selected')} "
            f"| {_ms('vendor')} | {_v('m2_batch')} | {_v('m5_batch')} "
            f"| {_v('selected')} | {r['all_correct']} |")
    canary = [r for r in summary["shapes"]
              if r["N"] == CANARY_SHAPE[0]
              and r["batch"] == CANARY_SHAPE[1]
              and r["direction"] == "dit"]
    if canary:
        cs = canary[0]["candidates"]
        lines += [
            "",
            "## P2 Td canary (representative shape N=1024 B=16)",
            "",
            "| candidate | median ms | speedup vs inc | td | verdict |",
            "|---|---:|---:|---|---|",
        ]
        for n in ["incumbent_m4_folded", "m2_batch", "m2_packet_td1",
                  "m2_packet_td2"]:
            c = cs.get(n, {})
            lines.append(
                f"| {n} | {c.get('median_ms', float('nan')):.4f} "
                f"| {c.get('speedup_over_incumbent', float('nan')):.3f}x "
                f"| {c.get('td')} | {c.get('verdict')} |")
        lines += [
            "",
            "`m2_packet_td2` is the true worker packet loop (tables loaded "
            "once, consumed by a static_range loop over 2 packets; DIT "
            "only; packets>2 rejected by the runner: this backend "
            "silently miscompiles programs beyond four dot operations).",
        ]
    lines += [
        "",
        "## Notes",
        "",
        "- P1.1 axes on every raw row: `us`/`ts`/`ud`/`td` are "
        "`not-lowered`, a real fixed value, or `external` -- never an "
        "aspirational name; folded batch rows carry the explicit "
        "flattened-index caveat.",
        "- `baseline_m0_rank1_loop` is literal M0 at batch shapes (host "
        "row loop); the batch incumbent control is radix-2 folded.",
        "- Vendor support was re-probed this round (see "
        "`manifest.vendor_probe`), not hardcoded from old conclusions.",
        "- 2% promotion gate: candidates are NOT promoted by this harness; "
        "promotion is a separate gated step against the incumbent.",
        "",
    ]
    lines += _ablation_lines(summary)
    with open(os.path.join(out_dir, "server_report.md"), "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=os.path.join(
        ROOT, "results", "p800", "batch_release"))
    ap.add_argument("--session", type=int, default=None,
                    help="internal: run one child session")
    ap.add_argument("--shapes", default=None,
                    help="comma NxB list (e.g. 1024x16,2048x64); default "
                    "the issue's four focus shapes")
    ap.add_argument("--full-space", action="store_true",
                    help="enumerate ALL published m2 (tile,rows) points "
                    "instead of the model top-K (holdout validation)")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.session is not None:
        run_child(args)
    else:
        run_parent(args)


def run_parent(args):
    out_dir = os.path.abspath(args.out)
    if args.aggregate_only:
        _aggregate(out_dir, args)
        return
    import torch  # noqa: F401  (device stack import before any session)
    if os.path.exists(os.path.join(out_dir, "summary.json")) \
            and not args.overwrite:
        raise SystemExit(f"{out_dir} already holds a finished run; "
                         "pass --overwrite to replace")
    os.makedirs(out_dir, exist_ok=True)
    man = _manifest(args.seed, args.warmup, args.trials, args.sessions,
                    shapes=_parse_shapes(args.shapes),
                    full_space=args.full_space)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
        f.write("\n")
    print(f"vendor probe: {man['vendor_probe']}", flush=True)
    for s in range(args.sessions):
        cmd = [sys.executable, os.path.abspath(__file__),
               "--session", str(s), "--trials", str(args.trials),
               "--warmup", str(args.warmup), "--seed", str(args.seed),
               "--out", out_dir]
        if args.shapes:
            cmd += ["--shapes", args.shapes]
        if args.full_space:
            cmd += ["--full-space"]
        env = dict(os.environ)
        env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
        print(f"=== session {s} ===", flush=True)
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    _aggregate(out_dir, args)


if __name__ == "__main__":
    main()
