#!/usr/bin/env python3
"""WP2 unified semantic release matrix (issue #6 PR-D).

One manifest drives raw rows, summary tables and the immutable
release snapshot. For every case x semantic the matrix records four
contenders under one correctness-first, interleaved protocol:

* ``incumbent`` — the M0/M4 baseline plan (``plan_fft`` abstains to M0);
* ``selector``  — the frozen production table plan;
* ``oracle``    — a bounded measured candidate set (m0/m1/m2/m4/m5);
* ``vendor``    — ``torch.fft`` with the same semantic (unsupported /
  error preserved verbatim, never enters a speedup).

Semantics: ``c2c-forward``, ``c2c-inverse``, ``r2c``, ``c2r`` (PR-B/PR-C
contracts; non-forward semantics wrap the same core through the entry
layer, so the bounded oracle measures the core set through the same
wrapper).

Timing tiers are kept strictly separate — plan creation, plan-reuse
steady API wall, one-shot latency, H2D->FFT->D2H end-to-end — and
never labelled as kernel time (no device counters on this stack).

Protocol: >= 3 independent process sessions, >= 30 trials/case,
randomized interleave within each trial round, correctness gate before
any timing, search (oracle candidates) measured in the same
confirmation sessions but attributed per candidate.

Usage::

    CUDA_VISIBLE_DEVICES=6 python3 scripts/run_semantic_matrix.py \
        --sessions 3 --trials 30 --scope main \
        --out results/p800/semantic_matrix
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xpu.manifest_contract import (  # noqa: E402
    SEMANTIC_MATRIX_PROTOCOL,
    cases_sha256,
    format_failures,
    result_sha256,
    validate_benchmark_output,
)

# The matrix locks the FFT decomposition direction (issue #8 scope:
# decomposition stability). Builders reading entry geometry source it
# from here so table rows and run code can never disagree.
DIRECTION = "dit"
PROTOCOL_ID = SEMANTIC_MATRIX_PROTOCOL
TIMING_SCOPE = ("api wall time (dispatch+alloc+layout+sync), ms; "
                "tiers: plan_creation / steady / one_shot / e2e")
SEMANTICS = ("c2c-forward", "c2c-inverse", "r2c", "c2r")
TIER_STEADY = "steady"
TIER_ONE_SHOT = "one_shot"
TIER_E2E = "e2e"
TIER_PLAN_CREATION = "plan_creation"

# Issue #8 PR-I round 2: "full" keeps the round-1 bounded oracle set;
# "winner-block" fixes the processing unit (the promoted production
# winner per cell, incl. composition cells) and sweeps ONLY the block
# (worker waves/tail geometry) -- a single-factor A/B over the same
# fixed cells. Set from --oracle-mode in main(); session children
# re-parse the same flag.
ORACLE_MODE = "full"
WINNER_BLOCK_SWEEP = (256, 512, 1024, 2048)

SCOPES = {
    "main": [(n, b) for n in (64, 128, 256, 512, 1024, 2048)
             for b in (1, 4, 16, 64)],
    "apps": [(1024, 16), (1024, 64), (2048, 16), (2048, 64),
             (1024, 256), (2048, 256)],   # OFDM + radar workload proxies
    "large": [(4096, 1), (4096, 16), (16384, 1), (16384, 16),
              (65536, 1), (65536, 16)],
    # issue #8 PR-G step 2: shapes owning the fixed-domain abstain cells
    # whose v2 bounded oracle measured clearly faster (regret >= 0.10,
    # i.e. oracle >= 1.10x the incumbent fallback), plus N2048B256 as
    # background evidence (same 2x-class gap, no gate).  Holdout rows in
    # this scope are confirmation evidence only -- never fitting input.
    "regret": [(64, 1), (512, 64), (2048, 4), (1024, 256), (2048, 256),
               (16384, 1), (16384, 16)],
    # issue #8 stress domain: the four N1024B256 cells whose high
    # regret the issue requires fixing; also the bootstrap chain's
    # default confirmation scope (WP1 step 2 -> exact-key promotion).
    "stress": [(1024, 256)],
    # issue #8 PR-H: fixed cells for the dedicated real-flow A/B --
    # packed half-length candidates vs the deployed compat entries on
    # the OFDM/radar proxies plus the (64,1) probe and the stress cell.
    "prh": [(64, 1), (1024, 16), (1024, 64), (1024, 256),
            (2048, 16), (2048, 64)],
    # issue #8 PR-I round 1 fixed axis: the PR-H cells (their entries
    # re-clear the guard on every re-measurement) plus the batch/large
    # gap cells the WP2 ledger attributed to the M4 folded core
    # (core-invoke >= 99.7% of wall; the registered M2 matrix batch
    # candidates measured 1.9-2.4x faster there).
    "pri": [(64, 1), (1024, 16), (1024, 64), (1024, 256),
            (2048, 16), (2048, 64),
            (64, 64), (128, 64), (256, 64), (2048, 256),
            (4096, 16), (4096, 64), (16384, 16),
            (65536, 16), (65536, 64)],
}


def parse_cells_spec(spec):
    """Parse ``--cells`` into ordered ``[(N, batch, [sems...]), ...]``.

    Formats per cell: ``N64B4`` (all four semantics), ``N64B4:c2c-forward``
    or ``N64B4:c2c-forward+r2c``; cells are comma-separated. Issue #9
    WP1 freezes experiment domains as exact cell+semantic sets (the
    10-cell closure domain, the 12-cell scale slice, ...) -- the matrix
    must measure exactly the declared set, not the cross product.
    """
    cells = []
    index = {}
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        head, _, sem_part = part.partition(":")
        m = re.match(r"^N(\d+)B(\d+)$", head)
        if not m:
            raise SystemExit(
                f"--cells: bad cell {part!r} (expected N<n>B<b>[:"
                "sem[+sem...]])")
        n, b = int(m.group(1)), int(m.group(2))
        if sem_part:
            sems = [x for x in sem_part.split("+") if x]
            unknown = [x for x in sems if x not in SEMANTICS]
            if unknown:
                raise SystemExit(
                    f"--cells: unknown semantic(s) {unknown} in {part!r}"
                    f" (known: {list(SEMANTICS)})")
        else:
            sems = list(SEMANTICS)
        key = (n, b)
        if key in index:
            merged = cells[index[key]][2]
            for sem in sems:
                if sem not in merged:
                    merged.append(sem)
        else:
            index[key] = len(cells)
            cells.append((n, b, list(sems)))
    if not cells:
        raise SystemExit("--cells: empty cell list")
    return cells


def _schedule(cells, sem_map=None):
    """Flattened ``[(N, batch, semantic), ...]`` measurement schedule."""
    sched = []
    for n, b in cells:
        for sem in (sem_map.get((n, b), SEMANTICS) if sem_map
                    else SEMANTICS):
            sched.append((n, b, sem))
    return sched


# Previous production selector: the frozen table as of the last commit
# BEFORE issue #8 started (v2 snapshot commit) — read from git, never
# from the working tree. Issue #8 PR-J A1: release confirmation needs a
# same-session previous-deployed contender, not a cross-day join.
PREVIOUS_SELECTOR_COMMIT = "54260cf9f2dfda5dc830f749c2debf4101d59d48"
PREVIOUS_TABLE_FILE = "previous_selector_table.json"
PREVIOUS_CANDIDATE = "previous-deployed"
# Cells whose PREVIOUS deployed behavior was the (now P1-B-forbidden)
# forward-composition inheritance — pre-registered exceptions that may
# never be re-enabled for speed (issue #8 PR-J A1).
PRE_INTENTIONAL_CORRECTIONS = {
    ("N2048B16", "c2c-inverse"),
    ("N2048B16", "r2c"),
    ("N2048B16", "c2r"),
}
INTENTIONAL_REASON = (
    "intentional semantic correction: previous deployed inherited the "
    "c2c-forward composition plan for this semantic; P1-B forbids it "
    "(never re-enabled)")


def _scope_cells(scope):
    """Unique (N, batch) schedule for a scope, order-preserving.

    ``all`` concatenates main+apps+large and the OFDM/radar proxy shapes
    sit in both main and apps (issue #8 domain bookkeeping). Without
    dedupe each shared cell is measured twice per session and emits
    duplicate (session, label, trial, candidate) raw keys, which the
    strict manifest gate rejects -- found by the post-promote
    full-domain confirmation run.
    """
    if scope == "all":
        raw = SCOPES["main"] + SCOPES["apps"] + SCOPES["large"]
    else:
        raw = SCOPES[scope]
    return list(dict.fromkeys(raw))


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def _pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    return xs[min(len(xs) - 1, max(0, int(q * len(xs))))]


def _git(*args):
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True).strip()


def _mad(xs):
    m = _median(xs)
    if m is None:
        return None
    return _median([abs(x - m) for x in xs])


def _err_vs_np(got, ref_np):
    import numpy as np
    g = got.detach().cpu().numpy()
    delta = np.maximum(np.abs(g.real - ref_np.real),
                       np.abs(g.imag - ref_np.imag))
    scale = np.abs(ref_np).max()
    return float(delta.max() / max(float(scale), 1e-12))


def _err_real_vs_np(got, ref_np):
    import numpy as np
    g = got.detach().cpu().numpy()
    scale = np.abs(ref_np).max()
    return float(np.abs(g - ref_np).max() / max(float(scale), 1e-12))


def _semantic_of(name):
    transform_type = "r2c" if name == "r2c" else (
        "c2c" if name.startswith("c2c") else "c2r")
    direction = ("inverse" if name in ("c2c-inverse", "c2r")
                 else "forward")
    return transform_type, direction


def _norm_of(name):
    return "backward"  # fixed release norm (torch default)


def _make_input(kind, n, b, semantic, seed):
    import numpy as np
    import torch
    rng = np.random.default_rng(seed)
    if semantic in ("r2c", "c2r"):
        a = rng.standard_normal((b, n))
        if semantic == "c2r":
            # input contract: complex64 half spectrum of the same
            # seeded real signal the reference inverts
            half = np.fft.rfft(a, axis=-1).astype(np.complex64)
            if b == 1:
                half = half[0]
            return torch.from_numpy(half).to("cuda")
        a = a.astype(np.float32)
        if b == 1:
            a = a[0]
        return torch.from_numpy(a).to("cuda")
    a = (rng.standard_normal((b, n))
         + 1j * rng.standard_normal((b, n))).astype(np.complex64)
    if b == 1:
        a = a[0]
    return torch.from_numpy(a).to("cuda")


def _ref_np(n, b, semantic, seed, norm="backward"):
    import numpy as np
    rng = np.random.default_rng(seed)
    if semantic in ("r2c", "c2r"):
        a = rng.standard_normal((b, n))
        if semantic == "r2c":
            return np.fft.rfft(a, axis=-1, norm=norm)
        half = np.fft.rfft(a, axis=-1)
        ref = np.fft.irfft(half, n=n, axis=-1, norm=norm)
        if norm in ("none", "forward"):
            ref = ref * n
        return ref
    a = (rng.standard_normal((b, n))
         + 1j * rng.standard_normal((b, n)))
    if semantic == "c2c-forward":
        if norm == "none":
            norm = "backward"  # numpy backward == unscaled forward
        return np.fft.fft(a, axis=-1, norm=norm)
    if norm == "ortho":
        return np.fft.ifft(a, axis=-1, norm="ortho")
    ref = np.fft.ifft(a, axis=-1, norm="backward")
    if norm in ("none", "forward"):
        ref = ref * n   # ours unscales the inverse for these norms
    return ref


def _vendor_call(x, semantic, n, b):
    import torch
    norm = _norm_of(semantic)
    if semantic == "c2c-forward":
        return torch.fft.fft(x, dim=-1, norm=norm)
    if semantic == "c2c-inverse":
        return torch.fft.ifft(x, dim=-1, norm=norm)
    if semantic == "r2c":
        return torch.fft.rfft(x, dim=-1, norm=norm)
    return torch.fft.irfft(x, n=n, dim=-1, norm=norm)


def _entry_plan(direction, n, b, semantic, profile_hash, table):
    from src.xpu.composition import load_composition_table
    from src.xpu.entry import plan_fft
    transform_type, transform_direction = _semantic_of(semantic)
    return plan_fft(direction, n, batch=b,
                    profile_hash=profile_hash, table=table,
                    transform_type=transform_type,
                    transform_direction=transform_direction,
                    norm=_norm_of(semantic),
                    composition_table=load_composition_table())


def _attribution(executed, params=None):
    """Flow/unit attribution for one measured plan (single source).

    The entry recorded in a promoted table must describe the lowering
    that actually ran; matrix oracle plans therefore carry their
    attribution from here (the same flow ids/unit names the bench
    builder and the schema-v2 table use), never re-derived downstream.
    """
    p = params or {}
    real_flow = p.get("real_flow")
    if real_flow:
        # Dedicated real flow (PR-H): the plan's real_flow IS the
        # attribution; realization/unit stay the underlying core's.
        real, unit = _attribution(
            executed, {k: v for k, v in p.items() if k != "real_flow"})[1:]
        return real_flow, real, unit
    variant = p.get("variant") or "folded"
    if executed in ("fft_dit_batch", "fft_dif_batch"):
        return f"batch-{variant}", "global-stage", "radix2-scalar"
    if executed in ("fft_dit_matrix", "fft_dif_matrix",
                    "fft_dit_matrix_batch", "fft_dif_matrix_batch"):
        return "local-fused-tile", "matrix-resident", "dense-dot-fft64"
    if executed == "fft_dit_fused":
        return "local-fused-tile", "fused-inplace", "radix2-scalar"
    if executed == "fft_dit_radix4":
        return "baseline-global-stage", "global-stage", "radix4-scalar"
    if executed in ("fft_dit_radix4_batch", "fft_dif_radix4_batch"):
        # Batched radix-4 cascade: same batch-flow family as M4 (the
        # lowering contract's spread branch says batch-spread for the
        # global stage; folded lowers radix-2 only). A radix4_batch
        # plan without an explicit spread variant cannot be attributed
        # -- fail closed instead of guessing.
        if variant != "spread":
            return None, None, None
        return "batch-spread", "global-stage", "radix4-scalar"
    if executed in ("fft_dit", "fft_dif"):
        return "baseline-global-stage", "global-stage", "radix2-scalar"
    return None, None, None


def _oracle_plans(n, b, direction, semantic, block):
    """Bounded measured candidate set, each as an executable plan."""
    from src.xpu.entry import ExecutionPlan, REAL_COMPAT_MARK, REAL_FLOW_MARK
    transform_type, transform_direction = _semantic_of(semantic)
    common = dict(n=n, batch=b, direction=direction, precision="fp32",
                  selection_source="oracle-candidate",
                  reason="bounded oracle set", fallback=False,
                  transform_type=transform_type,
                  transform_direction=transform_direction,
                  norm=_norm_of(semantic))
    compat = ({"real_compat": REAL_COMPAT_MARK}
              if transform_type in ("r2c", "c2r") else {})
    plans = []

    def add(name, executed, params=None, blk=None):
        flow, real, unit = _attribution(executed, params)
        # A dedicated real-flow candidate carries the dedicated mark
        # (PR-H); the compat mark stays for the full-length pipeline.
        mark = ({"real_compat": REAL_FLOW_MARK}
                if (params or {}).get("real_flow") else compat)
        plans.append(ExecutionPlan(
            **common, executed=executed, block=blk or block,
            flow_id=flow, flow_realization=real, processing_unit=unit,
            params={**(params or {}), "oracle_name": name, **mark}))

    if b == 1:
        add("m0_dit", "fft_dit")
        add("m0_dif", "fft_dif")
        add("m1_fused_t64", "fft_dit_fused", {"tile": 64})
        add("m2_dit_t64", "fft_dit_matrix", {"tile": 64, "rows": 8})
        add("m2_dif_t64", "fft_dif_matrix", {"tile": 64, "rows": 8})
        add("m5_r4_dit", "fft_dit_radix4")
    else:
        add("m0_dit", "fft_dit_batch", {"variant": "folded"})
        add("m4_spread", "fft_dit_batch", {"variant": "spread"})
        add("m4_folded", "fft_dit_batch", {"variant": "folded"})
        add("m2_dit_t64", "fft_dit_matrix_batch",
            {"tile": 64, "rows": 8, "variant": "spread"})
        add("m2_dif_t64", "fft_dif_matrix_batch",
            {"tile": 64, "rows": 8, "variant": "spread"})
        add("m5_r4_dit", "fft_dit_radix4_batch", {"variant": "spread"})
    if transform_type in ("r2c", "c2r"):
        # Dedicated half-length pipeline candidates (PR-H): same core
        # space, timed pack/stitch/pre/unpack epilogues; budget stays
        # 6 + 2 = 8 <= 24 per semantic/shape.
        real_flow = ("dedicated-r2c-pack-v1" if transform_type == "r2c"
                     else "dedicated-c2r-hermitian-v1")
        if b == 1:
            add("h_dit_t64", "fft_dit_matrix",
                {"tile": 64, "rows": 8, "real_flow": real_flow})
            add("h_dif_t64", "fft_dif_matrix",
                {"tile": 64, "rows": 8, "real_flow": real_flow})
        else:
            add("h_dif_t64", "fft_dif_matrix_batch",
                {"tile": 64, "rows": 8, "variant": "spread",
                 "real_flow": real_flow})
            add("h_spread", "fft_dit_batch",
                {"variant": "spread", "real_flow": real_flow})
    return plans


def _winner_block_plans(selector):
    """Single-factor round-2 candidates: the production winner plan
    (flow attribution and params exactly as promoted) re-blocked over
    the fixed block sweep; the selector's own block is always included
    so the A/B can only tie or improve the deployed geometry."""
    from dataclasses import replace
    plans, seen = [], set()
    for blk in (*WINNER_BLOCK_SWEEP, selector.block):
        # composition plans carry block=None (the composed core picks
        # its own geometry); the fixed sweep still applies.
        if not isinstance(blk, int) or blk <= 0 or blk in seen:
            continue
        seen.add(blk)
        plans.append(replace(
            selector, block=blk,
            params={**(selector.params or {}),
                    "oracle_name": f"winner-b{blk}"}))
    return plans


def _declared_inventory(cases, table, oracle_mode, sessions, trials,
                        warmup, one_shot_trials, e2e_trials,
                        profile_hash, has_previous=True, sem_map=None):
    """Pre-measurement declared candidate inventory (issue #9 WP0-B).

    Exactly the contenders :func:`run_case` will construct for every
    (cell x semantic), with the per-tier row counts each session owes
    (overhead tiers are declared under their actual row candidate names
    ``<base>:<tier>``) and the terminal-outcome vocabulary. Stamped into
    the manifest BEFORE session 0 runs; the strict gate then demands the
    raw evidence account for every declaration instead of letting
    observed rows (or observed oracle names) redefine the registry.
    """
    from src.xpu.manifest_contract import (
        DECLARED_INVENTORY_SCHEMA, DECLARED_TERMINAL_STATUSES)

    cells = []
    for n, b in cases:
        for semantic in (sem_map.get((n, b), SEMANTICS) if sem_map
                         else SEMANTICS):
            label = f"N{n}B{b}:{semantic}"
            incumbent = _entry_plan(DIRECTION, n, b, semantic, None, table)
            if oracle_mode == "winner-block":
                selector = _entry_plan(DIRECTION, n, b, semantic,
                                       profile_hash, table)
                if selector.fallback:
                    raise RuntimeError(
                        "winner-block declared inventory needs a "
                        "production entry, but the selector falls back "
                        f"for {label}")
                oplans = _winner_block_plans(selector)
            else:
                oplans = _oracle_plans(
                    n, b, DIRECTION, semantic,
                    block=incumbent.block or 1024)
            cands = {
                "incumbent": {"steady": trials},
                f"incumbent:{TIER_ONE_SHOT}": {TIER_ONE_SHOT:
                                               one_shot_trials},
                f"incumbent:{TIER_E2E}": {TIER_E2E: e2e_trials},
                "selector": {"steady": trials},
                f"selector:{TIER_ONE_SHOT}": {TIER_ONE_SHOT:
                                              one_shot_trials},
                f"selector:{TIER_E2E}": {TIER_E2E: e2e_trials},
                f"selector:{TIER_PLAN_CREATION}": {TIER_PLAN_CREATION: 1},
                "vendor": {"steady": trials},
            }
            if has_previous:
                cands[PREVIOUS_CANDIDATE] = {"steady": trials}
            for plan in oplans:
                cands[f"oracle:{plan.params['oracle_name']}"] = {
                    "steady": trials}
            cells.append({"label": label, "candidates": cands})
    return {
        "schema": DECLARED_INVENTORY_SCHEMA,
        "trials": trials,
        "sessions": sessions,
        "warmup": warmup,
        "one_shot_trials": one_shot_trials,
        "e2e_trials": e2e_trials,
        "allowed_terminal_statuses": list(DECLARED_TERMINAL_STATUSES),
        "cells": cells,
    }


def _timed_steady(fns, warmup, trials, seed):
    """Interleaved randomized trials across contenders (one call each)."""
    from src.xpu.runner import _sync
    import torch  # noqa: F401  (device stack ready via runners)
    for fn in fns.values():
        for _ in range(warmup):
            fn()
        _sync()
    order = list(fns)
    rng = random.Random(seed)
    samples = {k: [] for k in fns}
    for _ in range(trials):
        rng.shuffle(order)
        for key in order:
            t0 = time.perf_counter()
            fns[key]()
            _sync()
            samples[key].append((time.perf_counter() - t0) * 1e3)
    return samples


def _timed_once(fn, warmup, trials, seed):
    """Time ``fn`` and return ``(samples, last_output)``.

    The last output is captured so the caller can re-derive
    correctness ON the timed path itself (issue #9 re-review P1-1:
    one-shot/E2E rows may never inherit steady-tier correctness).
    """
    from src.xpu.runner import _sync
    for _ in range(warmup):
        fn()
    _sync()
    rng = random.Random(seed)
    xs = []
    last = None
    for _ in range(trials):
        t0 = time.perf_counter()
        last = fn()
        _sync()
        xs.append((time.perf_counter() - t0) * 1e3)
    return xs, last


def _validate_timed_path(out, ref, err_fn, semantic):
    """Independently validate one timed-path output for a semantic.

    Returns ``(err, problem)``: ``err`` is the semantic's own error
    against the reference (never inherited from the steady tier);
    ``problem`` is a human-readable terminal reason when the timed
    path fails its own shape/dtype/value contract.  Exactly one of the
    two is None.
    """
    try:
        if out is None:
            return None, "TimedPathNoOutput: timed path returned no output"
        shape = tuple(out.shape)
        ref_shape = tuple(ref.shape)
        # batched cells record ref with an explicit batch dim; b=1
        # inputs are squeezed by _make_input, so both shapes are legal
        legal = {ref_shape}
        if len(ref_shape) == 2 and ref_shape[0] == 1:
            legal.add(ref_shape[1:])
        if shape not in legal:
            return None, (f"TimedPathShape: output shape {shape} not in "
                          f"legal shapes {sorted(legal)}")
        dtype = str(getattr(out, "dtype", ""))
        want = "float32" if semantic == "c2r" else "complex64"
        if want not in dtype:
            return None, (f"TimedPathDtype: output dtype {dtype!r} != "
                          f"{want} ({semantic})")
        err = err_fn(out, ref)
        if err > 1e-4:
            return err, (f"TimedPathWrongValue: err={err:.3e} > 1e-4 "
                         "(independent timed-path check)")
        return err, None
    except Exception as e:  # noqa: BLE001
        return None, f"TimedPathError: {type(e).__name__}: {e}"


def _row(session, case_label, semantic, contender, tier, trial, **kw):
    from src.xpu.config import DeviceProfile
    row = {
        "session": session,
        "label": f"{case_label}:{semantic}",
        "trial": trial,
        "candidate": contender,
        "tier": tier,
        "protocol_id": PROTOCOL_ID,
        "profile_hash": DeviceProfile.load().hash(),
        "timed_scope": TIMING_SCOPE,
        "processing_unit": kw.get("processing_unit"),
        "flow_id": kw.get("flow_id"),
        "flow_realization": kw.get("flow_realization"),
        "params": kw.get("params"),
        # plan geometry the sample was measured under (issue #8: the
        # builder binds an entry to the measured block, never to a
        # re-derived default)
        "block": kw.get("block"),
        "executed": kw.get("executed"),
        "api_ms": kw.get("api_ms"),
        "correct": kw.get("correct"),
        "err": kw.get("err"),
        "status": kw.get("status", "ok"),
        "reason": kw.get("reason"),
        "notes": kw.get("notes"),
    }
    return row


def _plan_creation_row(session, case_label, semantic, xs, errs):
    """plan_creation tier emission (selector only).

    The strict contract requires every ``status='ok'`` row to carry a
    validated ``err``. The plan-creation probe itself never computes
    correctness, so it borrows the selector's setup-probe error; when
    the selector never passed that probe (setup raised, contender
    deleted), an ``ok`` row would be unvalidated -- record the row as
    an explicit error row instead (issue #9 WP5 strict gate).
    """
    err = errs.get("selector")
    notes = f"samples={xs}"
    if err is None:
        return _row(
            session, case_label, semantic, "selector:plan_creation",
            TIER_PLAN_CREATION, 0, api_ms=_median(xs) if xs else None,
            correct=False, err=None, status="error", notes=notes,
            reason="SelectorCorrectnessNotMeasured: selector setup "
                   "did not validate for this cell")
    return _row(
        session, case_label, semantic, "selector:plan_creation",
        TIER_PLAN_CREATION, 0, api_ms=_median(xs), correct=True,
        err=err, notes=notes)


def run_case(session, n, b, semantic, trials, warmup, seed, table,
             previous_table, raw, e2e_trials, one_shot_trials):
    import torch
    from src.xpu.composition import load_composition_table
    from src.xpu.entry import execute_fft_plan, run_fft
    from src.xpu.config import DeviceProfile

    case_label = f"N{n}B{b}"
    direction = DIRECTION
    prof = DeviceProfile.load().hash()
    norm = _norm_of(semantic)
    transform_type, transform_direction = _semantic_of(semantic)

    x = _make_input("random", n, b, semantic, seed)
    ref = _ref_np(n, b, semantic, seed, norm)
    if semantic == "r2c":
        err_fn = _err_vs_np
    elif semantic == "c2r":
        err_fn = _err_real_vs_np
    else:
        err_fn = _err_vs_np

    contenders = {}   # name -> (plan, fn)
    incumbent = _entry_plan(direction, n, b, semantic, None, table)
    selector = _entry_plan(direction, n, b, semantic, prof, table)
    contenders["incumbent"] = (incumbent,
                               (lambda p=incumbent: execute_fft_plan(p, x)))
    contenders["selector"] = (selector,
                              (lambda p=selector: execute_fft_plan(p, x)))
    if previous_table is not None:
        # Previous production selector resolved through the CURRENT
        # contracts (fail-closed: illegal old plans fall back, the
        # banned inverse-composition path can never re-enter). Same
        # profile/env binding, interleaved by _timed_steady below.
        previous = _entry_plan(direction, n, b, semantic, prof,
                               previous_table)
        contenders[PREVIOUS_CANDIDATE] = (
            previous, (lambda p=previous: execute_fft_plan(p, x)))
    if ORACLE_MODE == "winner-block":
        if selector.fallback:
            raise RuntimeError(
                "winner-block oracle mode needs a production entry to "
                f"sweep, but the selector fell back for "
                f"N{n}B{b}:{semantic}")
        oracle_plans = _winner_block_plans(selector)
    else:
        oracle_plans = _oracle_plans(n, b, direction, semantic,
                                     block=incumbent.block or 1024)
    for plan in oracle_plans:
        name = plan.params["oracle_name"]
        contenders[f"oracle:{name}"] = (
            plan, (lambda p=plan: execute_fft_plan(p, x)))
    # vendor
    contenders["vendor"] = (
        None, (lambda: _vendor_call(x, semantic, n, b)))

    # correctness gate first
    errs = {}
    for name, (plan, fn) in list(contenders.items()):
        try:
            y = fn()
            err = err_fn(y, ref)
            errs[name] = err
            if err > 1e-4:
                raw.append(_row(
                    session, case_label, semantic, name, "correctness",
                    0, api_ms=None, correct=False, err=err,
                    status="wrong-value",
                    reason=f"err={err:.3e} > 1e-4"))
                del contenders[name]
        except Exception as e:  # noqa: BLE001
            raw.append(_row(
                session, case_label, semantic, name, "correctness", 0,
                api_ms=None, correct=False, err=None,
                status="vendor-unsupported"
                if name == "vendor" else "error",
                reason=f"{type(e).__name__}: {e}"))
            del contenders[name]

    if not contenders:
        return

    # steady interleaved timing
    fns = {name: fn for name, (_, fn) in contenders.items()}
    samples = _timed_steady(fns, warmup, trials,
                            seed=seed * 1000 + 1)
    for name, xs in samples.items():
        for i, ms in enumerate(xs):
            plan = contenders[name][0]
            raw.append(_row(
                session, case_label, semantic, name, TIER_STEADY, i,
                api_ms=ms, correct=True, err=errs.get(name),
                block=plan.block if plan else None,
                executed=plan.executed if plan else "vendor-torch-fft",
                flow_id=plan.flow_id if plan else "vendor",
                flow_realization=(plan.flow_realization if plan
                                  else "global-stage"),
                processing_unit=(plan.processing_unit if plan
                                 else "vendor-torch-fft"),
                params={**((plan.params or {}) if plan else {}),
                        "selection_source": plan.selection_source
                        if plan else "vendor",
                        "fallback": plan.fallback if plan else False}))

    # one-shot + e2e for incumbent/selector only (overhead tiers).
    # P1-1: each timed path re-derives correctness on its own output
    # (semantic err_fn + shape/dtype contract); failures become
    # explicit terminal rows, never inherited steady correctness.
    for name in ("incumbent", "selector"):
        if name not in contenders:
            continue
        direction_arg = direction
        try:
            # run_fft returns (y, ExecutionReport); the timed path
            # yields the tensor so timed-path validation sees output
            prof_arg = prof if name == "selector" else None

            def one_shot(prof_arg=prof_arg):
                y, _rep = run_fft(
                    x, direction_arg, profile_hash=prof_arg,
                    table=table, transform_type=transform_type,
                    transform_direction=transform_direction,
                    norm=norm,
                    composition_table=load_composition_table())
                return y

            xs, out = _timed_once(
                one_shot,
                max(2, warmup // 2), one_shot_trials,
                seed=seed * 1000 + 2)
            err, problem = _validate_timed_path(out, ref, err_fn,
                                                semantic)
            if problem:
                raw.append(_row(
                    session, case_label, semantic, f"{name}:one_shot",
                    TIER_ONE_SHOT, 0, api_ms=None, correct=False,
                    err=err,
                    status=("wrong-value" if problem.startswith(
                        "TimedPathWrongValue") else "error"),
                    reason=problem + "; steady correctness not "
                    "inherited"))
            else:
                for i, ms in enumerate(xs):
                    raw.append(_row(
                        session, case_label, semantic,
                        f"{name}:one_shot", TIER_ONE_SHOT, i, api_ms=ms,
                        correct=True, err=err,
                        notes="includes per-call plan/dispatch "
                        "overhead; correctness independently "
                        "re-derived on the timed path"))
        except Exception as e:  # noqa: BLE001
            raw.append(_row(
                session, case_label, semantic, f"{name}:one_shot",
                TIER_ONE_SHOT, 0, api_ms=None, correct=False,
                status="error", reason=f"{type(e).__name__}: {e}"))
        try:
            x_cpu = x.cpu().contiguous()
            plan = contenders[name][0]

            def e2e(plan=plan):
                xd = x_cpu.to("cuda")
                y = execute_fft_plan(plan, xd)
                return y.cpu()

            xs, out = _timed_once(e2e, max(2, warmup // 2), e2e_trials,
                                  seed=seed * 1000 + 3)
            err, problem = _validate_timed_path(out, ref, err_fn,
                                                semantic)
            if problem:
                raw.append(_row(
                    session, case_label, semantic, f"{name}:e2e",
                    TIER_E2E, 0, api_ms=None, correct=False,
                    err=err,
                    status=("wrong-value" if problem.startswith(
                        "TimedPathWrongValue") else "error"),
                    reason=problem + "; steady correctness not "
                    "inherited"))
            else:
                for i, ms in enumerate(xs):
                    raw.append(_row(
                        session, case_label, semantic, f"{name}:e2e",
                        TIER_E2E, i, api_ms=ms, correct=True, err=err,
                        notes="H2D copy + execute + D2H copy; "
                        "correctness independently re-derived on the "
                        "timed path (post-D2H output)"))
        except Exception as e:  # noqa: BLE001
            raw.append(_row(
                session, case_label, semantic, f"{name}:e2e",
                TIER_E2E, 0, api_ms=None, correct=False,
                status="error", reason=f"{type(e).__name__}: {e}"))

    # plan-creation timing (selector only; host path)
    try:
        import time as _t
        xs = []
        for i in range(5):
            t0 = _t.perf_counter()
            _entry_plan(direction, n, b, semantic, prof, table)
            xs.append((_t.perf_counter() - t0) * 1e3)
        raw.append(_plan_creation_row(
            session, case_label, semantic, xs, errs))
    except Exception as e:  # noqa: BLE001
        raw.append(_row(
            session, case_label, semantic, "selector:plan_creation",
            TIER_PLAN_CREATION, 0, api_ms=None, correct=False,
            err=errs.get("selector"), status="error",
            reason=f"{type(e).__name__}: {e}"))


def run_child(payload):
    session, cases, trials, warmup, seed, out_dir, e2e_trials, \
        one_shot_trials = payload[:8]
    previous_table_path = payload[8] if len(payload) > 8 else None
    from src.xpu.selector import load_table
    table = load_table(str(ROOT / "config" / "selector_table.json"))
    previous_table = (load_table(str(previous_table_path))
                      if previous_table_path else None)
    raw_path = Path(out_dir) / f"raw_session{session}.jsonl"
    # WP1: the parent's schedule.json is the frozen exact cell set
    # (per-shape semantics); --scope stays the fallback for directly
    # spawned legacy children.
    sem_map = None
    sched_path = Path(out_dir) / "schedule.json"
    if sched_path.is_file():
        sched = json.loads(sched_path.read_text())
        cases = [(int(n), int(b)) for n, b, _sems in sched["cells"]]
        sem_map = {(int(n), int(b)): list(sems)
                   for n, b, sems in sched["cells"]}
    # per-session seeded interleaving of case order (and semantic
    # order): no session measures the grid in the same sequence
    order = list(cases)
    rnd = random.Random(seed + session * 10007)
    rnd.shuffle(order)
    with open(raw_path, "w") as raw_f:
        for n, b in order:
            sems = list(sem_map.get((n, b), SEMANTICS) if sem_map
                        else SEMANTICS)
            rnd.shuffle(sems)
            for semantic in sems:
                rows = []
                try:
                    run_case(session, n, b, semantic, trials, warmup,
                             seed + session * 10007 + n * 13 + b, table,
                             previous_table, rows, e2e_trials,
                             one_shot_trials)
                except Exception as e:  # noqa: BLE001
                    rows.append(_row(
                        session, f"N{n}B{b}", semantic, "case", "error",
                        0, api_ms=None, correct=False, status="error",
                        reason=f"{type(e).__name__}: {e}"))
                for r in rows:
                    raw_f.write(json.dumps(r) + "\n")
                raw_f.flush()
    return raw_path


def _previous_regression_reason(case_label, semantic, entry):
    """Disposition text for the previous-deployed comparison (A1).

    - pre-registered semantic fixes (banned inverse-composition era
      behavior) always carry the intentional-correction marker;
    - any other cell slower than the LEGAL previous deployed path by
      more than 2% is flagged for the review reply (keep old path as
      fallback unless it was itself a correctness/semantic fix).
    """
    key = (case_label, semantic)
    reg = entry.get("selector_vs_previous")
    slow = reg is not None and reg > 0.02
    if key in PRE_INTENTIONAL_CORRECTIONS:
        return INTENTIONAL_REASON
    if slow:
        return ("current selector slower than legal previous deployed "
                ">2% (candidate: retain previous path as fallback)")
    return None


def _aggregate(out_dir, cases, trials, warmup, sessions, manifest=None,
               write_dir=None, sem_map=None):
    from src.xpu.manifest_contract import (
        REQUIRED_ROW_KEYS, declared_inventory_failures, iter_raw_rows,
        rows_sha256, trial_completeness_failures)
    # issue #9 WP0-C: analysis may be written to a separate replay
    # directory while the raw evidence stays where it was recorded
    dest = Path(write_dir) if write_dir else Path(out_dir)

    # P2-c: aggregate reads through the SAME plain/gzip reader the
    # strict gate uses -- an --aggregate-only replay of a released
    # snapshot (.jsonl.gz) aggregates the identical bytes.
    raw_rows = []
    decode_failures = []
    for _, _, r in iter_raw_rows(Path(out_dir)):
        if "__decode_error__" in r:
            decode_failures.append(
                {"kind": "raw-decode-error", "reason":
                 r["__decode_error__"]})
            continue
        raw_rows.append(r)
    steady = {}
    meta = {}
    failures = list(decode_failures)
    # P2-a: the same session/label/candidate/tier census the strict
    # validator runs -- a truncated candidate or missing session shows
    # up in summary.failures instead of silently aggregating short.
    for msg in trial_completeness_failures(raw_rows, manifest or {}):
        failures.append({"kind": "trial-incompleteness", "detail": msg})
    # Issue #9 WP0-B: declared-inventory census -- a declared cell or
    # candidate with zero raw rows is a hard failure in the summary
    # itself, not only at snapshot validation time.
    for msg in declared_inventory_failures(raw_rows, manifest or {}):
        failures.append({"kind": "declared-inventory", "detail": msg})
    for r in raw_rows:
        missing = [k for k in REQUIRED_ROW_KEYS
                   if k not in r or r[k] in (None, "")]
        if missing and r.get("status") == "ok":
            failures.append({"kind": "missing-row-field",
                             "row": {k: r.get(k) for k in
                                     ("label", "candidate", "tier")},
                             "missing": missing})
        key = (r["label"], r["candidate"], r.get("tier"))
        if r.get("tier") == TIER_STEADY and r.get("status") == "ok" \
                and r.get("api_ms") is not None:
            steady.setdefault(key, []).append(r["api_ms"])
            meta[key] = r
        if r.get("status") in ("wrong-value", "vendor-unsupported",
                               "error"):
            failures.append({
                "kind": r["status"], "label": r["label"],
                "candidate": r["candidate"], "tier": r.get("tier"),
                "reason": r.get("reason"),
            })

    cases_out = []
    for n, b in cases:
        case_label = f"N{n}B{b}"
        for semantic in (sem_map.get((n, b), SEMANTICS) if sem_map
                         else SEMANTICS):
            label = f"{case_label}:{semantic}"
            med = {}
            mad = {}
            for (lab, cand, tier), xs in steady.items():
                if lab == label and cand.startswith(
                        ("incumbent", "selector", "oracle:", "vendor",
                         PREVIOUS_CANDIDATE)):
                    med[cand] = _median(xs)
                    mad[cand] = _mad(xs)
            oracle_keys = [k for k in med if k.startswith("oracle:")]
            oracle = min((med[k] for k in oracle_keys), default=None)
            oracle_name = (min(oracle_keys, key=lambda k: med[k])
                           if oracle_keys else None)
            sel = med.get("selector")
            inc = med.get("incumbent")
            ven = med.get("vendor")
            prev = med.get(PREVIOUS_CANDIDATE)
            # selector_vs_previous > 0: current selector is SLOWER
            # than the legal previous deployed path (regression).
            vs_prev = ((sel / prev - 1.0)
                       if (sel and prev) else None)
            entry = {
                "label": label, "N": n, "batch": b,
                "semantic": semantic, "direction": DIRECTION,
                "selector_ms": sel, "incumbent_ms": inc,
                "vendor_ms": ven, "oracle_ms": oracle,
                "selector_mad_ms": mad.get("selector"),
                "incumbent_mad_ms": mad.get("incumbent"),
                "vendor_mad_ms": mad.get("vendor"),
                "oracle_mad_ms": (
                    min((mad[k] for k in oracle_keys), default=None)
                    if oracle_keys else None),
                "oracle_candidate": oracle_name,
                "selector_regret": (
                    (sel / oracle - 1.0)
                    if (sel and oracle) else None),
                "speedup_vs_vendor": (
                    (ven / sel) if (sel and ven) else None),
                "speedup_vs_incumbent": (
                    (inc / sel) if (sel and inc) else None),
                "guard_vs_incumbent": (
                    ((inc - sel) / inc) if (sel and inc) else None),
                "vendor_status": ("ok" if ven else (
                    "unsupported" if any(
                        f.get("label") == label
                        and f.get("candidate") == "vendor"
                        for f in failures) else "missing")),
                "selector_source": meta.get(
                    (label, "selector", TIER_STEADY), {}
                ).get("params", {}).get("selection_source"),
                # Issue #8 Option A: did this cell deploy the
                # previous-deployed fallback? (raw rows carry
                # params.previous_fallback from plan_fft)
                "selector_previous_fallback": bool(meta.get(
                    (label, "selector", TIER_STEADY), {}
                ).get("params", {}).get("previous_fallback")),
                # A1: previous production column (issue #8 review)
                "previous_deployed_ms": prev,
                "previous_deployed_mad_ms": mad.get(PREVIOUS_CANDIDATE),
                "selector_vs_previous": vs_prev,
                "previous_deployed_source": meta.get(
                    (label, PREVIOUS_CANDIDATE, TIER_STEADY), {}
                ).get("params", {}).get("selection_source"),
                "sessions_with_data": len({
                    r["session"] for r in raw_rows
                    if r["label"] == label
                    and r["candidate"] == "selector"
                    and r.get("tier") == TIER_STEADY
                    and r.get("status") == "ok"}),
            }
            entry["guard_pass_2pct"] = (
                bool(entry["guard_vs_incumbent"] is not None
                     and entry["guard_vs_incumbent"] >= 0.02)
                if entry["selector_source"] in ("table", "composition")
                else None)
            entry["regression_reason"] = _previous_regression_reason(
                case_label, semantic, entry)
            cases_out.append(entry)

    supported = [c for c in cases_out
                 if c["speedup_vs_vendor"] is not None]
    geomean = (math.exp(sum(math.log(c["speedup_vs_vendor"])
                            for c in supported) / len(supported))
               if supported else None)
    wins = sum(1 for c in supported if c["speedup_vs_vendor"] > 1.02)
    ties = sum(1 for c in supported
               if 0.98 <= c["speedup_vs_vendor"] <= 1.02)
    losses = sum(1 for c in supported if c["speedup_vs_vendor"] < 0.98)
    worst = min((c["speedup_vs_vendor"] for c in supported),
                default=None)
    regrets = [c["selector_regret"] for c in cases_out
               if c["selector_regret"] is not None]
    prev_measured = sum(1 for c in cases_out
                        if c.get("previous_deployed_ms") is not None)
    prev_regressions = [c["label"] for c in cases_out
                        if (c.get("selector_vs_previous")
                            is not None
                            and c["selector_vs_previous"] > 0.02)]
    intentional = [c["label"] for c in cases_out
                   if c.get("regression_reason")
                   and "intentional" in c["regression_reason"]]
    # P2-b / issue #9 WP0-B: the oracle candidate registry is DECLARED
    # before measurement (manifest.oracle_candidates stamped from the
    # declared inventory). Observed names are checked against it and
    # may never overwrite it; only legacy manifests without the key
    # (pre-WP0-B artifacts) still record the observed set.
    observed_oracles = sorted(
        {r["candidate"] for r in raw_rows
         if str(r.get("candidate") or "").startswith("oracle:")})
    declared_oracles = (manifest.get("oracle_candidates")
                        if manifest else None)
    if declared_oracles is None:
        oracle_candidates = observed_oracles
    else:
        declared_oracles = sorted(str(c) for c in declared_oracles)
        if declared_oracles != observed_oracles:
            failures.append({
                "kind": "oracle-candidates-mismatch",
                "declared": declared_oracles,
                "observed": observed_oracles,
            })
        oracle_candidates = declared_oracles
    summary = {
        "protocol_id": PROTOCOL_ID,
        "cases": cases_out,
        "metrics": {
            "vendor_supported_cases": len(supported),
            "win": wins, "tie": ties, "loss": losses,
            "geomean_speedup_vs_vendor": geomean,
            "worst_speedup_vs_vendor": worst,
            "mean_selector_regret": (
                sum(regrets) / len(regrets) if regrets else None),
            "max_selector_regret": max(regrets, default=None),
            "previous_cells_measured": prev_measured,
            "previous_regressions_gt_2pct": prev_regressions,
            "intentional_semantic_exceptions": intentional,
        },
        "failures": failures,
        "raw_sha256": hashlib.sha256(
            json.dumps(raw_rows, sort_keys=True,
                       separators=(",", ":")).encode()).hexdigest(),
        "rows_sha256_check": rows_sha256({"rows": raw_rows}),
    }
    if manifest is not None:
        # WP0 integration (issue #8): summary.json embeds the same
        # manifest as manifest.json so strict validation/stamping of a
        # directory artifact works on summary.json alone. The embedded
        # copy also binds the cases array (P1-C): a summary edited
        # after stamping fails the strict gate.
        manifest = dict(manifest)
        manifest["cases_sha256"] = cases_sha256(cases_out)
        manifest["oracle_candidates"] = oracle_candidates
        summary["manifest"] = manifest
    (dest / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    cols = ["label", "semantic", "N", "batch", "selector_ms",
            "selector_mad_ms", "incumbent_ms", "oracle_ms",
            "oracle_candidate", "vendor_ms", "vendor_mad_ms",
            "vendor_status", "speedup_vs_vendor",
            "selector_regret", "guard_pass_2pct",
            "selector_source", "sessions_with_data"]
    with open(dest / "summary.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for c in cases_out:
            f.write(",".join(
                str(c.get(k)) for k in cols) + "\n")
    # A1: full previous-deployed disposition table (all cells, not just
    # regressions) -- required final evidence artifact.
    import csv as _csv
    reg_cols = ["label", "semantic", "N", "batch",
                "previous_deployed_ms", "selector_ms",
                "selector_vs_previous", "previous_deployed_source",
                "selector_source", "regression_reason"]
    with open(dest / "regression_vs_previous.csv", "w",
              newline="") as f:
        w = _csv.writer(f)
        w.writerow(reg_cols)
        for c in cases_out:
            w.writerow([c.get(k) for k in reg_cols])
    _write_figures(dest, summary)
    return summary


def _write_figures(out_dir, summary):
    """speedup-vs-vendor figure driven by the same summary/manifest."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    import numpy as np
    sems = list(SEMANTICS)
    fig, axes = plt.subplots(len(sems), 1, figsize=(13, 2.6 * len(sems)))
    for ax, sem in zip(np.atleast_1d(axes), sems):
        cs = [c for c in summary["cases"] if c["semantic"] == sem]
        sp = [(c["speedup_vs_vendor"]
               if c["speedup_vs_vendor"] else float("nan"))
              for c in cs]
        colors = ["#2a7" if (s or 0) >= 1.02 else
                  ("#c72" if (s or 0) < 0.98 else "#888")
                  for s in sp]
        ax.bar(range(len(cs)), sp, color=colors, width=0.85)
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_ylabel(f"{sem}\nspeedup vs vendor")
        ax.set_xticks(range(len(cs)))
        ax.set_xticklabels([c["label"].split(":")[0] for c in cs],
                           rotation=90, fontsize=6)
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle(
        f"semantic matrix {summary['protocol_id']}: "
        f"win/tie/loss="
        f"{summary['metrics']['win']}/"
        f"{summary['metrics']['tie']}/"
        f"{summary['metrics']['loss']}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(out_dir) / "speedup_vs_vendor.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def _write_snapshot(out_dir, release_id, manifest, env, args, cases):
    snap = ROOT / "results" / "p800" / "releases" / release_id
    if snap.exists():
        raise SystemExit(f"release snapshot already exists (immutable): "
                         f"{snap}")
    snap.mkdir(parents=True)
    manifest = dict(manifest)
    # same hash function validate_benchmark_output recomputes (the
    # former local bytes-only variant diverged and failed the gate)
    manifest["result_sha256"] = result_sha256(out_dir)
    if "source_hash" not in manifest:
        manifest["source_hash"] = _git("rev-parse", "HEAD")
    (snap / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (snap / "environment.json").write_text(
        json.dumps(env, indent=2, sort_keys=True) + "\n")
    (snap / "repro_command.txt").write_text(
        " ".join(sys.argv) + "\n")
    (snap / "cases.json").write_text(json.dumps(cases) + "\n")
    for name in sorted(Path(out_dir).glob("raw_session*.jsonl")):
        # rows are highly repetitive (~30x); snapshot keeps gzip'd raw
        # as the canonical evidence artifact
        with gzip.open(snap / (name.name + ".gz"), "wb") as f:
            f.write(name.read_bytes())
    for name in ("summary.json", "summary.csv",
                 "regression_vs_previous.csv", "speedup_vs_vendor.png",
                 "server_report.md", PREVIOUS_TABLE_FILE):
        src = Path(out_dir) / name
        if src.is_file():
            (snap / name).write_bytes(src.read_bytes())
    # A2: the finished snapshot must strict-validate immediately --
    # failure aborts this run. (Also covers a missing summary.json.)
    failures = validate_benchmark_output(snap, strict=True)
    if failures:
        raise SystemExit(
            "snapshot failed strict validation:\n"
            + format_failures(failures))
    # binding fields must agree between top manifest and summary.manifest
    top = json.loads((snap / "manifest.json").read_text())
    emb = json.loads((snap / "summary.json").read_text())["manifest"]
    bind_keys = ["result_sha256", "cases_sha256"]
    if ("declared_inventory_sha256" in top
            or "declared_inventory_sha256" in emb):
        # issue #9 WP0-B: the pre-measurement inventory stamp must
        # survive into both copies (legacy snapshots predate the key)
        bind_keys.append("declared_inventory_sha256")
    for key in bind_keys:
        if top.get(key) != emb.get(key):
            raise SystemExit(
                f"snapshot binding mismatch for {key}: top "
                f"{top.get(key)!r} != summary {emb.get(key)!r}")
    return snap


def _cases_from_shapes(shapes):
    """(N, batch) pairs parsed from declared manifest shapes."""
    pairs = []
    for shape in shapes or []:
        m = re.match(r"^N(\d+)B(\d+)", str(shape).split(":")[0])
        if m:
            pairs.append((int(m.group(1)), int(m.group(2))))
    return list(dict.fromkeys(pairs))


def _load_replay_cases(out_dir, man, args):
    """Replay schedule, bound to the recorded run (issue #9 WP0-C).

    Priority: the run's own ``cases.json`` > its declared ``shapes`` >
    the requested scope (legacy artifacts with neither). The replay
    must analyze exactly what was declared, never today's scope.
    """
    cases_json = Path(out_dir) / "cases.json"
    if cases_json.is_file():
        try:
            data = json.loads(cases_json.read_text())
            return [(int(n), int(b)) for n, b in data]
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            raise SystemExit(f"unreadable cases.json: {cases_json}: {e}")
    pairs = _cases_from_shapes(man.get("shapes"))
    return pairs or _scope_cells(args.scope)


def _replay_sem_map(man):
    """Per-shape semantics from the recorded shape labels (WP1 filtered
    schedules replay exactly; unknown semantics fail closed)."""
    sem_map = {}
    for shape in man.get("shapes") or []:
        head, _, sem = str(shape).partition(":")
        m = re.match(r"^N(\d+)B(\d+)$", head)
        if not m or not sem:
            continue
        if sem not in SEMANTICS:
            raise SystemExit(
                f"recorded shape {shape!r} names unknown semantic "
                f"{sem!r}")
        key = (int(m.group(1)), int(m.group(2)))
        sem_map.setdefault(key, [])
        if sem not in sem_map[key]:
            sem_map[key].append(sem)
    return sem_map or None


def _print_summary(summary, note=None):
    m = summary["metrics"]
    print(f"cases={len(summary['cases'])} "
          f"win/tie/loss={m['win']}/{m['tie']}/{m['loss']} "
          f"geomean={m['geomean_speedup_vs_vendor']} "
          f"worst={m['worst_speedup_vs_vendor']} "
          f"mean_regret={m['mean_selector_regret']}")
    print(f"failures={len(summary['failures'])}")
    if note:
        print(note)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--e2e-trials", type=int, default=10)
    ap.add_argument("--one-shot-trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--scope", choices=sorted(SCOPES) + ["all"],
                    default="main")
    ap.add_argument("--cells", default=None, metavar="SPEC",
                    help="exact cell schedule, e.g. "
                         "'N64B4:c2c-forward+c2r,N512B64:r2c' -- "
                         "overrides --scope so a frozen experiment "
                         "domain (issue #9 WP1) is measured exactly")
    ap.add_argument("--out", default=str(
        ROOT / "results" / "p800" / "semantic_matrix"))
    ap.add_argument("--snapshot", action="store_true",
                    help="write immutable results/p800/releases/"
                         "<release-id>/ snapshot")
    ap.add_argument("--release-id", default=None)
    ap.add_argument("--aggregate-only", action="store_true",
                    help="reuse existing raw sessions (no measuring)")
    ap.add_argument("--replay-out", default=None, metavar="DIR",
                    help="WP0-C: write the regenerated analysis into a "
                         "NEW directory (raw evidence and its recorded "
                         "identity stay untouched); required when "
                         "--aggregate-only finds an existing "
                         "manifest.json")
    ap.add_argument("--session", type=int, default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--oracle-mode", choices=("full", "winner-block"),
                    default="full",
                    help="candidate set: full round-1 bounded set, or "
                         "the promoted production winner re-blocked "
                         "over the fixed block sweep (PR-I round 2)")
    ap.add_argument("--previous-table", default=None,
                    help="path to the frozen previous-deployed selector "
                         "table for the A1 same-session comparison "
                         "(child sessions get it from the parent)")
    ap.add_argument("--previous-table-src", default=None,
                    help="materialize the previous-deployed table from "
                         "this FILE instead of the fixed A1 commit "
                         "(issue #9 P1-2 v5-deployed pairing); the file "
                         "sha256 is bound into the manifest")
    args = ap.parse_args()

    global ORACLE_MODE
    ORACLE_MODE = args.oracle_mode

    if args.cells:
        cell_specs = parse_cells_spec(args.cells)
        cases = list(dict.fromkeys((n, b) for n, b, _ in cell_specs))
        sem_map = {(n, b): list(sems) for n, b, sems in cell_specs}
    else:
        cases = _scope_cells(args.scope)
        sem_map = None
    schedule = _schedule(cases, sem_map)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.replay_out and (not args.aggregate_only or args.snapshot):
        raise SystemExit(
            "--replay-out is only for --aggregate-only analysis "
            "replays (no measuring, no --snapshot)")

    # WP0-C: --aggregate-only over a directory that already carries a
    # manifest replays the RUN'S OWN recorded identity (commit, env,
    # registry, hashes, schedule) instead of re-declaring from the
    # working tree -- and the regenerated analysis must go to a NEW
    # directory, never over the evidence.
    input_manifest = None
    if args.aggregate_only and (out_dir / "manifest.json").is_file():
        input_manifest = json.loads(
            (out_dir / "manifest.json").read_text())
        if not args.replay_out:
            raise SystemExit(
                "--aggregate-only found an existing manifest.json: "
                "replaying a recorded run requires --replay-out DIR "
                "(new directory; never rewriting evidence in place)")

    if args.session is not None:
        # isolated session process: freshly exec'd interpreter, no
        # shared mp Pool (pool teardown deadlocks against the device
        # runtime's listener threads); os._exit skips interpreter
        # atexit teardown where the runtime can also hang
        path = run_child((args.session, cases, args.trials, args.warmup,
                          args.seed, str(out_dir), args.e2e_trials,
                          args.one_shot_trials,
                          args.previous_table))
        print(f"session done: {path}", flush=True)
        os._exit(0)

    write_dir = Path(args.replay_out) if args.replay_out else out_dir
    if write_dir != out_dir:
        if write_dir.exists() and any(write_dir.iterdir()):
            raise SystemExit(
                f"--replay-out must be empty or absent: {write_dir}")
        write_dir.mkdir(parents=True, exist_ok=True)

    if input_manifest is not None:
        # --------- WP0-C replay of a recorded run (host-reachable) ----
        # gates FIRST: the input must strict-validate exactly as
        # released (schema, row provenance, trial census, result hash),
        # so tampered raw/manifest evidence never regenerates.
        strict_fails = validate_benchmark_output(out_dir, strict=True)
        if strict_fails:
            raise SystemExit(format_failures(
                strict_fails, "replay input failed strict validation"))
        # the identity the replay inherits (top manifest.json) must be
        # the SAME identity the strict gate just verified behind the
        # summary's embedded copy -- tampering either side fails here
        try:
            emb = json.loads(
                (out_dir / "summary.json").read_text())["manifest"]
        except (OSError, json.JSONDecodeError, KeyError) as e:
            raise SystemExit(
                f"replay input has no readable summary manifest: {e}")
        for key in ("commit", "protocol_id", "env_hash", "profile_hash",
                    "result_sha256", "cases_sha256", "oracle_candidates"):
            if input_manifest.get(key) != emb.get(key):
                raise SystemExit(
                    "replay input manifest drift: "
                    f"{key} top {input_manifest.get(key)!r} != summary "
                    f"{emb.get(key)!r}")
        man = dict(input_manifest)
        cases = _load_replay_cases(out_dir, man, args)
        # per-shape semantics come from the recorded shape labels, so
        # a filtered schedule (WP1 frozen domains) replays identically
        sem_map = _replay_sem_map(man)
        # self-contained replay dir: evidence bytes copied verbatim
        for src in sorted(out_dir.glob("raw_session*")):
            if src.is_file():
                shutil.copy2(src, write_dir / src.name)
        if (out_dir / "cases.json").is_file():
            shutil.copy2(out_dir / "cases.json", write_dir / "cases.json")
        # source vs evidence identity: commit/env/hashes stay the
        # RUN's; the replay block names what re-derived the analysis
        man["replay"] = {
            "replayed_by_commit": _git("rev-parse", "HEAD"),
            "replayed_by_dirty": bool(_git("status", "--porcelain")),
            "replay_command": " ".join(sys.argv),
            "replay_input": str(out_dir),
            "replay_input_manifest_sha256": hashlib.sha256(
                (out_dir / "manifest.json").read_bytes()).hexdigest(),
        }
        summary = _aggregate(out_dir, cases, man["trials"],
                             man["warmup"], man["sessions"],
                             manifest=man, write_dir=write_dir,
                             sem_map=sem_map)
        # analysis drift: the regenerated cases binding must equal the
        # one recorded with the evidence
        regen = summary["manifest"].get("cases_sha256")
        if (man.get("cases_sha256") is not None
                and regen != man["cases_sha256"]):
            raise SystemExit(
                "replay cases drift: regenerating the analysis from "
                f"raw yields cases_sha256 {regen} != recorded "
                f"{man['cases_sha256']}")
        res = result_sha256(out_dir)
        if (man.get("result_sha256") is not None
                and res != man["result_sha256"]):
            raise SystemExit(
                f"replay raw drift: result_sha256 {res} != recorded "
                f"{man['result_sha256']}")
        (write_dir / "manifest.json").write_text(
            json.dumps(man, indent=2, sort_keys=True) + "\n")
        (write_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n")
        _print_summary(
            summary,
            f"replay verified: input strict OK, recorded result/cases "
            f"hashes match; written to {write_dir}")
        return

    # A1: materialize the previous deployed table from the fixed pre-
    # issue commit (never the working tree), bind its hash into the
    # manifest, and hand the file to every session child. Raw bytes —
    # no strip, so the sha256 matches the committed blob exactly.
    # P1-2: --previous-table-src materializes from an explicit FILE
    # instead (v5-deployed pairing runs); its sha256 is bound the same
    # way and the source kind is recorded in previous_selector_source.
    if args.previous_table_src:
        src_path = Path(args.previous_table_src)
        if not src_path.is_file():
            raise SystemExit(
                f"--previous-table-src {src_path} does not exist")
        previous_bytes = src_path.read_bytes()
        previous_source = {
            "kind": "file",
            "path": str(src_path),
        }
    else:
        previous_bytes = subprocess.check_output(
            ["git", "show",
             f"{PREVIOUS_SELECTOR_COMMIT}:config/selector_table.json"],
            cwd=ROOT)
        previous_source = {
            "kind": "commit",
            "commit": PREVIOUS_SELECTOR_COMMIT,
        }
    previous_sha256 = hashlib.sha256(previous_bytes).hexdigest()
    previous_source["sha256"] = previous_sha256
    previous_table_path = out_dir / PREVIOUS_TABLE_FILE
    previous_table_path.write_bytes(previous_bytes)

    from src.xpu.config import DeviceProfile, build_manifest, git_commit
    from src.xpu.envcheck import current_env_hash, collect
    from src.xpu.manifest_contract import declared_inventory_sha256
    from src.xpu.selector import load_table

    env = collect()
    env_hash = current_env_hash()
    man = build_manifest(DeviceProfile.load()).to_dict()
    man.update({
        "env_hash": env_hash,
        "protocol_id": PROTOCOL_ID,
        "timing_scope": TIMING_SCOPE,
        "selection_semantics": (
            "correctness-first, interleaved randomized contender order, "
            "3+ independent process sessions, 2% incumbent guard, "
            "oracle = bounded measured set, vendor errors preserved"),
        "warmup": args.warmup,
        "trials": args.trials,
        "sessions": args.sessions,
        "seed_base": args.seed,
        "candidates": ["incumbent", "selector", "oracle:*", "vendor",
                       PREVIOUS_CANDIDATE],
        "previous_selector_source_commit": PREVIOUS_SELECTOR_COMMIT,
        "previous_selector_source": previous_source,
        "previous_selector_sha256": previous_sha256,
        "oracle_mode": args.oracle_mode,
        "block_sweep": (list(WINNER_BLOCK_SWEEP)
                        if args.oracle_mode == "winner-block" else None),
        "shapes": [f"N{n}B{b}:{s}" for n, b, s in schedule],
        "error_threshold": 1e-4,
        "repro_command": " ".join(sys.argv),
        "gm_bytes_kind": ("analytical estimate only; not used in this "
                          "matrix's reported metrics"),
        "kernel_time_status": (
            "not collected: this stack exposes reliable sync API wall "
            "time; device/kernel timers unavailable"),
        "source_hash": git_commit(),
        "toolchain": (f"python {sys.version.split()[0]} + "
                      "semantic-matrix-v1"),
    })
    # Issue #9 WP0-B: declare the candidate inventory and the planned
    # oracle names BEFORE any session runs (stamped + hashed here);
    # aggregation and the strict gate validate raw evidence against
    # these declarations instead of letting observation redefine them.
    cur_table = load_table(str(ROOT / "config" / "selector_table.json"))
    # P1-4: bind the measured precision into the evidence identity so
    # reconfirm/repair paths can check it against the summary instead
    # of trusting a self-reported field.
    man["precision"] = cur_table["precision"]
    inventory = _declared_inventory(
        cases, cur_table,
        args.oracle_mode, args.sessions, args.trials, args.warmup,
        args.one_shot_trials, args.e2e_trials,
        DeviceProfile.load().hash(), sem_map=sem_map)
    man["declared_inventory"] = inventory
    man["declared_inventory_sha256"] = declared_inventory_sha256(inventory)
    # WP1: freeze the exact schedule next to the evidence; session
    # children read this file instead of re-deriving from --scope
    sched_cells = [[n, b, list(sem_map.get((n, b), SEMANTICS))
                    if sem_map else list(SEMANTICS)]
                   for n, b in cases]
    (out_dir / "schedule.json").write_text(
        json.dumps({"cells": sched_cells, "protocol_id": PROTOCOL_ID},
                   indent=2) + "\n")
    man["oracle_candidates"] = sorted(
        {c for cell in inventory["cells"] for c in cell["candidates"]
         if c.startswith("oracle:")})
    (write_dir / "manifest.json").write_text(
        json.dumps(man, indent=2, sort_keys=True) + "\n")

    if not args.aggregate_only:
        # Sequential, each in its own freshly exec'd process (never
        # concurrent on one device): session independence without
        # contending for timing.
        child = [sys.executable, str(Path(__file__).resolve()),
                 "--session", "SESSION", "--scope", args.scope,
                 "--trials", str(args.trials),
                 "--warmup", str(args.warmup),
                 "--seed", str(args.seed),
                 "--e2e-trials", str(args.e2e_trials),
                  "--one-shot-trials", str(args.one_shot_trials),
                  "--oracle-mode", args.oracle_mode,
                  "--previous-table", str(previous_table_path),
                  "--out", str(out_dir)]
        for s in range(args.sessions):
            cmd = [str(s) if x == "SESSION" else x for x in child]
            subprocess.run(cmd, check=True, timeout=7200)
            print(f"session done: raw_session{s}.jsonl", flush=True)

    summary = _aggregate(out_dir, cases, args.trials, args.warmup,
                         args.sessions, manifest=man,
                         write_dir=write_dir, sem_map=sem_map)
    # Issue #8 P1-C strict gate: bind the raw evidence hash with the
    # same manifest_contract.result_sha256 the validator recomputes,
    # into both the standalone manifest.json and the summary's
    # embedded manifest copy (the snapshot below inherits it).
    res = result_sha256(out_dir)
    man["result_sha256"] = res
    summary["manifest"]["result_sha256"] = res
    # A2: top manifest carries the same cases binding as the summary's
    # embedded manifest (strict snapshot validation checks both).
    man["cases_sha256"] = summary["manifest"]["cases_sha256"]
    man["oracle_candidates"] = summary["manifest"]["oracle_candidates"]
    (write_dir / "manifest.json").write_text(
        json.dumps(man, indent=2, sort_keys=True) + "\n")
    (write_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _print_summary(summary)

    if args.snapshot:
        rid = args.release_id or (
            f"{time.strftime('%Y%m%d-%H%M%S')}-"
            f"{git_commit()[:8]}-semantic-matrix")
        # _write_snapshot strict-validates the finished artifact; a
        # failure aborts here (its summary copy is already stamped).
        snap = _write_snapshot(out_dir, rid, man, env, args, cases)
        print(f"immutable snapshot: {snap} (strict-validated OK)")


if __name__ == "__main__":
    main()
