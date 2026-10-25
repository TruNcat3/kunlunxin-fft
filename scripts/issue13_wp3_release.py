#!/usr/bin/env python3
"""issue #13 WP3: 16-cell release matrix — old incumbent vs new selector.

Measures the production selector end-to-end on the release matrix
N in {1024, 2048} x B in {1, 16, 64, 256} x {forward, inverse}
(16 exact cells), four paths per cell, >= 3 independent sessions
(same protocol family as the PR#12 holdout: complex64 non-zero-imag
input, one subprocess per (cell, path) with a fresh TRITON_CACHE_DIR,
sync-per-trial warm p50/MAD, spawn order shuffled per session, plus an
adaptive steady-state warmup: warm in batches of 16 until the last 3
batch means are within 2% of the running minimum \u2014 min 320 / cap 800
calls; run 1 with holdout warmup=5 left a 2-step ramp (ends after up
to 144 timed calls, 42/144 plan rows) inside the timed window, which
landed asymmetrically between paths of the same cell and fabricated a
regression at a cell whose dispatch is byte-identical):

  vendor         direct vendor torch.fft (capability baseline; not a
                 plan path)
  incumbent_old  profile-bound plan_fft with the PRE-issue13
                 composition table (entries without the hybrid4
                 ``variant`` stamp) — what was deployed before this
                 wiring
  selector_new   profile-bound plan_fft with the CURRENT composition
                 table — the new selector (hybrid4 at the 2 promoted
                 inverse keys, adapter at the forward key, butterfly/M0
                 elsewhere)
  m0_reference   profile_hash=None abstains to M0 (descriptive only,
                 never a gate)

Exit conditions (issue #13 WP3):

1. every path, cell, session correct (numpy rel_err <= 1e-4), and NO
   silent fallback at the 2 WP1-promoted keys (selector_new there must
   resolve the composition plan; fallback/abstain fails the worker
   closed);
2. on cells outside the WP1-promoted set, the new selector has NO
   confirmed >2% regression vs the old incumbent: confirmed regression
   = the independent-bootstrap CI high of old/new speedup is < 0.98
   (i.e. even the optimistic bound says new is >2% slower) in >= 2
   sessions, or the all-session pooled point is < 0.98;
3. the 2 WP1-promoted keys KEEP their dual-baseline threshold: in
   every session the independent-bootstrap CI LOW of the old/new AND
   vendor/new speedups must exceed 1.02 (WP1 promotion rule);
4. report: all 16 points win/tie/loss, geomean of pooled speedups,
   worst cell, fallback/binding reasons per cell.

Win = pooled point > 1.02 AND point > 1.02 in >= 2 sessions.
Tie = neither win nor confirmed regression.
The 2 WP1-promoted cells (2048x1 inverse, both batches) are EXPECTED
wins; they are reported under the same criteria, not gated.

Outputs (results/p800/issue13_release/):
  wp3_release.json        matrix verdict + exit conditions
  wp3_raw.csv             per-(cell, path, session) warm p50/MAD/rel
  wp3_sessions_raw.json   full raw trial rows (checkpoint, --resume)
  legacy_composition_table.json  pre-issue13 table snapshot used by
                          the incumbent_old path

Usage:
    CUDA_VISIBLE_DEVICES=2 python3 scripts/issue13_wp3_release.py
    ... --resume          # skip sessions already in the checkpoint
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "results/p800/issue13_release"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT = OUT_DIR / "wp3_sessions_raw.json"
LEGACY_TABLE = OUT_DIR / "legacy_composition_table.json"
SCRIPT = ROOT / "scripts/issue13_wp3_release.py"
SELECTOR_TABLE = ROOT / "config/selector_table.json"
COMPOSITION_TABLE = ROOT / "config/composition_table.json"
SEL_SRC = ROOT / "src/xpu/selector.py"
COMP_SRC = ROOT / "src/xpu/composition.py"
ENTRY_SRC = ROOT / "src/xpu/entry.py"
H4_SRC = ROOT / "src/xpu/hybrid4.py"
MARK = "ISSUE13_WP3_ROW:"

CELLS = [(n, b, inv) for n in (1024, 2048) for b in (1, 16, 64, 256)
         for inv in (0, 1)]
PATHS = ("vendor", "incumbent_old", "selector_new", "m0_reference")
SESSIONS = (1, 2, 3)
SEED = 13_102026
SPAWN_SEED = SEED + 1300
GATE_CI_LOW = 1.02          # win threshold (speedup point)
REGRESSION_PTS = 0.98       # confirmed-regression threshold (old/new)
ERR_MAX = 1e-4
BOOTSTRAP = 400
PROMOTED = {(2048, 1, 1), (2048, 16, 1)}   # WP1 promoted keys (n,b,inv)

# steady-state warmup (added after run 1: WARMUP=5 left a 2-step ramp
# inside the timed window on 42/144 plan-path rows — transition only
# reached after up to 144 timed calls, and with one subprocess per path
# the ramp landed asymmetrically between incumbent_old/selector_new of
# the same cell/session, fabricating "regressions" at cells whose
# dispatch is byte-identical)
STEADY_BATCH = 16          # calls per warmup batch
STEADY_MIN_CALLS = 320     # never check before this many warmup calls
                           # (run-1 ramps end by absolute call ~151;
                           # 20 batches keeps a wide margin and the
                           # middle plateau can never be mistaken for
                           # steady state at the first legal check)
STEADY_MAX_CALLS = 800     # hard cap (ramps observed end by ~150 calls)
STEADY_TOL = 0.02          # batch mean within 2% of the running minimum
STEADY_NEED = 3            # ... for this many consecutive batches


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_holdout():
    spec = importlib.util.spec_from_file_location(
        "issue13_holdout", ROOT / "scripts/issue10_wp3_holdout.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["issue13_holdout"] = m
    spec.loader.exec_module(m)
    return m


def _load_wp3():
    spec = importlib.util.spec_from_file_location(
        "issue13_wp3_perf", ROOT / "scripts/issue10_wp3_perf.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["issue13_wp3_perf"] = m
    spec.loader.exec_module(m)
    return m


def _write_legacy_table() -> None:
    """Snapshot the pre-issue13 composition table (entries that never
    carried the hybrid4 variant stamp) for the incumbent_old path."""
    from src.xpu.manifest_contract import rows_sha256
    cur = json.loads(COMPOSITION_TABLE.read_text())
    legacy_entries = [e for e in cur["entries"] if "variant" not in e]
    legacy = dict(cur)
    legacy["entries"] = legacy_entries
    legacy["result_sha256"] = rows_sha256({"rows": legacy_entries})
    legacy["_note"] = ("pre-issue13 deployed state: entries without the "
                       "hybrid4 variant stamp (snapshot taken at WP3 run "
                       "time from config/composition_table.json)")
    LEGACY_TABLE.write_text(json.dumps(legacy, indent=2, sort_keys=True)
                            + "\n")
    assert [e.get("flow_id") for e in legacy_entries] == ["hybrid-A1P1"], \
        legacy_entries


# ------------------------------------------------------------- worker

def steady_warmup_done(batch_means, tol=STEADY_TOL, need=STEADY_NEED) -> bool:
    """True once the last `need` batch means are all within (1+tol) of
    the minimum seen so far (the ramp never comes back after that in
    run 1; min/max call bounds are enforced by the caller)."""
    if len(batch_means) < need:
        return False
    gmin = min(batch_means)
    return all(m <= gmin * (1.0 + tol) for m in batch_means[-need:])


def worker(a) -> int:
    import torch
    H = _load_holdout()
    wp3 = _load_wp3()
    n, b, inverse, p = a.n, a.b, a.inverse, a.path
    out = {"session": a.session, "n": n, "batch": b,
           "direction": "c2c_inverse" if inverse else "c2c_forward",
           "path": p}
    stage = "import/context"
    try:
        with wp3.xpu_context(0):
            torch.cuda.synchronize()
        context_ms = (time.perf_counter() - H.PROCESS_T0) * 1e3

        stage = "input/contract"
        seed = H._seed_for(a.session, n, b)
        x = H._input_c(b, n, seed)
        wp3._vendor_canary()
        contract = H._contract_c(x, n, b, inverse)

        stage = "setup"
        segs = {"context_ms": context_ms}
        xb = x.squeeze(0) if b == 1 else x
        t0 = time.perf_counter()
        plan = None
        if p == "vendor":
            call, meta = wp3.run_vendor(xb, inverse)
        else:
            from src.xpu.entry import execute_fft_plan, plan_fft
            from src.xpu.selector import load_table
            table = load_table(str(SELECTOR_TABLE))
            if p == "incumbent_old":
                # pre-issue13 deployed state: monkeypatch the module-
                # global loader composition_select resolves through
                # (call-time lookup) so plan_fft sees the snapshot.
                import src.xpu.composition as comp_mod
                legacy = json.loads(LEGACY_TABLE.read_text())
                comp_mod.load_composition_table = (
                    lambda path=None: legacy)
                prof = legacy.get("profile_hash")
            elif p == "selector_new":
                prof = table.get("profile_hash")
            else:  # m0_reference
                prof = None
            tp0 = time.perf_counter()
            plan = plan_fft(direction="dit", n=n, batch=b,
                            profile_hash=prof, table=table,
                            env_hash="auto",
                            transform_type="c2c",
                            transform_direction=("inverse" if inverse
                                                 else "forward"),
                            norm="backward")
            segs["plan_fft_ms"] = (time.perf_counter() - tp0) * 1e3
            params = dict(plan.params or {})
            binding = {
                "path": p,
                "profile_hash": prof,
                "selection_source": plan.selection_source,
                "fallback": bool(plan.fallback),
                "executed": plan.executed,
                "flow_id": plan.flow_id,
                "flow_realization": plan.flow_realization,
                "processing_unit": plan.processing_unit,
                "block": plan.block,
                "plan_reason": str(plan.reason)[:300],
                "composition": params.get("flow_kind") == "composition",
                "variant": params.get("variant"),
                "params": {k: str(v)[:100]
                           for k, v in sorted(params.items())},
            }
            if p == "incumbent_old" and (plan.fallback
                                         or plan.selection_source
                                         == "abstain"):
                # pre-issue13 incumbent no-go: the old deployed selector
                # must resolve a real (non-M0) plan in every cell
                stage = "incumbent_old"
                raise RuntimeError(
                    "pre-issue13 incumbent no-go: selection_source="
                    f"{plan.selection_source} fallback={plan.fallback} "
                    f"reason={plan.reason}")
            if (p == "selector_new" and (n, b, inverse) in PROMOTED
                    and (plan.fallback
                         or plan.selection_source == "abstain")):
                # promoted keys are wired into the composition table:
                # falling back to M0 here would be a SILENT fallback
                # (issue #13 WP3 exit condition 1) -> fail closed
                stage = "selector_new_promoted"
                raise RuntimeError(
                    "promoted-key silent fallback: selection_source="
                    f"{plan.selection_source} fallback={plan.fallback} "
                    f"reason={plan.reason}")

            def call():
                y = execute_fft_plan(plan, xb)
                if y.dim() == 1:
                    y = y.reshape(b, n)
                return y.real, y.imag

            meta = {"launches": "production entry execute_fft_plan",
                    **binding}
        # holdout _API_ENTRY keys the plan paths under "incumbent_entry"
        # (plan_fft + execute_fft_plan production entry)
        meta.update(H._entry_meta(
            "incumbent_entry" if p in ("incumbent_old", "selector_new")
            else p, xb, b))
        segs["setup_ms"] = (time.perf_counter() - t0) * 1e3

        stage = "warm_probe"
        with torch.no_grad():
            if p == "vendor":
                t1 = time.perf_counter()
                y = (torch.fft.ifft(xb.clone()) if inverse
                     else torch.fft.fft(xb.clone()))
                del y
                torch.cuda.synchronize()
                segs["vendor_plan_probe_ms"] = (
                    time.perf_counter() - t1) * 1e3
            else:
                t1 = time.perf_counter()
                y = execute_fft_plan(plan, xb.clone())
                del y
                torch.cuda.synchronize()
                segs["entry_kernel_probe_ms"] = (
                    time.perf_counter() - t1) * 1e3
        probe_total = sum(v for k, v in segs.items()
                          if k.endswith("_probe_ms"))

        stage = "first_call"
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y_re, y_im = call()
        torch.cuda.synchronize()
        segs["first_call_ms"] = (time.perf_counter() - t0) * 1e3
        # reconstruction of a true unsegmented one-shot (descriptive;
        # same formula as the holdout — gate uses steady state only)
        segs["one_shot_ms"] = (segs["setup_ms"] + probe_total
                               + segs["first_call_ms"])

        stage = "correctness"
        err = float(wp3._check_batched(y_re, y_im, x, inverse))

        stage = "steady_warmup"
        wu = {"steady_warmup_calls": 0, "steady_warmup_ms": 0.0,
              "steady_warmup_batches": [], "steady_warmup_reached": False}
        tw0 = time.perf_counter()
        with torch.no_grad():
            while wu["steady_warmup_calls"] < STEADY_MAX_CALLS:
                torch.cuda.synchronize()
                tb = time.perf_counter()
                for _ in range(STEADY_BATCH):
                    call()
                torch.cuda.synchronize()
                wu["steady_warmup_batches"].append(
                    (time.perf_counter() - tb) * 1e3 / STEADY_BATCH)
                wu["steady_warmup_calls"] += STEADY_BATCH
                if (wu["steady_warmup_calls"] >= STEADY_MIN_CALLS
                        and steady_warmup_done(wu["steady_warmup_batches"])):
                    wu["steady_warmup_reached"] = True
                    break
        wu["steady_warmup_ms"] = (time.perf_counter() - tw0) * 1e3
        segs.update(wu)

        stage = "steady"
        trials = H._trials(wp3, call)
        out.update({
            "contract": contract,
            "meta": meta,
            "segments": segs,
            "trials": trials,
            "rel_err": err,
            "correct": bool(err <= ERR_MAX),
            "fail_closed": False,
        })
    except Exception as e:  # noqa: BLE001 — recorded as fail-closed
        out.update({"fail_closed": True, "error": f"{stage}: {e}"[:300]})
    print(MARK + json.dumps(out), flush=True)
    return 0


def _spawn(session: int, n: int, b: int, inverse: bool,
           path: str) -> dict | None:
    cache = tempfile.mkdtemp(prefix=f"wp3_s{session}_{path}_")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "2"
    env["TRITON_CACHE_DIR"] = cache
    env["PYTHONUNBUFFERED"] = "1"
    env["ISSUE13_LEGACY_TABLE"] = str(LEGACY_TABLE)
    try:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--worker",
             "--session", str(session), "--n", str(n), "--b", str(b),
             "--inverse", str(int(inverse)), "--path", path],
            capture_output=True, text=True, timeout=1200, env=env,
            cwd=str(ROOT))
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    for line in (proc.stdout or "").splitlines():
        if line.startswith(MARK):
            try:
                return json.loads(line[len(MARK):])
            except json.JSONDecodeError:
                return None
    print(f"  worker fail n={n} b={b} inv={int(inverse)} s={session} "
          f"path={path} rc={proc.returncode} "
          f"stderr={(proc.stderr or '')[-300:]}", flush=True)
    return None


# ------------------------------------------------------- aggregation

def _p50(xs):
    ys = sorted(xs)
    return ys[len(ys) // 2] if ys else None


def classify_cell(points: list[float], cis: list[dict],
                  promoted: bool) -> str:
    """win | tie | confirmed-regression for one cell.

    points/cis: per-session speedup old/new (>1 = new faster) and the
    independent bootstrap CI of that ratio ({"low","high"}).
    """
    if not points:
        return "tie"
    pooled = sorted(points)[len(points) // 2] if len(points) % 2 \
        else 0.5 * (sorted(points)[len(points) // 2 - 1]
                    + sorted(points)[len(points) // 2])
    n_reg = sum(1 for sp in cis
                if sp and float(sp["high"]) < REGRESSION_PTS)
    if pooled < REGRESSION_PTS or n_reg >= 2:
        return "confirmed-regression"
    n_win = sum(1 for p in points if p > GATE_CI_LOW)
    if pooled > GATE_CI_LOW and n_win >= 2:
        return "win"
    return "tie"


def _geomean(xs: list[float]) -> float | None:
    ys = [x for x in xs if x and x > 0]
    if not ys:
        return None
    return math.exp(sum(math.log(x) for x in ys) / len(ys))


def exit_conditions(cells_out: list[dict]) -> dict:
    """WP3 exit conditions from the aggregated matrix."""
    bad_correct = [
        c["label"] for c in cells_out
        if any(not fl.get("all_paths_correct")
               for fl in c["gate_flags_by_session"])]
    regressions = [
        {"label": c["label"], "verdict": c["comparison"]["verdict"],
         "pooled": c["comparison"]["pooled_speedup"],
         "points": c["comparison"]["points"]}
        for c in cells_out
        if (c["n"], c["batch"], c["inverse"]) not in PROMOTED
        and c["comparison"]["verdict"] == "confirmed-regression"]
    pooled = [c["comparison"]["pooled_speedup"] for c in cells_out
              if c["comparison"]["pooled_speedup"]]
    # promoted keys must hold the WP1 dual-baseline threshold (CI low
    # > 1.02 vs vendor AND incumbent_old in every session)
    dual_fail = [
        {"label": c["label"],
         "vendor_cis": (c["comparison"].get("dual_baseline")
                        or {}).get("vendor_cis"),
         "incumbent_cis": (c["comparison"].get("dual_baseline")
                           or {}).get("incumbent_cis")}
        for c in cells_out
        if (c["n"], c["batch"], c["inverse"]) in PROMOTED
        and not c["comparison"].get("dual_baseline_pass")]
    return {
        "correctness_all_paths": not bad_correct,
        "bad_correctness_cells": bad_correct,
        "no_confirmed_regression_on_uncovered_cells": not regressions,
        "regressions": regressions,
        "promoted_dual_baseline": not dual_fail,
        "promoted_dual_baseline_failures": dual_fail,
        "geomean_speedup_old_over_new": _geomean(pooled),
        "worst_cell": (min(cells_out,
                           key=lambda c: c["comparison"]["pooled_speedup"]
                          or 9e9)["label"]
                       if cells_out else None),
        "pass": (not bad_correct and not regressions and not dual_fail),
    }


def _session_flags(rows: dict) -> dict:
    present = all(p in rows for p in PATHS)
    fc = [p for p in PATHS
          if p in rows and (rows[p].get("fail_closed")
                            or not rows[p].get("correct"))]
    shapes = {p: list((rows[p].get("meta") or {})
                      .get("input_shape") or []) for p in PATHS
              if p in rows}
    return {"all_paths_present": present,
            "no_fail_closed": not fc,
            "all_paths_correct": present and not fc,
            "fail_closed_paths": fc,
            "entry_shapes": shapes}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--session", type=int, default=1)
    ap.add_argument("--n", type=int)
    ap.add_argument("--b", type=int)
    ap.add_argument("--inverse", type=int, default=0)
    ap.add_argument("--path", type=str, default="")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.worker:
        return worker(args)

    from src.xpu.selector import (assert_current_environment,
                                  load_table)
    table = load_table(str(SELECTOR_TABLE))
    live_env = assert_current_environment(table)
    profile_hash = table.get("profile_hash")
    print(f"env ok: profile={profile_hash} env={live_env}", flush=True)
    _write_legacy_table()
    print(f"legacy table -> {LEGACY_TABLE}", flush=True)

    raw: list = []
    sessions_meta: list = []
    done_sessions: set[int] = set()
    if args.resume and CHECKPOINT.is_file():
        ck = json.loads(CHECKPOINT.read_text())
        raw = ck.get("raw", [])
        sessions_meta = ck.get("sessions", [])
        done_sessions = {int(s.get("session")) for s in sessions_meta}
        print(f"resume: {len(raw)} rows, sessions {sorted(done_sessions)}",
              flush=True)

    rng = random.Random(SPAWN_SEED)
    for s in SESSIONS:
        if s in done_sessions:
            continue
        order = [(n, b, inv, p) for n, b, inv in CELLS for p in PATHS]
        rng.shuffle(order)
        ts = time.time()
        ok = 0
        for n, b, inv, p in order:
            r = _spawn(s, n, b, bool(inv), p)
            if r is not None:
                raw.append(r)
                if not r.get("fail_closed"):
                    ok += 1
        sessions_meta.append({
            "session": s, "role": "protocol",
            "jobs_ok": ok, "jobs_total": len(order),
            "elapsed_s": round(time.time() - ts, 1),
            "spawn_seed": SPAWN_SEED,
            "isolation": ("one subprocess per (cell, path) with its "
                          "own fresh TRITON_CACHE_DIR; spawn order "
                          "shuffled per session"),
        })
        CHECKPOINT.write_text(json.dumps({"raw": raw,
                                          "sessions": sessions_meta}))
        print(f"session {s}: {ok}/{len(order)} jobs "
              f"({time.time() - ts:.0f}s)", flush=True)

    # index rows: (n,b,dir,session) -> {path: row}
    idx: dict = {}
    for r in raw:
        key = (r.get("n"), r.get("batch"), r.get("direction"),
               r.get("session"))
        idx.setdefault(key, {})[r.get("path")] = r

    import numpy as np

    rng2 = np.random.default_rng(SEED)
    cells_out = []
    csv_rows = []
    for n, b, inv in CELLS:
        d = "c2c_inverse" if inv else "c2c_forward"
        label = f"N{n}B{b}:{d}"
        c = {"label": label, "n": n, "batch": b, "inverse": bool(inv),
             "direction": d, "sessions": {}, "gate_flags_by_session": []}
        pts, cis, per_session = [], [], {}
        vendor_pts, vendor_cis, per_session_vendor = [], [], {}
        for s in SESSIONS:
            rows = idx.get((n, b, d, s), {})
            flags = _session_flags(rows)
            c["gate_flags_by_session"].append(flags)
            entry = {"flags": flags, "paths": {}}
            if flags["all_paths_present"] and flags["no_fail_closed"]:
                for pth in PATHS:
                    pr = rows[pth]
                    meta = pr.get("meta") or {}
                    pv = {"warm_p50_ms": H_p50(pr["trials"]),
                          "warm_mad_ms": H_mad(pr["trials"]),
                          "rel_err": pr["rel_err"],
                          "correct": pr["correct"],
                          "one_shot_ms": (pr.get("segments") or {})
                          .get("one_shot_ms"),
                          "binding": {k: meta.get(k) for k in (
                              "selection_source", "fallback", "executed",
                              "flow_id", "composition", "variant",
                              "plan_reason")}}
                    entry["paths"][pth] = pv
                    csv_rows.append({
                        "label": label, "session": s, "path": pth,
                        "warm_p50_ms": pv["warm_p50_ms"],
                        "warm_mad_ms": pv["warm_mad_ms"],
                        "rel_err": pv["rel_err"],
                        "correct": pv["correct"],
                        "executed": meta.get("executed"),
                        "selection_source": meta.get("selection_source"),
                        "composition": meta.get("composition"),
                    })
                old = rows["incumbent_old"]["trials"]
                new = rows["selector_new"]["trials"]
                sp = H_boot(rng2, old, new)
                if sp:
                    sp["definition"] = (
                        "median(incumbent_old)/median(selector_new) "
                        "independent bootstrap (>1 = new selector "
                        "faster)")
                    pts.append(sp["point"])
                    cis.append(sp)
                per_session[s] = sp
                # WP1 dual-baseline kept for promoted keys: vendor
                # speedup of the new selector, same bootstrap family
                spv = H_boot(rng2, rows["vendor"]["trials"], new)
                if spv:
                    spv["definition"] = (
                        "median(vendor)/median(selector_new) "
                        "independent bootstrap (>1 = new selector "
                        "faster)")
                    vendor_pts.append(spv["point"])
                    vendor_cis.append(spv)
                per_session_vendor[s] = spv
            c["sessions"][str(s)] = entry
        pooled_point = _p50(pts) if pts else None
        verdict = classify_cell(pts, cis,
                                promoted=(n, b, inv) in PROMOTED)
        vendor_p50s, old_p50s, new_p50s, m0_p50s = [], [], [], []
        for s in SESSIONS:
            paths = (c["sessions"].get(str(s), {}) or {}).get("paths") or {}
            for store, key in ((vendor_p50s, "vendor"),
                               (old_p50s, "incumbent_old"),
                               (new_p50s, "selector_new"),
                               (m0_p50s, "m0_reference")):
                if key in paths and paths[key]["warm_p50_ms"] is not None:
                    store.append(paths[key]["warm_p50_ms"])
        sel_binding = ((c["sessions"].get(str(SESSIONS[0]), {}) or {})
                       .get("paths", {}).get("selector_new", {})
                       .get("binding") or {})
        # WP1 promoted-key dual baseline: per-session independent-
        # bootstrap CI low must exceed 1.02 vs BOTH vendor and
        # incumbent_old (issue #13 WP3 exit condition: keep the dual
        # baseline for promoted keys). Held for every session that
        # produced a CI; if any promoted session lacks a CI, the
        # baseline cannot be established -> False.
        dual_pass = True
        dual_detail = {"vendor_cis": vendor_cis, "incumbent_cis": cis}
        if (n, b, inv) in PROMOTED:
            dual_pass = (
                len(vendor_cis) == len(SESSIONS)
                and len(cis) == len(SESSIONS)
                and all(float(sp["low"]) > GATE_CI_LOW
                        for sp in vendor_cis)
                and all(float(sp["low"]) > GATE_CI_LOW
                        for sp in cis))
        c["comparison"] = {
            "metric": "median_incumbent_old / median_selector_new",
            "definition": ("per-session independent bootstrap of "
                           "warm p50; pooled = median of session "
                           "points"),
            "points": pts,
            "cis": cis,
            "pooled_speedup": pooled_point,
            "verdict": verdict,
            "promoted_key": (n, b, inv) in PROMOTED,
            "dual_baseline_pass": dual_pass if (n, b, inv) in PROMOTED
            else None,
            "dual_baseline": dual_detail if (n, b, inv) in PROMOTED
            else None,
            "selector_binding": sel_binding,
            "fallback_reason": (
                sel_binding.get("plan_reason")
                if sel_binding.get("fallback")
                or sel_binding.get("selection_source") in (
                    "abstain", None) else None),
            "path_p50_ms": {
                "vendor": _p50(vendor_p50s),
                "incumbent_old": _p50(old_p50s),
                "selector_new": _p50(new_p50s),
                "m0_reference": _p50(m0_p50s),
            },
        }
        cells_out.append(c)

    ec = exit_conditions(cells_out)
    doc = {
        "schema": "issue13-wp3-release-v1",
        "issue": "kunlunxin-fft#13",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": str(SCRIPT.relative_to(ROOT)),
        "environment": {
            "profile_hash": profile_hash,
            "env_hash_live": live_env,
            "selector_table_sha256": _sha256(SELECTOR_TABLE),
            "composition_table_sha256": _sha256(COMPOSITION_TABLE),
            "legacy_table_sha256": _sha256(LEGACY_TABLE),
            "producer_sha256": {
                "scripts/issue13_wp3_release.py": _sha256(SCRIPT),
                "src/xpu/composition.py": _sha256(COMP_SRC),
                "src/xpu/entry.py": _sha256(ENTRY_SRC),
                "src/xpu/hybrid4.py": _sha256(H4_SRC),
                "src/xpu/selector.py": _sha256(SEL_SRC),
            },
        },
        "protocol": {
            "cells": [{"N": n, "batch": b, "inverse": bool(inv)}
                      for n, b, inv in CELLS],
            "paths": list(PATHS),
            "sessions": list(SESSIONS),
            "seed": SEED, "spawn_seed": SPAWN_SEED,
            "bootstrap": BOOTSTRAP,
            "steady_warmup": {
                "batch_calls": STEADY_BATCH,
                "min_calls": STEADY_MIN_CALLS,
                "max_calls": STEADY_MAX_CALLS,
                "tol": STEADY_TOL,
                "consecutive_batches": STEADY_NEED,
                "holdout_warmup_calls": 5,
                "reason": ("run 1 (holdout warmup=5 only) left a 2-step "
                           "ramp inside the timed window on 42/144 "
                           "plan-path rows; asymmetric ramp placement "
                           "between paths fabricated a confirmed "
                           "regression at N2048B1:c2c_forward where "
                           "incumbent_old/selector_new dispatch is "
                           "byte-identical — run 1 archived under "
                           "run1-warmup5/"),
            },
            "input_semantics": ("complex64 non-zero imag, contiguous, "
                                "axis=-1, seeded per (session, n, b); "
                                "rank-1 (N,) for b==1, rank-2 (B,N) "
                                "otherwise — the production entry "
                                "contract"),
            "gates": {
                "win_point": GATE_CI_LOW,
                "regression_point": REGRESSION_PTS,
                "err_max": ERR_MAX,
                "promoted_keys": sorted(f"N{n}B{b}:{'inv' if inv else 'fwd'}"
                                        for n, b, inv in PROMOTED),
                "promoted_dual_baseline": (
                    "per-session independent-bootstrap CI low > 1.02 "
                    "vs BOTH vendor and incumbent_old (WP1 rule)"),
            },
        },
        "sessions": sessions_meta,
        "cells": cells_out,
        "exit_conditions": ec,
    }
    OUT_DIR.joinpath("wp3_release.json").write_text(
        json.dumps(doc, indent=2, sort_keys=True) + "\n")
    with open(OUT_DIR / "wp3_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "label", "session", "path", "warm_p50_ms", "warm_mad_ms",
            "rel_err", "correct", "executed", "selection_source",
            "composition"])
        w.writeheader()
        w.writerows(csv_rows)
    print(json.dumps({
        "cells": len(cells_out),
        "verdicts": {c["label"]: c["comparison"]["verdict"]
                     for c in cells_out},
        "exit_conditions": ec,
    }, indent=1), flush=True)
    print(f"-> {OUT_DIR / 'wp3_release.json'}", flush=True)


def H_p50(xs):
    ys = sorted(xs)
    return ys[len(ys) // 2] if ys else None


def H_mad(xs):
    m = H_p50(xs)
    return None if m is None else H_p50([abs(v - m) for v in xs])


def H_boot(rng, num, den, n_boot=BOOTSTRAP):
    """Independent bootstrap CI of median(num)/median(den).

    Call as H_boot(rng, incumbent_old, selector_new): point > 1 means
    the NEW selector is faster (matches classify_cell's 1.02/0.98
    gates and the recorded definition string).
    """
    import numpy as np
    a = np.asarray(num, dtype=float)
    b = np.asarray(den, dtype=float)
    if a.size == 0 or b.size == 0:
        return None
    num_m = np.median(a)
    den_m = np.median(b)
    if den_m <= 0:
        return None
    point = float(num_m / den_m)
    ratios = []
    for _ in range(n_boot):
        sa = float(np.median(a[rng.integers(0, a.size, a.size)]))
        sb = float(np.median(b[rng.integers(0, b.size, b.size)]))
        if sb > 0:
            ratios.append(sa / sb)
    if not ratios:
        return None
    ratios.sort()
    low = ratios[int(0.025 * len(ratios))]
    high = ratios[int(0.975 * len(ratios))]
    return {"point": point, "low": float(low), "high": float(high),
            "n_boot": n_boot, "bootstrap_kind": "independent"}


if __name__ == "__main__":
    main()
