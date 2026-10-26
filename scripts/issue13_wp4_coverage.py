#!/usr/bin/env python3
"""issue #13 WP4: coverage grid — 49-cell forward-C2C matrix + 12-cell
long-size slice (issue #13 sections 4.1-4.2).

Grid (same public forward-C2C grid as AscendFFT):

  C2C  N={64,128,256,512,1024,2048,4096} x B={1,4,16,64,256,1024,4096}
       -> 49 cells, correctness + current selector everywhere;
       vendor-successful points enter the same-semantics speedup
       statistics (median(vendor)/median(selector_new) independent
       bootstrap per session, pooled = median of session points);
       vendor-unsupported/failed points are kept EXPLICITLY (label +
       reason) — never filled in or dropped from the denominator.
  LONG N={8192,16384,32768,65536} x B={1,16,256} -> 12 points;
       correctness, latency, resolved path and memory/failure reason
       only — no vendor speedup verdicts.

Paths per cell (3 sessions, one subprocess per (cell, path) with a
fresh TRITON_CACHE_DIR, spawn order shuffled per session — same
protocol family as WP3 incl. the adaptive steady-state warmup):

  vendor          torch.fft last-dim op (capability baseline)
  incumbent_old   profile-bound plan_fft with the PRE-issue13
                  composition table (shared WP3 legacy snapshot)
  selector_new    profile-bound plan_fft with the CURRENT table

Heavy cells (>= HEAVY_CALL_MS per-call, e.g. M0-fallback long rows at
~0.2-1.6s/call) downshift to a bounded protocol variant (shorter
adaptive steady warmup, 3x32 sync-per-trial trials), recorded
per row as "protocol_variant": at that scale the launch/clock ramp
cannot contaminate the median. Verdicts: win/tie/loss with
win = pooled > 1.02 and > 1.02 in >= 2 sessions; loss = the same
confirmed-regression criterion as WP3 (pooled < 0.98 or >= 2 session
bootstrap CI-highs < 0.98) — i.e. vendor is >2% faster; else tie.
The promoted-keys fail-closed guard is inert on the forward grid
(the promoted keys are inverse cells) but kept for symmetry.

Coverage stats reported: complete per-point table, win/tie/loss
counts, vendor-supported geomean of vendor/selector AND
vendor/incumbent, fallback ratio (selector fallback/abstain cells)
and the retained vendor-unsupported/selector-failed lists.

Outputs (results/p800/issue13_release/):
  wp4_coverage.json       per-point tables + coverage stats + claims
  wp4_raw.csv             per-(point, path, session) warm stats
  wp4_sessions_raw.json   full raw trial rows (checkpoint, --resume)
  legacy_composition_table.json  shared pre-issue13 snapshot (WP3)

Usage:
    CUDA_VISIBLE_DEVICES=2 python3 scripts/issue13_wp4_coverage.py
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
CHECKPOINT = OUT_DIR / "wp4_sessions_raw.json"
LEGACY_TABLE = OUT_DIR / "legacy_composition_table.json"
SCRIPT = ROOT / "scripts/issue13_wp4_coverage.py"
SELECTOR_TABLE = ROOT / "config/selector_table.json"
COMPOSITION_TABLE = ROOT / "config/composition_table.json"
SEL_SRC = ROOT / "src/xpu/selector.py"
COMP_SRC = ROOT / "src/xpu/composition.py"
ENTRY_SRC = ROOT / "src/xpu/entry.py"
H4_SRC = ROOT / "src/xpu/hybrid4.py"
MARK = "ISSUE13_WP4_ROW:"

C2C_NS = (64, 128, 256, 512, 1024, 2048, 4096)
C2C_BS = (1, 4, 16, 64, 256, 1024, 4096)
C2C_CELLS = [(n, b) for n in C2C_NS for b in C2C_BS]      # 49, forward
LONG_NS = (8192, 16384, 32768, 65536)
LONG_BS = (1, 16, 256)
LONG_CELLS = [(n, b) for n in LONG_NS for b in LONG_BS]   # 12, forward
CELLS = C2C_CELLS + LONG_CELLS                            # 61, all forward
PATHS = ("vendor", "incumbent_old", "selector_new")
SESSIONS = (1, 2, 3)
SEED = 14_102026
SPAWN_SEED = SEED + 1400
WIN_POINT = 1.02
LOSS_POINT = 0.98         # confirmed-loss threshold (vendor/new)
ERR_MAX = 1e-4
# heavy-cell protocol downshift (smoke: N4096B4096 selector ~1.6s/call
# would spend ~500s in steady warmup + ~260s in trials at the full
# protocol; at >=30ms/call the launch/clock ramp is negligible
# relative to the call itself)
HEAVY_CALL_MS = 30.0
HEAVY_STEADY_MIN_CALLS = 64
HEAVY_STEADY_MAX_CALLS = 96
HEAVY_STEADY_NEED = 2
HEAVY_ROUNDS = 3
PROMOTED = {(2048, 1, 1), (2048, 16, 1)}   # WP1 promoted keys (n,b,inv)
                                           # — inverse cells, inert here


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_wp3():
    spec = importlib.util.spec_from_file_location(
        "issue13_wp3_release", ROOT / "scripts/issue13_wp3_release.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["issue13_wp3_release"] = m
    spec.loader.exec_module(m)
    return m


def _load_holdout():
    spec = importlib.util.spec_from_file_location(
        "issue13_holdout4", ROOT / "scripts/issue10_wp3_holdout.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["issue13_holdout4"] = m
    spec.loader.exec_module(m)
    return m


def _load_perf():
    spec = importlib.util.spec_from_file_location(
        "issue13_wp3_perf4", ROOT / "scripts/issue10_wp3_perf.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["issue13_wp3_perf4"] = m
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------- worker

def worker(a) -> int:
    import torch
    H = _load_holdout()
    wp3 = _load_perf()
    w3 = _load_wp3()
    n, b, p = a.n, a.b, a.path
    kind = "long" if n > 4096 else "c2c"
    out = {"session": a.session, "n": n, "batch": b, "kind": kind,
           "direction": "c2c_forward", "path": p}
    stage = "import/context"
    try:
        with wp3.xpu_context(0):
            torch.cuda.synchronize()
        context_ms = (time.perf_counter() - H.PROCESS_T0) * 1e3

        stage = "input/contract"
        seed = H._seed_for(a.session, n, b)
        x = H._input_c(b, n, seed)
        wp3._vendor_canary()
        contract = H._contract_c(x, n, b, False)

        stage = "setup"
        segs = {"context_ms": context_ms}
        xb = x.squeeze(0) if b == 1 else x
        t0 = time.perf_counter()
        plan = None
        if p == "vendor":
            call, meta = wp3.run_vendor(xb, False)
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
            else:  # not a WP4 path
                raise ValueError(f"unknown path {p}")
            tp0 = time.perf_counter()
            plan = plan_fft(direction="dit", n=n, batch=b,
                            profile_hash=prof, table=table,
                            env_hash="auto",
                            transform_type="c2c",
                            transform_direction="forward",
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
            # WP4 is coverage: an incumbent abstain/fallback at sizes
            # outside the deployed range is recorded as data (binding
            # + fallback ratio), never raised — unlike WP3 where the
            # no-go bound the 16 release cells.
            if (p == "selector_new" and (n, b, 0) in PROMOTED
                    and (plan.fallback
                         or plan.selection_source == "abstain")):
                # inert on the forward grid (promoted keys are inverse
                # cells) but kept for protocol symmetry with WP3
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
        meta.update(H._entry_meta(
            "incumbent_entry" if p in ("incumbent_old", "selector_new")
            else p, xb, b))
        segs["setup_ms"] = (time.perf_counter() - t0) * 1e3

        stage = "warm_probe"
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:  # noqa: BLE001 — mem telemetry is best-effort
            pass
        with torch.no_grad():
            if p == "vendor":
                t1 = time.perf_counter()
                y = torch.fft.fft(xb.clone())
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
        segs["one_shot_ms"] = (segs["setup_ms"] + probe_total
                               + segs["first_call_ms"])
        try:
            segs["peak_mem_mb"] = round(
                torch.cuda.max_memory_allocated() / 2 ** 20, 1)
        except Exception:  # noqa: BLE001 — best-effort telemetry
            pass

        stage = "correctness"
        err = float(wp3._check_batched(y_re, y_im, x, False))

        heavy = segs["first_call_ms"] > HEAVY_CALL_MS
        stage = "steady_warmup"
        w_min = HEAVY_STEADY_MIN_CALLS if heavy else w3.STEADY_MIN_CALLS
        w_max = HEAVY_STEADY_MAX_CALLS if heavy else w3.STEADY_MAX_CALLS
        w_need = HEAVY_STEADY_NEED if heavy else w3.STEADY_NEED
        wu = {"steady_warmup_calls": 0, "steady_warmup_ms": 0.0,
              "steady_warmup_batches": [], "steady_warmup_reached": False}
        tw0 = time.perf_counter()
        with torch.no_grad():
            while wu["steady_warmup_calls"] < w_max:
                if heavy:
                    torch.cuda.synchronize()
                    tb = time.perf_counter()
                    call()
                    torch.cuda.synchronize()
                    wu["steady_warmup_batches"].append(
                        (time.perf_counter() - tb) * 1e3)
                    wu["steady_warmup_calls"] += 1
                else:
                    torch.cuda.synchronize()
                    tb = time.perf_counter()
                    for _ in range(w3.STEADY_BATCH):
                        call()
                    torch.cuda.synchronize()
                    wu["steady_warmup_batches"].append(
                        (time.perf_counter() - tb) * 1e3 / w3.STEADY_BATCH)
                    wu["steady_warmup_calls"] += w3.STEADY_BATCH
                if (wu["steady_warmup_calls"] >= w_min
                        and w3.steady_warmup_done(
                            wu["steady_warmup_batches"], need=w_need)):
                    wu["steady_warmup_reached"] = True
                    break
        wu["steady_warmup_ms"] = (time.perf_counter() - tw0) * 1e3
        segs.update(wu)

        stage = "steady"
        trials = _trials4(wp3, H, call, heavy)
        out.update({
            "contract": contract,
            "meta": meta,
            "segments": segs,
            "trials": trials,
            "rel_err": err,
            "correct": bool(err <= ERR_MAX),
            "protocol_variant": "heavy" if heavy else "full",
            "fail_closed": False,
        })
    except Exception as e:  # noqa: BLE001 — recorded as fail-closed
        out.update({"fail_closed": True, "error": f"{stage}: {e}"[:300]})
    print(MARK + json.dumps(out), flush=True)


# ------------------------------------------------------------- spawn

def _trials4(wp3, H, call, heavy: bool) -> list:
    """timed trials: full holdout protocol (5x32 events, sync per
    trial) for light cells; heavy cells keep the sync-per-trial shape
    but cut to HEAVY_ROUNDS x wp3.TRIALS trials."""
    import torch
    if not heavy:
        return H._trials(wp3, call)
    torch.cuda.synchronize()
    for _ in range(wp3.WARMUP):
        call()
    torch.cuda.synchronize()
    ts = []
    for _ in range(HEAVY_ROUNDS):
        for _ in range(wp3.TRIALS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            call()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
    return ts


def _spawn(session: int, n: int, b: int, path: str) -> dict | None:
    cache = tempfile.mkdtemp(prefix=f"wp4_s{session}_{path}_")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "2"
    env["TRITON_CACHE_DIR"] = cache
    env["PYTHONUNBUFFERED"] = "1"
    env["ISSUE13_WP4_LEGACY_TABLE"] = str(LEGACY_TABLE)
    try:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--worker",
             "--session", str(session), "--n", str(n), "--b", str(b),
             "--path", path],
            capture_output=True, text=True, timeout=1800, env=env,
            cwd=str(ROOT))
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    for line in (proc.stdout or "").splitlines():
        if line.startswith(MARK):
            try:
                return json.loads(line[len(MARK):])
            except json.JSONDecodeError:
                return None
    print(f"  worker fail n={n} b={b} s={session} path={path} "
          f"rc={proc.returncode} stderr={(proc.stderr or '')[-300:]}",
          flush=True)
    return None


# -------------------------------------------------------- aggregation

def _verdict(points: list, cis: list, selector_ok: bool) -> tuple:
    """(verdict, pooled) for one cell: selector-failed has priority,
    then the WP3 win/confirmed-regression rule mapped to loss."""
    if not selector_ok:
        return "selector-failed", None
    if not points:
        return "vendor-unsupported", None
    pooled = (sorted(points)[len(points) // 2]
              if len(points) % 2 else
              0.5 * (sorted(points)[len(points) // 2 - 1]
                     + sorted(points)[len(points) // 2]))
    v = classify(points, cis)
    if v == "confirmed-regression":
        return "loss", pooled
    return v, pooled


def classify(points: list, cis: list) -> str:
    """same rule as WP3 classify_cell, vendor as denominator."""
    n_reg = sum(1 for sp in cis
                if sp and float(sp["high"]) < LOSS_POINT)
    if points and (pooled_of(points) < LOSS_POINT or n_reg >= 2):
        return "confirmed-regression"
    n_win = sum(1 for p in points if p > WIN_POINT)
    if points and pooled_of(points) > WIN_POINT and n_win >= 2:
        return "win"
    return "tie"


def pooled_of(xs: list) -> float:
    ys = sorted(xs)
    return (0.5 * (ys[len(ys) // 2 - 1] + ys[len(ys) // 2])
            if len(ys) % 2 == 0 else ys[len(ys) // 2])


def _geomean(xs) -> float | None:
    vals = [x for x in xs if x and x > 0]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--session", type=int, default=1)
    ap.add_argument("--n", type=int)
    ap.add_argument("--b", type=int)
    ap.add_argument("--path", type=str, default="")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.worker:
        return worker(args)

    from src.xpu.selector import (assert_current_environment,
                                  load_table)
    w3 = _load_wp3()
    table = load_table(str(SELECTOR_TABLE))
    live_env = assert_current_environment(table)
    profile_hash = table.get("profile_hash")
    print(f"env ok: profile={profile_hash} env={live_env}", flush=True)
    w3._write_legacy_table()   # shared WP3 snapshot (same bytes)
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
        order = [(n, b, p) for n, b in CELLS for p in PATHS]
        rng.shuffle(order)
        ts = time.time()
        ok = 0
        for n, b, p in order:
            r = _spawn(s, n, b, p)
            if r is not None:
                raw.append(r)
                if not r.get("fail_closed"):
                    ok += 1
        sessions_meta.append({
            "session": s, "role": "protocol",
            "jobs_ok": ok, "jobs_total": len(order),
            "elapsed_s": round(time.time() - ts, 1),
            "spawn_seed": SPAWN_SEED,
            "isolation": ("one subprocess per (point, path) with its "
                          "own fresh TRITON_CACHE_DIR; spawn order "
                          "shuffled per session"),
        })
        CHECKPOINT.write_text(json.dumps({"raw": raw,
                                          "sessions": sessions_meta}))
        print(f"session {s}: {ok}/{len(order)} jobs "
              f"({time.time() - ts:.0f}s)", flush=True)

    # index rows: (n,b,session) -> {path: row}
    idx: dict = {}
    for r in raw:
        key = (r.get("n"), r.get("batch"), r.get("session"))
        idx.setdefault(key, {})[r.get("path")] = r

    import numpy as np

    rng2 = np.random.default_rng(SEED)
    c2c_out, long_out, csv_rows = [], [], []
    loss_list, vendor_unsupported, selector_failed = [], [], []
    fallback_cells = []

    for n, b in C2C_CELLS:
        label = f"N{n}B{b}:c2c_forward"
        pts_v, cis_v, pts_i = [], [], []
        selector_all_ok, sessions_used = True, []
        entry = {"label": label, "n": n, "batch": b,
                 "direction": "c2c_forward", "sessions": {},
                 "gate_flags_by_session": []}
        for s in SESSIONS:
            rows = idx.get((n, b, s), {})
            present = [p for p in PATHS if p in rows]
            fc = [p for p in present if rows[p].get("fail_closed")]
            flags = {"all_paths_present": len(present) == len(PATHS),
                     "no_fail_closed": not fc,
                     "fail_closed_paths": fc}
            entry["gate_flags_by_session"].append(flags)
            per = {"flags": flags, "paths": {}}
            for pth in PATHS:
                pr = rows.get(pth)
                if pr is None:
                    per["paths"][pth] = {"missing": True}
                    continue
                meta = pr.get("meta") or {}
                # the worker inlines the plan binding into meta (WP3
                # contract: raw fields recorded, never re-keyed)
                binding = ({k: meta.get(k) for k in
                            ("executed", "selection_source", "fallback",
                             "flow_id", "composition", "variant",
                             "plan_reason", "profile_hash")}
                           if meta else None)
                if pth == "selector_new" and not pr.get("fail_closed"):
                    sel_valid = bool(
                        pr.get("correct")
                        and pr.get("rel_err", 1.0) <= ERR_MAX)
                    selector_all_ok = selector_all_ok and sel_valid
                    if binding is None:
                        selector_all_ok = False
                    if binding and (binding.get("fallback")
                                    or binding.get("selection_source")
                                    in ("m0", "abstain")):
                        fallback_cells.append(label)
                pv = {
                    "warm_p50_ms": (pooled_of(pr["trials"])
                                    if not pr.get("fail_closed")
                                    and pr.get("trials") else None),
                    "rel_err": pr.get("rel_err"),
                    "correct": pr.get("correct"),
                    "fail_closed": pr.get("fail_closed"),
                    "error": (pr.get("error")
                              if pr.get("fail_closed") else None),
                    "binding": binding,
                }
                per["paths"][pth] = pv
                csv_rows.append({
                    "label": label, "kind": "c2c", "session": s,
                    "path": pth, "warm_p50_ms": pv["warm_p50_ms"],
                    "rel_err": pv["rel_err"], "correct": pv["correct"],
                    "fail_closed": pv["fail_closed"],
                    "executed": (binding or {}).get("executed"),
                    "selection_source": (binding or {})
                    .get("selection_source"),
                    "composition": (binding or {}).get("composition"),
                    "fallback": (binding or {}).get("fallback"),
                    "error": (pv["error"] or "")[:160],
                })
            vend, news, inc = (rows.get("vendor"),
                               rows.get("selector_new"),
                               rows.get("incumbent_old"))

            def ok(r):
                return (r is not None and not r.get("fail_closed")
                        and r.get("correct"))

            # vendor/selector and vendor/incumbent comparisons are
            # independent — a failing incumbent plate must not erase
            # the vendor/selector statistics of the cell
            if ok(vend) and ok(news):
                spv = w3.H_boot(rng2, vend["trials"], news["trials"])
                if spv:
                    spv["definition"] = (
                        "median(vendor)/median(selector_new) "
                        "independent bootstrap (>1 = selector faster)")
                    pts_v.append(spv["point"])
                    cis_v.append(spv)
                    sessions_used.append(s)
            if ok(vend) and ok(inc):
                spi = w3.H_boot(rng2, vend["trials"], inc["trials"])
                if spi:
                    spi["definition"] = (
                        "median(vendor)/median(incumbent_old) "
                        "independent bootstrap (>1 = incumbent OK)")
                    pts_i.append(spi["point"])
            entry["sessions"][str(s)] = per
        entry["comparison"] = {
            "pooled_point": None, "per_session_points": pts_v,
            "cis": cis_v, "sessions": sessions_used,
            "incumbent_pooled_point": None,
            "definition": ("win = pooled vendor/selector_new point "
                           "> 1.02 AND point > 1.02 in >= 2 sessions; "
                           "loss = WP3 confirmed-regression criterion "
                           "(vendor > 2% faster in pooled point or "
                           ">= 2 session bootstrap CI-highs < 0.98)"),
        }
        verdict, pooled = _verdict(pts_v, cis_v, selector_all_ok)
        if verdict == "loss":
            loss_list.append({"label": label, "pooled_point": pooled})
        if verdict == "selector-failed":
            reasons = []
            for s in SESSIONS:
                pr = idx.get((n, b, s), {}).get("selector_new") or {}
                if pr.get("fail_closed") or pr.get("error"):
                    reasons.append(pr.get("error"))
                    break
            if not reasons:
                reasons = ["selector row missing/incorrect"]
            selector_failed.append({"label": label,
                                    "reason": reasons[0] or "incorrect"})
        if verdict == "vendor-unsupported":
            reasons = []
            for s in SESSIONS:
                vr = idx.get((n, b, s), {}).get("vendor") or {}
                if vr.get("fail_closed"):
                    reasons.append(f"s{s}: {vr.get('error')}")
            vendor_unsupported.append({"label": label,
                                       "reason": (reasons or
                                                  "vendor missing")
                                       if reasons else
                                       "vendor row missing/incorrect"})
        if verdict not in ("selector-failed",):
            entry["comparison"]["pooled_point"] = pooled
            entry["comparison"]["incumbent_pooled_point"] = (
                pooled_of(pts_i) if pts_i else None)
            entry["comparison"]["verdict"] = verdict
        else:
            entry["comparison"]["verdict"] = verdict
        c2c_out.append(entry)

    for n, b in LONG_CELLS:
        label = f"N{n}B{b}:c2c_forward"
        entry = {"label": label, "n": n, "batch": b,
                 "direction": "c2c_forward", "sessions": {}}
        for s in SESSIONS:
            rows = idx.get((n, b, s), {})
            per = {"paths": {}}
            for pth in PATHS:
                pr = rows.get(pth)
                if pr is None:
                    per["paths"][pth] = {"missing": True}
                    continue
                segs = pr.get("segments") or {}
                meta = pr.get("meta") or {}
                binding = ({k: meta.get(k) for k in
                            ("executed", "selection_source", "fallback",
                             "flow_id", "composition", "variant",
                             "plan_reason")}
                           if meta else None)
                rec = {
                    "warm_p50_ms": (pooled_of(pr["trials"])
                                    if not pr.get("fail_closed")
                                    and pr.get("trials") else None),
                    "warm_mad_ms": (w3.H_mad(pr["trials"])
                                    if not pr.get("fail_closed")
                                    and pr.get("trials") else None),
                    "one_shot_ms": segs.get("one_shot_ms"),
                    "peak_mem_mb": segs.get("peak_mem_mb"),
                    "rel_err": pr.get("rel_err"),
                    "correct": pr.get("correct"),
                    "fail_closed": pr.get("fail_closed"),
                    "executed": (binding or {}).get("executed"),
                    "selection_source": (binding or {})
                    .get("selection_source"),
                    "fallback": (binding or {}).get("fallback"),
                    "plan_reason": (binding or {}).get("plan_reason"),
                    "error": pr.get("error"),
                }
                per["paths"][pth] = rec
                csv_rows.append({
                    "label": label, "kind": "long", "session": s,
                    "path": pth, "warm_p50_ms": rec["warm_p50_ms"],
                    "rel_err": rec["rel_err"], "correct": rec["correct"],
                    "fail_closed": rec["fail_closed"],
                    "executed": rec["executed"],
                    "selection_source": rec["selection_source"],
                    "composition": (binding or {}).get("composition"),
                    "fallback": rec["fallback"],
                    "error": (rec["error"] or "")[:160],
                })
            entry["sessions"][str(s)] = per
        long_out.append(entry)

    supported = [e["label"] for e in c2c_out
                 if e["comparison"]["verdict"] not in
                 ("selector-failed", "vendor-unsupported")]
    verdict_counts = {"win": 0, "tie": 0, "loss": 0}
    for e in c2c_out:
        v = e["comparison"]["verdict"]
        if v in verdict_counts:
            verdict_counts[v] += 1
    geo_new = _geomean([e["comparison"]["pooled_point"]
                        for e in c2c_out
                        if e["comparison"]["pooled_point"] is not None])
    geo_inc = _geomean([e["comparison"]["incumbent_pooled_point"]
                        for e in c2c_out
                        if e["comparison"]["incumbent_pooled_point"]
                        is not None])
    fallback_ratio = len(set(fallback_cells)) / len(C2C_CELLS)
    coverage = {
        "c2c_cells_total": len(C2C_CELLS),
        "verdicts": verdict_counts,
        "vendor_supported_cells": supported,
        "vendor_unsupported": vendor_unsupported,
        "selector_failed": selector_failed,
        "fallback_ratio": round(fallback_ratio, 4),
        "fallback_cells": sorted(set(fallback_cells)),
        "losses": loss_list,
        "geomean_vendor_over_selector": geo_new,
        "geomean_vendor_over_incumbent": geo_inc,
        "definition": ("losses and vendor-unsupported points are "
                       "retained explicitly (never removed from the "
                       "denominator); geomeans cover the "
                       "vendor-supported subset with computed verdicts"),
    }
    claims = {
        "broad_coverage_geomean": geo_new,
        "broad_coverage_claim_supported": bool(
            geo_new is not None and geo_new > 1.0),
        "all_negative_points_listed": True,
        "note": ("per issue #13: broad-coverage acceleration claim is "
                 "only usable with geomean > 1 on the vendor-supported "
                 "subset and every negative point published — these "
                 "artifacts carry both"),
    }

    doc = {
        "issue": 13, "workpackage": "WP4",
        "schema": "issue13-wp4-coverage/1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                     time.gmtime()),
        "environment": {
            "profile_hash": profile_hash,
            "env_hash_live": live_env,
            "selector_table_sha256": _sha256(SELECTOR_TABLE),
            "composition_table_sha256": _sha256(COMPOSITION_TABLE),
            "legacy_table_sha256": _sha256(LEGACY_TABLE),
            "producer_sha256": {
                "scripts/issue13_wp4_coverage.py": _sha256(SCRIPT),
                "scripts/issue13_wp3_release.py": _sha256(
                    ROOT / "scripts/issue13_wp3_release.py"),
                "src/xpu/composition.py": _sha256(COMP_SRC),
                "src/xpu/entry.py": _sha256(ENTRY_SRC),
                "src/xpu/hybrid4.py": _sha256(H4_SRC),
                "src/xpu/selector.py": _sha256(SEL_SRC),
            },
        },
        "protocol": {
            "cells_c2c": [{"N": n, "batch": b} for n, b in C2C_CELLS],
            "cells_long": [{"N": n, "batch": b} for n, b in LONG_CELLS],
            "paths": list(PATHS),
            "sessions": list(SESSIONS),
            "seed": SEED, "spawn_seed": SPAWN_SEED,
            "steady_warmup": {
                "batch_calls": w3.STEADY_BATCH,
                "min_calls": w3.STEADY_MIN_CALLS,
                "max_calls": w3.STEADY_MAX_CALLS,
                "tol": w3.STEADY_TOL,
                "consecutive_batches": w3.STEADY_NEED,
                "note": "shared WP3 rule (imports and runs the exact "
                        "issue13_wp3_release implementation)",
            },
            "input_semantics": ("complex64 non-zero imag, contiguous, "
                                "axis=-1, forward unscaled, seeded per "
                                "(session, n, b); rank-1 (N,) for b==1, "
                                "rank-2 (B,N) otherwise — the "
                                "production entry contract"),
            "gates": {
                "win_point": WIN_POINT, "loss_point": LOSS_POINT,
                "err_max": ERR_MAX,
                "note": ("WP4 is coverage reporting, not a promotion "
                         "gate; loss keeps the WP3 confirmed-"
                         "regression criterion vs vendor"),
            },
            "heavy_cell_protocol": {
                "threshold_ms_per_call": HEAVY_CALL_MS,
                "steady_warmup": {"min_calls": HEAVY_STEADY_MIN_CALLS,
                                  "max_calls": HEAVY_STEADY_MAX_CALLS,
                                  "consecutive_batches":
                                      HEAVY_STEADY_NEED},
                "trials": f"{HEAVY_ROUNDS} rounds x wp3.TRIALS "
                          "(sync per trial)",
                "reason": ("smoke: >=30ms/call cells (M0-fallback "
                           "long/4M-elem rows) would spend ~500s in "
                           "steady warmup and ~260s in trials at the "
                           "full protocol; at that scale the "
                           "launch/clock ramp can no longer "
                           "contaminate the median"),
                "recorded_per_row": "segments.protocol_variant",
            },
        },
        "sessions": sessions_meta,
        "c2c_grid": c2c_out,
        "long_slice": long_out,
        "coverage": coverage,
        "claims": claims,
    }
    OUT_DIR.joinpath("wp4_coverage.json").write_text(
        json.dumps(doc, indent=2, sort_keys=True) + "\n")
    with open(OUT_DIR / "wp4_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "label", "kind", "session", "path", "warm_p50_ms",
            "rel_err", "correct", "fail_closed", "executed",
            "selection_source", "composition", "fallback", "error"])
        w.writeheader()
        w.writerows(csv_rows)
    print(json.dumps({
        "c2c_cells": len(c2c_out),
        "verdicts": {e["label"]: e["comparison"]["verdict"]
                     for e in c2c_out},
        "coverage": coverage,
        "claims": claims,
    }, indent=1), flush=True)
    print(f"-> {OUT_DIR / 'wp4_coverage.json'}", flush=True)


if __name__ == "__main__":
    main()
