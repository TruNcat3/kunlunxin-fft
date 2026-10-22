"""issue #10 review item 5: performance holdout (>=2 independent
sessions) before any selector promotion.

Cells (holdout set fixed by the review):
  N in {1024, 2048} x B in {1, 16, 64, 256} x {c2c_forward, c2c_inverse}
  = 16 cells, paths = vendor + hybrid_vendor_p + hybrid_fused_p
  (primary promotion candidate = hybrid_fused_p, ablation winner).

Per cell x session (independent subprocess, fresh process => true
one-shot cold timing; fresh TRITON_CACHE_DIR per session):
  cold  : first call in a fresh process (plan/compile included),
  warm  : raw trial array after warmup (plan-reuse steady state).

Statistics:
  - p50/MAD per cell/path/session from raw warm trials,
  - paired bootstrap 95% CI (B=2000) of median speedup
    vendor/hybrid within each session,
  - cross-session spread of the speedup,
  - same-semantic comparison vs issue9 v61 incumbent
    (results/p800/issue9_v61/reports/e2e_report.csv, incumbent:e2e and
    incumbent:one_shot; protocol differs -> documented caveat, used
    only for same-shape/axis/dtype magnitude comparison).

Promotion gate (review wording): a cell enters the selector only if
the improvement over BOTH vendor and incumbent is stably >2% in every
session (bootstrap CI lower bound > 1.02).

Output: results/p800/issue10_capability/wp3_holdout.json
"""
import argparse
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT = Path(__file__).resolve()
OUT = ROOT / "results/p800/issue10_capability/wp3_holdout.json"
INCUMBENT_CSV = (ROOT / "results/p800/issue9_v61/reports/"
                 "e2e_report.csv")
MARK = "@@HOLDOUT@@"
NS = (1024, 2048)
BS = (1, 16, 64, 256)
DIRS = (("c2c_forward", False), ("c2c_inverse", True))
SESSIONS = (1, 2)
PATHS = ("vendor", "hybrid_vendor_p", "hybrid_fused_p")
# promotion candidates (primary = ablation winner, baseline for contrast)
CANDIDATES = ("hybrid_fused_p", "hybrid_vendor_p")
BOOTSTRAP = 2000
SEED = 10_2026
GATE = 1.02


def _wp3():
    spec = importlib.util.spec_from_file_location(
        "issue10_wp3_perf", ROOT / "scripts/issue10_wp3_perf.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _trials(wp3, call):
    import torch
    torch.cuda.synchronize()
    for _ in range(wp3.WARMUP):
        call()
    torch.cuda.synchronize()
    ts = []
    for _ in range(wp3.ROUNDS):
        for _ in range(wp3.TRIALS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            call()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
    return ts


def worker(a) -> int:
    import torch
    wp3 = _wp3()
    n, b, inverse = a.n, a.b, a.inverse
    x = wp3._data(b, n)
    out = {"session": a.session, "n": n, "batch": b,
           "direction": "c2c_inverse" if inverse else "c2c_forward",
           "paths": {}}
    try:
        wp3._vendor_canary()
        cf = wp3._contract_fields(x, n, b, inverse)
    except Exception as e:  # noqa: BLE001
        out["fail_closed"] = f"{type(e).__name__}: {str(e)[:200]}"
        print(MARK + json.dumps(out))
        return 0
    out["contract"] = cf
    calls = {}
    for p in PATHS:
        call, meta = wp3.PATHS[p](x, inverse)
        calls[p] = (call, meta)
    # true one-shot cold: first call in this fresh process
    cold = {}
    for p in PATHS:
        call = calls[p][0]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        call()
        torch.cuda.synchronize()
        cold[p] = (time.perf_counter() - t0) * 1e3
    for p in PATHS:
        call, meta = calls[p]
        y_re, y_im = call()
        err = wp3._check_batched(y_re, y_im, x, inverse)
        out["paths"][p] = {
            "rel_err": float(err),
            "correct": bool(err <= wp3.TOL_REL),
            "cold_ms": cold[p],
            "trials": _trials(wp3, call),
            **meta,
        }
    print(MARK + json.dumps(out))
    return 0


def _spawn(session, n, b, inverse, cache) -> dict | None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "2"
    env["TRITON_CACHE_DIR"] = cache
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--worker",
         "--session", str(session), "--n", str(n), "--b", str(b),
         "--inverse", str(int(inverse))],
        capture_output=True, text=True, timeout=900, env=env,
        cwd=str(ROOT))
    for line in (proc.stdout or "").splitlines():
        if line.startswith(MARK):
            try:
                return json.loads(line[len(MARK):])
            except json.JSONDecodeError:
                return None
    print(f"  worker fail n={n} b={b} inv={int(inverse)} s={session} "
          f"rc={proc.returncode} stderr={(proc.stderr or '')[-300:]}",
          flush=True)
    return None


def _p50(xs):
    ys = sorted(xs)
    return ys[len(ys) // 2] if ys else None


def _mad(xs):
    m = _p50(xs)
    if m is None:
        return None
    return _p50([abs(v - m) for v in xs])


def _boot_ratio(rng, num, den, n_boot=BOOTSTRAP):
    """CI of median(num)/median(den) by resampling both arrays."""
    import numpy as np
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    if not len(num) or not len(den):
        return None
    ratio = np.empty(n_boot)
    for i in range(n_boot):
        a = num[rng.integers(0, len(num), len(num))]
        b = den[rng.integers(0, len(den), len(den))]
        ratio[i] = np.median(a) / max(np.median(b), 1e-30)
    lo, hi = np.percentile(ratio, [2.5, 97.5])
    return {"point": float(np.median(num) / max(np.median(den), 1e-30)),
            "ci95": [float(lo), float(hi)], "n_boot": n_boot}


def _incumbent() -> dict:
    """label -> {e2e: incumbent:e2e ms, one_shot: incumbent:one_shot}"""
    out = {}
    if not INCUMBENT_CSV.exists():
        return out
    with open(INCUMBENT_CSV) as f:
        for row in csv.DictReader(f):
            try:
                e2e = json.loads(row["e2e"])
                one = json.loads(row["one_shot"])
            except (KeyError, json.JSONDecodeError):
                continue
            out[row["label"]] = {
                "incumbent_e2e_ms": e2e.get("incumbent:e2e"),
                "incumbent_one_shot_ms": one.get("incumbent:one_shot"),
                "issue9_selector_e2e_ms": e2e.get("selector:e2e"),
            }
    return out


def _promotion_block(ready, spreads, cells_total) -> dict:
    per_cand = {}
    for ch in CANDIDATES:
        vs = ready[ch]["vendor"]
        iss = ready[ch]["incumbent"]
        per_cand[ch] = {
            "ready_vs_vendor": vs,
            "ready_vs_incumbent": iss,
            "ready_vs_vendor_and_incumbent": [
                l for l in vs if l in set(iss)],
            "max_cross_session_spread_rel": (
                max(spreads[ch]) if spreads[ch] else None),
        }
    primary = per_cand[CANDIDATES[0]]
    inter = primary["ready_vs_vendor_and_incumbent"]
    return {
        "candidate": CANDIDATES[0],
        "ready_vs_vendor": primary["ready_vs_vendor"],
        "ready_vs_incumbent": primary["ready_vs_incumbent"],
        "ready_vs_vendor_and_incumbent": inter,
        "cells_total": cells_total,
        "max_cross_session_spread_rel": primary[
            "max_cross_session_spread_rel"],
        "selector_entry_threshold": GATE,
        "release_quality_status": (
            "BLOCKED" if not inter else
            "PARTIAL: validated cells listed, rest of matrix still "
            "exploratory"),
        "candidates": per_cand,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--session", type=int, default=1)
    ap.add_argument("--n", type=int, required=False)
    ap.add_argument("--b", type=int, required=False)
    ap.add_argument("--inverse", type=int, default=0)
    a = ap.parse_args()
    if a.worker:
        return worker(a)

    import numpy as np
    t_start = time.time()
    cells = [(n, b, inv) for n in NS for b in BS
             for _, inv in DIRS]
    raw, session_meta = [], []
    for s in SESSIONS:
        cache = tempfile.mkdtemp(prefix=f"holdout_triton_cache_s{s}_")
        t0 = time.time()
        ok = 0
        for n, b, inv in cells:
            r = _spawn(s, n, b, inv, cache)
            if r is not None:
                raw.append(r)
                ok += 1
        session_meta.append({
            "session": s, "cells_ok": ok, "cells_total": len(cells),
            "elapsed_s": round(time.time() - t0, 1),
            "fresh_triton_cache": cache,
        })
        shutil.rmtree(cache, ignore_errors=True)
        print(f"session {s}: {ok}/{len(cells)} cells "
              f"({time.time() - t0:.0f}s)", flush=True)

    # index raw by (n, b, dir)
    idx = {}
    for r in raw:
        idx[(r["n"], r["batch"], r["direction"], r["session"])] = r

    inc = _incumbent()
    rng = np.random.default_rng(SEED)
    cells_out, fail_closed = [], []
    ready = {ch: {"vendor": [], "incumbent": []} for ch in CANDIDATES}
    spreads = {ch: [] for ch in CANDIDATES}
    for n, b, inv in cells:
        d = "c2c_inverse" if inv else "c2c_forward"
        label = f"N{n}B{b}:{d}"
        # issue9 csv uses hyphenated direction in labels
        csv_label = f"N{n}B{b}:c2c-{'inverse' if inv else 'forward'}"
        c = {"label": label, "n": n, "batch": b, "direction": d,
             "sessions": {}}
        cand_sessions = {ch: {"v": [], "i": []} for ch in CANDIDATES}
        cok = True
        for s in SESSIONS:
            r = idx.get((n, b, d, s))
            if r is None:
                cok = False
                continue
            if r.get("fail_closed"):
                fail_closed.append(f"{label}/s{s}")
                cok = False
                continue
            entry = {"contract": r.get("contract"), "paths": {}}
            for p in PATHS:
                pr = r["paths"].get(p)
                if pr is None:
                    cok = False
                    continue
                entry["paths"][p] = {
                    "cold_ms": pr["cold_ms"],
                    "warm_p50_ms": _p50(pr["trials"]),
                    "warm_mad_ms": _mad(pr["trials"]),
                    "n_trials": len(pr["trials"]),
                    "rel_err": pr["rel_err"],
                    "correct": pr["correct"],
                    "warm_trials_ms": pr["trials"],
                }
                if not pr["correct"]:
                    cok = False
            hv = entry["paths"].get("vendor")
            for ch in CANDIDATES:
                hh = entry["paths"].get(ch)
                if not (hv and hh):
                    continue
                rv = r["paths"]["vendor"]["trials"]
                rh = r["paths"][ch]["trials"]
                sp = _boot_ratio(rng, rv, rh)
                sp["definition"] = (
                    f"median(vendor)/median({ch}) "
                    "(>1 = hybrid faster)")
                entry[f"speedup_vendor_boot:{ch}"] = sp
                if sp:
                    cand_sessions[ch]["v"].append(sp)
                    entry[f"cold_speedup_vendor:{ch}"] = (
                        hv["cold_ms"] / hh["cold_ms"])
                ci = inc.get(csv_label, inc.get(label, {}))
                inc_e2e = ci.get("incumbent_e2e_ms")
                if inc_e2e and hh["warm_p50_ms"]:
                    # incumbent fixed -> resample hybrid trials only
                    den = np.asarray(rh, dtype=np.float64)
                    ratio = np.empty(BOOTSTRAP)
                    for i in range(BOOTSTRAP):
                        b_ = den[rng.integers(0, len(den), len(den))]
                        ratio[i] = inc_e2e / max(np.median(b_), 1e-30)
                    lo, hi = np.percentile(ratio, [2.5, 97.5])
                    e = {"point": float(inc_e2e / hh["warm_p50_ms"]),
                         "ci95": [float(lo), float(hi)],
                         "incumbent_e2e_ms": inc_e2e,
                         "incumbent_one_shot_ms":
                             ci.get("incumbent_one_shot_ms"),
                         "definition": f"incumbent_e2e/median({ch}) "
                                       "(>1 = hybrid faster)"}
                    entry[f"speedup_incumbent_boot:{ch}"] = e
                    cand_sessions[ch]["i"].append(e)
            c["sessions"][str(s)] = entry
        cross = {}
        for ch in CANDIDATES:
            vs, iss = cand_sessions[ch]["v"], cand_sessions[ch]["i"]
            if len(vs) == len(SESSIONS):
                pts = [sp["point"] for sp in vs]
                spr = abs(pts[0] - pts[1]) / max(
                    sum(pts) / len(pts), 1e-30)
                spreads[ch].append(spr)
                okv = all(sp["ci95"][0] > GATE for sp in vs)
                cross.setdefault(ch, {})["vendor"] = {
                    "points": pts, "spread_rel": spr,
                    "all_ci_low_gt_1.02": okv}
                if okv:
                    ready[ch]["vendor"].append(label)
            if len(iss) == len(SESSIONS):
                oki = all(e["ci95"][0] > GATE for e in iss)
                cross.setdefault(ch, {})["incumbent"] = {
                    "points": [e["point"] for e in iss],
                    "all_ci_low_gt_1.02": oki}
                if oki:
                    ready[ch]["incumbent"].append(label)
        c["cross_session_by_candidate"] = cross
        c["correct_all_sessions"] = cok
        cells_out.append(c)

    # protocol caveat for incumbent join
    caveat = (
        "issue9 v61 e2e protocol differs from this holdout "
        "(different warmup/rounds machinery, its own plan-reuse "
        "definition); join is same-shape/axis/dtype (fp32, last-axis "
        "c2c) magnitude comparison only, not a like-for-like timing "
        "protocol. hybrid cold 口径 compared against "
        "incumbent:one_shot where present.")
    doc = {
        "schema": "issue10-wp3-holdout-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": str(SCRIPT.relative_to(ROOT)),
        "protocol": {
            "cells": [f"N{n}B{b}:{'c2c_inverse' if inv else 'c2c_forward'}"
                      for n, b, inv in cells],
            "sessions": len(SESSIONS),
            "session_isolation": "independent subprocess per cell; "
                                 "fresh TRITON_CACHE_DIR per session",
            "cold": "first call in a fresh process (one-shot: plan + "
                    "compile included)",
            "warm": f"raw trials after warmup={_wp3().WARMUP} rounds="
                    f"{_wp3().ROUNDS} trials={_wp3().TRIALS} "
                    "(plan-reuse steady state)",
            "bootstrap": f"B={BOOTSTRAP} paired resample of warm trial "
                         f"arrays, seed={SEED}, percentile 95% CI",
            "promotion_gate": "cell enters selector only if bootstrap "
                              "CI lower bound of speedup > 1.02 in "
                              "EVERY session, vs vendor AND incumbent",
            "correctness_tol_rel": 1e-4,
            "incumbent_join_caveat": caveat,
            "candidates": list(CANDIDATES),
            "primary_candidate": CANDIDATES[0],
        },
        "sessions": session_meta,
        "cells": cells_out,
        "fail_closed_cells": fail_closed,
        "promotion": _promotion_block(ready, spreads, len(cells)),
        "elapsed_s": round(time.time() - t_start, 1),
    }
    OUT.write_text(json.dumps(doc, indent=2))
    print(f"written {OUT}")
    print(json.dumps({
        "sessions": session_meta,
        "promotion": doc["promotion"],
        "fail_closed": fail_closed,
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
