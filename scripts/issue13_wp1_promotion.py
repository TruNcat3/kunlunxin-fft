"""issue #13 WP1: stability gate for the frozen promotion candidates.

Scope (issue #13, no search expansion): the frozen positive keys from
the issue's table — 5 ``hybrid_fused_p`` cells
(N1024B1:inverse, N2048B1:both, N2048B16:both) and 3
``hybrid_vendor_p`` cells (N2048B1:both, N2048B16:inverse). Every cell
measures ALL five issue-10 holdout paths so the vendor / profile-bound
incumbent / M0 comparisons come from the same rows as PR #12.

Protocol: identical to the PR #12 holdout (issue10_wp3_holdout worker
reused verbatim — one subprocess per (cell, path, session), fresh
TRITON_CACHE_DIR, itemized segments, b==1 rank-1 entry contract on
all five paths, independent bootstrap), plus:

* 4 sessions total: sessions 1-3 are protocol sessions, session 4 is
  an independent confirmation session (role recorded); spawn order is
  shuffled per session (seed recorded) = random interleaving;
* environment/profile fail-closed BEFORE any spawn
  (assert_current_environment on the production table).

Promotion rules (issue #13 WP1, pure function
:func:`promotion_verdict`):

1. correctness: every path, every session, rel_err <= 1e-4, no
   fail_closed, all paths present, all-path contract (WP0), uniform
   entry shape;
2. every session: independent-bootstrap CI lower bound of speedup
   > 1.02 (GATE) vs BOTH vendor and the profile-bound incumbent;
   M0 is attribution only, never a gate;
3. cross-session spread: each session's speedup point vs the
   cross-session median deviates <= 10% for both baselines — exceed
   => not promoted (reason recorded; the issue allows promotion only
   with added sessions + explanation, which we do not take silently);
4. if fused and vendor-P both pass the same exact key, deploy the
   stable-measured faster one (pooled cross-session bootstrap of the
   two candidates' own warm trials — measured, never by name or
   static score).

Output: results/p800/issue13_release/wp1_promotion.json
        results/p800/issue13_release/wp1_raw.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT = Path(__file__).resolve()
OUT_DIR = ROOT / "results/p800/issue13_release"
OUT = OUT_DIR / "wp1_promotion.json"
RAW_CSV = OUT_DIR / "wp1_raw.csv"
CHECKPOINT = OUT_DIR / "wp1_sessions_raw.json"
SELECTOR_TABLE = ROOT / "config/selector_table.json"
COMPOSITION_TABLE = ROOT / "config/composition_table.json"
SEL_SRC = ROOT / "src/xpu/selector.py"
COMP_SRC = ROOT / "src/xpu/composition.py"
ENTRY_SRC = ROOT / "src/xpu/entry.py"

# frozen keys (issue #13): (n, batch, inverse, frozen candidates)
FROZEN_CELLS = (
    (1024, 1, 1, ("hybrid_fused_p",)),
    (2048, 1, 0, ("hybrid_fused_p", "hybrid_vendor_p")),
    (2048, 1, 1, ("hybrid_fused_p", "hybrid_vendor_p")),
    (2048, 16, 0, ("hybrid_fused_p",)),
    (2048, 16, 1, ("hybrid_fused_p", "hybrid_vendor_p")),
)
# sessions 1-3 protocol, session 4 = independent confirmation
SESSIONS = (1, 2, 3, 4)
CONFIRM_SESSION = 4
SEED = 13_102026
SPAWN_SEED = SEED + 700
GATE = 1.02          # CI lower bound must EXCEED this vs both baselines
SPREAD_REL_MAX = 0.10  # per-session point vs cross-session median
TOL_REL = 1e-4

# reuse the PR #12 holdout worker + helpers (single source of protocol)
_spec = importlib.util.spec_from_file_location(
    "issue10_wp3_holdout", ROOT / "scripts" / "issue10_wp3_holdout.py")
H = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("issue10_wp3_holdout", H)
_spec.loader.exec_module(H)

PATHS = H.PATHS
INCUMBENT_LABEL = H.INCUMBENT_LABEL
M0_LABEL = H.M0_LABEL
BOOTSTRAP = H.BOOTSTRAP


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def promotion_verdict(flags_per_session, vendor_ci, incumbent_ci,
                      vendor_pts, incumbent_pts, cell_ok: bool,
                      gate: float = GATE,
                      spread_max: float = SPREAD_REL_MAX) -> dict:
    """Pure promotion decision (host-testable, issue #13 WP1).

    ``flags_per_session``: list of per-session gate flag dicts (all
    paths present / no fail_closed / correct / all-path contract);
    ``vendor_ci``/``incumbent_ci``: per-session CI dicts
    (``{"ci95": [lo, hi]}``) or None; ``vendor_pts``/
    ``incumbent_pts``: per-session speedup points. ``cell_ok`` folds
    the cell-level entry-shape uniformity flag.
    """
    reasons = []
    if not cell_ok:
        reasons.append("entry_shape_uniform failed (cell-level)")
    for i, fl in enumerate(flags_per_session, 1):
        bad = [k for k in ("all_paths_present", "no_fail_closed",
                           "correct_all_sessions",
                           "contract_all_sessions") if not fl.get(k)]
        if bad:
            reasons.append(f"session {i}: gate flags failed: {bad}")
    for name, cis, pts in (("vendor", vendor_ci, vendor_pts),
                           ("incumbent", incumbent_ci, incumbent_pts)):
        if len(cis) != len(SESSIONS) or any(c is None for c in cis):
            reasons.append(f"{name}: missing bootstrap CI in a session")
            continue
        lows = [c["ci95"][0] for c in cis]
        if any(lo <= gate for lo in lows):
            reasons.append(
                f"{name}: CI lower bound <= {gate} in session(s) "
                + ", ".join(str(i + 1) for i, lo in enumerate(lows)
                            if lo <= gate)
                + f" (lows={['%.4f' % lo for lo in lows]})")
        if len(pts) == len(SESSIONS) and all(p for p in pts):
            med = sorted(pts)[len(pts) // 2] if len(pts) % 2 else \
                0.5 * (sorted(pts)[len(pts) // 2 - 1]
                       + sorted(pts)[len(pts) // 2])
            devs = [abs(p - med) / med for p in pts] if med else \
                [float("inf")]
            mx = max(devs)
            if mx > spread_max:
                reasons.append(
                    f"{name}: cross-session spread {mx:.3f} > "
                    f"{spread_max} (points="
                    + ", ".join("%.3f" % p for p in pts) + ")")
        else:
            reasons.append(f"{name}: missing spread points")
    return {
        "pass": not reasons,
        "reasons": reasons,
        "gate_ci_low_gt": gate,
        "spread_rel_max": spread_max,
        "ci_low_by_session": {
            "vendor": [c["ci95"][0] if c else None for c in vendor_ci],
            "incumbent": [c["ci95"][0] if c else None
                          for c in incumbent_ci]},
        "points_by_session": {"vendor": vendor_pts,
                              "incumbent": incumbent_pts},
    }


def choose_faster(pooled: dict, rng) -> dict:
    """Both candidates passed a key -> deploy the measured faster one.

    ``pooled``: {candidate: concatenated warm trial arrays across
    sessions}. Returns pairwise bootstrap of the slower/faster ratio.
    """
    import numpy as np
    names = sorted(pooled)
    if len(names) != 2:
        return {"selected": names[0] if names else None,
                "pairwise": None}
    a, b = names
    ratio = H._boot_ratio(rng, pooled[a], pooled[b])
    if ratio is None:
        return {"selected": None, "pairwise": None,
                "reason": "empty trials"}
    # point = median(a)/median(b) in ms: <= 1.0 means a is faster
    selected = a if ratio["point"] <= 1.0 else b
    return {
        "selected": selected,
        "pairwise": {
            "definition": f"median({a})/median({b}) pooled across "
                          f"sessions (independent bootstrap)",
            **ratio},
        "policy": "stable measured faster candidate wins "
                  "(never name / static score)",
    }


def _session_flags(rows: dict) -> tuple[dict, dict]:
    """Per-session gate flags + raw entry shapes from path rows."""
    flags = {"all_paths_present": True, "no_fail_closed": True,
             "correct_all_sessions": True, "contract_all_sessions": True}
    if any(p not in rows for p in PATHS):
        flags["all_paths_present"] = False
        return flags, {}
    bad = {p: rows[p].get("fail_closed") for p in PATHS
           if rows[p].get("fail_closed")}
    if bad:
        flags["no_fail_closed"] = False
        return flags, {}
    ok, _ = H._session_contract_gate(rows)
    flags["contract_all_sessions"] = ok
    if any(not rows[p].get("correct") for p in PATHS):
        flags["correct_all_sessions"] = False
    shapes = {p: ((rows[p].get("meta") or {}).get("input_shape")
                  or ["__missing__"]) for p in PATHS}
    return flags, shapes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true",
                    help="continue from wp1_sessions_raw.json")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import numpy as np

    # WP0/WP1: environment/profile fail-closed BEFORE any spawn
    from src.xpu.selector import (assert_current_environment,
                                  load_table)
    t0 = time.time()
    table = load_table(str(SELECTOR_TABLE))
    live_env = assert_current_environment(table)

    cells = [(n, b, inv) for n, b, inv, _ in FROZEN_CELLS]
    frozen_by_cell = {(n, b, inv): cands
                      for n, b, inv, cands in FROZEN_CELLS}

    raw, session_meta = [], []
    if a.resume and CHECKPOINT.exists():
        ck = json.loads(CHECKPOINT.read_text())
        raw, session_meta = ck["raw"], ck["sessions"]
        print(f"resume: {len(raw)} rows, {len(session_meta)} sessions",
              flush=True)
    done_sessions = {m["session"] for m in session_meta}
    total = len(SESSIONS) * len(cells) * len(PATHS)
    for s in SESSIONS:
        if s in done_sessions:
            continue
        ts = time.time()
        order = [(n, b, inv, p) for n, b, inv in cells for p in PATHS]
        idxs = np.arange(len(order))
        np.random.default_rng(SPAWN_SEED + s).shuffle(idxs)
        order = [order[int(i)] for i in idxs]
        spawn_order = [
            f"N{n}B{b}:c2c-{'inverse' if inv else 'forward'}:{p}"
            for n, b, inv, p in order]
        ok = 0
        for n, b, inv, p in order:
            r = H._spawn(s, n, b, inv, p)
            if r is not None:
                r["role"] = ("confirmation" if s == CONFIRM_SESSION
                             else "protocol")
                raw.append(r)
                if not r.get("fail_closed"):
                    ok += 1
        session_meta.append({
            "session": s,
            "role": "confirmation" if s == CONFIRM_SESSION else "protocol",
            "jobs_ok": ok, "jobs_total": len(order),
            "elapsed_s": round(time.time() - ts, 1),
            "spawn_seed": SPAWN_SEED + s, "spawn_order": spawn_order,
            "isolation": H._spawn.__doc__ or (
                "one subprocess per (cell, path) with its own fresh "
                "TRITON_CACHE_DIR; spawn order shuffled per session"),
        })
        CHECKPOINT.write_text(json.dumps({"raw": raw,
                                          "sessions": session_meta}))
        print(f"session {s} ({'confirmation' if s == CONFIRM_SESSION else 'protocol'})"
              f": {ok}/{len(order)} jobs ({time.time() - ts:.0f}s)",
              flush=True)
        _ = total

    # index raw rows: (n, b, dir_label, session) -> {path: row}
    idx = {}
    for r in raw:
        key = (r.get("n"), r.get("batch"), r.get("direction"),
               r.get("session"))
        idx.setdefault(key, {})[r.get("path")] = r

    rng = np.random.default_rng(SEED)
    cells_out = []
    for n, b, inv in cells:
        d = "c2c_inverse" if inv else "c2c_forward"
        label = f"N{n}B{b}:{d}"
        c = {"label": label, "n": n, "batch": b, "direction": d,
             "frozen_candidates": list(frozen_by_cell[(n, b, inv)]),
             "sessions": {}, "verdicts": {}}
        flags_per_session, shapes_all = [], {}
        vendor_ci = {ch: [] for ch in H.CANDIDATES}
        inc_ci = {ch: [] for ch in H.CANDIDATES}
        vendor_pt = {ch: [] for ch in H.CANDIDATES}
        inc_pt = {ch: [] for ch in H.CANDIDATES}
        pooled = {ch: [] for ch in H.CANDIDATES}
        for s in SESSIONS:
            rows = idx.get((n, b, d, s), {})
            flags, shapes = _session_flags(rows)
            flags_per_session.append(flags)
            shapes_all[s] = shapes
            entry = {"flags": flags, "role": (
                "confirmation" if s == CONFIRM_SESSION else "protocol"),
                "paths": {}}
            if flags["all_paths_present"] and flags["no_fail_closed"]:
                for p in PATHS:
                    pr = rows[p]
                    meta = pr.get("meta") or {}
                    entry["paths"][p] = {
                        "warm_p50_ms": H._p50(pr["trials"]),
                        "warm_mad_ms": H._mad(pr["trials"]),
                        "rel_err": pr["rel_err"],
                        "correct": pr["correct"],
                        "one_shot_ms": (pr.get("segments") or {}).get(
                            "one_shot_ms"),
                        # what actually ran (plan/executed attribution;
                        # promote reads these for entry provenance)
                        "binding": {k: meta[k] for k in (
                            "selection_source", "entry_tier", "executed",
                            "flow_id", "fallback", "plan_reason",
                            "composition", "composition_table",
                            "profile_hash", "input_shape")
                            if k in meta},
                    }
                for ch in H.CANDIDATES:
                    if ch not in entry["paths"]:
                        continue
                    rh = entry["paths"][ch]
                    rv = entry["paths"]["vendor"]["warm_p50_ms"]
                    ri = entry["paths"][INCUMBENT_LABEL]["warm_p50_ms"]
                    h_trials = rows[ch]["trials"]
                    pooled[ch].extend(h_trials)
                    for cis, pts, ref in (
                            (vendor_ci, vendor_pt, "vendor"),
                            (inc_ci, inc_pt, INCUMBENT_LABEL)):
                        sp = H._boot_ratio(rng, rows[ref]["trials"],
                                           h_trials)
                        if sp:
                            sp["definition"] = (
                                f"median({ref})/median({ch}) "
                                "independent bootstrap (>1 = "
                                "candidate faster)"
                                + ("; PROFILE-BOUND incumbent" if ref
                                   == INCUMBENT_LABEL else ""))
                            cis[ch].append(sp)
                            pts[ch].append(sp["point"])
            c["sessions"][str(s)] = entry
        cell_ok = (len(shapes_all) == len(SESSIONS)
                   and H._entry_shape_uniform(shapes_all))
        c["entry_shapes"] = {str(s): {p: list(shp)
                                      for p, shp in per.items()}
                             for s, per in shapes_all.items()}
        c["entry_shape_uniform"] = cell_ok
        c["gate_flags_by_session"] = flags_per_session
        for ch in H.CANDIDATES:
            v = promotion_verdict(flags_per_session, vendor_ci[ch],
                                  inc_ci[ch], vendor_pt[ch],
                                  inc_pt[ch], cell_ok)
            v["frozen_candidate"] = ch in c["frozen_candidates"]
            c["verdicts"][ch] = v
        # promotion decision inside the frozen set
        passing = [ch for ch in c["frozen_candidates"]
                   if c["verdicts"][ch]["pass"]]
        dec = {"passing_frozen_candidates": passing}
        if len(passing) == 1:
            dec["selected"] = passing[0]
            dec["selection_reason"] = "only passing frozen candidate"
        elif len(passing) >= 2:
            faster = choose_faster({ch: pooled[ch] for ch in passing},
                                   rng)
            dec.update(faster)
        else:
            dec["selected"] = None
            dec["selection_reason"] = "no frozen candidate passed"
        c["promotion"] = dec
        cells_out.append(c)

    promoted = [
        {"key": {"N": c["n"], "batch": c["batch"],
                 "direction": "dit",
                 "transform_type": "c2c",
                 "transform_direction": c["direction"].replace("c2c_", ""),
                 "normalization": "backward"},
         "label": c["label"],
         "candidate": c["promotion"].get("selected"),
         "verdict": (c["verdicts"].get(
             c["promotion"]["selected"]) if c["promotion"].get("selected")
             else None)}
        for c in cells_out if c["promotion"].get("selected")]
    not_promoted = [
        {"label": c["label"], "frozen_candidates": c["frozen_candidates"],
         "reasons": {ch: c["verdicts"][ch]["reasons"]
                     for ch in c["frozen_candidates"]
                     if not c["verdicts"][ch]["pass"]}}
        for c in cells_out if not c["promotion"].get("selected")]

    doc = {
        "schema": "issue13-wp1-promotion-v1",
        "issue": "kunlunxin-fft#13",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": str(SCRIPT.relative_to(ROOT)),
        "environment": {
            "profile_hash": table.get("profile_hash"),
            "env_hash_table": table.get("env_hash"),
            "env_hash_live": live_env,
            "env_fail_closed": (
                "assert_current_environment(load_table()) ran before "
                "any spawn; mismatch raises and aborts the run"),
            "selector_table_sha256": _sha256(SELECTOR_TABLE),
            "composition_table_sha256": (_sha256(COMPOSITION_TABLE)
                                         if COMPOSITION_TABLE.exists()
                                         else None),
            "producer_sha256": {
                "scripts/issue13_wp1_promotion.py": _sha256(SCRIPT),
                "scripts/issue10_wp3_holdout.py": _sha256(
                    ROOT / "scripts" / "issue10_wp3_holdout.py"),
                "scripts/issue10_wp3_perf.py": _sha256(
                    ROOT / "scripts" / "issue10_wp3_perf.py"),
                "src/xpu/selector.py": _sha256(SEL_SRC),
                "src/xpu/composition.py": _sha256(COMP_SRC),
                "src/xpu/entry.py": _sha256(ENTRY_SRC),
            },
        },
        "protocol": {
            "cells": [f"N{n}B{b}:c2c-{'inverse' if inv else 'forward'}"
                      for n, b, inv in cells],
            "frozen_candidates": {
                f"N{n}B{b}:c2c-{'inverse' if inv else 'forward'}": list(cs)
                for n, b, inv, cs in FROZEN_CELLS},
            "sessions": len(SESSIONS),
            "confirmation_session": CONFIRM_SESSION,
            "paths": list(PATHS),
            "spawn": ("one subprocess per (cell, path, session), fresh "
                      "TRITON_CACHE_DIR, spawn order shuffled per "
                      "session (seed recorded) = random interleaving; "
                      "protocol identical to the PR #12 holdout "
                      "(issue10_wp3_holdout worker reused verbatim)"),
            "bootstrap": (f"independent bootstrap B={BOOTSTRAP}, "
                          f"seed={SEED}"),
            "promotion_rules": {
                "correctness": (f"all paths all sessions rel_err <= "
                                f"{TOL_REL}, no fail_closed, all-path "
                                "contract (WP0), uniform entry shape"),
                "ci_gate": (f"per session: bootstrap CI lower bound > "
                            f"{GATE} vs vendor AND profile-bound "
                            "incumbent; M0 attribution only"),
                "spread": (f"per-session speedup point vs cross-session "
                           f"median deviates <= {SPREAD_REL_MAX} for "
                           "both baselines; exceed => not promoted"),
                "both_pass": ("pooled cross-session bootstrap of the "
                              "two candidates' own warm trials; the "
                              "measured faster wins")},
            "input_semantics": (
                "same as PR #12 holdout: complex64 non-zero imag, "
                "b==1 rank-1 (N,) on all five paths, B>1 (B,N), "
                "seeded per (session, n, b)"),
        },
        "sessions": session_meta,
        "cells": cells_out,
        "promotion_summary": {
            "promoted_keys": promoted,
            "not_promoted": not_promoted,
            "note": ("promoted_keys feed scripts/issue13_promote_"
                     "hybrid.py (composition table) — exact keys only, "
                     "no N/B extrapolation (issue #13 WP2)")},
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(doc, indent=2))

    # raw CSV (deliverable: raw rows for every cell/session/path)
    with open(RAW_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "session", "role", "path", "warm_p50_ms",
                    "warm_mad_ms", "rel_err", "correct", "one_shot_ms"])
        for c in cells_out:
            for s in SESSIONS:
                ent = c["sessions"][str(s)]
                for p, v in ent["paths"].items():
                    w.writerow([c["label"], s, ent["role"], p,
                                v["warm_p50_ms"], v["warm_mad_ms"],
                                v["rel_err"], v["correct"],
                                v["one_shot_ms"]])
    print(f"written {OUT}")
    print(f"written {RAW_CSV}")
    print(json.dumps({
        "promoted": [(p["label"], p["candidate"]) for p in promoted],
        "not_promoted": [(p["label"], p["reasons"])
                         for p in not_promoted],
        "elapsed_s": doc["elapsed_s"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
