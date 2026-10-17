#!/usr/bin/env python3
"""Bounded failure-cell diagnostics for issue #6 WP3 (PR-E).

Turns the semantic-matrix summary (and, when present, its raw rows)
into a per-cell verdict with an *auditable ladder*: every step records
which number fired it and whether it came from a fresh measurement, a
measured artifact (summary/raw), or an analytic model. Nothing here
relabels host composition / API wall time as a resident kernel counter.

What each stage is (and is not):

* classification        — summary-derived failure kind (vendor-unsupported,
                          correctness, loses/tie/wins)
* flow_sensitivity      — FIXED processing unit (triton local row core),
                          varies flow: per-oracle-candidate steady p50 from
                          the release raw rows; selector regret vs best
                          oracle recomputed from the same rows
* unit_sensitivity      — FIXED flow semantics at this cell, varies unit:
                          selector vs incumbent(triton) vs vendor whole
                          transform (+ composition table entry when the
                          cell is exact-shape promoted)
* batch_scaling         — summary-derived speedup_vs_vendor across batch
                          sizes at this N (worker-wave/tail indicator)
* dispatch_probe        — fresh MEASURED stage on device: first-call,
                          interleaved steady p50/MAD/p95 for
                          selector/incumbent/vendor, entry-vs-core probe
                          (entry total vs bare runner invoke), plan
                          creation cost
* ledger                — ANALYTIC launches/GM-bytes model for known
                          flow families; absent families are reported as
                          unmodelled, never invented

Usage (device, default top-4 losing cells of the newest release):

    python3 scripts/diagnose_failing_cells.py
    python3 scripts/diagnose_failing_cells.py --cells N1024B256:c2c-forward
    python3 scripts/diagnose_failing_cells.py --skip-timing   # host only

Outputs <out>/verdicts.json + <out>/report.md (default
results/p800/diagnostics/<YYYY-MM-DD>/).
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# ladder order (labels are stable contract for tests/docs)
LADDER = (
    "correctness",
    "external-capability(vendor)",
    "entry-dispatch-overhead",
    "sync-handoff-jitter",
    "worker-waves-tail",
    "flow-not-selected",
    "processing-unit-bound",
    "mixed",
)

DISPATCH_SHARE_MIN = 0.10
P95_P50_MAX = 1.15
BATCH_RATIO_MIN = 0.70
ORACLE_SPREAD_MIN = 0.50     # best oracle beats selector by >50% => flow
NEAR_ORACLE_MAX = 1.10       # selector within 10% of best oracle
UNIT_GAP_MIN = 1.20          # selector >= vendor * 1.2 => unit gap


def classify(case: dict, failures: list[dict]) -> dict:
    """Summary-derived classification of one case (never measured)."""
    label = case["label"]
    f = [x for x in failures if x.get("label") == label
         and x.get("tier") == "correctness"]
    if any(x.get("kind") == "wrong-value" for x in f):
        return {"kind": "correctness", "detail": f[0].get("reason")}
    if case.get("vendor_status") == "unsupported":
        return {"kind": "vendor-unsupported",
                "detail": "vendor rejected this shape"}
    if case.get("vendor_status") != "ok" or case.get("vendor_ms") is None:
        # failures list marks vendor correctness/unsupported rows
        vf = [x for x in failures if x.get("label") == label
              and x.get("candidate") == "vendor"]
        if vf:
            return {"kind": "vendor-unsupported",
                    "detail": vf[0].get("reason")}
        return {"kind": "no-vendor-data", "detail": None}
    sp = case.get("speedup_vs_vendor")
    if sp is None:
        return {"kind": "no-vendor-data", "detail": None}
    if sp > 1.02:
        return {"kind": "wins", "detail": sp}
    if sp >= 0.98:
        return {"kind": "tie", "detail": sp}
    return {"kind": "loses", "detail": sp}


def pick_cells(cases: list[dict], failures: list[dict],
               top_k: int) -> list[dict]:
    """Worst *actionable* cells: correctness first, then lowest
    speedup_vs_vendor among vendor-supported cases. vendor-unsupported
    cells are classified separately (no local action)."""
    by_label = {c["label"]: c for c in cases}
    correctness = [x for x in failures
                   if x.get("tier") == "correctness"
                   and x.get("kind") == "wrong-value"
                   and x.get("label") in by_label]
    picked, seen = [], set()
    for x in correctness:
        if x["label"] not in seen:
            picked.append(by_label[x["label"]])
            seen.add(x["label"])
    supported = [c for c in cases
                 if c.get("vendor_status") == "ok"
                 and c.get("speedup_vs_vendor") is not None
                 and c["label"] not in seen]
    supported.sort(key=lambda c: c["speedup_vs_vendor"])
    for c in supported:
        if len(picked) >= top_k:
            break
        if c["label"] not in seen:
            picked.append(c)
            seen.add(c["label"])
    return picked[:top_k]


def _load_raw_p50(release_dir: Path, label: str) -> dict[str, dict]:
    """Per-candidate steady p50/MAD/p95 from release raw rows
    (measured artifact; no device needed). Rows carry a combined
    ``label`` field (``N1024B256:c2c-forward``) + ``candidate``."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_smm_helpers", SCRIPTS / "run_semantic_matrix.py")
    smm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smm)
    per: dict[str, list[float]] = {}
    for path in sorted(release_dir.glob("raw_session*.jsonl*")):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                if (r.get("label") != label
                        or r.get("tier") != "steady"
                        or r.get("api_ms") is None):
                    continue
                per.setdefault(r["candidate"], []).append(
                    float(r["api_ms"]))
    out = {}
    for name, xs in per.items():
        out[name] = {"p50": smm._median(xs), "mad": smm._mad(xs),
                     "p95": smm._pct(xs, 95), "n": len(xs)}
    return out


def flow_sensitivity(case: dict, release_dir: Path | None) -> dict:
    """FIXED unit (triton local row candidates), varies FLOW: oracle
    candidate spread + selector regret recomputed from raw rows."""
    if release_dir is None or not release_dir.exists():
        return {"available": False,
                "why": "no release dir for raw rows"}
    stats = _load_raw_p50(release_dir, case["label"])
    oracles = {k: v for k, v in stats.items() if k.startswith("oracle:")}
    if not oracles:
        return {"available": False, "why": "no oracle rows in raw"}
    best_name = min(oracles, key=lambda k: oracles[k]["p50"])
    best = oracles[best_name]["p50"]
    sel = stats.get("selector", {}).get("p50")
    regret = (sel / best) if (sel and best) else case.get("selector_regret")
    spread = (max(v["p50"] for v in oracles.values()) - best) / best
    return {
        "available": True,
        "kind": "measured-artifact (release raw rows)",
        "unit_fixed": "triton local row core (oracle:* candidates)",
        "candidates": {k: v["p50"] for k, v in sorted(oracles.items())},
        "best": best_name, "best_ms": best,
        "selector_ms": sel,
        "regret_vs_best": regret,
        "spread_rel": spread,
    }


def unit_sensitivity(case: dict, cases: list[dict],
                     composition_table: dict | None) -> dict:
    """FIXED flow semantics at this cell, varies unit: selector vs
    incumbent (triton production) vs vendor whole transform."""
    same = [c for c in cases
            if c["N"] == case["N"] and c["batch"] == case["batch"]
            and c["semantic"] == case["semantic"]]
    c0 = same[0] if same else case
    comp_entry = None
    if composition_table:
        for e in composition_table.get("entries", []):
            if (e.get("N") == c0["N"] and e.get("batch") == c0["batch"]
                    and e.get("direction") == "dit"
                    and e.get("precision") == "fp32"):
                comp_entry = {
                    "flow_id": e.get("flow_id"),
                    "candidate_ms": e.get("candidate_ms"),
                    "incumbent_ms": e.get("incumbent_ms"),
                    "guard_pct": e.get("guard_pct"),
                }
    sel = c0.get("selector_ms")
    ven = c0.get("vendor_ms")
    inc = c0.get("incumbent_ms")
    return {
        "available": sel is not None and ven is not None,
        "kind": "measured-artifact (summary medians)",
        "selector_ms": sel, "incumbent_ms": inc, "vendor_ms": ven,
        "selector_source": c0.get("selector_source"),
        "selector_over_vendor": (sel / ven) if ven else None,
        "composition_entry": comp_entry,
    }


def batch_scaling(case: dict, cases: list[dict]) -> dict:
    """speedup_vs_vendor across batch sizes at this N (same semantic);
    a collapse with B points at worker waves / tail."""
    peers = [c for c in cases
             if c["N"] == case["N"]
             and c["semantic"] == case["semantic"]
             and c.get("speedup_vs_vendor") is not None]
    by_b = {c["batch"]: c["speedup_vs_vendor"] for c in peers}
    if len(by_b) < 2:
        return {"available": False, "why": "<2 batch points at this N"}
    vals = list(by_b.values())
    worst_b = min(by_b, key=by_b.get)
    best_b = max(by_b, key=by_b.get)
    return {
        "available": True,
        "kind": "summary-derived",
        "speedup_by_batch": {str(k): v for k, v in sorted(by_b.items())},
        "ratio_worst_over_best": (min(vals) / max(vals)) if max(vals) else 0,
        "worst_B": worst_b, "best_B": best_b,
    }


def ledger(plan_info: dict) -> dict:
    """ANALYTIC launches/GM-bytes for known flow families. Unknown
    families stay unmodelled (honest absence > invented numbers)."""
    flow = str(plan_info.get("flow_id") or "")
    n = plan_info.get("n")
    b = plan_info.get("batch")
    if n is None:
        return {"available": False, "why": "no plan n"}
    import math
    if not n or (n & (n - 1)) != 0:
        return {"available": False, "why": "n not a power of two"}
    lg = int(math.log2(n))
    tile = 64  # run_batch_release.TILE
    launches = None
    if "m2" in flow.lower():
        # bench/run_batch_release._m2_launches: bitrev + groups + tail
        launches = 2 + lg - int(math.log2(tile))
        kind = "analytic-model (run_batch_release m2 formula)"
    elif "m5" in flow.lower():
        launches = (lg + 1) // 2   # run_batch_release._m5_launches
        kind = "analytic-model (run_batch_release m5 formula)"
    if launches is None:
        return {"available": False,
                "why": f"no launch model for flow {flow!r}",
                "kind": "analytic-model"}
    # complex64 read+write per launch (same convention as _gm_bytes;
    # an analytical estimate, explicitly not a hardware counter)
    gm_bytes = launches * 2 * n * b * 8
    return {"available": True, "kind": kind, "launches": launches,
            "gm_bytes": gm_bytes, "estimate": True,
            "basis": "oracle_candidate (bounded-best flow from the "
                     "summary; the selector's own flow family is not "
                     "carried in summary rows)"}


def verdict(stage: dict) -> dict:
    """Deterministic ladder over stage results (host-testable pure).

    Each rung records fired/not + the exact number that decided it, so
    the verdict is re-checkable against the artifacts it cites.
    """
    rungs = []

    def rung(label, fired, why, evidence):
        rungs.append({"label": label, "fired": bool(fired),
                      "why": why, "evidence": evidence})

    cls = stage.get("classification") or {}
    kind = cls.get("kind")
    rung("correctness", kind == "correctness",
         "wrong-value failure", cls)
    rung("external-capability(vendor)",
         kind == "vendor-unsupported",
         "vendor rejected shape; no local action", cls)

    disp = stage.get("dispatch_probe") or {}
    share = disp.get("dispatch_share")
    rung("entry-dispatch-overhead",
         share is not None and share > DISPATCH_SHARE_MIN,
         f"entry-core share {share} > {DISPATCH_SHARE_MIN}",
         {"dispatch_share": share})

    p95p50 = disp.get("p95_over_p50")
    rung("sync-handoff-jitter",
         p95p50 is not None and p95p50 > P95_P50_MAX,
         f"selector p95/p50 {p95p50} > {P95_P50_MAX}",
         {"p95_over_p50": p95p50})

    bs = stage.get("batch_scaling") or {}
    ratio = bs.get("ratio_worst_over_best")
    rung("worker-waves-tail",
         ratio is not None and ratio < BATCH_RATIO_MIN,
         f"speedup worst/best batch {ratio} < {BATCH_RATIO_MIN}",
         bs if bs.get("available") else {})

    fs = stage.get("flow_sensitivity") or {}
    reg = fs.get("regret_vs_best")
    rung("flow-not-selected",
         reg is not None and reg > NEAR_ORACLE_MAX,
         f"selector/best-oracle {reg} > {NEAR_ORACLE_MAX}",
         {"regret_vs_best": reg, "best": fs.get("best")})

    us = stage.get("unit_sensitivity") or {}
    gap = us.get("selector_over_vendor")
    near = reg is not None and reg <= NEAR_ORACLE_MAX
    rung("processing-unit-bound",
         near and gap is not None and gap > UNIT_GAP_MIN,
         f"near-oracle ({reg}) yet selector/vendor {gap} > "
         f"{UNIT_GAP_MIN}: candidate set exhausted",
         {"gap": gap, "regret_vs_best": reg})

    fired = next((r for r in rungs if r["fired"]
                  and r["label"] in LADDER[:-1]), None)
    label = fired["label"] if fired else "mixed"
    return {"verdict": label, "ladder_order": list(LADDER),
            "rungs": rungs,
            "note": "first fired rung wins; mixed = no rung fired "
                    "within the bounded ladder (residual gap reported "
                    "as-is)"}


def _probe_case(case: dict, trials: int, warmup: int) -> dict:
    """MEASURED device stage: first call, interleaved steady timing,
    entry-vs-core dispatch probe, plan creation."""
    import importlib.util
    import time
    import torch
    spec = importlib.util.spec_from_file_location(
        "_smm_helpers", SCRIPTS / "run_semantic_matrix.py")
    smm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smm)
    from src.xpu.config import DeviceProfile
    from src.xpu.entry import execute_fft_plan
    from src.xpu.selector import load_table

    n, b, sem = case["N"], case["batch"], case["semantic"]
    direction = "dit"   # matrix direction for every semantic
    table = load_table()
    prof = DeviceProfile.load().hash()
    plan = smm._entry_plan(direction, n, b, sem, prof, table)
    x = smm._make_input("random", n, b, sem, 1234)

    def sync():
        torch.cuda.synchronize()

    # first call after plan creation (fresh execution path)
    t0 = time.perf_counter()
    execute_fft_plan(plan, x)
    sync()
    first_ms = (time.perf_counter() - t0) * 1e3

    incumbent = smm._entry_plan(direction, n, b, sem, None, table)
    fns = {
        "selector": (lambda: execute_fft_plan(plan, x)),
        "incumbent": (lambda: execute_fft_plan(incumbent, x)),
    }
    try:
        smm._vendor_call(x, sem, n, b)
        fns["vendor"] = (lambda: smm._vendor_call(x, sem, n, b))
    except Exception:  # vendor-unsupported cell: skip that contender
        pass
    samples = smm._timed_steady(fns, warmup, trials, seed=1234 * 1000 + 1)

    # dispatch probe: bare runner invoke (no plan checks/getattr).
    # Bounded to c2c-forward: r2c/c2r prep (real packing, Hermitian
    # reconstruction, crop) and inverse conj/scale are real work, not
    # dispatch, and are excluded by not probing those cells here.
    disp = {"applicable": (
        plan.params.get("flow_kind") != "composition"
        and plan.transform_type == "c2c"
        and plan.transform_direction == "forward")}
    if not disp["applicable"]:
        if plan.params.get("flow_kind") == "composition":
            disp["why"] = ("composition flow: core measured via canary "
                           "(run_hybrid), entry share not probed here")
        else:
            disp["why"] = ("dispatch probe bounded to c2c-forward "
                           "(real-compat/epilogue prep excluded)")
    if disp["applicable"]:
        module = __import__("src.xpu.runner", fromlist=[plan.executed])
        fn = getattr(module, plan.executed)
        from src.xpu.entry import _invoke
        block = plan.block or 1024
        inner_fns = {"core": (lambda: _invoke(fn, x, plan.direction,
                                              block, plan.params))}
        inner = smm._timed_steady(inner_fns, warmup, trials,
                                  seed=1234 * 1000 + 2)["core"]
        entry_ms = samples["selector"]
        e50 = smm._median(entry_ms)
        i50 = smm._median(inner)
        disp.update({
            "kind": "measured",
            "entry_p50": e50, "core_p50": i50,
            "dispatch_share": (e50 - i50) / e50 if e50 else None,
            "scope": "entry total vs bare runner invoke "
                     "(forward c2c: no epilogue cost)",
        })

    # plan creation cost (host plan resolution, measured)
    t0 = time.perf_counter()
    for _ in range(5):
        smm._entry_plan(direction, n, b, sem, prof, table)
    plan_creation_ms = (time.perf_counter() - t0) * 1e3 / 5

    sel = samples["selector"]
    return {
        "available": True, "kind": "measured (fresh, this process)",
        "first_call_ms": first_ms,
        "selector_p50": smm._median(sel),
        "selector_mad": smm._mad(sel),
        "selector_p95": smm._pct(sel, 95),
        "p95_over_p50": (smm._pct(sel, 95) / smm._median(sel)
                         if smm._median(sel) else None),
        "incumbent_p50": smm._median(samples["incumbent"]),
        "vendor_p50": (smm._median(samples["vendor"])
                       if "vendor" in samples else None),
        "dispatch": disp,
        "dispatch_share": disp.get("dispatch_share"),
        "plan_creation_ms": plan_creation_ms,
        "trials": trials,
    }


def analyze_cell(case: dict, cases: list[dict], failures: list[dict],
                 release_dir: Path | None, composition_table: dict | None,
                 skip_timing: bool, trials: int, warmup: int) -> dict:
    stage = {
        "cell": case["label"],
        "classification": classify(case, failures),
        "flow_sensitivity": flow_sensitivity(case, release_dir),
        "unit_sensitivity": unit_sensitivity(case, cases,
                                             composition_table),
        "batch_scaling": batch_scaling(case, cases),
    }
    plan_info = {
        "flow_id": case.get("oracle_candidate", ""),  # placeholder
        "n": case["N"], "batch": case["batch"],
    }
    stage["ledger"] = ledger(plan_info)
    if not skip_timing:
        try:
            stage["dispatch_probe"] = _probe_case(case, trials, warmup)
        except Exception as e:  # noqa: BLE001 - keep the ladder running
            stage["dispatch_probe"] = {
                "available": False,
                "why": f"{type(e).__name__}: {e}"}
    else:
        stage["dispatch_probe"] = {"available": False,
                                   "why": "--skip-timing"}
    stage["verdict"] = verdict(stage)
    return stage


def _render_md(results: list[dict], meta: dict) -> str:
    lines = [
        "# WP3 failure-cell diagnostics (bounded)",
        "",
        f"- generated: {meta['generated']}",
        f"- source summary: {meta['summary']}",
        f"- commit: {meta['commit']}",
        f"- protocol: {meta.get('protocol_id')}",
        "- labels are ladder verdicts; every rung carries the number "
        "that fired it and its evidence kind (measured / "
        "measured-artifact / summary-derived / analytic-model). "
        "No counter below is a resident-kernel or hardware performance "
        "counter; host/API timings are labelled as such.",
        "",
    ]
    for r in results:
        lines.append(f"## {r['cell']} — {r['verdict']['verdict']}")
        lines.append("")
        cls = r["classification"]
        lines.append(f"- classification: `{cls.get('kind')}` "
                     f"({cls.get('detail')})")
        for rung in r["verdict"]["rungs"]:
            mark = "**fired**" if rung["fired"] else "no"
            lines.append(f"  - {rung['label']}: {mark} — {rung['why']}")
        for stage in ("flow_sensitivity", "unit_sensitivity",
                      "batch_scaling", "ledger", "dispatch_probe"):
            s = r.get(stage) or {}
            lines.append(f"- {stage}: "
                         + (json.dumps(s, sort_keys=True)
                            if s.get("available") is not False
                            else f"unavailable ({s.get('why')})"))
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default=None,
                    help="summary.json (default: newest release)")
    ap.add_argument("--cells", default=None,
                    help="comma list of N<sep>B<sep>semantic labels, "
                         "e.g. N1024B256:c2c-forward")
    ap.add_argument("--top", type=int, default=4,
                    help="auto-pick the K worst actionable cells")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    if args.summary:
        summary_path = Path(args.summary)
    else:
        releases = sorted((ROOT / "results" / "p800" / "releases")
                          .glob("*/summary.json"))
        if not releases:
            print("no release summaries found", file=sys.stderr)
            return 1
        summary_path = releases[-1]
    summary = json.loads(summary_path.read_text())
    release_dir = summary_path.parent
    cases = summary["cases"]
    failures = summary.get("failures", [])

    if args.cells:
        wanted = {w.strip() for w in args.cells.split(",") if w.strip()}
        picked = [c for c in cases if c["label"].split(":", 1)[0] in wanted
                  or c["label"] in wanted]
        if not picked:
            print(f"no cases match {wanted}", file=sys.stderr)
            return 1
    else:
        picked = pick_cells(cases, failures, args.top)

    comp_table_path = ROOT / "config" / "composition_table.json"
    comp_table = (json.loads(comp_table_path.read_text())
                  if comp_table_path.exists() else None)

    results = []
    for case in picked:
        print(f"analyzing {case['label']} ...", flush=True)
        results.append(analyze_cell(
            case, cases, failures, release_dir, comp_table,
            args.skip_timing, args.trials, args.warmup))

    out_dir = Path(args.out) if args.out else (
        ROOT / "results" / "p800" / "diagnostics"
        / datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    out_dir.mkdir(parents=True, exist_ok=True)
    import subprocess
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT,
        capture_output=True, text=True).stdout.strip()
    meta = {"generated": datetime.now(timezone.utc).isoformat(),
            "summary": str(summary_path.relative_to(ROOT)
                            if ROOT in summary_path.parents
                            else summary_path),
            "commit": commit,
            "protocol_id": summary.get("protocol_id")}
    (out_dir / "verdicts.json").write_text(
        json.dumps({"meta": meta, "cells": results}, indent=1,
                   sort_keys=True) + "\n")
    (out_dir / "report.md").write_text(_render_md(results, meta))
    print(f"wrote {out_dir}/verdicts.json and report.md")
    for r in results:
        print(f"  {r['cell']}: {r['verdict']['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
