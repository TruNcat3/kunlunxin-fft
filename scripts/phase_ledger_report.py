#!/usr/bin/env python3
"""WP2 phase/launch ledger report (issue #8, PR-F part).

Per selected cell:

* **fresh measured** phase breakdown of the deployed selector and the
  incumbent (``ledger.measure_phases``: per-phase synchronize; attribution
  mode, explicitly not the headline protocol timing);
* **analytic estimate** launch/GM/flop ledger + real-compat op list
  (``ledger.launch_ledger``, reusing cost_model families; vendor launches
  stay opaque);
* **summary-derived** batch-fill table (per-point cost vs batch, from
  the release summary medians) for the cell's N/semantic.

Every block carries its evidence kind; no block claims hardware
performance counters or resident-kernel time for host/composition/API
wall measurements.

    CUDA_VISIBLE_DEVICES=6 python3 scripts/phase_ledger_report.py --top 4
    python3 scripts/phase_ledger_report.py --skip-measure   # host stages
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))


def _smm():
    spec = importlib.util.spec_from_file_location(
        "_smm_helpers", ROOT / "scripts" / "run_semantic_matrix.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pick(cases: list[dict], failures: list[dict], cells: str | None,
          top: int) -> list[dict]:
    if cells:
        wanted = {w.strip() for w in cells.split(",") if w.strip()}
        picked = [c for c in cases
                  if c["label"] in wanted
                  or c["label"].split(":", 1)[0] in wanted]
        if not picked:
            raise SystemExit(f"no cases match {wanted}")
        return picked
    from importlib import import_module
    diag = import_module("diagnose_failing_cells")
    return diag.pick_cells(cases, failures, top)


def _measure(plan, x, trials: int, warmup: int) -> dict:
    from src.xpu.ledger import measure_phases
    return measure_phases(plan, x, trials=trials, warmup=warmup)


def _vendor_block(x, sem: str, n: int, b: int,
                  trials: int, warmup: int) -> dict:
    """Fresh measured vendor wall (single host API call; no internals)."""
    import statistics
    import time
    import torch
    smm = _smm()
    try:
        smm._vendor_call(x, sem, n, b)
    except Exception as e:  # noqa: BLE001
        return {"available": False,
                "why": f"{type(e).__name__}: {e}"}
    xs = []
    for i in range(max(1, warmup) + max(1, trials)):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        smm._vendor_call(x, sem, n, b)
        torch.cuda.synchronize()
        if i >= max(1, warmup):
            xs.append((time.perf_counter() - t0) * 1e3)
    return {
        "available": True,
        "kind": "fresh measured (vendor host API wall; single call, "
                "extra synchronize added)",
        "wall_p50_ms": statistics.median(xs),
        "wall_mean_ms": statistics.fmean(xs),
        "trials": len(xs),
        "note": "vendor has no exposed internal phases on this stack",
    }


def analyze_cell(case: dict, summary_cases: list[dict], trials: int,
                 warmup: int, skip_measure: bool) -> dict:
    from src.xpu.ledger import batch_fill_analysis, launch_ledger
    n, b, sem = case["N"], case["batch"], case["semantic"]
    block = {"cell": case["label"],
             "summary_context": {
                 "selector_ms": case.get("selector_ms"),
                 "incumbent_ms": case.get("incumbent_ms"),
                 "vendor_ms": case.get("vendor_ms"),
                 "selector_source": case.get("selector_source"),
                 "kind": "summary-derived (release summary medians)"}}
    block["batch_fill"] = batch_fill_analysis(
        [c for c in summary_cases
         if c["N"] == n and c["semantic"] == sem])
    if skip_measure:
        block["phases"] = {"available": False, "why": "--skip-measure"}
        block["launches"] = {"available": False, "why": "--skip-measure"}
        return block

    import torch
    from src.xpu.config import DeviceProfile
    from src.xpu.selector import load_table
    smm = _smm()
    table = load_table()
    prof = DeviceProfile.load().hash()
    plan = smm._entry_plan("dit", n, b, sem, prof, table)
    inc = smm._entry_plan("dit", n, b, sem, None, table)
    x = smm._make_input("random", n, b, sem, 1234)

    block["launches"] = {
        "selector": launch_ledger(plan),
        "incumbent": launch_ledger(inc),
    }
    block["phases"] = {
        "selector": _measure(plan, x, trials, warmup),
        "incumbent": _measure(inc, x, trials, warmup),
        "vendor": _vendor_block(x, sem, n, b, trials, warmup),
        "plan_creation": {
            "kind": "fresh measured (host plan resolution, no device)",
            **_plan_creation(plan, smm, n, b, sem, prof, table),
        },
    }
    return block


def _plan_creation(plan, smm, n, b, sem, prof, table) -> dict:
    import time
    xs = []
    for _ in range(5):
        t0 = time.perf_counter()
        smm._entry_plan("dit", n, b, sem, prof, table)
        xs.append((time.perf_counter() - t0) * 1e3)
    return {"mean_ms": sum(xs) / len(xs), "reps": len(xs)}


def _render(results: list[dict], meta: dict) -> str:
    lines = [
        "# WP2 phase/launch ledger",
        "",
        f"- generated: {meta['generated']}",
        f"- source summary: {meta['summary']}",
        f"- commit: {meta['commit']}",
        "- evidence kinds: `fresh measured` (phase timers with extra "
        "synchronize — attribution only, headline protocol timings run "
        "with the ledger off), `analytic estimate` (cost_model launch "
        "families + real-compat op bytes), `summary-derived` (release "
        "medians). None of these are hardware performance counters or "
        "resident-kernel claims.",
        "",
    ]
    for r in results:
        lines.append(f"## {r['cell']}")
        ctx = r["summary_context"]
        lines.append(f"- summary: selector {ctx['selector_ms']} ms / "
                     f"incumbent {ctx['incumbent_ms']} ms / vendor "
                     f"{ctx['vendor_ms']} ms "
                     f"(source={ctx['selector_source']})")
        ph = r.get("phases") or {}
        if ph.get("available") is False or "why" in ph:
            lines.append(f"- phases: unavailable ({ph.get('why')})")
        else:
            for who in ("selector", "incumbent"):
                p = ph.get(who) or {}
                lines.append(
                    f"- phases[{who}] ({p.get('kind')}): "
                    f"trial wall p50 {p.get('trial_wall_p50_ms'):.4f} ms; "
                    + ", ".join(
                        f"{k}={v['mean_ms']:.4f}ms"
                        for k, v in (p.get("phases") or {}).items()))
                if p.get("inter_phase_host_overhead_ms") is not None:
                    lines.append(
                        f"  - inter-phase host overhead (unattributed): "
                        f"{p['inter_phase_host_overhead_ms']:.4f} ms")
            vw = ph.get("vendor") or {}
            lines.append(
                "- vendor: "
                + (f"wall p50 {vw.get('wall_p50_ms'):.4f} ms "
                   f"({vw.get('kind')})"
                   if vw.get("available")
                   else f"unavailable ({vw.get('why')})"))
            pc = ph.get("plan_creation") or {}
            lines.append(f"- plan creation: mean "
                         f"{pc.get('mean_ms', 0):.4f} ms "
                         f"({pc.get('kind')})")
        lg = r.get("launches") or {}
        if lg.get("selector"):
            s = lg["selector"]
            if s.get("available"):
                lines.append(
                    f"- launches[selector] ({s['kind']}): "
                    f"family={s['family']} launches={s['launches']} "
                    f"gm_bytes={s['gm_bytes']} flops={s['flops']}")
            else:
                lines.append(
                    f"- launches[selector]: unavailable ({s.get('why')})")
            for op in (s.get("real_compat_ops") or []):
                lines.append(f"  - real-compat op: {op['op']} "
                             f"(in {op['in_bytes']} B, out "
                             f"{op['out_bytes']} B, {op['kind']})")
        for bf in (r.get("batch_fill") or []):
            g = bf["selector_per_point_growth_smallest_to_largest_B"]
            lines.append(
                f"- batch_fill N={bf['N']} {bf['semantic']} "
                f"({bf['kind']}): per-point growth {g and round(g, 2)}x "
                f"smallB->largeB, poor_batch_fill={bf['poor_batch_fill']}"
                f"; {bf['conclusion_scope']}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--cells", default=None)
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-measure", action="store_true")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    if args.summary:
        summary_path = Path(args.summary)
    else:
        releases = sorted((ROOT / "results" / "p800" / "releases")
                          .glob("*/summary.json"))
        if not releases:
            raise SystemExit("no release summaries found")
        summary_path = releases[-1]
    summary = json.loads(summary_path.read_text())
    results = []
    for case in _pick(summary["cases"], summary.get("failures", []),
                      args.cells, args.top):
        print(f"ledger: {case['label']} ...", flush=True)
        results.append(analyze_cell(
            case, summary["cases"], args.trials, args.warmup,
            args.skip_measure))

    out_dir = Path(args.out) if args.out else (
        ROOT / "results" / "p800" / "diagnostics"
        / datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    out_dir.mkdir(parents=True, exist_ok=True)
    import subprocess
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    meta = {"generated": datetime.now(timezone.utc).isoformat(),
            "summary": str(summary_path), "commit": commit,
            "protocol_id": summary.get("protocol_id")}
    (out_dir / "phase_ledger.json").write_text(
        json.dumps({"meta": meta, "cells": results}, indent=1,
                   sort_keys=True, default=str) + "\n")
    (out_dir / "phase_ledger.md").write_text(_render(results, meta))
    print(f"wrote {out_dir}/phase_ledger.json and phase_ledger.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
