#!/usr/bin/env python3
"""Issue #9 release-correction P2-7: five machine-readable summaries.

From one immutable release snapshot (plus the recorded tune runs for
scope disclosure) emit, into a fresh directory:

    steady_report.{json,csv}          -- steady-tier coverage/baseline/
                                         correctness per label, with the
                                         five comparisons kept SEPARATE
                                         (current vs v5 production / vs
                                         A1 historical / vs incumbent /
                                         vs bounded oracle / vs vendor)
    e2e_report.{json,csv}             -- E2E tier scope: incumbent/
                                         selector only; vendor E2E
                                         unavailable (no vendor E2E
                                         speedup is ever reported)
    scale_report.{json,csv}           -- scale domain: C2C-forward
                                         single-semantic capability +
                                         correctness evidence (not a
                                         completed search optimization)
    shape_proxy_report.{json,csv}     -- apps domain: uniform-random
                                         shape proxies (no OFDM/radar
                                         structure); B=14/140 capability
                                         probe failures disclosed
    ablation_scope_report.{json,md}   -- what the two bounded rounds did
                                         and did NOT cover (no new
                                         physical kernels; A/P axes are
                                         not a controlled ablation)

Every report carries coverage, input type, timing scope, baseline,
correctness, unsupported handling and explicit limitations.

Usage:
    scripts/issue9_release_reports.py RELEASE_DIR --out DIR
        [--tune-p DIR] [--tune-a DIR]
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.xpu.manifest_contract import iter_raw_rows  # noqa: E402

SCHEMA = "issue9-release-reports-v1"
NONPOW2 = {"N2048B14:c2c-forward", "N2048B140:c2c-forward",
           "N4096B14:c2c-forward", "N4096B140:c2c-forward"}


def _median(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def _geomean(xs):
    import math
    xs = [x for x in xs if isinstance(x, (int, float)) and x > 0]
    if not xs:
        return None
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def _load_cfg():
    import run_issue9 as ri
    return ri.load_config()


def _labels_of(cfg, domain):
    import run_semantic_matrix as sm
    labs = set()
    for n, b, sems in sm.parse_cells_spec(",".join(cfg["domains"][domain]
                                                   ["cells"])):
        labs |= {f"N{n}B{b}:{s}" for s in sems}
    return labs


def _tier_stats(rows, tier, label):
    """Per-candidate median/err-count for one (label, tier)."""
    buckets = {}
    for _p, _ln, r in rows:
        if r.get("label") != label or r.get("tier") != tier:
            continue
        if r.get("status") == "ok" and isinstance(r.get("api_ms"), (int,
                                                                   float)):
            buckets.setdefault(r.get("candidate"), []).append(r)
    out = {}
    for cand, rs in buckets.items():
        errs = [r.get("err") for r in rs
                if isinstance(r.get("err"), (int, float))]
        out[cand] = {
            "median_ms": _median([r["api_ms"] for r in rs]),
            "samples": len(rs),
            "max_err": max(errs) if errs else None,
        }
    return out


def _load_rows(rel):
    return list(iter_raw_rows(rel))


def _case_by_label(summary):
    return {c["label"]: c for c in summary["cases"]}


def steady_report(rel, rows, summary, cfg):
    cases = _case_by_label(summary)
    labels = sorted(cases)
    m = summary["manifest"]
    recs = []
    for lab in labels:
        c = cases[lab]
        st = _tier_stats(rows, "steady", lab)
        recs.append({
            "label": lab,
            "incumbent_ms": c.get("incumbent_ms"),
            "selector_ms": c.get("selector_ms"),
            "previous_ms": c.get("previous_deployed_ms"),
            "oracle_ms": c.get("oracle_ms"),
            "vendor_ms": c.get("vendor_ms"),
            "selector_source": c.get("selector_source"),
            "selector_regret": c.get("selector_regret"),
            "guard_vs_incumbent": c.get("guard_vs_incumbent"),
            "selector_vs_previous": c.get("selector_vs_previous"),
            "vendor_status": c.get("vendor_status"),
            "selector_previous_fallback": c.get(
                "selector_previous_fallback"),
            "steady_candidates": sorted(st),
            "selector_max_err": (st.get("selector") or {}).get("max_err"),
            "steady_samples": sum(v["samples"] for v in st.values()),
        })
    # five comparisons, computed and stored separately (never mixed)
    def _vs(field, base):
        vals = []
        for c in summary["cases"]:
            a, b = c.get(field), c.get(base)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
                    and b > 0:
                vals.append(a / b)   # >1 = selector slower than base
        return {
            "cells": len(vals),
            "geomean_ratio": _geomean(vals),
            "mean_delta": (sum(v - 1 for v in vals) / len(vals)
                           if vals else None),
        }
    regs = [c["selector_regret"] for c in summary["cases"]
            if isinstance(c.get("selector_regret"), (int, float))]
    report = {
        "schema": SCHEMA,
        "report": "steady",
        "release": str(rel),
        "coverage": {
            "labels": len(labels),
            "semantic_cells": len(summary["cases"]),
            "sessions": m.get("sessions"), "trials": m.get("trials"),
            "warmup": m.get("warmup"),
        },
        "input_type": "uniform random complex/real inputs (seeded); "
                      "shape proxy only",
        "timing_scope": m.get("timing_scope"),
        "baseline": "incumbent (M0 production plan)",
        "correctness": {
            "threshold": m.get("error_threshold"),
            "protocol": m.get("protocol_id"),
            "selector_rows_all_ok": all(
                r.get("selector_max_err") is None
                or r["selector_max_err"] <= 1e-4 for r in recs),
        },
        "unsupported": {
            "vendor_unsupported_labels": sorted(
                c["label"] for c in summary["cases"]
                if c.get("vendor_status") not in ("ok", None)),
            "capability_probe_failures": sorted(NONPOW2),
        },
        "comparisons": {
            "current_vs_v5_production": {
                "basis": "P1-2 path identity (134 identical cells) + "
                         "10-cell paired confirmation run",
                "paired": "results/p800/issue9_v61/v5_pair",
                "see": "v5_path_identity.json for the identity split",
            },
            "current_vs_A1_historical_previous": {
                "previous_regressions_gt_2pct": summary["metrics"].get(
                    "previous_regressions_gt_2pct"),
                "intentional_semantic_exceptions": summary["metrics"].get(
                    "intentional_semantic_exceptions"),
                "measured_mean": _vs("selector_ms", "previous_deployed_ms"),
                "note": "informational only; A1 is NOT the v5 regression "
                        "gate (re-review P1-2)",
            },
            "current_vs_incumbent_M0": _vs("selector_ms",
                                           "incumbent_ms"),
            "current_vs_bounded_oracle": {
                "mean_regret": summary["metrics"].get(
                    "mean_selector_regret"),
                "max_regret": summary["metrics"].get(
                    "max_selector_regret"),
                "cells_gt_10pct": sum(1 for r in regs if r > 0.10),
            },
            "current_vs_vendor": {
                "win": summary["metrics"].get("win"),
                "tie": summary["metrics"].get("tie"),
                "loss": summary["metrics"].get("loss"),
                "geomean_speedup": summary["metrics"].get(
                    "geomean_speedup_vs_vendor"),
                "note": "vendor supported subset only; unsupported cells "
                        "are never counted as wins",
            },
        },
        "limitations": [
            "sync API wall time only; device/kernel timers unavailable",
            "steady tier = plan-reuse; one_shot/e2e are separate reports",
            "regret bounded by the declared candidate set (no physical "
            "kernel search this issue)",
        ],
        "cells": recs,
    }
    return report


def e2e_report(rel, rows, summary):
    labels = sorted(_case_by_label(summary))
    recs = []
    for lab in labels:
        e2e = _tier_stats(rows, "e2e", lab)
        one = _tier_stats(rows, "one_shot", lab)
        if not e2e and not one:
            continue
        recs.append({
            "label": lab,
            "e2e": {k: v["median_ms"] for k, v in e2e.items()},
            "one_shot": {k: v["median_ms"] for k, v in one.items()},
            "e2e_candidates": sorted(e2e),
            "e2e_max_err": max([v.get("max_err") or 0
                                for v in e2e.values()], default=None),
        })
    return {
        "schema": SCHEMA,
        "report": "e2e",
        "release": str(rel),
        "coverage": {
            "labels_with_e2e": len(recs),
            "timed_scope": "H2D copy + execute + D2H copy (combined; "
                           "per-hop split unavailable)",
            "candidates": ["incumbent", "selector"],
        },
        "input_type": "uniform random shape proxy (same tensor as "
                      "steady, round-tripped through host)",
        "timing_scope": "e2e and one_shot tiers",
        "baseline": "same-label steady tier (plan-reuse)",
        "correctness": "independent timed-path check per tier "
                       "(P1-1): semantic err_fn on the post-D2H / "
                       "one-shot output; errors are terminal rows",
        "vendor_e2e": {
            "status": "unavailable",
            "note": "no matched vendor E2E contender in the inventory; "
                    "vendor E2E speedup is not produced",
        },
        "unsupported": "cells without one_shot/e2e rows: contenders "
                       "failed the correctness gate first",
        "limitations": [
            "per-hop H2D/D2H split unavailable (combined scope)",
            "one_shot includes per-call plan/dispatch overhead by "
            "definition; not a launch-count measurement",
        ],
        "cells": recs,
    }


def scale_report(rel, summary, cfg):
    labs = _labels_of(cfg, "scale")
    cases = _case_by_label(summary)
    recs = []
    for lab in sorted(labs):
        c = cases.get(lab)
        if c is None:
            continue
        recs.append({
            "label": lab,
            "selector_ms": c.get("selector_ms"),
            "selector_source": c.get("selector_source"),
            "selector_regret": c.get("selector_regret"),
            "vendor_ms": c.get("vendor_ms"),
            "vendor_status": c.get("vendor_status"),
            "measured": c.get("selector_ms") is not None,
        })
    abstain = [r["label"] for r in recs
               if r["selector_source"] in (None, "abstain")]
    return {
        "schema": SCHEMA,
        "report": "scale",
        "release": str(rel),
        "coverage": {
            "domain": "scale (N65536..N524288 C2C-forward)",
            "labels": len(recs),
            "semantic": "c2c-forward only",
        },
        "input_type": "uniform random complex, seeded (shape proxy)",
        "timing_scope": "steady tier",
        "baseline": "incumbent per cell",
        "correctness": "per-cell err <= 1e-4 gate before timing",
        "unsupported": {
            "vendor_unsupported": [r["label"] for r in recs
                                   if r["vendor_status"] != "ok"],
            "selector_abstain": abstain,
        },
        "interpretation": "capability + correctness evidence for large-N "
                          "C2C-forward shapes; NOT a completed scale "
                          "search optimization (most new points abstain)",
        "limitations": [
            "single semantic (c2c-forward); no r2c/c2r scale coverage",
            "no selector search wave was run over this domain "
            "(closure/tune domains are frozen separately)",
        ],
        "cells": recs,
    }


def shape_proxy_report(rel, summary, cfg):
    labs = _labels_of(cfg, "apps")
    cases = _case_by_label(summary)
    recs = []
    for lab in sorted(labs):
        c = cases.get(lab)
        if c is None:
            continue
        probe_fail = lab in NONPOW2
        recs.append({
            "label": lab,
            "batch": int(lab.split("B")[1].split(":")[0])
            if "B" in lab else None,
            "selector_ms": c.get("selector_ms"),
            "selector_source": c.get("selector_source"),
            "vendor_ms": c.get("vendor_ms"),
            "vendor_status": c.get("vendor_status"),
            "capability_probe_failure": probe_fail,
        })
    return {
        "schema": SCHEMA,
        "report": "shape_proxy",
        "release": str(rel),
        "coverage": {
            "domain": "apps (OFDM/radar proxy shapes)",
            "labels": len(recs),
        },
        "input_type": "uniform random input per semantic -- a SHAPE "
                      "proxy only; no OFDM DC/guard-band/modulation "
                      "structure and no radar echo structure are "
                      "synthesized (config's app_proxy_disclosure "
                      "padded-proxy for B=14/140 was NOT executed: "
                      "every legal lowering rejects non-power-of-2 "
                      "batches)",
        "timing_scope": "steady tier",
        "baseline": "incumbent per cell",
        "correctness": "per-cell err <= 1e-4 gate before timing",
        "unsupported": {
            "capability_probe_failures": sorted(NONPOW2),
            "note": "B=14/140 cells are capability probes that FAILED "
                    "(ValueError: non-power-of-2); they are not "
                    "completed application proxies and no padded-proxy "
                    "substitute ran",
            "vendor_unsupported": [r["label"] for r in recs
                                   if r["vendor_status"] != "ok"],
        },
        "limitations": [
            "application claims are limited to shape-level proxies",
            "native OFDM/radar workloads need contract-legal lowerings "
            "for non-power-of-2 batches (Future Work)",
        ],
        "cells": recs,
    }


def ablation_scope_report(rel, summary, cfg, tune_p, tune_a):
    def _tune(path):
        p = Path(path) / "summary.json"
        if not p.is_file():
            return {"status": "not found", "path": str(path)}
        s = json.loads(p.read_text())
        return {
            "path": str(path),
            "oracle_mode": s["manifest"].get("oracle_mode"),
            "labels": len(s["cases"]),
            "mean_regret": s["metrics"].get("mean_selector_regret"),
            "max_regret": s["metrics"].get("max_selector_regret"),
            "deployed_equals_oracle_cells": sum(
                1 for c in s["cases"]
                if isinstance(c.get("selector_regret"), (int, float))
                and c["selector_regret"] < 0.005),
        }
    tp, ta = _tune(tune_p), _tune(tune_a)
    budget = cfg["budgets"]
    return {
        "schema": SCHEMA,
        "report": "ablation_scope",
        "release": str(rel),
        "scope_statement": (
            "Existing candidate sets were fully scanned (tune-p: full "
            "legal processing-unit/flow set) and the deployed winner "
            "re-blocked (tune-a: winner-block sweep) within the "
            "pre-declared two-round budget; no promotable >=2% gain was "
            "found. This is NOT a controlled A/P (architecture-mapping "
            "/ processing-unit) ablation and NOT 'two-axis "
            "implementation and optimization': a94b890..HEAD adds no "
            "new physical kernel."),
        "rounds": {"tune-p": tp, "tune-a": ta},
        "budget": {
            "implementation_rounds": budget["implementation_rounds"],
            "max_candidates_per_cell":
                budget["max_candidates_per_cell"],
            "device_hours_total": budget["device_hours_total"],
        },
        "not_covered": [
            "true batched M2/hybrid/h_dif kernels (dispatch per row/"
            "sub-FFT remains)",
            "resident/pipeline configurations (boundary_residence canary "
            "archived: contract admits only 'gm', no legal resident "
            "candidate)",
            "r2c/c2r adapter-core fusion (phase-split counters "
            "unavailable; fusion gate not demonstrated)",
            "external/third-party local FFT core integration",
            "non-power-of-2 batch lowerings (capability probes failed)",
        ],
        "future_work_ref": (
            "physical candidate-space optimization moves to the next "
            "performance issue (per re-review P2-6)"),
        "limitations": [
            "scan results are within the declared candidate sets only",
            "oracle bounded by measured candidates; regret is not a "
            "global optimality gap",
        ],
    }


def _write_csv(path, recs, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list))
                            else v) for k, v in r.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("release_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tune-p", default=str(ROOT / "results/p800/issue9"
                                             "/tune-p"))
    ap.add_argument("--tune-a", default=str(ROOT / "results/p800/issue9"
                                             "/tune-a"))
    args = ap.parse_args(argv)

    rel = Path(args.release_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((rel / "summary.json").read_text())
    rows = _load_rows(rel)
    cfg = _load_cfg()

    reports = {
        "steady_report": steady_report(rel, rows, summary, cfg),
        "e2e_report": e2e_report(rel, rows, summary),
        "scale_report": scale_report(rel, summary, cfg),
        "shape_proxy_report": shape_proxy_report(rel, summary, cfg),
        "ablation_scope_report": ablation_scope_report(
            rel, summary, cfg, args.tune_p, args.tune_a),
    }
    csv_fields = {
        "steady_report": ["label", "incumbent_ms", "selector_ms",
                          "previous_ms", "oracle_ms", "vendor_ms",
                          "selector_source", "selector_regret",
                          "guard_vs_incumbent", "selector_vs_previous",
                          "vendor_status", "selector_previous_fallback",
                          "selector_max_err"],
        "e2e_report": ["label", "e2e", "one_shot", "e2e_candidates",
                       "e2e_max_err"],
        "scale_report": ["label", "selector_ms", "selector_source",
                         "selector_regret", "vendor_ms", "vendor_status",
                         "measured"],
        "shape_proxy_report": ["label", "batch", "selector_ms",
                               "selector_source", "vendor_ms",
                               "vendor_status", "capability_probe_failure"],
    }
    for name, rep in reports.items():
        (out / f"{name}.json").write_text(
            json.dumps(rep, indent=2, sort_keys=True) + "\n")
        if name in csv_fields:
            _write_csv(out / f"{name}.csv", rep["cells"],
                       csv_fields[name])
        else:
            md = [f"# {name}", "",
                  f"release: {rel}", "",
                  "```json",
                  json.dumps(rep, indent=2, sort_keys=True),
                  "```", ""]
            (out / f"{name}.md").write_text("\n".join(md))
        print(f"report: {out / (name + '.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
