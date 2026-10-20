#!/usr/bin/env python3
"""Issue #9 WP3: same-source attribution report for profile anchors.

Reads one run_issue9 ``--phase profile`` run directory (summary.json +
raw sessions), emits per-anchor tier medians (plan creation, steady,
one-shot, e2e), the measurable overhead deltas, and a disclosed
hypothesis label for every anchor.

Release-correction (issue #9 re-review P1-3): the six WP3 class names
are now ``hypothesis`` labels with ``confidence`` attached, because the
dominance rules run over quantities that are themselves disclosed as
unavailable (phase splits, launches, workspace, waves).  A hypothesis
with ``confidence=unverified`` is NOT direct causal evidence and must
not be used as justification for skipping an optimization.  Only the
selector-coverage deficit (regret vs bounded oracle) is measured
directly and carries ``confidence=measured``.

Everything that needs hardware counters (visible launches,
workspace/live set, measured GM bytes, worker waves) is reported as
``unavailable`` or ``analytic estimate`` -- never as measured.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xpu.manifest_contract import iter_raw_rows  # noqa: E402

SCHEMA = "issue9-attribution-v2"

# six WP3 class names, now emitted as hypothesis labels (see docstring)
CLASS_SELECTOR = "selector-coverage dominated"
CLASS_LAUNCH = "launch-dispatch dominated"
CLASS_CORE = "core-invoke dominated"
CLASS_GM = "boundary-GM-traffic dominated"
CLASS_TAIL = "worker-occupancy-tail dominated"
CLASS_REAL = "real-adapter dominated"
CLASSES = (CLASS_SELECTOR, CLASS_LAUNCH, CLASS_CORE, CLASS_GM,
           CLASS_TAIL, CLASS_REAL)

# confidence vocabulary: only the regret deficit is a measured fact;
# every dominance rule over unavailable counters is an unverified
# hypothesis (issue #9 re-review P1-3).
CONF_MEASURED = "measured"
CONF_UNVERIFIED = "unverified"
EVIDENCE_STATUS = (
    "rule-derived hypotheses over disclosed-unavailable counters; not "
    "direct causal evidence; must not be used as justification for "
    "skipping an optimization")

# disclosed rule thresholds (analytic policy, not measured evidence)
DISPATCH_FRAC = 0.5    # one-shot delta >= 50% of steady -> launch bound
TRANSFER_FRAC = 1.0    # e2e delta >= steady -> GM/transfer bound
REGRET_MAX = 0.10      # selection still losing to the oracle

WORKER_FLOWS = ("baseline-global-stage", "batch-spread", "batch-folded")
UNAVAILABLE = [
    "visible launches: no hardware counter export (analytic only)",
    "workspace / live set: not instrumented this round (unavailable)",
    "GM bytes, bandwidth, occupancy: analytic estimate only",
    "pack/core/stitch/hermitian/crop/scale phase split: unavailable "
    "(no in-kernel counter)",
    "worker/block/wave/tail occupancy: analytic from plan geometry "
    "only",
    "E2D H2D/D2H split: timed as one combined e2e scope "
    "(per-hop split unavailable)",
]


def _median(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return statistics.median(xs) if xs else None


def load_run(run_dir):
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    rows = list(iter_raw_rows(run_dir))
    return summary, rows


def tier_medians(rows, label):
    """Per-candidate tier medians for one label."""
    buckets = {}
    for _p, _ln, row in rows:
        if row.get("label") != label or row.get("tier") is None:
            continue
        if row.get("status") != "ok" or not row.get("api_ms"):
            continue
        buckets.setdefault((row["candidate"], row["tier"]), []).append(
            row["api_ms"])
    return {k: _median(v) for k, v in buckets.items()}


def selector_geometry(rows, label, candidate):
    """Steady-row plan geometry for (label, candidate): flow, PU, params."""
    for _p, _ln, row in rows:
        if (row.get("label") == label
                and row.get("candidate") == candidate
                and row.get("tier") == "steady"
                and row.get("status") == "ok"):
            return {
                "executed": row.get("executed"),
                "flow_id": row.get("flow_id"),
                "flow_realization": row.get("flow_realization"),
                "processing_unit": row.get("processing_unit"),
                "block": row.get("block"),
                "params": row.get("params") or {},
            }
    return {}


def classify(rec):
    """Disclosed priority rules -> exactly one hypothesis label.

    ``rec`` keys: regret, steady_ms, dispatch_delta_ms,
    transfer_delta_ms, semantic, flow_id, has_real_flow.

    Every returned label is a hypothesis: the dominance rules compare
    measured tier medians against thresholds while the quantities that
    would prove dominance (launches, phase splits, waves) are
    disclosed as unavailable.
    """
    if rec.get("regret") is not None and rec["regret"] > REGRET_MAX:
        return CLASS_SELECTOR
    steady = rec.get("steady_ms") or 0.0
    if steady > 0 and (rec.get("dispatch_delta_ms") or 0.0) >= \
            DISPATCH_FRAC * steady:
        return CLASS_LAUNCH
    if steady > 0 and (rec.get("transfer_delta_ms") or 0.0) >= \
            TRANSFER_FRAC * steady:
        return CLASS_GM
    if rec.get("semantic") in ("r2c", "c2r") and rec.get("has_real_flow"):
        return CLASS_REAL
    flow = rec.get("flow_id")
    if flow == "local-fused-tile":
        return CLASS_CORE
    if flow in WORKER_FLOWS:
        return CLASS_TAIL
    return "unclassified-need-counters"


def analytic_gm_bytes(n, batch):
    """Disclosed analytic estimate of one forward c2c pass (fp32)."""
    return n * batch * 8 * 3  # input + twiddle-ish + output, rough


def attribute(run_dir):
    summary, rows = load_run(run_dir)
    out = []
    for case in summary["cases"]:
        label = case["label"]
        sem = case["semantic"]
        tms = tier_medians(rows, label)
        geo = selector_geometry(rows, label, "selector")
        inc_geo = selector_geometry(rows, label, "incumbent")

        # overhead-tier rows carry the tier inside the candidate name
        # ("selector:one_shot", ...) -- see run_case in the runner
        steady = tms.get(("selector", "steady"))
        one_shot = tms.get(("selector:one_shot", "one_shot"))
        e2e = tms.get(("selector:e2e", "e2e"))
        plan_c = tms.get(("selector:plan_creation", "plan_creation"))
        inc_steady = tms.get(("incumbent", "steady"))
        inc_one = tms.get(("incumbent:one_shot", "one_shot"))
        inc_e2e = tms.get(("incumbent:e2e", "e2e"))

        dispatch = (one_shot - steady) if (one_shot and steady) else None
        transfer = (e2e - steady) if (e2e and steady) else None

        params = geo.get("params") or {}
        has_real = "real_flow" in params or "real_compat" in params
        rec = {
            "label": label,
            "semantic": sem,
            "N": case["N"],
            "batch": case["batch"],
            "selector_source": case["selector_source"],
            "oracle_candidate": case["oracle_candidate"],
            "regret": case["selector_regret"],
            # fields classify() consumes directly
            "flow_id": geo.get("flow_id"),
            "has_real_flow": has_real,
            "vendor_status": case["vendor_status"],
            "steady_ms": steady,
            "incumbent_steady_ms": inc_steady,
            "plan_creation_ms": plan_c,
            "one_shot_ms": one_shot,
            "e2e_ms": e2e,
            "incumbent_one_shot_ms": inc_one,
            "incumbent_e2e_ms": inc_e2e,
            "dispatch_delta_ms": dispatch,
            "transfer_delta_ms": transfer,
            "dispatch_frac_of_steady": (
                dispatch / steady if (dispatch is not None and steady)
                else None),
            "transfer_frac_of_steady": (
                transfer / steady if (transfer is not None and steady)
                else None),
            "geometry": {k: geo.get(k) for k in (
                "executed", "flow_id", "flow_realization",
                "processing_unit", "block")},
            "plan_geometry_analytic": {
                "rows": params.get("rows"),
                "tile": params.get("tile"),
                "variant": params.get("variant"),
                "analytic_gm_bytes": analytic_gm_bytes(
                    case["N"], case["batch"]),
                "gm_bytes_kind": "analytic estimate; not measured",
            },
            "incumbent_geometry": {k: inc_geo.get(k) for k in (
                "flow_id", "processing_unit")},
        }
        cls = classify(rec)
        rec["hypothesis"] = cls
        # regret vs the bounded oracle is a directly measured deficit;
        # every dominance rule over unavailable counters is unverified
        rec["confidence"] = (
            CONF_MEASURED if cls == CLASS_SELECTOR else CONF_UNVERIFIED)
        rec["evidence"] = {
            "regret_vs_oracle": rec["regret"],
            "dispatch_delta_ms": dispatch,
            "transfer_delta_ms": transfer,
            "one_shot_includes": "per-call plan/dispatch overhead "
                                 "(runner note)",
            "e2e_scope": "H2D + execute + D2H combined; per-hop split "
                         "unavailable",
        }
        out.append(rec)

    report = {
        "schema": SCHEMA,
        "run": str(run_dir),
        "protocol_id": summary.get("protocol_id"),
        "evidence_status": EVIDENCE_STATUS,
        "hypotheses_used": sorted({r["hypothesis"] for r in out}),
        "confidence_levels": sorted({r["confidence"] for r in out}),
        "unavailable_quantities": UNAVAILABLE,
        "rule_thresholds": {
            "regret_max": REGRET_MAX,
            "dispatch_frac": DISPATCH_FRAC,
            "transfer_frac": TRANSFER_FRAC,
            "priority": ["selector", "launch", "GM", "real-adapter",
                         "core (local-fused-tile)",
                         "worker-tail (global/spread/folded)",
                         "unclassified-need-counters"],
        },
        "anchors": out,
    }
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True,
                    help="run_issue9 profile phase output directory")
    ap.add_argument("--out", default=None,
                    help="report path (default: <run>/attribution.json)")
    args = ap.parse_args(argv)

    run_dir = Path(args.run)
    if not (run_dir / "summary.json").is_file():
        raise SystemExit(f"{run_dir}: summary.json missing")
    report = attribute(run_dir)
    out = Path(args.out) if args.out else run_dir / "attribution.json"
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"attribution report: {out}")
    print("note: hypothesis labels are rule-derived over disclosed-"
          "unavailable counters (confidence=unverified except measured "
          "selector regret); not direct causal evidence")
    print(f"{'label':20} {'hypothesis':28} {'conf':>12} {'steady':>8} "
          f"{'dispatch':>9} {'transfer':>9} {'flow':>20}")
    for r in report["anchors"]:
        print(f"{r['label']:20} {r['hypothesis']:28} "
              f"{r['confidence']:>12} "
              f"{(r['steady_ms'] or 0):8.4f} "
              f"{(r['dispatch_delta_ms'] or 0):9.4f} "
              f"{(r['transfer_delta_ms'] or 0):9.4f} "
              f"{str(r['geometry'].get('flow_id')):>20}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
