#!/usr/bin/env python3
"""Issue #8 PR-G step 1: re-audit every v2 matrix cell from its own
summary.

For each of the 144 frozen cells report deployed selector, incumbent,
bounded oracle, vendor and the resulting regret, classified against the
fixed domains of issue #8 (main / stress / holdout / regression /
background).  Flags the acceptance-relevant points:

- main-domain cells with regret > 10% (gate: <= 10% where legal),
- stress cells (N1024B256) with known high regret (gate: must fix),
- abstain cells whose measured oracle is clearly faster (PR-G step 2
  independent-confirmation targets),
- guard-positive production points per non-C2C semantic (gate: >= 1 each).

Evidence class of every number: summary-derived (frozen v2 release
medians).  No new measurement is performed here.

Usage:
    python3 scripts/audit_v2_regret.py [--release DIR] [--out DIR]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = (ROOT / "results" / "p800" / "releases" /
                   "20261017-wp2-semantic-matrix-v2")

MAIN_N, MAIN_B = (1024, 2048), (16, 64)
STRESS_CELLS = [(1024, 256)]
HOLDOUT_CELLS = [(512, 64), (2048, 4)]
REGRESSION_CELLS = [(64, 1), (4096, 1), (4096, 16), (16384, 1),
                    (16384, 16), (65536, 1), (65536, 16)]
SEMANTICS = ("c2c-forward", "c2c-inverse", "r2c", "c2r")
# selector_regret == selector_ms / oracle_ms - 1 (frozen v2 convention):
# the issue's "<= 10%" gate therefore means regret <= 0.10, and a stress
# cell at "~2.0-2.2x" carries regret ~1.0-1.2.
REGRET_MAX = 0.10
# "clearly faster" for an abstained cell: measured oracle >= 1.10x the
# incumbent selector fallback (the same 10% line as the regret gate),
# restricted to the fixed domains of issue #8.
ABSTAIN_CONFIRM_MIN = 0.10
FIXED_DOMAINS = ("main", "stress", "holdout", "regression")


def oracle_speedup_regret(regret):
    """incumbent/oracle speedup when the selector abstained (ratio)."""
    return None if regret is None else round(1.0 + regret, 3)


def domain_of(n, b):
    if (n, b) in STRESS_CELLS:
        return "stress"
    if (n, b) in HOLDOUT_CELLS:
        return "holdout"
    if n in MAIN_N and b in MAIN_B:
        return "main"
    if (n, b) in REGRESSION_CELLS:
        return "regression"
    return "background"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir (default: <release>/..  "
                         "diagnostics/<date>)")
    args = ap.parse_args()
    release = args.release if args.release.is_absolute() else \
        (Path.cwd() / args.release)

    summary = json.loads((release / "summary.json").read_text())
    out_dir = args.out or (ROOT / "results" / "p800" / "diagnostics" /
                           "2026-10-17")
    out_dir.mkdir(parents=True, exist_ok=True)

    # De-duplicate: focus + matrix passes record identical rows for the
    # same label; keep one per (N, batch, semantic).
    cells = {}
    for c in summary["cases"]:
        key = (c["N"], c["batch"], c["semantic"])
        prev = cells.get(key)
        if prev is None:
            cells[key] = c
        elif prev != c:
            raise SystemExit(f"conflicting duplicate for {key}: "
                             f"{prev} != {c}")

    audit = []
    for (n, b, sem), c in sorted(cells.items()):
        audit.append({
            "label": c["label"], "N": n, "batch": b, "semantic": sem,
            "domain": domain_of(n, b),
            "selector_ms": c["selector_ms"],
            "selector_source": c.get("selector_source"),
            "incumbent_ms": c["incumbent_ms"],
            "vendor_ms": c["vendor_ms"],
            "vendor_status": c.get("vendor_status"),
            "oracle_ms": c["oracle_ms"],
            "oracle_candidate": c.get("oracle_candidate"),
            "selector_regret": c.get("selector_regret"),
            "guard_vs_incumbent": c.get("guard_vs_incumbent"),
            "oracle_vs_incumbent_speedup": (
                round(c["incumbent_ms"] / c["oracle_ms"], 3)
                if c.get("oracle_ms") else None),
        })

    def rows(pred):
        return [r for r in audit if pred(r)]

    main_over = rows(lambda r: r["domain"] == "main"
                     and (r["selector_regret"] or 0) > REGRET_MAX)
    stress_over = rows(lambda r: r["domain"] == "stress"
                       and (r["selector_regret"] or 0) > REGRET_MAX)
    confirm_targets = rows(
        lambda r: r["domain"] in FIXED_DOMAINS
        and r["selector_source"] == "abstain"
        and (r["selector_regret"] or 0) >= ABSTAIN_CONFIRM_MIN)
    positives = {}
    for sem in ("c2c-forward", "c2c-inverse", "r2c", "c2r"):
        positives[sem] = [
            {"label": r["label"], "guard": r["guard_vs_incumbent"]}
            for r in rows(lambda r: r["semantic"] == sem
                          and r["selector_source"] in ("table", "composition")
                          and (r["guard_vs_incumbent"] or 0) >= 0.02)
        ]

    result = {
        "evidence_class": "summary-derived",
        "source": (str(release.relative_to(ROOT))
                   if release.is_relative_to(ROOT) else str(release)),
        "protocol_id": summary.get("protocol_id"),
        "cells_total": len(audit),
        "note": (
            "frozen v2 summary holds 144 case entries but 128 unique "
            "cells: the 16 main-domain cells are listed twice "
            "(main+apps scope bookkeeping, byte-identical rows); "
            "regret = selector_ms/oracle_ms - 1"),
        "gate_main_regret_le_10pct": {
            "pass": not main_over,
            "regret_max": REGRET_MAX,
            "violations": [r["label"] for r in main_over],
        },
        "gate_stress_n1024b256_must_fix": {
            "pass": not stress_over,
            "violations": [
                {"label": r["label"], "regret": r["selector_regret"],
                 "oracle_vs_incumbent_speedup":
                     r["oracle_vs_incumbent_speedup"],
                 "selector_source": r["selector_source"]}
                for r in stress_over],
        },
        "step2_independent_confirmation_targets": [
            {"label": r["label"], "regret": r["selector_regret"],
             "oracle_candidate": r["oracle_candidate"],
             "oracle_vs_incumbent_speedup":
                 r["oracle_vs_incumbent_speedup"]}
            for r in confirm_targets],
        "guard_positive_production_points": positives,
        "cells": audit,
    }

    (out_dir / "pr-g-regret-audit.json").write_text(
        json.dumps(result, indent=1, sort_keys=True) + "\n")

    lines = [
        "# PR-G step 1: v2 regret audit",
        "",
        f"- evidence: summary-derived from `{result['source']}`",
        f"- protocol: {result['protocol_id']}",
        f"- cells: {len(audit)} (deduplicated)",
        "",
        "## Gates on the frozen v2 baseline",
        "",
        f"- main-domain regret <= 10%: "
        f"{'PASS' if result['gate_main_regret_le_10pct']['pass'] else 'FAIL'} "
        f"{result['gate_main_regret_le_10pct']['violations'] or ''}",
        f"- stress N1024B256 (must fix): "
        f"{'PASS' if result['gate_stress_n1024b256_must_fix']['pass'] else 'OPEN'}",
    ]
    for v in result["gate_stress_n1024b256_must_fix"]["violations"]:
        lines.append(f"  - {v['label']}: regret {v['regret']:.2f} "
                     f"({v['oracle_vs_incumbent_speedup']}x oracle vs "
                     f"incumbent, source={v['selector_source']})")
    lines += ["", "## PR-G step 2: fixed-domain abstain points whose "
              "measured oracle is clearly faster "
              "(regret >= 0.10, i.e. oracle >= 1.10x the incumbent "
              "fallback)", ""]
    for t in result["step2_independent_confirmation_targets"]:
        lines.append(f"- {t['label']}: regret {t['regret']:.2f}, "
                     f"oracle {t['oracle_candidate']}, "
                     f"{t['oracle_vs_incumbent_speedup']}x vs incumbent")
    lines += ["", "## Guard-positive production points (>=2%) per semantic",
              ""]
    for sem, pts in positives.items():
        shown = ", ".join(f"{p['label']} +{p['guard']:.3f}"
                          for p in pts[:4]) or "(none in v2)"
        lines.append(f"- {sem}: {len(pts)} point(s): {shown}")
    lines.append("")
    (out_dir / "pr-g-regret-audit.md").write_text("\n".join(lines))

    print(f"wrote {out_dir}/pr-g-regret-audit.json "
          f"and pr-g-regret-audit.md "
          f"({len(audit)} cells, "
          f"{len(confirm_targets)} confirmation targets)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
