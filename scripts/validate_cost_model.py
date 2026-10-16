#!/usr/bin/env python3
"""Evaluate the *constant-declared* cost model against measured runs.

Reads one or more run_batch_release ``summary.json`` artifacts (the focus
rounds and/or the full-space holdout round), turns the measured rows into
model records via
:meth:`src.xpu.cost_model.records_from_summary` (the same
completeness/finiteness/correctness gates the selector builder applies),
and runs :meth:`src.xpu.cost_model.validate`:

* per-family MDAPE over all scored records (descriptive scope),
* promotion-decision agreement at the formal ``GUARD_PCT`` (not
  sign-of-gain), **per focus round** (never pooled with holdout),
* legacy top-K hit and ``topk_full_space`` recall / selection regret over
  the published nine (tile, rows) points — computed **only from
  ``full_space=true`` holdout records**, m2 points deduplicated by
  structural parameters.

The result is written to ``results/p800/model_validation.json`` and
embedded in the regenerated batch-focus report. Nothing here tunes the
model: efficiency constants are declared in ``src/xpu/cost_model.py`` and
hardware anchors come from ``config/calibration.json``; this script only
measures.

Usage::

    python scripts/validate_cost_model.py \\
        results/p800/batch_focus_a/summary.json \\
        results/p800/batch_focus_b/summary.json \\
        results/p800/batch_holdout/summary.json \\
        [--out results/p800/model_validation.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xpu.cost_model import (  # noqa: E402
    GUARD_PCT,
    records_from_summary,
    validate,
)


def _load_summary(path: Path) -> dict:
    with open(path) as f:
        summary = json.load(f)
    if not isinstance(summary, dict) or "shapes" not in summary:
        raise SystemExit(f"{path}: not a run_batch_release summary.json")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("summaries", nargs="+", type=Path,
                    help="summary.json files (focus / holdout rounds; "
                         "classified per file by manifest.full_space)")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "p000" / "model_validation.json",
                    help="where to write the metrics JSON")
    args = ap.parse_args()

    sources = []
    records = []
    for path in args.summaries:
        summary = _load_summary(path)
        man = summary.get("manifest") or {}
        recs = records_from_summary(summary, source_id=str(path))
        sources.append({
            "path": str(path),
            "protocol_id": man.get("protocol_id"),
            "full_space": bool(man.get("full_space")),
            "commit": man.get("commit"),
            "dirty": man.get("dirty"),
            "env_hash": man.get("env_hash"),
            "records": len(recs),
            "shapes": len(summary.get("shapes") or []),
        })
        records.extend(recs)

    if not records:
        raise SystemExit("no usable measured records in the given summaries "
                         "(all rows skipped as non-finite / failing / "
                         "incomplete)")

    metrics = validate(records)
    if not metrics.get("n_records"):
        raise SystemExit("validate() found no records it could score "
                         "(model families did not predict these shapes)")

    full = metrics.get("topk_full_space") or {}
    result = {
        "kind": "issue-3-model-validation",
        "guard_pct": GUARD_PCT,
        "n_records": metrics["n_records"],
        "sources": sources,
        "metrics": metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    print(f"records: {metrics['n_records']} from {len(sources)} summary file(s) "
          f"({metrics['n_focus_records']} focus / "
          f"{metrics['n_holdout_records']} holdout)")
    for sid, g in sorted((metrics.get("guard_by_source") or {}).items()):
        print(f"guard={GUARD_PCT:.1f}% promotion-decision agreement "
              f"[{sid}]: {g['agreement']} over {g['pairs']} pairs")
    print(f"top-K hit (holdout only): {metrics['topk_hit']} "
          f"over {metrics['topk_shapes']} shapes")
    print(f"full-space holdout: recall={full.get('recall')} "
          f"shapes={full.get('shapes')} "
          f"mean_regret={full.get('mean_regret_pct')}% "
          f"(space={full.get('space_size')} points)")
    print(f"MDAPE %: {json.dumps(metrics['mdape_pct'], sort_keys=True)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
