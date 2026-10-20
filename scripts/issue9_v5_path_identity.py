#!/usr/bin/env python3
"""Issue #9 release-correction P1-2: v5 production -> current path identity.

Resolves every semantic cell of the frozen validate domain under BOTH
the v5 production selector table (the frozen-base-commit blob, the
table whose behaviour Issue #9 was supposed to close against) and the
current working-tree table, through the CURRENT contracts
(plan_fft/select), and classifies each cell:

* ``identical``            -- resolved runner/flow/block/unit/params all
                              match; the cell inherits v5 behaviour by
                              construction and needs no re-measurement;
* ``changed``              -- resolved geometry differs; the cell needs
                              a paired v5-deployed vs current
                              confirmation run;
* ``fallback-transition``  -- v5 fell back and the current table serves
                              an exact entry (or vice versa).

Only resolved-plan identity matters: table provenance keys and
``selection_source`` are excluded (they never reach execution).

The frozen v5 blob predates the WP0-A fallback carry contract, so it
cannot load under the current contracts as-is.  The script stamps each
carried fallback's provenance with the WP0-A binding keys sourced from
the v5 table's OWN env/profile/precision/device identity (a
self-binding normalization: entry geometry, params and selection
behaviour are untouched) and records both the original blob sha and
the bound-file sha.  The bound file is what a paired
``--previous-table-src`` run measures against.

Usage:
    scripts/issue9_v5_path_identity.py [--out DIR]
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CONFIG_PATH = ROOT / "config" / "issue9_p800.json"
SCHEMA = "issue9-v5-path-identity-v1"
# keys that never reach the executed plan
_NON_EXEC_KEYS = ("selection_source", "fallback")
_DEVICE_IDENTITY_KEYS = ("device_model", "multiprocessor_count",
                         "max_shared_mem")
_BINDING_KEYS = ("carry_contract", "source_env_hash",
                 "source_profile_hash", "source_precision",
                 "source_device", "source_result_sha256")


def bind_v5_fallback_provenance(table: dict, blob_sha: str = None) -> dict:
    """Stamp WP0-A binding keys onto a pre-WP0-A table's fallbacks.

    Self-binding only: every value comes from the table's own identity
    (env/profile/precision/device) or the blob sha of the frozen file
    itself (no result_sha256 in the pre-WP0-A schema).  Entry geometry,
    params and selection semantics are untouched.
    """
    from src.xpu.selector import FALLBACK_CARRY_CONTRACT

    ident = {k: (table.get("env") or {}).get(k)
             for k in _DEVICE_IDENTITY_KEYS}
    out = json.loads(json.dumps(table))
    for e in out.get("previous_fallbacks") or []:
        prov = e.setdefault("previous_fallback_provenance", {})
        defaults = {
            "carry_contract": FALLBACK_CARRY_CONTRACT,
            "source_env_hash": table.get("env_hash"),
            "source_profile_hash": table.get("profile_hash"),
            "source_precision": table.get("precision"),
            "source_device": ident,
            "source_result_sha256": (
                prov.get("source_result_sha256")
                or table.get("result_sha256")
                or blob_sha),
        }
        for k in _BINDING_KEYS:
            if prov.get(k) in (None, ""):
                prov[k] = defaults[k]
    return out


def _resolve(table, n, b, sem, prof):
    """Resolve one semantic cell to its executed-plan identity."""
    from src.xpu.entry import plan_fft
    import run_semantic_matrix as sm

    tt, tdir = sm._semantic_of(sem)
    norm = sm._norm_of(sem)
    plan = plan_fft(direction="dit", n=n, batch=b,
                    precision=table["precision"],
                    profile_hash=prof, table=table,
                    env_hash=table["env_hash"], transform_type=tt,
                    transform_direction=tdir, norm=norm)
    params = {k: v for k, v in (plan.params or {}).items()
              if k not in _NON_EXEC_KEYS}
    return {
        "executed": plan.executed,
        "flow_id": plan.flow_id,
        "flow_realization": plan.flow_realization,
        "processing_unit": plan.processing_unit,
        "block": plan.block,
        "fallback": plan.fallback,
        "params": params,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None,
                    help="report directory (default: "
                    "results/p800/issue9_v61)")
    args = ap.parse_args(argv)

    import run_issue9 as ri
    import run_semantic_matrix as sm
    from src.xpu.config import DeviceProfile
    from src.xpu.selector import load_table

    cfg = ri.load_config(CONFIG_PATH)
    frozen = cfg["source"]["frozen_base_commit"]
    out_dir = Path(args.out) if args.out else (
        ROOT / "results" / "p800" / "issue9_v61")
    out_dir.mkdir(parents=True, exist_ok=True)

    v5_bytes = subprocess.check_output(
        ["git", "show", f"{frozen}:config/selector_table.json"],
        cwd=ROOT)
    v5_path = out_dir / "v5_production_selector_table.json"
    v5_path.write_bytes(v5_bytes)
    v5_sha = hashlib.sha256(v5_bytes).hexdigest()
    v5_raw = json.loads(v5_bytes)
    v5_bound = bind_v5_fallback_provenance(v5_raw, blob_sha=v5_sha)
    v5_bound_path = out_dir / "v5_production_selector_table_bound.json"
    v5_bound_path.write_text(
        json.dumps(v5_bound, indent=2, sort_keys=True) + "\n")
    v5_bound_sha = hashlib.sha256(
        v5_bound_path.read_bytes()).hexdigest()
    cur_path = ROOT / "config" / "selector_table.json"
    cur_sha = hashlib.sha256(cur_path.read_bytes()).hexdigest()

    v5_table = load_table(str(v5_bound_path))
    cur_table = load_table(str(cur_path))
    prof = DeviceProfile.load().hash()

    cells = sm.parse_cells_spec(ri.phase_cells_spec(cfg, "validate"))
    identical, changed, transitions = [], [], []
    for n, b, sems in cells:
        for sem in sems:
            label = f"N{n}B{b}:{sem}"
            v5 = _resolve(v5_table, n, b, sem, prof)
            cur = _resolve(cur_table, n, b, sem, prof)
            rec = {"label": label, "N": n, "batch": b, "semantic": sem,
                   "v5": v5, "current": cur}
            if v5["fallback"] != cur["fallback"]:
                transitions.append(rec)
                changed.append(rec)
            elif v5 == cur:
                identical.append(label)
            else:
                changed.append(rec)

    ident_blob = json.dumps(sorted(identical), separators=(",", ":"))
    report = {
        "schema": SCHEMA,
        "frozen_base_commit": frozen,
        "v5_table_sha256": v5_sha,
        "v5_table_bound_sha256": v5_bound_sha,
        "v5_bound_normalization": (
            "WP0-A carry-contract binding keys stamped onto the "
            "pre-WP0-A fallback provenance, sourced from the v5 "
            "table's own env/profile/precision/device identity; "
            "entry geometry, params and selection behaviour untouched"),
        "current_table_sha256": cur_sha,
        "profile_hash": prof,
        "validate_semantic_cells": sum(len(s) for _, _, s in cells),
        "counts": {
            "identical": len(identical),
            "changed": len(changed),
            "fallback_transitions": len(transitions),
        },
        "identical_labels_sha256": hashlib.sha256(
            ident_blob.encode()).hexdigest(),
        "identical": sorted(identical),
        "changed": changed,
        "interpretation": (
            "identical cells inherit v5 production behaviour by "
            "construction (same resolved runner/geometry under the "
            "current contracts); only changed/fallback-transition "
            "cells require paired v5-deployed confirmation"),
    }
    out_json = out_dir / "v5_path_identity.json"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"v5 path identity: {out_json}")
    print(f"  frozen base: {frozen[:12]}  v5 sha: {v5_sha[:12]}  "
          f"bound: {v5_bound_sha[:12]}  current sha: {cur_sha[:12]}")
    print(f"  identical: {len(identical)}  changed: {len(changed)}  "
          f"fallback-transitions: {len(transitions)}")
    for rec in changed:
        print(f"    {rec['label']:28s} v5={rec['v5']['executed']}"
              f"/fb={rec['v5']['fallback']}  "
              f"cur={rec['current']['executed']}"
              f"/fb={rec['current']['fallback']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
