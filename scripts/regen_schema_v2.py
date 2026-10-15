#!/usr/bin/env python3
"""Normalize committed bench JSONs to schema v2 (offline, torch-free).

The committed phase-2/3 artifacts were captured under the v1 attribution
vocabulary (``processing_unit`` as ``radix-2``/``radix-4``, a single
``residence`` field, no ``flow_realization``/``local_residence``). Schema v2
of ``src/xpu/candidates.py`` splits those axes; the **measured rows are never
touched** — only derived sections are regenerated:

* manifest labels: ``residence`` -> ``boundary_residence``, processing-unit
  vocabulary mapped to ``radix2-scalar``/``radix4-scalar``;
* candidates: same label mapping plus derived ``flow_realization`` and
  ``local_residence`` (mirroring ``candidates.effective_*``) and the v2 knob
  defaults (``rows``/``glog``/``maxp``/``cells``);
* ``rows_long``: regenerated through ``report.long_rows`` so the section-9
  rows use the v2 fields (and the corrected analytic GM-bytes ledger);
* top-level ``schema_version: 2``.

Companion ``*_long.csv`` files are rewritten from the new ``rows_long`` when
they already exist. Markdown is NOT touched (``report.write_markdown`` does
not read the attribution fields; the golden byte-identity tests pin that).

Usage::

    python scripts/regen_schema_v2.py [results/p800/bench_phase3.json ...]
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from report import long_rows, write_long_csv  # noqa: E402

_UNIT_MAP = {
    "radix-2": "radix2-scalar",
    "radix-4": "radix4-scalar",
}
_DERIVED_LOCAL_RESIDENCE = {
    "global-stage": "none",
    "fused-inplace": "cache",
    "matrix-resident": "register+implicit-dot",
    "grouped-boundary": "register",
}


def _map_unit(unit: str) -> str:
    return _UNIT_MAP.get(unit, unit)


def _derive_attribution(c: dict) -> None:
    """Add v2 flow_realization/local_residence to a candidate dict in place."""
    flow = c["flow_id"]
    realization = c.get("flow_realization") or ""
    if not realization:
        if flow in ("baseline-global-stage", "batch-spread", "batch-folded"):
            realization = "global-stage"
        elif c.get("glog", 0) >= 1:
            realization = "grouped-boundary"
        elif c.get("rows", 1) > 1:
            realization = "matrix-resident"
        else:
            realization = "fused-inplace"
    c["flow_realization"] = realization
    if not c.get("local_residence"):
        c["local_residence"] = _DERIVED_LOCAL_RESIDENCE[realization]
    c.setdefault("rows", 1)
    c.setdefault("glog", 0)
    c.setdefault("maxp", 0)
    c.setdefault("cells", 1)


def _normalize_candidate_dict(c: dict) -> None:
    if "residence" in c:
        c["boundary_residence"] = c.pop("residence")
    c.setdefault("boundary_residence", "gm")
    c["processing_unit"] = _map_unit(c.get("processing_unit", "radix2-scalar"))
    _derive_attribution(c)


def normalize_payload(data: dict) -> dict:
    """Return a schema-v2 copy of a bench payload (measured rows untouched)."""
    man = data["manifest"]
    if "residence" in man:
        man["boundary_residence"] = man.pop("residence")
    man.setdefault("boundary_residence", "gm")
    man["processing_unit"] = _map_unit(
        man.get("processing_unit", "radix2-scalar"))

    cands = data.get("candidates") or {}
    for verdicts in (cands.get("legal"), cands.get("rejected")):
        for v in verdicts or []:
            _normalize_candidate_dict(v["candidate"])
    sel = cands.get("selected")
    if sel:
        _normalize_candidate_dict(sel["candidate"])

    if "rows" in data:
        data["rows_long"] = long_rows(data["rows"], man)
    data["schema_version"] = 2
    return data


def main(argv) -> None:
    paths = argv or [
        "results/p800/bench_phase2.json",
        "results/p800/bench_phase3.json",
        "results/p800/bench_block256.json",
        "results/p800/bench_block512.json",
        "results/p800/bench_block2048.json",
    ]
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        if "manifest" not in data:
            print(f"skip {path}: no manifest (different artifact schema)")
            continue
        normalize_payload(data)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        n_long = len(data.get("rows_long") or [])
        csv_path = path[:-len(".json")] + "_long.csv"
        wrote_csv = False
        if n_long and os.path.exists(csv_path):
            write_long_csv(csv_path, data["rows_long"])
            wrote_csv = True
        print(f"normalized {path}: schema_version=2 rows_long={n_long} "
              f"long_csv={'rewritten' if wrote_csv else 'n/a'}")


if __name__ == "__main__":
    main(sys.argv[1:])
