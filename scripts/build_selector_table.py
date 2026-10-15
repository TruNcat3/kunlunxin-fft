#!/usr/bin/env python3
"""Build config/selector_table.json from confirmation-round bench JSONs.

Reads the measured confirmation runs (block sweeps + batch probe), applies
the plan's promotion guard (challenger must beat the M0 incumbent by >=2%
paired-median improvement; otherwise the incumbent stays), and emits the
static per-size selector table. M0 is always in the closure: an entry whose
winner fails the guard records the M0 fallback explicitly.

Usage::

    python scripts/build_selector_table.py \
        results/p800/confirm_b1024.json results/p800/confirm_b256.json \
        --batch results/p800/confirm_batch.json \
        --out config/selector_table.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.xpu.config import ConfigError, DeviceProfile  # noqa: E402
from src.xpu.selector import GUARD_PCT  # noqa: E402

# candidate key -> (flow_id, flow_realization, processing_unit, params-beyond-block)
_UNIT = {
    "m0": ("baseline-global-stage", "global-stage", "radix2-scalar", {}),
    "m5": ("baseline-global-stage", "global-stage", "radix4-scalar", {"radix": 4}),
}


def _descriptor(key: str, block: int, extra: dict) -> dict:
    """Map a bench candidate key onto its schema-v2 attribution + params."""
    if key.startswith("m0_"):
        flow, real, unit, params = _UNIT["m0"]
    elif key.startswith("m1_t"):
        flow, real, unit = "local-fused-tile", "fused-inplace", "radix2-scalar"
        params = {"tile": int(key.split("_t")[-1])}
    elif key.startswith("m2_"):
        flow, real, unit = "local-fused-tile", "matrix-resident", "dense-dot-fft64"
        params = {"tile": int(key.split("_t")[-1]), "rows": int(extra["matrix_rows"])}
    elif key.startswith("m3_g_"):
        flow, real, unit = ("local-fused-tile", "grouped-boundary",
                            "composite-linear-map")
        params = {"tile": int(key.split("_t")[-1]), "glog": int(extra["glog"]),
                  "maxp": int(extra["maxp"]), "cells": int(extra["cells"])}
    elif key.startswith("m5_r4_"):
        flow, real, unit, params = _UNIT["m5"]
    elif key.startswith("m4_"):
        parts = key.split("_")           # m4, variant, bNN, dit|dif
        variant, bkey = parts[1], parts[2]
        flow = f"batch-{variant}"
        real, unit = "global-stage", "radix2-scalar"
        params = {"variant": variant, "batch": int(bkey[1:])}
    else:
        raise ConfigError(f"unmapped candidate key: {key}")
    return {"block": block, "flow_id": flow, "flow_realization": real,
            "processing_unit": unit, "params": params}


def _speedup(row, base_key: str, cand_key: str):
    m0, cand = row.get(f"{base_key}_ms"), row.get(f"{cand_key}_ms")
    if not m0 or not cand:
        return None
    return m0 / cand


def build_b1_entry(rows_by_n, direction, n, extra_by_file, guard=GUARD_PCT):
    """One (N, direction) B=1 entry: best challenger vs best-M0 incumbent."""
    d = direction
    candidates = []   # (ms, cand_key, block, file)
    incumbents = []
    for path, rows in rows_by_n.items():
        row = next((r for r in rows if r["n"] == n), None)
        if row is None:
            continue
        extra = extra_by_file[path]
        block = int(extra["block"])
        m0_key = f"m0_{d}_ms"
        if row.get(m0_key):
            incumbents.append((row[m0_key], f"m0_{d}", block, path))
        for key in (f"m1_t64", f"m2_{d}_t64", f"m3_g_{d}_t8", f"m5_r4_{d}"):
            if row.get(f"{key}_ms"):
                candidates.append((row[f"{key}_ms"], key, block, path))
    if not incumbents or not candidates:
        return None
    inc_ms, inc_key, inc_block, inc_path = min(incumbents)
    ch_ms, ch_key, ch_block, ch_path = min(candidates)
    gain_pct = (inc_ms - ch_ms) / inc_ms * 100.0
    if gain_pct >= guard:
        desc = _descriptor(ch_key, ch_block, extra_by_file[ch_path])
        reason = (f"challenger {ch_key}@block{ch_block} beats M0 incumbent "
                  f"{inc_key}@block{inc_block} by {gain_pct:.1f}% (>= {guard}%)")
    else:
        desc = _descriptor(inc_key, inc_block, extra_by_file[inc_path])
        reason = (f"no challenger clears the {guard}% guard "
                  f"(best {ch_key} gain {gain_pct:.1f}%); M0 incumbent kept")
    return {"N": n, "batch": 1, "direction": d,
            "incumbent_ms": inc_ms, "challenger_ms": ch_ms,
            "evidence": {"incumbent": f"{inc_key}@block{inc_block}",
                         "challenger": f"{ch_key}@block{ch_block}",
                         "gain_pct": round(gain_pct, 2)},
            "reason": reason, **desc}


def build_batch_entries(path, guard=GUARD_PCT):
    """Per (N, B, direction): pick the faster M4 variant (folded default)."""
    with open(path) as f:
        data = json.load(f)
    extra = data["manifest"]["extra"]
    block = int(data["manifest"]["block"])
    entries = []
    for row in data["rows"]:
        n = row["n"]
        for b in extra["batchs"]:
            if b == 1:
                # B=1 selection is owned by the B=1 confirmation runs; the
                # batch probe's B=1 rows only sanity-check the M4 path.
                continue
            for d in ("dit", "dif"):
                ms = {v: row.get(f"m4_{v}_b{b}_{d}_ms") for v in ("spread", "folded")}
                if not any(ms.values()):
                    continue
                if ms["spread"] and ms["folded"]:
                    gain = (ms["folded"] - ms["spread"]) / ms["folded"] * 100.0
                    if gain >= guard:
                        variant, why = "spread", (
                            f"spread beats folded by {gain:.1f}% (>= {guard}%)")
                    else:
                        variant, why = "folded", (
                            f"folded kept (spread gain {gain:.1f}% < {guard}%)")
                else:
                    variant = "spread" if ms["spread"] else "folded"
                    why = f"only {variant} measured"
                desc = _descriptor(f"m4_{variant}_b{b}_{d}", block, extra)
                entries.append({
                    "N": n, "batch": b, "direction": d,
                    "incumbent_ms": ms["folded"] if variant == "spread" else ms["spread"],
                    "challenger_ms": ms[variant],
                    "evidence": {"variant_ms": {k: v for k, v in ms.items() if v}},
                    "reason": why, **desc,
                })
    return entries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="B=1 confirmation JSONs")
    ap.add_argument("--batch", default=None, help="batch confirmation JSON")
    ap.add_argument("--out", default="config/selector_table.json")
    ap.add_argument("--guard", type=float, default=GUARD_PCT)
    args = ap.parse_args()

    profile = DeviceProfile.load()
    rows_by_n = {}
    extra_by_file = {}
    sizes = set()
    sources = []

    def _rel(path):
        try:
            return os.path.relpath(path)
        except ValueError:  # different drive (Windows); keep absolute
            return path

    for path in args.runs:
        with open(path) as f:
            data = json.load(f)
        rows_by_n[path] = data["rows"]
        extra_by_file[path] = {"block": data["manifest"]["block"],
                               **data["manifest"]["extra"]}
        sizes.update(r["n"] for r in data["rows"])
        sources.append(_rel(path))
    entries = []
    for n in sorted(sizes):
        for d in ("dit", "dif"):
            e = build_b1_entry(rows_by_n, d, n, extra_by_file, guard=args.guard)
            if e:
                entries.append(e)
    if args.batch:
        entries.extend(build_batch_entries(args.batch, guard=args.guard))
        sources.append(_rel(args.batch))

    table = {
        "schema_version": 1,
        "device": profile.device,
        "profile_hash": profile.hash(),
        "precision": "fp32",
        "guard_pct": args.guard,
        "sources": sources,
        "entries": entries,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(table, f, indent=2)
        f.write("\n")
    n_b1 = sum(1 for e in entries if e["batch"] == 1)
    n_m0 = sum(1 for e in entries if e["batch"] == 1
               and e["flow_id"] == "baseline-global-stage"
               and e["processing_unit"] == "radix2-scalar")
    print(f"wrote {args.out}: {len(entries)} entries "
          f"({n_b1} at B=1, {n_m0} M0-fallback) guard={args.guard}%")


if __name__ == "__main__":
    main()
