#!/usr/bin/env python3
"""Build config/selector_table.json from confirmation-round bench JSONs.

Safety gates (review round P0): a row may only enter the table when its
timing is a finite positive number, its recorded relative error is below
the run's correctness threshold, and every input run shares one
device/profile/precision/protocol manifest. Rejected rows and rejected runs
are logged with reasons; they never influence an entry.

Selection semantics: for each (N, direction) B=1 shape the builder compares
correctness-gated challengers against the correctness-gated M0 incumbent and
keeps the challenger only when its **search-stage median improvement**
(compare: aggregated medians measured in the same run schedule, not paired
per-trial medians) clears the promotion guard. Batch entries pick the faster
M4 variant the same way; the table itself — never a prose claim — is the
source of which variant wins where.

Usage::

    python scripts/build_selector_table.py \
        results/p800/confirm_b1024.json results/p800/confirm_b256.json \
        --batch results/p800/confirm_batch.json \
        --out config/selector_table.json
"""
from __future__ import annotations

import argparse
import json
import math
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

# M1 is DIT-only in this tree (fft_dit_fused); there is no DIF fused lowering.
_M1_DIRECTIONS = ("dit",)

_MANIFEST_KEYS = ("device", "profile_hash", "precision",
                  "correctness_threshold", "warmup", "trials", "dtype",
                  "timing_scope", "protocol_id")

_SUPPORTED_PRECISIONS = {"fp32": "complex64"}


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


def _err_field(key: str) -> str:
    """Bench row field holding the relative error for a candidate key."""
    if key.startswith("m0_dit"):
        return "dit_err"
    if key.startswith("m0_dif"):
        return "dif_err"
    return f"{key}_err"


def _finite_positive(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v) and v > 0


def _check_ms(row, key: str, threshold: float, path: str, n, rejections):
    """Return the timing when (ms, err) pass the safety gates, else None."""
    ms = row.get(f"{key}_ms")
    if not _finite_positive(ms):
        rejections.append(
            f"{path} N={n} key={key}: ms not finite positive ({ms!r})")
        return None
    err = row.get(_err_field(key))
    if not isinstance(err, (int, float)) or not math.isfinite(err):
        rejections.append(
            f"{path} N={n} key={key}: missing/non-finite error ({err!r})")
        return None
    if err >= threshold:
        rejections.append(
            f"{path} N={n} key={key}: max_rel_error {err:.3e} >= {threshold:g}")
        return None
    return ms


def _load_run(path: str, rejections) -> dict:
    """Load one confirmation run; reject it outright on manifest problems."""
    with open(path) as f:
        data = json.load(f)
    manifest = data.get("manifest") or {}
    missing = [k for k in _MANIFEST_KEYS if manifest.get(k) in (None, "")]
    if missing:
        raise ConfigError(
            f"run {path}: manifest missing required keys {missing}; "
            "refusing to build a table from an incomplete manifest")
    thr = manifest["correctness_threshold"]
    if (not isinstance(thr, (int, float)) or not math.isfinite(thr)
            or not (0 < float(thr) < 1)):
        raise ConfigError(
            f"run {path}: correctness_threshold {thr!r} must be a finite "
            "number in (0, 1); refusing to build")
    precision = manifest["precision"]
    want_dtype = _SUPPORTED_PRECISIONS.get(precision)
    if want_dtype is None:
        raise ConfigError(
            f"run {path}: precision {precision!r} is not a supported "
            f"table precision ({sorted(_SUPPORTED_PRECISIONS)}); "
            "refusing to merge mixed-precision measurements")
    if manifest["dtype"] != want_dtype:
        raise ConfigError(
            f"run {path}: dtype {manifest['dtype']!r} does not match "
            f"precision {precision!r} (expected {want_dtype}); "
            "refusing to build")
    return {
        "path": path,
        "rows": data["rows"],
        "manifest": manifest,
        "block": int(manifest["block"]),
        "extra": manifest.get("extra") or {},
    }


def _check_manifest_consistency(runs, table_profile_hash: str) -> None:
    """All runs (and the table itself) must share one device/profile/protocol."""
    ref = runs[0]
    for run in runs[1:]:
        for k in _MANIFEST_KEYS:
            if run["manifest"][k] != ref["manifest"][k]:
                raise ConfigError(
                    f"mixed-profile data: {run['path']} manifest {k}="
                    f"{run['manifest'][k]!r} != {ref['path']} "
                    f"{k}={ref['manifest'][k]!r}; refusing to build")
    if ref["manifest"]["profile_hash"] != table_profile_hash:
        raise ConfigError(
            f"run profile_hash {ref['manifest']['profile_hash']} != current "
            f"DeviceProfile hash {table_profile_hash}; re-measure under the "
            "current profile before building the table")


def build_b1_entry(rows_by_path, direction, n, run_by_path, guard=GUARD_PCT,
                   threshold=1e-4, rejections=None):
    """One (N, direction) B=1 entry: best challenger vs best-M0 incumbent."""
    d = direction
    rejections = rejections if rejections is not None else []
    candidates = []   # (ms, cand_key, block, path)
    incumbents = []
    for path, rows in rows_by_path.items():
        row = next((r for r in rows if r["n"] == n), None)
        if row is None:
            continue
        run = run_by_path[path]
        block = run["block"]
        m0_key = f"m0_{d}_ms"
        if _check_ms(row, f"m0_{d}", threshold, path, n, rejections) is not None:
            incumbents.append((row[m0_key], f"m0_{d}", block, path))
        keys = [f"m2_{d}_t64", f"m3_g_{d}_t8", f"m5_r4_{d}"]
        if d in _M1_DIRECTIONS:
            keys.insert(0, "m1_t64")
        for key in keys:
            ms = _check_ms(row, key, threshold, path, n, rejections)
            if ms is not None:
                candidates.append((ms, key, block, path))
    if not incumbents or not candidates:
        return None
    inc_ms, inc_key, inc_block, inc_path = min(incumbents)
    ch_ms, ch_key, ch_block, ch_path = min(candidates)
    gain_pct = (inc_ms - ch_ms) / inc_ms * 100.0
    if gain_pct >= guard:
        desc = _descriptor(ch_key, ch_block, run_by_path[ch_path]["extra"])
        reason = (f"challenger {ch_key}@block{ch_block} beats M0 incumbent "
                  f"{inc_key}@block{inc_block} by {gain_pct:.1f}% (>= {guard}%)")
    else:
        desc = _descriptor(inc_key, inc_block, run_by_path[inc_path]["extra"])
        reason = (f"no challenger clears the {guard}% guard "
                  f"(best {ch_key} gain {gain_pct:.1f}%); M0 incumbent kept")
    return {"N": n, "batch": 1, "direction": d,
            "incumbent_ms": inc_ms, "incumbent_block": inc_block,
            "challenger_ms": ch_ms,
            "evidence": {"incumbent": f"{inc_key}@block{inc_block}",
                         "challenger": f"{ch_key}@block{ch_block}",
                         "gain_pct": round(gain_pct, 2)},
            "reason": reason, **desc}


def build_batch_entries(run, guard=GUARD_PCT, threshold=1e-4, rejections=None):
    """Per (N, B, direction): pick the faster M4 variant (table-derived)."""
    rejections = rejections if rejections is not None else []
    extra = run["extra"]
    block = run["block"]
    entries = []
    for row in run["rows"]:
        n = row["n"]
        for b in extra.get("batchs", ()):
            if b == 1:
                # B=1 selection is owned by the B=1 confirmation runs; the
                # batch probe's B=1 rows only sanity-check the M4 path.
                continue
            for d in ("dit", "dif"):
                ms = {}
                for v in ("spread", "folded"):
                    key = f"m4_{v}_b{b}_{d}"
                    ms[v] = _check_ms(row, key, threshold, run["path"], n,
                                      rejections)
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
                    "incumbent_ms": (ms["folded"] if variant == "spread"
                                     else ms["spread"]),
                    "incumbent_block": block,
                    "challenger_ms": ms[variant],
                    "evidence": {"variant_ms": {k: v for k, v in ms.items()
                                                if v}},
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
    profile_hash = profile.hash()
    rejections = []

    def _rel(path):
        try:
            return os.path.relpath(path)
        except ValueError:  # different drive (Windows); keep absolute
            return path

    runs = [_load_run(p, rejections) for p in args.runs]
    all_runs = list(runs)
    if args.batch:
        all_runs = all_runs + [_load_run(args.batch, rejections)]
    _check_manifest_consistency(all_runs, profile_hash)
    threshold = float(runs[0]["manifest"]["correctness_threshold"])

    rows_by_path = {r["path"]: r["rows"] for r in runs}
    run_by_path = {r["path"]: r for r in runs}
    sizes = set()
    for r in runs:
        sizes.update(row["n"] for row in r["rows"])
    sources = [_rel(r["path"]) for r in runs]

    entries = []
    for n in sorted(sizes):
        for d in ("dit", "dif"):
            e = build_b1_entry(rows_by_path, d, n, run_by_path,
                               guard=args.guard, threshold=threshold,
                               rejections=rejections)
            if e:
                entries.append(e)
    if args.batch:
        batch_run = next(r for r in all_runs if r["path"] == args.batch)
        entries.extend(build_batch_entries(batch_run, guard=args.guard,
                                           threshold=threshold,
                                           rejections=rejections))
        sources.append(_rel(args.batch))

    table = {
        "schema_version": 2,
        "device": profile.device,
        "profile_hash": profile_hash,
        "precision": "fp32",
        "guard_pct": args.guard,
        "selection_semantics": (
            "search-stage median improvement (aggregated medians within one "
            "run schedule), not paired per-trial medians"),
        "sources": sources,
        "rejections": rejections,
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
          f"({n_b1} at B=1, {n_m0} M0-fallback) guard={args.guard}% "
          f"rejections={len(rejections)}")
    for r in rejections:
        print(f"  rejected: {r}", file=sys.stderr)


if __name__ == "__main__":
    main()
