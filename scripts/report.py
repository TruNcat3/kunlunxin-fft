#!/usr/bin/env python3
"""Torch-free report renderer: rows -> section-9 CSV rows and markdown.

This module is imported by ``scripts/bench.py`` (device path), by
``scripts/regen_markdown.py`` (offline rebuild from a saved JSON) and by the
host golden test ``tests/test_report.py``. It must stay importable in
ordinary CI with NO vendor stack, NO torch and NO numpy: the committed
markdown report has to be regenerable byte-identically from the raw JSON
alone.

Usage::

    python scripts/regen_markdown.py results/p800/bench_phase3.json \
        results/p800/bench_phase3.md
"""
from __future__ import annotations

import json
import math
import re

__all__ = [
    "LONG_FIELDS",
    "long_rows",
    "write_markdown",
    "write_long_csv",
    "scalar_fields",
    "derive_axes",
    "render_from_json",
    "gmean",
]


def gmean(xs):
    """Geometric mean (double precision, NaN-safe) without numpy."""
    xs = [x for x in xs if x]
    if not xs:
        return float("nan")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


LONG_FIELDS = [
    "commit", "device", "runtime", "profile_hash", "N", "batch", "direction",
    "precision", "flow_id", "flow_realization", "processing_unit",
    "Us", "Ts", "Ud", "Td", "boundary_residence", "local_residence",
    "layout", "compiled", "correct", "max_err", "kernel_ms", "api_ms",
    "api_p10", "api_p90", "launches",
    "gm_read_bytes", "gm_write_bytes", "local_bytes",
    "execution_groups", "stages_per_group", "occupancy", "notes",
]


def _tail_groups(tile: int, logn: int, glog: int, maxp: int):
    """Stdlib clone of ``src.xpu.matrices.tail_groups`` (host import guard).

    The report module must stay importable without numpy/torch, and
    ``matrices.py`` needs numpy for its verification helpers; the greedy
    partition is pure integer arithmetic, so it is cloned here and
    cross-checked against the original by ``tests/test_host.py``.
    """
    if glog < 1:
        raise ValueError(f"glog must be >= 1, got {glog}")
    stages = [1 << s for s in range(tile.bit_length() - 1, logn)]
    groups = []
    singles = []
    i = 0
    while i < len(stages):
        n0 = stages[i]
        take = 0
        for s in range(min(glog, len(stages) - i), 1, -1):
            if (n0 << s) <= maxp:
                take = s
                break
        if take:
            groups.append((n0, take))
            i += take
        else:
            singles.append(n0)
            i += 1
    return groups, singles


def _fmt_groups(groups) -> str:
    return ";".join(f"{n0}x{g}" for n0, g in groups)


def _fmt_singles(singles) -> str:
    return ";".join(str(s) for s in singles)


def long_rows(rows, man, threshold=1e-4):
    """Plan section 9 raw rows: one per (flow, N, batch, direction).

    Schema v2 (candidates.py vocabulary): ``flow_realization``,
    ``processing_unit`` and ``local_residence`` are separate attribution
    axes; ``boundary_residence`` records the stage/group-boundary memory.
    ``local_residence`` is ``none`` for the global-stage flows (M0/M4/M5 —
    no multi-stage residency inside a launch), ``cache`` for M1,
    ``register+implicit-dot`` for M2 and ``register`` for M3.

    GM bytes are **analytic** for the radix-2 global-stage flows (M0 and
    each M4 batched stage: one launch reads N complex64 inputs + N/2
    complex64 twiddles = 12N bytes and writes N complex64 = 8N bytes,
    multiplied by launches and by B for batch); M5's radix-4 twiddle
    ledger differs, so those columns stay empty rather than guessed, as
    do the fused/matrix/grouped flows (no counter instrumentation on this
    stack). Analytic columns carry the ``analytic-gm-bytes`` note.
    """
    out = []
    for r in rows:
        n = r["n"]
        logn = n.bit_length() - 1
        extra = man.get("extra") or {}
        glog = int(extra.get("glog", 2))
        maxp = int(extra.get("maxp", 4096))
        for k in r:
            if not k.endswith("_ms") or k == "vendor_ms" or k.endswith("_kernel_ms"):
                continue
            base = k[:-3]
            err = None
            flow = None
            realization = None
            unit = None
            local_res = "none"
            direction = None
            batch = 1
            ud, td = 1, 1
            notes = ""
            groups_str = ""
            singles_str = ""
            if base.startswith("m0_"):
                flow, realization = "baseline-global-stage", "global-stage"
                unit, direction = "radix2-scalar", base[3:]
                err = r.get(f"{direction}_err")
            elif base.startswith("m1_"):
                flow, realization = "local-fused-tile", "fused-inplace"
                unit, local_res, direction = "radix2-scalar", "cache", "dit"
                notes = "m1-fused-inplace"
                err = r.get(f"{base}_err")
            elif base.startswith("m2_"):
                flow, realization = "local-fused-tile", "matrix-resident"
                unit = "dense-dot-fft64"
                local_res = "register+implicit-dot"
                direction = "dit" if "_dit_" in base else "dif"
                notes = "m2-matrix-resident"
                err = r.get(f"{base}_err")
            elif base.startswith("m3_"):
                flow, realization = "local-fused-tile", "grouped-boundary"
                unit, local_res = "composite-linear-map", "register"
                direction = "dit" if "_dit_" in base else "dif"
                notes = "m3-grouped-boundary"
                err = r.get(f"{base}_err")
                m = re.fullmatch(r"m3_g_(?:dif|dit)_t(\d+)", base)
                if m:
                    tile = min(int(m.group(1)), n)
                    if tile >= 2 and glog >= 1:
                        gs, ss = _tail_groups(tile, logn, glog, maxp)
                        groups_str = _fmt_groups(gs)
                        singles_str = _fmt_singles(ss)
            elif base.startswith("m5_"):
                flow, realization = "baseline-global-stage", "global-stage"
                unit, direction = "radix4-scalar", base.rsplit("_", 1)[-1]
                err = r.get(f"{base}_err")
                notes = "m5-radix4-scalar"
            elif base.startswith("m4_"):
                parts = base.split("_")            # m4, variant, bNN, dit|dif
                if len(parts) != 4:
                    continue
                variant, bkey, direction = parts[1], parts[2], parts[3]
                flow = f"batch-{variant}"
                realization = "global-stage"
                unit = "radix2-scalar"
                batch = int(bkey[1:])
                err = r.get(f"{base}_err")
                if variant == "spread":
                    ud = batch
                else:
                    td = batch
                notes = f"m4-batch-{variant}"
            if flow is None or direction is None:
                continue
            launches = r.get(f"{base}_launches")
            analytic = unit == "radix2-scalar" and realization == "global-stage"
            read_b = write_b = None
            if analytic and launches:
                read_b = 12 * n * launches * batch
                write_b = 8 * n * launches * batch
                notes = (notes + " " if notes else "") + "analytic-gm-bytes"
            out.append({
                "commit": man["commit"],
                "device": man["device"],
                "runtime": man["driver"],
                "profile_hash": man["profile_hash"],
                "N": n,
                "batch": batch,
                "direction": direction,
                "precision": "fp32",
                "flow_id": flow,
                "flow_realization": realization,
                "processing_unit": unit,
                "Us": 1, "Ts": 1, "Ud": ud, "Td": td,
                "boundary_residence": "gm",
                "local_residence": local_res,
                "layout": "planar-soa",
                "compiled": True,
                "correct": err is not None and err <= threshold,
                "max_err": err,
                "kernel_ms": r.get(f"{base}_kernel_ms"),
                "api_ms": r[k],
                "api_p10": r.get(f"{base}_p10"),
                "api_p90": r.get(f"{base}_p90"),
                "launches": launches,
                "gm_read_bytes": read_b,
                "gm_write_bytes": write_b,
                "local_bytes": 0 if local_res == "none" else None,
                "execution_groups": groups_str,
                "stages_per_group": singles_str,
                "occupancy": None,
                "notes": notes,
            })
    return out


def write_long_csv(path, lrows):
    """Write the section-9 long rows with the canonical field order."""
    import csv

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LONG_FIELDS)
        w.writeheader()
        w.writerows(lrows)


def scalar_fields(rows):
    """Wide-CSV column set: every scalar key across rows (lists excluded)."""
    keys = set()
    for r in rows:
        for k, v in r.items():
            if not isinstance(v, list):
                keys.add(k)
    return sorted(keys)


def write_markdown(path, rows, tiles, matrix_tiles, grouped_tiles, man,
                    batchs=(), batch_variants=()):
    """Render the report table from raw rows so it never needs hand-editing."""
    t0 = tiles[0] if tiles else 1
    m0t = matrix_tiles[0] if matrix_tiles else None
    gm = gmean
    small = [r["m0_dit_ms"] / r[f"m1_t{t0}_ms"] for r in rows
             if r["n"] <= 1024 and r.get(f"m1_t{t0}_ms")]
    big = [r["m0_dit_ms"] / r[f"m1_t{t0}_ms"] for r in rows
           if r["n"] >= 4096 and r.get(f"m1_t{t0}_ms")]
    allr = [r["m0_dit_ms"] / r[f"m1_t{t0}_ms"] for r in rows if r.get(f"m1_t{t0}_ms")]
    m2k = f"m2_dit_t{m0t}" if m0t else None
    m2_small = [r["m0_dit_ms"] / r[f"{m2k}_ms"] for r in rows
                if m2k and r["n"] <= 1024 and r.get(f"{m2k}_ms")]
    m2_big = [r["m0_dit_ms"] / r[f"{m2k}_ms"] for r in rows
              if m2k and r["n"] >= 4096 and r.get(f"{m2k}_ms")]
    m2_all = [r["m0_dit_ms"] / r[f"{m2k}_ms"] for r in rows
              if m2k and r.get(f"{m2k}_ms")]
    g3t = grouped_tiles[0] if grouped_tiles else None
    g3k = f"m3_g_dit_t{g3t}" if g3t else None
    g3_small = [r["m0_dit_ms"] / r[f"{g3k}_ms"] for r in rows
                if g3k and r["n"] <= 1024 and r.get(f"{g3k}_ms")]
    g3_big = [r["m0_dit_ms"] / r[f"{g3k}_ms"] for r in rows
              if g3k and r["n"] >= 4096 and r.get(f"{g3k}_ms")]
    g3_all = [r["m0_dit_ms"] / r[f"{g3k}_ms"] for r in rows
              if g3k and r.get(f"{g3k}_ms")]
    lines = [
        f"<!-- generated by scripts/bench.py --markdown; commit {man.commit[:8]}, "
        f"profile {man.profile_hash}, trials={man.trials}, warmup={man.warmup} -->",
        "",
        "| N | M0 dit | M0 dif | M1 fused | M2 dit | M3 dit | vendor | M0 vs vendor | "
        "M1 vs M0 | M2 vs M0 | M3 vs M0 | dit_err | dif_err |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        vend = format(r["vendor_ms"], ".4f") if r["vendor_ms"] else "n/a"
        sp_v = format(r["m0_dit_speedup_vs_vendor"], ".2f") + "x" \
            if r.get("m0_dit_speedup_vs_vendor") else "n/a"
        if r.get(f"m1_t{t0}_ms"):
            ratio = r["m0_dit_ms"] / r[f"m1_t{t0}_ms"]
            sp_m1 = f"**{ratio:.2f}x**" if ratio > 1 else f"{ratio:.2f}x"
        else:
            sp_m1 = "n/a"
        if m2k and r.get(f"{m2k}_ms"):
            ratio2 = r["m0_dit_ms"] / r[f"{m2k}_ms"]
            sp_m2 = f"**{ratio2:.2f}x**" if ratio2 > 1 else f"{ratio2:.2f}x"
            m2_ms = format(r[f"{m2k}_ms"], ".4f")
        else:
            sp_m2 = "n/a"
            m2_ms = "n/a"
        if g3k and r.get(f"{g3k}_ms"):
            ratio3 = r["m0_dit_ms"] / r[f"{g3k}_ms"]
            sp_m3 = f"**{ratio3:.2f}x**" if ratio3 > 1 else f"{ratio3:.2f}x"
            m3_ms = format(r[f"{g3k}_ms"], ".4f")
        else:
            sp_m3 = "n/a"
            m3_ms = "n/a"
        lines.append(
            f"| {r['n']} | {r['m0_dit_ms']:.4f} | {r['m0_dif_ms']:.4f} "
            f"| {r.get(f'm1_t{t0}_ms', 0):.4f} | {m2_ms} | {m3_ms} | {vend} | {sp_v} "
            f"| {sp_m1} | {sp_m2} | {sp_m3} "
            f"| {r['dit_err']:.1e} | {r['dif_err']:.1e} |")
    lines += [
        "",
        f"M1 tile={t0} geometric-mean speedup vs M0: "
        f"`N<=1024` **{gm(small):.2f}x**, `N>=4096` **{gm(big):.2f}x**, "
        f"all sizes {gm(allr):.2f}x.",
        "",
        f"M2 dit tile={m0t} geometric-mean speedup vs M0: "
        f"`N<=1024` **{gm(m2_small):.2f}x**, `N>=4096` **{gm(m2_big):.2f}x**, "
        f"all sizes {gm(m2_all):.2f}x.",
        "",
        f"M3 dit grouped tile={g3t} geometric-mean speedup vs M0: "
        f"`N<=1024` **{gm(g3_small):.2f}x**, `N>=4096` **{gm(g3_big):.2f}x**, "
        f"all sizes {gm(g3_all):.2f}x.",
        "",
    ]
    # M5 radix-4: same baseline flow, different processing unit
    if rows and all(r.get("m5_r4_dit_ms") for r in rows):
        m5_small = [r["m0_dit_ms"] / r["m5_r4_dit_ms"] for r in rows if r["n"] <= 1024]
        m5_big = [r["m0_dit_ms"] / r["m5_r4_dit_ms"] for r in rows if r["n"] >= 4096]
        m5_all = [r["m0_dit_ms"] / r["m5_r4_dit_ms"] for r in rows]
        m5d_small = [r["m0_dif_ms"] / r["m5_r4_dif_ms"] for r in rows if r["n"] <= 1024]
        m5d_big = [r["m0_dif_ms"] / r["m5_r4_dif_ms"] for r in rows if r["n"] >= 4096]
        lines += [
            "### M5 radix-4 vs M0 radix-2 (same flow, median api_ms)",
            "",
            "| N | M5 r4 dit | M5 r4 dif | dit err | dif err | dit vs M0 | dif vs M0 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in rows:
            rd = r["m0_dit_ms"] / r["m5_r4_dit_ms"]
            rf = r["m0_dif_ms"] / r["m5_r4_dif_ms"]
            lines.append(
                f"| {r['n']} | {r['m5_r4_dit_ms']:.4f} | {r['m5_r4_dif_ms']:.4f} "
                f"| {r.get('m5_r4_dit_err', 0):.1e} | {r.get('m5_r4_dif_err', 0):.1e} "
                f"| {rd:.2f}x | {rf:.2f}x |")
        lines += [
            "",
            f"M5 radix-4 dit geometric-mean speedup vs M0: "
            f"`N<=1024` **{gm(m5_small):.2f}x**, `N>=4096` **{gm(m5_big):.2f}x**, "
            f"all sizes {gm(m5_all):.2f}x. DIF: "
            f"`N<=1024` **{gm(m5d_small):.2f}x**, `N>=4096` **{gm(m5d_big):.2f}x**.",
            "",
        ]
    # M4 batch tables (workload axis: raw api_ms, no cross-B speedup claim)
    if batchs and batch_variants:
        for d in ("dit", "dif"):
            lines += [
                f"### M4 batch {d} (median api_ms)",
                "",
                "| N | " + " | ".join(
                    f"{v} b={b}" for v in batch_variants for b in batchs) + " |",
                "|---:|" + "---:|" * (len(batchs) * len(batch_variants)),
            ]
            for r in rows:
                cells = []
                for v in batch_variants:
                    for b in batchs:
                        ms = r.get(f"m4_{v}_b{b}_{d}_ms")
                        cells.append(format(ms, ".4f") if ms else "n/a")
                lines.append(f"| {r['n']} | " + " | ".join(cells) + " |")
            lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


class _Manifest:
    """Attribute view over the manifest dict (matches bench's object)."""

    def __init__(self, d):
        self.__dict__.update(d)


def derive_axes(rows):
    """Recover the candidate axes from raw row keys (json is source of truth)."""
    keys = set()
    for r in rows:
        keys.update(r.keys())
    tiles = sorted({int(m.group(1)) for k in keys
                    if (m := re.fullmatch(r"m1_t(\d+)_ms", k))})
    matrix_tiles = sorted({int(m.group(1)) for k in keys
                           if (m := re.fullmatch(r"m2_dit_t(\d+)_ms", k))})
    grouped_tiles = sorted({int(m.group(1)) for k in keys
                            if (m := re.fullmatch(r"m3_g_dit_t(\d+)_ms", k))})
    bm = {re.fullmatch(r"m4_(\w+)_b(\d+)_dit_ms", k) for k in keys}
    bm = {m for m in bm if m}
    batch_variants = sorted({m.group(1) for m in bm})
    batchs = sorted({int(m.group(2)) for m in bm})
    return {"tiles": tiles, "matrix_tiles": matrix_tiles,
            "grouped_tiles": grouped_tiles, "batchs": batchs,
            "batch_variants": batch_variants}


def render_from_json(src, dst):
    """Regenerate the markdown report from a saved bench JSON (offline)."""
    with open(src) as f:
        data = json.load(f)
    axes = derive_axes(data["rows"])
    write_markdown(dst, data["rows"], axes["tiles"], axes["matrix_tiles"],
                   axes["grouped_tiles"], _Manifest(data["manifest"]),
                   batchs=axes["batchs"], batch_variants=axes["batch_variants"])
