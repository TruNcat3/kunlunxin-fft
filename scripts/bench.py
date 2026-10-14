#!/usr/bin/env python3
"""Phase-2 fair baseline benchmark (migration convergence plan section 5/7/9).

Protocol:

* Reads the P800 profile, size matrix, and design space from ``config/`` via
  the host-only loader, validates them, and emits a **manifest** for the run.
* Enumerates candidates and reports the lowering-contract verdict for each
  (``legal`` / ``not-lowered`` + reasons) before timing anything.
* For every length, times the candidates with **randomized interleaved trials**
  (each trial picks a candidate in shuffled order) so drift across the run does
  not bias one candidate. Reports ``api_ms`` (synchronized wall time) and, when
  device events report, ``kernel_ms`` separately — median/min/p10/p90.
* Correctness against numpy (complex128) is recorded with **separate DIT and DIF
  error columns**; the vendor ``torch.fft.fft`` row is retained even when it
  fails, with the **actual error string**.

Usage:
    CUDA_VISIBLE_DEVICES=6 python3 scripts/bench.py \
        [--sizes 8,16,...] [--block 1024] [--tiles 64] \
        [--trials 50] [--json out.json] [--csv out.csv]
"""

import argparse
import csv
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch_xmlir  # noqa: E402,F401

from src.xpu import (  # noqa: E402
    DesignSpace,
    DeviceProfile,
    SizeMatrix,
    build_manifest,
)
from src.xpu.candidates import enumerate_candidates, selected_candidate  # noqa: E402
from src.xpu.matrices import tail_groups  # noqa: E402
from src.xpu.runner import (  # noqa: E402
    device_name,
    fft_dif,
    fft_dif_grouped,
    fft_dif_matrix,
    fft_dit,
    fft_dit_fused,
    fft_dit_grouped,
    fft_dit_matrix,
    time_call_detailed,
    twiddle_tables,
)


def make_input(n: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.complex(
        torch.randn(n, dtype=torch.float32, device="cuda"),
        torch.randn(n, dtype=torch.float32, device="cuda"),
    )


def numpy_ref(x: torch.Tensor) -> np.ndarray:
    return np.fft.fft(x.real.cpu().numpy() + 1j * x.imag.cpu().numpy())


def err_vs(got: torch.Tensor, ref: np.ndarray) -> float:
    gr = got.real.cpu().numpy().astype(np.float64)
    gi = got.imag.cpu().numpy().astype(np.float64)
    den = float(np.abs(ref).max())
    return float(np.maximum(np.abs(gr - ref.real), np.abs(gi - ref.imag)).max() / den)


def vendor_fft(x: torch.Tensor):
    """Return (result_or_None, error_string_or_None) for torch.fft.fft."""
    try:
        return torch.fft.fft(x), None
    except Exception as e:  # the vendor caps at N<=2048 with [NOT IMPLEMENTED]
        return None, f"{type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default=None, help="comma-separated lengths")
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--tiles", default="64", help="comma-separated M1 fused tile sizes")
    ap.add_argument("--matrix-tiles", default="64",
                    help="comma-separated M2 register-matrix tile sizes")
    ap.add_argument("--matrix-rows", type=int, default=8,
                    help="tile-blocks per program for the M2 dot kernel")
    ap.add_argument("--grouped-tiles", default="8",
                    help="comma-separated M3 execution-group tile sizes")
    ap.add_argument("--glog", type=int, default=2,
                    help="M3 tail stages per execution group (1 disables grouping)")
    ap.add_argument("--maxp", type=int, default=4096,
                    help="M3 cap on the group period (register-block bound)")
    ap.add_argument("--cells", type=int, default=1,
                    help="M3 BLOCK windows per group-kernel program")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--json", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--markdown", default=None,
                    help="render the results table as markdown from raw rows")
    args = ap.parse_args()

    profile = DeviceProfile.load()
    sizes = SizeMatrix.load()
    space = DesignSpace.load()
    sizes_list = [int(s) for s in args.sizes.split(",")] if args.sizes else list(sizes.bench_sizes)
    tiles = [int(t) for t in args.tiles.split(",")] if args.tiles else list(space.fused_tile_candidates)
    matrix_tiles = [int(t) for t in args.matrix_tiles.split(",")]
    grouped_tiles = [int(t) for t in args.grouped_tiles.split(",")]

    # --- Phase 1: candidate legality before any launch ---
    legal, rejected = enumerate_candidates(space, profile, block=args.block)
    print("== lowering-contract verdicts ==")
    for v in legal:
        c = v.candidate
        print(f"  legal        flow={c.flow_id} tile={c.fused_tile} block={c.block}")
    for v in rejected:
        c = v.candidate
        print(f"  {v.status:<12} flow={c.flow_id} tile={c.fused_tile} :: {'; '.join(v.reasons)}")
    sel = selected_candidate(legal)
    print(f"selected_candidate: {sel.candidate.flow_id if sel else None} "
          f"(conservative pick = lowest fused_tile; M1 must prove itself by measurement)")

    manifest = build_manifest(
        profile,
        flow_id="m0-vs-m1-paired",
        block=args.block,
        warmup=args.warmup,
        trials=args.trials,
        command=" ".join(sys.argv),
        extra={"tiles": tiles, "matrix_tiles": matrix_tiles,
               "matrix_rows": args.matrix_rows,
               "grouped_tiles": grouped_tiles, "glog": args.glog,
               "maxp": args.maxp, "cells": args.cells,
               "sizes": sizes_list, "device_name": device_name()},
    )
    man = manifest.to_dict()
    print(f"manifest: commit={man['commit'][:8]} dirty={man['dirty']} "
          f"profile={man['profile_hash']} device={man['device']}")

    # --- Phase 2: interleaved randomized trials per length ---
    rows = []
    print(f"\n== timings (median api_ms; interleaved {args.trials} trials) ==")
    print(f"{'N':>8} {'M0_dit':>9} {'M0_dif':>9} "
          + "".join(f"{'M1_t'+str(t):>9}" for t in tiles)
          + "".join(f"{'M2d_t'+str(t):>9}" for t in matrix_tiles)
          + "".join(f"{'M3g_dit_t'+str(t):>9}" for t in grouped_tiles)
          + f" {'vendor':>9} {'dit_err':>9} {'dif_err':>9}")

    for n in sizes_list:
        x = make_input(n, seed=0)
        ref = numpy_ref(x)
        tw = twiddle_tables(n, x.device)

        # correctness (DIT and DIF reported separately)
        dit_err = err_vs(fft_dit(x, block=args.block, twiddle=tw), ref)
        dif_err = err_vs(fft_dif(x, block=args.block, twiddle=tw), ref)
        fused_errs = {}
        for t in tiles:
            fused_errs[t] = err_vs(fft_dit_fused(x, block=args.block, tile=t, twiddle=tw), ref)
        matrix_errs = {}
        for t in matrix_tiles:
            matrix_errs[f"dit_t{t}"] = err_vs(
                fft_dit_matrix(x, block=args.block, tile=t, rows=args.matrix_rows), ref)
            matrix_errs[f"dif_t{t}"] = err_vs(
                fft_dif_matrix(x, block=args.block, tile=t, rows=args.matrix_rows), ref)
        grouped_errs = {}
        for t in grouped_tiles:
            grouped_errs[f"dit_t{t}"] = err_vs(
                fft_dit_grouped(x, block=args.block, tile=t, glog=args.glog,
                                maxp=args.maxp, cells=args.cells), ref)
            grouped_errs[f"dif_t{t}"] = err_vs(
                fft_dif_grouped(x, block=args.block, tile=t, glog=args.glog,
                                maxp=args.maxp, cells=args.cells), ref)

        # vendor boundary
        vend, vend_err = vendor_fft(x)
        vend_ms = None
        if vend is not None:
            vend_ms = time_call_detailed(
                lambda: torch.fft.fft(x), warmup=args.warmup, trials=args.trials, seed=7
            )["api_ms"]["median"]

        # candidates
        cands = {
            "m0_dit": lambda: fft_dit(x, block=args.block, twiddle=tw),
            "m0_dif": lambda: fft_dif(x, block=args.block, twiddle=tw),
        }
        for t in tiles:
            cands[f"m1_t{t}"] = lambda t=t: fft_dit_fused(x, block=args.block, tile=t, twiddle=tw)
        for t in matrix_tiles:
            cands[f"m2_dit_t{t}"] = lambda t=t: fft_dit_matrix(
                x, block=args.block, tile=t, rows=args.matrix_rows, twiddle=tw)
            cands[f"m2_dif_t{t}"] = lambda t=t: fft_dif_matrix(
                x, block=args.block, tile=t, rows=args.matrix_rows, twiddle=tw)
        for t in grouped_tiles:
            cands[f"m3_g_dit_t{t}"] = lambda t=t: fft_dit_grouped(
                x, block=args.block, tile=t, glog=args.glog, maxp=args.maxp,
                cells=args.cells, twiddle=tw)
            cands[f"m3_g_dif_t{t}"] = lambda t=t: fft_dif_grouped(
                x, block=args.block, tile=t, glog=args.glog, maxp=args.maxp,
                cells=args.cells, twiddle=tw)

        # launch accounting (analytic, from the flow definitions)
        logn = int(math.log2(n))
        launches = {"m0_dit": logn, "m0_dif": logn}
        for t in tiles:
            fl = int(math.log2(min(t, n)))
            launches[f"m1_t{t}"] = 2 + (logn - fl)
        for t in matrix_tiles:
            fl = int(math.log2(min(t, n)))
            launches[f"m2_dit_t{t}"] = 2 + (logn - fl)
            launches[f"m2_dif_t{t}"] = 2 + (logn - fl)
        for t in grouped_tiles:
            # M3: bit-reversal pass + local matrix + one launch per tail unit
            # (execution group or leftover single stage); matches tail_groups.
            if t < 2:
                launches[f"m3_g_dit_t{t}"] = logn
                launches[f"m3_g_dif_t{t}"] = logn
            else:
                gs, ss = tail_groups(min(t, n), logn, args.glog, args.maxp)
                launches[f"m3_g_dit_t{t}"] = 2 + len(gs) + len(ss)
                launches[f"m3_g_dif_t{t}"] = 2 + len(gs) + len(ss)

        # randomized interleaved trials: round-robin over shuffled candidate order
        order = list(cands)
        rng_state = np.random.RandomState(n)
        timings = {k: [] for k in cands}
        ktime = {k: [] for k in cands}
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        for _ in range(args.warmup):
            for k in order:
                cands[k]()
        torch.cuda.synchronize()
        for _ in range(args.trials):
            rng_state.shuffle(order)
            for k in order:
                t0 = time.perf_counter()
                ev_start.record()
                cands[k]()
                ev_end.record()
                torch.cuda.synchronize()
                timings[k].append((time.perf_counter() - t0) * 1e3)
                kt = ev_start.elapsed_time(ev_end)
                if kt > 0:  # XPU events may return 0 if unsupported
                    ktime[k].append(kt)

        def med(xs):
            xs = sorted(xs)
            return xs[len(xs) // 2] if xs else None

        row = {
            "n": n,
            "dit_err": dit_err,
            "dif_err": dif_err,
            "vendor_ms": vend_ms,
            "vendor_error": vend_err,
        }
        for t in tiles:
            row[f"m1_t{t}_err"] = fused_errs[t]
        for k, e in matrix_errs.items():
            row[f"m2_{k}_err"] = e
        for k, e in grouped_errs.items():
            row[f"m3_g_{k}_err"] = e
        for k in cands:
            row[f"{k}_ms"] = med(timings[k])
            row[f"{k}_kernel_ms"] = med(ktime[k])
            row[f"{k}_launches"] = launches.get(k)
            if vend_ms and row[f"{k}_ms"]:
                row[f"{k}_speedup_vs_vendor"] = vend_ms / row[f"{k}_ms"]
            row[f"{k}_speedup_vs_m0_dit"] = (row["m0_dit_ms"] / row[f"{k}_ms"]
                                             if k != "m0_dit" else 1.0)
        rows.append(row)

        def fmt(v, w=9, spec=".4f"):
            return "n/a".rjust(w) if v is None else format(v, spec).rjust(w)

        print(f"{n:>8} {row['m0_dit_ms']:9.4f} {row['m0_dif_ms']:9.4f} "
              + "".join(f"{row[f'm1_t{t}_ms']:9.4f}" for t in tiles)
              + "".join(f"{row[f'm2_dit_t{t}_ms']:9.4f}" for t in matrix_tiles)
              + "".join(f"{row[f'm3_g_dit_t{t}_ms']:9.4f}" for t in grouped_tiles)
              + f" {fmt(vend_ms)} {dit_err:9.1e} {dif_err:9.1e}")
        k0 = row["m0_dit_kernel_ms"]
        k1 = row[f"m1_t{tiles[0]}_kernel_ms"] if tiles else None
        k2 = row[f"m2_dit_t{matrix_tiles[0]}_kernel_ms"] if matrix_tiles else None
        k3 = row[f"m3_g_dit_t{grouped_tiles[0]}_kernel_ms"] if grouped_tiles else None
        print(f"{'':>8} kernel_ms: M0_dit={fmt(k0, 7, '.4f')} "
              f"M1_t{tiles[0] if tiles else '-'}={fmt(k1, 7, '.4f')} "
              f"M2d_t{matrix_tiles[0] if matrix_tiles else '-'}={fmt(k2, 7, '.4f')} "
              f"M3g_t{grouped_tiles[0] if grouped_tiles else '-'}={fmt(k3, 7, '.4f')}"
              f"  launches: M0={row['m0_dit_launches']} "
              f"M1={row.get(f'm1_t{tiles[0]}_launches') if tiles else '-'} "
              f"M2={row.get(f'm2_dit_t{matrix_tiles[0]}_launches') if matrix_tiles else '-'} "
              f"M3={row.get(f'm3_g_dit_t{grouped_tiles[0]}_launches') if grouped_tiles else '-'}"
              f"  (kernel time = device events, 0 unsupported)")
        if vend_err:
            print(f"{'':>8} vendor unsupported: {vend_err[:110]}")

    # M1 vs M0 summary (paired per size)
    print("\n== M1 vs M0 (api_ms speedup, >1 = M1 faster) ==")
    for t in tiles:
        ratios = [r["m0_dit_ms"] / r[f"m1_t{t}_ms"] for r in rows if r.get(f"m1_t{t}_ms")]
        small = [r["m0_dit_ms"] / r[f"m1_t{t}_ms"] for r in rows
                 if r["n"] <= 1024 and r.get(f"m1_t{t}_ms")]
        big = [r["m0_dit_ms"] / r[f"m1_t{t}_ms"] for r in rows
               if r["n"] >= 4096 and r.get(f"m1_t{t}_ms")]
        gm = lambda xs: float(np.exp(np.mean(np.log(xs)))) if xs else float("nan")
        print(f"  tile={t}: N<=1024 gmean={gm(small):.2f}x  "
              f"N>=4096 gmean={gm(big):.2f}x  all gmean={gm(ratios):.2f}x")

    # M2 vs M0/M1 summary
    for t in matrix_tiles:
        for d in ("dit", "dif"):
            key = f"m2_{d}_t{t}"
            ratios = [r["m0_dit_ms"] / r[f"{key}_ms"] for r in rows if r.get(f"{key}_ms")]
            small = [r["m0_dit_ms"] / r[f"{key}_ms"] for r in rows
                     if r["n"] <= 1024 and r.get(f"{key}_ms")]
            big = [r["m0_dit_ms"] / r[f"{key}_ms"] for r in rows
                   if r["n"] >= 4096 and r.get(f"{key}_ms")]
            vs_m1 = [r[f"m1_t{tiles[0]}_ms"] / r[f"{key}_ms"] for r in rows
                     if tiles and r.get(f"{key}_ms") and r.get(f"m1_t{tiles[0]}_ms")]
            gm = lambda xs: float(np.exp(np.mean(np.log(xs)))) if xs else float("nan")
            print(f"  {key}: vs M0 N<=1024 gmean={gm(small):.2f}x  "
                  f"N>=4096 gmean={gm(big):.2f}x  all gmean={gm(ratios):.2f}x"
                  + (f"  vs M1 all gmean={gm(vs_m1):.2f}x" if vs_m1 else ""))

    # M3 execution-group vs M0/M2 summary
    for t in grouped_tiles:
        for d in ("dit", "dif"):
            key = f"m3_g_{d}_t{t}"
            ratios = [r["m0_dit_ms"] / r[f"{key}_ms"] for r in rows if r.get(f"{key}_ms")]
            small = [r["m0_dit_ms"] / r[f"{key}_ms"] for r in rows
                     if r["n"] <= 1024 and r.get(f"{key}_ms")]
            big = [r["m0_dit_ms"] / r[f"{key}_ms"] for r in rows
                   if r["n"] >= 4096 and r.get(f"{key}_ms")]
            m2key = f"m2_{d}_t{matrix_tiles[0]}" if matrix_tiles else None
            vs_m2 = [r[f"{m2key}_ms"] / r[f"{key}_ms"] for r in rows
                     if m2key and r.get(f"{key}_ms") and r.get(f"{m2key}_ms")]
            gm = lambda xs: float(np.exp(np.mean(np.log(xs)))) if xs else float("nan")
            print(f"  {key}: vs M0 N<=1024 gmean={gm(small):.2f}x  "
                  f"N>=4096 gmean={gm(big):.2f}x  all gmean={gm(ratios):.2f}x"
                  + (f"  vs M2 all gmean={gm(vs_m2):.2f}x" if vs_m2 else ""))

    payload = {"manifest": man, "candidates": {
        "legal": [v.to_dict() for v in legal],
        "rejected": [v.to_dict() for v in rejected],
        "selected": sel.to_dict() if sel else None,
    }, "rows": rows}

    if args.json:
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {args.json}")
    if args.csv:
        fields = sorted({k for r in rows for k in r})
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")

    if args.markdown:
        _write_markdown(args.markdown, rows, tiles, matrix_tiles, grouped_tiles,
                        manifest)
        print(f"wrote {args.markdown}")


def _write_markdown(path, rows, tiles, matrix_tiles, grouped_tiles, man):
    """Render the report table from raw rows so it never needs hand-editing."""
    t0 = tiles[0] if tiles else 1
    m0t = matrix_tiles[0] if matrix_tiles else None
    gm = lambda xs: float(np.exp(np.mean(np.log(xs)))) if xs else float("nan")
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
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
