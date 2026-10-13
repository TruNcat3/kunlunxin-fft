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
from src.xpu.runner import (  # noqa: E402
    device_name,
    fft_dif,
    fft_dit,
    fft_dit_fused,
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
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--json", default=None)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    profile = DeviceProfile.load()
    sizes = SizeMatrix.load()
    space = DesignSpace.load()
    sizes_list = [int(s) for s in args.sizes.split(",")] if args.sizes else list(sizes.bench_sizes)
    tiles = [int(t) for t in args.tiles.split(",")] if args.tiles else list(space.fused_tile_candidates)

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
        extra={"tiles": tiles, "sizes": sizes_list, "device_name": device_name()},
    )
    man = manifest.to_dict()
    print(f"manifest: commit={man['commit'][:8]} dirty={man['dirty']} "
          f"profile={man['profile_hash']} device={man['device']}")

    # --- Phase 2: interleaved randomized trials per length ---
    rows = []
    print(f"\n== timings (median api_ms; interleaved {args.trials} trials) ==")
    print(f"{'N':>8} {'M0_dit':>9} {'M0_dif':>9} "
          + "".join(f"{'M1_t'+str(t):>9}" for t in tiles)
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
        for k in cands:
            row[f"{k}_ms"] = med(timings[k])
            row[f"{k}_kernel_ms"] = med(ktime[k])
            if vend_ms and row[f"{k}_ms"]:
                row[f"{k}_speedup_vs_vendor"] = vend_ms / row[f"{k}_ms"]
            row[f"{k}_speedup_vs_m0_dit"] = (row["m0_dit_ms"] / row[f"{k}_ms"]
                                             if k != "m0_dit" else 1.0)
        rows.append(row)

        def fmt(v, w=9, spec=".4f"):
            return "n/a".rjust(w) if v is None else format(v, spec).rjust(w)

        print(f"{n:>8} {row['m0_dit_ms']:9.4f} {row['m0_dif_ms']:9.4f} "
              + "".join(f"{row[f'm1_t{t}_ms']:9.4f}" for t in tiles)
              + f" {fmt(vend_ms)} {dit_err:9.1e} {dif_err:9.1e}")
        k0 = row["m0_dit_kernel_ms"]
        k1 = row[f"m1_t{tiles[0]}_kernel_ms"] if tiles else None
        print(f"{'':>8} kernel_ms: M0_dit={fmt(k0, 7, '.4f')} "
              f"M1_t{tiles[0] if tiles else '-'}={fmt(k1, 7, '.4f')}"
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


if __name__ == "__main__":
    main()
