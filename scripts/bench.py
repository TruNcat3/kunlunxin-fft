#!/usr/bin/env python3
"""Benchmark the P800 Triton FFT against the vendor torch.fft.fft baseline.

Usage:
    CUDA_VISIBLE_DEVICES=6 python3 scripts/bench.py [--sizes 8,16,...] [--json out.json]

Notes:
* The vendor ``torch.fft.fft`` (xdnn) only supports N <= 2048 on this device;
  larger sizes are reported as baseline-unavailable.
* All timings are device-only (kernel launches + ``torch.cuda.synchronize``),
  excluding host-side Python setup such as twiddle-table construction.
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch_xmlir  # noqa: E402,F401

from src.xpu import device_name, fft_dif, fft_dit, twiddle_tables  # noqa: E402

DEFAULT_SIZES = [2**k for k in range(3, 21)]  # 8 .. 1M


def make_input(n: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.complex(
        torch.randn(n, dtype=torch.float32, device="cuda"),
        torch.randn(n, dtype=torch.float32, device="cuda"),
    )


def time_call(fn, warmup: int = 10, iters: int = 50) -> float:
    """Median wall-clock milliseconds for fn(), including device sync."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return samples[len(samples) // 2]


def bench_size(n: int, block: int, seed: int = 0) -> dict:
    x = make_input(n, seed)
    tw = twiddle_tables(n, x.device)
    row = {"n": n, "logn": int(math.log2(n))}

    row["dit_ms"] = time_call(lambda: fft_dit(x, block=block, twiddle=tw))
    row["dif_ms"] = time_call(lambda: fft_dif(x, block=block, twiddle=tw))

    if n <= 2048:
        row["vendor_ms"] = time_call(lambda: torch.fft.fft(x))
    else:
        row["vendor_ms"] = None

    # correctness spot check against numpy on a fresh sample
    y = fft_dit(x, block=block, twiddle=tw)
    host = x.real.cpu().numpy() + 1j * x.imag.cpu().numpy()
    ref = np.fft.fft(host)
    gr = y.real.cpu().numpy().astype(np.float64)
    gi = y.imag.cpu().numpy().astype(np.float64)
    err = float(np.maximum(np.abs(gr - ref.real), np.abs(gi - ref.imag)).max())
    row["max_err"] = err
    row["rel_err"] = err / float(np.abs(ref).max())

    if row["vendor_ms"]:
        row["speedup_vs_vendor_dit"] = row["vendor_ms"] / row["dit_ms"]
        row["speedup_vs_vendor_dif"] = row["vendor_ms"] / row["dif_ms"]
    else:
        row["speedup_vs_vendor_dit"] = None
        row["speedup_vs_vendor_dif"] = None
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default=None, help="comma-separated lengths")
    ap.add_argument("--block", type=int, default=1024, help="Triton BLOCK size")
    ap.add_argument("--json", default=None, help="write full results to this JSON file")
    args = ap.parse_args()

    if args.sizes:
        sizes = [int(s) for s in args.sizes.split(",")]
    else:
        sizes = DEFAULT_SIZES

    dev = device_name()
    print(f"device: {dev}")
    print(f"block : {args.block}")
    print(f"{'N':>10} {'dit_ms':>10} {'dif_ms':>10} {'vendor_ms':>10} "
          f"{'sp_dit':>8} {'sp_dif':>8} {'rel_err':>10}")
    print("-" * 72)

    rows = []
    for n in sizes:
        r = bench_size(n, args.block)
        rows.append(r)
        v = f"{r['vendor_ms']:10.4f}" if r["vendor_ms"] else f"{'n/a':>10}"
        sd = f"{r['speedup_vs_vendor_dit']:8.2f}" if r["speedup_vs_vendor_dit"] else f"{'n/a':>8}"
        sf = f"{r['speedup_vs_vendor_dif']:8.2f}" if r["speedup_vs_vendor_dif"] else f"{'n/a':>8}"
        print(f"{r['n']:>10} {r['dit_ms']:10.4f} {r['dif_ms']:10.4f} {v} {sd} {sf} "
              f"{r['rel_err']:10.2e}")

    # geometric-mean speedup over sizes where a baseline exists
    ratios = [r["speedup_vs_vendor_dit"] for r in rows if r["speedup_vs_vendor_dit"]]
    if ratios:
        gmean = math.exp(sum(math.log(v) for v in ratios) / len(ratios))
        print("-" * 72)
        print(f"geometric mean speedup (DIT vs torch.fft.fft, n={len(ratios)}): {gmean:.2f}x")

    if args.json:
        payload = {
            "device": dev,
            "block": args.block,
            "torch": torch.__version__,
            "rows": rows,
        }
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
