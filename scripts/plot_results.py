#!/usr/bin/env python3
"""Render the section-7 experiment-matrix plots from raw bench JSON.

Every published table gets an adjacent plot (plan section 7): this script
reads the raw rows emitted by ``scripts/bench.py --json`` and writes PNGs
next to them. No numbers are computed that are not already in the raw file,
so the plot and the table can never disagree.

Charts
------
1. ``api_ms_vs_n.png``          median api_ms vs N for every flow (log-log),
                                external baseline (vendor) included where
                                supported.
2. ``speedup_vs_m0.png``        per-size speedup vs the M0 baseline (DIT).
3. ``batch_throughput.png``     batch Gpts/s (b*N/t) vs N per variant/B.
4. ``block_sweep.png``          M0 DIT api_ms vs N for every declared legal
                                block (one JSON per block, section 7 axis).

Usage::

    python scripts/plot_results.py \
        --json results/p800/bench_phase3.json \
        --block-json results/p800/bench_block256.json \
        --block-json results/p800/bench_block512.json \
        --block-json results/p800/bench_block2048.json \
        --out results/p800/plots
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HILITE = (4096, 16384, 65536, 262144, 1048576)


def load(path):
    with open(path) as f:
        return json.load(f)


def _series(rows, key):
    return [(r["n"], r[key]) for r in rows if r.get(key)]


def _finish(ax, title, xlabel="N (points)", ylabel="median api_ms (ms)"):
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    for h in HILITE:
        ax.axvline(h, color="tab:red", alpha=0.08, linewidth=6)
    ax.legend(fontsize=8, loc="best")


def plot_api_ms(rows, path):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    lines = [
        ("M0 DIT (radix-2)", "m0_dit_ms", "C0-", "o"),
        ("M0 DIF (radix-2)", "m0_dif_ms", "C0--", "s"),
        ("M1 fused DIT", "m1_t64_ms", "C1-", "^"),
        ("M2 matrix DIT", "m2_dit_t64_ms", "C2-", "v"),
        ("M3 grouped DIT", "m3_g_dit_t8_ms", "C3-", "D"),
        ("M5 radix-4 DIT", "m5_r4_dit_ms", "C4-", "P"),
        ("M5 radix-4 DIF", "m5_r4_dif_ms", "C4--", "X"),
        ("vendor (native)", "vendor_ms", "k-", "*"),
    ]
    for label, key, style, marker in lines:
        pts = _series(rows, key)
        if pts:
            ax.plot(*zip(*pts), style, marker=marker, markersize=5,
                    label=label, linewidth=1.4)
    _finish(ax, "P800 c2c FFT latency by flow (red bands = highlighted sizes)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_speedup(rows, path):
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ns = [r["n"] for r in rows]

    def sp(key):
        return [r["m0_dit_ms"] / r[key] if r.get(key) else None for r in rows]

    for label, key, style, marker in (
        ("M1 fused", "m1_t64_ms", "C1-o", "o"),
        ("M2 matrix", "m2_dit_t64_ms", "C2-^", "^"),
        ("M3 grouped", "m3_g_dit_t8_ms", "C3-D", "D"),
        ("M5 radix-4 DIT", "m5_r4_dit_ms", "C4-P", "P"),
    ):
        y = sp(key)
        if any(v is not None for v in y):
            ax.plot(ns, y, style, markersize=5, label=label, linewidth=1.4)
    ax.axhline(1.0, color="k", linewidth=1, linestyle=":")
    ax.text(ns[0], 1.05, "M0 baseline", fontsize=8, color="k")
    _finish(ax, "Speedup vs M0 baseline, DIT (higher = better)",
            ylabel="speedup vs M0 (x)")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_batch(rows, path):
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    variants = sorted({k.split("_")[1] for r in rows
                       for k in r if k.startswith("m4_") and k.endswith("_ms")})
    batches = sorted({int(k.split("_b")[1].split("_")[0]) for r in rows
                      for k in r if k.startswith("m4_") and "_b" in k
                      and k.endswith("_ms")})
    cmap = plt.get_cmap("tab10")
    i = 0
    for v in variants:
        for b in batches:
            for d, ls in (("dit", "-"), ("dif", "--")):
                key = f"m4_{v}_b{b}_{d}_ms"
                pts = [(r["n"], b * r["n"] / (r[key] * 1e6))
                       for r in rows if r.get(key)]
                if pts:
                    ax.plot(*zip(*pts), color=cmap(i % 10), linestyle=ls,
                            marker="o", markersize=4, linewidth=1.3,
                            label=f"{v} b={b} {d}")
                i += 1
    _finish(ax, "Batch throughput (Gpts/s) by variant and batch",
            ylabel="Gpts/s")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_block_sweep(block_runs, path):
    """block_runs: list of (block, rows)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, (key, title, suffix) in zip(
        axes,
        (("m0_dit_ms", "M0 DIT (radix-2)", "dit"),
         ("m5_r4_dit_ms", "M5 radix-4 DIT", "r4-dit")),
    ):
        for block, rows in block_runs:
            pts = _series(rows, key)
            if pts:
                ax.plot(*zip(*pts), marker="o", markersize=4, linewidth=1.4,
                        label=f"block={block}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("N (points)")
        ax.set_title(f"{title} vs block", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("median api_ms (ms)")
    fig.suptitle("Block sweep across every declared legal block "
                 "(section 7 axis)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", required=True, help="main bench JSON")
    ap.add_argument("--block-json", action="append", default=[],
                    help="per-block bench JSON (repeatable)")
    ap.add_argument("--out", default="results/p800/plots")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    main_data = load(args.json)
    rows = main_data["rows"]

    written = []
    for name, fn in (
        ("api_ms_vs_n.png", lambda: plot_api_ms(rows,
         os.path.join(args.out, "api_ms_vs_n.png"))),
        ("speedup_vs_m0.png", lambda: plot_speedup(rows,
         os.path.join(args.out, "speedup_vs_m0.png"))),
        ("batch_throughput.png", lambda: plot_batch(rows,
         os.path.join(args.out, "batch_throughput.png"))),
    ):
        fn()
        written.append(os.path.join(args.out, name))

    if args.block_json:
        block_runs = []
        for p in args.block_json:
            data = load(p)
            block_runs.append((data["manifest"].get("block", "?"), data["rows"]))
        block_runs.sort(key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0)
        plot_block_sweep(block_runs, os.path.join(args.out, "block_sweep.png"))
        written.append(os.path.join(args.out, "block_sweep.png"))

    for w in written:
        print(f"wrote {w}")


if __name__ == "__main__":
    main()
