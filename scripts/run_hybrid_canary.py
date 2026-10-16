#!/usr/bin/env python3
"""P2 re-test: interleaved multi-session canary for the A1 adapter.

Issue #3 P2 acceptance asked for a bounded re-measure of the A1P1
composition cell under a clean protocol: 3 independent sessions, 30
trials each, cells interleaved per trial (random order) to average out
thermal/clock drift, explicit synchronize around every timed call.

Cells measured: A0P0 (frozen-table fused path), A1P0 (framework
decomposition + triton local core), A1P1 (framework decomposition +
vendor host-API sub-blocks), vendor_full (external whole-transform
baseline). Every cell passes the 1e-4 correctness gate on its first
call before it is timed.

Sessions are *in-process* but re-seed the input generator per session
and re-construct HybridPlans per session so plan caches (twiddle,
staging buffers, row tables) are cold at each session start. Each
session's samples are stored separately so thermal drift across
sessions is visible.

Usage::

    CUDA_VISIBLE_DEVICES=<healthy card> timeout -s KILL 1800 \\
      python scripts/run_hybrid_canary.py \\
        --shapes 1024x16,1024x64,2048x16 \\
        --out results/p800/hybrid_canary.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time as _time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from src.xpu.adapter_hybrid import (  # noqa: E402
    plan_hybrid,
    run_hybrid,
    time_cell,
)
from src.xpu.error_metric import max_rel_error  # noqa: E402
from src.xpu.config import git_dirty  # noqa: E402

ERR_MAX = 1e-4


def _parse_shapes(spec: str) -> list[tuple[int, int]]:
    out = []
    for part in spec.split(","):
        n_s, b_s = part.strip().lower().split("x")
        out.append((int(n_s), int(b_s)))
    return out


def _cells_for(n: int, batch: int, tile: int, seed: int, table):
    """Build the four cells for one shape with a *fresh* seed and fresh
    HybridPlans (cold caches at each session start)."""
    import torch
    import run_release_matrix as relm
    from src.xpu import runner

    x = relm._make_input(n, batch, seed)
    ref_np = relm._ref_np(n, batch, seed)
    entry = relm._entry_for(table, n, batch, "dit")

    def a0p0(xx):
        return runner.fft_dit_matrix_batch(
            xx, block=entry.get("block", 1024),
            tile=entry.get("params", {}).get("tile", 64),
            rows=entry.get("params", {}).get("rows", 8),
            variant=entry.get("params", {}).get("variant", "spread"))

    plan_triton = plan_hybrid(n, batch, tile=tile, core="triton")
    plan_vendor = plan_hybrid(n, batch, tile=tile, core="vendor")

    def a1p0(xx, _p=plan_triton):
        return run_hybrid(xx, _p)

    def a1p1(xx, _p=plan_vendor):
        return run_hybrid(xx, _p)

    def vendor_full(xx):
        return torch.fft.fft(xx, dim=-1)

    cells = {
        "A0P0": a0p0,
        "A1P0": a1p0,
        "A1P1": a1p1,
        "vendor_full": vendor_full,
    }
    return x, ref_np, cells


def _first_call_ok(fn, x, ref_np) -> tuple[bool, float]:
    """Correctness gate on the first call (which also pays JIT)."""
    t0 = _time.perf_counter()
    y = fn(x)
    import torch
    torch.cuda.synchronize(x.device)
    first_ms = (_time.perf_counter() - t0) * 1e3
    err = max_rel_error(y.detach().cpu().numpy(), ref_np)
    return err < ERR_MAX, first_ms


def run_session(n: int, batch: int, tile: int, seed: int, trials: int,
                warmup: int, table) -> dict:
    """One session: cold plan caches, warmup per cell, then interleaved
    steady trials with per-trial random cell order."""
    x, ref_np, cells = _cells_for(n, batch, tile, seed, table)

    # Gate + warmup each cell once (order fixed here; interleaving only
    # applies to the steady phase).
    gated = {}
    for name, fn in cells.items():
        ok, first_ms = _first_call_ok(fn, x, ref_np)
        if not ok:
            gated[name] = {"abstain": "correctness gate failed",
                           "first_ms": round(first_ms, 4)}
            continue
        for _ in range(warmup):
            fn(x)
        import torch
        torch.cuda.synchronize(x.device)
        gated[name] = {"first_ms": round(first_ms, 4)}

    live = [name for name in cells if "abstain" not in gated.get(name, {})]
    samples: dict[str, list[float]] = {name: [] for name in live}
    rng = random.Random(seed)
    for _ in range(trials):
        order = list(live)
        rng.shuffle(order)
        for name in order:
            fn = cells[name]
            import torch
            torch.cuda.synchronize(x.device)
            t0 = _time.perf_counter()
            fn(x)
            torch.cuda.synchronize(x.device)
            samples[name].append((_time.perf_counter() - t0) * 1e3)

    for name in live:
        s = sorted(samples[name])
        mid = s[len(s) // 2]
        stdev = (sum((v - mid) ** 2 for v in s) / max(1, len(s) - 1)) ** 0.5
        gated[name]["steady_ms"] = round(mid, 4)
        gated[name]["steady_stdev_ms"] = round(stdev, 4)
        gated[name]["trials"] = len(s)

    return {"seed": seed, "cells": gated, "samples": samples}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shapes", default="1024x16,1024x64,2048x16")
    ap.add_argument("--tile", type=int, default=64)
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=20261016)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "p000" / "hybrid_canary.json")
    args = ap.parse_args()

    import torch
    from src.xpu.selector import load_table

    if not torch.cuda.is_available():
        raise SystemExit("XPU device required for the hybrid canary")
    torch.cuda.set_device(0)
    table = load_table()

    shapes_out = []
    for n, batch in _parse_shapes(args.shapes):
        sessions = []
        for k in range(args.sessions):
            seed = args.seed_base + k * 1000
            sess = run_session(n, batch, args.tile, seed, args.trials,
                               args.warmup, table)
            sessions.append(sess)
            print(f"[N{n}B{batch} sess{k}] " + " ".join(
                f"{name}={c.get('steady_ms', c.get('abstain', '?'))}"
                for name, c in sess["cells"].items()), flush=True)

        # Pooled median across all sessions' steady samples.
        pooled: dict[str, list[float]] = {}
        for sess in sessions:
            for name, s in sess.get("samples", {}).items():
                pooled.setdefault(name, []).extend(s)
        summary = {}
        for name, all_s in pooled.items():
            s = sorted(all_s)
            mid = s[len(s) // 2]
            stdev = (sum((v - mid) ** 2 for v in s)
                     / max(1, len(s) - 1)) ** 0.5
            summary[name] = {
                "pooled_median_ms": round(mid, 4),
                "pooled_stdev_ms": round(stdev, 4),
                "pooled_trials": len(s),
            }
        # Guard evaluation: A1P1 vs A0P0 (incumbent in-table path) and
        # vs vendor_full, under the 2% threshold.
        a0 = (summary.get("A0P0") or {}).get("pooled_median_ms")
        ap1 = (summary.get("A1P1") or {}).get("pooled_median_ms")
        av = (summary.get("vendor_full") or {}).get("pooled_median_ms")
        guard = {}
        if a0 and ap1:
            guard["A1P1_vs_A0P0_pct"] = round((a0 - ap1) / a0 * 100, 3)
            guard["A1P1_beats_A0P0_by_2pct"] = (a0 - ap1) / a0 >= 0.02
        if av and ap1:
            guard["A1P1_vs_vendor_full_pct"] = round(
                (av - ap1) / av * 100, 3)
            guard["A1P1_beats_vendor_by_2pct"] = (av - ap1) / av >= 0.02
        shapes_out.append({
            "N": n, "batch": batch, "direction": "dit",
            "tile": args.tile,
            "sessions": sessions,
            "pooled": summary,
            "guard": guard,
        })

    result = {
        "kind": "issue-3-hybrid-canary",
        "schema": 1,
        "commit": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            text=True).strip(),
        "dirty": git_dirty(),
        "sessions": args.sessions,
        "trials": args.trials,
        "warmup": args.warmup,
        "seed_base": args.seed_base,
        "err_max": ERR_MAX,
        "protocol": (
            "3 independent in-process sessions, 30 interleaved steady "
            "trials each (random cell order per trial, seeded), explicit "
            "synchronize around every timed call. HybridPlans are "
            "re-constructed per session so twiddle/staging/row caches "
            "are cold at session start; first call per cell pays JIT "
            "and is correctness-gated before warmup. Pooled median "
            "across all sessions' samples is the headline number; "
            "per-session medians are retained to expose drift. Guard "
            "evaluation uses the 2% threshold against A0P0 (in-table "
            "incumbent) and vendor_full. A1P1 is an experimental "
            "canary: even a guard pass does not promote it into the "
            "selector table this round (cost model has no A1 family)."
        ),
        "shapes": shapes_out,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} (commit {result['commit'][:8]} "
          f"dirty={result['dirty']})")


if __name__ == "__main__":
    main()
