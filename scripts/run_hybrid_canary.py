#!/usr/bin/env python3
"""P2 re-test: multi-process interleaved canary for the A1 adapter.

Issue #3 review §2/§3: 3 independent *processes* (not in-process
sessions — a fresh Python allocator, JIT cache, and runtime state per
session), 30 interleaved steady trials each (random cell order per
trial), explicit synchronize around every timed call, HybridPlan
caches cold per session, 1e-4 correctness gate on the first call
before warmup. A fourth *confirmation* session is run whenever the
A1P1-vs-vendor_full guard passes at any shape in the first three.

Freshness: each worker process re-asserts the live environment against
the selector table (zero kernel launches on mismatch) and records full
provenance (commit, env/profile hashes, device/runtime versions,
argv). The parent refuses to start when the tracked tree is dirty
(override with --allow-dirty for non-publishable probes).

Cells measured: A0P0 (frozen-table fused path via selector.select),
A1P0 (framework decomposition + triton local core), A1P1 (framework
decomposition + vendor host-API sub-blocks), vendor_full (external
whole-transform baseline). A1P1 is an experimental canary: a guard
pass does not promote it (the cost model has no A1 family).

Statistics: pooled and per-session p50 / MAD / p95 / max / outlier
count (samples > p50 + 3*MAD); every outlier value is disclosed in the
JSON and the report — a pooled median is never shown alone.

Usage::

    CUDA_VISIBLE_DEVICES=<healthy card> timeout -s KILL 1800 \\
      python scripts/run_hybrid_canary.py \\
        --shapes 1024x16,1024x64,2048x16 \\
        --out results/p800/hybrid_canary.json
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import subprocess
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
    summarize_samples,
)
from src.xpu.error_metric import max_rel_error  # noqa: E402
from src.xpu.config import git_dirty  # noqa: E402

ERR_MAX = 1e-4
CELL_NAMES = ("A0P0", "A1P0", "A1P1", "vendor_full")


def _parse_shapes(spec: str) -> list[tuple[int, int]]:
    out = []
    for part in spec.split(","):
        n_s, b_s = part.strip().lower().split("x")
        out.append((int(n_s), int(b_s)))
    return out


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def _provenance(table: dict, env_hash: str, argv: list[str]) -> dict:
    from src.xpu.envcheck import collect
    env = collect() or {}
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": git_dirty(),
        "argv": argv,
        "pid": os.getpid(),
        "table_env_hash": table.get("env_hash"),
        "live_env_hash": env_hash,
        "profile_hash": table.get("profile_hash"),
        "device": {
            "driver": env.get("driver"),
            "device_model": env.get("device_model"),
            "multiprocessor_count": env.get("multiprocessor_count"),
            "max_shared_mem": env.get("max_shared_mem"),
        },
        "runtime": {
            "torch": env.get("torch"),
            "triton": env.get("triton"),
            "xpytorch": env.get("xpytorch"),
            "xre": env.get("xre"),
            "xhpc": env.get("xhpc"),
        },
    }


def _cells_for(n: int, batch: int, tile: int, seed: int, table: dict,
               env_hash: str):
    """Fresh input + fresh HybridPlans (cold caches) via the live
    selector path for A0P0 (env-verified entry)."""
    import torch
    import run_release_matrix as relm
    from src.xpu import runner
    from src.xpu.selector import select

    x = relm._make_input(n, batch, seed)
    ref_np = relm._ref_np(n, batch, seed)
    sel = select("dit", n, batch, precision=table["precision"],
                 profile_hash=table["profile_hash"], table=table,
                 env_hash=env_hash)
    if sel.source != "table":
        raise RuntimeError(
            f"selector abstained for N{n}B{batch} ({sel.reason}); "
            "cannot build the A0P0 cell")
    entry = {"block": sel.block, "params": sel.params}

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

    return x, ref_np, {
        "A0P0": a0p0, "A1P0": a1p0, "A1P1": a1p1,
        "vendor_full": vendor_full,
    }


def _run_session(job: dict) -> dict:
    """One independent process session: env gate, cold plans, gated
    warmup, interleaved steady trials. Returns raw samples + stats."""
    sys.path.insert(0, str(ROOT))
    from src.xpu.selector import assert_current_environment, load_table

    # Freshness gate in EVERY worker (review §2): zero launches on
    # mismatch, even if the parent already checked.
    table = load_table()
    env_hash = assert_current_environment(table)

    import torch
    torch.cuda.set_device(0)

    n, batch = job["n"], job["batch"]
    x, ref_np, cells = _cells_for(n, batch, job["tile"], job["seed"],
                                  table, env_hash)

    gated: dict[str, dict] = {}
    for name, fn in cells.items():
        t0 = _time.perf_counter()
        y = fn(x)
        torch.cuda.synchronize(x.device)
        first_ms = (_time.perf_counter() - t0) * 1e3
        err = max_rel_error(y.detach().cpu().numpy(), ref_np)
        if err >= ERR_MAX:
            gated[name] = {"abstain": f"correctness gate failed "
                           f"(max_rel_err={err:.2e})", "first_ms": round(
                               first_ms, 4)}
            continue
        for _ in range(job["warmup"]):
            fn(x)
        torch.cuda.synchronize(x.device)
        gated[name] = {"first_ms": round(first_ms, 4)}

    live = [name for name in cells if "abstain" not in gated.get(name, {})]
    samples: dict[str, list[float]] = {name: [] for name in live}
    rng = random.Random(job["seed"])
    for _ in range(job["trials"]):
        order = list(live)
        rng.shuffle(order)
        for name in order:
            fn = cells[name]
            torch.cuda.synchronize(x.device)
            t0 = _time.perf_counter()
            fn(x)
            torch.cuda.synchronize(x.device)
            samples[name].append((_time.perf_counter() - t0) * 1e3)

    for name in live:
        gated[name]["samples"] = samples[name]
        gated[name]["stats"] = summarize_samples(samples[name])

    return {
        "session_index": job["session_index"],
        "role": job["role"],
        "seed": job["seed"],
        "pid": os.getpid(),
        "provenance": _provenance(table, env_hash, job["argv"]),
        "cells": gated,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shapes", default="1024x16,1024x64,2048x16")
    ap.add_argument("--tile", type=int, default=64)
    ap.add_argument("--sessions", type=int, default=3,
                    help="protocol sessions (independent processes); a "
                    "confirmation session is added automatically when "
                    "the A1P1 guard passes anywhere")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=20261016)
    ap.add_argument("--allow-dirty", action="store_true",
                    help="measure anyway on a dirty tracked tree")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "p000" / "hybrid_canary.json")
    args = ap.parse_args()

    from src.xpu.selector import assert_current_environment, load_table

    # Parent-side freshness gate before spawning anything.
    if git_dirty() and not args.allow_dirty:
        raise SystemExit(
            "tracked tree is dirty; commit first so the artifact "
            "provenance is meaningful (or pass --allow-dirty)")
    table = load_table()
    env_hash = assert_current_environment(table)

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("XPU device required for the hybrid canary")

    shapes = _parse_shapes(args.shapes)
    # One job per (shape, session). Protocol sessions first; the
    # confirmation session is decided after the first three finish.
    base_jobs = []
    for n, batch in shapes:
        for k in range(args.sessions):
            base_jobs.append({
                "n": n, "batch": batch, "tile": args.tile,
                "seed": args.seed_base + k * 1000 + n + batch,
                "session_index": k, "role": "protocol",
                "trials": args.trials, "warmup": args.warmup,
                "argv": sys.argv,
            })

    ctx = mp.get_context("spawn")
    results: list[dict] = []
    result_jobs: list[dict] = []  # parallel to results
    with ctx.Pool(processes=1) as pool:  # serial: one healthy card
        for job in base_jobs:
            print(f"session {job['session_index']} "
                  f"(N{job['n']}B{job['batch']}, {job['role']}) ...",
                  flush=True)
            results.append(pool.apply(_run_session, (job,)))
            result_jobs.append(job)

    # Guard evaluation on the protocol sessions (per shape) -> maybe
    # one confirmation session per shape that passes.
    need_confirm_shapes: list[tuple[int, int]] = []
    for n, batch in shapes:
        pooled: dict[str, list[float]] = {}
        for job, r in zip(result_jobs, results):
            if (job["n"], job["batch"]) != (n, batch):
                continue
            if r["provenance"]["commit"] != results[0]["provenance"]["commit"]:
                continue
            for name, c in r["cells"].items():
                pooled.setdefault(name, []).extend(c.get("samples", []))
        med = {k: summarize_samples(v).get("p50")
               for k, v in pooled.items()}
        a0, ap1, av = med.get("A0P0"), med.get("A1P1"), med.get(
            "vendor_full")
        if ap1 and av and (av - ap1) / av >= 0.02:
            need_confirm_shapes.append((n, batch))

    if need_confirm_shapes:
        with ctx.Pool(processes=1) as pool:
            for n, batch in need_confirm_shapes:
                job = {
                    "n": n, "batch": batch, "tile": args.tile,
                    "seed": args.seed_base + 99 * 1000 + n + batch,
                    "session_index": args.sessions,
                    "role": "confirmation",
                    "trials": args.trials, "warmup": args.warmup,
                    "argv": sys.argv,
                }
                print(f"session {job['session_index']} "
                      f"(N{n}B{batch}, confirmation) ...", flush=True)
                results.append(pool.apply(_run_session, (job,)))
                result_jobs.append(job)

    # Merge: pooled + per-session stats + outlier disclosure.
    results_by_shape: dict[tuple, list] = {s: [] for s in
                                           ((j["n"], j["batch"])
                                            for j in base_jobs)}
    assert len(result_jobs) == len(results), (
        f"job/result count mismatch: {len(result_jobs)} vs "
        f"{len(results)}")
    for job, res in zip(result_jobs, results):
        results_by_shape[(job["n"], job["batch"])].append(res)

    shapes_out = []
    for (n, batch), sres in results_by_shape.items():
        pooled = {}
        for r in sres:
            for name, c in r["cells"].items():
                pooled.setdefault(name, []).extend(c.get("samples", []))
        pooled_stats = {
            name: {k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in summarize_samples(s).items()
                   if k != "outlier_values"}
            | {"outlier_values": [round(v, 4) for v in
                                  summarize_samples(s).get(
                                      "outlier_values", [])]}
            for name, s in pooled.items()
        }
        sessions_out = []
        for r in sres:
            cells_out = {}
            for name, c in r["cells"].items():
                cells_out[name] = {
                    "first_ms": c.get("first_ms"),
                    "abstain": c.get("abstain"),
                    "stats": ({k: (round(v, 4) if isinstance(v, float)
                                   else v)
                               for k, v in (c.get("stats") or {}).items()
                               if k != "outlier_values"}
                              if c.get("stats") else None),
                    "outlier_values": [round(v, 4) for v in
                                       (c.get("stats") or {}).get(
                                           "outlier_values", [])],
                    # raw samples kept for audit; ~30 floats/cell
                    "samples": [round(v, 4) for v in
                                c.get("samples", [])],
                }
            sessions_out.append({
                "session_index": r["session_index"],
                "role": r["role"],
                "seed": r["seed"],
                "pid": r["pid"],
                "provenance": r["provenance"],
                "cells": cells_out,
            })
        # Guard from pooled protocol-session medians only.
        proto_pool = {}
        for r in sres:
            if r["role"] != "protocol":
                continue
            for name, c in r["cells"].items():
                proto_pool.setdefault(name, []).extend(
                    c.get("samples", []))
        med = {k: summarize_samples(v).get("p50")
               for k, v in proto_pool.items()}
        a0, ap1, av = med.get("A0P0"), med.get("A1P1"), med.get(
            "vendor_full")
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
            "pooled": pooled_stats,
            "per_session_p50": {
                name: {
                    r["session_index"]: round(
                        ((r["cells"].get(name) or {}).get("stats")
                         or {}).get("p50", float("nan")), 4)
                    for r in sres
                    if (r["cells"].get(name) or {}).get("stats")
                }
                for name in CELL_NAMES
                if any((r["cells"].get(name) or {}).get("stats")
                       for r in sres)
            },
            "sessions": sessions_out,
            "guard": guard,
        })

    commits = {r["provenance"]["commit"] for r in results}
    single_commit = len(commits) == 1
    result = {
        "kind": "issue-3-hybrid-canary",
        "schema": 2,
        "commit": results[0]["provenance"]["commit"],
        "all_sessions_same_commit": single_commit,
        "dirty": any(r["provenance"]["dirty"] for r in results),
        "table_env_hash": table.get("env_hash"),
        "live_env_hash": env_hash,
        "profile_hash": table.get("profile_hash"),
        "sessions_protocol": args.sessions,
        "confirmation_session_added": bool(need_confirm_shapes),
        "trials": args.trials,
        "warmup": args.warmup,
        "seed_base": args.seed_base,
        "err_max": ERR_MAX,
        "process_model": (
            "each session is an independent spawned process (fresh "
            "Python allocator, JIT cache, and runtime state); the "
            "parent gates freshness once, every worker re-gates before "
            "any launch"),
        "protocol": (
            f"{args.sessions} independent process sessions, "
            f"{args.trials} interleaved steady trials each (random "
            "cell order per trial, seeded), explicit synchronize "
            "around every timed call. HybridPlans are re-constructed "
            "per session so twiddle/staging/row caches are cold at "
            "session start; first call per cell pays JIT and is "
            "correctness-gated before warmup. Statistics: pooled and "
            "per-session p50 / MAD / p95 / max / outlier count "
            "(outliers > p50+3*MAD); every outlier value is listed. "
            "A confirmation session is added automatically when the "
            "A1P1-vs-vendor guard passes in the protocol sessions. "
            "Guard evaluation uses the 2% threshold against A0P0 "
            "(in-table incumbent) and vendor_full, on protocol-session "
            "pooled medians. A1P1 is an experimental canary: even a "
            "guard pass does not promote it into the selector table "
            "(cost model has no A1 family)."
        ),
        "shapes": shapes_out,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} (commit {result['commit'][:8]} "
          f"dirty={result['dirty']} env={env_hash} "
          f"confirm={need_confirm_shapes})")


if __name__ == "__main__":
    main()
