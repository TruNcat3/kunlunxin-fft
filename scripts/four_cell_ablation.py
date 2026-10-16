#!/usr/bin/env python3
"""Four-cell ablation + T×R factor scan — issue #3 P1-5 / P2.

Cells (flow A0 fused / A1 framework decomposition × core P0 local /
P1 vendor host API):

- A0P0 — production fused lowering (m2, triton local core), measured
  through the *live selector path* (``selector.select`` with env gate).
- A0P1 — abstain by construction (a fused local program cannot host an
  external whole-tensor core).
- A1P0 — framework decomposition + triton local stage.
- A1P1 — framework decomposition + vendor ``torch.fft.fft`` on the same
  sub-block tensors (materialized external-subtransform / host-API
  subtransform composition; experimental canary).

Timing layers (issue #3 P1-5 / review §3):

- plan_ms   — plan construction (HybridPlan caches + buffers).
- first_ms  — the *first* execution call (JIT/compile + cache fill);
              correctness is judged on this result.
- warmup    — ``--warmup`` untimed calls after the correctness gate.
- steady    — ``--trials`` timed calls (explicit synchronize), reduced
              to p50 / MAD / p95 / max / outlier count (samples > p50 +
              3*MAD; MAD==0 falls back to 1.5*p50).

Freshness (issue #3 review §2): the script refuses to measure when the
tracked tree is dirty, records full provenance (commit, env/profile
hashes, device/runtime versions, argv), and asserts the live
environment against the selector table with
``selector.assert_current_environment`` — zero kernel launches on
mismatch. A0P0 goes through ``selector.select(..., env_hash="auto")``
so the entry itself is verified by the shared gate.

A phase-level profile of the A1 pipeline (gather / stage1 core /
fused transpose+twiddle / stage2 core / scatter, with hidden-copy
audit) is recorded once per shape for the ledger.

Usage::

    CUDA_VISIBLE_DEVICES=<healthy card> timeout -s KILL 1800 \
      python scripts/four_cell_ablation.py \
        --shapes 1024x16,1024x64,2048x16 \
        --out results/p800/four_cell.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time as _time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

# adapter_hybrid / error_metric pull in torch+triton at import time;
# they are deferred until AFTER the freshness gate in main() so the
# "gate before any device-stack import" contract is fully honoured
# (review P2). git_dirty is pure subprocess and stays module-level.
from src.xpu.config import git_dirty  # noqa: E402

ERR_MAX = 1e-4  # same correctness threshold the focus protocol uses

# Bounded T×R scan from the review (T = stage-1 length, R = tile).
# 1024: 16x64 / 32x32 / 64x16; 2048: 32x64 / 64x32.
FACTOR_SCAN = {
    1024: [(16, 64), (32, 32), (64, 16)],
    2048: [(32, 64), (64, 32)],
}


def _parse_shapes(spec: str) -> list[tuple[int, int]]:
    out = []
    for part in spec.split(","):
        n_s, b_s = part.strip().lower().split("x")
        out.append((int(n_s), int(b_s)))
    return out


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def _provenance(table: dict, env_hash: str) -> dict:
    """Everything needed to judge whether a number is interpretable."""
    from src.xpu.envcheck import collect
    env = collect() or {}
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": git_dirty(),
        "argv": sys.argv,
        "table_env_hash": table.get("env_hash"),
        "live_env_hash": env_hash,
        "profile_hash": table.get("profile_hash"),
        "guard_pct": table.get("guard_pct"),
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


def _timed_cell(fn, x, ref_np, warmup: int, trials: int) -> dict:
    """Correctness on the FIRST call, then warmup, then steady stats."""
    import torch
    t0 = _time.perf_counter()
    y = fn(x)
    torch.cuda.synchronize(x.device)
    first_ms = (_time.perf_counter() - t0) * 1e3
    err = max_rel_error(y.detach().cpu().numpy(), ref_np)
    if err >= ERR_MAX:
        return {
            "abstain": f"correctness gate failed (max_rel_err={err:.2e} "
                       f">= {ERR_MAX:.0e})",
            "max_rel_err": err,
            "first_ms": round(first_ms, 4),
        }
    _, stats = time_cell(fn, x, warmup, trials)
    return {
        "first_ms": round(first_ms, 4),
        "steady": {k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in stats.items() if k != "outlier_values"},
        "outlier_values": [round(v, 4) for v in stats.get("outlier_values",
                                                          [])],
        "max_rel_err": err,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shapes", default="1024x16,1024x64,2048x16")
    ap.add_argument("--tile", type=int, default=64,
                    help="decomposition tile R for the main cells "
                    "(default from the published space)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-factor-scan", action="store_true",
                    help="skip the bounded T×R factor scan")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="measure anyway on a dirty tracked tree "
                    "(provenance will say dirty=true; never publish)")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "p000" / "four_cell.json")
    args = ap.parse_args()

    # Freshness gate BEFORE any device-stack import or tensor/kernel
    # work (review §2 / follow-up P2): the selector gate is pure
    # Python (no torch); torch / runner / release-matrix imports are
    # deferred until after the gate so an import can never initialize
    # the runtime ahead of the check.
    from src.xpu.selector import (
        assert_current_environment,
        load_table,
        select,
    )

    if git_dirty() and not args.allow_dirty:
        raise SystemExit(
            "tracked tree is dirty; commit first so the artifact "
            "provenance is meaningful (or pass --allow-dirty for a "
            "non-publishable probe)")
    table = load_table()
    env_hash = assert_current_environment(table)  # raises on mismatch

    # Device-stack imports now happen strictly after the gate.
    from src.xpu.adapter_hybrid import (
        plan_hybrid,
        plan_ledger,
        profile_hybrid,
        run_hybrid,
        summarize_samples,
        time_cell,
    )
    from src.xpu.error_metric import max_rel_error
    import torch
    import run_release_matrix as relm
    from src.xpu import runner

    if not torch.cuda.is_available():
        raise SystemExit("XPU device required for the four-cell ablation")
    torch.cuda.set_device(0)
    prov = _provenance(table, env_hash)
    shapes_out = []

    for n, batch in _parse_shapes(args.shapes):
        x = relm._make_input(n, batch, args.seed)
        ref_np = relm._ref_np(n, batch, args.seed)

        # A0P0 through the live selector path: the same env-verified
        # entry production would use (not a raw table lookup).
        sel = select("dit", n, batch, precision=table["precision"],
                     profile_hash=table["profile_hash"], table=table,
                     env_hash=env_hash)
        if sel.source != "table":
            raise SystemExit(
                f"selector abstained for N{n}B{batch} ({sel.reason}); "
                "the four-cell needs the in-table A0P0 incumbent")
        entry = {"block": sel.block, "params": sel.params}

        def a0p0(xx):
            return runner.fft_dit_matrix_batch(
                xx, block=entry.get("block", 1024),
                tile=entry.get("params", {}).get("tile", 64),
                rows=entry.get("params", {}).get("rows", 8),
                variant=entry.get("params", {}).get("variant", "spread"))

        t0 = _time.perf_counter()
        plan_triton = plan_hybrid(n, batch, tile=args.tile, core="triton")
        plan_vendor = plan_hybrid(n, batch, tile=args.tile, core="vendor")
        plan_ms = (_time.perf_counter() - t0) * 1e3

        def a1p0(xx, _p=plan_triton):
            return run_hybrid(xx, _p)

        def a1p1(xx, _p=plan_vendor):
            return run_hybrid(xx, _p)

        def vendor_full(xx):
            return torch.fft.fft(xx, dim=-1)

        cells_shape = {}
        for name, fn, flow, core in (
            ("A0P0", a0p0, "fused (m2 runner, selector path)",
             "P0 triton local"),
            ("A1P0", a1p0, "framework decomposition (materialized "
             "sub-transforms)",
             "P0 triton local"),
            ("A1P1", a1p1, "framework decomposition (materialized "
             "sub-transforms)",
             "P1 vendor host-API sub-blocks (composition canary)"),
            ("vendor_full", vendor_full,
             "external whole-transform baseline (not a cell)",
             "P1 vendor torch.fft"),
        ):
            cell = _timed_cell(fn, x, ref_np, args.warmup, args.trials)
            cell["flow"] = flow
            cell["core"] = core
            cell["plan_ms"] = (0.0 if name == "A0P0"
                               else round(plan_ms, 4))
            cells_shape[name] = cell

        # Phase profile of the A1 pipeline (diagnostic; sync between
        # phases, so slower than steady — not mixed into steady stats).
        _, phases = profile_hybrid(x, plan_vendor)
        phase_profile = {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in phases.items()
        }

        # A0P1 is structural, not a runtime cell.
        cells_shape["A0P1"] = {
            "flow": "fused (m2 runner)",
            "core": "P1 vendor (whole-tensor API)",
            "plan_ms": 0.0,
            "abstain": "fused local program cannot host an external "
                       "whole-tensor core; the core boundary is the "
                       "whole kernel",
        }

        ledger = {
            "A1P0": plan_ledger(plan_triton),
            "A1P1": plan_ledger(plan_vendor),
            "A1P1_phase_profile": phase_profile,
        }

        # Bounded T×R factor scan (P2): same correctness gate, same
        # steady protocol, no expansion of N/B space.
        factor_rows = []
        if not args.no_factor_scan:
            for t1, r in FACTOR_SCAN.get(n, []):
                row = {"T": t1, "R": r}
                for core_name, core in (("A1P0", "triton"),
                                        ("A1P1", "vendor")):
                    p = plan_hybrid(n, batch, tile=r, core=core)

                    def fn_f(xx, _p=p):
                        return run_hybrid(xx, _p)

                    cell = _timed_cell(fn_f, x, ref_np, args.warmup,
                                       args.trials)
                    row[core_name] = cell
                factor_rows.append(row)

        shapes_out.append({
            "N": n, "batch": batch, "direction": "dit",
            "tile": args.tile,
            "cells": cells_shape,
            "ledger": ledger,
            "factor_scan": factor_rows,
        })
        print(f"[N{n}B{batch}_dit] " + " ".join(
            f"{k}={v.get('steady', {}).get('p50', v.get('abstain', v.get('first_ms', '?')))}"
            for k, v in cells_shape.items()), flush=True)
        if factor_rows:
            for row in factor_rows:
                a0 = row["A1P0"].get("steady", {}).get("p50")
                a1 = row["A1P1"].get("steady", {}).get("p50")
                print(f"  T{row['T']}xR{row['R']}: A1P0={a0} A1P1={a1}",
                      flush=True)

    result = {
        "kind": "issue-3-four-cell-ablation",
        "schema": 3,
        "provenance": prov,
        "warmup": args.warmup,
        "trials": args.trials,
        "seed": args.seed,
        "err_max": ERR_MAX,
        "factor_scan_enabled": not args.no_factor_scan,
        "shapes": shapes_out,
        "note": (
            "A0P1 abstains by construction (fused core boundary). The "
            "vendor whole transform is an external baseline, not a cell. "
            "Timing layers: plan_ms (HybridPlan construction, caches "
            "filled lazily), first_ms (first execution incl. JIT/compile "
            "+ cache fill; correctness judged on this result), warmup "
            "(untimed), steady = p50/MAD/p95/max/outlier count over "
            "trials (explicit synchronize; outliers > p50+3*MAD). A1P1 "
            "is an experimental canary (materialized external-"
            "subtransform), never promoted on this artifact. Ledger "
            "records per-call sub-transform shapes, core-call counts "
            "(vendor launches opaque), analytical GM bytes for the "
            "fused pipeline, and a phase profile with hidden-copy "
            "audit. Factor scan: bounded T×R pairs, no N/B expansion. "
            "Freshness: dirty tree or env mismatch aborts with zero "
            "kernel launches; A0P0 is fetched via selector.select with "
            "the shared env gate."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} (commit {prov['commit'][:8]} "
          f"dirty={prov['dirty']} env={prov['live_env_hash']})")


if __name__ == "__main__":
    main()
