#!/usr/bin/env python3
"""Four-cell ablation: flow (A0 fused / A1 framework decomposition) x
local core (P0 triton / P1 vendor) — issue #3 review §4-§5.

Cells:

- A0P0 — production fused lowering (m2, triton local core), measured.
- A0P1 — abstain by construction (a fused local program cannot host an
  external whole-tensor core).
- A1P0 — framework decomposition + triton local stage.
- A1P1 — framework decomposition + vendor ``torch.fft.fft`` on the same
  sub-block tensors (the only legal host-callable external core here).

The vendor *whole* transform (``torch.fft.fft`` on the full signal) is
measured separately as the external baseline — never as a cell. Setup
(first call incl. compile) and execution (median, explicit synchronize)
are timed separately, and every cell passes a correctness gate against
the NumPy reference before it is timed.

Usage::

    CUDA_VISIBLE_DEVICES=<healthy card> timeout -s KILL 1800 \
      python scripts/four_cell_ablation.py \
        --shapes 1024x16,1024x64,2048x16 \
        --out results/p800/four_cell.json
"""

from __future__ import annotations

import argparse
import json
import sys
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

ERR_MAX = 1e-4  # same correctness threshold the focus protocol uses


def _parse_shapes(spec: str) -> list[tuple[int, int]]:
    out = []
    for part in spec.split(","):
        n_s, b_s = part.strip().lower().split("x")
        out.append((int(n_s), int(b_s)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shapes", default="1024x16,1024x64,2048x16")
    ap.add_argument("--tile", type=int, default=64,
                    help="decomposition tile R (framework axis; default "
                    "from the published space)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "p000" / "four_cell.json")
    args = ap.parse_args()

    import torch
    import run_release_matrix as relm
    from src.xpu import runner
    from src.xpu.selector import load_table

    # Kunlunxin stack exposes the device through the torch.cuda API
    # surface (xpucuda shim); same convention as the rest of the repo.
    if not torch.cuda.is_available():
        raise SystemExit("XPU device required for the four-cell ablation")
    torch.cuda.set_device(0)
    table = load_table()
    shapes_out = []
    cells_global: dict[str, dict] = {}

    for n, batch in _parse_shapes(args.shapes):
        x = relm._make_input(n, batch, args.seed)
        ref_np = relm._ref_np(n, batch, args.seed)
        # The shared NaN-safe CPU metric (complex ops on device are not
        # all implemented by xdnn; the protocol always diffs on host).
        ref = torch.from_numpy(ref_np).to(device=x.device, dtype=x.dtype)

        # A0P0: the production entry for this shape from the frozen table.
        entry = relm._entry_for(table, n, batch, "dit")

        def a0p0(xx):
            return runner.fft_dit_matrix_batch(
                xx, block=entry.get("block", 1024),
                tile=entry.get("params", {}).get("tile", 64),
                rows=entry.get("params", {}).get("rows", 8),
                variant=entry.get("params", {}).get("variant", "spread"))

        plan_triton = plan_hybrid(n, batch, tile=args.tile, core="triton")
        plan_vendor = plan_hybrid(n, batch, tile=args.tile, core="vendor")

        def a1p0(xx, _p=plan_triton):
            return run_hybrid(xx, _p)

        def a1p1(xx, _p=plan_vendor):
            return run_hybrid(xx, _p)

        def vendor_full(xx):
            return torch.fft.fft(xx, dim=-1)

        measured = {}
        cells_shape = {}
        for name, fn, flow, core in (
            ("A0P0", a0p0, "fused (m2 runner)", "P0 triton local"),
            ("A1P0", a1p0, "framework decomposition", "P0 triton local"),
            ("A1P1", a1p1, "framework decomposition",
             "P1 vendor torch.fft on sub-blocks"),
            ("vendor_full", vendor_full, "external baseline (not a cell)",
             "P1 vendor whole transform"),
        ):
            y = fn(x)
            err = max_rel_error(y.detach().cpu().numpy(), ref_np)
            if err >= ERR_MAX:
                cells_shape[name] = {
                    "flow": flow, "core": core,
                    "abstain": f"correctness gate failed (max_rel_err="
                               f"{err:.2e} >= {ERR_MAX:.0e})",
                }
                continue
            setup_ms, exec_ms = time_cell(fn, x, args.warmup, args.trials)
            cells_shape[name] = {
                "flow": flow, "core": core,
                "setup_ms": round(setup_ms, 4),
                "exec_ms": round(exec_ms, 4),
                "max_rel_err": err,
            }
            measured[name] = exec_ms

        # A0P1 is structural, not a runtime cell.
        cells_shape["A0P1"] = {
            "flow": "fused (m2 runner)",
            "core": "P1 vendor (whole-tensor API)",
            "abstain": "fused local program cannot host an external "
                       "whole-tensor core; the core boundary is the "
                       "whole kernel",
        }
        shapes_out.append({
            "N": n, "batch": batch, "direction": "dit",
            "tile": args.tile,
            "cells": cells_shape,
            "a0p0_speedup_of_a1": {
                k: round(measured["A0P0"] / v, 3)
                for k, v in measured.items()
                if k != "A0P0" and k != "vendor_full" and k in measured
            } if "A0P0" in measured else {},
        })
        print(f"[N{n}B{batch}_dit] " + " ".join(
            f"{k}={v.get('exec_ms', v.get('abstain', '?'))}"
            for k, v in cells_shape.items()), flush=True)

    result = {
        "kind": "issue-3-four-cell-ablation",
        "commit": relm._git("rev-parse", "HEAD"),
        "dirty": git_dirty(),
        "warmup": args.warmup,
        "trials": args.trials,
        "seed": args.seed,
        "err_max": ERR_MAX,
        "shapes": shapes_out,
        "note": (
            "A0P1 abstains by construction (fused core boundary). The "
            "vendor whole transform is an external baseline, not a cell. "
            "Setup = first call incl. compile; exec = median wall ms, "
            "explicit synchronize. A1 cells pay the framework's gathers, "
            "twiddle materialization, transposes, and digit-reversal "
            "transpose — the IO the review asked us to attribute "
            "separately."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} (commit {result['commit'][:8]} "
          f"dirty={result['dirty']})")


if __name__ == "__main__":
    main()
