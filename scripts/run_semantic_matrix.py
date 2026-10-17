#!/usr/bin/env python3
"""WP2 unified semantic release matrix (issue #6 PR-D).

One manifest drives raw rows, summary tables and the immutable
release snapshot. For every case x semantic the matrix records four
contenders under one correctness-first, interleaved protocol:

* ``incumbent`` — the M0/M4 baseline plan (``plan_fft`` abstains to M0);
* ``selector``  — the frozen production table plan;
* ``oracle``    — a bounded measured candidate set (m0/m1/m2/m4/m5);
* ``vendor``    — ``torch.fft`` with the same semantic (unsupported /
  error preserved verbatim, never enters a speedup).

Semantics: ``c2c-forward``, ``c2c-inverse``, ``r2c``, ``c2r`` (PR-B/PR-C
contracts; non-forward semantics wrap the same core through the entry
layer, so the bounded oracle measures the core set through the same
wrapper).

Timing tiers are kept strictly separate — plan creation, plan-reuse
steady API wall, one-shot latency, H2D->FFT->D2H end-to-end — and
never labelled as kernel time (no device counters on this stack).

Protocol: >= 3 independent process sessions, >= 30 trials/case,
randomized interleave within each trial round, correctness gate before
any timing, search (oracle candidates) measured in the same
confirmation sessions but attributed per candidate.

Usage::

    CUDA_VISIBLE_DEVICES=6 python3 scripts/run_semantic_matrix.py \
        --sessions 3 --trials 30 --scope main \
        --out results/p800/semantic_matrix
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import random
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_ID = ("semantic-matrix-v1:plan-reuse:interleave:"
               "correctness-first")
TIMING_SCOPE = ("api wall time (dispatch+alloc+layout+sync), ms; "
                "tiers: plan_creation / steady / one_shot / e2e")
SEMANTICS = ("c2c-forward", "c2c-inverse", "r2c", "c2r")
TIER_STEADY = "steady"
TIER_ONE_SHOT = "one_shot"
TIER_E2E = "e2e"

SCOPES = {
    "main": [(n, b) for n in (64, 128, 256, 512, 1024, 2048)
             for b in (1, 4, 16, 64)],
    "apps": [(1024, 16), (1024, 64), (2048, 16), (2048, 64),
             (1024, 256), (2048, 256)],   # OFDM + radar workload proxies
    "large": [(4096, 1), (4096, 16), (16384, 1), (16384, 16),
              (65536, 1), (65536, 16)],
    # issue #8 PR-G step 2: shapes owning the fixed-domain abstain cells
    # whose v2 bounded oracle measured clearly faster (regret >= 0.10,
    # i.e. oracle >= 1.10x the incumbent fallback), plus N2048B256 as
    # background evidence (same 2x-class gap, no gate).  Holdout rows in
    # this scope are confirmation evidence only -- never fitting input.
    "regret": [(64, 1), (512, 64), (2048, 4), (1024, 256), (2048, 256),
               (16384, 1), (16384, 16)],
}


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def _pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    return xs[min(len(xs) - 1, max(0, int(q * len(xs))))]


def _git(*args):
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True).strip()


def _mad(xs):
    m = _median(xs)
    if m is None:
        return None
    return _median([abs(x - m) for x in xs])


def _err_vs_np(got, ref_np):
    import numpy as np
    g = got.detach().cpu().numpy()
    delta = np.maximum(np.abs(g.real - ref_np.real),
                       np.abs(g.imag - ref_np.imag))
    scale = np.abs(ref_np).max()
    return float(delta.max() / max(float(scale), 1e-12))


def _err_real_vs_np(got, ref_np):
    import numpy as np
    g = got.detach().cpu().numpy()
    scale = np.abs(ref_np).max()
    return float(np.abs(g - ref_np).max() / max(float(scale), 1e-12))


def _semantic_of(name):
    transform_type = "r2c" if name == "r2c" else (
        "c2c" if name.startswith("c2c") else "c2r")
    direction = ("inverse" if name in ("c2c-inverse", "c2r")
                 else "forward")
    return transform_type, direction


def _norm_of(name):
    return "backward"  # fixed release norm (torch default)


def _make_input(kind, n, b, semantic, seed):
    import numpy as np
    import torch
    rng = np.random.default_rng(seed)
    if semantic in ("r2c", "c2r"):
        a = rng.standard_normal((b, n))
        if semantic == "c2r":
            # input contract: complex64 half spectrum of the same
            # seeded real signal the reference inverts
            half = np.fft.rfft(a, axis=-1).astype(np.complex64)
            if b == 1:
                half = half[0]
            return torch.from_numpy(half).to("cuda")
        a = a.astype(np.float32)
        if b == 1:
            a = a[0]
        return torch.from_numpy(a).to("cuda")
    a = (rng.standard_normal((b, n))
         + 1j * rng.standard_normal((b, n))).astype(np.complex64)
    if b == 1:
        a = a[0]
    return torch.from_numpy(a).to("cuda")


def _ref_np(n, b, semantic, seed, norm="backward"):
    import numpy as np
    rng = np.random.default_rng(seed)
    if semantic in ("r2c", "c2r"):
        a = rng.standard_normal((b, n))
        if semantic == "r2c":
            return np.fft.rfft(a, axis=-1, norm=norm)
        half = np.fft.rfft(a, axis=-1)
        ref = np.fft.irfft(half, n=n, axis=-1, norm=norm)
        if norm in ("none", "forward"):
            ref = ref * n
        return ref
    a = (rng.standard_normal((b, n))
         + 1j * rng.standard_normal((b, n)))
    if semantic == "c2c-forward":
        if norm == "none":
            norm = "backward"  # numpy backward == unscaled forward
        return np.fft.fft(a, axis=-1, norm=norm)
    if norm == "ortho":
        return np.fft.ifft(a, axis=-1, norm="ortho")
    ref = np.fft.ifft(a, axis=-1, norm="backward")
    if norm in ("none", "forward"):
        ref = ref * n   # ours unscales the inverse for these norms
    return ref


def _vendor_call(x, semantic, n, b):
    import torch
    norm = _norm_of(semantic)
    if semantic == "c2c-forward":
        return torch.fft.fft(x, dim=-1, norm=norm)
    if semantic == "c2c-inverse":
        return torch.fft.ifft(x, dim=-1, norm=norm)
    if semantic == "r2c":
        return torch.fft.rfft(x, dim=-1, norm=norm)
    return torch.fft.irfft(x, n=n, dim=-1, norm=norm)


def _entry_plan(direction, n, b, semantic, profile_hash, table):
    from src.xpu.entry import plan_fft
    transform_type, transform_direction = _semantic_of(semantic)
    return plan_fft(direction, n, batch=b,
                    profile_hash=profile_hash, table=table,
                    transform_type=transform_type,
                    transform_direction=transform_direction,
                    norm=_norm_of(semantic))


def _oracle_plans(n, b, direction, semantic, block):
    """Bounded measured candidate set, each as an executable plan."""
    from src.xpu.entry import ExecutionPlan, REAL_COMPAT_MARK
    transform_type, transform_direction = _semantic_of(semantic)
    common = dict(n=n, batch=b, direction=direction, precision="fp32",
                  selection_source="oracle-candidate",
                  reason="bounded oracle set", fallback=False,
                  transform_type=transform_type,
                  transform_direction=transform_direction,
                  norm=_norm_of(semantic))
    compat = ({"real_compat": REAL_COMPAT_MARK}
              if transform_type in ("r2c", "c2r") else {})
    plans = []

    def add(name, executed, params=None, blk=None):
        plans.append(ExecutionPlan(
            **common, executed=executed, block=blk or block,
            params={**(params or {}), "oracle_name": name, **compat}))

    if b == 1:
        add("m0_dit", "fft_dit")
        add("m0_dif", "fft_dif")
        add("m1_fused_t64", "fft_dit_fused", {"tile": 64})
        add("m2_dit_t64", "fft_dit_matrix", {"tile": 64, "rows": 8})
        add("m2_dif_t64", "fft_dif_matrix", {"tile": 64, "rows": 8})
        add("m5_r4_dit", "fft_dit_radix4")
    else:
        add("m0_dit", "fft_dit_batch", {"variant": "folded"})
        add("m4_spread", "fft_dit_batch", {"variant": "spread"})
        add("m4_folded", "fft_dit_batch", {"variant": "folded"})
        add("m2_dit_t64", "fft_dit_matrix_batch",
            {"tile": 64, "rows": 8, "variant": "spread"})
        add("m2_dif_t64", "fft_dif_matrix_batch",
            {"tile": 64, "rows": 8, "variant": "spread"})
        add("m5_r4_dit", "fft_dit_radix4_batch", {"variant": "spread"})
    return plans


def _timed_steady(fns, warmup, trials, seed):
    """Interleaved randomized trials across contenders (one call each)."""
    from src.xpu.runner import _sync
    import torch  # noqa: F401  (device stack ready via runners)
    for fn in fns.values():
        for _ in range(warmup):
            fn()
        _sync()
    order = list(fns)
    rng = random.Random(seed)
    samples = {k: [] for k in fns}
    for _ in range(trials):
        rng.shuffle(order)
        for key in order:
            t0 = time.perf_counter()
            fns[key]()
            _sync()
            samples[key].append((time.perf_counter() - t0) * 1e3)
    return samples


def _timed_once(fn, warmup, trials, seed):
    from src.xpu.runner import _sync
    for _ in range(warmup):
        fn()
    _sync()
    rng = random.Random(seed)
    xs = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        _sync()
        xs.append((time.perf_counter() - t0) * 1e3)
    return xs


def _row(session, case_label, semantic, contender, tier, trial, **kw):
    from src.xpu.config import DeviceProfile
    row = {
        "session": session,
        "label": f"{case_label}:{semantic}",
        "trial": trial,
        "candidate": contender,
        "tier": tier,
        "protocol_id": PROTOCOL_ID,
        "profile_hash": DeviceProfile.load().hash(),
        "timed_scope": TIMING_SCOPE,
        "processing_unit": kw.get("processing_unit"),
        "flow_id": kw.get("flow_id"),
        "params": kw.get("params"),
        "api_ms": kw.get("api_ms"),
        "correct": kw.get("correct"),
        "err": kw.get("err"),
        "status": kw.get("status", "ok"),
        "reason": kw.get("reason"),
        "notes": kw.get("notes"),
    }
    return row


def run_case(session, n, b, semantic, trials, warmup, seed, table,
             raw, e2e_trials, one_shot_trials):
    import torch
    from src.xpu.entry import execute_fft_plan, run_fft
    from src.xpu.config import DeviceProfile

    case_label = f"N{n}B{b}"
    direction = "dit"
    prof = DeviceProfile.load().hash()
    norm = _norm_of(semantic)
    transform_type, transform_direction = _semantic_of(semantic)

    x = _make_input("random", n, b, semantic, seed)
    ref = _ref_np(n, b, semantic, seed, norm)
    if semantic == "r2c":
        err_fn = _err_vs_np
    elif semantic == "c2r":
        err_fn = _err_real_vs_np
    else:
        err_fn = _err_vs_np

    contenders = {}   # name -> (plan, fn)
    incumbent = _entry_plan(direction, n, b, semantic, None, table)
    selector = _entry_plan(direction, n, b, semantic, prof, table)
    contenders["incumbent"] = (incumbent,
                               (lambda p=incumbent: execute_fft_plan(p, x)))
    contenders["selector"] = (selector,
                              (lambda p=selector: execute_fft_plan(p, x)))
    for plan in _oracle_plans(n, b, direction, semantic,
                              block=incumbent.block or 1024):
        name = plan.params["oracle_name"]
        contenders[f"oracle:{name}"] = (
            plan, (lambda p=plan: execute_fft_plan(p, x)))
    # vendor
    contenders["vendor"] = (
        None, (lambda: _vendor_call(x, semantic, n, b)))

    # correctness gate first
    errs = {}
    for name, (plan, fn) in list(contenders.items()):
        try:
            y = fn()
            err = err_fn(y, ref)
            errs[name] = err
            if err > 1e-4:
                raw.append(_row(
                    session, case_label, semantic, name, "correctness",
                    0, api_ms=None, correct=False, err=err,
                    status="wrong-value",
                    reason=f"err={err:.3e} > 1e-4"))
                del contenders[name]
        except Exception as e:  # noqa: BLE001
            raw.append(_row(
                session, case_label, semantic, name, "correctness", 0,
                api_ms=None, correct=False, err=None,
                status="vendor-unsupported"
                if name == "vendor" else "error",
                reason=f"{type(e).__name__}: {e}"))
            del contenders[name]

    if not contenders:
        return

    # steady interleaved timing
    fns = {name: fn for name, (_, fn) in contenders.items()}
    samples = _timed_steady(fns, warmup, trials,
                            seed=seed * 1000 + 1)
    for name, xs in samples.items():
        for i, ms in enumerate(xs):
            plan = contenders[name][0]
            raw.append(_row(
                session, case_label, semantic, name, TIER_STEADY, i,
                api_ms=ms, correct=True, err=errs.get(name),
                flow_id=plan.flow_id if plan else "vendor",
                processing_unit=(plan.processing_unit if plan
                                 else "vendor-torch-fft"),
                params={"selection_source": plan.selection_source
                        if plan else "vendor",
                        "fallback": plan.fallback if plan else False}))

    # one-shot + e2e for incumbent/selector only (overhead tiers)
    for name in ("incumbent", "selector"):
        if name not in contenders:
            continue
        direction_arg = direction
        try:
            xs = _timed_once(
                lambda: run_fft(
                    x, direction_arg, profile_hash=(
                        prof if name == "selector" else None),
                    table=table, transform_type=transform_type,
                    transform_direction=transform_direction,
                    norm=norm),
                max(2, warmup // 2), one_shot_trials,
                seed=seed * 1000 + 2)
            for i, ms in enumerate(xs):
                raw.append(_row(
                    session, case_label, semantic, f"{name}:one_shot",
                    TIER_ONE_SHOT, i, api_ms=ms, correct=True,
                    err=errs.get(name),
                    notes="includes per-call plan/dispatch overhead"))
        except Exception as e:  # noqa: BLE001
            raw.append(_row(
                session, case_label, semantic, f"{name}:one_shot",
                TIER_ONE_SHOT, 0, api_ms=None, correct=False,
                status="error", reason=f"{type(e).__name__}: {e}"))
        try:
            x_cpu = x.cpu().contiguous()
            plan = contenders[name][0]

            def e2e(plan=plan):
                xd = x_cpu.to("cuda")
                y = execute_fft_plan(plan, xd)
                return y.cpu()

            xs = _timed_once(e2e, max(2, warmup // 2), e2e_trials,
                             seed=seed * 1000 + 3)
            for i, ms in enumerate(xs):
                raw.append(_row(
                    session, case_label, semantic, f"{name}:e2e",
                    TIER_E2E, i, api_ms=ms, correct=True,
                    err=errs.get(name),
                    notes="H2D copy + execute + D2H copy"))
        except Exception as e:  # noqa: BLE001
            raw.append(_row(
                session, case_label, semantic, f"{name}:e2e",
                TIER_E2E, 0, api_ms=None, correct=False,
                status="error", reason=f"{type(e).__name__}: {e}"))

    # plan-creation timing (selector only; host path)
    try:
        import time as _t
        xs = []
        for i in range(5):
            t0 = _t.perf_counter()
            _entry_plan(direction, n, b, semantic, prof, table)
            xs.append((_t.perf_counter() - t0) * 1e3)
        raw.append(_row(
            session, case_label, semantic, "selector:plan_creation",
            "plan_creation", 0, api_ms=_median(xs), correct=True,
            err=errs.get("selector"), notes=f"samples={xs}"))
    except Exception as e:  # noqa: BLE001
        raw.append(_row(
            session, case_label, semantic, "selector:plan_creation",
            "plan_creation", 0, api_ms=None, correct=False,
            err=errs.get("selector"), status="error",
            reason=f"{type(e).__name__}: {e}"))


def run_child(payload):
    session, cases, trials, warmup, seed, out_dir, e2e_trials, \
        one_shot_trials = payload
    from src.xpu.selector import load_table
    table = load_table(str(ROOT / "config" / "selector_table.json"))
    raw_path = Path(out_dir) / f"raw_session{session}.jsonl"
    # per-session seeded interleaving of case order (and semantic
    # order): no session measures the grid in the same sequence
    order = list(cases)
    rnd = random.Random(seed + session * 10007)
    rnd.shuffle(order)
    sems = list(SEMANTICS)
    rnd.shuffle(sems)
    with open(raw_path, "w") as raw_f:
        for n, b in order:
            for semantic in sems:
                rows = []
                try:
                    run_case(session, n, b, semantic, trials, warmup,
                             seed + session * 10007 + n * 13 + b, table,
                             rows, e2e_trials, one_shot_trials)
                except Exception as e:  # noqa: BLE001
                    rows.append(_row(
                        session, f"N{n}B{b}", semantic, "case", "error",
                        0, api_ms=None, correct=False, status="error",
                        reason=f"{type(e).__name__}: {e}"))
                for r in rows:
                    raw_f.write(json.dumps(r) + "\n")
                raw_f.flush()
    return raw_path


def _aggregate(out_dir, cases, trials, warmup, sessions):
    from src.xpu.manifest_contract import (
        REQUIRED_ROW_KEYS, rows_sha256)

    raw_rows = []
    for s in range(sessions):
        p = Path(out_dir) / f"raw_session{s}.jsonl"
        if p.is_file():
            with open(p) as f:
                raw_rows += [json.loads(l) for l in f if l.strip()]
    steady = {}
    meta = {}
    failures = []
    for r in raw_rows:
        missing = [k for k in REQUIRED_ROW_KEYS
                   if k not in r or r[k] in (None, "")]
        if missing and r.get("status") == "ok":
            failures.append({"kind": "missing-row-field",
                             "row": {k: r.get(k) for k in
                                     ("label", "candidate", "tier")},
                             "missing": missing})
        key = (r["label"], r["candidate"], r.get("tier"))
        if r.get("tier") == TIER_STEADY and r.get("status") == "ok" \
                and r.get("api_ms") is not None:
            steady.setdefault(key, []).append(r["api_ms"])
            meta[key] = r
        if r.get("status") in ("wrong-value", "vendor-unsupported",
                               "error"):
            failures.append({
                "kind": r["status"], "label": r["label"],
                "candidate": r["candidate"], "tier": r.get("tier"),
                "reason": r.get("reason"),
            })

    cases_out = []
    for n, b in cases:
        case_label = f"N{n}B{b}"
        for semantic in SEMANTICS:
            label = f"{case_label}:{semantic}"
            med = {}
            mad = {}
            for (lab, cand, tier), xs in steady.items():
                if lab == label and cand.startswith(
                        ("incumbent", "selector", "oracle:", "vendor")):
                    med[cand] = _median(xs)
                    mad[cand] = _mad(xs)
            oracle_keys = [k for k in med if k.startswith("oracle:")]
            oracle = min((med[k] for k in oracle_keys), default=None)
            oracle_name = (min(oracle_keys, key=lambda k: med[k])
                           if oracle_keys else None)
            sel = med.get("selector")
            inc = med.get("incumbent")
            ven = med.get("vendor")
            entry = {
                "label": label, "N": n, "batch": b,
                "semantic": semantic,
                "selector_ms": sel, "incumbent_ms": inc,
                "vendor_ms": ven, "oracle_ms": oracle,
                "selector_mad_ms": mad.get("selector"),
                "incumbent_mad_ms": mad.get("incumbent"),
                "vendor_mad_ms": mad.get("vendor"),
                "oracle_mad_ms": (
                    min((mad[k] for k in oracle_keys), default=None)
                    if oracle_keys else None),
                "oracle_candidate": oracle_name,
                "selector_regret": (
                    (sel / oracle - 1.0)
                    if (sel and oracle) else None),
                "speedup_vs_vendor": (
                    (ven / sel) if (sel and ven) else None),
                "speedup_vs_incumbent": (
                    (inc / sel) if (sel and inc) else None),
                "guard_vs_incumbent": (
                    ((inc - sel) / inc) if (sel and inc) else None),
                "vendor_status": ("ok" if ven else (
                    "unsupported" if any(
                        f.get("label") == label
                        and f.get("candidate") == "vendor"
                        for f in failures) else "missing")),
                "selector_source": meta.get(
                    (label, "selector", TIER_STEADY), {}
                ).get("params", {}).get("selection_source"),
                "sessions_with_data": len({
                    r["session"] for r in raw_rows
                    if r["label"] == label
                    and r["candidate"] == "selector"
                    and r.get("tier") == TIER_STEADY
                    and r.get("status") == "ok"}),
            }
            entry["guard_pass_2pct"] = (
                bool(entry["guard_vs_incumbent"] is not None
                     and entry["guard_vs_incumbent"] >= 0.02)
                if entry["selector_source"] in ("table", "composition")
                else None)
            cases_out.append(entry)

    supported = [c for c in cases_out
                 if c["speedup_vs_vendor"] is not None]
    geomean = (math.exp(sum(math.log(c["speedup_vs_vendor"])
                            for c in supported) / len(supported))
               if supported else None)
    wins = sum(1 for c in supported if c["speedup_vs_vendor"] > 1.02)
    ties = sum(1 for c in supported
               if 0.98 <= c["speedup_vs_vendor"] <= 1.02)
    losses = sum(1 for c in supported if c["speedup_vs_vendor"] < 0.98)
    worst = min((c["speedup_vs_vendor"] for c in supported),
                default=None)
    regrets = [c["selector_regret"] for c in cases_out
               if c["selector_regret"] is not None]
    summary = {
        "protocol_id": PROTOCOL_ID,
        "cases": cases_out,
        "metrics": {
            "vendor_supported_cases": len(supported),
            "win": wins, "tie": ties, "loss": losses,
            "geomean_speedup_vs_vendor": geomean,
            "worst_speedup_vs_vendor": worst,
            "mean_selector_regret": (
                sum(regrets) / len(regrets) if regrets else None),
            "max_selector_regret": max(regrets, default=None),
        },
        "failures": failures,
        "raw_sha256": hashlib.sha256(
            json.dumps(raw_rows, sort_keys=True,
                       separators=(",", ":")).encode()).hexdigest(),
        "rows_sha256_check": rows_sha256({"rows": raw_rows}),
    }
    (Path(out_dir) / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    cols = ["label", "semantic", "N", "batch", "selector_ms",
            "selector_mad_ms", "incumbent_ms", "oracle_ms",
            "oracle_candidate", "vendor_ms", "vendor_mad_ms",
            "vendor_status", "speedup_vs_vendor",
            "selector_regret", "guard_pass_2pct",
            "selector_source", "sessions_with_data"]
    with open(Path(out_dir) / "summary.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for c in cases_out:
            f.write(",".join(
                str(c.get(k)) for k in cols) + "\n")
    _write_figures(out_dir, summary)
    return summary


def _write_figures(out_dir, summary):
    """speedup-vs-vendor figure driven by the same summary/manifest."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    import numpy as np
    sems = list(SEMANTICS)
    fig, axes = plt.subplots(len(sems), 1, figsize=(13, 2.6 * len(sems)))
    for ax, sem in zip(np.atleast_1d(axes), sems):
        cs = [c for c in summary["cases"] if c["semantic"] == sem]
        sp = [(c["speedup_vs_vendor"]
               if c["speedup_vs_vendor"] else float("nan"))
              for c in cs]
        colors = ["#2a7" if (s or 0) >= 1.02 else
                  ("#c72" if (s or 0) < 0.98 else "#888")
                  for s in sp]
        ax.bar(range(len(cs)), sp, color=colors, width=0.85)
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_ylabel(f"{sem}\nspeedup vs vendor")
        ax.set_xticks(range(len(cs)))
        ax.set_xticklabels([c["label"].split(":")[0] for c in cs],
                           rotation=90, fontsize=6)
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle(
        f"semantic matrix {summary['protocol_id']}: "
        f"win/tie/loss="
        f"{summary['metrics']['win']}/"
        f"{summary['metrics']['tie']}/"
        f"{summary['metrics']['loss']}")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(out_dir) / "speedup_vs_vendor.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def _rows_sha256(out_dir):
    h = hashlib.sha256()
    for name in sorted(Path(out_dir).glob("raw_session*.jsonl")):
        h.update(name.read_bytes())
    return h.hexdigest()


def _write_snapshot(out_dir, release_id, manifest, env, args, cases):
    snap = ROOT / "results" / "p800" / "releases" / release_id
    if snap.exists():
        raise SystemExit(f"release snapshot already exists (immutable): "
                         f"{snap}")
    snap.mkdir(parents=True)
    manifest = dict(manifest)
    manifest["result_sha256"] = _rows_sha256(out_dir)
    if "source_hash" not in manifest:
        manifest["source_hash"] = _git("rev-parse", "HEAD")
    (snap / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (snap / "environment.json").write_text(
        json.dumps(env, indent=2, sort_keys=True) + "\n")
    (snap / "repro_command.txt").write_text(
        " ".join(sys.argv) + "\n")
    (snap / "cases.json").write_text(json.dumps(cases) + "\n")
    for name in sorted(Path(out_dir).glob("raw_session*.jsonl")):
        # rows are highly repetitive (~30x); snapshot keeps gzip'd raw
        # as the canonical evidence artifact
        with gzip.open(snap / (name.name + ".gz"), "wb") as f:
            f.write(name.read_bytes())
    for name in ("summary.json", "summary.csv",
                 "speedup_vs_vendor.png", "server_report.md"):
        src = Path(out_dir) / name
        if src.is_file():
            (snap / name).write_bytes(src.read_bytes())
    return snap


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--e2e-trials", type=int, default=10)
    ap.add_argument("--one-shot-trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--scope", choices=sorted(SCOPES) + ["all"],
                    default="main")
    ap.add_argument("--out", default=str(
        ROOT / "results" / "p800" / "semantic_matrix"))
    ap.add_argument("--snapshot", action="store_true",
                    help="write immutable results/p800/releases/"
                         "<release-id>/ snapshot")
    ap.add_argument("--release-id", default=None)
    ap.add_argument("--aggregate-only", action="store_true",
                    help="reuse existing raw sessions (no measuring)")
    ap.add_argument("--session", type=int, default=None,
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.scope == "all":
        cases = (SCOPES["main"] + SCOPES["apps"] + SCOPES["large"])
    else:
        cases = SCOPES[args.scope]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.session is not None:
        # isolated session process: freshly exec'd interpreter, no
        # shared mp Pool (pool teardown deadlocks against the device
        # runtime's listener threads); os._exit skips interpreter
        # atexit teardown where the runtime can also hang
        path = run_child((args.session, cases, args.trials, args.warmup,
                          args.seed, str(out_dir), args.e2e_trials,
                          args.one_shot_trials))
        print(f"session done: {path}", flush=True)
        os._exit(0)

    from src.xpu.config import DeviceProfile, build_manifest, git_commit
    from src.xpu.envcheck import current_env_hash, collect

    env = collect()
    env_hash = current_env_hash()
    man = build_manifest(DeviceProfile.load()).to_dict()
    man.update({
        "env_hash": env_hash,
        "protocol_id": PROTOCOL_ID,
        "timing_scope": TIMING_SCOPE,
        "selection_semantics": (
            "correctness-first, interleaved randomized contender order, "
            "3+ independent process sessions, 2% incumbent guard, "
            "oracle = bounded measured set, vendor errors preserved"),
        "warmup": args.warmup,
        "trials": args.trials,
        "sessions": args.sessions,
        "seed_base": args.seed,
        "candidates": ["incumbent", "selector", "oracle:*", "vendor"],
        "shapes": [f"N{n}B{b}:{s}" for n, b in cases
                   for s in SEMANTICS],
        "error_threshold": 1e-4,
        "repro_command": " ".join(sys.argv),
        "gm_bytes_kind": ("analytical estimate only; not used in this "
                          "matrix's reported metrics"),
        "kernel_time_status": (
            "not collected: this stack exposes reliable sync API wall "
            "time; device/kernel timers unavailable"),
        "source_hash": git_commit(),
        "toolchain": (f"python {sys.version.split()[0]} + "
                      "semantic-matrix-v1"),
    })
    (out_dir / "manifest.json").write_text(
        json.dumps(man, indent=2, sort_keys=True) + "\n")

    if not args.aggregate_only:
        # Sequential, each in its own freshly exec'd process (never
        # concurrent on one device): session independence without
        # contending for timing.
        child = [sys.executable, str(Path(__file__).resolve()),
                 "--session", "SESSION", "--scope", args.scope,
                 "--trials", str(args.trials),
                 "--warmup", str(args.warmup),
                 "--seed", str(args.seed),
                 "--e2e-trials", str(args.e2e_trials),
                 "--one-shot-trials", str(args.one_shot_trials),
                 "--out", str(out_dir)]
        for s in range(args.sessions):
            cmd = [str(s) if x == "SESSION" else x for x in child]
            subprocess.run(cmd, check=True, timeout=7200)
            print(f"session done: raw_session{s}.jsonl", flush=True)

    summary = _aggregate(out_dir, cases, args.trials, args.warmup,
                         args.sessions)
    m = summary["metrics"]
    print(f"cases={len(summary['cases'])} "
          f"win/tie/loss={m['win']}/{m['tie']}/{m['loss']} "
          f"geomean={m['geomean_speedup_vs_vendor']} "
          f"worst={m['worst_speedup_vs_vendor']} "
          f"mean_regret={m['mean_selector_regret']}")
    print(f"failures={len(summary['failures'])}")

    if args.snapshot:
        rid = args.release_id or (
            f"{time.strftime('%Y%m%d-%H%M%S')}-"
            f"{git_commit()[:8]}-semantic-matrix")
        snap = _write_snapshot(out_dir, rid, man, env, args, cases)
        (snap / "summary.json").write_text(
            (out_dir / "summary.json").read_text())
        print(f"immutable snapshot: {snap}")


if __name__ == "__main__":
    main()
