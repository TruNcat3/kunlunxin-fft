#!/usr/bin/env python3
"""Release-matrix validation runner (review round 2 fixes).

Fixes over ``run_selected_sessions.py`` (that directory stays immutable
historical evidence):

* **per-candidate raw attribution** — ``m0_incumbent`` rows record
  ``selection_source=baseline-control`` with the baseline runner/block/
  flow/unit actually called; ``selected`` rows carry the full
  ``ExecutionReport``; the two never share a report;
* **plan-reuse timing** — the table is queried once per shape
  (``plan_fft``) and every timed call is ``execute_fft_plan`` (no
  per-call dispatch); one-shot ``run_fft`` overhead is measured separately
  at N=16/32 and reported as plan overhead, never as a kernel regression;
* **incumbent-block baseline** — the M0 control runs at the table entry's
  ``incumbent_block``; the 2% guard applies to plan-reuse selected vs this
  same-block incumbent;
* **real (B, N) vendor** — ``torch.fft.fft`` receives the same rank-2
  fp32 forward C2C tensor; failures are stored as ``unsupported`` with the
  original error and never enter a speedup;
* **illegal semantics are host-rejected** (inverse/r2c/fp64/dtype) and are
  not run on the device.

Canonical DIT shapes (3 independent process sessions, correctness-first,
randomized interleave within each trial round):

    small M1        N=16    B=1
    matched vendor  N=1024,2048  B=1,16,64
    batch trend     N=4096  B=1,16,64   (+spread/folded ablation at B=16)
    long            N=65536 B=1,4       (+M2 ablation at B=1)
    capacity        N=1048576 B=1       (correctness + latency; reduced trials)
    fallback        N=32    B=1         (out-of-table) and one profile mismatch

Output (results/p800/release_matrix/, immutable once written):

    manifest.json  raw_session{S}.jsonl  summary.json  summary.csv
    server_report.md

Usage::

    CUDA_VISIBLE_DEVICES=6 python scripts/run_release_matrix.py \
        --sessions 3 --trials 30 --out results/p800/release_matrix
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PROTOCOL_ID = "release-matrix-v2:plan-reuse:interleave:correctness-first:3sess"

# (N, B, direction, trials_override|None)
SHAPES = [
    (16, 1, "dit", None),
    (32, 1, "dit", None),                  # out-of-table fallback probe
    (1024, 1, "dit", None),
    (1024, 16, "dit", None),
    (1024, 64, "dit", None),
    (2048, 1, "dit", None),
    (2048, 16, "dit", None),
    (2048, 64, "dit", None),
    (4096, 1, "dit", None),
    (4096, 16, "dit", None),               # + spread/folded ablation
    (4096, 64, "dit", None),
    (65536, 1, "dit", None),               # + M2 ablation
    (65536, 4, "dit", None),
    (1048576, 1, "dit", 10),               # capacity: reduced trials
]
PROFILE_MISMATCH_SHAPE = (2048, 1, "dit", 10)
VENDOR_MAX_N = 2048


def _median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def _pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest(seed, warmup, trials):
    import torch
    from src.xpu.config import DeviceProfile
    p = DeviceProfile.load()
    table_path = os.path.join(ROOT, "config", "selector_table.json")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "device": p.device,
        "driver": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "toolchain": f"torch {torch.__version__} + torch_xmlir",
        "profile_hash": p.hash(),
        "config_hash": p.hash(),
        "selector_table_sha256": _sha256(table_path),
        "protocol_id": PROTOCOL_ID,
        "warmup": warmup,
        "trials": trials,
        "seed_base": seed,
        "timing_scope": "api wall time (dispatch+alloc+layout+sync), ms",
        "error_threshold": 1e-4,
        "error_definition": "max_rel_error = max(|got-ref|)/max(|ref|), "
                            "numpy complex128 reference",
        "selection_semantics": (
            "frozen table; search-stage median improvement guard; "
            "plan-reuse timings use execute_fft_plan after one plan_fft"),
    }


def _err_vs_np(got, ref_np):
    import numpy as np
    g = got.detach().cpu().numpy()
    scale = max(np.abs(ref_np).max(), 1e-12)
    return float(max(np.abs(g.real - ref_np.real).max(),
                     np.abs(g.imag - ref_np.imag).max()) / scale)


def _make_input(n, batch, seed):
    import numpy as np
    import torch
    rng = np.random.default_rng(seed)
    if batch == 1:
        re = rng.standard_normal(n).astype(np.float32)
        im = rng.standard_normal(n).astype(np.float32)
        return torch.complex(torch.from_numpy(re), torch.from_numpy(im)).cuda()
    re = rng.standard_normal((batch, n)).astype(np.float32)
    im = rng.standard_normal((batch, n)).astype(np.float32)
    return torch.complex(torch.from_numpy(re), torch.from_numpy(im)).cuda()


def _ref_np(n, batch, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    if batch == 1:
        re = rng.standard_normal(n).astype(np.float32)
        im = rng.standard_normal(n).astype(np.float32)
        return np.fft.fft(re + 1j * im)
    re = rng.standard_normal((batch, n)).astype(np.float32)
    im = rng.standard_normal((batch, n)).astype(np.float32)
    return np.fft.fft(re + 1j * im)


def _baseline_invoke(x, direction, batch, block):
    from src.xpu import runner
    if batch > 1:
        fn = runner.fft_dit_batch if direction == "dit" \
            else runner.fft_dif_batch
        return fn(x, block=block, variant="folded")
    fn = runner.fft_dit if direction == "dit" else runner.fft_dif
    return fn(x, block=block)


def _baseline_desc(direction, batch):
    if batch > 1:
        return {
            "executed": "fft_dit_batch" if direction == "dit"
                        else "fft_dif_batch",
            "flow_realization": "global-stage",
            "processing_unit": "radix2-scalar",
            "params": {"variant": "folded", "batch": batch},
        }
    return {
        "executed": "fft_dit" if direction == "dit" else "fft_dif",
        "flow_realization": "global-stage",
        "processing_unit": "radix2-scalar",
        "params": {},
    }


def _entry_for(table, n, batch, direction):
    for e in table.get("entries", ()):
        if (e["N"] == n and e["batch"] == batch
                and e["direction"] == direction):
            return e
    return None


def _timed_call(fn, warmup):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


def run_shape(session, n, batch, direction, trials, seed, table, raw,
              warmup, profile_override=None, label=None, extra_candidates=()):
    """One shape: correctness gates, then interleaved plan-reuse timing."""
    import torch
    from src.xpu.config import DeviceProfile
    from src.xpu.entry import execute_fft_plan, plan_fft

    tag = label or f"N{n}B{batch}"
    x = _make_input(n, batch, seed)
    ref = _ref_np(n, batch, seed)
    ph = profile_override if profile_override is not None else \
        DeviceProfile.load().hash()

    plan = plan_fft(direction, n, batch=batch, profile_hash=ph, table=table)
    entry = _entry_for(table, n, batch, direction)
    inc_block = int((entry or {}).get("incumbent_block")
                    or (entry or {}).get("block") or 1024)

    # --- correctness gates (before any timing) --------------------------
    y_sel = execute_fft_plan(plan, x)
    sel_err = _err_vs_np(y_sel, ref)
    if sel_err >= 1e-4:
        raise SystemExit(f"[{tag}] selected correctness failed: {sel_err}")
    bdesc = _baseline_desc(direction, batch)
    y_m0 = _baseline_invoke(x, direction, batch, inc_block)
    m0_err = _err_vs_np(y_m0, ref)
    if m0_err >= 1e-4:
        raise SystemExit(f"[{tag}] m0 correctness failed: {m0_err}")

    # candidate registry: label -> (callable, attribution dict)
    cands = {
        "m0_incumbent": (
            (lambda: _baseline_invoke(x, direction, batch, inc_block)),
            {"selection_source": "baseline-control",
             "executed": bdesc["executed"], "block": inc_block,
             "flow_realization": bdesc["flow_realization"],
             "processing_unit": bdesc["processing_unit"],
             "params": bdesc["params"],
             "fallback": False, "reason": "table incumbent M0 control"}),
        "selected": (
            (lambda: execute_fft_plan(plan, x)),
            {"selection_source": plan.selection_source,
             "executed": plan.executed, "block": plan.block,
             "flow_realization": plan.flow_realization,
             "processing_unit": plan.processing_unit,
             "params": plan.params,
             "fallback": plan.fallback, "reason": plan.reason}),
    }
    vend_status = None
    if n <= VENDOR_MAX_N:
        def _vendor():
            return torch.fft.fft(x, dim=-1)
        try:
            y_v = _vendor()
            v_err = _err_vs_np(y_v, ref)
            vend_status = "ok" if v_err < 1e-4 else "incorrect"
            if vend_status != "ok":
                raise SystemExit(f"[{tag}] vendor correctness failed: {v_err}")
            cands["vendor"] = (
                _vendor,
                {"selection_source": "external-baseline",
                 "executed": "torch.fft.fft", "block": None,
                 "flow_realization": None, "processing_unit": None,
                 "params": {}, "fallback": False,
                 "reason": "matched (B,N) vendor call"})
        except Exception as e:  # noqa: BLE001 - store structured unsupported
            vend_status = f"unsupported: {type(e).__name__}: {e}"
            print(f"  [{tag}] vendor unsupported: {e}", flush=True)

    for extra in extra_candidates:
        name, fn, desc = extra
        err = _err_vs_np(fn(), ref)
        if err >= 1e-4:
            raise SystemExit(f"[{tag}] extra {name} correctness failed: {err}")
        desc = dict(desc, err=err)
        cands[name] = (fn, desc)

    # --- plan-reuse interleaved timing ----------------------------------
    rng = random.Random(seed * 1000003 + session)
    samples = {k: [] for k in cands}
    for _ in range(3):  # short interleave warmup
        for fn in cands.values():
            fn[0]()
    torch.cuda.synchronize()
    rows = []
    order_names = list(cands)
    for t in range(trials):
        order = list(order_names)
        rng.shuffle(order)
        for pos, name in enumerate(order):
            fn, desc = cands[name]
            api = _timed_call(fn, warmup=0)
            samples[name].append(api)
            rows.append({
                "session": session, "trial": t, "order": pos,
                "candidate": name, "N": n, "batch": batch,
                "direction": direction, "label": tag,
                "api_ms": api, "correct": True, "status": "ok",
                "selection_source": desc["selection_source"],
                "executed": desc["executed"], "block": desc["block"],
                "flow_realization": desc["flow_realization"],
                "processing_unit": desc["processing_unit"],
                "params": desc["params"], "fallback": desc["fallback"],
                "selection_reason": desc.get("reason", ""),
                "profile_hash": ph,
            })
    with open(raw, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    m0_ms = _median(samples["m0_incumbent"])
    summary = {
        "label": tag, "N": n, "batch": batch, "direction": direction,
        "incumbent_block": inc_block,
        "m0_incumbent": {**bdesc, "block": inc_block,
                         "median_ms": m0_ms, "err": m0_err},
        "selected": {"executed": plan.executed, "block": plan.block,
                     "flow_realization": plan.flow_realization,
                     "processing_unit": plan.processing_unit,
                     "params": plan.params, "fallback": plan.fallback,
                     "selection_source": plan.selection_source,
                     "reason": plan.reason,
                     "median_ms": _median(samples["selected"]),
                     "err": sel_err},
        "vendor_status": vend_status,
        "vendor_median_ms": _median(samples.get("vendor", [])),
        "speedup_m0_over_selected": (m0_ms / _median(samples["selected"]))
        if samples.get("selected") and _median(samples["selected"]) else None,
        "speedup_m0_over_vendor": (m0_ms / _median(samples["vendor"]))
        if samples.get("vendor") and _median(samples["vendor"]) else None,
        "extra": {name: {"median_ms": _median(samples[name]),
                         "executed": cands[name][1]["executed"],
                         "block": cands[name][1]["block"],
                         "flow_realization": cands[name][1]["flow_realization"],
                         "processing_unit": cands[name][1]["processing_unit"],
                         "err": cands[name][1].get("err")}
                  for name in extra_candidates_names(extra_candidates)},
    }
    print(f"  [{tag}] sel={plan.executed}@{plan.block} fb={plan.fallback} "
          f"m0={m0_ms:.4f}ms sel_ms={summary['selected']['median_ms']:.4f} "
          f"sp={summary['speedup_m0_over_selected']:.3f}x "
          f"vend={summary['vendor_status']}", flush=True)
    return summary


def extra_candidates_names(extra_candidates):
    return [e[0] for e in extra_candidates]


def _extra_batch_variants(x, n, batch, direction, block):
    """spread/folded ablation callables at the incumbent block."""
    from src.xpu import runner
    out = []
    for v in ("spread", "folded"):
        fn = runner.fft_dit_batch if direction == "dit" \
            else runner.fft_dif_batch
        out.append((f"m4_{v}", (lambda fn=fn, v=v: fn(
            x, block=block, variant=v)),
            {"selection_source": "ablation",
             "executed": fn.__name__, "block": block,
             "flow_realization": "global-stage",
             "processing_unit": "radix2-scalar",
             "params": {"variant": v, "batch": batch},
             "fallback": False, "reason": "spread/folded ablation",
             "err": None}))
    return out


def _extra_m2(x, n, batch, direction, block):
    from src.xpu import runner
    fn = runner.fft_dit_matrix if direction == "dit" \
        else runner.fft_dif_matrix
    return [("m2_ablation", (lambda: fn(x, block=block, tile=64, rows=8)),
             {"selection_source": "ablation",
              "executed": fn.__name__, "block": block,
              "flow_realization": "matrix-resident",
              "processing_unit": "dense-dot-fft64",
              "params": {"tile": 64, "rows": 8},
              "fallback": False,
              "reason": "long-size M2 ablation", "err": None})]


def _plan_overhead_probe(session, seed, table, raw):
    """One-shot run_fft vs plan-reuse at N=16 and N=32 (plan cost column)."""
    import torch
    from src.xpu.config import DeviceProfile
    from src.xpu.entry import execute_fft_plan, plan_fft, run_fft
    ph = DeviceProfile.load().hash()
    out = []
    for n in (16, 32):
        x = _make_input(n, 1, seed)
        plan = plan_fft("dit", n, profile_hash=ph, table=table)
        reuse = _median([_timed_call(
            (lambda: execute_fft_plan(plan, x)), warmup=5)
            for _ in range(20)])
        one_shot = _median([_timed_call(
            (lambda: run_fft(x, "dit", profile_hash=ph, table=table)),
            warmup=5) for _ in range(20)])
        overhead = one_shot - reuse
        out.append({"N": n, "plan_reuse_ms": reuse, "one_shot_ms": one_shot,
                    "plan_overhead_ms": overhead,
                    "overhead_pct": overhead / reuse * 100.0})
        print(f"  [plan-overhead] N={n}: reuse={reuse:.4f}ms "
              f"one-shot={one_shot:.4f}ms overhead={overhead:.4f}ms "
              f"({overhead / reuse * 100:.1f}%)", flush=True)
        for cand, ms in (("plan_reuse", reuse), ("one_shot", one_shot)):
            with open(raw, "a") as f:
                f.write(json.dumps({
                    "session": session, "trial": -1, "order": 0,
                    "candidate": f"plan-overhead:{cand}", "N": n,
                    "batch": 1, "direction": "dit", "label": "plan-overhead",
                    "api_ms": ms, "correct": True, "status": "ok",
                    "selection_source": "diagnostic",
                    "executed": plan.executed, "block": plan.block,
                    "flow_realization": plan.flow_realization,
                    "processing_unit": plan.processing_unit,
                    "params": plan.params, "fallback": plan.fallback,
                    "selection_reason": "plan overhead probe",
                    "profile_hash": ph,
                }) + "\n")
    return out


def run_child(args):
    from src.xpu.config import DeviceProfile
    from src.xpu.selector import load_table

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    raw = os.path.join(out_dir, f"raw_session{args.session}.jsonl")
    open(raw, "w").close()
    table = load_table()
    seed = args.seed + args.session * 7919
    results = []
    print(f"session {args.session}: seed={seed}", flush=True)
    for n, b, d, tr in SHAPES:
        trials = tr or args.trials
        extras = ()
        if (n, b) == (4096, 16):
            extras = _extra_batch_variants(_make_input(n, b, seed), n, b, d,
                                           _entry_for(table, n, b, d)
                                           ["incumbent_block"])
        if (n, b) == (65536, 1):
            extras = _extra_m2(_make_input(n, b, seed), n, b, d,
                               _entry_for(table, n, b, d)["incumbent_block"])
        results.append(run_shape(args.session, n, b, d, trials, seed, table,
                                 raw, args.warmup, extra_candidates=extras))
    # profile-mismatch fallback probe
    results.append(run_shape(
        args.session, *PROFILE_MISMATCH_SHAPE[:3],
        PROFILE_MISMATCH_SHAPE[3], seed, table, raw, args.warmup,
        profile_override="f" * 16, label="fallback-profile-mismatch"))
    overhead = _plan_overhead_probe(args.session, seed, table, raw)
    with open(os.path.join(out_dir, f"summary_session{args.session}.json"),
              "w") as f:
        json.dump({"session": args.session, "shapes": results,
                   "plan_overhead": overhead}, f, indent=2)
        f.write("\n")
    print(f"session {args.session}: wrote {raw}", flush=True)


def run_parent(args):
    import torch
    out_dir = os.path.abspath(args.out)
    if os.path.exists(os.path.join(out_dir, "summary.json")) \
            and not args.overwrite:
        raise SystemExit(f"{out_dir} already holds a finished run; "
                         "pass --overwrite to replace")
    os.makedirs(out_dir, exist_ok=True)
    man = _manifest(args.seed, args.warmup, args.trials)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
        f.write("\n")
    for s in range(args.sessions):
        cmd = [sys.executable, os.path.abspath(__file__),
               "--session", str(s), "--trials", str(args.trials),
               "--warmup", str(args.warmup), "--seed", str(args.seed),
               "--out", out_dir]
        env = dict(os.environ)
        env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
        print(f"=== session {s} ===", flush=True)
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    import statistics
    per_session = []
    for s in range(args.sessions):
        with open(os.path.join(out_dir, f"summary_session{s}.json")) as f:
            per_session.append(json.load(f))
    shapes = {}
    overhead = per_session[0]["plan_overhead"]
    for ss in per_session:
        for r in ss["shapes"]:
            shapes.setdefault(r["label"], []).append(r)
    agg = []
    for label, rs in sorted(shapes.items()):
        sp = [r["speedup_m0_over_selected"] for r in rs
              if r["speedup_m0_over_selected"]]
        agg.append({
            "label": label, "N": rs[0]["N"], "batch": rs[0]["batch"],
            "direction": rs[0]["direction"],
            "incumbent_block": rs[0]["incumbent_block"],
            "selected_executed": rs[0]["selected"]["executed"],
            "selected_block": rs[0]["selected"]["block"],
            "flow_realization": rs[0]["selected"]["flow_realization"],
            "processing_unit": rs[0]["selected"]["processing_unit"],
            "fallback": rs[0]["selected"]["fallback"],
            "selection_source": rs[0]["selected"]["selection_source"],
            "sessions": len(rs),
            "m0_median_ms": _median([r["m0_incumbent"]["median_ms"]
                                     for r in rs]),
            "selected_median_ms": _median([r["selected"]["median_ms"]
                                           for r in rs]),
            "speedup_median": _median(sp),
            "speedup_all": sp,
            "vendor_status": rs[0]["vendor_status"],
            "vendor_median_ms": _median([r["vendor_median_ms"] for r in rs
                                         if r["vendor_median_ms"]]),
            "all_correct": all(r["selected"]["err"] < 1e-4
                               and r["m0_incumbent"]["err"] < 1e-4
                               for r in rs),
            "selection_reason": rs[0]["selected"]["reason"],
            "extras": rs[0]["extra"],
        })
    summary = {"manifest": man, "sessions": args.sessions,
               "shapes": agg, "plan_overhead": overhead}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    _write_csv(out_dir, agg)
    _write_report(out_dir, summary)
    print(f"aggregated {len(agg)} shapes -> {out_dir}/summary.json")


def _write_csv(out_dir, agg):
    path = os.path.join(out_dir, "summary.csv")
    cols = ["label", "N", "batch", "direction", "incumbent_block",
            "selected_executed", "selected_block", "flow_realization",
            "processing_unit", "fallback", "selection_source",
            "m0_median_ms", "selected_median_ms", "speedup_median",
            "vendor_status", "vendor_median_ms", "all_correct"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in agg:
            w.writerow(r)


def _write_report(out_dir, summary):
    m = summary["manifest"]
    lines = [
        "# Release matrix server report",
        "",
        f"commit `{m['commit'][:12]}` dirty={m['dirty']} device="
        f"{m['device']} driver={m['driver']} protocol=`{m['protocol_id']}`",
        f"profile_hash=`{m['profile_hash']}` table_sha256="
        f"`{m['selector_table_sha256'][:16]}…` warmup={m['warmup']} "
        f"trials={m['trials']} seed={m['seed_base']}",
        "",
        "| label | N | B | incumbent block | selected | sel block | "
        "flow | unit | fb | m0 ms | sel ms | speedup | vendor | correct |",
        "|---|---:|---:|---:|---|---:|---|---|---|---:|---:|---:|---|---|",
    ]
    for r in summary["shapes"]:
        sp = r["speedup_median"]
        vend = r["vendor_status"] or "n/a (N>2048)"
        if vend == "ok":
            vend = f"ok ({r['vendor_median_ms']:.4f} ms)"
        lines.append(
            f"| {r['label']} | {r['N']} | {r['batch']} "
            f"| {r['incumbent_block']} | {r['selected_executed']} "
            f"| {r['selected_block']} | {r['flow_realization']} "
            f"| {r['processing_unit']} | {r['fallback']} "
            f"| {r['m0_median_ms']:.4f} | {r['selected_median_ms']:.4f} "
            f"| {sp:.3f}x | {vend} | {r['all_correct']} |")
    lines += ["", "## Plan overhead (one-shot run_fft vs plan-reuse)", "",
              "| N | plan-reuse ms | one-shot ms | overhead ms | % |",
              "|---:|---:|---:|---:|---:|"]
    for o in summary["plan_overhead"]:
        lines.append(f"| {o['N']} | {o['plan_reuse_ms']:.4f} "
                     f"| {o['one_shot_ms']:.4f} "
                     f"| {o['plan_overhead_ms']:.4f} "
                     f"| {o['overhead_pct']:.1f}% |")
    lines += [
        "",
        "The 2% guard applies to plan-reuse `selected` vs the same-block "
        "`m0_incumbent`; one-shot plan overhead is a separate convenience-"
        "API column, never a kernel regression. Vendor rows enter speedups "
        "only when `torch.fft.fft` executed the identical (B, N) fp32 "
        "forward C2C tensor successfully; failures stay as "
        "`unsupported + error`. Speedups are session-level medians over "
        f"{summary['sessions']} independent process sessions.",
        "",
    ]
    with open(os.path.join(out_dir, "server_report.md"), "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--out", default=os.path.join(
        ROOT, "results", "p800", "release_matrix"))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--session", type=int, default=None,
                    help="internal: run one child session")
    args = ap.parse_args()
    if args.session is not None:
        run_child(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
