#!/usr/bin/env python3
"""Independent validation sessions for the frozen selector (review P1).

Protocol (review exit criteria):

* 3 independent process sessions (one child process per session);
* per shape: correctness gate first, then randomized/interleaved M0 vs the
  table-selected implementation, every trial stored raw;
* fallback shapes (out-of-table N, profile mismatch, missing profile) must
  actually execute M0 — verified by the ExecutionReport, not assumed;
* one controlled ablation at N=4096 (M0 -> flow-only M1 -> core-change
  M2/M5) with launch counts, to attribute where the gain comes from.

Output (results/p800/selected_sessions/):
  raw_session{S}.jsonl   one row per (session, shape, trial, candidate)
  manifest.json          commit/dirty/device/profile/protocol
  summary.json           per-shape medians, speedups, fallback checks
  report.md              human-readable summary (regenerable from raw)

Usage::

    CUDA_VISIBLE_DEVICES=6 python scripts/run_selected_sessions.py \
        --sessions 3 --trials 30 --out results/p800/selected_sessions
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# (N, batch, direction): the representative validation matrix from the
# review: small M1, mid M2, long M5, mature batch path, plus fallback probes.
DEFAULT_SHAPES = [
    (16, 1, "dit"),
    (2048, 1, "dit"),
    (65536, 1, "dit"),
    (4096, 16, "dit"),
    (4096, 64, "dit"),
]
FALLBACK_SHAPES = [
    (3000, 1, "dit"),      # out-of-table N
    (2048, 1, "dif"),      # profile-mismatch probe (run with a fake hash)
]
ABLATION_N = 4096
ABLATION_CANDIDATES = ("m0", "m1_t64", "m2_dit_t64", "m5_r4_dit")


def _median(xs):
    """Standard median (even sample counts average the two centers)."""
    import statistics
    xs = list(xs)
    return statistics.median(xs) if xs else None


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT,
                                   text=True).strip()


def _manifest(table_hash_note=None):
    from src.xpu.config import DeviceProfile
    p = DeviceProfile.load()
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "device": p.device,
        "profile_hash": p.hash(),
        "driver": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "correctness_threshold": 1e-4,
        "seed_base": None,
        "table_path": os.path.relpath(
            os.path.join(ROOT, "config", "selector_table.json"), ROOT),
        "protocol": (
            "3 independent process sessions; per shape correctness first, "
            "then interleaved randomized M0 vs selected; raw per-trial "
            "samples retained; 2% guard is an engineering band, not a "
            "significance claim"),
        "selection_semantics": (
            "search-stage median improvement from the frozen table; "
            "session-level speedups reported separately"),
        **({"table_note": table_hash_note} if table_hash_note else {}),
    }


def _err_vs_np(got, ref_np):
    """numpy complex128 reference; shared contract in src/xpu/error_metric."""
    from src.xpu.error_metric import max_rel_error
    return max_rel_error(got.detach().cpu().numpy(), ref_np)


def _make_input(n, batch, seed):
    import numpy as np
    import torch
    rng = np.random.default_rng(seed)
    re = rng.standard_normal((batch, n)).astype(np.float32)
    im = rng.standard_normal((batch, n)).astype(np.float32)
    x = torch.complex(torch.from_numpy(re), torch.from_numpy(im)).cuda()
    return x[0] if batch == 1 else x


def _ref_np(n, batch, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    re = rng.standard_normal((batch, n)).astype(np.float32)
    im = rng.standard_normal((batch, n)).astype(np.float32)
    return np.fft.fft(re + 1j * im)


def _baseline_name(direction, batch):
    if batch > 1:
        return "fft_dit_batch" if direction == "dit" else "fft_dif_batch"
    return "fft_dit" if direction == "dit" else "fft_dif"


def _invoke_baseline(x, direction, batch):
    from src.xpu import runner
    fn = getattr(runner, _baseline_name(direction, batch))
    if batch > 1:
        return fn(x, block=1024, variant="folded")
    return fn(x, block=1024)


def _run_shape(session, n, batch, direction, trials, seed, table, out,
               profile_override=None, label=None):
    """One shape: correctness gate, then interleaved M0 vs selected."""
    import torch
    from src.xpu.config import DeviceProfile
    from src.xpu.entry import run_fft

    torch.manual_seed(seed)
    x = _make_input(n, batch, seed)
    ref = _ref_np(n, batch, seed)

    ph = profile_override if profile_override is not None else \
        DeviceProfile.load().hash()
    # 1) selected path + report (this also performs the execution)
    y_sel, report = run_fft(x, direction, profile_hash=ph, table=table)
    sel_err = _err_vs_np(y_sel, ref)
    # 2) baseline
    y0 = _invoke_baseline(x, direction, batch)
    m0_err = _err_vs_np(y0, ref)
    thr = 1e-4
    ok = sel_err < thr and m0_err < thr
    print(f"  [{label or 'shape'}] N={n} B={batch} {direction}: "
          f"sel={report.executed} fallback={report.fallback} "
          f"sel_err={sel_err:.2e} m0_err={m0_err:.2e} "
          f"correct={ok}", flush=True)
    if not ok:
        raise SystemExit(f"correctness gate failed at N={n} B={batch}")

    # 3) interleaved timing: per trial, random order of (m0, selected)
    rng = random.Random(seed * 1000003 + session)
    m0_fn = (lambda: _invoke_baseline(x, direction, batch))
    sel_fn = (lambda: run_fft(x, direction, profile_hash=ph, table=table)[0])
    for _ in range(5):
        m0_fn()
        sel_fn()
    torch.cuda.synchronize()
    rows = []
    for t in range(trials):
        order = ["m0", "selected"]
        rng.shuffle(order)
        for pos, cand in enumerate(order):
            fn = m0_fn if cand == "m0" else sel_fn
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            api = (time.perf_counter() - t0) * 1e3
            rows.append({
                "session": session, "trial": t, "order": pos,
                "candidate": cand, "N": n, "batch": batch,
                "direction": direction, "api_ms": api,
                "correct": True, "status": "ok",
                "selection_source": report.selection_source,
                "selection_reason": report.reason,
                "fallback": report.fallback,
                "executed": report.executed,
                "block": report.block,
                "flow_realization": report.flow_realization,
                "processing_unit": report.processing_unit,
                "params": report.params,
                "profile_hash": ph,
            })
    with open(out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    m0_ms = _median([r["api_ms"] for r in rows if r["candidate"] == "m0"])
    sel_ms = _median([r["api_ms"] for r in rows
                      if r["candidate"] == "selected"])
    return {
        "N": n, "batch": batch, "direction": direction,
        "label": label or "shape",
        "executed": report.executed, "fallback": report.fallback,
        "selection_source": report.selection_source,
        "selection_reason": report.reason,
        "block": report.block,
        "flow_realization": report.flow_realization,
        "processing_unit": report.processing_unit,
        "params": report.params,
        "m0_median_ms": m0_ms, "selected_median_ms": sel_ms,
        "speedup_m0_over_selected": (m0_ms / sel_ms) if sel_ms else None,
        "m0_err": m0_err, "selected_err": sel_err, "correct": ok,
    }


def _run_ablation(session, trials, seed, out):
    """Controlled ablation at N=4096 DIT: flow-only vs core changes."""
    import math
    import torch
    from src.xpu import runner

    n, direction = ABLATION_N, "dit"
    x = _make_input(n, 1, seed)
    ref = _ref_np(n, 1, seed)
    thr = 1e-4
    calls = {
        "m0": lambda: runner.fft_dit(x, block=1024),
        "m1_t64": lambda: runner.fft_dit_fused(x, block=1024, tile=64),
        "m2_dit_t64": lambda: runner.fft_dit_matrix(x, block=1024, tile=64,
                                                    rows=8),
        "m5_r4_dit": lambda: runner.fft_dit_radix4(x, block=1024),
    }
    logn = int(math.log2(n))
    launches = {
        "m0": logn,
        "m1_t64": 2 + (logn - 6),
        "m2_dit_t64": 2 + (logn - 6),
        "m5_r4_dit": (logn + 1) // 2,
    }
    realization = {
        "m0": "global-stage",
        "m1_t64": "fused-inplace",
        "m2_dit_t64": "matrix-resident",
        "m5_r4_dit": "global-stage",
    }
    unit = {
        "m0": "radix2-scalar",
        "m1_t64": "radix2-scalar",
        "m2_dit_t64": "dense-dot-fft64",
        "m5_r4_dit": "radix4-scalar",
    }
    rows = []
    summary = []
    rng = random.Random(seed * 104729 + session)
    samples = {k: [] for k in ABLATION_CANDIDATES}
    for _ in range(5):  # warmup each candidate once through the loop
        for k in ABLATION_CANDIDATES:
            calls[k]()
    torch.cuda.synchronize()
    for t in range(trials):
        order = list(ABLATION_CANDIDATES)
        rng.shuffle(order)
        for pos, key in enumerate(order):
            t0 = time.perf_counter()
            calls[key]()
            torch.cuda.synchronize()
            api = (time.perf_counter() - t0) * 1e3
            samples[key].append(api)
            rows.append({
                "session": session, "trial": t, "order": pos,
                "candidate": f"ablation:{key}", "N": n, "batch": 1,
                "direction": direction, "api_ms": api,
                "correct": True, "status": "ok",
                "launches": launches[key],
            })
    for key in ABLATION_CANDIDATES:
        y = calls[key]()
        err = _err_vs_np(y, ref)
        if err >= thr:
            raise SystemExit(f"ablation correctness failed for {key}: {err}")
        med = _median(samples[key])
        summary.append({"candidate": key, "median_ms": med, "err": err,
                        "launches": launches[key],
                        "flow_realization": realization[key],
                        "processing_unit": unit[key]})
    with open(out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return summary


def run_child(args):
    """One session (one process)."""
    from src.xpu.config import DeviceProfile
    from src.xpu.selector import load_table

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    raw = os.path.join(out_dir, f"raw_session{args.session}.jsonl")
    open(raw, "w").close()
    table = load_table()
    ph = DeviceProfile.load().hash()
    seed = args.seed + args.session * 7919
    results = []
    print(f"session {args.session}: seed={seed}", flush=True)
    for n, b, d in DEFAULT_SHAPES:
        results.append(_run_shape(args.session, n, b, d, args.trials, seed,
                                  table, raw))
    # fallback probes: a legal-but-unmeasured pow2 N with the real profile,
    # and a profile mismatch on an in-table shape — both must execute M0.
    results.append(_run_shape(args.session, 32, 1, "dit", max(10,
                              args.trials // 3), seed, table, raw,
                              label="fallback-out-of-table"))
    results.append(_run_shape(args.session, 2048, 1, "dif", max(10,
                              args.trials // 3), seed, table, raw,
                              profile_override="f" * 16,
                              label="fallback-profile-mismatch"))
    # a non-pow2 N is rejected before launch at the validation layer; record
    # the structured outcome instead of pretending it is a timing row.
    from src.xpu.entry import run_fft as _run_fft
    try:
        _run_fft(_make_input(3000, 1, seed), "dit",
                 profile_hash=DeviceProfile.load().hash(), table=table)
        raise SystemExit("N=3000 should be rejected before launch")
    except ValueError as e:
        rejection = {
            "session": args.session, "label": "rejected-non-pow2",
            "N": 3000, "batch": 1, "direction": "dit",
            "status": "rejected-before-launch", "error": str(e),
            "correct": None,
        }
        results.append(rejection)
        with open(raw, "a") as f:
            f.write(json.dumps(rejection) + "\n")
        print(f"  [rejected-non-pow2] N=3000: {e}", flush=True)
    ablation = _run_ablation(args.session, args.trials, seed, raw)
    with open(os.path.join(out_dir, f"summary_session{args.session}.json"),
              "w") as f:
        json.dump({"session": args.session, "shapes": results,
                   "ablation": ablation}, f, indent=2)
        f.write("\n")
    print(f"session {args.session}: wrote {raw}", flush=True)


def run_parent(args):
    """Spawn one child process per session, then aggregate."""
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    man = _manifest()
    man["seed_base"] = args.seed
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
        f.write("\n")
    for s in range(args.sessions):
        cmd = [sys.executable, os.path.abspath(__file__),
               "--session", str(s), "--trials", str(args.trials),
               "--seed", str(args.seed), "--out", out_dir]
        env = dict(os.environ)
        env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
        print(f"=== session {s} ===", flush=True)
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    # aggregate
    import statistics
    per_session = []
    for s in range(args.sessions):
        with open(os.path.join(out_dir, f"summary_session{s}.json")) as f:
            per_session.append(json.load(f))
    shapes = {}
    rejections = []
    for ss in per_session:
        for r in ss["shapes"]:
            if "executed" not in r:
                rejections.append(r)
                continue
            key = (r["N"], r["batch"], r["direction"], r["label"])
            shapes.setdefault(key, []).append(r)
    agg = []
    for key, rs in sorted(shapes.items()):
        sp = [r["speedup_m0_over_selected"] for r in rs
              if r["speedup_m0_over_selected"]]
        agg.append({
            "N": key[0], "batch": key[1], "direction": key[2],
            "label": key[3],
            "executed": rs[0]["executed"],
            "fallback": rs[0]["fallback"],
            "selection_source": rs[0]["selection_source"],
            "flow_realization": rs[0]["flow_realization"],
            "processing_unit": rs[0]["processing_unit"],
            "sessions": len(rs),
            "speedup_median": _median(sp),
            "speedup_mean": statistics.fmean(sp) if sp else None,
            "all_correct": all(r["correct"] for r in rs),
            "fallback_verified": all(
                (not r["fallback"]) or r["executed"].startswith("fft_d")
                for r in rs),
            "selection_reason": rs[0]["selection_reason"],
        })
    ablation = per_session[0]["ablation"]
    ab_m0 = next(a for a in ablation if a["candidate"] == "m0")
    for a in ablation:
        a["speedup_vs_m0"] = ab_m0["median_ms"] / a["median_ms"]
    summary = {"manifest": man, "sessions": args.sessions,
               "shapes": agg, "rejections": rejections,
               "ablation": ablation}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    _write_report(out_dir, summary)
    print(f"aggregated {len(agg)} shapes -> {out_dir}/summary.json")


def _write_report(out_dir, summary):
    lines = ["# Selector validation sessions",
             "",
             f"commit `{summary['manifest']['commit'][:12]}` dirty="
             f"{summary['manifest']['dirty']} device="
             f"{summary['manifest']['device']} sessions="
             f"{summary['sessions']}", "",
             "| N | B | dir | executed | fallback | source | "
             "flow/realization | unit | speedup (m0/selected, median "
             "over sessions) | correct |",
             "|---:|---:|---|---|---|---|---|---|---:|---|"]
    for r in summary["shapes"]:
        sp = r["speedup_median"]
        lines.append(
            f"| {r['N']} | {r['batch']} | {r['direction']} | {r['executed']} "
            f"| {r['fallback']} | {r['selection_source']} "
            f"| {r['flow_realization']} | {r['processing_unit']} "
            f"| {sp:.3f}x | {r['all_correct']} |"
            if sp is not None else
            f"| {r['N']} | {r['batch']} | {r['direction']} | {r['executed']} "
            f"| {r['fallback']} | {r['selection_source']} "
            f"| {r['flow_realization']} | {r['processing_unit']} "
            f"| n/a | {r['all_correct']} |")
    lines += ["", "## Ablation (N=4096, DIT)", "",
              "| candidate | flow | unit | launches | median ms | "
              "speedup vs M0 | err |",
              "|---|---|---|---:|---:|---:|---:|"]
    for a in summary["ablation"]:
        lines.append(
            f"| {a['candidate']} | {a['flow_realization']} "
            f"| {a['processing_unit']} | {a['launches']} "
            f"| {a['median_ms']:.4f} | {a['speedup_vs_m0']:.3f}x "
            f"| {a['err']:.1e} |")
    lines += ["",
              "Speedups are session-level medians of interleaved M0/selected "
              "API wall times from independent process sessions; the 2% "
              "guard band is an engineering protection, not a significance "
              "claim. Raw per-trial rows: `raw_session*.jsonl`.",
              ""]
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=3)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--out", default=os.path.join(
        ROOT, "results", "p800", "selected_sessions"))
    ap.add_argument("--session", type=int, default=None,
                    help="internal: run one child session")
    args = ap.parse_args()
    if args.session is not None:
        run_child(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
