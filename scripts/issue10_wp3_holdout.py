"""issue #10 review item 5 (v2): performance holdout (>=2 independent
sessions) before any selector promotion — rewritten for PR #12 review.

Cells (holdout set fixed by the review):
  N in {1024, 2048} x B in {1, 16, 64, 256} x {c2c_forward, c2c_inverse}
  = 16 cells; paths = vendor + hybrid_vendor_p + hybrid_fused_p +
  incumbent_entry (same-protocol remeasurement, see below).

Review fixes (PR #12 comments):
  evidence-only: nothing here wires the selector; the block is named
    eligible_for_selector and selector_diff.json proves
    config/selector_table.json / src/xpu/selector.py are untouched.
    "eligible" means "meets the promotion evidence threshold" only.
  cold isolation: every (cell, path, session) runs in its OWN
    subprocess with its OWN fresh TRITON_CACHE_DIR; spawn order is
    randomized per session (seed recorded); segments are itemized:
    context / setup (incl. incumbent plan_fft_ms) / warm probes
    (vendor_plan, stage_plan, triton_jit, m0_kernel) / first_call /
    steady. one_shot_ms = setup + probe total + first_call is a
    RECONSTRUCTION of a true unsegmented one-shot (reported
    descriptively; the gate uses steady state only).
  incumbent: remeasured in this run as a 4th path with the production
    contract — plan_fft(direction="dit", profile_hash=None -> M0
    abstain, table=config/selector_table.json) + execute_fft_plan on
    the same input/segments/sessions. issue9 v61 e2e values are kept
    only as descriptive_issue9_reference (different protocol, never a
    gate input).
  stats: independent bootstrap (both warm trial arrays resampled
    independently) — NOT paired.
  input: complex64 non-zero imag, contiguous, axis=-1, norm=backward
    (fwd unscaled, inv 1/N), seeded per (session, n, b); b==1
    incumbent measured as the 1-D (N,) production contract.

Gate (code-level, steady plan-reuse only): a cell is
eligible_for_selector for a candidate iff in EVERY session: no
fail_closed, all paths correct (rel_err <= 1e-4), contract present,
and bootstrap CI lower bound of speedup > GATE vs BOTH vendor and the
same-protocol incumbent. One-shot ratios are reported separately.

Output: results/p800/issue10_capability/wp3_holdout.json
        results/p800/issue10_capability/selector_diff.json
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT = Path(__file__).resolve()
OUT = ROOT / "results/p800/issue10_capability/wp3_holdout.json"
DIFF_OUT = ROOT / "results/p800/issue10_capability/selector_diff.json"
SELECTOR_TABLE = ROOT / "config/selector_table.json"
SELECTOR_SRC = ROOT / "src/xpu/selector.py"
INCUMBENT_CSV = (ROOT / "results/p800/issue9_v61/reports/"
                 "e2e_report.csv")
# process context baseline: measured from this module import (stdlib
# only) through device-context ready inside the worker
PROCESS_T0 = time.perf_counter()
MARK = "@@HOLDOUT@@"
NS = (1024, 2048)
BS = (1, 16, 64, 256)
DIRS = (("c2c_forward", False), ("c2c_inverse", True))
SESSIONS = (1, 2)
PATHS = ("vendor", "hybrid_vendor_p", "hybrid_fused_p", "incumbent_entry")
INCUMBENT_LABEL = "incumbent_entry"
# promotion candidates (primary = ablation winner, baseline for contrast)
CANDIDATES = ("hybrid_fused_p", "hybrid_vendor_p")
BOOTSTRAP = 2000
SEED = 10_2026
SPAWN_SEED = SEED + 700
GATE = 1.02


def _wp3():
    spec = importlib.util.spec_from_file_location(
        "issue10_wp3_perf", ROOT / "scripts/issue10_wp3_perf.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _seed_for(session: int, n: int, b: int) -> int:
    return SEED + session * 100_000 + n * 10 + b


def _input_c(b: int, n: int, seed: int):
    """complex64 non-zero imag input, same construction as issue9's
    _make_input c2c branch (rng.standard_normal + 1j*standard_normal),
    kept (B, N) for every path; the incumbent path squeezes b==1."""
    import numpy as np
    import torch
    rng = np.random.default_rng(seed)
    a = (rng.standard_normal((b, n))
         + 1j * rng.standard_normal((b, n))).astype(np.complex64)
    return torch.from_numpy(a).to("cuda")


def _contract_c(x, n: int, b: int, inverse: bool) -> dict:
    """fail-closed selector-grade input contract (raises before timing)."""
    import torch
    if tuple(x.shape) != (b, n):
        raise ValueError(f"input shape {tuple(x.shape)} != {(b, n)}")
    if x.dtype != torch.complex64:
        raise ValueError(f"input dtype {x.dtype} != complex64")
    if x.stride() != (n, 1):
        raise ValueError(f"input stride {x.stride()} != contiguous {(n, 1)}")
    if not bool(torch.isfinite(x.real).all()
                and torch.isfinite(x.imag).all()):
        raise ValueError("input contains non-finite values")
    if not bool((x.imag != 0).any()):
        raise ValueError("zero imaginary part: not selector-grade input")
    return {
        "in_shape": [b, n],
        "out_shape": [b, n],
        "dtype": "complex64",
        "axis": -1,
        "in_stride": list(x.stride()),
        "layout": "interleaved complex64 (public layout); batched (B,N) "
                  "plan/execute via plan_fft(batch=b) + "
                  "execute_fft_plan; the public 1-D "
                  "validate_input/run_fft contract applies to B=1",
        "normalization": ("ifft 1/N (norm='backward')"
                          if inverse else
                          "forward unscaled (norm='backward')"),
        "semantic_lock": "vendor_last_dim_canary=PASS",
        "input_semantics": "complex64 non-zero imag, contiguous, axis=-1, "
                           "seeded per (session, n, b)",
    }


def _trials(wp3, call):
    import torch
    torch.cuda.synchronize()
    for _ in range(wp3.WARMUP):
        call()
    torch.cuda.synchronize()
    ts = []
    for _ in range(wp3.ROUNDS):
        for _ in range(wp3.TRIALS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            call()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
    return ts


def worker(a) -> int:
    import torch
    n, b, inverse, p = a.n, a.b, a.inverse, a.path
    out = {"session": a.session, "n": n, "batch": b,
           "direction": "c2c_inverse" if inverse else "c2c_forward",
           "path": p}
    stage = "import/context"
    try:
        wp3 = _wp3()
        with wp3.xpu_context(0):
            torch.cuda.synchronize()
        context_ms = (time.perf_counter() - PROCESS_T0) * 1e3

        stage = "input/contract"
        seed = _seed_for(a.session, n, b)
        x = _input_c(b, n, seed)
        wp3._vendor_canary()
        contract = _contract_c(x, n, b, inverse)

        stage = "setup"
        segs = {"context_ms": context_ms}
        xinc = x.squeeze(0) if b == 1 else x
        meta = None
        plan = None
        t0 = time.perf_counter()
        if p == "vendor":
            call, meta = wp3.run_vendor(x, inverse)
        elif p in ("hybrid_vendor_p", "hybrid_fused_p"):
            call, meta = wp3.PATHS[p](x, inverse)
        elif p == INCUMBENT_LABEL:
            from src.xpu.entry import execute_fft_plan, plan_fft
            from src.xpu.selector import load_table
            tp0 = time.perf_counter()
            table = load_table(str(SELECTOR_TABLE))
            plan = plan_fft(direction="dit", n=n, batch=b,
                            profile_hash=None, table=table,
                            transform_type="c2c",
                            transform_direction=("inverse" if inverse
                                                 else "forward"),
                            norm="backward")
            segs["plan_fft_ms"] = (time.perf_counter() - tp0) * 1e3

            def call():
                y = execute_fft_plan(plan, xinc)
                if y.dim() == 1:
                    y = y.reshape(b, n)
                return y.real, y.imag

            meta = {"launches": "production entry execute_fft_plan",
                    "gm_bytes_analytic": None,
                    "executed": plan.executed,
                    "selection_source": plan.selection_source,
                    "fallback": plan.fallback,
                    "plan_reason": str(plan.reason)[:200]}
        else:
            raise ValueError(f"unknown path {p}")
        segs["setup_ms"] = (time.perf_counter() - t0) * 1e3

        # warm probes on SCRATCH state (fresh TRITON_CACHE_DIR): cover
        # torch fft plan creation, triton JIT, M0 kernels so that
        # first_call measures the first REAL execution in plan-reuse
        # state; probe ms are itemized and summed into one_shot_ms.
        stage = "warm_probe"
        with torch.no_grad():
            if p == "vendor":
                t1 = time.perf_counter()
                y = (torch.fft.ifft(x.clone()) if inverse
                     else torch.fft.fft(x.clone()))
                del y
                torch.cuda.synchronize()
                segs["vendor_plan_probe_ms"] = (
                    time.perf_counter() - t1) * 1e3
            elif p in ("hybrid_vendor_p", "hybrid_fused_p"):
                t, r = wp3._factors(n)
                t1 = time.perf_counter()
                scr = x.clone()
                op = torch.fft.ifft if inverse else torch.fft.fft
                z1 = op(scr.view(b, r, t), dim=-1)
                z2 = op(scr.view(b, t, r), dim=-1)
                del scr, z1, z2
                torch.cuda.synchronize()
                segs["stage_plan_probe_ms"] = (
                    time.perf_counter() - t1) * 1e3
                t1 = time.perf_counter()
                fb = torch.randn((b, n), device="cuda")
                xc = torch.complex(fb, torch.zeros_like(fb))
                x_flat = torch.view_as_real(xc).reshape(-1)
                tw = torch.empty((b, r, t), dtype=torch.complex64,
                                 device="cuda")
                tw_flat = torch.view_as_real(tw).reshape(-1)
                k = (wp3.k_twiddle_fused if p == "hybrid_fused_p"
                     else wp3.k_twiddle)
                with wp3.xpu_context(0):
                    k[(b * r,)](x_flat, tw_flat, n, r, t, t,
                                1 if inverse else 0)
                    torch.cuda.synchronize()
                del fb, xc, x_flat, tw, tw_flat
                segs["triton_jit_probe_ms"] = (
                    time.perf_counter() - t1) * 1e3
            else:  # incumbent_entry: M0 kernel warm on scratch clone
                t1 = time.perf_counter()
                y = execute_fft_plan(plan, xinc.clone())
                del y
                torch.cuda.synchronize()
                segs["m0_kernel_probe_ms"] = (
                    time.perf_counter() - t1) * 1e3
        probe_total = sum(v for k, v in segs.items()
                          if k.endswith("_probe_ms"))

        stage = "first_call"
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y_re, y_im = call()
        torch.cuda.synchronize()
        segs["first_call_ms"] = (time.perf_counter() - t0) * 1e3

        stage = "correctness"
        err = float(wp3._check_batched(y_re, y_im, x, inverse))

        stage = "steady"
        trials = _trials(wp3, call)

        segs["warm_probe_total_ms"] = probe_total
        segs["one_shot_ms"] = (segs["setup_ms"] + probe_total
                               + segs["first_call_ms"])
        segs["one_shot_with_context_ms"] = (context_ms
                                            + segs["one_shot_ms"])
        out.update({
            "contract": contract,
            "meta": meta,
            "segments": segs,
            "trials": trials,
            "rel_err": err,
            "correct": bool(err <= wp3.TOL_REL),
            "seed": seed,
        })
        print(MARK + json.dumps(out))
        return 0
    except Exception as e:  # noqa: BLE001
        out["fail_closed"] = (f"{stage}: {type(e).__name__}: "
                              f"{str(e)[:200]}")
        print(MARK + json.dumps(out))
        return 0


def _spawn(session, n, b, inverse, path) -> dict | None:
    cache = tempfile.mkdtemp(prefix=f"holdout_s{session}_{path}_")
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "2"
    env["TRITON_CACHE_DIR"] = cache
    env["PYTHONUNBUFFERED"] = "1"
    try:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--worker",
             "--session", str(session), "--n", str(n), "--b", str(b),
             "--inverse", str(int(inverse)), "--path", path],
            capture_output=True, text=True, timeout=900, env=env,
            cwd=str(ROOT))
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    for line in (proc.stdout or "").splitlines():
        if line.startswith(MARK):
            try:
                return json.loads(line[len(MARK):])
            except json.JSONDecodeError:
                return None
    print(f"  worker fail n={n} b={b} inv={int(inverse)} s={session} "
          f"path={path} rc={proc.returncode} "
          f"stderr={(proc.stderr or '')[-300:]}", flush=True)
    return None


def _p50(xs):
    ys = sorted(xs)
    return ys[len(ys) // 2] if ys else None


def _mad(xs):
    m = _p50(xs)
    if m is None:
        return None
    return _p50([abs(v - m) for v in xs])


def _boot_ratio(rng, num, den, n_boot=BOOTSTRAP):
    """independent bootstrap CI of median(num)/median(den): each array
    is resampled independently (NOT a paired/per-trial resample)."""
    import numpy as np
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    if not len(num) or not len(den):
        return None
    ratio = np.empty(n_boot)
    for i in range(n_boot):
        a = num[rng.integers(0, len(num), len(num))]
        b = den[rng.integers(0, len(den), len(den))]
        ratio[i] = np.median(a) / max(np.median(b), 1e-30)
    lo, hi = np.percentile(ratio, [2.5, 97.5])
    return {"point": float(np.median(num) / max(np.median(den), 1e-30)),
            "ci95": [float(lo), float(hi)], "n_boot": n_boot,
            "bootstrap_kind": "independent"}


def _issue9_hist() -> dict:
    """descriptive only: issue9 v61 e2e/one_shot ms per label —
    different protocol, never used in any gate."""
    out = {}
    if not INCUMBENT_CSV.exists():
        return out
    with open(INCUMBENT_CSV) as f:
        for row in csv.DictReader(f):
            try:
                e2e = json.loads(row["e2e"])
                one = json.loads(row["one_shot"])
            except (KeyError, json.JSONDecodeError):
                continue
            out[row["label"]] = {
                "incumbent_e2e_ms": e2e.get("incumbent:e2e"),
                "incumbent_one_shot_ms": one.get("incumbent:one_shot"),
                "issue9_selector_e2e_ms": e2e.get("selector:e2e"),
            }
    return out


def _eligibility_block(ready, spreads, cells_total) -> dict:
    per_cand = {}
    for ch in CANDIDATES:
        per_cand[ch] = {
            "correctness_and_contract_pass": ready[ch]["correct"],
            "meets_vendor_gate": ready[ch]["vendor"],
            "meets_incumbent_same_protocol_gate": ready[ch]["incumbent"],
            "eligible_for_selector": [
                l for l in ready[ch]["vendor"]
                if l in set(ready[ch]["incumbent"])],
            "max_cross_session_spread_rel": (
                max(spreads[ch]["vendor"]) if spreads[ch]["vendor"]
                else None),
        }
    primary = per_cand[CANDIDATES[0]]
    inter = primary["eligible_for_selector"]
    return {
        "block": "eligibility (evidence-only; selector NOT wired)",
        "selector_wiring": {
            "changes_default_path": False,
            "selector_table_modified": False,
            "selector_source_modified": False,
            "note": ("eligible_for_selector means the promotion evidence "
                     "threshold is met for the listed cells; PR #12 is "
                     "evidence-only and does not touch "
                     "config/selector_table.json or src/xpu/selector.py. "
                     "Runtime wiring lands in a separate integration PR "
                     "(see selector_diff.json)."),
        },
        "stats_method": (
            f"independent bootstrap, B={BOOTSTRAP}, seed={SEED}: candidate "
            "and reference warm trial arrays resampled independently "
            "(not paired)"),
        "gate": (
            f"steady-state plan-reuse only: eligible_for_selector "
            f"requires (every session) no fail_closed, all paths correct "
            f"(rel_err <= 1e-4), contract present, and bootstrap CI "
            f"lower bound of speedup > {GATE} vs vendor AND vs the "
            f"same-protocol incumbent_entry; one-shot ratios "
            f"(one_shot_ratio_*) are reported separately and NOT gated"),
        "candidate": CANDIDATES[0],
        "eligible_for_selector": inter,
        "cells_total": cells_total,
        "max_cross_session_spread_rel": primary[
            "max_cross_session_spread_rel"],
        "release_quality_status": (
            "BLOCKED" if not inter else
            "PARTIAL: validated cells listed, rest of matrix still "
            "exploratory"),
        "candidates": per_cand,
        "issue9_policy": (
            "issue9 v61 e2e values are descriptive_issue9_reference "
            "only (different warmup/rounds machinery); the hard "
            "incumbent gate uses the same-protocol incumbent_entry "
            "path measured in this run"),
    }


def _holdout_cell_state(n, b, inverse) -> dict:
    t = json.loads(SELECTOR_TABLE.read_text())
    dstr = "inverse" if inverse else "forward"

    def applies(e):
        return (e.get("N") == n and e.get("batch") == b
                and e.get("direction") == "dit"
                and e.get("transform_type") in (None, "c2c")
                and e.get("transform_direction") in (None, dstr)
                and e.get("normalization") in (None, "backward"))

    in_entries = any(applies(e) for e in t.get("entries", []))
    in_fallbacks = any(applies(e) for e in t.get("previous_fallbacks", []))
    state = ("in-table-retained" if in_entries else
             "in-previous_fallbacks-retained" if in_fallbacks else
             "absent-abstains-to-default")
    return {"label": f"N{n}B{b}:c2c_{'inverse' if inverse else 'forward'}",
            "n": n, "batch": b, "direction": "dit",
            "transform_direction": dstr, "state": state,
            "in_previous_fallbacks": in_fallbacks}


def _selector_diff(elig: dict) -> dict:
    table_sha = hashlib.sha256(SELECTOR_TABLE.read_bytes()).hexdigest()
    src_sha = hashlib.sha256(SELECTOR_SRC.read_bytes()).hexdigest()
    t = json.loads(SELECTOR_TABLE.read_text())
    cells = [_holdout_cell_state(n, b, inv)
             for n in NS for b in BS for _, inv in DIRS]
    return {
        "schema": "issue10-selector-diff-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": ("PR #12 is evidence-only: this diff proves the "
                    "runtime selector is untouched and records where "
                    "the holdout outcome would land in the later "
                    "integration PR"),
        "wiring": "none",
        "changes_default_path": False,
        "table_path": "config/selector_table.json",
        "table_sha256": table_sha,
        "table_entries": len(t.get("entries", [])),
        "previous_fallbacks": len(t.get("previous_fallbacks", [])),
        "selector_source_sha256": src_sha,
        "new_entries": [],
        "removed_entries": [],
        "modified_entries": [],
        "holdout_cell_state": cells,
        "state_counts": {
            s: sum(1 for c in cells if c["state"] == s)
            for s in sorted({c["state"] for c in cells})},
        "eligible_cells_pending_separate_pr": {
            ch: elig["candidates"][ch]["eligible_for_selector"]
            for ch in CANDIDATES},
        "note": ("eligible_for_selector = evidence threshold met only; "
                 "the integration PR will add entries here and present "
                 "this diff with new / retained-fallback / abstain "
                 "cells"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--session", type=int, default=1)
    ap.add_argument("--n", type=int, required=False)
    ap.add_argument("--b", type=int, required=False)
    ap.add_argument("--inverse", type=int, default=0)
    ap.add_argument("--path", type=str, default="")
    a = ap.parse_args()
    if a.worker:
        return worker(a)

    import numpy as np
    t_start = time.time()
    cells = [(n, b, inv) for n in NS for b in BS
             for _, inv in DIRS]
    raw, session_meta = [], []
    total_jobs = len(SESSIONS) * len(cells) * len(PATHS)
    done = 0
    for s in SESSIONS:
        t0 = time.time()
        order = [(n, b, inv, p) for n, b, inv in cells for p in PATHS]
        idxs = np.arange(len(order))
        np.random.default_rng(SPAWN_SEED + s).shuffle(idxs)
        order = [order[int(i)] for i in idxs]
        spawn_order = [
            f"N{n}B{b}:c2c-{'inverse' if inv else 'forward'}:{p}"
            for n, b, inv, p in order]
        ok = 0
        for n, b, inv, p in order:
            r = _spawn(s, n, b, inv, p)
            if r is not None:
                raw.append(r)
                if not r.get("fail_closed"):
                    ok += 1
            done += 1
            if done % 8 == 0:
                print(f"  [{done}/{total_jobs}] elapsed "
                      f"{time.time() - t_start:.0f}s", flush=True)
        session_meta.append({
            "session": s, "jobs_ok": ok, "jobs_total": len(order),
            "elapsed_s": round(time.time() - t0, 1),
            "spawn_seed": SPAWN_SEED + s,
            "spawn_order": spawn_order,
            "isolation": "one subprocess per (cell, path) with its own "
                         "fresh TRITON_CACHE_DIR; spawn order shuffled "
                         "per session",
        })
        print(f"session {s}: {ok}/{len(order)} jobs "
              f"({time.time() - t0:.0f}s)", flush=True)

    # index raw path rows by (n, b, dir, session, path)
    idx = {}
    for r in raw:
        key = (r.get("n"), r.get("batch"), r.get("direction"),
               r.get("session"))
        idx.setdefault(key, {})[r.get("path")] = r

    i9hist = _issue9_hist()
    rng = np.random.default_rng(SEED)
    cells_out, fail_closed = [], []
    ready = {ch: {"vendor": [], "incumbent": [], "correct": []}
             for ch in CANDIDATES}
    spreads = {ch: {"vendor": [], "incumbent": []} for ch in CANDIDATES}
    for n, b, inv in cells:
        d = "c2c_inverse" if inv else "c2c_forward"
        label = f"N{n}B{b}:{d}"
        csv_label = f"N{n}B{b}:c2c-{'inverse' if inv else 'forward'}"
        c = {"label": label, "n": n, "batch": b, "direction": d,
             "sessions": {}}
        hist = i9hist.get(csv_label, i9hist.get(label, {}))
        c["descriptive_issue9_reference"] = {
            **hist, "policy": (
                "issue9 v61 protocol differs (its own warmup/rounds/"
                "plan-reuse definition); magnitude reference only, "
                "never a gate input")}
        cand_sessions = {ch: {"v": [], "i": []} for ch in CANDIDATES}
        cok = True
        contract_ok = True
        for s in SESSIONS:
            rows = idx.get((n, b, d, s), {})
            if any(p not in rows for p in PATHS):
                missing = [p for p in PATHS if p not in rows]
                fail_closed.append(f"{label}/s{s}: missing {missing}")
                cok = False
                contract_ok = False
                continue
            bad = {p: rows[p].get("fail_closed") for p in PATHS
                   if rows[p].get("fail_closed")}
            if bad:
                fail_closed.append(f"{label}/s{s}: {bad}")
                cok = False
                contract_ok = False
                continue
            contract = next((rows[p].get("contract") for p in PATHS
                             if rows[p].get("contract")), None)
            if not contract:
                contract_ok = False
            entry = {"contract": contract, "paths": {}}
            for p in PATHS:
                pr = rows[p]
                segs = pr.get("segments", {})
                entry["paths"][p] = {
                    "warm_p50_ms": _p50(pr["trials"]),
                    "warm_mad_ms": _mad(pr["trials"]),
                    "n_trials": len(pr["trials"]),
                    "rel_err": pr["rel_err"],
                    "correct": pr["correct"],
                    "segments": segs,
                    "one_shot_ms": segs.get("one_shot_ms"),
                    "one_shot_with_context_ms": segs.get(
                        "one_shot_with_context_ms"),
                    "context_ms": segs.get("context_ms"),
                    "warm_trials_ms": pr["trials"],
                    **(pr.get("meta") or {}),
                }
                if not pr["correct"]:
                    cok = False
            vrow = entry["paths"]["vendor"]
            inc_row = entry["paths"][INCUMBENT_LABEL]
            for ch in CANDIDATES:
                h = entry["paths"][ch]
                rv = vrow["warm_trials_ms"]
                rh = h["warm_trials_ms"]
                sp = _boot_ratio(rng, rv, rh)
                if sp:
                    sp["definition"] = (
                        f"median(vendor)/median({ch}) "
                        "(>1 = candidate faster)")
                    entry[f"speedup_vendor_boot:{ch}"] = sp
                    cand_sessions[ch]["v"].append(sp)
                ri = inc_row["warm_trials_ms"]
                spi = _boot_ratio(rng, ri, rh)
                if spi:
                    spi["definition"] = (
                        f"median({INCUMBENT_LABEL})/median({ch}) "
                        "(>1 = candidate faster), same-protocol "
                        "incumbent measured in this run")
                    entry[f"speedup_incumbent_same_protocol_boot:{ch}"] = spi
                    cand_sessions[ch]["i"].append(spi)
                if (vrow["one_shot_ms"] and h["one_shot_ms"]):
                    entry[f"one_shot_ratio_vendor:{ch}"] = (
                        vrow["one_shot_ms"] / h["one_shot_ms"])
                if (inc_row["one_shot_ms"] and h["one_shot_ms"]):
                    entry[f"one_shot_ratio_incumbent:{ch}"] = (
                        inc_row["one_shot_ms"] / h["one_shot_ms"])
            c["sessions"][str(s)] = entry

        correct_ok = cok
        for ch in CANDIDATES:
            vs, iss = cand_sessions[ch]["v"], cand_sessions[ch]["i"]
            if len(vs) == len(SESSIONS) and len(iss) == len(SESSIONS):
                ready[ch]["correct"].append(label)
            if len(vs) == len(SESSIONS):
                pts = [sp["point"] for sp in vs]
                spr = abs(pts[0] - pts[1]) / max(
                    sum(pts) / len(pts), 1e-30)
                spreads[ch]["vendor"].append(spr)
                okv = all(sp["ci95"][0] > GATE for sp in vs)
                if okv and correct_ok and contract_ok:
                    ready[ch]["vendor"].append(label)
            if len(iss) == len(SESSIONS):
                pts = [e["point"] for e in iss]
                spr = abs(pts[0] - pts[1]) / max(
                    sum(pts) / len(pts), 1e-30)
                spreads[ch]["incumbent"].append(spr)
                oki = all(e["ci95"][0] > GATE for e in iss)
                if oki and correct_ok and contract_ok:
                    ready[ch]["incumbent"].append(label)
            c.setdefault("cross_session_by_candidate", {})[ch] = {
                "vendor": {"points": [sp["point"] for sp in vs],
                           "spread_rel": spreads[ch]["vendor"][-1]
                           if vs else None},
                "incumbent_same_protocol": {
                    "points": [e["point"] for e in iss],
                    "spread_rel": spreads[ch]["incumbent"][-1]
                    if iss else None},
            }
        c["correct_all_sessions"] = cok
        c["contract_all_sessions"] = contract_ok
        cells_out.append(c)

    elig = _eligibility_block(ready, spreads, len(cells))
    wp3 = _wp3()
    doc = {
        "schema": "issue10-wp3-holdout-v2",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": str(SCRIPT.relative_to(ROOT)),
        "protocol": {
            "cells": [f"N{n}B{b}:{'c2c_inverse' if inv else 'c2c_forward'}"
                      for n, b, inv in cells],
            "sessions": len(SESSIONS),
            "paths": list(PATHS),
            "session_isolation": (
                "one subprocess per (cell, path, session), own fresh "
                "TRITON_CACHE_DIR; spawn order randomized per session "
                "(seed + record in sessions[].spawn_order)"),
            "segments": (
                "context (module import -> device context ready), setup "
                "(path construction; incumbent setup includes table load "
                "+ plan_fft, sub-timed as plan_fft_ms), warm probes on "
                "scratch (vendor_plan_probe / stage_plan_probe / "
                "triton_jit_probe / m0_kernel_probe — itemized, cover "
                "torch plan creation, triton JIT and M0 kernels in a "
                "fresh process), first_call (first execution on the real "
                "input in plan-reuse state), steady (warmup="
                f"{wp3.WARMUP} rounds={wp3.ROUNDS} trials={wp3.TRIALS} "
                "raw trials)"),
            "one_shot": (
                "reconstructed: setup_ms + warm_probe_total_ms + "
                "first_call_ms (one_shot_with_context_ms adds "
                "context_ms); an unsegmented fresh-process one-shot "
                "would pay these serially inside its first call; "
                "reported descriptively, NOT gated"),
            "warm": (f"plan-reuse steady state: warmup={wp3.WARMUP} "
                     f"rounds={wp3.ROUNDS} trials={wp3.TRIALS}"),
            "bootstrap": (
                f"independent bootstrap, B={BOOTSTRAP}, seed={SEED}: "
                "candidate and reference warm trial arrays resampled "
                "independently (not paired), percentile 95% CI"),
            "gate": (
                f"steady-state plan-reuse only; eligible_for_selector "
                f"requires correctness + contract in EVERY session and "
                f"bootstrap CI lower bound > {GATE} vs vendor AND vs "
                f"same-protocol incumbent, every session; one-shot "
                f"ratios reported separately"),
            "input_semantics": (
                "complex64 non-zero imag, contiguous (B,N), axis=-1, "
                "norm=backward (fwd unscaled, inv 1/N); b==1 incumbent "
                "measured as the 1-D (N,) production contract; input "
                "seeded per (session, n, b) and identical for all four "
                "paths"),
            "incumbent": (
                "same-protocol 4th path: plan_fft(direction='dit', "
                "profile_hash=None -> M0 abstain, table=config/"
                "selector_table.json) + execute_fft_plan on this run's "
                "sessions/segments"),
            "issue9_join": (
                "results/p800/issue9_v61/reports/e2e_report.csv kept "
                "only as descriptive_issue9_reference; never used in a "
                "gate"),
            "batched_contract": (
                "batched (B,N) runs through plan_fft(batch=b) + "
                "execute_fft_plan (entry contract); the public 1-D "
                "validate_input/run_fft contract applies to B=1 — the "
                "wp3_summary matrix stays an internal batched real-input "
                "microbenchmark"),
            "selector_use": (
                "evidence-only PR: no selector wiring; "
                "config/selector_table.json and src/xpu/selector.py "
                "untouched (see selector_diff.json)"),
            "correctness_tol_rel": 1e-4,
            "candidates": list(CANDIDATES),
            "primary_candidate": CANDIDATES[0],
        },
        "sessions": session_meta,
        "cells": cells_out,
        "fail_closed_cells": fail_closed,
        "eligibility": elig,
        "elapsed_s": round(time.time() - t_start, 1),
    }
    OUT.write_text(json.dumps(doc, indent=2))
    DIFF_OUT.write_text(json.dumps(_selector_diff(elig), indent=2))
    print(f"written {OUT}")
    print(f"written {DIFF_OUT}")
    print(json.dumps({
        "sessions": [{k: v for k, v in s.items() if k != "spawn_order"}
                     for s in session_meta],
        "eligibility": {k: v for k, v in elig.items()
                        if k != "candidates"},
        "per_candidate": {
            ch: {k: v for k, v in cand.items()
                 if k != "correctness_and_contract_pass"}
            for ch, cand in elig["candidates"].items()},
        "fail_closed": fail_closed,
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
