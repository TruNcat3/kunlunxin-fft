"""issue #10 review item 5 (v4): performance holdout (>=2 independent
sessions) before any selector promotion — PR #12 review rounds 3-4.

Cells (holdout set fixed by the review):
  N in {1024, 2048} x B in {1, 16, 64, 256} x {c2c_forward, c2c_inverse}
  = 16 cells; paths = vendor + hybrid_vendor_p + hybrid_fused_p +
  incumbent_entry (profile-bound deployed incumbent) + m0_reference.

Review fixes (PR #12 comments, rounds 1-3):
  evidence-only: nothing here wires the selector; the block is named
    eligible_for_selector and selector_diff.json proves
    config/selector_table.json / src/xpu/selector.py are untouched.
    "eligible" means "meets the promotion evidence threshold" only.
  cold isolation: every (cell, path, session) runs in its OWN
    subprocess with its OWN fresh TRITON_CACHE_DIR; spawn order is
    randomized per session (seed recorded); segments are itemized:
    context / setup (incl. entry plan_fft_ms) / warm probes
    (vendor_plan, stage_plan, triton_jit, entry/m0_kernel) / first_call
    / steady. one_shot_ms = setup + probe total + first_call is a
    RECONSTRUCTION of a true unsegmented one-shot (reported
    descriptively; the gate uses steady state only).
  incumbent (round-3 fix): the hard gate runs the PROFILE-BOUND
    deployed selector — plan_fft(profile_hash=table["profile_hash"],
    env_hash="auto", table=config/selector_table.json) +
    execute_fft_plan — recording selection_source / entry_tier /
    executed / flow / plan_reason per row. entry_tier distinguishes
    table/semantic-or-legacy, previous-fallback, m0-abstain and
    composition-table (a composition_select match comes from the
    independent config/composition_table.json — its SHA/profile/env/
    entry key/contract_id are recorded per row in binding
    ["composition_table"]). If the plan abstains/falls back in a cell
    the row fails closed as "profile-bound incumbent no-go" (cell
    ineligible), never silently replaced by M0. m0_reference
    (profile_hash=None -> M0 abstain) is a separate descriptive
    baseline; issue9 v61 e2e values stay descriptive_issue9_reference
    only.
  stats: independent bootstrap (both warm trial arrays resampled
    independently) — NOT paired.
  input (round-4 fix): complex64 non-zero imag, contiguous, axis=-1,
    norm=backward (fwd unscaled, inv 1/N), seeded per (session, n,
    b). At b==1 EVERY path (vendor, both hybrids, incumbent, m0)
    receives the same rank-1 (N,) production-contract tensor: the
    candidate runners accept rank-1 natively (wp3_perf run_vendor /
    run_hybrid / run_hybrid_fused; the (N,)->(1,r,t) step is a
    zero-copy view at setup, outputs keep the stable (b,n) layout).
    Every path records api_entry / input_shape / input_stride /
    batch_contract / entry_contract in its meta, and a cell only
    stays eligible when all five paths' input_shape agree
    (entry_shape_uniform). B>1 stays (B,N) entry-level contract on
    all paths.
  batched boundary: B>1 rows are entry-level contract evidence
    (plan_fft(batch=b) + execute_fft_plan -> fft_*_batch with
    _validate_batch_input; batch runner tests in tests/test_fft.py,
    run_fft rank-2 test). The public validate_input helper stays the
    1-D single-length contract (docstring synced); a full
    public-batched helper sync is a separate contract PR before
    selector integration.

Gate (code-level, steady plan-reuse only): a cell is
eligible_for_selector for a candidate iff in EVERY session: all paths
present, no fail_closed (incl. profile-bound incumbent no-go), all
paths correct (rel_err <= 1e-4), contract present, entry shapes
uniform across the five paths, and bootstrap CI lower bound of
speedup > GATE vs BOTH vendor and the profile-bound incumbent.
One-shot ratios are reported separately. Promotion additionally
requires the separate integration PR (see eligibility
promotion_prerequisites): exact-key/fallback/abstain binding plus an
independent session or explicit spread guard.

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
PATHS = ("vendor", "hybrid_vendor_p", "hybrid_fused_p",
         "incumbent_entry", "m0_reference")
# profile-bound deployed incumbent (table profile_hash + env auto);
# abstain/fallback here = cell no-go, never silently replaced by M0
INCUMBENT_LABEL = "incumbent_entry"
# M0 abstain baseline (profile_hash=None): descriptive only, never a gate
M0_LABEL = "m0_reference"
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


# which API/shape actually enters the measurement, per path — the
# round-4 review requires these recorded raw so a cell label can never
# stand in for the shape
_API_ENTRY = {
    "vendor": ("torch.fft last-dim op via wp3_perf.run_vendor "
               "(rank-1 capable; 1-D output lifted view(1,N))"),
    "hybrid_vendor_p": ("wp3_perf.run_hybrid 4-step (rank-1 entry; "
                        "(N,)->(1,r,t) zero-copy view at setup)"),
    "hybrid_fused_p": ("wp3_perf.run_hybrid_fused 4-step (rank-1 "
                       "entry; (N,)->(1,r,t) zero-copy view at setup)"),
    "incumbent_entry": ("plan_fft + execute_fft_plan (production "
                        "entry; rank-1 (N,) for b==1)"),
    "m0_reference": ("plan_fft + execute_fft_plan (M0 abstain path; "
                     "rank-1 (N,) for b==1)"),
}


def _entry_meta(path: str, xb, b: int) -> dict:
    """Per-path raw record of API entry + input shape/stride/batch
    contract (round-4: B=1 must be the same (N,) contract on every
    path, verifiable from the artifact alone)."""
    return {
        "api_entry": _API_ENTRY[path],
        "input_shape": list(xb.shape),
        "input_stride": list(xb.stride()),
        "batch_contract": (
            "rank-1 (N,) production 1-D contract (b==1)"
            if b == 1 else
            "rank-2 (B,N) entry-level batched contract (B>1)"),
        "entry_contract": (
            f"complex64 {'(N,)' if b == 1 else '(B,N)'} axis=-1 "
            "norm=backward"),
    }


def _entry_tier(selection_source, params, fallback) -> str:
    """Honest entry tier for a resolved plan (round-4: composition
    rows come from the independent composition registry, not from the
    butterfly selector table)."""
    if params.get("previous_fallback"):
        return "previous-fallback"
    if fallback:
        return "m0-abstain"
    if selection_source == "composition":
        return "composition-table"
    return "table/semantic-or-legacy"


def _composition_record(plan, n: int, b: int, inverse: bool) -> dict:
    """Provenance of a composition_select match (round-4): the table
    that answered, its sha/profile/env, the exact entry key and
    contract, so composition-table rows are auditable next to
    butterfly-table rows."""
    ct_path = ROOT / "config" / "composition_table.json"
    rec = {"table": "config/composition_table.json", "matched": False}
    if not ct_path.exists():
        return rec
    ct = json.loads(ct_path.read_text())
    rec.update({
        "table_sha256": hashlib.sha256(ct_path.read_bytes()).hexdigest(),
        "table_profile_hash": ct.get("profile_hash"),
        "table_env_hash": ct.get("env_hash"),
        "entry_key": {"N": n, "batch": b, "direction": plan.direction,
                      "transform_type": "c2c",
                      "transform_direction": ("inverse" if inverse
                                              else "forward")},
        "contract_id": ct.get("contract_id"),
        "flow_id": plan.flow_id,
        "executed": plan.executed,
        "guard_pct": (plan.params or {}).get("composition_guard_pct"),
    })
    hit = next((e for e in ct.get("entries", [])
                if e.get("N") == n and e.get("batch") == b
                and e.get("direction") == plan.direction), None)
    if hit is not None:
        rec["matched"] = True
        rec["contract_id"] = hit.get("contract_id") or rec["contract_id"]
        rec["flow_id"] = hit.get("flow_id", rec["flow_id"])
        ev = hit.get("evidence") or {}
        rec["evidence_artifact"] = ev.get("artifact")
        rec["evidence_artifact_sha256"] = ev.get("artifact_sha256")
    return rec


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
        # THE entry tensor for every path (round-4): at b==1 all five
        # paths measure the same rank-1 (N,) production contract
        xb = x.squeeze(0) if b == 1 else x
        meta = None
        plan = None
        t0 = time.perf_counter()
        if p == "vendor":
            call, meta = wp3.run_vendor(xb, inverse)
        elif p in ("hybrid_vendor_p", "hybrid_fused_p"):
            call, meta = wp3.PATHS[p](xb, inverse)
        elif p in (INCUMBENT_LABEL, M0_LABEL):
            from src.xpu.entry import execute_fft_plan, plan_fft
            from src.xpu.selector import load_table
            tp0 = time.perf_counter()
            table = load_table(str(SELECTOR_TABLE))
            if p == INCUMBENT_LABEL:
                # profile-bound deployed incumbent: the table's own
                # profile binding + on-device env verification (auto).
                # An abstain/fallback plan here means the profile-bound
                # incumbent cannot legally resolve in this cell ->
                # fail closed as no-go, NEVER silently replaced by M0.
                prof = table.get("profile_hash")
                plan = plan_fft(direction="dit", n=n, batch=b,
                                profile_hash=prof, table=table,
                                env_hash="auto",
                                transform_type="c2c",
                                transform_direction=("inverse"
                                                     if inverse
                                                     else "forward"),
                                norm="backward")
            else:
                # m0_reference: profile_hash=None abstains to M0 by the
                # selector contract — descriptive baseline only, it is
                # NOT the deployed incumbent and never a hard gate.
                prof = None
                plan = plan_fft(direction="dit", n=n, batch=b,
                                profile_hash=None, table=table,
                                transform_type="c2c",
                                transform_direction=("inverse"
                                                     if inverse
                                                     else "forward"),
                                norm="backward")
            segs["plan_fft_ms"] = (time.perf_counter() - tp0) * 1e3

            params = dict(plan.params or {})
            binding = {
                "role": ("profile-bound deployed incumbent "
                         "(profile_hash + env_hash=auto; composition "
                         "registry outranks the butterfly table for "
                         "exact-shape c2c-forward)"
                         if p == INCUMBENT_LABEL else
                         "m0_reference (profile_hash=None abstains to "
                         "M0; NOT the deployed incumbent)"),
                "profile_hash": prof,
                "env_hash_mode": "auto" if p == INCUMBENT_LABEL else "n/a",
                "table_env_hash": table.get("env_hash"),
                "selection_source": plan.selection_source,
                "fallback": bool(plan.fallback),
                "entry_tier": _entry_tier(plan.selection_source, params,
                                          plan.fallback),
                "executed": plan.executed,
                "flow_id": plan.flow_id,
                "flow_realization": plan.flow_realization,
                "processing_unit": plan.processing_unit,
                "block": plan.block,
                "plan_reason": str(plan.reason)[:300],
                "composition": params.get("flow_kind") == "composition",
                "composition_table": (
                    _composition_record(plan, n, b, inverse)
                    if plan.selection_source == "composition" else None),
                "params": {k: str(v)[:100]
                           for k, v in sorted(params.items())},
            }
            if p == INCUMBENT_LABEL and (plan.fallback
                                         or plan.selection_source
                                         == "abstain"):
                stage = "profile_bound_incumbent"
                raise RuntimeError(
                    "profile-bound incumbent no-go: selection_source="
                    f"{plan.selection_source} fallback={plan.fallback} "
                    f"reason={plan.reason}")

            def call():
                y = execute_fft_plan(plan, xb)
                if y.dim() == 1:
                    y = y.reshape(b, n)
                return y.real, y.imag

            meta = {"launches": "production entry execute_fft_plan",
                    "gm_bytes_analytic": None,
                    **binding}
        else:
            raise ValueError(f"unknown path {p}")
        # round-4: raw API/shape record on EVERY path (aggregator gates
        # on entry_shape_uniform across the five paths)
        meta.update(_entry_meta(p, xb, b))
        segs["setup_ms"] = (time.perf_counter() - t0) * 1e3

        # warm probes on SCRATCH state (fresh TRITON_CACHE_DIR): cover
        # torch fft plan creation, triton JIT, M0 kernels so that
        # first_call measures the first REAL execution in plan-reuse
        # state; probe ms are itemized and summed into one_shot_ms.
        stage = "warm_probe"
        with torch.no_grad():
            if p == "vendor":
                t1 = time.perf_counter()
                y = (torch.fft.ifft(xb.clone()) if inverse
                     else torch.fft.fft(xb.clone()))
                del y
                torch.cuda.synchronize()
                segs["vendor_plan_probe_ms"] = (
                    time.perf_counter() - t1) * 1e3
            elif p in ("hybrid_vendor_p", "hybrid_fused_p"):
                t, r = wp3._factors(n)
                t1 = time.perf_counter()
                scr = xb.clone()
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
            else:  # incumbent_entry / m0_reference: entry-kernel warm
                t1 = time.perf_counter()
                y = execute_fft_plan(plan, xb.clone())
                del y
                torch.cuda.synchronize()
                segs[("m0_kernel_probe_ms" if p == M0_LABEL
                      else "entry_kernel_probe_ms")] = (
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
            "meets_incumbent_profile_bound_gate": ready[ch]["incumbent"],
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
            f"requires (every session) all_paths_present AND "
            f"no_fail_closed AND correct_all_sessions (rel_err <= 1e-4) "
            f"AND contract_all_sessions AND entry_shape_uniform "
            f"(all five paths measured on ONE entry shape — B=1 is the "
            f"same rank-1 (N,) production contract everywhere) AND "
            f"bootstrap CI lower bound of speedup > {GATE} vs vendor "
            f"AND vs the PROFILE-BOUND deployed incumbent_entry "
            f"(profile_hash + env_hash=auto); a profile-bound "
            f"abstain/fallback is a cell no-go; one-shot ratios "
            f"(one_shot_ratio_*) and the m0_reference are reported "
            f"separately and NOT gated"),
        "cell_field_definitions": {
            "all_paths_present": "every path row exists in every session",
            "no_fail_closed": "no path row failed closed in any session "
                              "(includes profile-bound incumbent no-go)",
            "correct_all_sessions": "every path correct in every session",
            "contract_all_sessions": "selector-grade input contract "
                                     "present in every session",
            "entry_shape_uniform": "all five paths' meta.input_shape "
                                   "agree within and across sessions "
                                   "(B=1 = rank-1 (N,) on every path; "
                                   "missing record fails the flag)",
        },
        "promotion_prerequisites": (
            "eligible_for_selector is a promotion CANDIDATE list only; "
            "the separate integration PR must (1) bind exact keys with "
            "profile_hash+env_hash+evidence_sha256, (2) keep "
            "fallback/abstain for uncovered cells, (3) add "
            "plan_fft->execute_fft_plan dispatch regressions, and "
            "(4) add an independent session or an explicit "
            "cross-session spread guard before any default-path switch "
            f"(max observed spread this run: {primary['max_cross_session_spread_rel']})"),
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
        "incumbent_policy": (
            "the hard gate uses incumbent_entry = the PROFILE-BOUND "
            "deployed selector (plan_fft(profile_hash=table"
            "[profile_hash], env_hash=auto) + execute_fft_plan) "
            "measured in this run's sessions; selection_source/"
            "entry_tier/executed/flow are recorded per row, with "
            "entry_tier distinguishing table/semantic-or-legacy, "
            "previous-fallback, m0-abstain and composition-table; "
            "m0_reference (profile_hash=None -> M0 abstain) is a "
            "descriptive baseline only and never a gate"),
        "composition_policy": (
            "plan_fft queries composition_select (config/"
            "composition_table.json) BEFORE the butterfly selector; a "
            "match returns selection_source=composition, "
            "executed=run_hybrid, entry_tier=composition-table — the "
            "per-row binding records the composition table sha256/"
            "profile/env/exact entry key/contract_id under "
            "binding.composition_table, and selector_diff.json records "
            "both tables' provenance; the integration PR must treat "
            "composition entries with their own contract checks"),
        "issue9_policy": (
            "issue9 v61 e2e values (incl. its historical selector:e2e) "
            "are descriptive_issue9_reference only (different "
            "warmup/rounds machinery); the hard incumbent gate is the "
            "profile-bound incumbent_entry measured in this run"),
        "batched_boundary": (
            "B>1 eligibility is entry-level contract evidence "
            "(plan_fft(batch=b)+execute_fft_plan -> fft_*_batch / "
            "_validate_batch_input, run_fft rank-2); the public "
            "validate_input helper remains the 1-D single-length "
            "contract — a full public-batched helper sync is a "
            "separate contract PR before selector integration"),
    }


def _cell_passes(flags: dict) -> bool:
    """Code-level correctness/contract gate (pure, unit-tested).

    A cell only feeds eligibility when EVERY session had all path rows
    present, no fail_closed row (incl. profile-bound incumbent no-go),
    all paths correct, the selector-grade input contract, and one
    uniform entry shape across the five paths (round-4: B=1 must be
    the same (N,) contract everywhere).
    """
    return bool(flags.get("all_paths_present")
                and flags.get("no_fail_closed")
                and flags.get("correct_all_sessions")
                and flags.get("contract_all_sessions")
                and flags.get("entry_shape_uniform"))


def _entry_shape_uniform(shapes) -> bool:
    """``shapes``: {session: {path: [dims]}} for present paths (pure,
    unit-tested). True iff every present path in every session agrees
    on ONE shape, no record is lost ("__missing__" sentinel), and at
    least one session reported."""
    seen = set()
    for per_path in shapes.values():
        for shp in per_path.values():
            seen.add(tuple(shp))
    if not seen or any("__missing__" in s for s in seen):
        return False
    return len(seen) == 1


def _collect_ready(ready, spreads, ch, label, vs, iss, cell_ok,
                   n_sessions) -> dict:
    """Single append-condition site for the eligibility lists (pure,
    unit-tested).

    - correctness_and_contract_pass: ``cell_ok`` ONLY (the name promises
      correctness+contract, so CI must not gate it — but a failing cell
      must never appear);
    - meets_vendor_gate / meets_incumbent_profile_bound_gate: cell_ok
      AND independent-bootstrap CI low > GATE in EVERY session;
    - returns the cross-session display dict for this candidate.
    """
    out = {}
    if not cell_ok:
        return out
    if len(vs) != n_sessions or len(iss) != n_sessions:
        return out              # partial session data: no decision at all
    if len(vs) == n_sessions:
        pts = [sp["point"] for sp in vs]
        spr = abs(pts[0] - pts[1]) / max(sum(pts) / len(pts), 1e-30)
        spreads[ch]["vendor"].append(spr)
        ok = all(sp["ci95"][0] > GATE for sp in vs)
        out["vendor"] = {"points": pts, "spread_rel": spr,
                         "all_ci_low_gt_gate": ok}
        if ok:
            ready[ch]["vendor"].append(label)
    if len(iss) == n_sessions:
        pts = [e["point"] for e in iss]
        spr = abs(pts[0] - pts[1]) / max(sum(pts) / len(pts), 1e-30)
        spreads[ch]["incumbent"].append(spr)
        ok = all(e["ci95"][0] > GATE for e in iss)
        out["incumbent_profile_bound"] = {
            "points": pts, "spread_rel": spr, "all_ci_low_gt_gate": ok}
        if ok:
            ready[ch]["incumbent"].append(label)
    if len(vs) == n_sessions and len(iss) == n_sessions:
        ready[ch]["correct"].append(label)
    return out


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
    comp_path = ROOT / "config" / "composition_table.json"
    comp_sha = (hashlib.sha256(comp_path.read_bytes()).hexdigest()
                if comp_path.exists() else None)
    ct = (json.loads(comp_path.read_text())
          if comp_path.exists() else {})
    cells = [_holdout_cell_state(n, b, inv)
             for n in NS for b in BS for _, inv in DIRS]
    return {
        "schema": "issue10-selector-diff-v2",
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
        "tables": {
            "selector_table": {
                "path": "config/selector_table.json",
                "sha256": table_sha,
                "profile_hash": t.get("profile_hash"),
                "env_hash": t.get("env_hash"),
                "n_entries": len(t.get("entries", [])),
                "n_previous_fallbacks": len(t.get("previous_fallbacks",
                                                   [])),
            },
            "composition_table": {
                "path": "config/composition_table.json",
                "sha256": comp_sha,
                "profile_hash": ct.get("profile_hash"),
                "env_hash": ct.get("env_hash"),
                "n_entries": len(ct.get("entries", [])),
                "contract_id": ct.get("contract_id"),
            },
            "relation": (
                "plan_fft queries composition_select (composition_table"
                ".json) BEFORE the butterfly selector: a match returns "
                "selection_source='composition' (executed=run_hybrid, "
                "entry_tier=composition-table) and outranks the "
                "butterfly table for that exact shape (c2c + forward + "
                "backward-norm only; fail-closed to the butterfly path "
                "otherwise). Butterfly-table rows stay "
                "selection_source='table'; previous_fallbacks keep "
                "entry_tier='previous-fallback'. The integration PR "
                "must keep composition entries on their own contract "
                "and runner checks."),
        },
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
        "evidence_basis": (
            "eligibility was measured against the profile-bound "
            "deployed incumbent (table profile_hash + env_hash=auto, "
            "incl. composition-registry matches recorded as "
            "entry_tier=composition-table) in the same protocol, not "
            "against M0 or issue9 history, on ONE uniform entry shape "
            "per cell (B=1 = rank-1 (N,) on all five paths), see "
            "wp3_holdout.json eligibility "
            "(meets_incumbent_profile_bound_gate, entry_shape_uniform) "
            "and per-row selection_source/entry_tier/api_entry/"
            "input_shape"),
        "batched_boundary": elig.get("batched_boundary"),
        "promotion_prerequisites": elig.get("promotion_prerequisites"),
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
        flags = {"all_paths_present": True, "no_fail_closed": True,
                 "correct_all_sessions": True,
                 "contract_all_sessions": True}
        entry_shapes = {}
        for s in SESSIONS:
            rows = idx.get((n, b, d, s), {})
            if any(p not in rows for p in PATHS):
                missing = [p for p in PATHS if p not in rows]
                fail_closed.append(f"{label}/s{s}: missing {missing}")
                flags["all_paths_present"] = False
                continue
            bad = {p: rows[p].get("fail_closed") for p in PATHS
                   if rows[p].get("fail_closed")}
            if bad:
                fail_closed.append(f"{label}/s{s}: {bad}")
                flags["no_fail_closed"] = False
                continue
            contract = next((rows[p].get("contract") for p in PATHS
                             if rows[p].get("contract")), None)
            if not contract:
                flags["contract_all_sessions"] = False
            # round-4: raw per-path entry shapes (missing record fails
            # the uniformity flag via the "__missing__" sentinel)
            entry_shapes[s] = {
                p: ((rows[p].get("meta") or {}).get("input_shape")
                    or ["__missing__"])
                for p in PATHS}
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
                    flags["correct_all_sessions"] = False
            vrow = entry["paths"]["vendor"]
            inc_row = entry["paths"][INCUMBENT_LABEL]
            m0_row = entry["paths"][M0_LABEL]
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
                        "(>1 = candidate faster), PROFILE-BOUND "
                        "deployed incumbent (table profile_hash + "
                        "env_hash=auto) measured in this run")
                    entry[f"speedup_incumbent_profile_bound_boot:{ch}"] = spi
                    cand_sessions[ch]["i"].append(spi)
                rm = m0_row["warm_trials_ms"]
                spd = _boot_ratio(rng, rm, rh)
                if spd:
                    spd["definition"] = (
                        f"median({M0_LABEL})/median({ch}) "
                        "(>1 = candidate faster), descriptive only — "
                        "M0 abstain baseline, NOT the deployed "
                        "incumbent and never a gate")
                    entry[f"speedup_m0_reference_boot:{ch}"] = spd
                if (vrow["one_shot_ms"] and h["one_shot_ms"]):
                    entry[f"one_shot_ratio_vendor:{ch}"] = (
                        vrow["one_shot_ms"] / h["one_shot_ms"])
                if (inc_row["one_shot_ms"] and h["one_shot_ms"]):
                    entry[f"one_shot_ratio_incumbent:{ch}"] = (
                        inc_row["one_shot_ms"] / h["one_shot_ms"])
                if (m0_row["one_shot_ms"] and h["one_shot_ms"]):
                    entry[f"one_shot_ratio_m0:{ch}"] = (
                        m0_row["one_shot_ms"] / h["one_shot_ms"])
            c["sessions"][str(s)] = entry

        # round-4: B=1 (and every cell) must measure ONE entry shape
        # across all five paths in all sessions before eligibility
        flags["entry_shape_uniform"] = _entry_shape_uniform(entry_shapes)
        c["entry_shapes"] = {str(s): {p: list(shp) for p, shp in per.items()}
                             for s, per in entry_shapes.items()}
        cell_ok = _cell_passes(flags)
        c.update(flags)
        c["eligible_inputs_ok"] = cell_ok
        for ch in CANDIDATES:
            vs, iss = cand_sessions[ch]["v"], cand_sessions[ch]["i"]
            c.setdefault("cross_session_by_candidate", {})[ch] = (
                _collect_ready(ready, spreads, ch, label, vs, iss,
                               cell_ok, len(SESSIONS)))
        cells_out.append(c)

    elig = _eligibility_block(ready, spreads, len(cells))
    wp3 = _wp3()
    doc = {
        "schema": "issue10-wp3-holdout-v4",
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
                "(path construction; entry paths include table load + "
                "plan_fft, sub-timed as plan_fft_ms), warm probes on "
                "scratch (vendor_plan_probe / stage_plan_probe / "
                "triton_jit_probe / entry_kernel_probe / "
                "m0_kernel_probe — itemized, cover torch plan creation, "
                "triton JIT and entry kernels in a fresh process), "
                "first_call (first execution on the real input in "
                "plan-reuse state), steady (warmup="
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
                f"requires (every session) all_paths_present AND "
                f"no_fail_closed AND correct_all_sessions AND "
                f"contract_all_sessions AND entry_shape_uniform AND "
                f"independent-bootstrap CI lower bound > {GATE} vs "
                f"vendor AND vs the profile-bound incumbent_entry, "
                f"every session; one-shot ratios and m0_reference "
                f"reported separately"),
            "input_semantics": (
                "complex64 non-zero imag, contiguous (B,N), axis=-1, "
                "norm=backward (fwd unscaled, inv 1/N); b==1: EVERY "
                "path (vendor/hybrids/incumbent/m0) receives the same "
                "rank-1 (N,) production-contract tensor — candidate "
                "runners accept rank-1 natively (zero-copy "
                "(N,)->(1,r,t) setup view), per-path api_entry/"
                "input_shape/input_stride/batch_contract recorded in "
                "meta and gated by entry_shape_uniform; input seeded "
                "per (session, n, b) and identical for all five paths"),
            "incumbent": (
                "PROFILE-BOUND deployed selector path: "
                "plan_fft(direction='dit', profile_hash="
                "table['profile_hash'], env_hash='auto', table=config/"
                "selector_table.json) + execute_fft_plan on this run's "
                "sessions/segments; per-row record of selection_source, "
                "entry_tier (previous-fallback), executed, flow_id/"
                "flow_realization/processing_unit, plan_reason; an "
                "abstain/fallback plan fails the row closed as "
                "'profile-bound incumbent no-go' (cell ineligible), "
                "never silently replaced by M0"),
            "m0_reference": (
                "m0_reference path: plan_fft(profile_hash=None) "
                "abstains to M0 by the selector contract; recorded as "
                "speedup_m0_reference_boot / one_shot_ratio_m0 for "
                "descriptive comparison only — NOT the deployed "
                "incumbent, never a gate input"),
            "issue9_join": (
                "results/p800/issue9_v61/reports/e2e_report.csv kept "
                "only as descriptive_issue9_reference (incl. its "
                "historical selector:e2e); never used in a gate"),
            "batched_contract": (
                "B>1 rows are entry-level contract evidence: "
                "plan_fft(batch=b) + execute_fft_plan -> fft_*_batch "
                "runners with _validate_batch_input (rank-2, tested in "
                "tests/test_fft.py) and run_fft rank-2 (test added "
                "with this round); the public validate_input helper "
                "stays the 1-D single-length contract (docstring "
                "synced) — full public-batched helper sync is a "
                "separate contract PR before selector integration; "
                "wp3_summary stays an internal batched real-input "
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
