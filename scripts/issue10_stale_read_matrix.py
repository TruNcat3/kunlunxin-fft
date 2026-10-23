"""issue #10 review item 4: stale-read root-cause matrix.

Classifies the WP2 same-buffer store -> permutation-load stale-read
phenomenon across: address permutation, allocation mode, grid size,
size, repetition, compile-cache state and independent sessions.

Modes (all report stale/error lane counts; prefill buffers with a
SENTINEL each trial so pre-store reads are detectable):

  same_kernel  : store S0[idx]=v; gather S0[perm(idx)] in ONE kernel
  diff_buffer  : store S0[idx]=v; gather S1[perm(idx)] (S1 never
                 written by the kernel; prefilled with expected values)
  cross_launch : launch 1 stores S0[idx]=v; launch 2 gathers
                 S0[perm(idx)] (explicit launch boundary)

Conclusion wording fixed by issue #10 review:
  same-buffer in-kernel permutation read = not-lowered;
  allowed alternatives = register residency or cross-launch read-only
  handoff.

Output: results/p800/issue10_capability/stale_read_matrix.json
Sessions: the main process spawns 2 independent worker processes, each
with a FRESH TRITON_CACHE_DIR (compile-cache cleared per session).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch  # noqa: F401
import torch_xmlir  # noqa: F401
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.xpu.context import xpu_context  # noqa: E402

SCRIPT = Path(__file__).resolve()
OUT = ROOT / "results/p800/issue10_capability/stale_read_matrix.json"
SENTINEL: tl.constexpr = -98765.4321
PERMS = ("xor1", "reverse", "stride2", "blockswap")
MODES = ("same_kernel", "diff_buffer", "cross_launch")
SIZES = (16, 32, 64, 128, 256)
GRIDS = (1, 2, 4, 8)
TRIALS = 16
MARK = "@@STALE_RESULT@@"
PMAP = {name: i for i, name in enumerate(PERMS)}


@triton.jit
def _perm_idx(idx, n, p):
    # p: 0=xor1 1=reverse 2=stride2 3=blockswap (half-block rotate)
    x = idx
    x = tl.where(p == 0, idx + 1 - 2 * (idx % 2), x)
    x = tl.where(p == 1, (n - 1) - idx, x)
    x = tl.where(p == 2, (idx * 2) % n, x)
    x = tl.where(p == 3, (idx + n // 2) % n, x)
    return x


@triton.jit
def k_same_kernel(S0, O, N: tl.constexpr, P: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * N
    idx = tl.arange(0, N)
    mask = idx < N
    v = (base + idx).to(tl.float32) + 1.0
    tl.store(S0 + base + idx, v, mask=mask)
    q = _perm_idx(idx, N, P)
    got = tl.load(S0 + base + q, mask=mask, other=SENTINEL)
    tl.store(O + base + idx, got, mask=mask)


@triton.jit
def k_diff_buffer(S0, S1, O, N: tl.constexpr, P: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * N
    idx = tl.arange(0, N)
    mask = idx < N
    v = (base + idx).to(tl.float32) + 1.0
    tl.store(S0 + base + idx, v, mask=mask)
    q = _perm_idx(idx, N, P)
    got = tl.load(S1 + base + q, mask=mask, other=SENTINEL)
    tl.store(O + base + idx, got, mask=mask)


@triton.jit
def k_writer(S0, N: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * N
    idx = tl.arange(0, N)
    mask = idx < N
    v = (base + idx).to(tl.float32) + 1.0
    tl.store(S0 + base + idx, v, mask=mask)


@triton.jit
def k_reader(S0, O, N: tl.constexpr, P: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * N
    idx = tl.arange(0, N)
    mask = idx < N
    q = _perm_idx(idx, N, P)
    got = tl.load(S0 + base + q, mask=mask, other=SENTINEL)
    tl.store(O + base + idx, got, mask=mask)


def _expected(g, n, pidx):
    idx = torch.arange(n, device="cuda")
    q = _host_perm(idx, n, pidx)
    v = (g * n + idx).float() + 1.0
    return v[q]


def _host_perm(idx, n, pidx):
    if pidx == 0:
        return idx + 1 - 2 * (idx % 2)
    if pidx == 1:
        return (n - 1) - idx
    if pidx == 2:
        return (idx * 2) % n
    return (idx + n // 2) % n


def run_matrix(session: int) -> dict:
    entries = []
    for perm in PERMS:
        pidx = PMAP[perm]
        for n in SIZES:
            for grid in GRIDS:
                for mode in MODES:
                    stale = 0
                    err = 0
                    for _ in range(TRIALS):
                        s0 = torch.full((grid * n,), SENTINEL,
                                        device="cuda")
                        o = torch.full((grid * n,), SENTINEL,
                                       device="cuda")
                        if mode == "same_kernel":
                            with xpu_context(0):
                                k_same_kernel[(grid,)](s0, o, n, pidx)
                            torch.cuda.synchronize()
                            for g in range(grid):
                                exp = _expected(g, n, pidx)
                                got = o[g * n:(g + 1) * n]
                                stale += int((got == SENTINEL).sum())
                                err += int(
                                    ((got - exp).abs() > 1e-3).sum()
                                    - int((got == SENTINEL).sum()))
                        elif mode == "diff_buffer":
                            # S1 never written by the kernel; prefilled
                            # with v+1000 (value-shifted vs S0's v so
                            # cross-buffer confusion is detectable)
                            s1 = torch.empty((grid * n,), device="cuda")
                            for g in range(grid):
                                s1[g * n:(g + 1) * n] = (
                                    (g * n + torch.arange(
                                        n, device="cuda")).float()
                                    + 1001.0)
                            with xpu_context(0):
                                k_diff_buffer[(grid,)](s0, s1, o, n, pidx)
                            torch.cuda.synchronize()
                            for g in range(grid):
                                exp = _expected(g, n, pidx) + 1000.0
                                got = o[g * n:(g + 1) * n]
                                err += int(((got - exp).abs() > 1e-3).sum())
                        else:  # cross_launch
                            with xpu_context(0):
                                k_writer[(grid,)](s0, n)
                                k_reader[(grid,)](s0, o, n, pidx)
                            torch.cuda.synchronize()
                            for g in range(grid):
                                exp = _expected(g, n, pidx)
                                got = o[g * n:(g + 1) * n]
                                stale += int((got == SENTINEL).sum())
                                err += int(
                                    ((got - exp).abs() > 1e-3).sum()
                                    - int((got == SENTINEL).sum()))
                    entries.append({
                        "session": session, "perm": perm, "n": n,
                        "grid": grid, "mode": mode, "trials": TRIALS,
                        "stale_lanes": stale,
                        "wrong_lanes": max(err, 0),
                        "total_lanes": grid * n * TRIALS,
                    })
        print(f"[session {session}] perm={perm} done", flush=True)
    return {"session": session, "entries": entries}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=None)
    ap.add_argument("--marker", action="store_true")
    a = ap.parse_args()
    if a.worker is not None:
        res = run_matrix(a.worker)
        print(MARK + json.dumps(res))
        return 0

    # main: two independent sessions, fresh compile cache each
    results, session_meta = [], []
    for s in (1, 2):
        cache = tempfile.mkdtemp(prefix=f"stale_triton_cache_s{s}_")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "2"
        env["TRITON_CACHE_DIR"] = cache
        env["PYTHONUNBUFFERED"] = "1"
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, str(SCRIPT), "--worker", str(s)],
            capture_output=True, text=True, timeout=3600, env=env,
            cwd=str(ROOT))
        payload = None
        for line in (proc.stdout or "").splitlines():
            if line.startswith(MARK):
                payload = json.loads(line[len(MARK):])
        session_meta.append({
            "session": s, "rc": proc.returncode,
            "elapsed_s": round(time.time() - t0, 1),
            "fresh_triton_cache": cache,
            "parsed": payload is not None,
            "stderr_tail": (proc.stderr or "")[-400:],
        })
        if payload:
            results.extend(payload["entries"])
        shutil.rmtree(cache, ignore_errors=True)

    # env / versions (review item 4: record vendor/runtime versions)
    versions = {
        "torch": getattr(torch, "__version__", "?"),
        "triton": getattr(triton, "__version__", "?"),
        "python": sys.version.split()[0],
    }
    wp0 = ROOT / "results/p800/issue10_capability/wp0_environment.json"
    if wp0.exists():
        try:
            w0 = json.loads(wp0.read_text())
            versions["wp0_env_hash"] = w0.get("env_hash")
            versions["wp0_profile_hash"] = w0.get("profile_hash")
        except Exception:  # noqa: BLE001
            pass

    # aggregates: mode x perm x n x grid over both sessions
    def agg(pred):
        st = sum(e["stale_lanes"] for e in results if pred(e))
        wr = sum(e["wrong_lanes"] for e in results if pred(e))
        to = sum(e["total_lanes"] for e in results if pred(e))
        return {"stale_lanes": st, "wrong_lanes": wr, "total_lanes": to}

    by_mode = {m: agg(lambda e, m=m: e["mode"] == m) for m in MODES}
    same = [e for e in results if e["mode"] == "same_kernel"]
    by_size = {str(n): agg(lambda e, n=n: e["mode"] == "same_kernel"
                           and e["n"] == n) for n in SIZES}
    by_grid = {str(g): agg(lambda e, g=g: e["mode"] == "same_kernel"
                           and e["grid"] == g) for g in GRIDS}
    by_perm = {p: agg(lambda e, p=p: e["mode"] == "same_kernel"
                      and e["perm"] == p) for p in PERMS}
    # cross-session consistency: same config, session 1 vs 2
    idx = {(e["session"], e["perm"], e["n"], e["grid"], e["mode"]): e
           for e in results}
    disagree = 0
    for e in results:
        if e["session"] != 1:
            continue
        o = idx.get((2, e["perm"], e["n"], e["grid"], e["mode"]))
        if o is None:
            continue
        a_hit = (e["stale_lanes"] > 0) != (o["stale_lanes"] > 0)
        disagree += int(a_hit)

    doc = {
        "schema": "issue10-stale-read-matrix-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": str(SCRIPT.relative_to(ROOT)),
        "versions": versions,
        "sessions": session_meta,
        "protocol": {
            "trials_per_config": TRIALS,
            "perms": list(PERMS), "modes": list(MODES),
            "sizes": list(SIZES), "grids": list(GRIDS),
            "prefill": f"S0/O prefilled with SENTINEL={SENTINEL}; "
                       "diff_buffer S1 prefilled with v[j]+1000 "
                       "(value-shifted vs S0's v[j])",
            "cache_policy": "fresh TRITON_CACHE_DIR per session "
                            "(compile cache cleared), independent "
                            "process per session",
        },
        "by_mode": by_mode,
        "same_kernel_by_size": by_size,
        "same_kernel_by_grid": by_grid,
        "same_kernel_by_perm": by_perm,
        "cross_session_hit_disagreements": disagree,
        "entries": results,
        "conclusion": (
            "same-buffer in-kernel permutation read = not-lowered; "
            "allowed alternatives: register residency or cross-launch "
            "read-only handoff. different-buffer and cross-launch "
            "controls are expected clean; any staleness there would "
            "widen the finding beyond same-kernel same-buffer access."),
        "root_cause_status": (
            "UNASSIGNED (hardware cache vs compiler alias analysis vs "
            "runtime memory model not separable with the available "
            "tools; matrix constrains the phenomenon by access mode / "
            "permutation / grid / size / session)"),
    }
    OUT.write_text(json.dumps(doc, indent=2))
    print(f"written {OUT}")
    print(json.dumps({"by_mode": by_mode,
                      "same_kernel_by_size": by_size,
                      "same_kernel_by_grid": by_grid,
                      "by_perm": by_perm,
                      "disagree": disagree}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
