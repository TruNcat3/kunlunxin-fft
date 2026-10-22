"""issue #10 review next-step (1): twiddle implementation ablation.

Three implementations of the hybrid 4-step twiddle, compared on the
same cells with the same timing protocol (plus one output-side probe):

  base_sin_cos : current k_twiddle, runtime sin/cos libdevice every
                 element every call (wp3 hybrid_vendor_p baseline),
  table        : PLAN-TIME precomputed twiddle table (built once per
                 (n, direction) at path setup, reused every call),
                 kernel does table loads only,
  fused        : plan-time table + ZERO-COPY stage1->stage2 handoff
                 (reads r-FFT output directly in [b, k, j] layout,
                 removes the transpose-back + reshape GM copy that the
                 table/base layouts pay),
  fused_sin_cos: zero-copy handoff with runtime sin/cos (isolates
                 layout fusion from the table load effect),
  tperm        : baseline sin/cos twiddle + Triton digit-reversal
                 gather kernel instead of torch permute+contiguous
                 (isolates the output permute implementation).

Cells: {1024, 2048} x {64, 16, 256 (plus 2048x1)} x {fwd, inv} ->
  (1024,64), (1024,256), (2048,1), (2048,16), (2048,256), both dirs.

Sessions: 2 independent processes, fresh TRITON_CACHE_DIR each.
Cold = first call in the fresh process; warm = raw trials after
warmup (plan-reuse; table build is plan-time, excluded from call
timing and reported separately).

Output: results/p800/issue10_capability/wp3_ablation.json

Input semantics: INTERNAL BATCHED REAL-INPUT microbenchmark (float32 +
zero imag) — exploration only, NOT selector-grade evidence and no
selector entry derives from it; selector-grade complex64 evidence is
issue10_wp3_holdout.py -> wp3_holdout.json.
"""
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch  # noqa: F401
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPT = Path(__file__).resolve()
OUT = ROOT / "results/p800/issue10_capability/wp3_ablation.json"
MARK = "@@ABLATION@@"
CELLS = ((1024, 64), (1024, 256), (2048, 1), (2048, 16), (2048, 256))
DIRS = (("c2c_forward", False), ("c2c_inverse", True))
SESSIONS = (1, 2)
KINDS = ("base_sin_cos", "table", "fused", "fused_sin_cos",
         "tperm")
WARMUP, ROUNDS, TRIALS = 5, 5, 32


def _wp3():
    spec = importlib.util.spec_from_file_location(
        "issue10_wp3_perf", ROOT / "scripts/issue10_wp3_perf.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_kernels(wp3):
    from src.xpu.context import xpu_context  # noqa: F401

    @triton.jit
    def k_twiddle_table(X, O, TBL, R: tl.constexpr, T: tl.constexpr,
                        BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        j = pid % R
        b = pid // R
        base = ((b * R + j) * T) * 2
        k = tl.arange(0, BLOCK)
        mask = k < T
        re = tl.load(X + base + k * 2, mask=mask, other=0.0)
        im = tl.load(X + base + k * 2 + 1, mask=mask, other=0.0)
        wr = tl.load(TBL + j * T + k, mask=mask, other=1.0)
        wi = tl.load(TBL + j * T + k + R * T, mask=mask, other=0.0)
        tl.store(O + base + k * 2, re * wr - im * wi, mask=mask)
        tl.store(O + base + k * 2 + 1, re * wi + im * wr, mask=mask)

    @triton.jit
    def k_twiddle_fused(X, O, TBL, R: tl.constexpr, T: tl.constexpr,
                        BLOCK: tl.constexpr):
        # X: r-FFT output kept in [b, t, r] (k-major, contiguous over
        # j); element (b,k,j) at ((b*T + k)*R + j) -> no transpose-back
        # / reshape copy on the handoff. O: [b, r, t] for stage2.
        pid = tl.program_id(0)
        j = pid % R
        b = pid // R
        k = tl.arange(0, BLOCK)
        mask = k < T
        xoff = (b * T * R + k * R + j) * 2
        re = tl.load(X + xoff + k * 0, mask=mask, other=0.0)
        im = tl.load(X + xoff + k * 0 + 1, mask=mask, other=0.0)
        wr = tl.load(TBL + j * T + k, mask=mask, other=1.0)
        wi = tl.load(TBL + j * T + k + R * T, mask=mask, other=0.0)
        obase = ((b * R + j) * T) * 2
        tl.store(O + obase + k * 2, re * wr - im * wi, mask=mask)
        tl.store(O + obase + k * 2 + 1, re * wi + im * wr, mask=mask)

    @triton.jit
    def k_twiddle_fused_sin_cos(X, O, N: tl.constexpr, R: tl.constexpr,
                                T: tl.constexpr, BLOCK: tl.constexpr,
                                CONJ: tl.constexpr):
        # same [b, k, j] zero-copy handoff as k_twiddle_fused but with
        # runtime sin/cos (isolates layout fusion from table load)
        pid = tl.program_id(0)
        j = pid % R
        b = pid // R
        k = tl.arange(0, BLOCK)
        mask = k < T
        xoff = (b * T * R + k * R + j) * 2
        re = tl.load(X + xoff, mask=mask, other=0.0)
        im = tl.load(X + xoff + 1, mask=mask, other=0.0)
        jf = j.to(tl.float32) + (k * 0.0)
        ang = (6.283185307179586 / N) * jf * k.to(tl.float32)
        if CONJ == 1:
            ang = -ang
        wr = tl.cos(ang)
        wi = -tl.sin(ang)
        obase = ((b * R + j) * T) * 2
        tl.store(O + obase + k * 2, re * wr - im * wi, mask=mask)
        tl.store(O + obase + k * 2 + 1, re * wi + im * wr, mask=mask)

    @triton.jit
    def k_digit_reverse(S, OUT, R: tl.constexpr, T: tl.constexpr):
        # S: stage2 output [b, r, t] as interleaved real view;
        # OUT: [b, n]; out[b, k*R + j] = S[b, j, k]
        pid = tl.program_id(0)
        k = pid % T
        b = pid // T
        j = tl.arange(0, R)
        mask = j < R
        soff = (b * R * T + j * T + k) * 2
        re = tl.load(S + soff, mask=mask, other=0.0)
        im = tl.load(S + soff + 1, mask=mask, other=0.0)
        ooff = (b * (R * T) + k * R + j) * 2
        tl.store(OUT + ooff, re, mask=mask)
        tl.store(OUT + ooff + 1, im, mask=mask)

    return (k_twiddle_table, k_twiddle_fused, k_twiddle_fused_sin_cos,
            k_digit_reverse)


def _table(wr_wi_shape, inverse, device):
    import torch
    r, t = wr_wi_shape
    jj = torch.arange(r, device=device, dtype=torch.float32)
    kk = torch.arange(t, device=device, dtype=torch.float32)
    ang = (6.283185307179586 / (r * t)) * jj[:, None] * kk[None, :]
    if inverse:
        ang = -ang
    wr = torch.cos(ang)
    wi = -torch.sin(ang)
    return torch.cat([wr.reshape(-1), wi.reshape(-1)]).contiguous()


def run_variant(wp3, kerns, x, inverse: bool, kind: str):
    import torch
    from src.xpu.context import xpu_context
    (k_twiddle_table, k_twiddle_fused, k_twiddle_fused_sin_cos,
     k_digit_reverse) = kerns
    b, n = x.shape
    t, r = wp3._factors(n)
    xc = torch.complex(x, torch.zeros_like(x))
    view = xc.view(b, r, t)
    tw = torch.empty((b, r, t), dtype=torch.complex64, device="cuda")
    tw_flat = torch.view_as_real(tw).reshape(-1)
    t_start = time.perf_counter()
    tbl = _table((r, t), inverse, "cuda")     # PLAN-TIME precompute
    table_build_ms = (time.perf_counter() - t_start) * 1e3
    out_buf = None
    if kind == "tperm":
        out_buf = torch.empty((b, n, 2), dtype=torch.float32,
                              device="cuda")
    conj = 1 if inverse else 0

    def call():
        vt = view.transpose(1, 2)                       # [k, j]
        s1 = (torch.fft.ifft(vt, dim=-1) if inverse
              else torch.fft.fft(vt, dim=-1))           # [k, j]
        if kind in ("fused", "fused_sin_cos"):
            x_flat = torch.view_as_real(s1).reshape(-1)  # contiguous
            with xpu_context(0):
                if kind == "fused":
                    k_twiddle_fused[(b * r,)](
                        x_flat, tw_flat, tbl, r, t, t)
                else:
                    k_twiddle_fused_sin_cos[(b * r,)](
                        x_flat, tw_flat, n, r, t, t, conj)
                torch.cuda.synchronize()
        else:
            stage1 = s1.transpose(1, 2)                  # [j, k] view
            x_flat = torch.view_as_real(stage1).reshape(-1)  # copy
            with xpu_context(0):
                if kind == "table":
                    k_twiddle_table[(b * r,)](
                        x_flat, tw_flat, tbl, r, t, t)
                else:  # base_sin_cos / tperm
                    wp3.k_twiddle[(b * r,)](
                        x_flat, tw_flat, n, r, t, t, conj)
                torch.cuda.synchronize()
        stage2 = torch.fft.ifft(tw, dim=2) if inverse else \
            torch.fft.fft(tw, dim=2)
        if kind == "tperm":
            s_flat = torch.view_as_real(stage2).reshape(-1)
            with xpu_context(0):
                k_digit_reverse[(b * t,)](
                    s_flat, out_buf.reshape(-1), r, t)
                torch.cuda.synchronize()
            return out_buf[..., 0], out_buf[..., 1]
        out = stage2.permute(0, 2, 1).contiguous().reshape(b, n)
        return out.real, out.imag

    meta = {"table_build_ms": table_build_ms, "factors": {"t": t, "r": r}}
    return call, meta


def _trials(call):
    import torch
    torch.cuda.synchronize()
    for _ in range(WARMUP):
        call()
    torch.cuda.synchronize()
    ts = []
    for _ in range(ROUNDS):
        for _ in range(TRIALS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            call()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
    return ts


def worker(a) -> int:
    import torch
    wp3 = _wp3()
    kerns = _build_kernels(wp3)
    out = {"session": a.session, "cells": []}
    wp3._vendor_canary()
    for n, b in CELLS:
        for dname, inverse in DIRS:
            cell = {"n": n, "batch": b, "direction": dname, "paths": {}}
            x = wp3._data(b, n)
            wp3._contract_fields(x, n, b, inverse)
            for kind in KINDS:
                if kind == "base_sin_cos":
                    call, meta = wp3.run_hybrid(x, inverse)
                else:
                    call, meta = run_variant(wp3, kerns, x, inverse, kind)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                call()
                torch.cuda.synchronize()
                cold = (time.perf_counter() - t0) * 1e3
                y_re, y_im = call()
                err = wp3._check_batched(y_re, y_im, x, inverse)
                cell["paths"][kind] = {
                    "cold_ms": cold,
                    "trials": _trials(call),
                    "rel_err": float(err),
                    "correct": bool(err <= wp3.TOL_REL),
                    **meta,
                }
                del y_re, y_im, call
                torch.cuda.synchronize()
            out["cells"].append(cell)
            print(f"[s{a.session}] N{n}B{b} {dname} done", flush=True)
    print(MARK + json.dumps(out))
    return 0


def _p50(xs):
    ys = sorted(xs)
    return ys[len(ys) // 2] if ys else None


def _spawn(session, cache) -> dict | None:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "2"
    env["TRITON_CACHE_DIR"] = cache
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--worker",
         "--session", str(session)],
        capture_output=True, text=True, timeout=1800, env=env,
        cwd=str(ROOT))
    for line in (proc.stdout or "").splitlines():
        if line.startswith(MARK):
            try:
                return json.loads(line[len(MARK):])
            except json.JSONDecodeError:
                return None
    print(f"  worker fail s={session} rc={proc.returncode} "
          f"stderr={(proc.stderr or '')[-500:]}")
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--session", type=int, default=1)
    a = ap.parse_args()
    if a.worker:
        return worker(a)

    t_start = time.time()
    runs, session_meta = [], []
    for s in SESSIONS:
        cache = tempfile.mkdtemp(prefix=f"abl_triton_cache_s{s}_")
        t0 = time.time()
        r = _spawn(s, cache)
        session_meta.append({
            "session": s, "ok": r is not None,
            "elapsed_s": round(time.time() - t0, 1),
            "fresh_triton_cache": cache,
        })
        if r:
            runs.append(r)
        shutil.rmtree(cache, ignore_errors=True)
        print(f"session {s}: {'ok' if r else 'FAIL'} "
              f"({time.time() - t0:.0f}s)", flush=True)

    # aggregate: cell -> kind -> per-session p50
    agg = {}
    for r in runs:
        for c in r["cells"]:
            key = (c["n"], c["batch"], c["direction"], r["session"])
            agg[key] = c

    cells_out, winners, table_wins, negative = [], {}, 0, []
    for n, b in CELLS:
        for dname, _ in DIRS:
            entry = {"label": f"N{n}B{b}:{dname}", "n": n, "batch": b,
                     "direction": dname, "sessions": {}}
            per_kind_session = {k: [] for k in KINDS}
            errs_ok = True
            for s in SESSIONS:
                c = agg.get((n, b, dname, s))
                if c is None:
                    errs_ok = False
                    continue
                sentry = {"paths": {}}
                for kind, p in c["paths"].items():
                    p50 = _p50(p["trials"])
                    sentry["paths"][kind] = {
                        "warm_p50_ms": p50,
                        "cold_ms": p["cold_ms"],
                        "rel_err": p["rel_err"],
                        "correct": p["correct"],
                        "table_build_ms": p.get("table_build_ms"),
                        "n_trials": len(p["trials"]),
                    }
                    per_kind_session[kind].append(p50)
                    errs_ok = errs_ok and p["correct"]
                entry["sessions"][str(s)] = sentry
            # deltas vs baseline (mean of per-session deltas)
            base = per_kind_session["base_sin_cos"]
            deltas = {}
            for kind in KINDS:
                if len(per_kind_session[kind]) == len(base) == len(SESSIONS):
                    ds = [100.0 * (per_kind_session[kind][i] - base[i])
                          / base[i] for i in range(len(base))]
                    deltas[kind] = {"delta_pct_vs_base": ds,
                                    "mean_delta_pct": sum(ds) / len(ds)}
            entry["deltas"] = deltas
            entry["correct_all"] = errs_ok
            cells_out.append(entry)
            # winner = lowest mean p50
            means = {k: (sum(v) / len(v)) if v else None
                     for k, v in per_kind_session.items()}
            means = {k: v for k, v in means.items() if v is not None}
            if means:
                w = min(means, key=means.get)
                winners[entry["label"]] = w
                if w == "table":
                    table_wins += 1
                if w == "base_sin_cos":
                    negative.append(entry["label"])

    doc = {
        "schema": "issue10-wp3-ablation-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": str(SCRIPT.relative_to(ROOT)),
        "protocol": {
            "kinds": list(KINDS),
            "cells": [f"N{n}B{b}:{d}" for n, b in CELLS
                      for d, _ in DIRS],
            "sessions": len(SESSIONS),
            "session_isolation": "one fresh process per session with "
                                 "fresh TRITON_CACHE_DIR",
            "timing": f"cold=first call in fresh process; warm: "
                      f"warmup={WARMUP}, rounds={ROUNDS}, trials={TRIALS}"
                      " per cell x path, pooled wall p50",
            "plan_time": "twiddle table build (and tperm out_buf "
                         "alloc) happen at path setup, excluded from "
                         "call timing; table_build_ms reported",
            "baseline": "base_sin_cos == wp3 hybrid_vendor_p "
                        "(run_hybrid, runtime sin/cos)",
            "input_semantics": "internal batched real-input "
                               "microbenchmark (float32 + zero imag); "
                               "NOT selector-grade evidence",
            "selector_use": "none — ablation exploration only",
            "correctness": "rel_err <= 1e-4 vs numpy, every path x "
                           "cell x session",
        },
        "sessions": session_meta,
        "cells": cells_out,
        "winner_by_cell": winners,
        "summary": {
            "cells_total": len(CELLS) * len(DIRS),
            "table_wins": table_wins,
            "baseline_wins": negative,
            "all_correct": all(c["correct_all"] for c in cells_out),
        },
        "elapsed_s": round(time.time() - t_start, 1),
    }
    OUT.write_text(json.dumps(doc, indent=2))
    print(f"written {OUT}")
    print(json.dumps({"summary": doc["summary"],
                      "winners": winners}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
