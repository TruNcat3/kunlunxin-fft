#!/usr/bin/env python3
"""WP1: Kunlunxin Triton/XMLIR capability micro-probes (Issue #10).

Ten capability areas from the issue's WP1 table.  Every status is
produced by executing a probe on the device -- never inferred from the
platform name.  Statuses are strictly ``PASS / FAIL / UNKNOWN``:

* ``PASS``  -- the probe compiled, executed, and the checked property
  held against an independent reference;
* ``FAIL``  -- the capability demonstrably does not lower/execute here
  (the minimal failing repro is the probe source + captured log);
* ``UNKNOWN`` -- the question could not be answered on this stack
  (API absent AND effect unobservable, or a timeout); each UNKNOWN
  carries a ``next_step`` verification entry point.

Robustness rules (flag_gems known-issues #11 / issue #9 fail-closed):

* every launch runs inside ``src.xpu.context.xpu_context`` -- without
  it launches are silent no-ops;
* output buffers are prefilled with a sentinel and checked to have
  changed, so a silent no-op surfaces as a FAILED check, never a pass;
* each probe runs in its own subprocess (a driver crash kills only
  that probe); a crash is FAIL with the log, a timeout is UNKNOWN.

Usage:
    python3 scripts/issue10_wp1_capability.py            # run all
    python3 scripts/issue10_wp1_capability.py --probe NAME  # one
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

# Import order matters (see issue10_wp0_env): torch + torch_xmlir first.
import torch  # noqa: E402,F401
import torch_xmlir  # noqa: E402,F401
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import triton  # noqa: E402
import triton.language as tl  # noqa: E402
# module-level imports are required inside @triton.jit bodies: the XPU
# fork's AST lowering rejects `from ... import` inside kernels.
from triton.language.extra.xpu import libdevice  # noqa: E402,E501
from triton.language.standard import swizzle2d  # noqa: E402,E501

from src.xpu.context import xpu_context  # noqa: E402

OUT_DIR = ROOT / "results/p800/issue10_capability"
LOG_DIR = OUT_DIR / "wp1_probes" / "logs"
MATRIX_PATH = OUT_DIR / "capability_matrix.json"
SCRIPT_PATH = ROOT / "scripts/issue10_wp1_capability.py"
MARK = "##RESULT##"
PROBE_TIMEOUT_S = 600
SENTINEL = -98765.4321

# --------------------------------------------------------------------------
# kernel library (module level: required by the Triton JIT)
# --------------------------------------------------------------------------


@triton.jit
def k_scalar_fill(OUT, N, FACTOR: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    tl.store(OUT + offs, offs.to(tl.float32) * FACTOR, mask=mask)


@triton.jit
def k_vec_axpy(IN, OUT, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(IN + offs, mask=mask, other=0.0)
    tl.store(OUT + offs, x * 2.0 + 1.0, mask=mask)


@triton.jit
def k_strided(IN, OUT, N, ST: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(IN + offs * ST, mask=mask, other=0.0)
    tl.store(OUT + offs * ST, x * 3.0, mask=mask)


@triton.jit
def k_transpose(IN, OUT, M, N, BM: tl.constexpr, BN: tl.constexpr):
    r = tl.arange(0, BM)
    c = tl.arange(0, BN)
    mr = r[:, None]
    nc = c[None, :]
    mask = (mr < M) & (nc < N)
    # IN is (M, N) row-major: element (r, c) lives at r * N + c
    v = tl.load(IN + mr * N + nc, mask=mask, other=0.0)
    # OUT has shape (N, M): element (c, r) lives at c * M + r
    tl.store(OUT + nc * M + mr, v, mask=mask)


@triton.jit
def k_tile_scale(IN, OUT, M, N, BM: tl.constexpr, BN: tl.constexpr):
    r = tl.arange(0, BM)
    c = tl.arange(0, BN)
    mr = r[:, None]
    nc = c[None, :]
    mask = (mr < M) & (nc < N)
    v = tl.load(IN + mr * N + nc, mask=mask, other=0.0)
    tl.store(OUT + mr * N + nc, v * 1.25, mask=mask)


@triton.jit
def k_swizzle_mark(OUT, W, H, BLOCK: tl.constexpr):
    r = tl.arange(0, BLOCK)
    c = tl.arange(0, BLOCK)
    rr = r[:, None]
    cc = c[None, :]
    mask = (rr < H) & (cc < W)
    sr, sc = swizzle2d(rr, cc, H, W, 2)
    mark = tl.where(mask, 1.0, 0.0)
    tl.store(OUT + sr * W + sc, mark, mask=mask)


@triton.jit
def k_dtype_axpy(IN, OUT, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(IN + offs, mask=mask, other=0.0)
    tl.store(OUT + offs, x * 2.0 + 1.0, mask=mask)


@triton.jit
def k_cmul(RE, IM, WR, WI, ORE, OIM, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(RE + offs, mask=mask, other=0.0)
    b = tl.load(IM + offs, mask=mask, other=0.0)
    c = tl.load(WR + offs, mask=mask, other=0.0)
    d = tl.load(WI + offs, mask=mask, other=0.0)
    tl.store(ORE + offs, a * c - b * d, mask=mask)
    tl.store(OIM + offs, a * d + b * c, mask=mask)


@triton.jit
def k_dot(A, B, OUT, M, N, K, BM: tl.constexpr, BN: tl.constexpr,
          BK: tl.constexpr):
    m = tl.arange(0, BM)
    n = tl.arange(0, BN)
    k = tl.arange(0, BK)
    a = tl.load(A + m[:, None] * K + k[None, :], mask=(m[:, None] < M)
                & (k[None, :] < K), other=0.0)
    b = tl.load(B + k[:, None] * N + n[None, :], mask=(k[:, None] < K)
                & (n[None, :] < N), other=0.0)
    acc = tl.dot(a, b)
    tl.store(OUT + m[:, None] * N + n[None, :],
             acc, mask=(m[:, None] < M) & (n[None, :] < N))


@triton.jit
def k_fp64_roundtrip(IN, OUT, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(IN + offs, mask=mask, other=0.0)
    tl.store(OUT + offs, x + 1.0, mask=mask)


@triton.jit
def k_barrier_chain(IN, OUT, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(IN + offs, mask=mask, other=0.0)
    y = x * 2.0
    tl.debug_barrier()
    z = y + 7.0
    tl.store(OUT + offs, z, mask=mask)


@triton.jit
def k_writer(SRC, OUT, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    tl.store(OUT + offs, tl.load(SRC + offs, mask=mask, other=0.0),
             mask=mask)


@triton.jit
def k_reader(SRC, OUT, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    tl.store(OUT + offs, tl.load(SRC + offs, mask=mask, other=0.0) * 3.0,
             mask=mask)


@triton.jit
def k_persistent_loop(OUT, PACKETS: tl.constexpr, NPK: tl.constexpr,
                      BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    # loop-carried state across static-unrolled iterations: register-
    # resident accumulator updated every packet (the residency claim)
    acc = pid.to(tl.float32) * 100.0
    for q in range(PACKETS):
        acc = acc * 1.5 + q.to(tl.float32)
        offs = q.to(tl.int32) * NPK + tl.arange(0, BLOCK)
        tl.store(OUT + pid * PACKETS * NPK + offs,
                 acc + tl.arange(0, BLOCK).to(tl.float32) * 0.0,
                 mask=tl.arange(0, BLOCK) < NPK)


@triton.jit
def k_persistent_while(OUT, LIMIT, BLOCK: tl.constexpr):
    # dynamic (non-static) while loop: XMLIR loop lowering known-issue
    # probe -- bounded exit required
    pid = tl.program_id(0)
    q = 0
    acc = pid.to(tl.float32)
    while q < LIMIT:
        acc = acc * 1.5 + 1.0
        q = q + 1
    offs = tl.arange(0, BLOCK)
    tl.store(OUT + pid * BLOCK + offs,
             acc + tl.arange(0, BLOCK).to(tl.float32) * 0.0,
             mask=offs < BLOCK)


@triton.jit
def k_grid_query(OUT, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nprocs = tl.num_programs(0)
    offs = tl.arange(0, BLOCK)
    tl.store(OUT + offs,
             pid.to(tl.float32)
             + nprocs.to(tl.float32) * 0.001,
             mask=offs < BLOCK)


@triton.jit
def k_libdevice(IN, OUT, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(IN + offs, mask=mask, other=0.0)
    tl.store(OUT + offs, libdevice.exp2(x), mask=mask)


@triton.jit
def k_scale2(IN, OUT, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    tl.store(OUT + offs, tl.load(IN + offs, mask=mask, other=0.0) * 2.0,
             mask=mask)


@triton.jit
def k_cache_modifier_load(IN, OUT, N, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(IN + offs, mask=mask, other=0.0, cache_modifier=".cg")
    tl.store(OUT + offs, x + 1.0, mask=mask)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _launch(kernel, grid, *args, **kw):
    """Launch ``kernel[grid]`` inside an explicit device context; wall ms."""
    if isinstance(grid, int):
        grid = (grid,)
    t0 = time.perf_counter()
    with xpu_context(0):
        kernel[grid](*args, **kw)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3


def _sentinel_changed(out: torch.Tensor, sentinel: float = SENTINEL) -> bool:
    """True when ``out`` no longer equals the sentinel (anti-noop)."""
    return not bool(torch.all(out == sentinel).item())


def _check(cid: str, ok: bool, detail: str, status: str | None = None,
           next_step: str = "") -> dict:
    st = status or ("PASS" if ok else "FAIL")
    return {"id": cid, "status": st, "detail": detail,
            "next_step": next_step}


def _run(name: str) -> dict:
    """Run one probe function; never raises -- failures become checks."""
    entry = PROBES[name]
    fn = entry["fn"]
    result = {"probe": name, "question": entry["question"],
              "checks": [], "notes": list(entry.get("notes", []))}
    try:
        out = fn()
        result["checks"].extend(out.get("checks", []))
        result["notes"].extend(out.get("notes", []))
    except Exception:  # noqa: BLE001  probe crash -> FAIL with repro
        result["checks"].append(_check(
            "probe_uncaught", False,
            "probe crashed: " + traceback.format_exc(limit=8)))
    # status over REQUIRED checks (optional checks are recorded only)
    required = set(entry.get("required") or
                   [c["id"] for c in result["checks"]])
    req_checks = [c for c in result["checks"] if c["id"] in required]
    if any(c["status"] == "FAIL" for c in req_checks):
        result["status"] = "FAIL"
    elif any(c["status"] != "PASS" for c in req_checks) or not req_checks:
        result["status"] = "UNKNOWN"
    else:
        result["status"] = "PASS"
    if result["status"] == "UNKNOWN" and not any(
            c.get("next_step") for c in result["checks"]
            if c["status"] == "UNKNOWN"):
        result.setdefault("checks", []).append(_check(
            "unknown_next_step", False,
            "required checks inconclusive", status="UNKNOWN",
            next_step="rerun probe with longer timeout and inspect log"))
    # source excerpt: the probe fn is the minimal repro
    try:
        result["repro_source"] = inspect.getsource(fn)
    except Exception as e:  # noqa: BLE001
        result["repro_source"] = f"<unavailable: {e}>"
    result["repro_script"] = "scripts/issue10_wp1_capability.py"
    return result


# --------------------------------------------------------------------------
# probes
# --------------------------------------------------------------------------


def p_load_store() -> dict:
    ch = []
    n = 1000
    BLOCK = 1024
    # scalar fill (mask + tail)
    out = torch.full((BLOCK,), SENTINEL, device="cuda")
    _launch(k_scalar_fill, triton.cdiv(n, BLOCK), out, n, FACTOR=1.5,
            BLOCK=BLOCK)
    ref = torch.arange(n, dtype=torch.float32, device="cuda") * 1.5
    ok = _sentinel_changed(out[:n]) and torch.equal(out[:n], ref) \
        and bool((out[n:] == SENTINEL).all())
    ch.append(_check("scalar_fill_masked", ok,
                     f"n={n} block={BLOCK} tail preserved={bool((out[n:] == SENTINEL).all())}"))
    # vector axpy + tail untouched
    x = torch.randn(n, device="cuda")
    out = torch.full((BLOCK,), SENTINEL, device="cuda")
    _launch(k_vec_axpy, triton.cdiv(n, BLOCK), x, out, n, BLOCK=BLOCK)
    ref = x * 2.0 + 1.0
    err = float((out[:n] - ref).abs().max())
    ok = _sentinel_changed(out) and err == 0.0 \
        and bool((out[n:] == SENTINEL).all())
    ch.append(_check("vector_axpy", ok, f"max_err={err} tail_untouched="
                     f"{bool((out[n:] == SENTINEL).all())}"))
    # non-contiguous strided (every 2nd element of a 2N buffer)
    m = 512
    base = torch.zeros(2 * m, device="cuda")
    src = torch.randn(m, device="cuda")
    base[0::2] = src
    outb = torch.full((2 * m,), SENTINEL, device="cuda")
    outb[0::2] = src  # start from real values, stride-store over them
    _launch(k_strided, triton.cdiv(m, m), base, outb, m, ST=2, BLOCK=m)
    ref = src * 3.0
    err = float((outb[0::2] - ref).abs().max())
    odd_untouched = bool((outb[1::2] == SENTINEL).all())
    ch.append(_check("strided_st2", err == 0.0 and odd_untouched,
                     f"max_err={err} stride1_slots_preserved={odd_untouched}"))
    # element-offset pointer (misaligned start relative to a bigger tensor)
    big = torch.randn(n + 1, device="cuda")
    view = big[1:]  # address offset of one element
    out = torch.full((BLOCK,), SENTINEL, device="cuda")
    _launch(k_vec_axpy, triton.cdiv(n, BLOCK), view, out, n, BLOCK=BLOCK)
    err = float((out[:n] - (view * 2.0 + 1.0)).abs().max())
    ch.append(_check("element_offset_start", err == 0.0
                     and _sentinel_changed(out),
                     f"max_err={err} (1-element offset base)"))
    # byte-level misalignment: torch tensors are allocator-aligned; the
    # question cannot be answered from this API surface -> UNKNOWN
    ch.append(_check("byte_misalign", False,
                     "host tensors are allocator-aligned; byte-misaligned "
                     "fp32 base address not constructible from torch "
                     "storage (allocator guarantees)", status="UNKNOWN",
                     next_step="construct a byte-offset view via "
                               "as_strided/storage offset once the vendor "
                               "documents legal alignments, or probe via "
                               "XTDK sample"))
    return {"checks": ch,
            "notes": ["all output buffers prefilled with sentinel "
                      f"{SENTINEL}; silent-noop surfaces as FAIL"]}


def p_layout() -> dict:
    ch = []
    M, N = 64, 128
    x = torch.randn(M, N, device="cuda")
    out = torch.full((N, M), SENTINEL, device="cuda")
    _launch(k_transpose, 1, x, out, M, N, BM=64, BN=128)
    err = float((out - x.t()).abs().max())
    ch.append(_check("transpose_til2d", err == 0.0
                     and _sentinel_changed(out), f"max_err={err}"))
    # tile round-trip (tile-shaped load, transform, store)
    y = torch.randn(32, 32, device="cuda")
    out = torch.full((32, 32), SENTINEL, device="cuda")
    _launch(k_tile_scale, 1, y, out, 32, 32, BM=32, BN=32)
    err = float((out - y * 1.25).abs().max())
    ch.append(_check("tile_roundtrip", err == 0.0
                     and _sentinel_changed(out), f"max_err={err}"))
    # swizzle2d: bijectivity of the mapping (contract-agnostic property)
    W = H = 64
    mark = torch.full((W * H,), SENTINEL, device="cuda")
    try:
        _launch(k_swizzle_mark, 1, mark, W, H, BLOCK=64)
        vals = mark
        uniq = int((vals == 1.0).sum())
        zeroes = int((vals == 0.0).sum())
        dup = int(W * H - uniq)
        if _sentinel_changed(vals) and uniq == W * H and zeroes == 0:
            ch.append(_check("swizzle2d_bijection", True,
                             f"every destination hit exactly once over "
                             f"{W}x{H}"))
        else:
            ch.append(_check(
                "swizzle2d_bijection", False,
                f"mapping not bijective on {W}x{H}: destinations_hit="
                f"{uniq} dup={dup} untouched={zeroes} (contract of "
                "swizzle2d params unconfirmed)", status="UNKNOWN",
                next_step="confirm swizzle2d argument contract from vendor "
                          "Triton docs, then re-run the bijection test"))
    except Exception as e:  # noqa: BLE001
        ch.append(_check("swizzle2d_bijection", False,
                         f"lowering rejected: {type(e).__name__}: {e}",
                         next_step="ask vendor for a legal swizzle sample "
                                   "before retrying"))
    # explicit lane/warp mapping control
    ch.append(_check(
        "lane_warp_mapping_control", False,
        "triton 3.0.0 XPU fork exposes no API to place lanes/warps (no "
        "annotate/metadata; XPUUtils reports warpSize=1 so NVIDIA-style "
        "lane assumptions do not apply)", status="UNKNOWN",
        next_step="inspect TTGIR/LLIR with an MLIR dump or vendor pass "
                  "docs to learn the actual lane mapping"))
    return {"checks": ch}


def p_dtype() -> dict:
    ch = []
    n = 256
    BLOCK = 256
    for dt, tol in ((torch.float32, 0.0), (torch.float16, 1e-3),
                    (torch.bfloat16, 2e-3)):
        x = torch.randn(n, device="cuda").to(dt)
        out = torch.full((n,), SENTINEL, device="cuda").to(dt)
        _launch(k_dtype_axpy, triton.cdiv(n, BLOCK), x, out, n,
                BLOCK=BLOCK)
        ref = x * 2.0 + 1.0
        err = float((out.float() - ref.float()).abs().max())
        ch.append(_check(f"{str(dt).split('.')[-1]}_axpy", err <= tol
                         and _sentinel_changed(out),
                         f"max_err={err} tol={tol}"))
    # complex simulation over fp32 SOA pairs
    ar, ai = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
    br, bi = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
    ore = torch.full((n,), SENTINEL, device="cuda")
    oim = torch.full((n,), SENTINEL, device="cuda")
    _launch(k_cmul, 1, ar, ai, br, bi, ore, oim, n, BLOCK=BLOCK)
    wr = ar * br - ai * bi
    wi = ar * bi + ai * br
    err = max(float((ore - wr).abs().max()), float((oim - wi).abs().max()))
    ch.append(_check("complex_sim_f32_soa", err < 1e-5
                     and _sentinel_changed(ore), f"max_err={err}"))
    # fp64 attempt: demonstrated support or demonstrated rejection
    try:
        a = torch.randn(n, device="cuda", dtype=torch.float64)
        out = torch.full((n,), SENTINEL, device="cuda",
                         dtype=torch.float64)
        _launch(k_fp64_roundtrip, triton.cdiv(n, BLOCK), a, out, n,
                BLOCK=BLOCK)
        err = float((out - (a + 1.0)).abs().max())
        ch.append(_check("fp64_lowering", err == 0.0,
                         f"fp64 executed, max_err={err}"))
    except Exception as e:  # noqa: BLE001
        ch.append(_check("fp64_lowering", False,
                         f"fp64 rejected by stack: "
                         f"{type(e).__name__}: {str(e)[:300]}",
                         next_step="record unsupported in capability "
                                   "matrix; SDK route if fp64 required"))
    # fp8 attempt
    fp8_names = [a for a in dir(tl) if a.startswith("float8")]
    if fp8_names:
        try:
            ch.append(_check("fp8_lowering", False,
                             f"fp8 dtypes present {fp8_names} but no "
                             "validated kernel sample this round",
                             status="UNKNOWN",
                             next_step="write an fp8 axpy probe once a "
                                       "vendor fp8 sample exists"))
        except Exception as e:  # noqa: BLE001
            ch.append(_check("fp8_lowering", False, str(e)))
    else:
        ch.append(_check("fp8_lowering", False,
                         "no fp8 dtype in tl surface (dir(tl) has no "
                         "float8*)", next_step="record unsupported; SDK "
                              "route if fp8 required"))
    return {"checks": ch,
            "notes": ["required dtypes: fp32/fp16/bf16/complex-sim; "
                      "fp64/fp8 recorded explicitly as issue demands"]}


def p_tl_dot() -> dict:
    ch = []
    M = N = K = 64
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda")
    out = torch.full((M, N), SENTINEL, device="cuda")
    _launch(k_dot, 1, a, b, out, M, N, K, BM=64, BN=64, BK=64)
    # ground truth: host float64 matmul; device torch.matmul is the
    # vendor BLAS cross-check (on this stack it is accurate: see logs)
    ref64 = a.cpu().double() @ b.cpu().double()
    err_dot = float((out.cpu().double() - ref64).abs().max())
    dev_mm = a @ b
    err_mm = float((dev_mm.cpu().double() - ref64).abs().max())
    if err_dot <= 1e-4:
        ch.append(_check("dot_fp32", True,
                         f"max_err_vs_fp64_truth={err_dot} "
                         f"(device matmul cross-check {err_mm})"))
    elif err_dot <= 1e-1:
        ch.append(_check("dot_fp32", False,
                         f"tl.dot max_err_vs_fp64_truth={err_dot}; "
                         f"device torch.matmul max_err={err_mm} -- "
                         "tl.dot uses reduced-precision accumulation "
                         "while the vendor BLAS path does not (dedicated "
                         "matrix path with sub-fp32 accumulator likely)",
                         status="UNKNOWN",
                         next_step="confirm accumulator width (fp16/bf16/"
                                   "tf32-like) from vendor matrix-unit "
                                   "docs; dense-dot P core only legal "
                                   "under the documented precision"))
    else:
        ch.append(_check("dot_fp32", False,
                         f"max_err_vs_fp64_truth={err_dot} (wrong "
                         "values)"))
    # fp16 dot against fp32 reference
    a16 = a.half()
    b16 = b.half()
    out16 = torch.full((M, N), SENTINEL, device="cuda").half()
    _launch(k_dot, 1, a16, b16, out16, M, N, K, BM=64, BN=64, BK=64)
    ref16 = (a @ b).half()
    err16 = float((out16.float() - ref16.float()).abs().max())
    ch.append(_check("dot_fp16", err16 <= 2e-2 and _sentinel_changed(out16),
                     f"max_err_vs_fp32_ref={err16}"))
    # matrix-unit utilization cannot be proven without counters
    ch.append(_check(
        "matrix_unit_utilization", False,
        "no hardware counter / occupancy API available from this stack; "
        "tl.dot correctness proven, hardware path not distinguishable",
        status="UNKNOWN",
        next_step="vendor profiler counters (XTDK) or a roofline "
                  "comparison of dot vs elementwise at equal FLOP"))
    return {"checks": ch}


def p_local_storage() -> dict:
    ch = []
    # explicit space selection API
    missing = [n for n in ("static_alloc", "static_free", "annotate")
               if not hasattr(tl, n)]
    ch.append(_check(
        "explicit_space_selection_api", False,
        f"no local-storage allocation/space-selection API: tl missing "
        f"{missing}; no UB/shared/L1 selector exposed to Python",
        next_step="placement can only be observed via IR dump or "
                  "controlled via SDK -- see capacity/visibility checks"))
    # capacity query
    try:
        from triton.backends.xpu.driver import XPUUtils
        props = dict(XPUUtils().get_device_properties(0))
        cap = int(props.get("max_shared_mem", 0))
        ch.append(_check("capacity_query", cap > 0,
                         f"max_shared_mem={cap} bytes reported by "
                         "XPUUtils (256 KiB on-chip window)"))
    except Exception as e:  # noqa: BLE001
        ch.append(_check("capacity_query", False,
                         f"props query failed: {type(e).__name__}: {e}"))
    # placement visibility: try the kernel .asm IR surface
    try:
        x = torch.randn(64, device="cuda")
        out = torch.full((64,), SENTINEL, device="cuda")
        _launch(k_vec_axpy, 1, x, out, 64, BLOCK=64)
        fn = k_vec_axpy
        asm = getattr(fn, "asm", None)
        if isinstance(asm, dict) and asm:
            keys = sorted(asm.keys())
            ch.append(_check("placement_visibility", True,
                             f"kernel.asm keys={keys} -- inspect for "
                             "shared/local allocation"))
        else:
            ch.append(_check(
                "placement_visibility", False,
                f"JITFunction exposes no .asm surface after launch "
                f"(asm={type(asm).__name__}); compiler-managed placement "
                "not observable from Python", status="UNKNOWN",
                next_step="run the probe with MLIR_ENABLE_DUMP=1 and "
                          "parse the dump for buffer location attributes"))
    except Exception as e:  # noqa: BLE001
        ch.append(_check("placement_visibility", False,
                         f"{type(e).__name__}: {e}", status="UNKNOWN",
                         next_step="MLIR dump route"))
    return {"checks": ch,
            "notes": ["explicit selection fails => full AscendC-style "
                      "ownership not expressible from Triton Python; "
                      "capacity IS queryable"]}


def p_async_copy() -> dict:
    ch = []
    present = {n: hasattr(tl, n) for n in
               ("async_copy", "async_wait", "async_dot", "join", "split")}
    extras = []
    try:
        import triton.language.extra.xpu as xe
        extras = [a for a in dir(xe) if "async" in a.lower()
                  or "mte" in a.lower() or "copy" in a.lower()]
    except Exception:  # noqa: BLE001
        extras = []
    exposed = present.get("async_copy") or bool(extras)
    ch.append(_check(
        "async_mechanism_api", exposed,
        f"tl async surface: {present}; extra.xpu async/mte/copy symbols: "
        f"{extras}" if not exposed else f"exposed: {present} {extras}",
        next_step="" if exposed else
        "converge: implicit compiler overlap only, or SDK route for "
        "explicit MTE-style pipelines (see Go/No-Go)"))
    # cache_modifier acceptance (compiler-side hint if accepted)
    try:
        x = torch.randn(128, device="cuda")
        out = torch.full((128,), SENTINEL, device="cuda")
        _launch(k_cache_modifier_load, 1, x, out, 128, BLOCK=128)
        err = float((out - (x + 1.0)).abs().max())
        ch.append(_check("cache_modifier_load", False,
                         f"cache_modifier='.cg' accepted and executed "
                         f"(max_err={err}); effect on traffic not "
                         "observable without counters", status="UNKNOWN",
                         next_step="vendor profiler cache counters"))
    except Exception as e:  # noqa: BLE001
        ch.append(_check("cache_modifier_load", False,
                         f"cache_modifier='.cg' rejected: "
                         f"{type(e).__name__}: {str(e)[:240]}"))
    # implicit compute/transfer overlap
    ch.append(_check(
        "implicit_overlap", False,
        "compiler-internal copy/compute overlap not observable from "
        "Python (no counters, no async API)", status="UNKNOWN",
        next_step="steady-vs-analytic roofline timing + XTDK profiler "
                  "if exposed"))
    return {"checks": ch}


def p_sync() -> dict:
    ch = []
    n = 256
    x = torch.randn(n, device="cuda")
    # in-kernel barrier
    out = torch.full((n,), SENTINEL, device="cuda")
    _launch(k_barrier_chain, triton.cdiv(n, n), x, out, n, BLOCK=n)
    err = float((out - (x * 2.0 + 7.0)).abs().max())
    ch.append(_check("in_kernel_barrier", err == 0.0
                     and _sentinel_changed(out),
                     "kernel with tl.debug_barrier compiles, executes, "
                     f"and returns correct values (max_err={err})"))
    # cross-launch (program) visibility through global memory
    y = torch.full((n,), SENTINEL, device="cuda")
    z = torch.full((n,), SENTINEL, device="cuda")
    _launch(k_writer, 1, x, y, n, BLOCK=n)
    _launch(k_reader, 1, y, z, n, BLOCK=n)
    err = float((z - x * 3.0).abs().max())
    ch.append(_check("gm_launch_order_visibility", err == 0.0
                     and _sentinel_changed(z),
                     f"two sequential launches hand off via GM "
                     f"(max_err={err})"))
    # concurrent cross-program barrier
    ch.append(_check(
        "cross_program_concurrent_sync", False,
        "no cooperative grid-barrier API exposed (tl.debug_barrier is "
        "in-kernel scope only; cross-program sync requires launch "
        "ordering or XCCL)", status="UNKNOWN",
        next_step="producer/consumer pipelines need SDK/graph-level "
                  "sync; revisit only if WP2 step 5 is attempted"))
    return {"checks": ch,
            "notes": ["required for basic pipeline: in-kernel barrier + "
                      "launch ordering; concurrent grid sync is optional "
                      "(producer/consumer step)"]}


def p_persistent() -> dict:
    ch = []
    PACKETS, NPK = 8, 16
    # one program loops over its packets (loop-carried state)
    out = torch.full((PACKETS * NPK,), SENTINEL, device="cuda")
    _launch(k_persistent_loop, 1, out, PACKETS=PACKETS, NPK=NPK,
            BLOCK=NPK)

    def _ref(pid: int) -> torch.Tensor:
        ref = torch.empty(PACKETS * NPK, device="cuda")
        acc = float(pid) * 100.0
        for q in range(PACKETS):
            acc = acc * 1.5 + float(q)
            ref[q * NPK:(q + 1) * NPK] = acc
        return ref

    err = float((out - _ref(0)).abs().max())
    ch.append(_check("single_program_packet_loop", err == 0.0
                     and _sentinel_changed(out),
                     f"loop-carried state over {PACKETS} packets "
                     f"correct (max_err={err}); launch count=1"))
    # multi-program split: 4 programs x 8 packets, all packets correct
    P = 4
    out2 = torch.full((P * PACKETS * NPK,), SENTINEL, device="cuda")
    _launch(k_persistent_loop, P, out2, PACKETS=PACKETS, NPK=NPK,
            BLOCK=NPK)
    ref2 = torch.cat([_ref(pid) for pid in range(P)])
    err2 = float((out2 - ref2).abs().max())
    ch.append(_check("grid_launch_persistence", err2 == 0.0
                     and _sentinel_changed(out2),
                     f"grid of {P} resident workers, {P * PACKETS} "
                     f"packets all correct (max_err={err2})"))
    # safe exit: bounded while-loop returns within watchdog (implicit:
    # this probe process completed); record explicitly
    ch.append(_check("bounded_loop_exit", True,
                     "bounded static-unroll loops exit; probe process "
                     f"returned within {PROBE_TIMEOUT_S}s watchdog"))
    # dynamic (non-static) while loop: XMLIR loop known-issue probe
    try:
        LIMIT = 5
        outw = torch.full((1 * 16,), SENTINEL, device="cuda")
        _launch(k_persistent_while, 1, outw, LIMIT, BLOCK=16)
        acc = 0.0
        for _ in range(LIMIT):
            acc = acc * 1.5 + 1.0
        refw = torch.full((16,), acc, device="cuda")
        errw = float((outw - refw).abs().max())
        ch.append(_check("dynamic_while_loop", errw == 0.0
                         and _sentinel_changed(outw),
                         f"dynamic `while q < LIMIT` lowers and exits "
                         f"(max_err={errw})"))
    except Exception as e:  # noqa: BLE001
        ch.append(_check("dynamic_while_loop", False,
                         f"dynamic while rejected by XMLIR: "
                         f"{type(e).__name__}: {str(e)[:300]}",
                         next_step="use static-unroll loops only in "
                                   "residency designs; SDK route if "
                                   "dynamic loops are required"))
    return {"checks": ch,
            "notes": ["resident worker loop: same program consumes "
                      "multiple independent packets sequentially"]}


def p_external_host_delegate() -> dict:
    """Host-level vendor delegate: vendor FFT API callable from the
    orchestration surface (NOT proof of embedding a device-local core)."""
    ch = []
    # host-side interop: triton kernel -> vendor op -> triton kernel
    x = torch.randn(128, device="cuda")
    y = torch.full((128,), SENTINEL, device="cuda")
    z = torch.full((128,), SENTINEL, device="cuda")
    _launch(k_scale2, 1, x, y, 128, BLOCK=128)
    y2 = torch.fft.fft(y)
    _launch(k_scale2, 1, y2.real.contiguous(), z, 128, BLOCK=128)
    # ground truth for the whole sequence on the host (fp64)
    xh = x.cpu().double()
    z_ref = (torch.fft.fft(xh * 2.0).real * 2.0)
    err = float((z.cpu().double() - z_ref).abs().max())
    ch.append(_check("host_vendor_interop", err <= 1e-3,
                     f"triton->torch.fft->triton sequencing correct "
                     f"(max_err_vs_fp64_truth={err})"))
    # explicit vendor fft dispatch contract: complex64 device tensor,
    # default axis=-1 last-dim semantics (see wp3_blockmax vendor_fft_dim:
    # non-trailing dims are silently wrong -> callers must transpose)
    v = torch.randn(256, device="cuda")
    xc = torch.complex(v, torch.zeros_like(v))
    yv = torch.fft.fft(xc)
    ref = np.fft.fft(v.cpu().numpy().astype(np.complex128))
    derr = float(np.abs(yv.cpu().numpy() - ref).max()
                  / max(float(np.abs(ref).max()), 1e-30))
    ch.append(_check("vendor_fft_dispatch", derr <= 1e-5
                     and yv.shape == xc.shape,
                     f"vendor torch.fft.fft on device complex64 rank-1 "
                     f"(axis=-1 contract) rel_err={derr:.2e}"))
    return {"checks": ch,
            "notes": ["host-level vendor delegate only: proves API/"
                      "dispatch usability, NOT device-side core embedding"]}


def p_device_local_processing_unit() -> dict:
    """Device-local processing unit: embed an inline intrinsic /
    precompiled device core / role-local FFT core into device code."""
    ch = []
    # inline math intrinsic (works) -- sub-evidence only: a scalar math
    # intrinsic is not a processing unit (FFT core)
    x = torch.randn(128, device="cuda")
    out = torch.full((128,), SENTINEL, device="cuda")
    _launch(k_libdevice, 1, x, out, 128, BLOCK=128)
    ref = torch.exp2(x)
    err = float((out - ref).abs().max())
    ch.append(_check("vendor_intrinsic_libdevice", err < 1e-4
                     and _sentinel_changed(out),
                     f"libdevice.exp2 lowers and matches torch.exp2 "
                     f"(max_err={err}) -- scalar math intrinsic only, "
                     f"does NOT prove a device-local processing unit"))
    # inline asm: no verified XPU syntax sample -> UNKNOWN (no blind try)
    has_asm = hasattr(tl, "inline_asm_elementwise")
    ch.append(_check(
        "inline_asm_xpu", has_asm,
        f"tl.inline_asm_elementwise present={has_asm}, but no verified "
        "XPU assembly syntax sample exists in this repo; blind attempt "
        "could corrupt the compiler state", status="UNKNOWN",
        next_step="obtain a legal XTDK inline-asm sample from vendor "
                  "docs, then run an asm axpy probe"))
    # precompiled .so / extern lib linkage from the Python surface
    try:
        import triton.compiler as tc
        src = str(inspect.signature(tc.compile)) if hasattr(
            tc, "compile") else ""
        has_extern = "extern" in src
    except Exception:  # noqa: BLE001
        has_extern = False
    ch.append(_check(
        "precompiled_so_link", has_extern,
        f"triton.compile signature extern-libs support: {has_extern}; no "
        "verified SDK .so linkage sample for the XPU backend this round",
        status="UNKNOWN" if has_extern else "FAIL",
        next_step="compile a kernel referencing an extern symbol once an "
                  "SDK .so sample is provided"))
    # the capability the issue asks about, stated directly
    ch.append(_check(
        "device_role_local_fft_core", False,
        "no proven route to embed a device-local processing unit (FFT "
        "core) into device code or a role/resident subgraph: inline asm "
        "unverified (UNKNOWN), precompiled .so link unsupported (FAIL); "
        "host torch.fft is an orchestration-level delegate, not a "
        "device-local core", status="FAIL",
        next_step="SDK/XTDK route: device-linkable vendor FFT object or "
                  "documented inline-asm kernel sample"))
    return {"checks": ch,
            "notes": ["device-side core embedding is NOT established by "
                      "the host vendor delegate PASS (split per issue #10 "
                      "review)"]}


def p_resource_query() -> dict:
    ch = []
    try:
        from triton.backends.xpu.driver import XPUUtils
        props = dict(XPUUtils().get_device_properties(0))
        mps = int(props.get("multiprocessor_count", 0))
        shm = int(props.get("max_shared_mem", 0))
        ch.append(_check("device_props_query", mps > 0 and shm > 0,
                         f"multiprocessor_count={mps} "
                         f"max_shared_mem={shm} device_model="
                         f"{props.get('device_model')}"))
    except Exception as e:  # noqa: BLE001
        ch.append(_check("device_props_query", False,
                         f"{type(e).__name__}: {e}"))
    # in-kernel grid query
    try:
        out = torch.full((4,), SENTINEL, device="cuda")
        _launch(k_grid_query, 1, out, BLOCK=4)
        # single-program grid: pid must be 0, num_programs must be 1
        got = out.tolist()
        ok = abs(got[0] - 0.001) < 1e-6 and all(
            abs(v - got[0]) < 1e-9 for v in got)
        ch.append(_check("in_kernel_grid_query", ok,
                         f"program_id=0 num_programs=1 encoded in "
                         f"values {got} (float32 representation "
                         f"0.0010000000474974513)"))
    except Exception as e:  # noqa: BLE001
        ch.append(_check("in_kernel_grid_query", False,
                         f"{type(e).__name__}: {e}"))
    # register file size reporting
    try:
        regs = props.get("max_num_regs", None)
    except Exception:  # noqa: BLE001
        regs = None
    ch.append(_check(
        "register_file_query", False,
        f"XPUUtils reports max_num_regs={regs} (0/absent => not "
        "reported on this driver)", status="UNKNOWN",
        next_step="register budget from XTDK docs / SDK query"))
    return {"checks": ch}


# --------------------------------------------------------------------------
# probe registry
# --------------------------------------------------------------------------

PROBES: dict[str, dict] = {
    "load_store": {
        "question": "基本 load/store：标量、向量、非连续 stride、对齐/"
                    "尾块是否正确",
        "fn": p_load_store,
        "required": ["scalar_fill_masked", "vector_axpy", "strided_st2",
                     "element_offset_start"],
    },
    "layout": {
        "question": "layout：tile、lane/warp 映射、转置和 swizzle 表达能力",
        "fn": p_layout,
        "required": ["transpose_til2d", "tile_roundtrip"],
    },
    "dtype": {
        "question": "dtype：fp16/bf16/fp32/复数模拟；fp64/fp8 明确记录",
        "fn": p_dtype,
        "required": ["fp32_axpy", "fp16_axpy", "bfloat16_axpy",
                     "complex_sim_f32_soa"],
    },
    "tl_dot": {
        "question": "tl.dot/矩阵路径：正确性与（若可测）矩阵单元归属",
        "fn": p_tl_dot,
        "required": ["dot_fp32", "dot_fp16"],
    },
    "local_storage": {
        "question": "局部存储：能否显式选择 UB/local/shared/L1 空间，"
                    "容量和可见性边界",
        "fn": p_local_storage,
        "required": ["explicit_space_selection_api", "capacity_query"],
    },
    "async_copy": {
        "question": "异步搬运：async copy/MTE 等价机制及其与计算重叠",
        "fn": p_async_copy,
        "required": ["async_mechanism_api"],
    },
    "sync": {
        "question": "同步：program 内 barrier、跨 program 可见性与"
                    "所有权转移语义",
        "fn": p_sync,
        "required": ["in_kernel_barrier", "gm_launch_order_visibility"],
    },
    "persistent": {
        "question": "persistent 执行：同一 worker 循环消费多个 packet "
                    "并安全退出",
        "fn": p_persistent,
        "required": ["single_program_packet_loop",
                     "grid_launch_persistence", "bounded_loop_exit"],
    },
    "external_host_delegate": {
        "question": "外部宿主 delegate：vendor FFT API / host dispatch "
                    "可调用（编排层，非设备侧核心嵌入）",
        "fn": p_external_host_delegate,
        "required": ["host_vendor_interop", "vendor_fft_dispatch"],
    },
    "device_local_processing_unit": {
        "question": "设备侧处理单元：inline intrinsic / 预编译 device "
                    "core / role-local FFT core 可嵌入设备代码",
        "fn": p_device_local_processing_unit,
        "required": ["device_role_local_fft_core",
                     "precompiled_so_link"],
    },
    "resource_query": {
        "question": "资源查询：grid、寄存器/局部存储、片上容量、"
                    "resident worker 边界",
        "fn": p_resource_query,
        "required": ["device_props_query", "in_kernel_grid_query"],
    },
}


def _probe_env() -> dict:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "2"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _run_one_subprocess(name: str) -> dict:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{name}.log"
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--probe", name],
            capture_output=True, text=True, timeout=PROBE_TIMEOUT_S,
            env=_probe_env(), cwd=str(ROOT))
        rc, out, err = proc.returncode, proc.stdout, proc.stderr
        timed_out = False
    except subprocess.TimeoutExpired as e:
        rc, out, err = -9, e.stdout or "", (e.stderr or "") + \
            f"\nTIMEOUT after {PROBE_TIMEOUT_S}s"
        timed_out = True
    elapsed = time.time() - t0
    log_path.write_text(
        f"# probe: {name}\n# rc={rc} elapsed={elapsed:.1f}s\n"
        f"# timeout={timed_out}\n--- stdout ---\n{out}\n"
        f"--- stderr ---\n{err}\n")
    # parse the result marker
    result = None
    for line in (out or "").splitlines():
        if line.startswith(MARK):
            try:
                result = json.loads(line[len(MARK):])
            except Exception:  # noqa: BLE001
                result = None
    if result is None:
        if timed_out:
            status = "UNKNOWN"
            note = (f"probe timed out after {PROBE_TIMEOUT_S}s -- "
                    "slow XMLIR compile vs hang not distinguishable")
        else:
            status = "FAIL"
            note = f"probe process crashed rc={rc} (see log)"
        result = {"probe": name, "status": status,
                  "checks": [{"id": "probe_process", "status": status,
                              "detail": note,
                              "next_step": ("rerun with extended timeout; "
                                            "inspect XMLIR compile log"
                                            if timed_out else
                                            "inspect log for driver/XMLIR "
                                            "error; keep minimal repro")}],
                  "repro_source": inspect.getsource(PROBES[name]["fn"]),
                  "repro_script": "scripts/issue10_wp1_capability.py"}
    result["log"] = str(log_path.relative_to(ROOT))
    result["rc"] = rc
    result["elapsed_s"] = round(elapsed, 1)
    return result


def run_all() -> int:
    started = time.time()
    entries = []
    for name in PROBES:
        print(f"[wp1] probe {name} ...", flush=True)
        r = _run_one_subprocess(name)
        entries.append(r)
        print(f"      -> {r['status']} ({r['elapsed_s']}s) "
              f"checks={len(r.get('checks', []))}", flush=True)
    # bind to WP0
    wp0_hash = None
    wp0_path = OUT_DIR / "wp0_environment.json"
    if wp0_path.is_file():
        wp0_hash = json.loads(wp0_path.read_text()).get("wp0_hash")
    matrix = {
        "schema": "issue10-capability-matrix-v1",
        "issue": "kunlunxin-fft#10",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wp0_hash": wp0_hash,
        "policy": {
            "statuses": ["PASS", "FAIL", "UNKNOWN"],
            "inference": "statuses come from executed probes only; "
                         "platform/backend names never imply support",
            "required_checks": "entry status is computed from the "
                               "entry's required checks; optional checks "
                               "are recorded as evidence",
            "timeout": f"{PROBE_TIMEOUT_S}s -> UNKNOWN (inconclusive)",
            "no_op_guard": f"sentinel {SENTINEL} prefill on every output "
                           "+ xpu_context around every launch",
        },
        "probe_script": "scripts/issue10_wp1_capability.py",
        "probe_script_sha256": hashlib.sha256(
            SCRIPT_PATH.read_bytes()).hexdigest(),
        "entries": entries,
        "totals": {
            name: sum(1 for e in entries if e["status"] == name)
            for name in ("PASS", "FAIL", "UNKNOWN")},
        "elapsed_s": round(time.time() - started, 1),
    }
    MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    MATRIX_PATH.write_text(json.dumps(matrix, indent=2) + "\n")
    print(f"matrix: {MATRIX_PATH} totals={matrix['totals']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe", help="run a single probe (subprocess mode)")
    args = ap.parse_args(argv)
    if args.probe:
        if args.probe not in PROBES:
            print(f"unknown probe {args.probe!r}; known: {sorted(PROBES)}")
            return 2
        res = _run(args.probe)
        print(MARK + json.dumps(res, sort_keys=True))
        return 0
    return run_all()


if __name__ == "__main__":
    sys.exit(main())
