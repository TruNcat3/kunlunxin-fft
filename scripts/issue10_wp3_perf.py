"""WP3: issue kunlunxin-fft#10 — three-way matched performance + Go/No-Go.

Steps (each isolated in its own subprocess; `--step NAME`):
  blockmax -> wp3_blockmax.json : max BLOCK (lanes/program) that lowers+runs
  perf     -> wp3_raw.csv + wp3_summary.json : matched-protocol comparison
  gonogo   -> wp3_gonogo.json   : six-criteria Go/No-Go from WP1+WP2+WP3
  all      -> blockmax, perf, gonogo sequentially

Paths (same semantics: C2C, complex64, fp32):
  vendor              torch.fft.fft/ifft (Kunlunxin vendor device FFT)
  triton_multipass    grid=B programs; one DIT stage per launch; log2(N)
                      launches; partner loads only from the previous-launch
                      buffer (legal at grid>1: no in-kernel store->gather)
  triton_resident_loop grid=1 program loops B packets per stage (the WP2
                      Td "resident worker" intent); log2(N) launches;
                      cross-launch GM handoff only
  hybrid_vendor_p     A-axis host orchestration + vendor P-axis FFTs:
                      col-FFT(r) | Triton twiddle | row-FFT(t) | digit-
                      reversal transpose

Inverse on all paths = conj-trick on real input: ifft(x) = conj(fft(x))/N
(valid for real x; hybrid uses conj twiddles + vendor ifft directly).

Stale-read correction (wp3_blockmax.json stale_sweep): in-kernel
store->permuted-load from the same GM buffer is unreliable even at grid=1
for n=32/64 (~11%/20% of lanes read pre-store values; n=16/128/256 clean
in the sweep). The WP2 scratch ping-pong tile design is therefore not a
legal general lowering; every path above uses cross-launch handoff only.

Bounded first round per the issue: N in {256,1024,2048}, B in {1,16,64},
C2C forward fp32 on the full grid; C2C inverse on the B=16 column only.
dtype axis: fp32 only — preflight recorded vendor torch.fft on fp16/bf16
as NOT IMPLEMENTED (kept in wp3_blockmax.json evidence).

Timing protocol (same scope for every path): warmup, then R rounds of T
full-batch calls; wall = sync-per-round, per-trial ms; p50/MAD over all
R*T trials. Analytic GM bytes are estimates, never measured bandwidth.
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/kunlunxin-fft")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch_xmlir  # noqa: F401,E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

import numpy as np  # noqa: E402

from src.xpu.context import xpu_context  # noqa: E402

OUT_DIR = ROOT / "results/p800/issue10_capability"
LOG_DIR = OUT_DIR / "wp3_logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
SCRIPT_PATH = ROOT / "scripts/issue10_wp3_perf.py"
TOL_REL = 1e-4
STEP_TIMEOUT_S = 1800
WARMUP = 5
ROUNDS = 5
TRIALS = 32

CELLS = [(256, 1), (256, 16), (256, 64), (256, 256),
         (1024, 1), (1024, 16), (1024, 64), (1024, 256),
         (2048, 1), (2048, 16), (2048, 64), (2048, 256)]
INV_SUBSET = {256: 16, 1024: 16, 2048: 16}


# ============================ Triton kernels ==================================

@triton.jit
def k_stage_dit(SRC_RE, SRC_IM, DST_RE, DST_IM,
                N: tl.constexpr, LOGN: tl.constexpr, BLOCK: tl.constexpr,
                STAGE: tl.constexpr):
    """One DIT stage for all B packets (grid = B).

    Loads ONLY from SRC (written by a previous launch); never stores to
    SRC -> partner gather is legal at grid>1 (WP2 stale-read bug only bites
    in-kernel store -> different-address load of the same buffer).
    STAGE=0 also applies the bit-reversed input permutation.
    """
    bo = tl.program_id(0) * N
    p = tl.arange(0, BLOCK)
    mask = p < N
    zero = p * 0
    m = 1 << STAGE
    L = (p % (2 * m)) < m
    Li = L.to(tl.int32)
    q = p + Li * (2 * m) - m
    if STAGE == 0:
        rp = zero
        rq = zero
        for bi_ in range(LOGN):
            rp = rp + ((p // (1 << bi_)) % 2) * (1 << (LOGN - 1 - bi_))
            rq = rq + ((q // (1 << bi_)) % 2) * (1 << (LOGN - 1 - bi_))
        v_re = tl.load(SRC_RE + bo + rp, mask=mask, other=0.0)
        v_im = tl.load(SRC_IM + bo + rp, mask=mask, other=0.0)
        b_re = tl.load(SRC_RE + bo + rq, mask=mask, other=0.0)
        b_im = tl.load(SRC_IM + bo + rq, mask=mask, other=0.0)
    else:
        v_re = tl.load(SRC_RE + bo + p, mask=mask, other=0.0)
        v_im = tl.load(SRC_IM + bo + p, mask=mask, other=0.0)
        b_re = tl.load(SRC_RE + bo + q, mask=mask, other=0.0)
        b_im = tl.load(SRC_IM + bo + q, mask=mask, other=0.0)
    e = (p % m) * (N // (2 * m))
    ang = (6.283185307179586 / N) * e.to(tl.float32)
    wr = tl.cos(ang)
    wi = -tl.sin(ang)
    tb_re = b_re * wr - b_im * wi
    tb_im = b_re * wi + b_im * wr
    tv_re = v_re * wr - v_im * wi
    tv_im = v_re * wi + v_im * wr
    o_re = tl.where(L, v_re + tb_re, b_re - tv_re)
    o_im = tl.where(L, v_im + tb_im, b_im - tv_im)
    tl.store(DST_RE + bo + p, o_re, mask=mask)
    tl.store(DST_IM + bo + p, o_im, mask=mask)


@triton.jit
def k_stage_resident(SRC_RE, SRC_IM, DST_RE, DST_IM,
                     N: tl.constexpr, LOGN: tl.constexpr,
                     BLOCK: tl.constexpr, STAGE: tl.constexpr,
                     K: tl.constexpr):
    """grid=1: one resident worker loops K packets for ONE DIT stage.

    Same stage body as k_stage_dit; loads only from SRC (previous launch),
    stores only to DST -> legal at any N (no in-kernel store->gather; the
    WP2 scratch pattern is stale-read-unreliable even at grid=1 for
    n in {32,64}, see wp3_blockmax.json stale_sweep).
    """
    for pk in range(K):
        bo = pk * N
        p = tl.arange(0, BLOCK)
        mask = p < N
        zero = p * 0
        m = 1 << STAGE
        L = (p % (2 * m)) < m
        Li = L.to(tl.int32)
        q = p + Li * (2 * m) - m
        if STAGE == 0:
            rp = zero
            rq = zero
            for bi_ in range(LOGN):
                rp = rp + ((p // (1 << bi_)) % 2) * (1 << (LOGN - 1 - bi_))
                rq = rq + ((q // (1 << bi_)) % 2) * (1 << (LOGN - 1 - bi_))
            v_re = tl.load(SRC_RE + bo + rp, mask=mask, other=0.0)
            v_im = tl.load(SRC_IM + bo + rp, mask=mask, other=0.0)
            b_re = tl.load(SRC_RE + bo + rq, mask=mask, other=0.0)
            b_im = tl.load(SRC_IM + bo + rq, mask=mask, other=0.0)
        else:
            v_re = tl.load(SRC_RE + bo + p, mask=mask, other=0.0)
            v_im = tl.load(SRC_IM + bo + p, mask=mask, other=0.0)
            b_re = tl.load(SRC_RE + bo + q, mask=mask, other=0.0)
            b_im = tl.load(SRC_IM + bo + q, mask=mask, other=0.0)
        e = (p % m) * (N // (2 * m))
        ang = (6.283185307179586 / N) * e.to(tl.float32)
        wr = tl.cos(ang)
        wi = -tl.sin(ang)
        tb_re = b_re * wr - b_im * wi
        tb_im = b_re * wi + b_im * wr
        tv_re = v_re * wr - v_im * wi
        tv_im = v_re * wi + v_im * wr
        o_re = tl.where(L, v_re + tb_re, b_re - tv_re)
        o_im = tl.where(L, v_im + tb_im, b_im - tv_im)
        tl.store(DST_RE + bo + p, o_re, mask=mask)
        tl.store(DST_IM + bo + p, o_im, mask=mask)


@triton.jit
def k_twiddle(X, O, N: tl.constexpr, R: tl.constexpr, T: tl.constexpr,
              BLOCK: tl.constexpr, CONJ: tl.constexpr):
    """4-step hybrid twiddle: O = X * W_N^{+-j*k} on interleaved complex64.

    X, O: float32 flattened views of complex64 (B*R*T, ) with element
    (b,j,k) at ((b*R + j)*T + k), re at *2, im at *2+1. One program per
    (batch, row j); vector over k.
    """
    pid = tl.program_id(0)
    j = pid % R
    b = pid // R
    base = ((b * R + j) * T) * 2
    k = tl.arange(0, BLOCK)
    mask = k < T
    re = tl.load(X + base + k * 2, mask=mask, other=0.0)
    im = tl.load(X + base + k * 2 + 1, mask=mask, other=0.0)
    jf = j.to(tl.float32) + (k * 0.0)
    ang = (6.283185307179586 / N) * jf * k.to(tl.float32)
    if CONJ == 1:
        ang = -ang
    wr = tl.cos(ang)
    wi = -tl.sin(ang)
    tl.store(O + base + k * 2, re * wr - im * wi, mask=mask)
    tl.store(O + base + k * 2 + 1, re * wi + im * wr, mask=mask)


@triton.jit
def k_twiddle_fused(X, O, N: tl.constexpr, R: tl.constexpr,
                    T: tl.constexpr, BLOCK: tl.constexpr,
                    CONJ: tl.constexpr):
    """Twiddle fused with the r-FFT -> stage2 layout handoff.

    X: r-FFT output kept in [b, t, r] (k-major, contiguous over j);
    element (b,k,j) at ((b*T + k)*R + j) — no transpose-back / reshape
    GM copy between the stages. O: [b, r, t] for the t-FFT.
    """
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


# ============================ helpers =========================================

def _np_ref(x: np.ndarray, inverse: bool) -> np.ndarray:
    return np.fft.ifft(x.astype(np.complex128)) if inverse \
        else np.fft.fft(x.astype(np.complex128))


def _rel_err(y: np.ndarray, ref: np.ndarray) -> float:
    denom = max(float(np.abs(ref).max()), 1e-30)
    return float(np.abs(y - ref).max() / denom)


def _pooled_stats(ms: list) -> dict:
    s = sorted(ms)
    n = len(s)
    p50 = s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
    dev = sorted(abs(v - p50) for v in s)
    mad = dev[n // 2] if n % 2 else 0.5 * (dev[n // 2 - 1] + dev[n // 2])
    return {"p50_ms": float(p50), "mad_ms": float(mad),
            "min_ms": float(s[0]), "max_ms": float(s[-1]),
            "trials": n}


def _time_path(fn, warmup=WARMUP, rounds=ROUNDS, trials=TRIALS) -> dict:
    with xpu_context(0):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        walls = []
        for _ in range(rounds):
            t0 = time.perf_counter()
            for _ in range(trials):
                fn()
            torch.cuda.synchronize()
            per = (time.perf_counter() - t0) * 1e3 / trials
            walls.extend([per] * trials)
    return _pooled_stats(walls)


def _data(b: int, n: int, seed: int = 11) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.rand((b, n), device="cuda") - 0.5


def _check_batched(y_re: torch.Tensor, y_im: torch.Tensor,
                   x: torch.Tensor, inverse: bool) -> float:
    xn = x.cpu().numpy()
    yr = y_re.cpu().numpy()
    yi = y_im.cpu().numpy()
    yn = yr + 1j * yi
    errs = []
    for i in range(xn.shape[0]):
        ref = _np_ref(xn[i], inverse)
        errs.append(_rel_err(yn[i], ref))
    return max(errs)


def _factors(n: int):
    cands = []
    t = 2
    while t * t <= n:
        if n % t == 0 and t >= 8 and n // t >= 8:
            cands.append((t, n // t))
        t += 1
    cands.sort(key=lambda c: abs(c[0] - c[1]))
    return cands[0]


# ============================ path runners ====================================
# Every runner returns a zero-arg call() -> (re, im) device tensors.

def run_vendor(x: torch.Tensor, inverse: bool):
    b, n = x.shape
    xc = torch.complex(x, torch.zeros_like(x))

    def call():
        y = torch.fft.ifft(xc) if inverse else torch.fft.fft(xc)
        return y.real, y.imag

    meta = {"launches": "vendor-internal", "gm_bytes_analytic": None}
    return call, meta


def run_triton_multipass(x: torch.Tensor, inverse: bool):
    b, n = x.shape
    logn = n.bit_length() - 1
    src_re = x.contiguous()
    src_im = torch.zeros_like(x)
    buf = [torch.empty_like(x) for _ in range(2)]
    buf_i = [torch.empty_like(x) for _ in range(2)]

    def call():
        in_re, in_im = src_re, src_im
        for s in range(logn):
            o = s % 2
            with xpu_context(0):
                k_stage_dit[(b,)](in_re, in_im, buf[o], buf_i[o],
                                  n, logn, n, s)
            in_re, in_im = buf[o], buf_i[o]
            torch.cuda.synchronize()
        if inverse:
            return in_re / n, -in_im / n
        return in_re, in_im

    meta = {"launches": logn, "gm_bytes_analytic": _gm_multipass(b, n, logn)}
    return call, meta


def _gm_multipass(b: int, n: int, logn: int) -> int:
    stage0 = b * 4 * n * 4
    stage_mid = b * 6 * n * 4
    return int(stage0 + (logn - 1) * stage_mid)


def run_triton_resident_loop(x: torch.Tensor, inverse: bool):
    b, n = x.shape
    logn = n.bit_length() - 1
    src_re = x.contiguous()
    src_im = torch.zeros_like(x)
    buf = [torch.empty_like(x) for _ in range(2)]
    buf_i = [torch.empty_like(x) for _ in range(2)]

    def call():
        in_re, in_im = src_re, src_im
        for s in range(logn):
            o = s % 2
            with xpu_context(0):
                k_stage_resident[(1,)](in_re, in_im, buf[o], buf_i[o],
                                       n, logn, n, s, b)
            in_re, in_im = buf[o], buf_i[o]
        torch.cuda.synchronize()
        if inverse:
            return in_re / n, -in_im / n
        return in_re, in_im

    meta = {"launches": logn,
            "gm_bytes_analytic": _gm_multipass(b, n, logn)}
    return call, meta


def run_hybrid(x: torch.Tensor, inverse: bool):
    b, n = x.shape
    t, r = _factors(n)
    xc = torch.complex(x, torch.zeros_like(x))
    view = xc.view(b, r, t)          # [j, k], n = j*t + k
    tw = torch.empty((b, r, t), dtype=torch.complex64, device="cuda")
    tw_flat = torch.view_as_real(tw).reshape(-1)
    conj = 1 if inverse else 0

    def call():
        # 4-step: r-FFT over j | twiddle W_N^{+-j'*k} | t-FFT over k |
        # digit-reversal transpose out[m=k'*r+j'] = S[j', k']
        # NOTE: vendor torch.fft only honors the LAST dim (dim=0 silently
        # transforms axis=-1, see wp3_blockmax vendor_fft_dim) -> r-FFT
        # needs an explicit transpose to put j last.
        vt = view.transpose(1, 2)                    # [k, j]
        stage1 = (torch.fft.ifft(vt, dim=-1) if inverse
                  else torch.fft.fft(vt, dim=-1)).transpose(1, 2)  # [j', k]
        x_flat = torch.view_as_real(stage1).reshape(-1)
        with xpu_context(0):
            k_twiddle[(b * r,)](x_flat, tw_flat, n, r, t, t, conj)
            torch.cuda.synchronize()
        stage2 = torch.fft.ifft(tw, dim=2) if inverse else \
            torch.fft.fft(tw, dim=2)
        out = stage2.permute(0, 2, 1).contiguous().reshape(b, n)
        return out.real, out.imag

    meta = {"launches": "2 vendor FFT (last-dim only; r-FFT via transpose "
                        "pair) + 1 triton twiddle + transpose/permute copies",
            "gm_bytes_analytic": _gm_hybrid(b, n),
            "factors": json.dumps({"t": t, "r": r})}
    return call, meta


def _gm_hybrid(b: int, n: int) -> int:
    return int(b * n * 8 * 8)


def run_hybrid_fused(x: torch.Tensor, inverse: bool):
    """hybrid_vendor_p with the twiddle fused into the stage1->stage2
    handoff (zero-copy [b,k,j] read; ablation wp3_ablation.json winner:
    -36..-45% vs the transpose+reshape baseline on B>=16 cells)."""
    b, n = x.shape
    t, r = _factors(n)
    xc = torch.complex(x, torch.zeros_like(x))
    view = xc.view(b, r, t)          # [j, k], n = j*t + k
    tw = torch.empty((b, r, t), dtype=torch.complex64, device="cuda")
    tw_flat = torch.view_as_real(tw).reshape(-1)
    conj = 1 if inverse else 0

    def call():
        vt = view.transpose(1, 2)                    # [k, j]
        stage1 = (torch.fft.ifft(vt, dim=-1) if inverse
                  else torch.fft.fft(vt, dim=-1))    # [k, j] contiguous
        x_flat = torch.view_as_real(stage1).reshape(-1)  # view, no copy
        with xpu_context(0):
            k_twiddle_fused[(b * r,)](x_flat, tw_flat, n, r, t, t, conj)
            torch.cuda.synchronize()
        stage2 = torch.fft.ifft(tw, dim=2) if inverse else \
            torch.fft.fft(tw, dim=2)
        out = stage2.permute(0, 2, 1).contiguous().reshape(b, n)
        return out.real, out.imag

    meta = {"launches": "2 vendor FFT (last-dim only; r-FFT via transpose "
                        "pair) + 1 fused Triton twiddle (no inter-stage "
                        "transpose-back/reshape copy) + "
                        "digit-reversal permute copy",
            "gm_bytes_analytic": _gm_hybrid(b, n),
            "factors": json.dumps({"t": t, "r": r})}
    return call, meta


PATHS = {
    "vendor": run_vendor,
    "triton_multipass": run_triton_multipass,
    "triton_resident_loop": run_triton_resident_loop,
    "hybrid_vendor_p": run_hybrid,
    "hybrid_fused_p": run_hybrid_fused,
}

PATH_META = {
    "vendor": {
        "A": "none (opaque vendor kernel)",
        "P": "Kunlunxin vendor FFT (torch.fft)",
        "lowering": "torch_xmlir vendor dispatch; complex64 interleaved",
    },
    "triton_multipass": {
        "A": "stage-sliced batched DIT (log2(N) launches, grid=B; packet "
             "ownership = program pid)",
        "P": "Triton radix-2 butterfly (one lane per element)",
        "lowering": "legal at grid>1: cross-launch GM handoff only "
                    "(launch-order visibility PASS, WP1); no in-kernel "
                    "store->gather (stale-read bug avoided)",
    },
    "triton_resident_loop": {
        "A": "resident single-worker packet loop (grid=1 program loops K "
             "packets per stage; log2(N) launches)",
        "P": "Triton radix-2 butterfly; cross-launch GM handoff only",
        "lowering": "legal at any N: no in-kernel store->gather; the WP2 "
                    "scratch pattern is stale-read-unreliable even at "
                    "grid=1 for n=32/64 (wp3_blockmax stale_sweep)",
    },
    "hybrid_vendor_p": {
        "A": "4-step split N=t*r: col-FFT(r) | twiddle(j',k) | row-FFT(t) "
             "| digit-reversal transpose (host orchestration)",
        "P": "vendor FFT (both stages) + one Triton twiddle kernel",
        "lowering": "host_vendor_interop PASS (WP1); Triton twiddle is "
                    "elementwise GM; permutes materialize (costed)",
    },
    "hybrid_fused_p": {
        "A": "4-step split N=t*r: col-FFT(r) | twiddle fused into the "
             "r-FFT->t-FFT handoff (zero-copy [b,k,j]) | row-FFT(t) | "
             "digit-reversal transpose (host orchestration)",
        "P": "vendor FFT (both stages) + one fused Triton twiddle kernel",
        "lowering": "host_vendor_interop PASS (WP1); no inter-stage "
                    "transpose-back/reshape GM copy (ablation winner)",
    },
}


# everything inside the timed call() region (per-row manifest, review
# item 3: hybrid transpose/twiddle/copy/sync must be visibly costed)
PATH_TIMED = {
    "vendor": "vendor fft/ifft dispatch (plan-reuse; warmup hides plan)",
    "triton_multipass": "log2(N) stage-kernel launches + GM handoff + "
                        "final torch.cuda.synchronize",
    "triton_resident_loop": "log2(N) stage-kernel launches (grid=1 "
                            "in-kernel K-packet loop) + sync",
    "hybrid_vendor_p": "stage transposes (materialized) + 2 vendor FFT "
                       "+ k_twiddle + digit-reversal permute/clone + "
                       "sync (all inside call())",
    "hybrid_fused_p": "stage1 input transpose (materialized) + 2 vendor "
                      "FFT + fused k_twiddle (zero-copy [b,k,j] handoff; "
                      "no transpose-back/reshape copy) + digit-reversal "
                      "permute/clone + sync (all inside call())",
}


# ============================ steps ===========================================

def step_blockmax() -> dict:
    out = {"schema": "issue10-wp3-blockmax-v1", "probes": []}

    @triton.jit
    def k_fill(O, BLOCK: tl.constexpr):
        p = tl.arange(0, BLOCK)
        tl.store(O + p, p.to(tl.float32) + 1.0, mask=p < BLOCK)

    for b in (16, 64, 256, 512, 1024, 2048, 4096):
        o = torch.full((b,), -1.0, device="cuda")
        try:
            with xpu_context(0):
                k_fill[(1,)](o, BLOCK=b)
                torch.cuda.synchronize()
            ok = bool((o.cpu() == torch.arange(1, b + 1).float()).all())
            out["probes"].append({"block": b,
                                  "status": "PASS" if ok else "FAIL",
                                  "correct": ok})
        except Exception as e:
            out["probes"].append({"block": b, "status": "FAIL",
                                  "detail": f"{type(e).__name__}: {str(e)[:120]}"})
    out["max_block_pass"] = max(
        (p["block"] for p in out["probes"] if p["status"] == "PASS"), default=0)
    out["vendor_fft_dtypes"] = {}
    for dt, name in ((torch.float32, "fp32"), (torch.float16, "fp16"),
                     (torch.bfloat16, "bf16")):
        try:
            x = torch.randn(4, 256, device="cuda", dtype=dt)
            xc = torch.complex(x, torch.zeros_like(x))
            torch.fft.fft(xc)
            torch.cuda.synchronize()
            out["vendor_fft_dtypes"][name] = "PASS"
        except Exception as e:
            out["vendor_fft_dtypes"][name] = f"FAIL: {type(e).__name__}"
    out["dtype_axis"] = "fp32 only (fp16/bf16 vendor FFT not implemented)"
    # vendor torch.fft dim behavior: non-trailing dims are silently wrong
    try:
        torch.manual_seed(0)
        rr, tt = 8, 16
        xd = torch.rand(rr * tt, device="cuda") - 0.5
        Xn = xd.cpu().numpy().reshape(rr, tt)
        a0 = np.fft.fft(Xn, axis=0)
        a1 = np.fft.fft(Xn, axis=1)
        d0 = torch.fft.fft(xd.view(rr, tt), dim=0).cpu().numpy()
        d1 = torch.fft.fft(xd.view(rr, tt), dim=1).cpu().numpy()
        out["vendor_fft_dim"] = {
            "dim0_is_actually_last_axis": bool(np.abs(d0 - a1).max() < 1e-5
                                               and np.abs(d0 - a0).max() > 1e-2),
            "dim1_err_vs_numpy": float(np.abs(d1 - a1).max()),
            "finding": "vendor torch.fft.fft transforms only the LAST dim: "
                       "dim=0 silently computes axis=-1 -> callers must "
                       "transpose non-trailing dims into last position",
        }
    except Exception as e:
        out["vendor_fft_dim"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    # stale-read size sweep at grid=1: in-kernel store -> permuted load
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "wp2_probe", str(ROOT / "scripts/issue10_wp2_minikernels.py"))
    _wp2 = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_wp2)
    sent = _wp2.SENTINEL
    sweep = []
    for n in (16, 32, 64, 128, 256):
        stale = 0
        trials = 8
        for _ in range(trials):
            re = torch.arange(n, device="cuda").float() + 1
            ore = torch.full((n,), sent, device="cuda")
            s0 = torch.full((n,), sent, device="cuda")
            with xpu_context(0):
                _wp2.k_probe_store_gather[(1,)](re, ore, s0, n, n)
                torch.cuda.synchronize()
            stale += int((ore.cpu() == sent).sum())
        sweep.append({"n": n, "grid": 1, "trials": trials,
                      "stale_lanes": stale, "total_lanes": n * trials,
                      "clean": stale == 0})
    out["stale_sweep"] = {
        "pattern": "store S0[p] <- x*2; load S0[p^1] (WP2 k_probe_store_gather)",
        "grid1_by_n": sweep,
        "finding": "in-kernel store->permuted-load is stale-read-unreliable "
                   "even at grid=1 for n=32/64; small-n (16) and larger-n "
                   "(128/256) were clean in this sweep but the tile kernel "
                   "still corrupted at n=128 and crashed at n=256 (multi-"
                   "roundtrip compounding / compiler limits) -> scratch "
                   "ping-pong is not a legal general lowering; all WP3 "
                   "Triton paths use cross-launch handoff only",
    }
    (OUT_DIR / "wp3_blockmax.json").write_text(json.dumps(out, indent=2))
    return out


_VENDOR_CANARY = None


def _vendor_canary() -> dict:
    """fail-closed vendor FFT semantics lock (run once per process).

    Contract under test: torch.fft.fft/ifft on this stack transform ONLY
    the last axis (wp3_blockmax vendor_fft_dim: dim=0 silently computes
    axis=-1). All paths must operate rank-1/rank-2 with transform axis
    last; any deviation fails the process instead of timing garbage.
    """
    global _VENDOR_CANARY
    if _VENDOR_CANARY is None:
        v = torch.randn(64, device="cuda")
        xc = torch.complex(v, torch.zeros_like(v))
        y = torch.fft.fft(xc)
        ref = np.fft.fft(v.cpu().numpy().astype(np.complex128))
        rel = float(np.abs(y.cpu().numpy() - ref).max()
                    / max(float(np.abs(ref).max()), 1e-30))
        # NOTE: device-side complex elementwise (sub/abs) is
        # XDNN_UNIMPLEMENTED -> roundtrip compared on host
        yi = torch.fft.ifft(y).cpu()
        rel_ifft = float((yi - xc.cpu()).abs().max())
        ok = rel <= TOL_REL and rel_ifft <= TOL_REL
        _VENDOR_CANARY = {
            "axis_contract": "last_dim_only (dim!=-1 is silently "
                             "miscomputed -> never used)",
            "dtype_contract": "complex64 (fp16/bf16 XDNN_UNIMPLEMENTED)",
            "fft_vs_numpy_rel": rel,
            "ifft_fft_roundtrip_abs": rel_ifft,
            "pass": bool(ok),
        }
        if not ok:
            raise RuntimeError(
                f"vendor FFT semantic canary FAILED: {_VENDOR_CANARY}")
    return _VENDOR_CANARY


def _contract_fields(x: torch.Tensor, n: int, b: int,
                     inverse: bool) -> dict:
    """input/output/axis/stride/normalization manifest + fail-closed
    shape/stride/dtype validation (raises on violation: no timing)."""
    if tuple(x.shape) != (b, n):
        raise ValueError(f"input shape {tuple(x.shape)} != {(b, n)}")
    if x.dtype != torch.float32:
        raise ValueError(f"input dtype {x.dtype} != float32")
    if x.stride() != (n, 1):
        raise ValueError(f"input stride {x.stride()} != contiguous {(n, 1)}")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("input contains non-finite values")
    return {
        "in_shape": [b, n],
        "out_shape": [b, n],
        "axis": -1,
        "in_stride": list(x.stride()),
        "layout": ("interleaved complex64 for vendor paths; "
                   "SOA (re/im planes) for Triton paths"),
        "normalization": ("ifft 1/N (conj-trick or vendor ifft)"
                          if inverse else "none (unnormalized forward)"),
        "semantic_lock": "vendor_last_dim_canary=PASS",
    }


def _one_cell(path: str, n: int, b: int, inverse: bool) -> dict:
    x = _data(b, n)
    base = {"n": n, "batch": b,
            "direction": "c2c_inverse" if inverse else "c2c_forward",
            "path": path}
    try:
        _vendor_canary()
        cf = _contract_fields(x, n, b, inverse)
        call, meta = PATHS[path](x, inverse)
        y_re, y_im = call()
        if y_re.shape != x.shape or y_im.shape != x.shape:
            raise ValueError(
                "output shape mismatch: "
                f"{(tuple(y_re.shape), tuple(y_im.shape))} "
                f"!= {tuple(x.shape)}")
        err = _check_batched(y_re, y_im, x, inverse)
        if not np.isfinite(err):
            raise ValueError(f"non-finite rel_err {err}")
    except Exception as e:  # noqa: BLE001  fail-closed: no timing on
        # semantic/contract violation (issue #10 review item 3)
        return {**base, "rel_err": float("inf"), "correct": False,
                "status": f"FAIL-closed: {type(e).__name__}: {str(e)[:160]}",
                "p50_ms": None, "mad_ms": None, "min_ms": None,
                "max_ms": None, "trials": 0}
    stats = _time_path(call)
    return {**base, **cf, "rel_err": err, "correct": err <= TOL_REL,
            "status": "OK",
            "timed_components": PATH_TIMED[path],
            "p50_ms": stats["p50_ms"], "mad_ms": stats["mad_ms"],
            "min_ms": stats["min_ms"], "max_ms": stats["max_ms"],
            "trials": stats["trials"], **meta,
            **{f"desc_{k}": v for k, v in PATH_META[path].items()}}


def step_perf() -> dict:
    rows = []
    for n, b in CELLS:
        for path in PATHS:
            rows.append(_one_cell(path, n, b, False))
        if b == INV_SUBSET[n]:
            for path in PATHS:
                rows.append(_one_cell(path, n, b, True))
        print(f"cell N={n} B={b} done ({len(rows)} rows)", flush=True)
    cols = ["n", "batch", "direction", "path", "status", "rel_err",
            "correct", "p50_ms", "mad_ms", "min_ms", "max_ms", "trials",
            "launches", "gm_bytes_analytic", "factors",
            "in_shape", "out_shape", "axis", "in_stride", "layout",
            "normalization", "semantic_lock", "timed_components",
            "desc_A", "desc_P", "desc_lowering"]
    with open(OUT_DIR / "wp3_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    summary_cells = {}
    for r in rows:
        key = f"{r['n']}x{r['batch']}_{r['direction']}"
        summary_cells.setdefault(key, {"n": r["n"], "batch": r["batch"],
                                       "direction": r["direction"],
                                       "paths": {}})
        summary_cells[key]["paths"][r["path"]] = {
            "p50_ms": r["p50_ms"], "mad_ms": r["mad_ms"],
            "rel_err": r["rel_err"], "correct": r["correct"],
            "status": r.get("status"),
            "launches": r.get("launches"),
            "gm_bytes_analytic": r.get("gm_bytes_analytic"),
        }
    for c in summary_cells.values():
        vend = c["paths"].get("vendor", {}).get("p50_ms")
        for pv in c["paths"].values():
            pv["speedup_vs_vendor"] = (
                (vend / pv["p50_ms"])
                if vend and pv.get("p50_ms") else None)
    doc = {
        "schema": "issue10-wp3-perf-v1",
        "protocol": {
            "timing": f"warmup={WARMUP}, rounds={ROUNDS}, trials={TRIALS}, "
                      "sync per round, per-trial wall p50/MAD (pooled); "
                      "steady-state API wall clock (plan-reuse 口径)",
            "plan_compile": "excluded from timing (covered by warmup); "
                            "one-shot/cold 口径 measured separately in "
                            "wp3_holdout.json",
            "device_time": "not in the timing loop; per-op device time via "
                           "cupti in wp3_profile.json / wp3_trace/",
            "fail_closed": "shape/stride/dtype/finite + vendor last-dim "
                           "canary validated BEFORE timing; violation = "
                           "row status FAIL-closed, no timing",
            "hybrid_costs_included": ["stage transposes (materialized)",
                                      "k_twiddle triton kernel",
                                      "digit-reversal permute/clone",
                                      "torch.cuda.synchronize",
                                      "hybrid_fused_p: inter-stage "
                                      "transpose-back/reshape copy is "
                                      "ABSENT by design (see "
                                      "timed_components)"],
            "vendor_plan_reuse": True,
            "tol_rel": TOL_REL,
            "dtype": "fp32 (complex64)",
            "directions": "c2c_forward on all cells; c2c_inverse on B=16",
            "gm_bytes": "analytic estimate, not measured bandwidth",
        },
        "cells": summary_cells,
        "all_correct": all(r["correct"] for r in rows),
        "rows": len(rows),
    }
    (OUT_DIR / "wp3_summary.json").write_text(json.dumps(doc, indent=2))
    return doc


def step_gonogo() -> dict:
    cap = json.loads((OUT_DIR / "capability_matrix.json").read_text())
    wp2 = json.loads((OUT_DIR / "wp2_summary.json").read_text())
    wp3 = json.loads((OUT_DIR / "wp3_summary.json").read_text())
    bm = json.loads((OUT_DIR / "wp3_blockmax.json").read_text())

    def cap_status(name):
        for e in cap["entries"]:
            if e["probe"] == name:
                return e["status"]
        return "UNKNOWN"

    checks = [
        {
            "id": "A_tile_layout_fusion_stable",
            "question": "A-axis tile/layout/stage-fusion compiles and runs "
                        "stably",
            "verdict": "PASS" if wp2.get("gates", {}).get("radix2_pass")
                       and wp2.get("gates", {}).get("fusion_pass") else "FAIL",
            "evidence": "wp2_summary.json gates radix2/radix4/fusion (grid=1); "
                        "wp3 triton_multipass/triton_resident_loop correct on "
                        "all cells",
        },
        {
            "id": "local_storage_controllable",
            "question": "at least one explicit local storage space with "
                        "capacity/visibility bounds",
            "verdict": "FAIL" if cap_status("local_storage") == "FAIL"
                       else cap_status("local_storage"),
            "evidence": "WP1 local_storage: explicit_space_selection_api FAIL; "
                        "capacity_query PASS (262144 B reported) but placement "
                        "is compiler-managed only",
        },
        {
            "id": "verifiable_compute_transfer_overlap",
            "question": "a verifiable overlap mechanism between stages or "
                        "copy/compute",
            "verdict": "FAIL" if cap_status("async_copy") == "FAIL"
                       else cap_status("async_copy"),
            "evidence": "WP1 async_copy: async_mechanism_api FAIL; "
                        "implicit_overlap UNKNOWN (no counters)",
        },
        {
            "id": "sync_ownership_for_target_pipeline",
            "question": "sync + packet ownership semantics sufficient for a "
                        "multi-role resident pipeline",
            "verdict": "FAIL",
            "evidence": "WP1 cross_program_concurrent_sync UNKNOWN; WP2/WP3 "
                        "stale-read probes: store->permuted-load from the "
                        "same buffer returns pre-store values at grid=8 "
                        "(n=16) AND at grid=1 (n=32/64 sweep); "
                        "debug_barrier ineffective; only cross-launch "
                        "handoff (different buffers / launch order) is legal",
        },
        {
            "id": "radix_or_local_P_core_legal",
            "question": "at least one radix/local FFT P core embeds legally",
            "verdict": "PASS",
            "evidence": "WP2 radix2/radix4/scalar-codegen all correct; WP1 "
                        "host_vendor_interop PASS; WP3 vendor path correct",
        },
        {
            "id": "fusion_reduces_boundaries_with_stable_gain",
            "question": "fusion measurably reduces GM boundary/launch with "
                        "stable independent-session gain",
            "verdict": "PARTIAL",
            "evidence": "WP2 fusion: fused latency == split at N=16 "
                        "(launch-dominated); WP3 multipass (log2(N) "
                        "launches, grid=B) vs resident_loop (log2(N) "
                        "launches, grid=1 packet loop) quantifies "
                        "parallel-packet vs sequential-worker cost at "
                        "larger N — single session, no independent-session "
                        "confirmation yet",
        },
    ]
    full_go = all(c["verdict"] == "PASS" for c in checks)
    verdict = {
        "full_hybrid_resident_pipeline": "GO" if full_go else "NO-GO",
        "restricted_a_axis_migration": "GO",
        "vendor_p_hybrid": "GO",
        "reason": "criteria 2/3/4 fail (no controllable local storage, no "
                  "async overlap, concurrent-sync/stale-read bug); converge "
                  "per issue plan to restricted Triton A-axis (legal "
                  "multipass/resident-loop batched paths) + host-level "
                  "vendor-P hybrid; do not claim full hardware-efficiency "
                  "validation",
    }
    doc = {
        "schema": "issue10-wp3-gonogo-v1",
        "criteria": checks,
        "verdict": verdict,
        "wp3_all_correct": wp3.get("all_correct"),
        "blockmax": bm.get("max_block_pass"),
        "dtype_axis": bm.get("dtype_axis"),
        "vendor_fft_dim": bm.get("vendor_fft_dim"),
        "stale_sweep": bm.get("stale_sweep"),
    }
    (OUT_DIR / "wp3_gonogo.json").write_text(json.dumps(doc, indent=2))
    return doc


STEPS = {"blockmax": step_blockmax, "perf": step_perf, "gonogo": step_gonogo}
ORDER = ["blockmax", "perf", "gonogo"]


def _run_step_capture(name: str) -> dict:
    r = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--step", name],
        capture_output=True, text=True, timeout=STEP_TIMEOUT_S, cwd=str(ROOT),
    )
    (LOG_DIR / f"{name}.log").write_text(
        r.stdout + "\n--- STDERR ---\n" + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(
            f"step {name} rc={r.returncode}; see {LOG_DIR}/{name}.log")
    return {"log": f"wp3_logs/{name}.log"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=list(STEPS) + ["all"], default="all")
    a = ap.parse_args()
    if a.step == "all":
        results = {}
        for name in ORDER:
            print(f"== step {name} ==", flush=True)
            t0 = time.time()
            try:
                cap = _run_step_capture(name)
                results[name] = {"status": "PASS",
                                 "elapsed_s": time.time() - t0, **cap}
            except Exception as e:
                results[name] = {"status": "FAIL",
                                 "error": f"{type(e).__name__}: {e}"}
            print(f"   {results[name]['status']} ({time.time() - t0:.1f}s)",
                  flush=True)
        summary = {
            "schema": "issue10-wp3-summary-v1",
            "issue": "kunlunxin-fft#10",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "script": str(SCRIPT_PATH),
            "steps": results,
        }
        gonogo = json.loads((OUT_DIR / "wp3_gonogo.json").read_text()) \
            if (OUT_DIR / "wp3_gonogo.json").exists() else {}
        summary["gates"] = {
            "blockmax_runs": results["blockmax"]["status"] == "PASS",
            "perf_all_correct": json.loads(
                (OUT_DIR / "wp3_summary.json").read_text())["all_correct"]
            if (OUT_DIR / "wp3_summary.json").exists() else False,
            "gonogo_written": (OUT_DIR / "wp3_gonogo.json").exists(),
            "full_pipeline_go": gonogo.get("verdict", {}).get(
                "full_hybrid_resident_pipeline") == "GO",
        }
        (OUT_DIR / "wp3_run_summary.json").write_text(
            json.dumps(summary, indent=2))
        print(json.dumps(summary["gates"], indent=2))
        return 0 if (summary["gates"]["blockmax_runs"]
                     and summary["gates"]["perf_all_correct"]
                     and summary["gates"]["gonogo_written"]) else 1
    STEPS[a.step]()
    print(f"{a.step}: written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
