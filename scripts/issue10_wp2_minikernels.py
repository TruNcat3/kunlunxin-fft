"""WP2: issue kunlunxin-fft#10 — small-packet mini kernels.

Steps (each isolated in its own subprocess; `--step NAME`):
  opsubset       -> ops_supported.json : Triton op-subset evidence (isolated probes)
  radix2         -> wp2_radix2.json    : single-tile radix-2 DIT, N=8/16
  radix4         -> wp2_radix4.json    : single-tile radix-4/mixed, N=8/16
  fusion         -> wp2_fusion.json    : fused 1-kernel vs split 2-kernel
  concurrency    -> wp2_concurrency.json : multi-program stale-read probe +
                   scalar in-register concurrent packet FFT (validated)
  resident       -> wp2_resident.json  : K packets resident, ownership descriptor
  producer_consumer -> wp2_producer_consumer.json : async pair record-only (WP1 FAIL)
  p_axis         -> wp2_p_axis.json    : fixed A, P in {rad2,rad4,dense,torch}
  all            -> runs the steps above sequentially, writes wp2_summary.json

Concurrency bug (P800, wp2_concurrency.json): with grid>1, a kernel that
stores to a GM buffer then loads the SAME buffer at a *different* (permuted /
gathered) address sporadically reads pre-store values (stale); contiguous
store->load (same address) and permuted loads with no prior store to the
target buffer are unaffected; tl.debug_barrier does not fix it.  At grid=1
the pattern is clean for the n=8/16 tiles exercised here, but WP3's size
sweep (wp3_blockmax.json stale_sweep) later showed stale reads at grid=1
for n=32/64 as well — in-kernel scratch ping-pong is NOT a legal general
lowering at any grid.  This breaks the k_rad2_tile family as a scalable
FFT path.  Workaround (validated): python-codegen'd fully-unrolled scalar
kernel — one thread per packet, all butterflies in registers, twiddles
baked as constants; only the initial (never-stored) permuted input loads +
final stores touch GM (make_scalar_rad2).  For larger N, WP3's
cross-launch multipass/resident-loop stage kernels are the legal path.

Constraints established by step `opsubset` (see descriptor `op_constraints`):
  no slicing / scalar index / register advanced index / tl.trans / tl.sum(axis)
  / tl.join / tl.split / tl.cat / expand_dims / chained None-index / python math
  inside jit bodies.  Available: address-computed load/store (GM gather),
  reshape, where (same shape), single-level None index, //, %, tl.cos/tl.sin,
  full reduce tl.sum(x).  FFT kernels therefore move partner/group values
  through GM with per-stage address arithmetic (one kernel, one launch).
"""
import argparse
import cmath
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/workspace/kunlunxin-fft")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- module-level (required: jit bodies reject `from ... import`) -------------
import torch  # noqa: E402
import torch_xmlir  # noqa: F401,E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from src.xpu.context import xpu_context  # noqa: E402

OUT_DIR = ROOT / "results/p800/issue10_capability"
LOG_DIR = OUT_DIR / "wp2_logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
SCRIPT_PATH = ROOT / "scripts/issue10_wp2_minikernels.py"
SENTINEL = -98765.4321
TOL_REL = 1e-5
STEP_TIMEOUT_S = 600
PROBE_TIMEOUT_S = 60
REF_ITERS = 64
REF_WARMUP = 8
REF_ROUNDS = 5


# ============================ FFT kernels ====================================
# Design (op-constrained): every butterfly stage carries its own register
# vector v and fetches the *partner* value with an address-computed load from
# GM (scratch ping-pong), selecting low/high slots with tl.where.
# Complex SoA: separate re/im fp32 buffers (WP1 complex_sim_f32_soa PASS).

@triton.jit
def k_rad2_tile(RE, IM, ORE, OIM, S0RE, S0IM, S1RE, S1IM,
                N: tl.constexpr, LOGN: tl.constexpr, BLOCK: tl.constexpr,
                K: tl.constexpr):
    bo = tl.program_id(0) * N
    p = tl.arange(0, BLOCK)
    mask = p < N
    zero = p * 0
    v_re = zero.to(tl.float32)
    v_im = zero.to(tl.float32)
    b_re = zero.to(tl.float32)
    b_im = zero.to(tl.float32)
    for s in range(LOGN):                     # python-unrolled (all constexpr)
        m = 1 << s
        L = (p % (2 * m)) < m
        Li = L.to(tl.int32)
        q = p + Li * (2 * m) - m
        if s == 0:
            # DIT: register i holds x[bitrev(i)]; fetch value + partner.
            rp = zero
            rq = zero
            for bi_ in range(LOGN):
                rp = rp + ((p // (1 << bi_)) % 2) * (1 << (LOGN - 1 - bi_))
                rq = rq + ((q // (1 << bi_)) % 2) * (1 << (LOGN - 1 - bi_))
            v_re = tl.load(RE + bo + rp, mask=mask, other=0.0)
            v_im = tl.load(IM + bo + rp, mask=mask, other=0.0)
            b_re = tl.load(RE + bo + rq, mask=mask, other=0.0)
            b_im = tl.load(IM + bo + rq, mask=mask, other=0.0)
        else:
            # v carried in registers from previous stage (same values as the
            # stage store); only the partner needs an addressed GM load.
            src = S0RE if ((s - 1) % 2) == 0 else S1RE
            srci = S0IM if ((s - 1) % 2) == 0 else S1IM
            b_re = tl.load(src + bo + q, mask=mask, other=0.0)
            b_im = tl.load(srci + bo + q, mask=mask, other=0.0)
        e = (p % m) * (N // (2 * m))
        ang = (6.283185307179586 / N) * e.to(tl.float32)
        wr = tl.cos(ang)
        wi = -tl.sin(ang)
        # butterfly: low slot = v + w*b ; high slot = b - w*v
        tb_re = b_re * wr - b_im * wi
        tb_im = b_re * wi + b_im * wr
        tv_re = v_re * wr - v_im * wi
        tv_im = v_re * wi + v_im * wr
        v_re = tl.where(L, v_re + tb_re, b_re - tv_re)
        v_im = tl.where(L, v_im + tb_im, b_im - tv_im)
        if s == LOGN - 1:
            tl.store(ORE + bo + p, v_re, mask=mask)
            tl.store(OIM + bo + p, v_im, mask=mask)
        else:
            dst = S0RE if (s % 2) == 0 else S1RE
            dsti = S0IM if (s % 2) == 0 else S1IM
            tl.store(dst + bo + p, v_re, mask=mask)
            tl.store(dsti + bo + p, v_im, mask=mask)


@triton.jit
def k_rad4_tile(RE, IM, ORE, OIM, S0RE, S0IM,
                N: tl.constexpr, LOGN: tl.constexpr, BLOCK: tl.constexpr):
    bo = tl.program_id(0) * N
    p = tl.arange(0, BLOCK)
    mask = p < N
    zero = p * 0
    # ---- stage 0: four contiguous 4-pt DFTs over rev4-loaded groups (N=16)
    # digit reversal base 4: i = a + 4b -> rev = b + 4a  (a = i % 4, b = i // 4)
    acc_re = zero.to(tl.float32)
    acc_im = zero.to(tl.float32)
    j = p % 4
    for t in range(4):
        idx = (p // 4) * 4 + t
        ridx = (idx // 4) + 4 * (idx % 4)
        ar = tl.load(RE + bo + ridx, mask=mask, other=0.0)
        ai = tl.load(IM + bo + ridx, mask=mask, other=0.0)
        e = (j * t) % 4
        w4r = tl.where(e == 2, -1.0, tl.where(e == 0, 1.0, 0.0))
        w4i = tl.where(e == 1, -1.0, tl.where(e == 3, 1.0, 0.0))
        acc_re = acc_re + (ar * w4r - ai * w4i)
        acc_im = acc_im + (ar * w4i + ai * w4r)
    tl.store(S0RE + bo + p, acc_re, mask=mask)
    tl.store(S0IM + bo + p, acc_im, mask=mask)
    # ---- stage 1: out[k*4+j] = sum_t s1[t*4+j] * W16^(t*j) * W4^(k*t)
    out_re = zero.to(tl.float32)
    out_im = zero.to(tl.float32)
    k = p // 4
    for t in range(4):
        idx = t * 4 + j
        br = tl.load(S0RE + bo + idx, mask=mask, other=0.0)
        bi = tl.load(S0IM + bo + idx, mask=mask, other=0.0)
        e1 = t * j
        ang = (6.283185307179586 / N) * e1.to(tl.float32)
        wr = tl.cos(ang)
        wi = -tl.sin(ang)
        ur = br * wr - bi * wi
        ui = br * wi + bi * wr
        e2 = (k * t) % 4
        w4r = tl.where(e2 == 2, -1.0, tl.where(e2 == 0, 1.0, 0.0))
        w4i = tl.where(e2 == 1, -1.0, tl.where(e2 == 3, 1.0, 0.0))
        out_re = out_re + (ur * w4r - ui * w4i)
        out_im = out_im + (ur * w4i + ui * w4r)
    tl.store(ORE + bo + p, out_re, mask=mask)
    tl.store(OIM + bo + p, out_im, mask=mask)


@triton.jit
def k_rad4mix_tile(RE, IM, ORE, OIM, S0RE, S0IM,
                    N: tl.constexpr, LOGN: tl.constexpr, BLOCK: tl.constexpr):
    bo = tl.program_id(0) * N
    p = tl.arange(0, BLOCK)
    mask = p < N
    zero = p * 0
    # ---- stage 0: two contiguous 4-pt DFTs, perm rev42(i) = (i//4) + 2*(i%4)
    acc_re = zero.to(tl.float32)
    acc_im = zero.to(tl.float32)
    j = p % 4
    for t in range(4):
        idx = (p // 4) * 4 + t
        ridx = (idx // 4) + 2 * (idx % 4)
        ar = tl.load(RE + bo + ridx, mask=mask, other=0.0)
        ai = tl.load(IM + bo + ridx, mask=mask, other=0.0)
        e = (j * t) % 4
        w4r = tl.where(e == 2, -1.0, tl.where(e == 0, 1.0, 0.0))
        w4i = tl.where(e == 1, -1.0, tl.where(e == 3, 1.0, 0.0))
        acc_re = acc_re + (ar * w4r - ai * w4i)
        acc_im = acc_im + (ar * w4i + ai * w4r)
    tl.store(S0RE + bo + p, acc_re, mask=mask)
    tl.store(S0IM + bo + p, acc_im, mask=mask)
    # ---- stage 1: radix-2 combine m=4, pairs (i, i+4), tw = W8^(p%4)
    L = p < 4
    Li = L.to(tl.int32)
    q = p + Li * 8 - 4   # m=4: low -> p+4, high -> p-4
    br = tl.load(S0RE + bo + q, mask=mask, other=0.0)
    bi = tl.load(S0IM + bo + q, mask=mask, other=0.0)
    e = p % 4
    ang = (6.283185307179586 / N) * e.to(tl.float32)
    wr = tl.cos(ang)
    wi = -tl.sin(ang)
    tb_re = br * wr - bi * wi
    tb_im = br * wi + bi * wr
    tv_re = acc_re * wr - acc_im * wi
    tv_im = acc_re * wi + acc_im * wr
    nv_re = tl.where(L, acc_re + tb_re, br - tv_re)
    nv_im = tl.where(L, acc_im + tb_im, bi - tv_im)
    tl.store(ORE + bo + p, nv_re, mask=mask)
    tl.store(OIM + bo + p, nv_im, mask=mask)


@triton.jit
def k_dense_tile(RE, IM, ORE, OIM, N: tl.constexpr, BLOCK: tl.constexpr):
    bo = tl.program_id(0) * N
    p = tl.arange(0, BLOCK)
    mask = p < N
    zero = p * 0
    acc_re = zero.to(tl.float32)
    acc_im = zero.to(tl.float32)
    for i in range(16):                 # BLOCK<=16; i>=N masked by load
        ia = zero + i
        br = tl.load(RE + bo + ia, mask=ia < N, other=0.0)
        bi = tl.load(IM + bo + ia, mask=ia < N, other=0.0)
        e = (p * i) % N
        ang = (6.283185307179586 / N) * e.to(tl.float32)
        wr = tl.cos(ang)
        wi = -tl.sin(ang)
        acc_re = acc_re + (br * wr - bi * wi)
        acc_im = acc_im + (br * wi + bi * wr)
    tl.store(ORE + bo + p, acc_re, mask=mask)
    tl.store(OIM + bo + p, acc_im, mask=mask)


# ------- split kernels for the fusion probe (rad2 stage split at h stages) --
@triton.jit
def k_rad2_half_a(RE, IM, HANDRE, HANDIM, S0RE, S0IM, S1RE, S1IM,
                  N: tl.constexpr, LOGN: tl.constexpr, H: tl.constexpr,
                  BLOCK: tl.constexpr):
    bo = tl.program_id(0) * N
    p = tl.arange(0, BLOCK)
    mask = p < N
    zero = p * 0
    v_re = zero.to(tl.float32)
    v_im = zero.to(tl.float32)
    b_re = zero.to(tl.float32)
    b_im = zero.to(tl.float32)
    for s in range(H):
        m = 1 << s
        L = (p % (2 * m)) < m
        Li = L.to(tl.int32)
        q = p + Li * (2 * m) - m
        if s == 0:
            rp = zero
            rq = zero
            for bi_ in range(LOGN):
                rp = rp + ((p // (1 << bi_)) % 2) * (1 << (LOGN - 1 - bi_))
                rq = rq + ((q // (1 << bi_)) % 2) * (1 << (LOGN - 1 - bi_))
            v_re = tl.load(RE + bo + rp, mask=mask, other=0.0)
            v_im = tl.load(IM + bo + rp, mask=mask, other=0.0)
            b_re = tl.load(RE + bo + rq, mask=mask, other=0.0)
            b_im = tl.load(IM + bo + rq, mask=mask, other=0.0)
        else:
            src = S0RE if ((s - 1) % 2) == 0 else S1RE
            srci = S0IM if ((s - 1) % 2) == 0 else S1IM
            b_re = tl.load(src + bo + q, mask=mask, other=0.0)
            b_im = tl.load(srci + bo + q, mask=mask, other=0.0)
        e = (p % m) * (N // (2 * m))
        ang = (6.283185307179586 / N) * e.to(tl.float32)
        wr = tl.cos(ang)
        wi = -tl.sin(ang)
        tb_re = b_re * wr - b_im * wi
        tb_im = b_re * wi + b_im * wr
        tv_re = v_re * wr - v_im * wi
        tv_im = v_re * wi + v_im * wr
        v_re = tl.where(L, v_re + tb_re, b_re - tv_re)
        v_im = tl.where(L, v_im + tb_im, b_im - tv_im)
        if s == H - 1:
            tl.store(HANDRE + bo + p, v_re, mask=mask)
            tl.store(HANDIM + bo + p, v_im, mask=mask)
        else:
            dst = S0RE if (s % 2) == 0 else S1RE
            dsti = S0IM if (s % 2) == 0 else S1IM
            tl.store(dst + bo + p, v_re, mask=mask)
            tl.store(dsti + bo + p, v_im, mask=mask)


@triton.jit
def k_rad2_half_b(HANDRE, HANDIM, ORE, OIM, S0RE, S0IM, S1RE, S1IM,
                  N: tl.constexpr, LOGN: tl.constexpr, H: tl.constexpr,
                  BLOCK: tl.constexpr):
    bo = tl.program_id(0) * N
    p = tl.arange(0, BLOCK)
    mask = p < N
    # own vector must be re-materialized across the kernel boundary
    v_re = tl.load(HANDRE + bo + p, mask=mask, other=0.0)
    v_im = tl.load(HANDIM + bo + p, mask=mask, other=0.0)
    for s in range(H, LOGN):
        m = 1 << s
        L = (p % (2 * m)) < m
        Li = L.to(tl.int32)
        q = p + Li * (2 * m) - m
        if s == H:
            b_re = tl.load(HANDRE + bo + q, mask=mask, other=0.0)
            b_im = tl.load(HANDIM + bo + q, mask=mask, other=0.0)
        else:
            src = S0RE if ((s - 1) % 2) == 0 else S1RE
            srci = S0IM if ((s - 1) % 2) == 0 else S1IM
            b_re = tl.load(src + bo + q, mask=mask, other=0.0)
            b_im = tl.load(srci + bo + q, mask=mask, other=0.0)
        e = (p % m) * (N // (2 * m))
        ang = (6.283185307179586 / N) * e.to(tl.float32)
        wr = tl.cos(ang)
        wi = -tl.sin(ang)
        tb_re = b_re * wr - b_im * wi
        tb_im = b_re * wi + b_im * wr
        tv_re = v_re * wr - v_im * wi
        tv_im = v_re * wi + v_im * wr
        v_re = tl.where(L, v_re + tb_re, b_re - tv_re)
        v_im = tl.where(L, v_im + tb_im, b_im - tv_im)
        if s == LOGN - 1:
            tl.store(ORE + bo + p, v_re, mask=mask)
            tl.store(OIM + bo + p, v_im, mask=mask)
        else:
            dst = S0RE if (s % 2) == 0 else S1RE
            dsti = S0IM if (s % 2) == 0 else S1IM
            tl.store(dst + bo + p, v_re, mask=mask)
            tl.store(dsti + bo + p, v_im, mask=mask)


# ============================ op-subset probes ===============================
# Each probe kernel runs in its own subprocess (native compiler crashes would
# otherwise take the whole runner down).

@triton.jit
def k_p_slice(A, O, BLOCK: tl.constexpr):
    x = tl.load(A + tl.arange(0, BLOCK), mask=tl.arange(0, BLOCK) < BLOCK, other=0.0)
    s = x.reshape(4, 4)[:, 1:2]
    tl.store(O + tl.arange(0, 4), s.reshape(4), mask=tl.arange(0, 4) < 4)


@triton.jit
def k_p_adv(A, O, BLOCK: tl.constexpr):
    x = tl.load(A + tl.arange(0, BLOCK), mask=tl.arange(0, BLOCK) < BLOCK, other=0.0)
    idx = (tl.arange(0, BLOCK) + 1) % BLOCK
    tl.store(O + tl.arange(0, BLOCK), x[idx], mask=tl.arange(0, BLOCK) < BLOCK)


@triton.jit
def k_p_scalar(A, O, BLOCK: tl.constexpr):
    x = tl.load(A + tl.arange(0, BLOCK), mask=tl.arange(0, BLOCK) < BLOCK, other=0.0)
    tl.store(O + tl.arange(0, 1), x[3], mask=tl.arange(0, 1) < 1)


@triton.jit
def k_p_trans(A, O, BLOCK: tl.constexpr):
    x = tl.load(A + tl.arange(0, BLOCK), mask=tl.arange(0, BLOCK) < BLOCK, other=0.0)
    tl.store(O + tl.arange(0, BLOCK), tl.trans(x.reshape(4, 4)).reshape(BLOCK),
             mask=tl.arange(0, BLOCK) < BLOCK)


@triton.jit
def k_p_sum_axis(A, O, BLOCK: tl.constexpr):
    x = tl.load(A + tl.arange(0, BLOCK), mask=tl.arange(0, BLOCK) < BLOCK, other=0.0)
    tl.store(O + tl.arange(0, 4), tl.sum(x.reshape(4, 4), axis=0),
             mask=tl.arange(0, 4) < 4)


@triton.jit
def k_p_sum_full(A, O, BLOCK: tl.constexpr):
    x = tl.load(A + tl.arange(0, BLOCK), mask=tl.arange(0, BLOCK) < BLOCK, other=0.0)
    s = tl.sum(x) + tl.arange(0, 1).to(tl.float32) * 0.0
    tl.store(O + tl.arange(0, 1), s, mask=tl.arange(0, 1) < 1)


@triton.jit
def k_p_join(A, O, BLOCK: tl.constexpr):
    x = tl.load(A + tl.arange(0, BLOCK), mask=tl.arange(0, BLOCK) < BLOCK, other=0.0)
    j = tl.join(x, x)
    tl.store(O + tl.arange(0, 2 * BLOCK), j, mask=tl.arange(0, 2 * BLOCK) < 2 * BLOCK)


@triton.jit
def k_p_split(A, O, BLOCK: tl.constexpr):
    x = tl.load(A + tl.arange(0, BLOCK), mask=tl.arange(0, BLOCK) < BLOCK, other=0.0)
    a, b = tl.split(x.reshape(8, 2))
    tl.store(O + tl.arange(0, 8), a, mask=tl.arange(0, 8) < 8)
    tl.store(O + tl.arange(8, 16), b, mask=tl.arange(0, 8) < 8)


@triton.jit
def k_p_cat(A, O, BLOCK: tl.constexpr):
    x = tl.load(A + tl.arange(0, BLOCK), mask=tl.arange(0, BLOCK) < BLOCK, other=0.0)
    c = tl.cat(x, x, True)
    tl.store(O + tl.arange(0, 2 * BLOCK), c, mask=tl.arange(0, 2 * BLOCK) < 2 * BLOCK)


@triton.jit
def k_p_cos(A, O, BLOCK: tl.constexpr):
    e = tl.arange(0, 4).to(tl.float32)
    a = 1.5707963267948966 * e
    tl.store(O + tl.arange(0, 4), tl.cos(a) + tl.sin(a), mask=tl.arange(0, 4) < 4)


@triton.jit
def k_p_where(A, O, BLOCK: tl.constexpr):
    x = tl.load(A + tl.arange(0, BLOCK), mask=tl.arange(0, BLOCK) < BLOCK, other=0.0)
    sel = tl.arange(0, BLOCK) < 8
    tl.store(O + tl.arange(0, BLOCK), tl.where(sel, x, x * 2.0),
             mask=tl.arange(0, BLOCK) < BLOCK)


@triton.jit
def k_p_fdiv(A, O, BLOCK: tl.constexpr):
    p = tl.arange(0, BLOCK)
    tl.store(O + tl.arange(0, BLOCK), (p // 4) * 4 + (p % 4),
             mask=tl.arange(0, BLOCK) < BLOCK)


@triton.jit
def k_p_reshape(A, O, BLOCK: tl.constexpr):
    x = tl.load(A + tl.arange(0, BLOCK), mask=tl.arange(0, BLOCK) < BLOCK, other=0.0)
    y = x.reshape(4, 4)
    mr = tl.arange(0, 4)[:, None]
    nc = tl.arange(0, 4)[None, :]
    tl.store(O + mr * 4 + nc, y, mask=(mr < 4) & (nc < 4))


@triton.jit
def k_p_reshape_back(A, O, BLOCK: tl.constexpr):
    x = tl.load(A + tl.arange(0, BLOCK), mask=tl.arange(0, BLOCK) < BLOCK, other=0.0)
    y = x.reshape(4, 4).reshape(16)
    tl.store(O + tl.arange(0, 16), y, mask=tl.arange(0, 16) < 16)


@triton.jit
def k_p_expanddims(A, O, BLOCK: tl.constexpr):
    x = tl.load(A + tl.arange(0, BLOCK), mask=tl.arange(0, BLOCK) < BLOCK, other=0.0)
    tl.store(O + tl.arange(0, BLOCK), tl.expand_dims(x, 0).reshape(BLOCK),
             mask=tl.arange(0, BLOCK) < BLOCK)


@triton.jit
def k_p_nones2(A, O, BLOCK: tl.constexpr):
    r = tl.arange(0, 4)[:, None, None]
    tl.store(O + tl.arange(0, 4), (r * 0.0 + 1.0).reshape(4),
             mask=tl.arange(0, 4) < 4)


@triton.jit
def k_p_trace_math(A, O, BLOCK: tl.constexpr):
    x = tl.load(A + tl.arange(0, BLOCK), mask=tl.arange(0, BLOCK) < BLOCK, other=0.0)
    w = math.cos(0.0)   # python math inside jit body (probe)
    tl.store(O + tl.arange(0, BLOCK), x * w, mask=tl.arange(0, BLOCK) < BLOCK)


@triton.jit
def k_p_addr_gather(A, O, BLOCK: tl.constexpr):
    idx = (tl.arange(0, BLOCK) + 3) % BLOCK
    x = tl.load(A + idx, mask=tl.arange(0, BLOCK) < BLOCK, other=0.0)
    tl.store(O + tl.arange(0, BLOCK), x, mask=tl.arange(0, BLOCK) < BLOCK)


PROBE_FN = {
    "slice_get": k_p_slice,
    "reg_adv_index": k_p_adv,
    "reg_scalar_index": k_p_scalar,
    "tl_trans": k_p_trans,
    "tl_sum_axis": k_p_sum_axis,
    "tl_sum_full": k_p_sum_full,
    "tl_join": k_p_join,
    "tl_split": k_p_split,
    "tl_cat": k_p_cat,
    "tl_cos_sin": k_p_cos,
    "where_same_shape": k_p_where,
    "floordiv_mod": k_p_fdiv,
    "reshape": k_p_reshape,
    "reshape_back": k_p_reshape_back,
    "expand_dims": k_p_expanddims,
    "none_chained_index": k_p_nones2,
    "python_math_in_jit": k_p_trace_math,
    "load_addr_gather": k_p_addr_gather,
}

PROBE_EXPECT = {
    "tl_cos_sin": [1.0, 1.0, -1.0, -1.0],
    "floordiv_mod": [0.0, 1.0, 2.0, 3.0],
    "reshape": [1.0, 2.0, 3.0, 4.0],
    "reshape_back": [1.0, 2.0, 3.0, 4.0],
    "load_addr_gather": [4.0, 5.0, 6.0, 7.0],
    "tl_sum_full": [136.0],
}


def _probe_one(name: str) -> int:
    fn = PROBE_FN[name]
    v = torch.arange(16, dtype=torch.float32, device="cuda") + 1.0
    o = torch.full((64,), SENTINEL, device="cuda")
    try:
        with xpu_context(0):
            fn[(1,)](v, o, BLOCK=16)
            torch.cuda.synchronize()
    except Exception as e:  # CompilationError / unsupported value errors
        msg = str(e).replace("\n", " ").strip()
        print(f"RESULT\t{name}\tFAIL\t{type(e).__name__}: {msg[:180]}")
        return 0
    exp = PROBE_EXPECT.get(name)
    if exp is not None:
        got = o[: len(exp)].tolist()
        ok = all(abs(g - e) < 1e-4 for g, e in zip(got, exp))
        if not ok:
            print(f"RESULT\t{name}\tFAIL\tvalue mismatch got={got[:4]}")
            return 0
    print(f"RESULT\t{name}\tPASS\t")
    return 0


def step_opsubset() -> dict:
    entries = []
    for name in PROBE_FN:
        try:
            r = subprocess.run(
                [sys.executable, str(SCRIPT_PATH), "--probe-one", name],
                capture_output=True, text=True, timeout=PROBE_TIMEOUT_S,
                cwd=str(ROOT), env=None,
            )
            line = ""
            for ln in r.stdout.splitlines():
                if ln.startswith("RESULT\t"):
                    line = ln
            if line:
                _, nm, st, detail = (line.split("\t") + [""])[:4]
                entries.append({"probe": nm, "status": st, "detail": detail})
            else:
                detail = (r.stderr or "")[-200:].replace("\n", " ")
                entries.append({"probe": name, "status": "FAIL",
                                "detail": f"native compiler crash (no result line); rc={r.returncode}; {detail}"})
        except subprocess.TimeoutExpired:
            entries.append({"probe": name, "status": "UNKNOWN",
                            "detail": f"timeout {PROBE_TIMEOUT_S}s"})
    supported = [e["probe"] for e in entries if e["status"] == "PASS"]
    unsupported = [e["probe"] for e in entries if e["status"] == "FAIL"]
    doc = {
        "schema": "issue10-wp2-opsubset-v1",
        "probe_script": str(SCRIPT_PATH),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "policy": "status from executed probes only; isolated subprocess per probe "
                  "(native crashes cannot poison the runner)",
        "entries": entries,
        "supported": supported,
        "unsupported": unsupported,
        "totals": {
            "PASS": sum(1 for e in entries if e["status"] == "PASS"),
            "FAIL": sum(1 for e in entries if e["status"] == "FAIL"),
            "UNKNOWN": sum(1 for e in entries if e["status"] == "UNKNOWN"),
        },
    }
    (OUT_DIR / "ops_supported.json").write_text(json.dumps(doc, indent=2))
    return doc


# ============================ helpers ========================================

def _ref_fft(x: torch.Tensor) -> torch.Tensor:
    # NOTE: this torch build's .double() on a *complex* tensor silently
    # discards the imaginary part (Copy.cpp cast); never cast complex.
    return torch.fft.fft(x.cpu())


def _rel_err(y: torch.Tensor, ref: torch.Tensor) -> float:
    r = ref.abs().max().item()
    return float((y.cpu() - ref).abs().max().item() / max(r, 1e-30))


def _launch(fn, grid, args, **kw):
    with xpu_context(0):
        fn[grid](*args, **kw)
        torch.cuda.synchronize()


def _latency(fn, grid, args, iters=REF_ITERS, warmup=REF_WARMUP,
             rounds=REF_ROUNDS, **kw) -> float:
    with xpu_context(0):
        for _ in range(warmup):
            fn[grid](*args, **kw)
        torch.cuda.synchronize()
        best = []
        for _ in range(rounds):
            t0 = time.perf_counter()
            for _ in range(iters):
                fn[grid](*args, **kw)
            torch.cuda.synchronize()
            best.append((time.perf_counter() - t0) * 1e3 / iters)
    return float(min(best))


def _buffers(n: int, k: int = 1):
    shp = (k * n,)
    return [torch.full(shp, SENTINEL, device="cuda", dtype=torch.float32)
            for _ in range(10)]


def _descriptor(probe: str, launch: int, gm: dict, ownership: str,
                live: str, boundaries: list, sync: str, extra: dict | None = None) -> dict:
    d = {
        "probe": probe,
        "ownership": ownership,
        "local_live_data_range": live,
        "required_global_boundaries": boundaries,
        "synchronization_scope": sync,
        "expected_launch_count": launch,
        "expected_gm_bytes": gm,
    }
    if extra:
        d.update(extra)
    return d


def gm_rad2(n: int, logn: int) -> int:
    """Per-packet GM bytes for the fused radix-2 tile.

    stage0: v + partner loads (2 vectors x 2 comps) ; stages 1..L-1: partner
    load (1 vector x 2 comps) each ; every non-final stage stores its result
    (2 comps) ; final stage stores output (2 comps).
    """
    v = 2 * n * 4                      # one re/im vector in bytes
    return 4 * v + (logn - 1) * v + (logn - 1) * v + v


OP_CONSTRAINT = (
    "op-subset (ops_supported.json): FFT stages cannot slice/index-register "
    "values; partner/group values are fetched via address-computed GM loads "
    "inside one kernel (in-kernel scratch ping-pong). Constraints: "
    "supported = load/store(addr-arithmetic gather), where(same shape), "
    "//, %, cos/sin, full sum; unsupported = slice, reg/scalar index, trans, "
    "sum(axis), join/split/cat, chained None, python math in jit; reshape "
    "passes the frontend but crashes the XPU lowering (GlobalEncodingAttr "
    "UNREACHABLE) -> no kernel here uses reshape.  NOTE (stale-read bug): "
    "the GM ping-pong pattern is unreliable: store->permuted-load from the "
    "same buffer returns stale values at grid>1 (this WP) and — per WP3's "
    "size sweep — also at grid=1 for n=32/64; see wp2_concurrency.json and "
    "wp3_blockmax.json.  The scalar codegen kernel sidesteps the whole "
    "op-subset: unrolling happens in python, the jit body is pure scalar "
    "load/compute/store."
)


# =================== scalar codegen kernel (workaround) =======================
# Fully-unrolled radix-2 DIT generated in python: one thread per packet,
# butterflies stay in scalar registers, twiddle constants are baked in.
# GM traffic = bit-reversed input loads (buffer never stored by this kernel)
# + final stores only -> no store->gather roundtrip -> concurrency-safe.

CODEGEN_DIR = OUT_DIR / "wp2_codegen"
CODEGEN_DIR.mkdir(parents=True, exist_ok=True)


def make_scalar_rad2(n: int, logn: int):
    """Generate + import a fully-unrolled scalar radix-2 DIT kernel.

    Signature: k_rad2_scalar[(grid,)](RE, IM, ORE, OIM, K: tl.constexpr)
    Program pid computes the size-n FFT of packet slice [pid*n, (pid+1)*n).
    """
    br = [0] * n
    for i in range(n):
        v = 0
        for b in range(logn):
            v += ((i // (1 << b)) % 2) << (logn - 1 - b)
        br[i] = v
    lines = [
        "import triton, triton.language as tl",
        "@triton.jit",
        "def k_rad2_scalar(RE, IM, ORE, OIM, K: tl.constexpr):",
        "    bo = tl.program_id(0) * %d" % n,
    ]
    for i in range(n):
        lines.append("    vr%d = tl.load(RE + bo + %d)" % (i, br[i]))
        lines.append("    vi%d = tl.load(IM + bo + %d)" % (i, br[i]))
    for s in range(logn):
        m = 1 << s
        for base in range(0, n, 2 * m):
            for r in range(m):
                e = r * (n // (2 * m))
                w = cmath.exp(-2j * math.pi * e / n)
                a = base + r
                b = base + r + m
                lines.append("    tr%d = vr%d * %.17f - vi%d * %.17f"
                             % (b, b, w.real, b, w.imag))
                lines.append("    ti%d = vr%d * %.17f + vi%d * %.17f"
                             % (b, b, w.imag, b, w.real))
                lines.append("    nr%d = vr%d + tr%d" % (a, a, b))
                lines.append("    ni%d = vi%d + ti%d" % (a, a, b))
                lines.append("    vr%d = vr%d - tr%d" % (b, a, b))
                lines.append("    vi%d = vi%d - ti%d" % (b, a, b))
                lines.append("    vr%d = nr%d" % (a, a))
                lines.append("    vi%d = ni%d" % (a, a))
    for i in range(n):
        lines.append("    tl.store(ORE + bo + %d, vr%d)" % (i, i))
        lines.append("    tl.store(OIM + bo + %d, vi%d)" % (i, i))
    path = CODEGEN_DIR / ("k_rad2_scalar_n%d.py" % n)
    path.write_text("\n".join(lines) + "\n")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "wp2_k_rad2_scalar_n%d" % n, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.k_rad2_scalar


# ---- stale-read probe: store S0[p] then gather S0[p^1] -----------------------
# At grid=1 the gather sees the just-stored values; at grid>1 some lanes
# sporadically return the pre-store sentinel (hardware/compiler bug).

@triton.jit
def k_probe_store_gather(RE, ORE, S0RE, N: tl.constexpr, BLOCK: tl.constexpr):
    bo = tl.program_id(0) * N
    p = tl.arange(0, BLOCK)
    mask = p < N
    x = tl.load(RE + bo + p, mask=mask, other=0.0)
    tl.store(S0RE + bo + p, x * 2.0, mask=mask)
    Li = ((p % 2) < 1).to(tl.int32)
    q = p + Li * 2 - 1
    w = tl.load(S0RE + bo + q, mask=mask, other=0.0)
    tl.store(ORE + bo + p, w, mask=mask)


def _stale_sentinel_count(grid: int, n: int, trials: int = 6) -> dict:
    """Launch k_probe_store_gather `trials` times; count stale/incorrect lanes."""
    stale = 0
    wrong = 0
    total = 0
    for _ in range(trials):
        re = torch.arange(grid * n, device="cuda", dtype=torch.float32) + 1.0
        ore = torch.full((grid * n,), SENTINEL, device="cuda")
        s0 = torch.full((grid * n,), SENTINEL, device="cuda")
        _launch(k_probe_store_gather, (grid,), (re, ore, s0, n, n))
        got = ore.cpu()
        stale += int((got == SENTINEL).sum().item())
        y = re.cpu()
        exp = torch.empty_like(y)
        exp[0::2] = y[1::2] * 2.0
        exp[1::2] = y[0::2] * 2.0
        wrong += int((got != exp).sum().item())
        total += grid * n
    return {"grid": grid, "trials": trials, "total_lanes": total,
            "stale_sentinel_lanes": stale, "wrong_lanes": wrong,
            "clean": stale == 0}


# ============================ steps ==========================================

def step_radix2() -> dict:
    out = {"schema": "issue10-wp2-radix2-v1", "op_constraints": OP_CONSTRAINT,
           "cases": []}
    for n in (8, 16):
        logn = n.bit_length() - 1
        re, im, ore, oim, s0r, s0i, s1r, s1i, _pad, _pad2 = _buffers(n)
        re.copy_(torch.rand(n, device="cuda"))
        im.copy_(torch.rand(n, device="cuda"))
        x = torch.complex(re.cpu().double(), im.cpu().double())
        ref = _ref_fft(x)
        args = (re, im, ore, oim, s0r, s0i, s1r, s1i, n, logn, n)
        _launch(k_rad2_tile, (1,), args, K=1)
        y = torch.complex(ore.cpu().double(), oim.cpu().double())
        err = _rel_err(y, ref)
        lat = _latency(k_rad2_tile, (1,), args, K=1)
        # gm bytes: stage0: v+b loads 2*2N; stages1..L-1: partner 2N each;
        # stores: L-1 scratch rounds 2N + final 2N
        gm_per = gm_rad2(n, logn)
        desc = _descriptor(
            "radix2_single_tile", 1,
            {"per_packet_bytes": gm_per,
             "formula": "4*N*4 (stage0 v+b loads) + (L-1)*2*N*4 (partner loads) "
                        "+ (L-1)*2*N*4 (scratch stores) + 2*N*4 (final stores), L=log2(N)"},
            ownership="grid=1 program owns the whole packet: input load, all "
                      "log2(N) DIT stages, output store; registers v_re/v_im are "
                      "the packet-private live set",
            live=f"register v (2x{2 * n}B fp32 SoA) across {logn} stages; "
                 f"GM scratch ping-pong S0/S1 (2x2x{2 * n}B)",
            boundaries=["input RE/IM buffers (rev-addressed load)",
                        "scratch S0/S1 (per-stage partner load + stage store)",
                        "output ORE/OIM (final store)"],
            sync="single program; in-program program-order RAW only; no "
                 "barrier, no cross-program sync (grid=1)",
            extra={"latency_ms": lat, "rel_err": err,
                   "pass": err <= TOL_REL,
                   "n": n, "logn": logn})
        out["cases"].append(desc)
    out["all_pass"] = all(c["pass"] for c in out["cases"])
    (OUT_DIR / "wp2_radix2.json").write_text(json.dumps(out, indent=2))
    return out


def step_radix4() -> dict:
    out = {"schema": "issue10-wp2-radix4-v1", "op_constraints": OP_CONSTRAINT,
           "cases": []}
    cases = [
        ("radix4_pure_n16", 16, k_rad4_tile,
         {"stage0": "4 contiguous 4-pt DFTs over rev4-loaded groups",
          "stage1": "twiddle W16^(t*j) + 4-pt combine W4^(k*t)",
          "digit_reversal": "i=a+4b -> b+4a (in-kernel //,% addressing)"}),
        ("radix4_mixed_n8", 8, k_rad4mix_tile,
         {"stage0": "2 contiguous 4-pt DFTs over rev42 perm "
                    "(i -> (i//4)+2*(i%4))",
          "stage1": "radix-2 combine m=4, twiddle W8^(p%4)",
          "note": "N=8 = 4x2 mixed radix (single radix-4 stage + radix-2 tail)"}),
    ]
    for name, n, fn, alg in cases:
        logn = n.bit_length() - 1
        re, im, ore, oim, s0r, s0i, _a, _b, _c, _d = _buffers(n)
        re.copy_(torch.rand(n, device="cuda"))
        im.copy_(torch.rand(n, device="cuda"))
        x = torch.complex(re.cpu().double(), im.cpu().double())
        ref = _ref_fft(x)
        args = (re, im, ore, oim, s0r, s0i, n, logn, n)
        _launch(fn, (1,), args)
        y = torch.complex(ore.cpu().double(), oim.cpu().double())
        err = _rel_err(y, ref)
        lat = _latency(fn, (1,), args)
        if n == 16:
            gm_per = 8 * n * 4 + 2 * n * 4 + 8 * n * 4 + 2 * n * 4  # 4 loads x2 stages + 2 stores
            formula = "(4 terms x2 comp + 4 terms x2 comp)*N*4 + 2 stores*2*N*4"
        else:
            gm_per = 8 * n * 4 + 2 * n * 4 + 2 * n * 4 + 2 * n * 4
            formula = "(4 terms x2 comp)*N*4 + S0 store + partner load + final store"
        desc = _descriptor(
            name, 1,
            {"per_packet_bytes": gm_per, "formula": formula},
            ownership="grid=1 program owns the packet; stage results live in "
                      "registers then materialize to one scratch buffer "
                      "(stage0) before the addressed second stage load",
            live=f"register accumulator 2x{2 * n}B; GM scratch S0 2x{2 * n}B",
            boundaries=["input RE/IM (digit-reversed addressed loads)",
                        "scratch S0 (stage0 materialization)",
                        "output ORE/OIM"],
            sync="single program, program-order memory ops; no barrier",
            extra={"latency_ms": lat, "rel_err": err, "pass": err <= TOL_REL,
                   "n": n, "algorithm": alg})
        out["cases"].append(desc)
    out["all_pass"] = all(c["pass"] for c in out["cases"])
    (OUT_DIR / "wp2_radix4.json").write_text(json.dumps(out, indent=2))
    return out


def step_fusion() -> dict:
    out = {"schema": "issue10-wp2-fusion-v1", "op_constraints": OP_CONSTRAINT,
           "n": 16}
    n, logn = 16, 4
    h = logn // 2  # split point
    re, im, ore, oim, s0r, s0i, s1r, s1i, hr, hi = _buffers(n)
    re.copy_(torch.rand(n, device="cuda"))
    im.copy_(torch.rand(n, device="cuda"))
    x = torch.complex(re.cpu().double(), im.cpu().double())
    ref = _ref_fft(x)
    fused_args = (re, im, ore, oim, s0r, s0i, s1r, s1i, n, logn, n)
    _launch(k_rad2_tile, (1,), fused_args, K=1)
    err_f = _rel_err(torch.complex(ore.cpu().double(), oim.cpu().double()), ref)
    lat_f = _latency(k_rad2_tile, (1,), fused_args, K=1)
    ore.fill_(SENTINEL); oim.fill_(SENTINEL)
    a_args = (re, im, hr, hi, s0r, s0i, s1r, s1i, n, logn, h, n)
    b_args = (hr, hi, ore, oim, s0r, s0i, s1r, s1i, n, logn, h, n)
    with xpu_context(0):
        k_rad2_half_a[(1,)](*a_args)
        torch.cuda.synchronize()
        k_rad2_half_b[(1,)](*b_args)
        torch.cuda.synchronize()
    err_s = _rel_err(torch.complex(ore.cpu().double(), oim.cpu().double()), ref)
    with xpu_context(0):
        for _ in range(REF_WARMUP):
            k_rad2_half_a[(1,)](*a_args); k_rad2_half_b[(1,)](*b_args)
        torch.cuda.synchronize()
        best = []
        for _ in range(REF_ROUNDS):
            t0 = time.perf_counter()
            for _ in range(REF_ITERS):
                k_rad2_half_a[(1,)](*a_args); k_rad2_half_b[(1,)](*b_args)
            torch.cuda.synchronize()
            best.append((time.perf_counter() - t0) * 1e3 / REF_ITERS)
    lat_s = float(min(best))
    gm_fused = gm_rad2(n, logn)
    # split pays one extra vector: kernel B must re-load its own vector from
    # the handoff buffer (register state cannot cross kernel boundaries).
    gm_split = gm_fused + 2 * n * 4
    out.update({
        "fused": _descriptor(
            "fusion_fused", 1,
            {"per_packet_bytes": gm_fused,
             "formula": "gm_rad2(N,L): stage0 v+partner loads, per-stage "
                        "partner load + result store, final output store"},
            ownership="one program owns all 4 stages of the packet",
            live="register v across 4 stages + scratch ping-pong",
            boundaries=["input", "scratch S0/S1 (per stage)", "output"],
            sync="program-order only; no cross-launch visibility needed",
            extra={"latency_ms": lat_f, "rel_err": err_f, "pass": err_f <= TOL_REL}),
        "split_2kernel": _descriptor(
            "fusion_split", 2,
            {"per_packet_bytes": gm_split,
             "formula": "gm_rad2(N,L) + 2*N*4 (kernel B re-loads its own "
                        "vector from the handoff: register state cannot cross "
                        "kernel boundaries)"},
            ownership="kernel A owns stages 0..h-1 + handoff buffer; kernel B "
                      "owns stages h..3 from handoff to output",
            live="register v per kernel; GM handoff RE/IM (2x2x16B)",
            boundaries=["input", "handoff buffer (cross-launch GM boundary)",
                        "scratch", "output"],
            sync="launch-order (host stream) establishes handoff visibility "
                 "via gm_launch_order_visibility=PASS (WP1)",
            extra={"latency_ms": lat_s, "rel_err": err_s, "pass": err_s <= TOL_REL}),
        "delta": {
            "latency_ms_fused_minus_split": lat_f - lat_s,
            "launches_fused": 1, "launches_split": 2,
            "gm_bytes_delta": 2 * n * 4,
            "interpretation": "fusion saves one host launch plus one full "
                              "vector reload (split kernel B re-materializes "
                              "its register state from the handoff buffer); "
                              "N=16 is launch-dominated so latency ordering is "
                              "reported as measured, not interpreted"},
        "both_pass": (err_f <= TOL_REL) and (err_s <= TOL_REL),
    })
    (OUT_DIR / "wp2_fusion.json").write_text(json.dumps(out, indent=2))
    return out


def step_resident() -> dict:
    out = {"schema": "issue10-wp2-resident-v2", "op_constraints": OP_CONSTRAINT,
           "kernel": "scalar_codegen"}
    k, n, logn = 8, 16, 4
    kf = make_scalar_rad2(n, logn)
    re, im, ore, oim, *_ = _buffers(n, k)
    re.copy_(torch.rand(k * n, device="cuda"))
    im.copy_(torch.rand(k * n, device="cuda"))
    args = (re, im, ore, oim, k)
    _launch(kf, (k,), args)
    xr = re.reshape(k, n).cpu().double(); xi = im.reshape(k, n).cpu().double()
    yr = ore.reshape(k, n).cpu().double(); yi = oim.reshape(k, n).cpu().double()
    errs = []
    for i in range(k):
        ref = _ref_fft(torch.complex(xr[i], xi[i]))
        errs.append(float((torch.complex(yr[i], yi[i]) - ref).abs().max().item()
                          / max(ref.abs().max().item(), 1e-30)))
    lat = _latency(kf, (k,), args)
    # comparison: GM ping-pong tile kernel at the same grid (known stale-read
    # bug at grid>1; recorded as evidence, not gated)
    re2, im2, ore2, oim2, s0r, s0i, s1r, s1i, _a, _b = _buffers(n, k)
    re2.copy_(re); im2.copy_(im)
    tile_args = (re2, im2, ore2, oim2, s0r, s0i, s1r, s1i, n, logn, n)
    tile_errs = []
    try:
        _launch(k_rad2_tile, (k,), tile_args, K=1)
        yr2 = ore2.reshape(k, n).cpu().double(); yi2 = oim2.reshape(k, n).cpu().double()
        for i in range(k):
            ref = _ref_fft(torch.complex(xr[i], xi[i]))
            tile_errs.append(float(
                (torch.complex(yr2[i], yi2[i]) - ref).abs().max().item()
                / max(ref.abs().max().item(), 1e-30)))
        tile_status = "PASS" if max(tile_errs) <= TOL_REL else "FAIL"
    except Exception as e:  # native crash also counts as tile failure evidence
        tile_errs = []
        tile_status = "CRASH: %s: %s" % (type(e).__name__, str(e)[:160])
    # per-packet GM bytes for the scalar kernel: bitrev input loads + stores
    gm_per = 4 * n * 4
    out.update({
        "packets": k,
        "per_packet_max_rel_err": max(errs),
        "all_pass": max(errs) <= TOL_REL,
        "latency_batch_ms": lat,
        "tile_kernel_comparison": {
            "kernel": "k_rad2_tile (GM scratch ping-pong)",
            "grid": k,
            "status": tile_status,
            "per_packet_max_rel_err": max(tile_errs) if tile_errs else None,
            "interpretation": "expected FAIL: store->permuted-load stale-read "
                              "bug at grid>1 (wp2_concurrency.json)",
        },
        "descriptor": _descriptor(
            "resident_packets", 1,
            {"per_packet_bytes": gm_per, "total_bytes": gm_per * k,
             "formula": "grid=k programs x per-packet traffic: bitrev "
                        "input loads (re+im = 2*N*4B) + final output stores "
                        "(re+im = 2*N*4B); no scratch"},
            ownership=f"program pid owns packet slice [pid*N, (pid+1)*N) of "
                      f"RE/IM/ORE/OIM; one thread per packet holds all N "
                      f"complex values in scalar registers across all {logn} "
                      f"unrolled DIT stages (python codegen)",
            live=f"per-thread scalar registers 2x{n} fp32 (+ twiddle "
                 f"constants baked into the binary); no GM scratch",
            boundaries=["input tensor slices (disjoint, rev-addressed loads; "
                        "buffer never stored by this kernel)",
                        "output tensor slices (disjoint, final stores)"],
            sync="no cross-program sync required: packets are disjoint; only "
                 "in-program program-order RAW order matters (register "
                 "dependencies, no GM roundtrips)",
            extra={"grid": k, "packets_per_program": 1,
                   "codegen": str(CODEGEN_DIR / ("k_rad2_scalar_n%d.py" % n))}),
    })
    (OUT_DIR / "wp2_resident.json").write_text(json.dumps(out, indent=2))
    return out


def step_concurrency() -> dict:
    out = {"schema": "issue10-wp2-concurrency-v1",
           "op_constraints": OP_CONSTRAINT}
    n, logn = 16, 4
    probe_g1 = _stale_sentinel_count(grid=1, n=n, trials=6)
    probe_g8 = _stale_sentinel_count(grid=8, n=n, trials=6)
    out["stale_read_probe"] = {
        "pattern": "store S0[p] <- x*2 ; load S0[p^1] ; store ORE",
        "sentinel": SENTINEL,
        "grid1": probe_g1,
        "grid8": probe_g8,
        "bug_confirmed": (probe_g1["clean"]
                          and probe_g8["stale_sentinel_lanes"] > 0),
        "classification": (
            "HARDWARE/COMPILER BUG (store->permuted-load stale read at "
            "grid>1): sporadic pre-store values returned by gather loads "
            "after an earlier store to the same buffer; contiguous "
            "store->load and gather-without-store unaffected; "
            "tl.debug_barrier does not fix; grid=1 clean"),
    }
    # scalar concurrent FFT (validated workaround)
    cases = []
    for k, nn in ((8, 16), (8, 8), (16, 16)):
        lg = nn.bit_length() - 1
        kf = make_scalar_rad2(nn, lg)
        re, im, ore, oim, *_ = _buffers(nn, k)
        re.copy_(torch.rand(k * nn, device="cuda"))
        im.copy_(torch.rand(k * nn, device="cuda"))
        args = (re, im, ore, oim, k)
        _launch(kf, (k,), args)
        xr = re.reshape(k, nn).cpu().double()
        xi = im.reshape(k, nn).cpu().double()
        yr = ore.reshape(k, nn).cpu().double()
        yi = oim.reshape(k, nn).cpu().double()
        errs = []
        for i in range(k):
            ref = _ref_fft(torch.complex(xr[i], xi[i]))
            errs.append(float(
                (torch.complex(yr[i], yi[i]) - ref).abs().max().item()
                / max(ref.abs().max().item(), 1e-30)))
        cases.append({"grid": k, "n": nn,
                      "per_packet_max_rel_err": max(errs),
                      "pass": max(errs) <= TOL_REL})
    out["scalar_concurrent_fft"] = {
        "kernel": "make_scalar_rad2 (fully-unrolled in-register DIT)",
        "cases": cases,
        "all_pass": all(c["pass"] for c in cases),
        "why_it_works": "no GM store->load roundtrip: input loads hit a "
                        "buffer this kernel never stores; butterflies are "
                        "scalar register ops; output is a final store",
    }
    out["all_pass"] = (out["scalar_concurrent_fft"]["all_pass"]
                       and probe_g1["clean"])
    (OUT_DIR / "wp2_concurrency.json").write_text(json.dumps(out, indent=2))
    return out


def step_producer_consumer() -> dict:
    matrix = json.loads((OUT_DIR / "capability_matrix.json").read_text())
    async_entry = next(e for e in matrix["entries"] if e["probe"] == "async_copy")
    status = "FAIL"
    for c in async_entry["checks"]:
        if c["id"] == "async_mechanism_api":
            async_status = c["status"]
            async_detail = c["detail"]
    out = {
        "schema": "issue10-wp2-producer-consumer-v1",
        "probe": "producer_consumer_async_pair",
        "status": status,
        "attempted": False,
        "reason": "record-only: WP1 async_copy async_mechanism_api = "
                  f"{async_status} ({async_detail}); without an async copy "
                  "primitive, a producer/consumer packet pair would degrade to "
                  "ordinary GM round-trips already covered by fusion/resident "
                  "steps — no new capability would be demonstrated",
        "source": "results/p800/issue10_capability/capability_matrix.json "
                  "# async_copy.async_mechanism_api",
        "next_step_if_enabled": "re-probe async_mechanism_api on a vendor "
                                "release exposing the async copy API, then run "
                                "a two-kernel producer/consumer pair with "
                                "explicit event/scope sync",
        "descriptor": _descriptor(
            "producer_consumer_async_pair", 0,
            {"bytes": 0, "formula": "not executed"},
            ownership="n/a (not attempted)",
            live="n/a",
            boundaries="n/a (would be a cross-program GM handoff)",
            sync="n/a (cross-program concurrent sync UNKNOWN in WP1)",
        ),
    }
    (OUT_DIR / "wp2_producer_consumer.json").write_text(json.dumps(out, indent=2))
    return out


def step_p_axis() -> dict:
    out = {"schema": "issue10-wp2-p-axis-v1", "op_constraints": OP_CONSTRAINT,
           "fixed_A": "single-tile natural->natural FFT, N=16 (one launch, "
                      "register-resident stage sequence per op-subset)",
           "variants": []}
    n, logn = 16, 4
    ref_src = None
    for pname in ("rad2", "rad4", "dense", "torch"):
        re, im, ore, oim, s0r, s0i, s1r, s1i, _a, _b = _buffers(n)
        re.copy_(torch.rand(n, device="cuda"))
        im.copy_(torch.rand(n, device="cuda"))
        x = torch.complex(re.cpu().double(), im.cpu().double())
        ref = _ref_fft(x)
        if ref_src is None:
            ref_src = ref
        if pname == "rad2":
            fn, args, launches = k_rad2_tile, (re, im, ore, oim, s0r, s0i, s1r,
                                               s1i, n, logn, n), 1
            lat = _latency(fn, (1,), args, K=1)
            note = "stage sequence via addressed partner loads"
        elif pname == "rad4":
            fn, args, launches = k_rad4_tile, (re, im, ore, oim, s0r, s0i, n,
                                               logn, n), 1
            lat = _latency(fn, (1,), args)
            note = "radix-4 butterflies, digit-reversed addressed loads"
        elif pname == "dense":
            fn, args, launches = k_dense_tile, (re, im, ore, oim, n, n), 1
            lat = _latency(fn, (1,), args)
            note = "O(N^2) addressed loads + twiddle (no tl.dot: fp32 dot is " \
                   "UNKNOWN and reduced-precision)"
        else:
            t0 = time.perf_counter()
            with xpu_context(0):
                for _ in range(REF_WARMUP):
                    torch.fft.fft(torch.complex(re, im)); torch.cuda.synchronize()
                best = []
                for _ in range(REF_ROUNDS):
                    ta = time.perf_counter()
                    for _ in range(REF_ITERS):
                        torch.fft.fft(torch.complex(re, im))
                    torch.cuda.synchronize()
                    best.append((time.perf_counter() - ta) * 1e3 / REF_ITERS)
            lat = float(min(best))
            y = torch.fft.fft(torch.complex(re, im))
            ore.copy_(y.real); oim.copy_(y.imag)
            fn = None; args = (); launches = 1
            note = "vendor torch.fft.fft on device tensor"
        if pname != "torch":
            y = torch.complex(ore.cpu().double(), oim.cpu().double())
        else:
            y = torch.complex(ore.cpu().double(), oim.cpu().double())
        err = _rel_err(y, ref)
        out["variants"].append({
            "P": pname, "latency_ms": lat, "rel_err": err,
            "pass": err <= TOL_REL, "launches": launches, "note": note,
        })
    lats = [v["latency_ms"] for v in out["variants"]]
    out["disclosure"] = (
        "N=16 single tile: all variants are launch-dominated (host launch "
        "overhead >> kernel work); latency ordering carries no architectural "
        "signal at this scale and is reported only as a swap table for the "
        "P axis of the probe matrix"
    )
    out["all_pass"] = all(v["pass"] for v in out["variants"])
    out["latency_spread_ms"] = max(lats) - min(lats)
    (OUT_DIR / "wp2_p_axis.json").write_text(json.dumps(out, indent=2))
    return out


STEPS = {
    "opsubset": step_opsubset,
    "radix2": step_radix2,
    "radix4": step_radix4,
    "fusion": step_fusion,
    "concurrency": step_concurrency,
    "resident": step_resident,
    "producer_consumer": step_producer_consumer,
    "p_axis": step_p_axis,
}
ORDER = ["opsubset", "radix2", "radix4", "fusion", "concurrency", "resident",
         "producer_consumer", "p_axis"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=list(STEPS) + ["all"], default="all")
    ap.add_argument("--probe-one", default=None)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.probe_one:
        return _probe_one(a.probe_one)
    if a.step == "all":
        results = {}
        for name in ORDER:
            print(f"== step {name} ==", flush=True)
            t0 = time.time()
            try:
                cap = _run_step_capture(name)
                results[name] = {"status": "PASS", "elapsed_s": time.time() - t0,
                                 **cap}
            except Exception as e:
                results[name] = {"status": "FAIL", "error": f"{type(e).__name__}: {e}"}
            print(f"   {results[name]['status']} ({time.time() - t0:.1f}s)",
                  flush=True)
        summary = {
            "schema": "issue10-wp2-summary-v1",
            "issue": "kunlunxin-fft#10",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "script": str(SCRIPT_PATH),
            "tol_rel": TOL_REL,
            "steps": {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()},
        }
        # compact booleans for gates
        def pick(p):
            d = json.loads((OUT_DIR / p).read_text())
            return d
        for fname, key in [("ops_supported.json", "opsubset"),
                           ("wp2_radix2.json", "radix2"),
                           ("wp2_radix4.json", "radix4"),
                           ("wp2_fusion.json", "fusion"),
                           ("wp2_concurrency.json", "concurrency"),
                           ("wp2_resident.json", "resident"),
                           ("wp2_producer_consumer.json", "producer_consumer"),
                           ("wp2_p_axis.json", "p_axis")]:
            if (OUT_DIR / fname).exists():
                d = pick(fname)
                results[key]["artifact"] = fname
                if "all_pass" in d:
                    results[key]["all_pass"] = d["all_pass"]
                if "both_pass" in d:
                    results[key]["both_pass"] = d["both_pass"]
                if "totals" in d:
                    results[key]["totals"] = d["totals"]
        summary["gates"] = {
            "opsubset_runs": results["opsubset"].get("status") == "PASS",
            "radix2_pass": results["radix2"].get("all_pass", False),
            "radix4_pass": results["radix4"].get("all_pass", False),
            "fusion_pass": results["fusion"].get("both_pass", False),
            "concurrency_pass": results["concurrency"].get("all_pass", False),
            "resident_pass": results["resident"].get("all_pass", False),
            "producer_consumer_recorded": results["producer_consumer"].get("status") == "PASS",
            "p_axis_pass": results["p_axis"].get("all_pass", False),
        }
        summary["gates"]["all"] = all(summary["gates"].values())
        (OUT_DIR / "wp2_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary["gates"], indent=2))
        return 0 if summary["gates"]["all"] else 1
    res = STEPS[a.step]()
    print(f"{a.step}: written")
    return 0


def _run_step_capture(name: str) -> dict:
    r = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--step", name],
        capture_output=True, text=True, timeout=STEP_TIMEOUT_S, cwd=str(ROOT),
    )
    (LOG_DIR / f"{name}.log").write_text(r.stdout + "\n--- STDERR ---\n" + r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"step {name} rc={r.returncode}; see {LOG_DIR}/{name}.log")
    return {"log": f"wp2_logs/{name}.log"}


if __name__ == "__main__":
    sys.exit(main())
