"""issue #13 WP2 hardware tests: hybrid4 == measured holdout winner.

Requires the XPU device (``-m hardware``; pytest.ini deselects these
by default). Contract coverage:

1. output parity: ``src/xpu/hybrid4`` runners produce the same values
   as the benchmark implementations the WP1 promotion measured
   (``scripts/issue10_wp3_perf.run_hybrid`` / ``run_hybrid_fused``) —
   both variants, forward and inverse, B=1 and B=16;
2. correctness vs numpy (rel_err <= 1e-4, the holdout gate);
3. production rank-1 dispatch: plan_fft -> execute_fft_plan through a
   promoted-shape composition entry, forward AND inverse semantics,
   plan-reuse cache, exact-key fail-closed outside the table;
4. epilogue norm math through the entry for all four norms.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.hardware

import torch  # noqa: E402

import src.xpu.composition as comp  # noqa: E402
from src.xpu.entry import (  # noqa: E402
    _HYBRID_CACHE, execute_fft_plan, norm_scale, plan_fft)
from src.xpu.hybrid4 import (  # noqa: E402
    Hybrid4Plan, plan_hybrid4, run_hybrid4, run_hybrid4_fused)
from src.xpu.manifest_contract import rows_sha256  # noqa: E402


def _wp3():
    spec = importlib.util.spec_from_file_location(
        "issue13_wp3_perf", ROOT / "scripts" / "issue10_wp3_perf.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["issue13_wp3_perf"] = m
    spec.loader.exec_module(m)
    return m


def _x(b, n, seed=13):
    g = torch.Generator(device="cuda").manual_seed(seed)
    re = torch.rand((b, n), device="cuda", generator=g) - 0.5
    im = torch.rand((b, n), device="cuda", generator=g) - 0.5
    return torch.complex(re, im).to(torch.complex64)


def _max_rel(a: torch.Tensor, b: torch.Tensor) -> float:
    an = a.detach().cpu().numpy().astype(np.complex128)
    bn = b.detach().cpu().numpy().astype(np.complex128)
    denom = max(float(np.abs(an).max()), 1e-30)
    return float(np.abs(an - bn).max()) / denom


def _np_ref(x: np.ndarray, inverse: bool) -> np.ndarray:
    return (np.fft.ifft(x.astype(np.complex128)) if inverse
            else np.fft.fft(x.astype(np.complex128)))


@pytest.mark.parametrize("variant", ["4step-transpose", "4step-fused"])
@pytest.mark.parametrize("n,b,inverse", [(2048, 16, False),
                                         (2048, 16, True),
                                         (2048, 1, False),
                                         (2048, 1, True),
                                         (1024, 16, True),
                                         (1024, 1, True)])
def test_hybrid4_parity_vs_measured_benchmark(variant, n, b, inverse):
    wp3 = _wp3()
    x = _x(b, n)
    plan = plan_hybrid4(n, b, variant=variant)
    got = (run_hybrid4_fused if variant == "4step-fused"
           else run_hybrid4)(x, plan, inverse=inverse)
    xb = x.squeeze(0) if b == 1 else x
    name = ("hybrid_fused_p" if variant == "4step-fused"
            else "hybrid_vendor_p")
    call, _meta = wp3.PATHS[name](xb, inverse)
    re, im = call()
    want = torch.complex(re, im).reshape(b, n)
    assert _max_rel(got, want) <= 1e-5, (
        f"{variant} N{n}B{b} inv={inverse}: deployed != measured "
        f"(max_rel={_max_rel(got, want):g})")


@pytest.mark.parametrize("variant", ["4step-transpose", "4step-fused"])
@pytest.mark.parametrize("inverse", [False, True])
def test_hybrid4_correctness_vs_numpy(variant, inverse):
    n, b = 2048, 16
    x = _x(b, n, seed=17)
    plan = plan_hybrid4(n, b, variant=variant)
    y = (run_hybrid4_fused if variant == "4step-fused"
         else run_hybrid4)(x, plan, inverse=inverse)
    xn = x.cpu().numpy()
    errs = [float(np.abs(y[i].cpu().numpy()
                         - _np_ref(xn[i], inverse)).max()
                  / max(float(np.abs(_np_ref(xn[i], inverse)).max()),
                        1e-30))
            for i in range(b)]
    assert max(errs) <= 1e-4


def _table_with(entries):
    t = {
        "schema_version": 1, "kind": comp.KIND,
        "profile_hash": "p" * 16, "env_hash": "e" * 16,
        "precision": "fp32", "guard_pct": 2.0,
        "contract_id": "a1p1_materialized_subtransform",
        "source_hash": "b" * 40, "dirty": False, "entries": entries,
    }
    t["result_sha256"] = rows_sha256({"rows": entries})
    return t


def _entry(n, b, tdir, variant, executed, cand, inc):
    guard = comp.guard_pct_of(cand, inc)
    return {
        "N": n, "batch": b, "direction": "dit", "precision": "fp32",
        "tile": 64, "core": "vendor",
        "flow_id": "hybrid-A1P1-4step-fused",
        "flow_realization": "decomposition-composition",
        "processing_unit": "vendor-subfft-host-composition",
        "candidate_ms": cand, "incumbent_ms": inc,
        "incumbent_flow": "fft_dit_matrix (local-fused-tile)",
        "guard_pct": guard, "guard_pass": True, "reason": "hw test",
        "contract_id": "a1p1_materialized_subtransform",
        "variant": variant, "executed": executed,
        "transform_type": "c2c", "transform_direction": tdir,
        "normalization": "backward",
        "entry_contract": "rank-1 (N,) plan_fft->execute_fft_plan",
        "evidence": {
            "artifact": "results/p800/issue13_release/wp1_promotion.json",
            "artifact_sha256": "c" * 64, "commit": "a" * 40,
            "sessions": 4, "trials": 640, "confirmation_session": True,
            "err_max": 1e-6, "p50_ms": cand, "mad_ms": 0.0,
            "p95_ms": cand * 1.05,
        },
    }


@pytest.mark.parametrize("variant,executed", [
    ("4step-transpose", "run_hybrid4"),
    ("4step-fused", "run_hybrid4_fused"),
])
@pytest.mark.parametrize("inverse", [False, True])
def test_rank1_entry_dispatch(monkeypatch, variant, executed, inverse):
    n = 2048
    tdir = "inverse" if inverse else "forward"
    table = _table_with([
        _entry(n, 1, tdir, variant, executed, 0.1858, 0.5536)])
    monkeypatch.setattr(comp, "load_composition_table",
                        lambda path=comp.TABLE_PATH: table)
    plan = plan_fft("dit", n, 1, profile_hash="p" * 16,
                    env_hash="e" * 16,
                    transform_direction=tdir, norm="backward")
    assert plan.selection_source == "composition"
    assert plan.executed == executed
    x = _x(1, n, seed=21).squeeze(0)            # rank-1 production input
    y = execute_fft_plan(plan, x)
    assert tuple(y.shape) == (n,)
    ref = _np_ref(x.cpu().numpy().reshape(1, n), inverse)[0]
    err = float(np.abs(y.cpu().numpy() - ref).max()
                / max(float(np.abs(ref).max()), 1e-30))
    assert err <= 1e-4, err
    # plan-reuse: cached Hybrid4Plan for this plan, second call equal
    cached = _HYBRID_CACHE.get(id(plan))
    assert cached is not None and isinstance(cached[0], Hybrid4Plan)
    y2 = execute_fft_plan(plan, x)
    # complex torch.equal NOT IMPLEMENTED on this stack — numpy compare
    assert np.array_equal(y.cpu().numpy(), y2.cpu().numpy())
    assert _HYBRID_CACHE.get(id(plan))[0] is cached[0]
    # exact-key fail-closed: a query outside the entry stays butterfly
    other = plan_fft("dit", n, 16, profile_hash="p" * 16,
                     env_hash="e" * 16, transform_direction=tdir,
                     norm="backward")
    assert other.selection_source != "composition"


def test_entry_epilogue_norms(monkeypatch):
    n = 1024
    table = _table_with([
        _entry(n, 1, "inverse", "4step-fused", "run_hybrid4_fused",
               0.1, 1.0)])
    monkeypatch.setattr(comp, "load_composition_table",
                        lambda path=comp.TABLE_PATH: table)
    x = _x(1, n, seed=33).squeeze(0)
    xn = x.cpu().numpy()
    ifft = np.fft.ifft(xn.astype(np.complex128))
    base = plan_fft("dit", n, 1, profile_hash="p" * 16,
                    env_hash="e" * 16, transform_direction="inverse",
                    norm="backward")
    assert base.selection_source == "composition"
    assert base.executed == "run_hybrid4_fused"
    # a composition entry is promoted under its own semantic (backward
    # here); other-norm QUERIES fail closed (covered host-side). The
    # epilogue math for every norm is exercised by re-stamping the norm
    # on the resolved plan and re-executing (plan-reuse contract).
    from dataclasses import replace
    for norm, want in (("none", ifft * n), ("forward", ifft * n),
                       ("backward", ifft),
                       ("ortho", ifft * (n ** 0.5))):
        plan = replace(base, norm=norm)
        y = execute_fft_plan(plan, x).cpu().numpy()
        err = float(np.abs(y - want).max()
                    / max(float(np.abs(want).max()), 1e-30))
        assert err <= 1e-4, f"norm={norm} err={err:g}"
    # wrong-norm QUERIES never inherit this entry (issue #13 WP2)
    for norm in ("none", "forward", "ortho"):
        q = plan_fft("dit", n, 1, profile_hash="p" * 16,
                     env_hash="e" * 16, transform_direction="inverse",
                     norm=norm)
        assert q.selection_source != "composition", norm
    # norm_scale sanity shared with the epilogue implementation
    assert norm_scale(n, "inverse", "backward") * n == pytest.approx(1.0)


def test_entry_wrapper_overhead_bounded(monkeypatch):
    """Deployed entry path must not lose the measured margin to host
    wrapper overhead (loose bound; WP3 re-measures entry-level)."""
    import time
    n, b = 2048, 16
    table = _table_with([
        _entry(n, b, "inverse", "4step-fused", "run_hybrid4_fused",
               0.3096, 1.8839)])
    monkeypatch.setattr(comp, "load_composition_table",
                        lambda path=comp.TABLE_PATH: table)
    plan = plan_fft("dit", n, b, profile_hash="p" * 16,
                    env_hash="e" * 16, transform_direction="inverse",
                    norm="backward")
    assert plan.executed == "run_hybrid4_fused"
    x = _x(b, n, seed=41)
    wp3 = _wp3()
    hp = plan_hybrid4(n, b, variant="4step-fused")
    for _ in range(3):
        execute_fft_plan(plan, x)
        run_hybrid4_fused(x, hp, inverse=True)
    torch.cuda.synchronize()

    def _p50(fn, trials=64):
        ts = []
        for _ in range(trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        ts.sort()
        return ts[len(ts) // 2]

    p_entry = _p50(lambda: execute_fft_plan(plan, x))
    p_runner = _p50(lambda: run_hybrid4_fused(x, hp, inverse=True))
    ratio = p_entry / max(p_runner, 1e-9)
    assert ratio <= 2.0, (
        f"entry wrapper overhead too high: entry {p_entry:.4f}ms vs "
        f"runner {p_runner:.4f}ms (ratio {ratio:.2f})")
    _ = wp3
