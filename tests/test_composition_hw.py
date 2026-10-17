"""Composition path correctness on device (issue #6 PR-E).

Hardware-marked: needs the vendor xpytorch stack + reachable P800.
Exercises the A1P1 composition execution (adapter_hybrid run_hybrid)
through the entry plan/execute contract, including norm/inverse
epilogues, the c2c-only boundary, plan-reuse caching, and - when a
real promotion exists - the exact-shape plan_fft wiring.
"""

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

try:
    import torch
    import torch_xmlir  # noqa: F401
except Exception as e:  # pragma: no cover - absent in ordinary CI
    pytest.skip(f"vendor xpytorch stack unavailable: {e}",
                allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytestmark = pytest.mark.hardware

import src.xpu.composition as comp  # noqa: E402
from src.xpu.config import DeviceProfile  # noqa: E402
from src.xpu.entry import (  # noqa: E402
    ExecutionPlan, execute_fft_plan, plan_fft)

PROF = DeviceProfile.load().hash()
ENV = "e" * 16  # synthetic binding: explicit env_hash pins the test


def _entry(n, b, cand, inc):
    g = comp.guard_pct_of(cand, inc)
    return {
        "N": n, "batch": b, "direction": "dit", "precision": "fp32",
        "tile": 64, "core": "vendor",
        "flow_id": "hybrid-A1P1",
        "flow_realization": "decomposition-composition",
        "processing_unit": "vendor-subfft-host-composition",
        "candidate_ms": cand, "incumbent_ms": inc,
        "incumbent_flow": "A0P0-production",
        "guard_pct": g, "guard_pass": True, "reason": "hw test",
        "contract_id": "a1p1_materialized_subtransform",
        "evidence": {
            "artifact": "results/p800/composition_confirm.json",
            "artifact_sha256": "c" * 64, "commit": "a" * 40,
            "sessions": 4, "trials": 30, "confirmation_session": True,
            "err_max": 1e-4, "p50_ms": cand, "mad_ms": 0.0,
            "p95_ms": cand,
        },
    }


def _table(*cells):
    from src.xpu.manifest_contract import rows_sha256
    entries = [_entry(*c) for c in cells]
    return {
        "schema_version": 1, "kind": comp.KIND,
        "profile_hash": PROF, "env_hash": ENV,
        "precision": "fp32", "guard_pct": 2.0,
        "contract_id": "a1p1_materialized_subtransform",
        "created_at": "2026-10-17T00:00:00+0000",
        "source_hash": "b" * 40, "dirty": False,
        "entries": entries,
        "result_sha256": rows_sha256({"rows": entries}),
    }


def _input(n, b, seed=1234):
    g = torch.Generator().manual_seed(seed)
    if b == 1:
        re = torch.randn(n, dtype=torch.float32, generator=g)
        im = torch.randn(n, dtype=torch.float32, generator=g)
        return torch.complex(re, im).to("cuda")
    re = torch.randn(b, n, dtype=torch.float32, generator=g)
    im = torch.randn(b, n, dtype=torch.float32, generator=g)
    return torch.complex(re, im).to("cuda")


def _err(y, ref):
    a = y.detach().cpu().numpy()
    return float(np.abs(a - ref).max() / max(np.abs(ref).max(), 1e-30))


def test_composition_forward_b16(monkeypatch):
    table = _table((2048, 16, 0.2345, 1.5064))
    monkeypatch.setattr(comp, "load_composition_table",
                        lambda path=comp.TABLE_PATH: table)
    plan = plan_fft("dit", 2048, 16, profile_hash=PROF, env_hash=ENV)
    assert plan.selection_source == "composition"
    x = _input(2048, 16)
    y = execute_fft_plan(plan, x)
    ref = np.fft.fft(x.detach().cpu().numpy().astype(np.complex128),
                     axis=-1)
    assert _err(y, ref) <= 1e-4
    # plan-reuse cache: one HybridPlan per ExecutionPlan
    from src.xpu import entry as entry_mod
    assert len(entry_mod._HYBRID_CACHE) >= 1
    key = id(plan)
    hp = entry_mod._HYBRID_CACHE[key]
    _ = execute_fft_plan(plan, x)
    assert entry_mod._HYBRID_CACHE[key] is hp


def test_composition_forward_b1_rank1(monkeypatch):
    table = _table((64, 1, 0.05, 0.30))
    monkeypatch.setattr(comp, "load_composition_table",
                        lambda path=comp.TABLE_PATH: table)
    plan = plan_fft("dit", 64, 1, profile_hash=PROF, env_hash=ENV)
    assert plan.selection_source == "composition"
    x = _input(64, 1)
    assert x.dim() == 1
    y = execute_fft_plan(plan, x)
    assert y.dim() == 1
    ref = np.fft.fft(x.detach().cpu().numpy().astype(np.complex128))
    assert _err(y, ref) <= 1e-4


def test_composition_inverse_and_norms(monkeypatch):
    table = _table((1024, 4, 0.10, 0.50))
    monkeypatch.setattr(comp, "load_composition_table",
                        lambda path=comp.TABLE_PATH: table)
    x = _input(1024, 4)
    xn = x.detach().cpu().numpy().astype(np.complex128)
    cases = [
        ("backward", np.fft.ifft(xn, axis=-1)),
        ("ortho", np.fft.ifft(xn, norm="ortho", axis=-1)),
        ("forward", np.fft.ifft(xn, axis=-1) * 1024),  # inverse*1 = N-scaled
    ]
    for norm, ref in cases:
        plan = plan_fft("dit", 1024, 4, profile_hash=PROF, env_hash=ENV,
                        transform_direction="inverse", norm=norm)
        assert plan.selection_source == "composition"
        y = execute_fft_plan(plan, x)
        assert _err(y, ref) <= 1e-4, norm


def test_composition_c2c_only_boundary(monkeypatch):
    table = _table((2048, 16, 0.2345, 1.5064))
    monkeypatch.setattr(comp, "load_composition_table",
                        lambda path=comp.TABLE_PATH: table)
    # r2c at the same shape keeps the butterfly compat path
    plan = plan_fft("dit", 2048, 16, profile_hash=PROF, env_hash=ENV,
                    transform_type="r2c")
    assert plan.selection_source != "composition"
    # a hand-built composition plan with a real transform raises
    bad = ExecutionPlan(
        n=2048, batch=16, direction="dit", precision="fp32",
        selection_source="composition", reason="t", fallback=False,
        executed="run_hybrid",
        params={"flow_kind": "composition", "tile": 64,
                "core": "vendor"},
        transform_type="r2c")
    real = _input(2048, 16).real.contiguous()  # r2c contract input
    with pytest.raises(ValueError, match="c2c"):
        execute_fft_plan(bad, real)


def test_real_promotion_wiring_if_present():
    """After scripts/promote_composition.py runs, plan_fft must route
    the exact shape through the real table (no monkeypatch)."""
    table = comp.load_composition_table()
    if table is None:
        pytest.skip("no composition table promoted yet")
    matched = [e for e in table["entries"]
               if e["N"] == 2048 and e["batch"] == 16
               and e["direction"] == "dit"]
    if not matched:
        pytest.skip("2048x16 not promoted in this table")
    plan = plan_fft("dit", 2048, 16, profile_hash=PROF)
    assert plan.selection_source == "composition"
    assert plan.params["flow_kind"] == "composition"
    x = _input(2048, 16)
    y = execute_fft_plan(plan, x)
    ref = np.fft.fft(x.detach().cpu().numpy().astype(np.complex128),
                     axis=-1)
    assert _err(y, ref) <= 1e-4
