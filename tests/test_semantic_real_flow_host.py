"""PR-H dedicated real-transform flows (host-only).

Flow registration / attribution / bounded-oracle budget / plan_fft
wiring of ``plan.params["real_flow"]``, the analytic ledger ops, and
exact torch references of the pack/stitch/pre/unpack epilogues (CPU --
the device parity tests compare the launched kernels against these).
"""

import sys
from pathlib import Path

import pytest

# plain import machinery: pytest.importorskip("torch") leaves the vendor
# import hook in a state where torch_xmlir cannot load libtorch_python
try:
    import torch
except ImportError as e:  # pragma: no cover - ordinary CI without torch
    pytest.skip(f"torch unavailable: {e}", allow_module_level=True)

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_semantic_matrix as sm  # noqa: E402
from src.xpu.entry import (  # noqa: E402
    REAL_COMPAT_MARK,
    REAL_FLOW_MARK,
    SUPPORTED_NORMS,
    ExecutionPlan,
    _check_plan_input,
    execute_fft_plan,
    norm_scale,
    plan_fft,
)
from src.xpu.error_metric import max_rel_error  # noqa: E402
from src.xpu.runner import (  # noqa: E402
    c2r_pre_ref,
    c2r_unpack_ref,
    r2c_stitch_ref,
)
from src.xpu.semantic import CORE_FLOWS, family, family_allows  # noqa: E402

_R2C_FLOW = "dedicated-r2c-pack-v1"
_C2R_FLOW = "dedicated-c2r-hermitian-v1"


# --------------------------- registration ---------------------------


def test_specialized_flows_registered_per_transform():
    assert _R2C_FLOW in family("r2c")
    assert _C2R_FLOW in family("c2r")
    assert CORE_FLOWS <= family("r2c") <= CORE_FLOWS | {_R2C_FLOW}
    assert CORE_FLOWS <= family("c2r") <= CORE_FLOWS | {_C2R_FLOW}
    assert not family_allows("c2c", _R2C_FLOW)
    assert not family_allows("c2c", _C2R_FLOW)
    # the builder's rejected-id lock stays closed for r2c
    assert not family_allows("r2c", "dedicated-hermitian-pack-v1")


# --------------------------- attribution ----------------------------


def test_attribution_real_flow_is_single_sourced():
    assert sm._attribution("fft_dif_matrix_batch",
                           {"real_flow": _R2C_FLOW,
                            "variant": "spread"}) == (
        _R2C_FLOW, "matrix-resident", "dense-dot-fft64")
    assert sm._attribution("fft_dit_matrix",
                           {"real_flow": _C2R_FLOW, "tile": 64}) == (
        _C2R_FLOW, "matrix-resident", "dense-dot-fft64")
    # without real_flow nothing changes
    assert sm._attribution("fft_dif_batch", {"variant": "spread"}) == (
        "batch-spread", "global-stage", "radix2-scalar")
    assert sm._attribution("fft_dit") == (
        "baseline-global-stage", "global-stage", "radix2-scalar")


# ------------------------- bounded oracle set -----------------------


def test_oracle_plans_add_two_dedicated_real_flow_candidates():
    for semantic, flow in (("r2c", _R2C_FLOW), ("c2r", _C2R_FLOW)):
        plans = sm._oracle_plans(1024, 16, "dif", semantic, block=1024)
        assert len(plans) == 8 <= 24
        h = [p for p in plans if p.params.get("real_flow")]
        assert len(h) == 2
        for p in h:
            assert p.params["real_flow"] == flow
            assert p.params["real_compat"] == REAL_FLOW_MARK
            assert p.flow_id == flow
            assert p.flow_realization and p.processing_unit
            assert p.executed  # executed stays the core runner
        compat = [p for p in plans if not p.params.get("real_flow")]
        assert len(compat) == 6
        assert all(p.params.get("real_compat") == REAL_COMPAT_MARK
                   for p in compat)


def test_oracle_plans_batch_one_use_single_runner_dedicated():
    plans = sm._oracle_plans(64, 1, "dit", "r2c", block=64)
    h = [p for p in plans if p.params.get("real_flow")]
    assert {p.executed for p in h} == {"fft_dit_matrix", "fft_dif_matrix"}
    assert all(p.params["real_flow"] == _R2C_FLOW for p in h)


def test_oracle_plans_c2c_unchanged_by_prh():
    plans = sm._oracle_plans(1024, 16, "dit", "c2c-forward", block=1024)
    assert len(plans) == 6
    assert all(not p.params.get("real_flow") for p in plans)
    assert all("real_compat" not in p.params for p in plans)


# --------------------------- plan_fft wiring ------------------------


def _flow_table(params, transform_type, executed="fft_dif_matrix_batch",
                profile_hash="0" * 16, env_hash="e" * 16):
    flow = _R2C_FLOW if transform_type == "r2c" else _C2R_FLOW
    entry = {
        "N": 1024, "batch": 16, "direction": "dit",
        "transform_type": transform_type,
        "transform_direction": ("inverse" if transform_type == "c2r"
                                else "forward"),
        "normalization": "backward",
        "block": 1024, "flow_id": flow,
        "flow_realization": "matrix-resident",
        "processing_unit": "dense-dot-fft64",
        "params": params, "executed": executed,
        "incumbent_ms": 2.0, "challenger_ms": 1.0, "reason": "pr-h test",
    }
    return {"profile_hash": profile_hash, "precision": "fp32",
            "guard_pct": 2.0, "env_hash": env_hash, "entries": [entry]}


def test_plan_fft_marks_dedicated_real_flow():
    params = {"tile": 64, "rows": 8, "variant": "spread",
              "real_flow": _R2C_FLOW}
    table = _flow_table(params, "r2c")
    plan = plan_fft("dit", 1024, batch=16, profile_hash="0" * 16,
                    table=table, env_hash="e" * 16, transform_type="r2c",
                    transform_direction="forward", norm="backward")
    assert not plan.fallback
    assert plan.executed == "fft_dif_matrix_batch"
    assert plan.params["real_flow"] == _R2C_FLOW
    assert plan.params["real_compat"] == REAL_FLOW_MARK
    assert plan.flow_id == _R2C_FLOW


def test_plan_fft_rejects_family_illegal_real_flow():
    params = {"tile": 64, "rows": 8, "variant": "spread",
              "real_flow": "dedicated-hermitian-pack-v1"}
    table = _flow_table(params, "r2c")
    with pytest.raises(ValueError, match="not in the r2c family"):
        plan_fft("dit", 1024, batch=16, profile_hash="0" * 16,
                 table=table, env_hash="e" * 16, transform_type="r2c",
                 transform_direction="forward", norm="backward")


def test_plan_fft_rejects_real_flow_on_c2c():
    params = {"tile": 64, "real_flow": _R2C_FLOW}
    table = _flow_table(params, "c2c", executed="fft_dif_matrix")
    with pytest.raises(ValueError, match="not in the c2c family"):
        plan_fft("dit", 1024, batch=16, profile_hash="0" * 16,
                 table=table, env_hash="e" * 16, transform_type="c2c",
                 transform_direction="forward", norm="backward")


# --------------------- execute contract (fail-closed) ---------------


def _cpu_plan(transform_type, real_flow, executed="fft_dif",
              n=64, batch=1, transform_direction="forward"):
    return ExecutionPlan(
        n=n, batch=batch, direction="dif", precision="fp32",
        selection_source="oracle-candidate", reason="pr-h host gate",
        fallback=False, executed=executed, block=64,
        flow_id="x", flow_realization="global-stage",
        processing_unit="radix2-scalar",
        params={"real_flow": real_flow},
        transform_type=transform_type,
        transform_direction=transform_direction, norm="backward")


def test_execute_rejects_unknown_real_flow_before_any_launch():
    plan = _cpu_plan("r2c", "dedicated-hermitian-pack-v1")
    x = torch.zeros(64, dtype=torch.float32)  # contract-valid CPU tensor
    with pytest.raises(ValueError, match="not in the r2c family"):
        execute_fft_plan(plan, x)


def test_execute_rejects_real_flow_on_c2c_before_any_launch():
    plan = _cpu_plan("c2c", _R2C_FLOW,
                     transform_direction="forward")
    x = torch.zeros(64, dtype=torch.complex64)
    with pytest.raises(ValueError, match="r2c/c2r only"):
        execute_fft_plan(plan, x)


def test_check_plan_input_contract_unaffected_by_real_flow():
    # dedicated c2r keeps the half-spectrum input contract
    plan = _cpu_plan("c2r", _C2R_FLOW, transform_direction="inverse",
                     n=64, batch=4)
    _check_plan_input(plan, (4, 33), "complex64")
    with pytest.raises(ValueError, match=r"\(4, 33\)"):
        _check_plan_input(plan, (4, 64), "complex64")
    with pytest.raises(TypeError, match="complex64"):
        _check_plan_input(plan, (4, 33), "float32")


# --------------------------- analytic ledger ------------------------


def _ledger_plan(transform_type, real_flow, executed, n=1024, batch=16):
    return ExecutionPlan(
        n=n, batch=batch, direction="dif", precision="fp32",
        selection_source="oracle-candidate", reason="pr-h ledger",
        fallback=False, executed=executed, block=1024,
        flow_id=real_flow, flow_realization="matrix-resident",
        processing_unit="dense-dot-fft64",
        params={"real_flow": real_flow, "real_compat": REAL_FLOW_MARK,
                "tile": 64, "rows": 8, "variant": "spread"},
        transform_type=transform_type,
        transform_direction=("forward" if transform_type == "r2c"
                             else "inverse"),
        norm="backward")


def test_launch_ledger_dedicated_r2c_ops():
    from src.xpu.ledger import launch_ledger
    out = launch_ledger(_ledger_plan("r2c", _R2C_FLOW,
                                     "fft_dif_matrix_batch"))
    assert out["available"] is True
    ops = [o["op"] for o in out["real_compat_ops"]]
    assert any("stitch" in op for op in ops)
    assert any("pack" in op for op in ops)
    # crop + separate normalization are gone: fused into the stitch store
    assert not any("crop" in op for op in ops)
    assert not any("normalization epilogue" in op for op in ops)


def test_launch_ledger_dedicated_c2r_ops():
    from src.xpu.ledger import launch_ledger
    out = launch_ledger(_ledger_plan("c2r", _C2R_FLOW,
                                     "fft_dif_matrix_batch"))
    assert out["available"] is True
    ops = [o["op"] for o in out["real_compat_ops"]]
    assert any("pre-twiddle" in op for op in ops)
    assert any("mirror" in op and "fused" in op for op in ops)
    assert not any("hermitian-rebuild" in op for op in ops)
    assert not any("normalization epilogue" in op for op in ops)


def test_launch_ledger_compat_marks_unchanged():
    from src.xpu.ledger import launch_ledger
    plan = _ledger_plan("r2c", _R2C_FLOW, "fft_dif_matrix_batch")
    compat_plan = ExecutionPlan(
        **{**plan.__dict__,
           "params": {"real_compat": REAL_COMPAT_MARK, "tile": 64,
                      "rows": 8, "variant": "spread"}})
    out = launch_ledger(compat_plan)
    ops = [o["op"] for o in out["real_compat_ops"]]
    assert any("crop" in op for op in ops)
    assert any("normalization epilogue" in op for op in ops)


# --------------------- exact torch references (CPU) ------------------


def _torch_norm(norm):
    return None if norm == "none" else norm


def _torch_irfft(X, n, norm):
    # repo semantics: norm="none" means NO normalization anywhere, so the
    # inverse is unscaled (torch's norm=None is backward = 1/N)
    if norm == "none":
        return torch.fft.irfft(X, n=n, norm="backward") * n
    return torch.fft.irfft(X, n=n, norm=norm)


@pytest.mark.parametrize("norm", SUPPORTED_NORMS)
@pytest.mark.parametrize("n,b", [(64, 1), (256, 8)])
def test_r2c_stitch_ref_matches_torch_rfft(norm, n, b):
    rng = np.random.default_rng(n * 17 + b)
    a = rng.standard_normal((b, n)).astype(np.float32)
    x = torch.from_numpy(a if b > 1 else a[0])
    z = torch.fft.fft(torch.complex(x[..., 0::2], x[..., 1::2]))
    got = r2c_stitch_ref(z, n, norm_scale(n, "forward", norm))
    want = torch.fft.rfft(x, norm=_torch_norm(norm))
    assert got.shape == want.shape
    rel = max_rel_error(got.numpy(), want.numpy())
    assert rel < 1e-4, f"r2c ref norm={norm} N={n} B={b}: rel={rel:.3e}"


@pytest.mark.parametrize("norm", SUPPORTED_NORMS)
@pytest.mark.parametrize("n,b", [(64, 1), (256, 8)])
def test_c2r_chain_ref_matches_torch_irfft(norm, n, b):
    rng = np.random.default_rng(n * 31 + b)
    a = rng.standard_normal((b, n)).astype(np.float32)
    x = torch.from_numpy(a if b > 1 else a[0])
    X = torch.fft.rfft(x, norm="backward")
    # torch.fft ignores DC/Nyquist imaginary parts: garbage must vanish
    Xg = X.clone()
    Xg[..., 0] = Xg[..., 0] + 0.125j
    Xg[..., -1] = Xg[..., -1] + 0.75j
    zpre = c2r_pre_ref(Xg, n)
    u = torch.fft.fft(zpre)  # unnormalized forward core
    got = c2r_unpack_ref(u, n, 2.0 * norm_scale(n, "inverse", norm))
    want = _torch_irfft(X, n, norm)
    assert got.shape == want.shape and got.dtype == torch.float32
    rel = max_rel_error(got.numpy(), want.numpy())
    assert rel < 1e-4, f"c2r ref norm={norm} N={n} B={b}: rel={rel:.3e}"


def test_ref_bin_contracts():
    z = torch.zeros(32, dtype=torch.complex64)
    with pytest.raises(ValueError, match="N/2"):
        r2c_stitch_ref(z, 128)
    X = torch.zeros(64, dtype=torch.complex64)  # needs N/2+1 = 65
    with pytest.raises(ValueError, match="N/2\\+1"):
        c2r_pre_ref(X, 128)
    u = torch.zeros(63, dtype=torch.complex64)
    with pytest.raises(ValueError, match="N/2"):
        c2r_unpack_ref(u, 128)


# --- design-space / lowering contract on dedicated entries (PR-H) ---

def _dedicated_entry(flow, executed, realization, unit, params, n=1024,
                     batch=16, ttype="r2c"):
    return {
        "N": n, "batch": batch, "direction": "dit",
        "transform_type": ttype,
        "transform_direction": "forward" if ttype == "r2c" else "inverse",
        "normalization": "backward",
        "block": 1024,
        "flow_id": flow,
        "flow_realization": realization,
        "processing_unit": unit,
        "executed": executed,
        "params": {**params, "batch": batch, "real_flow": flow,
                   "real_compat": REAL_FLOW_MARK,
                   "oracle_name": "h_test"},
        "evidence": {"incumbent": "matrix-incumbent", "challenger": "h_test",
                     "gain_pct": 50.0, "sessions": 3,
                     "protocol_id": "semantic-matrix-v1", "semantic": "r2c",
                     "source": "semantic-matrix confirmation",
                     "executed": executed},
        "reason": "test entry",
    }


def test_dedicated_entry_passes_design_space_contract():
    from src.xpu.selector import entry_contract_violations
    cases = [
        # B=1 matrix core (h_dit/h_dif t64)
        _dedicated_entry("dedicated-r2c-pack-v1", "fft_dit_matrix",
                         "matrix-resident", "dense-dot-fft64",
                         {"tile": 64, "rows": 8}, n=64, batch=1),
        # batch matrix core (h_dif t64, spread)
        _dedicated_entry("dedicated-r2c-pack-v1", "fft_dif_matrix_batch",
                         "matrix-resident", "dense-dot-fft64",
                         {"tile": 64, "rows": 8, "variant": "spread"}),
        # batch radix-2 core (h_spread, spread)
        _dedicated_entry("dedicated-c2r-hermitian-v1", "fft_dit_batch",
                         "global-stage", "radix2-scalar",
                         {"variant": "spread"}, ttype="c2r"),
    ]
    for entry in cases:
        bad = entry_contract_violations(entry)
        assert bad == [], f"{entry['flow_id']} {entry['params']}: {bad}"


def test_contract_rejects_unknown_flow_on_same_geometry():
    from src.xpu.selector import entry_contract_violations
    entry = _dedicated_entry("bogus-flow", "fft_dit_matrix",
                             "matrix-resident", "dense-dot-fft64",
                             {"tile": 64, "rows": 8}, n=64, batch=1)
    bad = entry_contract_violations(entry)
    assert bad, "unknown flow id must still violate the flow-identity rules"


def test_specialized_routing_registry_matches_families():
    from src.xpu.candidates import SPECIALIZED_ROUTING
    from src.xpu.semantic import SPECIALIZED_FLOWS
    union = frozenset().union(*SPECIALIZED_FLOWS.values())
    assert SPECIALIZED_ROUTING == union
    assert SPECIALIZED_ROUTING == frozenset(
        {"dedicated-r2c-pack-v1", "dedicated-c2r-hermitian-v1"})
