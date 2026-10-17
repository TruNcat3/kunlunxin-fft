"""Host-only entry tests: the plan must match the table and M0-close.

Review P1: ``plan_fft`` resolves the frozen table onto the exact runner
function; abstain/undispatchable combinations fall back to M0 with the
reason. No torch, no device.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.xpu.config import DeviceProfile  # noqa: E402
from src.xpu.entry import plan_fft  # noqa: E402
from src.xpu.entry import _dispatch  # noqa: E402
from src.xpu.entry import _check_plan_input  # noqa: E402
from src.xpu.entry import norm_scale  # noqa: E402
from src.xpu.entry import ExecutionPlan  # noqa: E402
from src.xpu.selector import Selection, load_table  # noqa: E402

TABLE = ROOT / "config" / "selector_table.json"

_KNOWN = {
    ("global-stage", "radix2-scalar", "dit", 1): "fft_dit",
    ("global-stage", "radix2-scalar", "dif", 1): "fft_dif",
    ("global-stage", "radix2-scalar", "dit", 4): "fft_dit_batch",
    ("global-stage", "radix4-scalar", "dit", 1): "fft_dit_radix4",
    ("global-stage", "radix4-scalar", "dit", 4): "fft_dit_radix4_batch",
    ("global-stage", "radix4-scalar", "dif", 16): "fft_dif_radix4_batch",
    ("fused-inplace", "radix2-scalar", "dit", 1): "fft_dit_fused",
    ("matrix-resident", "dense-dot-fft64", "dif", 1): "fft_dif_matrix",
    ("matrix-resident", "dense-dot-fft64", "dit", 4): "fft_dit_matrix_batch",
    ("matrix-resident", "dense-dot-fft64", "dif", 64): "fft_dif_matrix_batch",
    ("grouped-boundary", "composite-linear-map", "dit", 1): "fft_dit_grouped",
}


def _sel(real, unit, direction, batch=1):
    return Selection(n=2048, direction=direction, batch=batch,
                     precision="fp32", source="table", reason="t",
                     block=256, flow_id="x", flow_realization=real,
                     processing_unit=unit, params={})


@pytest.mark.parametrize("real,unit,direction,batch,expect",
                         [(k[0], k[1], k[2], k[3], v)
                          for k, v in _KNOWN.items()])
def test_dispatch_covers_known_lowerings(real, unit, direction, batch, expect):
    assert _dispatch(_sel(real, unit, direction, batch)) == expect


@pytest.mark.parametrize("kwargs", [
    ("fused-inplace", "radix2-scalar", "dif", 1),   # no DIF fused lowering
    ("fused-inplace", "radix2-scalar", "dit", 4),  # batched M1: no lowering
    ("grouped-boundary", "composite-linear-map", "dit", 16),  # batched M3: none
])
def test_dispatch_returns_none_for_unlowered(kwargs):
    real, unit, direction, batch = kwargs
    assert _dispatch(_sel(real, unit, direction, batch)) is None


def test_plan_rejects_invalid_direction():
    """inverse/r2c must raise, never fall back onto a different transform."""
    with pytest.raises(ValueError, match="unsupported direction"):
        plan_fft("inverse", 16, profile_hash="0" * 16, table={"entries": []})
    with pytest.raises(ValueError, match="unsupported direction"):
        plan_fft("r2c", 16, profile_hash="0" * 16, table={"entries": []})


def test_plan_rejects_invalid_precision():
    with pytest.raises(ValueError, match="unsupported precision"):
        plan_fft("dit", 16, precision="fp64", profile_hash="0" * 16,
                 table={"entries": []})


def test_plan_executes_table_winner():
    if not TABLE.exists():
        pytest.skip("selector table not built")
    t = load_table(str(TABLE))
    p = DeviceProfile.load()
    plan = plan_fft("dit", 2048, profile_hash=p.hash(), table=t,
                    env_hash=t["env_hash"])
    assert not plan.fallback
    assert plan.selection_source == "table"
    assert plan.executed in ("fft_dit_matrix", "fft_dit_grouped",
                             "fft_dit_radix4", "fft_dit")
    assert plan.block and plan.flow_realization and plan.processing_unit


def test_plan_falls_back_on_unknown_n():
    if not TABLE.exists():
        pytest.skip("selector table not built")
    t = load_table(str(TABLE))
    p = DeviceProfile.load()
    plan = plan_fft("dit", 3000, profile_hash=p.hash(), table=t)
    assert plan.fallback
    assert plan.executed == "fft_dit"
    assert plan.selection_source == "abstain"
    assert plan.reason


def test_plan_falls_back_on_missing_profile():
    """profile_hash=None can never bypass into a table execution."""
    if not TABLE.exists():
        pytest.skip("selector table not built")
    t = load_table(str(TABLE))
    plan = plan_fft("dit", 2048, profile_hash=None, table=t)
    assert plan.fallback
    assert plan.executed == "fft_dit"
    assert "profile_hash is required" in plan.reason


def test_plan_falls_back_on_profile_mismatch():
    if not TABLE.exists():
        pytest.skip("selector table not built")
    t = load_table(str(TABLE))
    plan = plan_fft("dit", 2048, profile_hash="f" * 16, table=t)
    assert plan.fallback
    assert plan.executed == "fft_dit"
    assert "profile hash" in plan.reason


def test_plan_falls_back_for_dif_fused_table_entry():
    """A (hypothetical) DIF fused table entry must not claim execution."""
    if not TABLE.exists():
        pytest.skip("selector table not built")
    t = load_table(str(TABLE))
    fake = dict(t)
    fake["entries"] = [{
        "N": 16, "batch": 1, "direction": "dif",
        "block": 1024, "flow_id": "local-fused-tile",
        "flow_realization": "fused-inplace",
        "processing_unit": "radix2-scalar", "params": {"tile": 64},
        "incumbent_ms": 1.0, "challenger_ms": 0.5, "reason": "fake",
    }]
    p = DeviceProfile.load()
    plan = plan_fft("dif", 16, profile_hash=p.hash(), table=fake,
                    env_hash=fake["env_hash"])
    assert plan.fallback
    assert plan.executed == "fft_dif"
    assert "no lowering" in plan.reason


def test_plan_batch_entry_dispatches_m4():
    if not TABLE.exists():
        pytest.skip("selector table not built")
    t = load_table(str(TABLE))
    p = DeviceProfile.load()
    plan = plan_fft("dit", 4096, batch=16, profile_hash=p.hash(), table=t,
                    env_hash=t["env_hash"])
    assert not plan.fallback
    assert plan.executed == "fft_dit_batch"
    assert plan.params.get("variant") in ("spread", "folded")


def test_plan_falls_back_for_batched_m1_table_entry():
    """A (hypothetical) batched M1 table entry must fall back to M4/M0."""
    if not TABLE.exists():
        pytest.skip("selector table not built")
    t = load_table(str(TABLE))
    fake = dict(t)
    fake["entries"] = [{
        "N": 16, "batch": 4, "direction": "dit",
        "block": 1024, "flow_id": "local-fused-tile",
        "flow_realization": "fused-inplace",
        "processing_unit": "radix2-scalar", "params": {"tile": 64},
        "incumbent_ms": 1.0, "challenger_ms": 0.5, "reason": "fake",
    }]
    p = DeviceProfile.load()
    plan = plan_fft("dit", 16, batch=4, profile_hash=p.hash(), table=fake,
                    env_hash=fake["env_hash"])
    assert plan.fallback
    assert plan.executed == "fft_dit_batch"
    assert "no lowering" in plan.reason


def test_entry_module_does_not_import_torch():
    src = (ROOT / "src" / "xpu" / "entry.py").read_text()
    # torch may appear only as a function-local import
    lines = [ln for ln in src.splitlines()
             if "import torch" in ln and not ln.strip().startswith("#")]
    assert all("    import torch" in ln for ln in lines), lines


def _plan(n=16, batch=1, precision="fp32"):
    return ExecutionPlan(
        n=n, batch=batch, direction="dit", precision=precision,
        selection_source="table", reason="t", fallback=False,
        executed="fft_dit", block=1024, flow_id="x",
        flow_realization="global-stage", processing_unit="radix2-scalar",
        params={})


def test_plan_input_accepts_matching_shapes():
    _check_plan_input(_plan(16, 1), (16,), "complex64")
    _check_plan_input(_plan(4096, 16), (16, 4096), "complex64")


def test_plan_input_rejects_wrong_n():
    """An N=16 plan must not silently execute an N=32 tensor."""
    with pytest.raises(ValueError, match="N=16"):
        _check_plan_input(_plan(16, 1), (32,), "complex64")


def test_plan_input_rejects_wrong_rank_for_b1():
    with pytest.raises(ValueError, match="rank-1"):
        _check_plan_input(_plan(16, 1), (1, 16), "complex64")


def test_plan_input_rejects_wrong_batch_or_rank():
    with pytest.raises(ValueError, match=r"\(4, 4096\)"):
        _check_plan_input(_plan(4096, 4), (16, 4096), "complex64")
    with pytest.raises(ValueError, match=r"\(4, 4096\)"):
        _check_plan_input(_plan(4096, 4), (4096,), "complex64")


def test_plan_input_rejects_wrong_dtype():
    with pytest.raises(TypeError, match="complex64"):
        _check_plan_input(_plan(16, 1), (16,), "float32")


# ---------------- WP1 inverse C2C + normalization (issue #6 PR-B) ----------


def test_plan_defaults_keep_forward_unscaled_semantics():
    if not TABLE.exists():
        pytest.skip("selector table not built")
    table = load_table(str(TABLE))
    plan = plan_fft("dit", 1024, profile_hash="0" * 16, table=table)
    assert plan.transform_type == "c2c"
    assert plan.transform_direction == "forward"
    assert plan.norm == "backward"
    assert plan.as_report().transform_direction == "forward"


def test_norm_scale_matches_torch_fft_semantics():
    n = 16
    assert norm_scale(n, "forward", "none") == 1.0
    assert norm_scale(n, "forward", "forward") == 1.0 / n
    assert norm_scale(n, "forward", "backward") == 1.0
    assert norm_scale(n, "inverse", "none") == 1.0
    assert norm_scale(n, "inverse", "forward") == 1.0
    assert norm_scale(n, "inverse", "backward") == 1.0 / n
    for d in ("forward", "inverse"):
        assert norm_scale(n, d, "ortho") == 1.0 / (n ** 0.5)


@pytest.mark.parametrize("kwargs", [
    dict(transform_type="r2c"),
    dict(transform_type="c2r"),
    dict(transform_direction="sideways"),
    dict(norm="l2"),
])
def test_plan_rejects_unsupported_transform_semantics(kwargs):
    with pytest.raises(ValueError):
        plan_fft("dit", 16, profile_hash="0" * 16,
                 table={"entries": []}, **kwargs)


def test_plan_rejects_inverse_before_table_query():
    """A raising table proves validation happens before any query."""
    class _Boom:
        def __getitem__(self, key):
            raise AssertionError("table queried before validation")

    for bad in (dict(transform_type="r2c"), dict(norm="madeup"),
                dict(transform_direction="x")):
        with pytest.raises(ValueError):
            plan_fft("dit", 16, table=_Boom(), **bad)


def test_plan_inverse_carries_semantics_on_abstain_and_table():
    if not TABLE.exists():
        pytest.skip("selector table not built")
    table = load_table(str(TABLE))
    abstain = plan_fft("dit", 1024, profile_hash=None, table=table,
                       transform_direction="inverse", norm="ortho")
    assert abstain.transform_direction == "inverse"
    assert abstain.norm == "ortho"
    assert abstain.fallback is True
    prof = DeviceProfile.load().hash()
    winner = plan_fft("dit", 1024, profile_hash=prof, table=table,
                      transform_direction="inverse", norm="backward")
    fwd = plan_fft("dit", 1024, profile_hash=prof, table=table)
    # conjugate-trick lowerings reuse the forward incumbent
    assert winner.executed == fwd.executed
    assert winner.block == fwd.block
    assert winner.transform_direction == "inverse"
    assert winner.fallback == fwd.fallback


def test_plan_executes_table_winner_inverse_reports_same_lowering():
    if not TABLE.exists():
        pytest.skip("no production table")
    table = load_table(str(TABLE))
    prof = DeviceProfile.load().hash()
    plan = plan_fft("dif", 2048, batch=16, profile_hash=prof, table=table,
                    transform_direction="inverse", norm="ortho")
    fwd = plan_fft("dif", 2048, batch=16, profile_hash=prof, table=table)
    assert plan.executed == fwd.executed
    assert plan.params == fwd.params
    assert plan.norm == "ortho"
