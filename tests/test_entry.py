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
from src.xpu.selector import Selection, load_table  # noqa: E402

TABLE = ROOT / "config" / "selector_table.json"

_KNOWN = {
    ("global-stage", "radix2-scalar", "dit", 1): "fft_dit",
    ("global-stage", "radix2-scalar", "dif", 1): "fft_dif",
    ("global-stage", "radix2-scalar", "dit", 4): "fft_dit_batch",
    ("global-stage", "radix4-scalar", "dit", 1): "fft_dit_radix4",
    ("fused-inplace", "radix2-scalar", "dit", 1): "fft_dit_fused",
    ("matrix-resident", "dense-dot-fft64", "dif", 1): "fft_dif_matrix",
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
    ("matrix-resident", "dense-dot-fft64", "dit", 4),
    ("global-stage", "radix4-scalar", "dit", 4),
])
def test_dispatch_returns_none_for_unlowered(kwargs):
    real, unit, direction, batch = kwargs
    assert _dispatch(_sel(real, unit, direction, batch)) is None


def test_plan_executes_table_winner():
    if not TABLE.exists():
        pytest.skip("selector table not built")
    t = load_table(str(TABLE))
    p = DeviceProfile.load()
    plan = plan_fft("dit", 2048, profile_hash=p.hash(), table=t)
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
    plan = plan_fft("dif", 16, profile_hash=p.hash(), table=fake)
    assert plan.fallback
    assert plan.executed == "fft_dif"
    assert "no lowering" in plan.reason


def test_plan_batch_entry_dispatches_m4():
    if not TABLE.exists():
        pytest.skip("selector table not built")
    t = load_table(str(TABLE))
    p = DeviceProfile.load()
    plan = plan_fft("dit", 4096, batch=16, profile_hash=p.hash(), table=t)
    assert not plan.fallback
    assert plan.executed == "fft_dit_batch"
    assert plan.params.get("variant") in ("spread", "folded")


def test_entry_module_does_not_import_torch():
    src = (ROOT / "src" / "xpu" / "entry.py").read_text()
    # torch may appear only as a function-local import
    lines = [ln for ln in src.splitlines()
             if "import torch" in ln and not ln.strip().startswith("#")]
    assert all("    import torch" in ln for ln in lines), lines
