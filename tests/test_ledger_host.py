"""Host tests for src/xpu/ledger.py (issue #8 PR-F, WP2 ledger)."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.xpu.entry import ExecutionPlan  # noqa: E402
from src.xpu import ledger  # noqa: E402


def _plan(executed="fft_dit", n=1024, batch=16, params=None,
          transform_type="c2c", transform_direction="forward"):
    return ExecutionPlan(
        n=n, batch=batch, direction="dit", precision="fp32",
        selection_source="table", reason="t", fallback=False,
        executed=executed, block=1024, params=params or {},
        transform_type=transform_type,
        transform_direction=transform_direction)


def test_phase_noop_when_inactive():
    with ledger.phase("core-invoke"):
        pass
    assert ledger.current_ledger() is None
    # reusable no-op: enter many times
    for _ in range(3):
        with ledger.phase("core-invoke"):
            pass


def test_phase_records_on_active_ledger():
    led = ledger.PhaseLedger(sync=False)
    tok = ledger.ACTIVE.set(led)
    try:
        with ledger.phase("plan-validate"):
            pass
        with ledger.phase("core-invoke"):
            pass
    finally:
        ledger.ACTIVE.reset(tok)
    assert [p["name"] for p in led.phases] == [
        "plan-validate", "core-invoke"]
    assert ledger.current_ledger() is None  # reset restores default
    s = led.summary()
    assert s["kind"].startswith("fresh measured")
    assert s["sync_added"] is False
    assert s["phase_order"] == ["plan-validate", "core-invoke"]
    assert s["phases"]["plan-validate"]["calls"] == 1
    assert s["phases"]["core-invoke"]["mean_ms"] >= 0.0
    assert s["errors"] == []


def test_phase_records_error_and_reraises():
    led = ledger.PhaseLedger(sync=False)
    with pytest.raises(ValueError, match="boom"):
        with led.phase("core-invoke"):
            raise ValueError("boom")
    assert led.phases[0]["error"].startswith("ValueError")
    assert led.summary()["errors"][0]["name"] == "core-invoke"


def test_launch_ledger_m0_batch1():
    lg = ledger.launch_ledger(_plan("fft_dit", n=1024, batch=1))
    assert lg["available"] and lg["family"] == "baseline_m0_rank1_loop"
    assert lg["launches"] == 1 * 10  # batch * log2(n)
    assert lg["gm_bytes"] == 10 * 2 * 1024 * 1 * 8
    assert lg["kind"].startswith("analytic estimate")
    assert lg["estimate"] is True


def test_launch_ledger_m4_folded_batch():
    lg = ledger.launch_ledger(_plan("fft_dit_batch", n=1024, batch=64,
                                    params={"variant": "folded"}))
    assert lg["family"] == "incumbent_m4_folded"
    assert lg["launches"] == 10          # log2(n), batch folded into grid
    assert lg["gm_bytes"] == 10 * 2 * 1024 * 64 * 8


def test_launch_ledger_m2_and_packet():
    lg = ledger.launch_ledger(
        _plan("fft_dit_matrix_batch", n=2048, batch=16,
              params={"tile": 64, "rows": 8}))
    assert lg["family"] == "m2_batch"
    assert lg["launches"] == 2 + 11 - 6  # 2 + log2(n) - log2(tile)
    pk = ledger.launch_ledger(
        _plan("fft_dit_matrix_packet_batch", n=1024, batch=16,
              params={"tile": 64}))
    assert pk["family"] == "m2_packet"


def test_launch_ledger_m5_radix4():
    lg = ledger.launch_ledger(_plan("fft_dit_radix4_batch", n=1024,
                                    batch=64))
    assert lg["family"] == "m5_batch"
    assert lg["launches"] == (10 + 1) // 2


def test_launch_ledger_unmodelled_honest_absence():
    for executed, needle in (("fft_dit_fused", "cost_model"),
                             ("fft_dit_grouped", "cost_model"),
                             ("fft_unknown_fn", "no registered")):
        lg = ledger.launch_ledger(_plan(executed))
        assert lg["available"] is False, executed
        assert needle in lg["why"], executed
        assert "launches" not in lg


def test_launch_ledger_composition_opaque():
    lg = ledger.launch_ledger(_plan("run_hybrid", params={
        "flow_kind": "composition", "tile": 64, "core": "vendor"}))
    assert lg["available"] is False
    assert "plan_ledger" in lg["why"] and "opaque" in lg["why"]
    assert lg["real_compat_ops"] == []


def test_launch_ledger_real_compat_ops():
    r2c = ledger.launch_ledger(_plan("fft_dit", batch=8,
                                     transform_type="r2c"))
    ops = [o["op"] for o in r2c["real_compat_ops"]]
    assert any("pack" in o for o in ops)
    assert any("crop" in o for o in ops)
    assert all(o["kind"] == "analytic estimate" for o in r2c["real_compat_ops"])

    c2r = ledger.launch_ledger(_plan("fft_dit", batch=8,
                                     transform_type="c2r",
                                     transform_direction="inverse"))
    ops = [o["op"] for o in c2r["real_compat_ops"]]
    assert any("hermitian" in o for o in ops)
    assert any("conjugate" in o for o in ops)
    assert any("real-output" in o for o in ops)

    inv = ledger.launch_ledger(_plan(
        "fft_dit", batch=8, transform_direction="inverse"))
    ops = [o["op"] for o in inv["real_compat_ops"]]
    assert any("inverse conjugate" in o for o in ops)


def test_json_serializable():
    lg = ledger.launch_ledger(_plan("fft_dit_matrix_batch",
                                    params={"tile": 64}))
    assert json.dumps(lg)


def _case(n, b, sem, sel, ven):
    return {"N": n, "batch": b, "semantic": sem,
            "selector_ms": sel, "vendor_ms": ven}


def test_batch_fill_analysis_growth():
    cases = [
        _case(1024, 1, "c2c-forward", 1.0, 0.1),
        _case(1024, 16, "c2c-forward", 40.0, 0.1),   # per-pt 2.5 vs 0.000976
        _case(1024, 64, "c2c-forward", 320.0, 0.1),
        _case(1024, 256, "c2c-forward", 2000.0, 0.1),
        _case(512, 64, "c2c-forward", 1.0, 0.1),     # other N, ignored group
    ]
    out = ledger.batch_fill_analysis(cases, n_values=[1024])
    assert len(out) == 1
    row = out[0]
    assert row["kind"].startswith("summary-derived")
    assert len(row["points"]) == 4
    # per-point: B1 = 1e3us/1024pt ... B256 = 2e6us/(1024*256)
    g = row["selector_per_point_growth_smallest_to_largest_B"]
    assert g == pytest.approx(
        (2000.0 * 1e3 / (1024 * 256)) / (1.0 * 1e3 / 1024))
    assert row["poor_batch_fill"] is True
    assert "no worker/wave hardware counters" in row["conclusion_scope"]
    # vendor wall stays flat => per-point cost FALLS with B (growth 1/256)
    assert row["vendor_per_point_growth_smallest_to_largest_B"] == (
        pytest.approx(1.0 / 256))


def test_batch_fill_analysis_needs_three_points():
    cases = [_case(1024, 1, "c2c-forward", 1.0, 0.1),
             _case(1024, 16, "c2c-forward", 2.0, 0.1)]
    assert ledger.batch_fill_analysis(cases) == []
    # rows with missing data skipped
    assert ledger.batch_fill_analysis(
        [{"N": 1, "batch": 1, "semantic": "c2c-forward",
          "selector_ms": None, "vendor_ms": 1.0}]) == []
