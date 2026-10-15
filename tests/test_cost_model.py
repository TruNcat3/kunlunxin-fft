"""Issue #3 P1.3: structured cost model, bounded search, holdout validation."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.xpu import cost_model as cm  # noqa: E402


def _calib():
    try:
        from src.xpu.calibration import load
        return load()
    except FileNotFoundError:
        pytest.skip("config/calibration.json missing (run hw_probe)")


@pytest.mark.parametrize(
    "family,n,batch,launches",
    [("baseline_m0_rank1_loop", 1024, 16, 160),
     ("incumbent_m4_folded", 1024, 16, 10),
     ("m4_spread", 1024, 16, 10),
     ("m5_batch", 1024, 16, 5),
     ("m2_batch", 1024, 16, 6)],
)
def test_axes_launch_counts(family, n, batch, launches):
    ax = cm.axes(family, n, batch, {"tile": 64, "rows": 8})
    assert ax["launches"] == launches
    assert ax["gm_bytes"] > 0 and ax["flops"] > 0


def test_m0_gm_bytes_not_scaled_by_batch_again():
    # Each m0 launch moves one row: gm = B*logn * 16N, not B*logn * 16NB.
    ax = cm.axes("baseline_m0_rank1_loop", 1024, 16)
    assert ax["gm_bytes"] == 160 * 2 * 1024 * 8


def test_m2_axes_track_tile():
    small = cm.axes("m2_batch", 1024, 16, {"tile": 32, "rows": 8})
    big = cm.axes("m2_batch", 1024, 16, {"tile": 128, "rows": 8})
    assert small["launches"] > big["launches"]       # fewer tail stages
    assert small["flops"] < big["flops"]             # dot cost ~ tile
    assert small["gm_bytes"] != big["gm_bytes"]      # table bytes ~ tile


def test_predict_positive_and_vendor_refused():
    _calib()
    for fam in ("baseline_m0_rank1_loop", "incumbent_m4_folded", "m4_spread",
                "m2_batch", "m5_batch", "m2_packet"):
        assert cm.predict(fam, 1024, 16) > 0
    with pytest.raises(ValueError):
        cm.predict("vendor", 1024, 16)


def test_bounded_search_returns_k_from_space():
    cal = _calib()
    tops = cm.bounded_search("m2_batch", 1024, 16, k=3, calib=cal)
    assert 1 <= len(tops) <= 3
    assert len({(p["tile"], p["rows"]) for p in tops}) == len(tops)
    for p in tops:
        assert p["tile"] in (32, 64, 128)
        assert p["rows"] in (4, 8, 16)
    # model ranking: cheapest first (deterministic)
    costs = [cm.predict("m2_batch", 1024, 16, p, calib=cal) for p in tops]
    assert costs == sorted(costs)


def test_bounded_search_rejects_other_families():
    _calib()
    with pytest.raises(ValueError):
        cm.bounded_search("m5_batch", 1024, 16)


def _records():
    """Synthetic measurements where the model agrees on every pair."""
    rows = []
    for name, mult, params in (
            ("incumbent_m4_folded", 2.0, {}),
            ("m4_spread", 2.05, {}),
            ("m2_batch", 1.0, {"tile": 64, "rows": 8}),
            ("m2_t32_rows8", 1.1, {"tile": 32, "rows": 8}),
            ("m2_t128_rows8", 1.3, {"tile": 128, "rows": 8})):
        base = cm.predict("m2_batch", 1024, 16,
                          params or {"tile": 64, "rows": 8})
        rows.append({"name": name, "N": 1024, "batch": 16,
                     "measured_ms": base * mult, "params": params})
    return rows


def test_validate_reports_holdout_metrics():
    _calib()
    m = cm.validate(_records())
    assert m["n_records"] == 5
    assert m["fitted"] is False and m["constants_declared"] is True
    assert set(m["mdape_pct"]) >= {"m2_batch", "incumbent_m4_folded"}
    assert m["guard_pairs"] > 0
    assert m["guard_agreement"] == 1.0
    assert m["topk_shapes"] >= 1
    assert 0.0 <= m["topk_hit"] <= 1.0


def test_validate_against_repo_focus_summary_if_present():
    _calib()
    path = ROOT / "results/p800/batch_release/summary.json"
    if not path.exists():
        pytest.skip("focus run not present")
    summary = json.loads(path.read_text())
    recs = []
    for sh in summary["shapes"]:
        for name, c in sh["candidates"].items():
            if isinstance(c.get("median_ms"), (int, float)):
                recs.append({"name": name, "N": sh["N"], "batch": sh["batch"],
                             "measured_ms": c["median_ms"],
                             "params": c.get("params")})
    m = cm.validate(recs)
    assert m["n_records"] > 0
    # decision-level agreement must be reported, whatever its value
    assert m["guard_agreement"] is None or 0.0 <= m["guard_agreement"] <= 1.0
