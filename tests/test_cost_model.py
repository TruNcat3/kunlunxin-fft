"""Issue #3 P1.3: structured cost model, bounded search, holdout validation."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.xpu import cost_model as cm  # noqa: E402


def _calib():
    """Explicitly offline calibration read (no live-env verification):
    validation tests must not depend on the test host being the P800 the
    profile was probed on (issue #3 review: host CI without torch)."""
    try:
        from src.xpu.calibration import load
        return load(verify_env=False)
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
    cal = _calib()
    for fam in ("baseline_m0_rank1_loop", "incumbent_m4_folded", "m4_spread",
                "m2_batch", "m5_batch", "m2_packet"):
        assert cm.predict(fam, 1024, 16, calib=cal) > 0
    with pytest.raises(ValueError):
        cm.predict("vendor", 1024, 16, calib=cal)


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
    cal = _calib()
    with pytest.raises(ValueError):
        cm.bounded_search("m5_batch", 1024, 16, calib=cal)


def _records(cal=None):
    """Synthetic measurements where the model agrees on every pair."""
    cal = cal if cal is not None else _calib()
    rows = []
    for name, mult, params in (
            ("incumbent_m4_folded", 2.0, {}),
            ("m4_spread", 2.05, {}),
            ("m2_batch", 1.0, {"tile": 64, "rows": 8}),
            ("m2_t32_rows8", 1.1, {"tile": 32, "rows": 8}),
            ("m2_t128_rows8", 1.3, {"tile": 128, "rows": 8})):
        base = cm.predict("m2_batch", 1024, 16,
                          params or {"tile": 64, "rows": 8}, calib=cal)
        rows.append({"name": name, "N": 1024, "batch": 16,
                     "measured_ms": base * mult, "params": params})
    return rows


def test_validate_reports_focus_guard_and_holdout_topk():
    cal = _calib()
    # focus-role rows (full_space false): guard only, no holdout top-K
    m = cm.validate(_records(cal), calib=cal)
    assert m["n_records"] == 5 and m["n_focus_records"] == 5
    assert m["n_holdout_records"] == 0
    assert m["fitted"] is False and m["constants_declared"] is True
    assert set(m["mdape_pct"]) >= {"m2_batch", "incumbent_m4_folded"}
    g = m["guard_by_source"]["ad-hoc"]
    assert g["pairs"] > 0 and g["agreement"] == 1.0
    assert m["topk_shapes"] == 0 and m["topk_hit"] is None
    # holdout-role rows: selection metrics only, guard not pooled
    holdout = [{**r, "full_space": True} for r in _records(cal)]
    m2 = cm.validate(holdout, calib=cal)
    assert m2["n_holdout_records"] == 5
    assert m2["guard_by_source"] == {}
    assert m2["topk_shapes"] >= 1
    assert 0.0 <= m2["topk_hit"] <= 1.0


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
    m = cm.validate(recs, calib=_calib())
    assert m["n_records"] > 0
    # decision-level agreement must be reported, whatever its value
    g = m["guard_by_source"].get("ad-hoc") or {}
    assert g.get("agreement") is None or 0.0 <= g["agreement"] <= 1.0


def test_validate_guard_uses_two_percent_line_not_sign():
    # A 1.5% measured gain does NOT clear the 2% promotion guard while the
    # model predicts a clear win: sign-of-gain would call this agreement,
    # the formal guard line must call it a disagreement.
    cal = _calib()
    pred_inc = cm.predict("incumbent_m4_folded", 1024, 16, calib=cal)
    pred_m2 = cm.predict("m2_batch", 1024, 16, {"tile": 128, "rows": 8},
                         calib=cal)
    assert (pred_inc - pred_m2) / pred_inc * 100.0 >= cm.GUARD_PCT  # model: promote
    recs = [
        {"name": "incumbent_m4_folded", "family": "incumbent_m4_folded",
         "N": 1024, "batch": 16, "direction": "dit",
         "measured_ms": 100.0, "params": {}},
        {"name": "m2_batch", "family": "m2_batch", "N": 1024, "batch": 16,
         "direction": "dit", "measured_ms": 98.5,  # 1.5% < 2% guard
         "params": {"tile": 128, "rows": 8}},
    ]
    m = cm.validate(recs, calib=_calib())
    assert m["guard_pct"] == cm.GUARD_PCT
    g = m["guard_by_source"]["ad-hoc"]
    assert g["pairs"] == 1
    assert g["agreement"] == 0.0


def test_validate_full_space_recall_and_regret():
    cal = _calib()
    rows = [{"name": "incumbent_m4_folded", "family": "incumbent_m4_folded",
             "N": 1024, "batch": 16, "direction": "dit",
             "measured_ms": 5.0, "params": {}, "full_space": True}]
    for s in cm.m2_space():
        pred = cm.predict("m2_batch", 1024, 16, s, calib=cal)
        rows.append({"name": f"m2_t{s['tile']}_rows{s['rows']}",
                     "family": "m2_batch", "N": 1024, "batch": 16,
                     "direction": "dit", "measured_ms": pred,
                     "params": dict(s), "full_space": True})
    m = cm.validate(rows, calib=cal)
    fs = m["topk_full_space"]
    assert fs["shapes"] == 1
    assert fs["space_size"] == 9
    assert fs["recall"] == 1.0          # model-perfect measurements
    assert fs["mean_regret_pct"] == 0.0
    assert m["topk_shapes"] >= 1        # subset metric still reported


def _summary(candidates, sessions=3, trials=30):
    return {"manifest": {"error_threshold": 1e-4, "sessions": sessions,
                         "trials": trials, "protocol_id": "test"},
            "shapes": [{"N": 1024, "batch": 16, "direction": "dit",
                        "label": "N1024B16_dit",
                        "candidates": candidates}]}


def _ok_cand(**over):
    c = {"median_ms": 1.0, "err_max": 1e-6, "complete": True,
         "sessions": 3, "n_trials": 90,
         "params": {"tile": 64, "rows": 8}}
    c.update(over)
    return c


def test_records_from_summary_skips_bad_rows():
    summary = _summary({
        "ok": _ok_cand(),
        "wrong_value": _ok_cand(median_ms=0.5, err_max=0.1),
        "incomplete": _ok_cand(complete=False),
        "nan": _ok_cand(median_ms=float("nan")),
        "short_sessions": _ok_cand(sessions=1, n_trials=30),
        "no_complete_flag": {"median_ms": 1.0, "err_max": 1e-6,
                             "sessions": 3, "n_trials": 90},
    })
    recs = cm.records_from_summary(summary, source_id="synthetic")
    assert [r["name"] for r in recs] == ["ok"]
    assert recs[0]["full_space"] is False
    assert recs[0]["protocol_id"] == "test"


def test_builder_and_records_share_one_completeness_verdict():
    # Reviewer negative: 1/3 session + missing complete flag + wrong-value
    # timing must be rejected by BOTH consumers -- one shared helper.
    from src.xpu.evidence import focus_candidate_verdict

    bad = {"median_ms": "fast?", "err_max": 1e-6,
           "sessions": 1, "n_trials": 30}
    shape = {"label": "N1024B16_dit", "N": 1024, "batch": 16,
             "direction": "dit", "candidates": {"m2_batch": dict(bad)},
             "extras_failed": []}
    ms, why = focus_candidate_verdict(
        shape, "m2_batch", threshold=1e-4, expected_sessions=3,
        trials=30, label="synthetic")
    assert ms is None and "complete" in why
    # same candidate with the flag present still fails on session count
    shape["candidates"]["m2_batch"] = dict(bad, complete=True, median_ms=1.0)
    ms, why = focus_candidate_verdict(
        shape, "m2_batch", threshold=1e-4, expected_sessions=3,
        trials=30, label="synthetic")
    assert ms is None and "session evidence" in why
    # and the records path applies the same verdict
    assert cm.records_from_summary(_summary({
        "m2_batch": dict(bad, complete=True, median_ms=1.0)})) == []


def test_validate_holdout_metrics_invariant_to_inputs():
    cal = _calib()
    focus = ROOT / "results/p800/batch_focus_a/summary.json"
    holdout = ROOT / "results/p800/batch_holdout/summary.json"
    if not (focus.exists() and holdout.exists()):
        pytest.skip("repo focus/holdout artifacts not present")

    def recs(paths):
        out = []
        for p in paths:
            s = json.loads(p.read_text())
            out.extend(cm.records_from_summary(s, source_id=str(p)))
        return out

    base = cm.validate(recs([holdout]), calib=cal)
    more = cm.validate(recs([focus, holdout, focus]), calib=cal)
    flipped = cm.validate(recs([holdout, focus]), calib=cal)
    for m in (base, more, flipped):
        assert m["topk_full_space"] == base["topk_full_space"]
        assert m["topk_hit"] == base["topk_hit"]
        assert m["n_holdout_records"] == base["n_holdout_records"]
    # holdout-only is the correct single-source result (review: 6/6)
    assert base["topk_full_space"]["recall"] == 1.0
    assert base["topk_full_space"]["shapes"] >= 1
    # guard stays per focus round; duplicated focus inputs share a bucket
    assert set(more["guard_by_source"]) == {str(focus)}
