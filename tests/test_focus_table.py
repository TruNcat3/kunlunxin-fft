"""Focus (--focus) table entries: issue #3 selector closure + 2% guard."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.xpu.config import ConfigError, DeviceProfile  # noqa: E402
from scripts.build_selector_table import (  # noqa: E402
    _load_focus,
    build_focus_entries,
)


def _manifest(profile_hash=None):
    return {
        "device": "kunlunxin-p800",
        "profile_hash": profile_hash or DeviceProfile.load().hash(),
        "protocol_id": "batch-release-v1:focus:test",
        "error_threshold": 1e-4,
        "shapes": ["N1024B16:dit"],
        "candidates": ["m2_batch", "m5_batch"],
        "timing_scope": "api wall time",
        "sessions": 3,
        "trials": 30,
    }


def _shape(n=1024, batch=16, direction="dit", m2_ms=1.0, m4_ms=2.0,
           m2_err=1e-5, block=256, extra=None):
    cands = {
        "incumbent_m4_folded": {
            "median_ms": m4_ms, "err_max": 1e-6, "block": block,
            "params": {"variant": "folded", "batch": batch}},
        "m4_spread": {
            "median_ms": m4_ms * 1.01, "err_max": 1e-6, "block": block,
            "params": {"variant": "spread", "batch": batch}},
        "m2_batch": {
            "median_ms": m2_ms, "err_max": m2_err, "block": block,
            "params": {"variant": "spread", "tile": 64, "rows": 8,
                       "batch": batch}},
        "m5_batch": {
            "median_ms": m2_ms * 1.1, "err_max": 1e-6, "block": block,
            "params": {"variant": "spread", "batch": batch}},
        "vendor": {"median_ms": 0.2, "err_max": 1e-7, "block": None},
    }
    cands.update(extra or {})
    # complete evidence by default (3 sessions x 30 trials); negative
    # tests override sessions/n_trials per candidate.
    for c in cands.values():
        c.setdefault("sessions", 3)
        c.setdefault("n_trials", 90)
    return {"label": f"N{n}B{batch}_{direction}", "N": n, "batch": batch,
            "direction": direction, "incumbent_block": block,
            "candidates": cands}


def _focus_run(tmp_path, shapes, sessions=3, manifest=None, name="s.json"):
    data = {"manifest": manifest or _manifest(), "sessions": sessions,
            "shapes": shapes}
    import json
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return p


def test_focus_run_below_session_floor_is_refused(tmp_path):
    # 1 aggregated session vs a manifest declaring 3 is a failed run:
    # _load_focus refuses it outright (promotion needs the full set).
    p = _focus_run(tmp_path, [_shape()], sessions=1)
    with pytest.raises(ConfigError, match="session failed or was dropped"):
        _load_focus(str(p), [])


def test_focus_challenger_clears_guard_and_lands_in_table(tmp_path):
    p = _focus_run(tmp_path, [_shape(m2_ms=1.0, m4_ms=2.0)])
    run = _load_focus(str(p), [])
    entries = build_focus_entries([run], guard=2.0, min_sessions=3,
                                  rejections=[])
    assert len(entries) == 1
    e = entries[0]
    assert (e["N"], e["batch"], e["direction"]) == (1024, 16, "dit")
    assert e["flow_id"] == "local-fused-tile"
    assert e["flow_realization"] == "matrix-resident"
    assert e["processing_unit"] == "dense-dot-fft64"
    assert e["params"]["tile"] == 64 and e["params"]["rows"] == 8
    assert e["challenger_ms"] == 1.0 and e["incumbent_ms"] == 2.0
    assert "beats stable incumbent" in e["reason"]


def test_focus_guard_keeps_incumbent_when_gain_below_threshold(tmp_path):
    # 1.5% gain < 2% guard -> M4 incumbent entry, challenger recorded
    p = _focus_run(tmp_path, [_shape(m2_ms=1.97, m4_ms=2.0)])
    run = _load_focus(str(p), [])
    entries = build_focus_entries([run], guard=2.0, min_sessions=3,
                                  rejections=[])
    assert len(entries) == 1
    e = entries[0]
    assert e["flow_id"] == "batch-folded"
    assert e["flow_realization"] == "global-stage"
    assert e["processing_unit"] == "radix2-scalar"
    assert "no focus challenger clears the 2.0% guard" in e["reason"]


def test_focus_wrong_value_challenger_is_gated_out(tmp_path):
    bad_m5 = {"m5_batch": {"median_ms": 0.55, "err_max": 1e-2, "block": 256,
                           "params": {"variant": "spread", "batch": 16}}}
    p = _focus_run(tmp_path, [_shape(m2_ms=0.5, m4_ms=2.0, m2_err=1e-2,
                                     extra=bad_m5)])
    rejections = []
    run = _load_focus(str(p), rejections)
    entries = build_focus_entries([run], guard=2.0, min_sessions=3,
                                  rejections=rejections)
    assert len(entries) == 1
    assert entries[0]["flow_id"] == "batch-folded"  # incumbent kept
    assert any("max_rel_error" in r for r in rejections)


def test_focus_m2_t_variant_becomes_challenger(tmp_path):
    extra = {"m2_t32_rows8": {"median_ms": 0.6, "err_max": 1e-6, "block": 512,
                              "params": {"variant": "spread", "tile": 32,
                                         "rows": 8, "batch": 16}}}
    p = _focus_run(tmp_path, [_shape(extra=extra)])
    run = _load_focus(str(p), [])
    entries = build_focus_entries([run], guard=2.0, min_sessions=3,
                                  rejections=[])
    assert len(entries) == 1
    e = entries[0]
    assert e["challenger_ms"] == 0.6
    assert e["params"]["tile"] == 32
    assert e["block"] == 512  # entry block follows the winner
    assert e["evidence"]["challenger"] == "m2_t32_rows8"


def test_focus_loader_rejects_incomplete_manifest(tmp_path):
    import json
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"manifest": {"device": "x"}, "sessions": 3,
                             "shapes": [_shape()]}))
    with pytest.raises(ConfigError):
        _load_focus(str(p), [])


def test_focus_packet_canary_never_enters_table(tmp_path):
    # m2_packet_* is experimental (P2): not a challenger name
    extra = {"m2_packet_td2": {"median_ms": 0.4, "err_max": 1e-6,
                               "block": 256,
                               "params": {"packet": 2}}}
    p = _focus_run(tmp_path, [_shape(extra=extra)])
    run = _load_focus(str(p), [])
    entries = build_focus_entries([run], guard=2.0, min_sessions=3,
                                  rejections=[])
    assert len(entries) == 1
    assert entries[0]["challenger_ms"] == 1.0  # m2_batch, not the packet
    assert entries[0]["evidence"].get("challenger") == "m2_batch"


# --- promotion integrity (issue #3 review fix: no partial evidence) ---


def test_focus_incomplete_session_evidence_blocks_promotion(tmp_path):
    # m2_batch present in only 2 of 3 sessions -> excluded entirely;
    # m5 is complete but slower than the incumbent -> incumbent kept.
    slow_m5 = {"m5_batch": {"median_ms": 2.1, "err_max": 1e-6, "block": 256,
                            "params": {"variant": "spread", "batch": 16},
                            "sessions": 3, "n_trials": 90}}
    partial = {"m2_batch": {"median_ms": 1.0, "err_max": 1e-5, "block": 256,
                            "params": {"variant": "spread", "tile": 64,
                                       "rows": 8, "batch": 16},
                            "sessions": 2, "n_trials": 60}}
    p = _focus_run(tmp_path, [_shape(extra=dict(slow_m5, **partial))])
    run = _load_focus(str(p), [])
    rejections = []
    entries = build_focus_entries([run], guard=2.0, min_sessions=3,
                                  rejections=rejections)
    assert len(entries) == 1
    assert entries[0]["flow_id"] == "batch-folded"  # incumbent kept
    assert any("session evidence" in r for r in rejections)


def test_focus_missing_trials_blocks_promotion(tmp_path):
    short_trials = {
        "m2_batch": {"median_ms": 1.0, "err_max": 1e-5, "block": 256,
                     "params": {"variant": "spread", "tile": 64,
                                "rows": 8, "batch": 16},
                     "sessions": 3, "n_trials": 60},
        "m5_batch": {"median_ms": 2.1, "err_max": 1e-6, "block": 256,
                     "params": {"variant": "spread", "batch": 16},
                     "sessions": 3, "n_trials": 90},
    }
    p = _focus_run(tmp_path, [_shape(extra=short_trials)])
    run = _load_focus(str(p), [])
    rejections = []
    entries = build_focus_entries([run], guard=2.0, min_sessions=3,
                                  rejections=rejections)
    assert len(entries) == 1
    assert entries[0]["flow_id"] == "batch-folded"
    assert any("trial evidence" in r for r in rejections)


def test_focus_recorded_failure_blocks_promotion(tmp_path):
    # extras_failed in any session disqualifies that candidate outright.
    slow_m5 = {"m5_batch": {"median_ms": 2.1, "err_max": 1e-6, "block": 256,
                            "params": {"variant": "spread", "batch": 16},
                            "sessions": 3, "n_trials": 90}}
    p = _focus_run(tmp_path, [_shape(extra=slow_m5)])
    import json
    data = json.loads(p.read_text())
    data["shapes"][0]["extras_failed"] = [
        {"candidate": "m2_batch", "status": "wrong-value",
         "max_rel_error": 0.5}]
    p.write_text(json.dumps(data))
    run = _load_focus(str(p), [])
    rejections = []
    entries = build_focus_entries([run], guard=2.0, min_sessions=3,
                                  rejections=rejections)
    assert len(entries) == 1
    assert entries[0]["flow_id"] == "batch-folded"
    assert any("recorded failure" in r for r in rejections)


def test_focus_incomplete_incumbent_refuses_entry(tmp_path):
    bad_inc = {"incumbent_m4_folded": {"median_ms": 2.0, "err_max": 1e-6,
                                       "block": 256,
                                       "params": {"variant": "folded",
                                                  "batch": 16},
                                       "sessions": 2, "n_trials": 60}}
    p = _focus_run(tmp_path, [_shape(extra=bad_inc)])
    run = _load_focus(str(p), [])
    rejections = []
    entries = build_focus_entries([run], guard=2.0, min_sessions=3,
                                  rejections=rejections)
    assert entries == []
    assert any("incomplete incumbent control" in r for r in rejections)


def test_focus_loader_refuses_failed_session_run(tmp_path):
    # summary aggregated 2 sessions while the manifest declares 3.
    import json
    p = _focus_run(tmp_path, [_shape()], sessions=2)
    with pytest.raises(ConfigError, match="re-run the focus protocol"):
        _load_focus(str(p), [])
