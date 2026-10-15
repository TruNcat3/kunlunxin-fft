"""Calibration profile verification (issue #3 P1.2)."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.xpu.calibration import (  # noqa: E402
    CalibrationMismatch,
    compute_hash,
    load,
)


def _profile():
    return {"schema_version": 1, "kind": "p800-calibration",
            "baseline_costs": {"triton_launch_ms": 0.03}}


def test_repo_calibration_loads_when_present():
    p = ROOT / "config" / "calibration.json"
    if not p.exists():
        pytest.skip("hw_probe has not run on this machine")
    profile = load()
    assert profile["profile_hash"] == compute_hash(profile)


def test_hash_mismatch_raises(tmp_path):
    profile = _profile()
    profile["profile_hash"] = compute_hash(profile)
    profile["baseline_costs"]["triton_launch_ms"] = 0.0  # tampered
    f = tmp_path / "calibration.json"
    f.write_text(json.dumps(profile))
    with pytest.raises(CalibrationMismatch):
        load(f)


def test_missing_hash_raises(tmp_path):
    f = tmp_path / "calibration.json"
    f.write_text(json.dumps(_profile()))
    with pytest.raises(CalibrationMismatch):
        load(f)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "nope.json")
