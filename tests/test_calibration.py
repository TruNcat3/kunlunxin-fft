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
    # explicit offline inspection: the self-hash check must not depend
    # on this host being the P800 the profile was probed on (review P0).
    profile = load(verify_env=False)
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


def test_env_mismatch_refuses_reuse(tmp_path):
    # A profile bound to a different environment must not load (freshness).
    from src.xpu.envcheck import collect, env_hash

    live = collect()
    if live is None:
        pytest.skip("live environment not probeable on this host")
    profile = _profile()
    profile["env"] = dict(live, driver="0.0-fake")
    profile["env_hash"] = env_hash(profile["env"])
    profile["profile_hash"] = compute_hash(profile)
    f = tmp_path / "calibration.json"
    f.write_text(json.dumps(profile))
    with pytest.raises(CalibrationMismatch, match="environment"):
        load(f)


def test_missing_env_fingerprint_refuses_reuse(tmp_path):
    # Pre-binding profiles (self-hash only) fail closed with guidance.
    profile = _profile()
    profile["profile_hash"] = compute_hash(profile)
    f = tmp_path / "calibration.json"
    f.write_text(json.dumps(profile))
    with pytest.raises(CalibrationMismatch, match="environment fingerprint"):
        load(f)


def test_matching_env_fingerprint_loads(tmp_path):
    from src.xpu.envcheck import collect, env_hash

    live = collect()
    if live is None:
        pytest.skip("live environment not probeable on this host")
    profile = _profile()
    profile["env"] = live
    profile["env_hash"] = env_hash(live)
    profile["profile_hash"] = compute_hash(profile)
    f = tmp_path / "calibration.json"
    f.write_text(json.dumps(profile))
    assert load(f)["env_hash"] == profile["env_hash"]
    # explicit opt-out only (offline inspection)
    assert load(f, verify_env=False)["schema_version"] == 1


def test_repo_calibration_verifies_live_environment():
    p = ROOT / "config" / "calibration.json"
    if not p.exists():
        pytest.skip("hw_probe has not run on this machine")
    from src.xpu.envcheck import collect
    if collect() is None:
        pytest.skip("live environment not probeable on this host")
    profile = load(verify_env=True)  # must match THIS host
    assert profile.get("env_hash")
