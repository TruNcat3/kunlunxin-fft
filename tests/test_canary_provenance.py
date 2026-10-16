"""Regression tests: canary merge must reject mixed worker provenance.

Review follow-up P1 (issue #3, 2026-10-16): the parent used to record
``all_sessions_same_commit`` as a post-hoc diagnostic while still
merging every session into the pooled statistics. The strict gate is
now a pure function that must abort on any provenance-contract
mismatch before a publishable artifact can be written.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_hybrid_canary as rhc  # noqa: E402


def _prov(**overrides):
    base = {
        "commit": "a" * 40,
        "dirty": False,
        "table_env_hash": "6b50bd316032c0fe",
        "live_env_hash": "6b50bd316032c0fe",
        "profile_hash": "a78b86793c7142a0",
        "device": {"driver": "515.58", "device_model": 3,
                   "multiprocessor_count": 12, "max_shared_mem": 262144},
        "runtime": {"torch": "2.9.0", "triton": "3.0.0",
                    "xpytorch": "dev", "xre": "5.13", "xhpc": None},
        "pid": 1000,
        "argv": ["x"],
    }
    base.update(overrides)
    return base


def test_identical_provenance_passes():
    ref = _prov()
    rhc._validate_worker_provenance(ref, _prov(pid=2000), where="w")


@pytest.mark.parametrize("field,mutate", [
    ("commit", lambda p: dict(p, commit="b" * 40)),
    ("dirty", lambda p: dict(p, dirty=True)),
    ("table_env_hash", lambda p: dict(p, table_env_hash="0" * 16)),
    ("live_env_hash", lambda p: dict(p, live_env_hash="1" * 16)),
    ("profile_hash", lambda p: dict(p, profile_hash="f" * 16)),
    ("device", lambda p: dict(p, device=dict(p["device"],
                                             driver="999.0"))),
    ("runtime", lambda p: dict(p, runtime=dict(p["runtime"],
                                               torch="0.0.1"))),
])
def test_mismatched_contract_aborts(field, mutate):
    ref = _prov()
    bad = mutate(ref)
    with pytest.raises(RuntimeError, match=field):
        rhc._validate_worker_provenance(ref, bad, where="session 0")


def test_missing_field_aborts():
    ref = _prov()
    bad = _prov()
    del bad["profile_hash"]
    with pytest.raises(RuntimeError, match="profile_hash"):
        rhc._validate_worker_provenance(ref, bad)


def test_contract_covers_all_required_fields():
    # The gate must compare every field the review listed — commit,
    # dirty flag, table/live env hash, profile hash, device, runtime.
    assert set(rhc._PROVENANCE_CONTRACT) == {
        "commit", "dirty", "table_env_hash", "live_env_hash",
        "profile_hash", "device", "runtime",
    }


def test_pid_and_argv_are_not_part_of_the_contract():
    # Workers legitimately differ in pid (independent processes) and
    # may carry per-job argv; neither may trigger the gate.
    ref = _prov()
    rhc._validate_worker_provenance(
        ref, _prov(pid=99999, argv=["different"]), where="w")
