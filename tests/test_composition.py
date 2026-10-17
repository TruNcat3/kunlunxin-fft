"""Composition registry + entry integration (issue #6 PR-E, host-only).

Contract validation, fail-closed selection, exact-shape promotion
guard arithmetic, and plan_fft wiring — all without torch.
"""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.xpu.composition as comp  # noqa: E402
from src.xpu.config import ConfigError  # noqa: E402
from src.xpu.entry import plan_fft  # noqa: E402
from src.xpu.manifest_contract import rows_sha256  # noqa: E402


def _valid_table():
    cand, inc = 0.2345, 1.5064
    ev = {
        "artifact": "results/p800/composition_confirm.json",
        "artifact_sha256": "c" * 64,
        "commit": "a" * 40,
        "sessions": 4,
        "trials": 30,
        "confirmation_session": True,
        "err_max": 1e-4,
        "p50_ms": cand,
        "mad_ms": 0.0144,
        "p95_ms": 0.2663,
    }
    entry = {
        "N": 2048, "batch": 16, "direction": "dit", "precision": "fp32",
        "tile": 64, "core": "vendor",
        "flow_id": "hybrid-A1P1",
        "flow_realization": "decomposition-composition",
        "processing_unit": "vendor-subfft-host-composition",
        "candidate_ms": cand, "incumbent_ms": inc,
        "incumbent_flow": "A0P0-production",
        "guard_pct": comp.guard_pct_of(cand, inc),
        "guard_pass": True,
        "reason": "test cell",
        "contract_id": "a1p1_materialized_subtransform",
        "evidence": ev,
    }
    table = {
        "schema_version": 1,
        "kind": comp.KIND,
        "profile_hash": "p" * 16,
        "env_hash": "e" * 16,
        "precision": "fp32",
        "guard_pct": 2.0,
        "contract_id": "a1p1_materialized_subtransform",
        "created_at": "2026-10-17T00:00:00+0000",
        "source_hash": "b" * 40,
        "dirty": False,
        "entries": [entry],
    }
    table["result_sha256"] = rows_sha256({"rows": table["entries"]})
    return table


def test_absent_table_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(comp, "TABLE_PATH",
                        str(tmp_path / "missing.json"))
    assert comp.load_composition_table() is None
    assert comp.composition_select("dit", 2048, 16,
                                   profile_hash="p" * 16,
                                   env_hash="e" * 16) is None


def test_valid_table_loads_and_selects(monkeypatch):
    table = _valid_table()
    monkeypatch.setattr(comp, "load_composition_table",
                        lambda path=comp.TABLE_PATH: table)
    got = comp.composition_select("dit", 2048, 16,
                                  profile_hash="p" * 16,
                                  env_hash="e" * 16, table=table)
    assert got is not None and got["flow_id"] == "hybrid-A1P1"


def test_selection_fail_closed_matrix():
    table = _valid_table()
    assert comp.composition_select(
        "dif", 2048, 16, profile_hash="p" * 16,
        env_hash="e" * 16, table=table) is None          # DIF
    assert comp.composition_select(
        "dit", 2048, 4, profile_hash="p" * 16,
        env_hash="e" * 16, table=table) is None          # batch
    assert comp.composition_select(
        "dit", 1024, 16, profile_hash="p" * 16,
        env_hash="e" * 16, table=table) is None          # N
    assert comp.composition_select(
        "dit", 2048, 16, precision="fp64", profile_hash="p" * 16,
        env_hash="e" * 16, table=table) is None          # precision
    assert comp.composition_select(
        "dit", 2048, 16, profile_hash=None,
        env_hash="e" * 16, table=table) is None          # no profile
    assert comp.composition_select(
        "dit", 2048, 16, profile_hash="x" * 16,
        env_hash="e" * 16, table=table) is None          # wrong profile
    assert comp.composition_select(
        "dit", 2048, 16, profile_hash="p" * 16,
        env_hash="z" * 16, table=table) is None          # wrong env
    # env_hash="auto" resolves through envcheck; with an explicit
    # bound table the auto path is exercised only on device, so pin
    # it here to a mismatching explicit value instead
    assert comp.composition_select(
        "dit", 2048, 16, profile_hash="p" * 16,
        env_hash=None, table=table) is None


def test_contract_violations_catch_tampering():
    # clean table has no violations
    assert comp.table_violations(_valid_table()) == []

    # guard arithmetic mismatch (stale stamp)
    bad = _valid_table()
    bad["entries"][0]["guard_pct"] += 1.0
    assert any("guard_pct" in v for v in comp.table_violations(bad))

    # guard below the 2% threshold
    bad = _valid_table()
    bad["entries"][0]["candidate_ms"] = 1.50
    bad["entries"][0]["guard_pct"] = comp.guard_pct_of(1.50, 1.5064)
    bad["entries"][0]["evidence"]["p50_ms"] = 1.50
    bad["result_sha256"] = rows_sha256({"rows": bad["entries"]})
    assert any("guard" in v for v in comp.table_violations(bad))

    # missing confirmation session
    bad = _valid_table()
    bad["entries"][0]["evidence"]["confirmation_session"] = False
    bad["result_sha256"] = rows_sha256({"rows": bad["entries"]})
    assert any("confirmation" in v for v in comp.table_violations(bad))

    # correctness gate
    bad = _valid_table()
    bad["entries"][0]["evidence"]["err_max"] = 1e-3
    bad["result_sha256"] = rows_sha256({"rows": bad["entries"]})
    assert any("err_max" in v for v in comp.table_violations(bad))

    # unknown capability id (contract must exist in the registry)
    bad = _valid_table()
    bad["contract_id"] = "not-a-registered-core"
    bad["entries"][0]["contract_id"] = "not-a-registered-core"
    bad["result_sha256"] = rows_sha256({"rows": bad["entries"]})
    assert any("contract_id" in v for v in comp.table_violations(bad))

    # tampered entries after stamping
    bad = _valid_table()
    bad["entries"][0]["candidate_ms"] = 0.10
    bad["entries"][0]["guard_pct"] = comp.guard_pct_of(0.10, 1.5064)
    assert any("result_sha256" in v for v in comp.table_violations(bad))

    # duplicate exact keys
    bad = _valid_table()
    bad["entries"].append(copy.deepcopy(bad["entries"][0]))
    bad["result_sha256"] = rows_sha256({"rows": bad["entries"]})
    assert any("duplicate" in v for v in comp.table_violations(bad))


def test_load_raises_on_invalid_table(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps({"schema_version": 1}))
    with pytest.raises(ConfigError):
        comp.load_composition_table(str(p))


def test_plan_fft_composition_wiring(monkeypatch):
    table = _valid_table()
    monkeypatch.setattr(comp, "load_composition_table",
                        lambda path=comp.TABLE_PATH: table)
    plan = plan_fft("dit", 2048, 16, profile_hash="p" * 16,
                    env_hash="e" * 16)
    assert plan.selection_source == "composition"
    assert plan.fallback is False
    assert plan.executed == "run_hybrid"
    assert plan.params["flow_kind"] == "composition"
    assert plan.params["tile"] == 64
    assert plan.transform_type == "c2c"
    assert "guard" in plan.reason and "composition" in plan.reason
    assert plan.flow_realization == "decomposition-composition"
    # honest attribution: composition, never resident kernel
    assert "composition" in plan.processing_unit

    # inverse semantics ride the same composition plan
    inv = plan_fft("dit", 2048, 16, profile_hash="p" * 16,
                   env_hash="e" * 16, transform_direction="inverse",
                   norm="ortho")
    assert inv.selection_source == "composition"
    assert inv.transform_direction == "inverse"
    assert inv.norm == "ortho"

    # real-compat path never takes the composition plan
    r2c = plan_fft("dit", 2048, 16, profile_hash="p" * 16,
                   env_hash="e" * 16, transform_type="r2c")
    assert r2c.selection_source != "composition"

    # binding mismatch falls back to the butterfly path
    wrong = plan_fft("dit", 2048, 16, profile_hash="p" * 16,
                     env_hash="e" * 99)
    assert wrong.selection_source != "composition"
    none_profile = plan_fft("dit", 2048, 16, profile_hash=None,
                            env_hash="e" * 16)
    assert none_profile.selection_source != "composition"


def test_guard_arithmetic():
    # load-time recomputation uses exactly the stamped medians
    cand, inc = 0.2345, 1.5064
    g = comp.guard_pct_of(cand, inc)
    assert abs(g - (1.5064 - 0.2345) / 1.5064 * 100.0) < 1e-12
    assert g > 2.0
    # loser -> negative guard (must never promote)
    assert comp.guard_pct_of(1.60, 1.5064) < 0
