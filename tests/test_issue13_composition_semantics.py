"""issue #13 WP2: per-entry semantic composition + hybrid4 dispatch.

Host-only (no torch): composition_select per-entry semantic matching,
variant/executed contract, semantic-key uniqueness, plan_fft wiring
onto the 4-step entries, and hybrid4 plan validation/attribution.
Hardware coverage (runner output parity vs the measured benchmark
implementations, rank-1 plan->execute both directions) lives in
tests/test_issue13_hybrid4_hw.py.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.xpu.composition as comp  # noqa: E402
from src.xpu.entry import plan_fft  # noqa: E402
from src.xpu.entry import _H4_EXECUTED, norm_scale  # noqa: E402
from src.xpu.hybrid4 import (  # noqa: E402
    EXECUTED_VARIANT, VARIANTS, VARIANT_EXECUTED, factors, plan_hybrid4,
    plan_ledger,
)
from src.xpu.manifest_contract import rows_sha256  # noqa: E402


def _ev(p50=0.1234):
    return {
        "artifact": "results/p800/issue13_release/wp1_promotion.json",
        "artifact_sha256": "c" * 64, "commit": "a" * 40,
        "sessions": 4, "trials": 30, "confirmation_session": True,
        "err_max": 1e-5, "p50_ms": p50, "mad_ms": 0.004,
        "p95_ms": p50 * 1.1,
    }


def _entry(**over):
    cand = float(over.pop("candidate_ms", 0.1))
    inc = float(over.pop("incumbent_ms", 1.5))
    e = {
        "N": 2048, "batch": 16, "direction": "dit", "precision": "fp32",
        "tile": 64, "core": "vendor",
        "flow_id": "hybrid-A1P1-4step-fused",
        "flow_realization": "decomposition-composition",
        "processing_unit": "vendor-subfft-host-composition",
        "candidate_ms": cand, "incumbent_ms": inc,
        "incumbent_flow": "composition-a1p1-adapter",
        "guard_pct": comp.guard_pct_of(cand, inc),
        "guard_pass": True, "reason": "test cell",
        "contract_id": "a1p1_materialized_subtransform",
        "variant": "4step-fused", "executed": "run_hybrid4_fused",
        "transform_type": "c2c", "transform_direction": "forward",
        "normalization": "backward",
        "entry_contract": "rank-1 (N,) plan_fft->execute_fft_plan",
        "evidence": _ev(),
    }
    e.update(over)
    return e


def _table(entries):
    t = {
        "schema_version": 1, "kind": comp.KIND, "profile_hash": "p" * 16,
        "env_hash": "e" * 16, "precision": "fp32", "guard_pct": 2.0,
        "contract_id": "a1p1_materialized_subtransform",
        "source_hash": "b" * 40, "dirty": False, "entries": entries,
    }
    t["result_sha256"] = rows_sha256({"rows": entries})
    return t


def _select(table, **kw):
    args = dict(profile_hash="p" * 16, env_hash="e" * 16, table=table,
                transform_type="c2c", transform_direction="forward",
                normalization="backward")
    args.update(kw)
    return comp.composition_select("dit", 2048, 16, **args)


# ---------------------------------------------------------------- select

def test_per_entry_semantic_dispatch_both_directions():
    fwd = _entry(variant="4step-transpose", executed="run_hybrid4",
                 transform_direction="forward")
    inv = _entry(variant="4step-fused", executed="run_hybrid4_fused",
                 transform_direction="inverse")
    t = _table([inv, fwd])   # order must not matter
    assert _select(t)["executed"] == "run_hybrid4"
    assert _select(t, transform_direction="inverse")["executed"] == \
        "run_hybrid4_fused"


def test_same_shape_different_semantics_coexist_table_valid():
    fwd = _entry(variant="4step-transpose", executed="run_hybrid4")
    inv = _entry(transform_direction="inverse")
    t = _table([fwd, inv])
    assert comp.table_violations(t) == []


def test_unmatched_semantics_fail_closed():
    t = _table([_entry()])
    assert _select(t, transform_direction="inverse") is None
    assert _select(t, normalization="ortho") is None
    assert _select(t, transform_type="r2c") is None


def test_legacy_entry_defaults_answer_only_default_triple():
    legacy = _entry()
    for key in ("variant", "executed", "transform_type",
                "transform_direction", "normalization", "entry_contract"):
        legacy.pop(key)
    assert comp.entry_semantic(legacy) == ("c2c", "forward", "backward")
    assert comp.entry_executed(legacy) == "run_hybrid"
    t = _table([legacy])
    assert _select(t) is not None
    assert _select(t, transform_direction="inverse") is None


def test_duplicate_key_is_shape_plus_semantic():
    fwd = _entry()
    inv = _entry(transform_direction="inverse")
    dup = _entry(executed="run_hybrid4", variant="4step-transpose")
    for entries, expect_bad in (
            ([fwd, inv], False), ([fwd, dup], True)):
        t = _table([dict(e) for e in entries])
        bad = comp.table_violations(t)
        assert any("duplicate" in b for b in bad) is expect_bad, bad


# ------------------------------------------------- entry contract checks

def test_variant_executed_pairing_enforced():
    # matched pair (4step-fused <-> run_hybrid4_fused) is clean
    assert comp.composition_entry_violations(_entry(), _table([])) == []
    bad = _entry(variant="4step-fused", executed="run_hybrid4")
    assert any("executed" in v
               for v in comp.composition_entry_violations(bad, _table([])))
    bad3 = _entry(variant="nope")
    assert any("variant" in v for v in
               comp.composition_entry_violations(bad3, _table([])))


def test_executed_without_variant_must_be_legacy_adapter():
    e = _entry(executed="run_hybrid4_fused")
    e.pop("variant")
    assert any("legacy adapter" in v for v in
               comp.composition_entry_violations(e, _table([])))


def test_semantic_taxonomy_violations_caught():
    e = _entry(transform_direction="upside-down")
    assert any("transform_direction" in v for v in
               comp.composition_entry_violations(e, _table([])))
    e2 = _entry(normalization="unitary")
    assert any("normalization" in v for v in
               comp.composition_entry_violations(e2, _table([])))


def test_entry_contract_record_must_be_string():
    e = _entry(entry_contract=["not", "a", "string"])
    assert any("entry_contract" in v for v in
               comp.composition_entry_violations(e, _table([])))


def test_valid_hybrid4_entry_has_no_violations():
    assert comp.composition_entry_violations(
        _entry(), _table([])) == []


# ---------------------------------------------------------- plan wiring

def test_plan_fft_wires_hybrid4_entry(monkeypatch):
    fwd = _entry(variant="4step-transpose", executed="run_hybrid4")
    inv = _entry(transform_direction="inverse")
    t = _table([fwd, inv])
    monkeypatch.setattr(comp, "load_composition_table",
                        lambda path=comp.TABLE_PATH: t)
    plan = plan_fft("dit", 2048, 16, profile_hash="p" * 16,
                    env_hash="e" * 16)
    assert plan.selection_source == "composition"
    assert plan.executed == "run_hybrid4"
    assert plan.params["variant"] == "4step-transpose"
    assert plan.params["flow_kind"] == "composition"

    ip = plan_fft("dit", 2048, 16, profile_hash="p" * 16,
                  env_hash="e" * 16, transform_direction="inverse",
                  norm="backward")
    assert ip.selection_source == "composition"
    assert ip.executed == "run_hybrid4_fused"
    assert ip.transform_direction == "inverse"

    # unmatched semantics still fail closed to the butterfly path
    op = plan_fft("dit", 2048, 16, profile_hash="p" * 16,
                  env_hash="e" * 16, norm="ortho")
    assert op.selection_source != "composition"


def test_h4_executed_set_matches_module_mapping():
    assert _H4_EXECUTED == frozenset(EXECUTED_VARIANT)
    assert _H4_EXECUTED == {"run_hybrid4", "run_hybrid4_fused"}


# ------------------------------------------------------- hybrid4 plans

def test_factors_match_measured_geometry():
    assert factors(1024) == (32, 32)
    assert factors(2048) == (32, 64)
    assert factors(4096) == (64, 64)
    with pytest.raises(ValueError):
        factors(16)                        # no legal t, r >= 8 pair


def test_plan_hybrid4_validation():
    # direction lock: EXECUTED_VARIANT is executed -> variant (the
    # lookup entry.py uses), VARIANT_EXECUTED is variant -> executed.
    assert EXECUTED_VARIANT["run_hybrid4"] == "4step-transpose"
    assert EXECUTED_VARIANT["run_hybrid4_fused"] == "4step-fused"
    for variant in VARIANTS:
        p = plan_hybrid4(2048, 16, variant=variant)
        assert p.t * p.r == 2048 and p.variant == variant
        assert plan_ledger(p)["executed"] == VARIANT_EXECUTED[variant]
    with pytest.raises(ValueError):
        plan_hybrid4(2048, 16, variant="4step-naive")
    with pytest.raises(ValueError):
        plan_hybrid4(1000, 16)              # not power of two
    with pytest.raises(ValueError):
        plan_hybrid4(2048, 16, direction="dif")
    with pytest.raises(ValueError):
        plan_hybrid4(2048, 0)


def test_native_inverse_epilogue_math():
    # hybrid4 runner output is ifft-scaled for inverse; the epilogue
    # factor must be norm_scale * N so backward == exactly 1.0.
    n = 2048
    for norm, want in (("none", float(n)), ("forward", float(n)),
                       ("backward", 1.0), ("ortho", n ** 0.5)):
        assert norm_scale(n, "inverse", norm) * n == pytest.approx(want)
    # forward: runner is unscaled fft -> plain norm_scale
    assert norm_scale(n, "forward", "backward") == 1.0
    assert norm_scale(n, "forward", "forward") == pytest.approx(1.0 / n)
