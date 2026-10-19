"""Issue #8 PR-G: exact semantic selector key (host-only).

Tier rules: an entry carrying ``transform_type/transform_direction/
normalization`` answers ONLY its own triple; legacy entries (absent
triple) remain the deployed old path for every semantic until that
semantic's own entry clears its own guard. Inverse/R2C/C2R never
inherit a forward ordering, and the taxonomy is single-sourced in
``src/xpu/config.py``.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.xpu.config import (  # noqa: E402
    NORMALIZATIONS,
    TRANSFORM_DIRECTIONS,
    TRANSFORM_TYPES,
    ConfigError,
)
from src.xpu.entry import (  # noqa: E402
    SUPPORTED_NORMS,
    SUPPORTED_TRANSFORM_DIRECTIONS,
    SUPPORTED_TRANSFORMS,
    _entry_block,
    plan_fft,
)
from src.xpu.semantic import (  # noqa: E402
    CORE_FLOWS,
    SPECIALIZED_FLOWS,
    family,
    family_allows,
    family_size,
    spec,
    validate_taxonomy,
)
from src.xpu.selector import (  # noqa: E402
    entry_contract_violations,
    find_entry,
    semantic_key_of,
    select,
)

ENV = "e" * 16
PROFILE = "f" * 16


def _base_entry(**over):
    entry = {
        "N": 1024, "batch": 256, "direction": "dit",
        "incumbent_ms": 2.20, "challenger_ms": 2.03,
        "incumbent_block": 1024, "block": 1024,
        "flow_id": "batch-spread", "flow_realization": "global-stage",
        "processing_unit": "radix2-scalar",
        "params": {"variant": "spread", "batch": 256},
        "evidence": {"incumbent": "matrix-incumbent", "sessions": 3},
        "reason": "fixture",
    }
    entry.update(over)
    return entry


def _semantic_entry(tt, tdir, norm="backward", **over):
    return _base_entry(
        transform_type=tt, transform_direction=tdir, normalization=norm,
        **over)


def _table(entries):
    return {
        "schema_version": 2,
        "device": "kunlunxin-p800",
        "profile_hash": PROFILE,
        "env_hash": ENV,
        "precision": "fp32",
        "guard_pct": 2.0,
        "selection_semantics": "fixture",
        "sources": ["fixture"],
        "rejections": [],
        "entries": entries,
    }


def _select(table, **over):
    kwargs = dict(direction="dit", n=1024, batch=256,
                  precision="fp32", profile_hash=PROFILE,
                  table=table, env_hash=ENV)
    kwargs.update(over)
    return select(**kwargs)


# --- taxonomy coherence -------------------------------------------------

def test_entry_module_aliases_are_the_config_taxonomy():
    assert SUPPORTED_TRANSFORMS == TRANSFORM_TYPES
    assert SUPPORTED_TRANSFORM_DIRECTIONS == TRANSFORM_DIRECTIONS
    assert SUPPORTED_NORMS == NORMALIZATIONS


def test_semantic_specs_are_in_taxonomy():
    assert validate_taxonomy() == []
    for label in ("c2c-forward", "c2c-inverse", "r2c", "c2r"):
        sp = spec(label)
        assert sp.transform_type in TRANSFORM_TYPES
        assert sp.transform_direction in TRANSFORM_DIRECTIONS
        assert sp.normalization in NORMALIZATIONS


def test_unknown_semantic_label_fails_closed():
    with pytest.raises(KeyError):
        spec("c2c-sideways")


def test_real_transform_family_excludes_unknown_specialized_flows():
    # PR-H hook: before specialized flows exist, r2c/c2r accept the
    # measured core only -- and a dedicated name must not leak in.
    assert family("r2c") == CORE_FLOWS | SPECIALIZED_FLOWS["r2c"]
    assert not family_allows("r2c", "dedicated-hermitian-pack-v1")
    assert family_allows("r2c", "batch-spread")
    assert family_allows("c2c", "local-fused-tile")
    assert family_size("c2c") <= 24


# --- tier rules ---------------------------------------------------------

def test_semantic_entry_answers_only_its_own_triple():
    table = _table([_semantic_entry("r2c", "forward")])
    hit = find_entry(table, 1024, 256, "dit", "r2c", "forward", "backward")
    assert hit is not None
    assert semantic_key_of(hit) == ("r2c", "forward", "backward")
    # the same shape under c2c/forward must NOT see the r2c entry
    assert find_entry(table, 1024, 256, "dit", "c2c", "forward",
                      "backward") is None
    # nor must inverse c2c
    assert find_entry(table, 1024, 256, "dit", "c2c", "inverse",
                      "backward") is None


def test_legacy_entry_is_the_fallback_for_every_semantic():
    legacy = _base_entry()
    table = _table([legacy])
    for tt, tdir in (("c2c", "forward"), ("c2c", "inverse"),
                     ("r2c", "forward"), ("c2r", "inverse")):
        hit = find_entry(table, 1024, 256, "dit", tt, tdir, "backward")
        assert hit is legacy, (tt, tdir)
    assert semantic_key_of(legacy) is None


def test_semantic_tier_wins_over_legacy_on_the_same_shape():
    legacy = _base_entry()
    exact = _semantic_entry("r2c", "forward",
                            params={"variant": "spread", "batch": 256,
                                    "tile": 64},
                            block=512, incumbent_block=512)
    table = _table([legacy, exact])
    hit = find_entry(table, 1024, 256, "dit", "r2c", "forward", "backward")
    assert hit is exact
    # every other semantic still sees the legacy entry
    assert find_entry(table, 1024, 256, "dit", "c2c", "forward",
                      "backward") is legacy


def test_select_tier_passes_semantic_through():
    exact = _semantic_entry("c2c", "inverse",
                            params={"variant": "folded", "batch": 256},
                            flow_id="batch-folded")
    table = _table([exact])
    s = _select(table, transform_type="c2c",
                transform_direction="inverse", normalization="backward")
    assert s.source == "table"
    assert s.transform_type == "c2c"
    assert s.transform_direction == "inverse"
    assert s.normalization == "backward"
    assert s.params == exact["params"]
    assert s.block == exact["block"]
    # a forward query on the same shape must abstain (no legacy tier)
    miss = _select(table)
    assert miss.source == "abstain"
    assert miss.transform_type == "c2c"
    assert miss.transform_direction == "forward"


def test_select_rejects_illegal_semantics():
    table = _table([_base_entry()])
    s = _select(table, transform_type="g-toeplitz")
    assert s.source == "abstain"
    assert "transform_type" in s.reason


def test_entry_contract_accepts_semantic_triples():
    for tt, tdir in (("c2c", "forward"), ("c2c", "inverse"),
                     ("r2c", "forward"), ("c2r", "inverse")):
        assert entry_contract_violations(
            _semantic_entry(tt, tdir)) == []
    bad = _semantic_entry("c2c", "forward", "swapped")
    assert entry_contract_violations(bad)


# --- plan_fft / _entry_block integration --------------------------------

def test_entry_block_resolves_through_the_semantic_tier():
    exact = _semantic_entry("r2c", "forward", block=512,
                            incumbent_block=512)
    table = _table([_base_entry(), exact])
    sel_r2c = _select(table, transform_type="r2c",
                      transform_direction="forward")
    assert sel_r2c.source == "table"
    assert _entry_block(table, sel_r2c) == 512
    sel_c2c = _select(table)
    assert sel_c2c.source == "table"
    assert sel_c2c.params == {"variant": "spread", "batch": 256}
    assert _entry_block(table, sel_c2c) == 1024


def test_plan_fft_records_the_queried_semantics():
    table = _table([_semantic_entry("c2c", "inverse")])
    p = plan_fft("dit", 1024, batch=256, profile_hash=PROFILE,
                 table=table, env_hash=ENV,
                 transform_type="c2c", transform_direction="inverse",
                 norm="backward")
    assert p.transform_type == "c2c"
    assert p.transform_direction == "inverse"
    assert p.fallback is False
    assert p.params == {"variant": "spread", "batch": 256}


# --- runner closure (issue #8 P1-A) --------------------------------------

def _matrix_semantic_entry(executed, **over):
    """A semantic entry measured on the matrix-resident family."""
    return _semantic_entry(
        "c2c", "forward",
        flow_id="local-fused-tile", flow_realization="matrix-resident",
        processing_unit="dense-dot-fft64",
        params={"tile": 64, "rows": 8, "batch": 256, "variant": "spread"},
        executed=executed, **over)


def test_executed_contract_allows_orientation_flip_within_family():
    # a DIF winner inside the dit-locked matrix is legal: the measured
    # runner is authoritative, only family + batch tier are contracted
    e = _matrix_semantic_entry("fft_dif_matrix_batch")
    assert entry_contract_violations(e) == []


def test_executed_contract_rejects_illegal_runners():
    for bad in ("fft_dit_matrix",    # non-batch runner at batch=256
                "fft_dit_fused",     # runner of a different family
                "not_a_runner"):
        e = _matrix_semantic_entry(bad)
        assert any("executed" in v
                   for v in entry_contract_violations(e)), bad


def test_executed_contract_absent_field_is_legal_for_legacy_entries():
    # legacy entries carry no executed field: the contract only fires
    # when the field is present
    assert entry_contract_violations(_base_entry()) == []


def test_select_populates_executed_and_plan_fft_deploys_it():
    entry = _matrix_semantic_entry("fft_dif_matrix_batch")
    table = _table([entry])
    s = _select(table)
    assert s.source == "table"
    assert s.executed == "fft_dif_matrix_batch"
    p = plan_fft("dit", 1024, batch=256, profile_hash=PROFILE,
                 table=table, env_hash=ENV)
    # deploys the measured DIF runner, NOT the dit-locked name that
    # _dispatch would derive from direction
    assert p.executed == "fft_dif_matrix_batch"
    assert p.fallback is False


def test_legacy_entry_dispatches_by_direction():
    table = _table([_base_entry()])
    s = _select(table)
    assert s.source == "table"
    assert s.executed is None
    p = plan_fft("dit", 1024, batch=256, profile_hash=PROFILE,
                 table=table, env_hash=ENV)
    assert p.executed == "fft_dit_batch"


# --- previous-deployed per-cell fallback (issue #8 Option A) ------------

def _fallback_entry(tdir="forward", **over):
    """A declared previous_fallbacks cell: the old deployed geometry
    claiming one exact triple (as load_table requires)."""
    entry = _base_entry(
        transform_type="c2c", transform_direction=tdir,
        normalization="backward",
        params={"variant": "spread", "batch": 256, "tile": 128, "rows": 8},
        reason="focus challenger (old deployed winner)",
        previous_fallback_provenance={"issue": 8, "source_commit": "x"},
    )
    entry.update(over)
    return entry


def _fb_table(entries, fallbacks):
    t = _table(entries)
    t["previous_fallbacks"] = fallbacks
    return t


def test_previous_fallback_outranks_semantic_entry_at_its_cell():
    fb = _fallback_entry()
    sem = _semantic_entry("c2c", "forward", block=512,
                          params={"variant": "spread", "batch": 256,
                                  "tile": 64, "rows": 8})
    table = _fb_table([sem], [fb])
    hit = find_entry(table, 1024, 256, "dit", "c2c", "forward", "backward")
    assert hit is fb
    # the fallback is a declared cell, not an entry -- identity tier
    from src.xpu.selector import is_previous_fallback
    assert is_previous_fallback(table, hit)
    assert not is_previous_fallback(table, sem)


def test_previous_fallback_answers_only_its_claimed_triple():
    fb = _fallback_entry("forward")
    sem_inv = _semantic_entry("c2c", "inverse")
    sem_r2c = _semantic_entry("r2c", "forward")
    legacy = _base_entry()
    table = _fb_table([sem_inv, sem_r2c, legacy], [fb])
    # its own key -> fallback
    assert find_entry(table, 1024, 256, "dit", "c2c", "forward",
                      "backward") is fb
    # inverse / r2c / c2r / non-backward norms fall through untouched
    assert find_entry(table, 1024, 256, "dit", "c2c", "inverse",
                      "backward") is sem_inv
    assert find_entry(table, 1024, 256, "dit", "r2c", "forward",
                      "backward") is sem_r2c
    assert find_entry(table, 1024, 256, "dit", "c2c", "forward",
                      "ortho") is legacy
    # other directions never see a dit-scoped fallback: a dif legacy
    # entry keeps answering dif queries even while dit is fallback-held
    legacy_dif = _base_entry(direction="dif")
    table = _fb_table([sem_inv, sem_r2c, legacy, legacy_dif], [fb])
    assert find_entry(table, 1024, 256, "dif", "c2c", "forward",
                      "backward") is legacy_dif
    assert find_entry(table, 1024, 256, "dit", "c2c", "forward",
                      "backward") is fb


def test_select_previous_fallback_tier_reason_and_params():
    fb = _fallback_entry()
    sem = _semantic_entry("c2c", "forward", block=512,
                          params={"variant": "spread", "batch": 256,
                                  "tile": 64, "rows": 8})
    table = _fb_table([sem], [fb])
    s = _select(table, transform_type="c2c",
                transform_direction="forward", normalization="backward")
    assert s.source == "table"
    assert s.entry_tier == "previous-fallback"
    assert s.params == fb["params"]
    assert s.block == fb["block"]
    assert s.reason.startswith(
        "previous-deployed per-cell fallback (issue #8):")
    # Selection.params carries no extra marker -- only plan_fft does
    assert "previous_fallback" not in s.params
    # the shadowed semantic entry still answers nothing else wrong:
    # its inverse sibling key is untouched when not fallback-claimed
    inv = _semantic_entry("c2c", "inverse", block=512)
    table2 = _fb_table([sem, inv], [fb])
    s2 = _select(table2, transform_type="c2c",
                 transform_direction="inverse", normalization="backward")
    assert s2.entry_tier == "semantic"
    assert s2.block == 512


def test_plan_fft_marks_the_fallback_geometry_and_reason():
    fb = _fallback_entry(block=256, incumbent_block=256)
    sem = _semantic_entry("c2c", "forward", block=512,
                          params={"variant": "spread", "batch": 256,
                                  "tile": 64, "rows": 8})
    table = _fb_table([sem], [fb])
    p = plan_fft("dit", 1024, batch=256, profile_hash=PROFILE,
                 table=table, env_hash=ENV,
                 transform_type="c2c", transform_direction="forward",
                 norm="backward")
    assert p.fallback is False
    assert p.selection_source == "table"
    raw = dict(p.params)
    assert raw.pop("previous_fallback") is True
    assert raw == fb["params"]        # deployed geometry is the old one
    assert p.block == fb["block"]
    assert p.executed == "fft_dit_batch"
    assert p.reason.startswith(
        "previous-deployed per-cell fallback (issue #8):")
    # the incumbent-block lookup shares the fallback tier (same source
    # of truth): M0 block follows the fallback entry, not the shadowed
    # semantic one
    sel = _select(table, transform_type="c2c",
                  transform_direction="forward", normalization="backward")
    assert _entry_block(table, sel) == fb["incumbent_block"]


def test_load_table_rejects_fallback_without_semantic_triple(tmp_path):
    import json
    from src.xpu.selector import load_table
    table = _fb_table([_base_entry()], [_base_entry()])  # no triple!
    path = tmp_path / "table.json"
    path.write_text(json.dumps(table))
    with pytest.raises(ConfigError, match="semantic triple"):
        load_table(str(path))


def test_load_table_rejects_duplicate_previous_fallback_cell(tmp_path):
    import json
    from src.xpu.selector import load_table
    table = _fb_table([], [_fallback_entry(), _fallback_entry()])
    path = tmp_path / "table.json"
    path.write_text(json.dumps(table))
    with pytest.raises(ConfigError, match="duplicate"):
        load_table(str(path))


def test_load_table_accepts_declared_previous_fallbacks(tmp_path):
    import json
    from src.xpu.selector import load_table
    table = _fb_table([_base_entry()], [_fallback_entry()])
    path = tmp_path / "table.json"
    path.write_text(json.dumps(table))
    t = load_table(str(path))
    assert len(t["previous_fallbacks"]) == 1
