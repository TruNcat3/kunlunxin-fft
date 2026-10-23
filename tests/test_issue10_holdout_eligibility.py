"""issue #10 PR #12 round-3 review: eligibility field protection.

Pure-Python unit tests (host-only, no torch) for the code-level
correctness/contract gate and the eligibility append conditions in
scripts/issue10_wp3_holdout.py — the review required
``correctness_and_contract_pass`` to explicitly depend on
``all_paths_present``, ``no_fail_closed``, ``correct_all_sessions``
and ``contract_all_sessions``, never on CI alone, and the hard gates
to use the profile-bound incumbent.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_holdout():
    spec = importlib.util.spec_from_file_location(
        "issue10_wp3_holdout",
        ROOT / "scripts" / "issue10_wp3_holdout.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("issue10_wp3_holdout", mod)
    spec.loader.exec_module(mod)
    return mod


H = _load_holdout()


def _flags(**over):
    f = {"all_paths_present": True, "no_fail_closed": True,
         "correct_all_sessions": True, "contract_all_sessions": True,
         "entry_shape_uniform": True}
    f.update(over)
    return f


def _ready():
    return {ch: {"vendor": [], "incumbent": [], "correct": []}
            for ch in H.CANDIDATES}


def _spreads():
    return {ch: {"vendor": [], "incumbent": []} for ch in H.CANDIDATES}


def _boot(point, lo):
    return {"point": point, "ci95": [lo, lo * 1.5 + 1e-3], "n_boot": 10}


def _n():
    return len(H.SESSIONS)


def test_cell_passes_requires_all_gate_flags():
    assert H._cell_passes(_flags()) is True
    for k in _flags():
        assert H._cell_passes(_flags(**{k: False})) is False, k


def test_collect_ready_excludes_failing_cell_entirely():
    ready, spreads = _ready(), _spreads()
    ch = H.CANDIDATES[0]
    vs = [_boot(1.5, 1.4)] * _n()
    iss = [_boot(2.0, 1.5)] * _n()
    out = H._collect_ready(ready, spreads, ch, "L", vs, iss, False, _n())
    assert out == {}
    assert ready[ch] == {"vendor": [], "incumbent": [], "correct": []}
    assert spreads[ch] == {"vendor": [], "incumbent": []}


def test_correct_list_depends_on_cell_ok_not_ci():
    ready, spreads = _ready(), _spreads()
    ch = H.CANDIDATES[0]
    low = [_boot(1.5, 0.5)] * _n()          # CI far below the gate
    H._collect_ready(ready, spreads, ch, "L", low, low, True, _n())
    # correctness+contract passed, so the field must contain the cell...
    assert ready[ch]["correct"] == ["L"]
    # ...while both CI gates stay empty and the intersection is empty
    assert ready[ch]["vendor"] == []
    assert ready[ch]["incumbent"] == []
    inter = [l for l in ready[ch]["vendor"]
             if l in set(ready[ch]["incumbent"])]
    assert inter == []


def test_gates_and_correct_populated_when_all_pass():
    ready, spreads = _ready(), _spreads()
    ch = H.CANDIDATES[1]
    vs = [_boot(1.5, 1.4)] * _n()
    iss = [_boot(3.0, 2.5)] * _n()
    H._collect_ready(ready, spreads, ch, "L", vs, iss, True, _n())
    assert ready[ch]["correct"] == ["L"]
    assert ready[ch]["vendor"] == ["L"]
    assert ready[ch]["incumbent"] == ["L"]
    assert spreads[ch]["vendor"] and spreads[ch]["incumbent"]


def test_incomplete_sessions_never_pass_correct():
    ready, spreads = _ready(), _spreads()
    ch = H.CANDIDATES[0]
    vs = [_boot(1.5, 1.4)]                   # only one session of data
    iss = [_boot(3.0, 2.5)] * _n()
    H._collect_ready(ready, spreads, ch, "L", vs, iss, True, _n())
    assert ready[ch]["correct"] == []
    assert ready[ch]["vendor"] == []
    assert ready[ch]["incumbent"] == []


def test_only_one_gate_passing_is_not_eligible():
    ready, spreads = _ready(), _spreads()
    vs = [_boot(1.5, 1.4)] * _n()
    iss_low = [_boot(3.0, 0.5)] * _n()       # incumbent CI fails
    H._collect_ready(ready, spreads, H.CANDIDATES[1], "A", vs, iss_low,
                     True, _n())
    block = H._eligibility_block(ready, spreads, 16)
    c1 = block["candidates"][H.CANDIDATES[1]]
    assert c1["meets_vendor_gate"] == ["A"]
    assert c1["meets_incumbent_profile_bound_gate"] == []
    assert c1["eligible_for_selector"] == []
    assert block["eligible_for_selector"] == []


def test_eligibility_block_is_evidence_only_and_profile_bound():
    ready, spreads = _ready(), _spreads()
    vs = [_boot(1.5, 1.4)] * _n()
    iss = [_boot(3.0, 2.5)] * _n()
    H._collect_ready(ready, spreads, H.CANDIDATES[0], "A", vs, iss,
                     True, _n())
    block = H._eligibility_block(ready, spreads, 16)
    assert block["selector_wiring"]["changes_default_path"] is False
    assert block["selector_wiring"]["selector_table_modified"] is False
    assert block["candidates"][H.CANDIDATES[0]][
        "eligible_for_selector"] == ["A"]
    assert "PROFILE-BOUND" in block["gate"]
    assert "m0_reference" in block["incumbent_policy"]


def test_field_definitions_and_policies_documented():
    ready, spreads = _ready(), _spreads()
    block = H._eligibility_block(ready, spreads, 16)
    for key in ("all_paths_present", "no_fail_closed",
                "correct_all_sessions", "contract_all_sessions",
                "entry_shape_uniform"):
        assert key in block["cell_field_definitions"]
    assert "descriptive" in block["issue9_policy"]
    assert "entry-level" in block["batched_boundary"]
    assert "composition_table" in block["composition_policy"]
    assert "spread guard" in block["promotion_prerequisites"]


def test_entry_shape_uniform_gate():
    ok = {1: {"vendor": [1024], "incumbent_entry": [1024],
              "m0_reference": [1024], "hybrid_vendor_p": [1024],
              "hybrid_fused_p": [1024]},
          2: {"vendor": [1024], "incumbent_entry": [1024],
              "m0_reference": [1024], "hybrid_vendor_p": [1024],
              "hybrid_fused_p": [1024]}}
    assert H._entry_shape_uniform(ok) is True
    bad = {1: {"vendor": [1, 1024], "incumbent_entry": [1024]}}
    assert H._entry_shape_uniform(bad) is False          # rank-2 vs 1-D
    assert H._entry_shape_uniform({}) is False           # no sessions
    assert H._entry_shape_uniform(
        {1: {"vendor": ["__missing__"]}}) is False        # lost record
    mixed_sessions = {1: {"vendor": [1024]}, 2: {"vendor": [1, 1024]}}
    assert H._entry_shape_uniform(mixed_sessions) is False


def test_entry_tier_distinguishes_composition():
    assert H._entry_tier("composition", {"flow_kind": "composition"},
                         False) == "composition-table"
    assert H._entry_tier("table", {"previous_fallback": True},
                         False) == "previous-fallback"
    assert H._entry_tier("abstain", {}, True) == "m0-abstain"
    assert H._entry_tier("table", {}, False) == "table/semantic-or-legacy"
    # composition wins over plain table but never over explicit
    # previous-fallback / abstain attribution
    assert H._entry_tier("composition", {}, False) == "composition-table"


def test_entry_meta_records_shape_and_contract():
    class Fake1D:
        shape = (1024,)

        def stride(self):
            return (1,)

    m1 = H._entry_meta("hybrid_fused_p", Fake1D(), 1)
    assert m1["input_shape"] == [1024]
    assert m1["batch_contract"].startswith("rank-1")
    assert "rank-1" in m1["entry_contract"] or "(N,)" in m1["entry_contract"]
    assert "run_hybrid_fused" in m1["api_entry"]
    assert isinstance(m1["input_stride"], list)

    class Fake2D:
        shape = (16, 1024)
        _stride = (1024, 1)

        def stride(self):
            return (1024, 1)

    m2 = H._entry_meta("vendor", Fake2D(), 16)
    assert m2["input_shape"] == [16, 1024]
    assert m2["batch_contract"].startswith("rank-2")
    # b==1 uniformity: every path reports the same shape/contract
    shapes = {p: H._entry_meta(p, Fake1D(), 1)["input_shape"]
              for p in H.PATHS}
    assert len({tuple(v) for v in shapes.values()}) == 1
