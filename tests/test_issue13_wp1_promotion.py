"""issue #13 WP1 promotion rules — pure host tests.

Locks the issue's promotion contract: per-session dual-baseline CI
gate (>1.02 vs vendor AND profile-bound incumbent), cross-session
spread <= 10%, all-path correctness/contract flags, and the
"measured faster wins" policy between passing candidates.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "issue13_wp1_promotion",
        ROOT / "scripts" / "issue13_wp1_promotion.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("issue13_wp1_promotion", mod)
    spec.loader.exec_module(mod)
    return mod


M = _load()


def _flags(**over):
    f = {"all_paths_present": True, "no_fail_closed": True,
         "correct_all_sessions": True, "contract_all_sessions": True}
    f.update(over)
    return f


def _ci(lo):
    return {"ci95": [lo, lo * 1.1]}


def test_all_green_passes():
    v = M.promotion_verdict(
        [_flags()] * 4, [_ci(1.05)] * 4, [_ci(1.03)] * 4,
        [1.10, 1.08, 1.12, 1.09], [1.05, 1.04, 1.06, 1.05], True)
    assert v["pass"] is True and v["reasons"] == []


@pytest.mark.parametrize("baseline", ["vendor", "incumbent"])
def test_ci_gate_must_exceed_1_02_in_every_session(baseline):
    vcs = [_ci(1.05)] * 4
    ics = [_ci(1.05)] * 4
    target = vcs if baseline == "vendor" else ics
    target[2] = _ci(1.019)          # fails only in session 3
    v = M.promotion_verdict([_flags()] * 4, vcs, ics,
                            [1.1] * 4, [1.1] * 4, True)
    assert v["pass"] is False
    assert any(baseline in r and "session(s) 3" in r
               for r in v["reasons"]), v["reasons"]


def test_ci_gate_is_lower_bound_not_point():
    # point 1.5 but lower bound 1.01 <= gate -> fail
    lo = {"ci95": [1.01, 1.5]}
    v = M.promotion_verdict([_flags()] * 4, [lo] * 4, [_ci(1.5)] * 4,
                            [1.5] * 4, [1.5] * 4, True)
    assert v["pass"] is False


def test_spread_over_10_percent_blocks_promotion():
    v = M.promotion_verdict([_flags()] * 4, [_ci(1.5)] * 4,
                            [_ci(1.5)] * 4,
                            [1.50, 1.52, 1.30, 1.51],   # 1.30: -13%
                            [1.5] * 4, True)
    assert v["pass"] is False
    assert any("spread" in r for r in v["reasons"]), v["reasons"]
    # within 10% passes (1.50 vs 1.44: -4%)
    v2 = M.promotion_verdict([_flags()] * 4, [_ci(1.5)] * 4,
                             [_ci(1.5)] * 4,
                             [1.50, 1.52, 1.44, 1.51], [1.5] * 4, True)
    assert v2["pass"] is True


def test_gate_flags_and_uniformity_required():
    assert not M.promotion_verdict(
        [_flags(correct_all_sessions=False)] + [_flags()] * 3,
        [_ci(1.5)] * 4, [_ci(1.5)] * 4, [1.5] * 4, [1.5] * 4,
        True)["pass"]
    assert not M.promotion_verdict(
        [_flags()] * 4, [_ci(1.5)] * 4, [_ci(1.5)] * 4,
        [1.5] * 4, [1.5] * 4, False)["pass"]  # entry_shape_uniform


def test_missing_ci_session_fails():
    cis = [_ci(1.5), None, _ci(1.5), _ci(1.5)]
    v = M.promotion_verdict([_flags()] * 4, cis, [_ci(1.5)] * 4,
                            [1.5] * 4, [1.5] * 4, True)
    assert v["pass"] is False


def test_choose_faster_is_measured_not_by_name():
    rng = np.random.default_rng(0)
    # fused faster (0.30 ms) vs vendor_p (0.50 ms)
    a = M.choose_faster({
        "hybrid_fused_p": list(np.full(50, 0.30)),
        "hybrid_vendor_p": list(np.full(50, 0.50))}, rng)
    assert a["selected"] == "hybrid_fused_p"
    assert a["pairwise"]["point"] < 1.0
    # vendor_p faster wins regardless of the fused primary name
    b = M.choose_faster({
        "hybrid_fused_p": list(np.full(50, 0.60)),
        "hybrid_vendor_p": list(np.full(50, 0.20))}, rng)
    assert b["selected"] == "hybrid_vendor_p"
