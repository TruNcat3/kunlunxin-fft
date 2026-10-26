"""issue #13 WP4 coverage grid — pure host tests.

Locks the grid shape (49 forward-C2C + 12 long-size points, no
inverse), the win/tie/loss mapping over the WP3 confirmed-regression
criterion, the retention semantics (selector-failed and
vendor-unsupported points can never be folded into win/tie/loss), the
geomean/median helpers, and that the measurement core (steady-state
warmup rule and bootstrap gate) is the shared WP3 implementation, not
a fork.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


M4 = _load("issue13_wp4_coverage_test", "scripts/issue13_wp4_coverage.py")
M3 = _load("issue13_wp3_release_test2", "scripts/issue13_wp3_release.py")


# ------------------------------------------------------------------ grid

def test_c2c_grid_is_the_49_cell_public_matrix():
    assert M4.C2C_CELLS == [(n, b) for n in M4.C2C_NS for b in M4.C2C_BS]
    assert len(M4.C2C_CELLS) == 49
    assert set(M4.C2C_NS) == {64, 128, 256, 512, 1024, 2048, 4096}
    assert set(M4.C2C_BS) == {1, 4, 16, 64, 256, 1024, 4096}


def test_long_slice_is_the_12_point_capability_grid():
    assert M4.LONG_CELLS == [(n, b) for n in M4.LONG_NS
                             for b in M4.LONG_BS]
    assert len(M4.LONG_CELLS) == 12
    assert set(M4.LONG_CELLS) >= {(8192, 1), (65536, 256)}
    assert len(set(M4.LONG_CELLS) & set(M4.C2C_CELLS)) == 0


def test_cells_join_is_c2c_plus_long():
    assert M4.CELLS == M4.C2C_CELLS + M4.LONG_CELLS
    assert len(M4.CELLS) == 61


# -------------------------------------------------------------- verdicts

def _cis(boot):
    return [{"high": boot["point"] * 1.01, "low": boot["point"] * 0.99}
            for _ in range(3)]


def test_verdict_win_and_tie():
    v, pooled = M4._verdict([1.05, 1.06, 1.05], _cis({"point": 1.05}),
                            True)
    assert v == "win" and pooled == pytest.approx(1.05)
    v2, pooled2 = M4._verdict([1.0, 1.0, 1.0], _cis({"point": 1.0}),
                              True)
    assert v2 == "tie" and pooled2 == pytest.approx(1.0)


def test_verdict_loss_is_wp3_confirmed_regression():
    # pooled below 0.98 triggers the loss regardless of session spread
    v, pooled = M4._verdict([0.5, 0.5, 0.5], _cis({"point": 0.5}), True)
    assert v == "loss" and pooled == pytest.approx(0.5)
    # pooled above 0.98 but two bootstrap CI-highs below 0.98 also lose
    cis = [{"high": 0.97, "low": 0.90}, {"high": 0.97, "low": 0.90},
           {"high": 1.10, "low": 1.00}]
    v2, _ = M4._verdict([0.99, 0.99, 1.05], cis, True)
    assert v2 == "loss"


def test_verdict_retention_semantics():
    # a failing selector can never be scored win/tie/loss even with
    # winning session points, and a cell with no vendor comparison is
    # retained as vendor-unsupported
    v1, p1 = M4._verdict([1.30, 2.0, 3.0], _cis({"point": 2.0}), False)
    assert v1 == "selector-failed" and p1 is None
    v2, p2 = M4._verdict([], [], True)
    assert v2 == "vendor-unsupported" and p2 is None


# --------------------------------------------------------------- helpers

def test_pooled_median_even_and_odd():
    assert M4.pooled_of([3.0, 1.0, 2.0]) == 2.0
    assert M4.pooled_of([1.0, 2.0, 3.0, 4.0]) == 2.5


def test_geomean_and_empty_safety():
    assert M4._geomean([1.0, 4.0]) == pytest.approx(2.0)
    assert M4._geomean([]) is None
    assert M4._geomean([None, 0]) is None


# ------------------------------------------------- shared WP3 core

def test_measurement_core_is_shared_wp3_implementation():
    w3 = M4._load_wp3()
    assert Path(w3.__file__) == Path(M3.__file__)
    for k in ("STEADY_BATCH", "STEADY_MIN_CALLS", "STEADY_MAX_CALLS",
              "STEADY_TOL", "STEADY_NEED"):
        assert getattr(w3, k) == getattr(M3, k)
    # same run-2 steady-state warmup stop rule drives both workers
    assert w3.steady_warmup_done([1.0] * M3.STEADY_NEED) is True
    assert w3.steady_warmup_done([1.1, 1.0, 0.9]) is False
    # WP4 loss/win points must equal the WP3 gate points
    assert M4.LOSS_POINT == M3.REGRESSION_PTS
    assert M4.WIN_POINT == M3.GATE_CI_LOW
