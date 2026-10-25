"""issue #13 WP3 release matrix — pure host tests.

Locks the aggregation/exit-condition contract of
scripts/issue13_wp3_release.py: H_boot direction (>1 = new selector
faster), classify_cell win/tie/regression semantics with promoted
keys exempt from the regression gate, the WP1 dual-baseline threshold
kept on promoted keys, the legacy-table snapshot filter, and the
steady-state warmup stop rule (run 2 protocol fix after run 1's
warmup=5 left an asymmetric ramp in the timed window).
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "issue13_wp3_release",
        ROOT / "scripts" / "issue13_wp3_release.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("issue13_wp3_release", mod)
    spec.loader.exec_module(mod)
    return mod


M = _load()


# ----------------------------------------------------------------- H_boot

def test_h_boot_direction_old_over_new():
    rng = np.random.default_rng(0)
    # old incumbent 10ms, new selector 5ms -> point ~2 (>1 = new faster)
    sp = M.H_boot(rng, [10.0, 10.1, 9.9] * 20, [5.0, 5.1, 4.9] * 20,
                  n_boot=200)
    assert sp and sp["point"] > 1.9
    assert sp["low"] > 1.8 and sp["high"] > 1.9
    assert sp["bootstrap_kind"] == "independent"


def test_h_boot_direction_regression_inverted():
    rng = np.random.default_rng(0)
    # new selector SLOWER than old -> point < 1
    sp = M.H_boot(rng, [5.0] * 60, [10.0] * 60, n_boot=200)
    assert sp and sp["point"] < 0.55


def test_h_boot_empty_is_none():
    assert M.H_boot(np.random.default_rng(0), [], [1.0]) is None


# ----------------------------------------------------------- classify_cell

def _ci(lo, hi):
    return {"point": 0.5 * (lo + hi), "low": lo, "high": hi}


def test_classify_cell_win():
    pts = [1.05, 1.06, 1.04]
    cis = [_ci(1.03, 1.07)] * 3
    assert M.classify_cell(pts, cis, promoted=False) == "win"
    assert M.classify_cell(pts, cis, promoted=True) == "win"


def test_classify_cell_tie():
    pts = [1.00, 1.01, 0.99]
    cis = [_ci(0.98, 1.02)] * 3
    assert M.classify_cell(pts, cis, promoted=False) == "tie"


def test_classify_cell_confirmed_regression_pooled():
    pts = [0.95, 0.96, 0.94]
    cis = [_ci(0.93, 0.97)] * 3
    assert M.classify_cell(pts, cis, promoted=False) == \
        "confirmed-regression"


def test_classify_cell_confirmed_regression_two_ci_sessions():
    pts = [1.00, 0.95, 0.95]
    cis = [_ci(0.99, 1.01), _ci(0.90, 0.96), _ci(0.91, 0.97)]
    assert M.classify_cell(pts, cis, promoted=False) == \
        "confirmed-regression"


def test_classify_cell_no_points_is_tie():
    assert M.classify_cell([], [], promoted=False) == "tie"


# --------------------------------------------------------- exit_conditions

def _cell(n, b, inv, verdict, pooled, dual=None, correct=True):
    """Minimal cells_out entry for exit_conditions."""
    flags = [{"all_paths_correct": correct}] * len(M.SESSIONS)
    return {
        "label": f"N{n}B{b}:{'inv' if inv else 'fwd'}",
        "n": n, "batch": b, "inverse": bool(inv),
        "gate_flags_by_session": flags,
        "comparison": {
            "verdict": verdict, "pooled_speedup": pooled,
            "points": [pooled] * 3,
            "promoted_key": (n, b, int(inv)) in M.PROMOTED,
            "dual_baseline_pass": dual,
        },
    }


def test_exit_conditions_all_green():
    cells = [_cell(n, b, inv, "win", 1.05, dual=True)
             for n, b, inv in M.CELLS]
    ec = M.exit_conditions(cells)
    assert ec["pass"] is True
    assert ec["bad_correctness_cells"] == []
    assert ec["regressions"] == []
    assert ec["promoted_dual_baseline"] is True
    assert ec["worst_cell"] is not None
    assert ec["geomean_speedup_old_over_new"] > 1.0


def test_exit_conditions_bad_correctness_fails():
    cells = [_cell(n, b, inv, "win", 1.05, dual=True)
             for n, b, inv in M.CELLS]
    cells[3]["gate_flags_by_session"] = [{"all_paths_correct": False}]
    ec = M.exit_conditions(cells)
    assert ec["pass"] is False
    assert cells[3]["label"] in ec["bad_correctness_cells"]


def test_exit_conditions_regression_on_uncovered_cell_fails():
    cells = [_cell(n, b, inv, "win", 1.05, dual=True)
             for n, b, inv in M.CELLS]
    # (1024, 1, 0) is NOT promoted
    cells[0].update({
        "label": "N1024B1:fwd", "n": 1024, "batch": 1, "inverse": False,
        "comparison": {**cells[0]["comparison"],
                       "verdict": "confirmed-regression",
                       "pooled_speedup": 0.94}})
    ec = M.exit_conditions(cells)
    assert ec["pass"] is False
    assert ec["regressions"][0]["label"] == "N1024B1:fwd"
    assert ec["no_confirmed_regression_on_uncovered_cells"] is False


def test_exit_conditions_promoted_regression_not_in_regressions():
    cells = [_cell(n, b, inv, "win", 1.05, dual=True)
             for n, b, inv in M.CELLS]
    # (2048, 1, 1) IS promoted -> confirmed-regression reported but
    # not gated as an uncovered-cell regression
    cells[0].update({
        "label": "N2048B1:inv", "n": 2048, "batch": 1, "inverse": True,
        "comparison": {**cells[0]["comparison"],
                       "verdict": "confirmed-regression",
                       "pooled_speedup": 0.90}})
    ec = M.exit_conditions(cells)
    assert ec["regressions"] == []
    assert ec["no_confirmed_regression_on_uncovered_cells"] is True


def test_exit_conditions_dual_baseline_gate():
    cells = [_cell(n, b, inv, "win", 1.05, dual=True)
             for n, b, inv in M.CELLS]
    # flip ONE promoted key's dual-baseline flag
    for c in cells:
        if (c["n"], c["batch"], int(c["inverse"])) in M.PROMOTED:
            c["comparison"]["dual_baseline_pass"] = False
            break
    ec = M.exit_conditions(cells)
    assert ec["pass"] is False
    assert ec["promoted_dual_baseline"] is False
    assert ec["promoted_dual_baseline_failures"][0]["label"]


def test_exit_conditions_missing_dual_baseline_fails():
    cells = [_cell(n, b, inv, "win", 1.05, dual=None)
             for n, b, inv in M.CELLS]
    ec = M.exit_conditions(cells)
    assert ec["promoted_dual_baseline"] is False
    assert ec["pass"] is False


def test_exit_conditions_worst_cell_is_min_pooled():
    cells = [_cell(n, b, inv, "win", 1.05, dual=True)
             for n, b, inv in M.CELLS]
    cells[5].update({
        "label": "N1024B64:fwd", "n": 1024, "batch": 64,
        "inverse": False,
        "comparison": {**cells[5]["comparison"], "pooled_speedup": 1.005}})
    ec = M.exit_conditions(cells)
    assert ec["worst_cell"] == "N1024B64:fwd"


# -------------------------------------------------------- _write_legacy_table

def test_write_legacy_table_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "LEGACY_TABLE",
                        tmp_path / "legacy_composition_table.json")
    M._write_legacy_table()
    out = json.loads(M.LEGACY_TABLE.read_text())
    assert [e.get("flow_id") for e in out["entries"]] == ["hybrid-A1P1"]
    assert all("variant" not in e for e in out["entries"])
    from src.xpu.manifest_contract import rows_sha256
    assert out["result_sha256"] == rows_sha256({"rows": out["entries"]})


# --------------------------------------------------- steady_warmup_done

def test_steady_warmup_needs_min_batches():
    assert M.steady_warmup_done([]) is False
    assert M.steady_warmup_done([1.0]) is False
    assert M.steady_warmup_done([1.0, 1.0]) is False  # fewer than need
    # a descending ramp is not settled either (global-min rule)
    assert M.steady_warmup_done([1.1, 1.0, 0.9]) is False


def test_steady_warmup_flat_is_done():
    assert M.steady_warmup_done([1.0] * M.STEADY_NEED) is True


def test_steady_warmup_ramp_never_stops_early():
    # run-1 shape: slow plateau -> middle plateau -> steady; the worker
    # gate (min-calls + stop rule, mimicked below) must never fire
    # before STEADY_MIN_CALLS, so the middle plateau is unreachable
    ramp = [1.13] * 4 + [0.77] * 3 + [0.41] * 40

    def stop(batch_means):
        calls = 0
        for i, _ in enumerate(batch_means):
            calls += M.STEADY_BATCH
            if (calls >= M.STEADY_MIN_CALLS
                    and M.steady_warmup_done(batch_means[:i + 1])):
                return i
        return None

    # first legal check is batch index 20 (20 * 16 = 320 calls), by
    # then the ramp is long over -> stops there, flat afterwards
    assert stop(ramp) == M.STEADY_MIN_CALLS // M.STEADY_BATCH - 1
    # a flat profile stops at the same first legal check
    assert stop([1.0] * 60) == M.STEADY_MIN_CALLS // M.STEADY_BATCH - 1
    # a profile that keeps jittering beyond tolerance never stops
    # (caller hits STEADY_MAX_CALLS and records reached=false)
    jitter = [1.0 if i % 2 == 0 else 1.05 for i in
              range(M.STEADY_MAX_CALLS // M.STEADY_BATCH)]
    assert stop(jitter) is None
    # never stops before the min-calls gate by construction of `stop`


def test_steady_warmup_tolerance_and_reset_on_new_dip():
    # 1% noise around the minimum still settles...
    assert M.steady_warmup_done([1.0, 1.005, 0.99, 1.002]) is True
    # ...3% above the running minimum does not...
    assert M.steady_warmup_done([1.0, 1.03, 1.04]) is False
    # ...and a fresh dip resets the streak (global-min rule)
    assert M.steady_warmup_done([1.0, 0.99, 0.99, 0.90]) is False
    assert M.steady_warmup_done([0.90, 0.90, 0.90, 0.90]) is True


def test_steady_warmup_bounds_cover_run1_ramp():
    # run-1 ramps settled by timed call ~144 (~absolute call 151) ->
    # the first legal check (20 batches) must sit well past it, and
    # the cap must stay well above the min
    assert M.STEADY_MIN_CALLS >= 20 * M.STEADY_BATCH
    assert M.STEADY_MAX_CALLS > M.STEADY_MIN_CALLS
    assert M.STEADY_MIN_CALLS >= M.STEADY_NEED * M.STEADY_BATCH
