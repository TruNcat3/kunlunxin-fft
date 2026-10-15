"""Host-only selector tests: the static table must be reproducible and safe.

Round C contract: table-driven per-size selection, M0 closure with the 2%
promotion guard, abstain outside the measured matrix. No vendor stack, no
torch, no device.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.xpu.config import ConfigError, DeviceProfile  # noqa: E402
from src.xpu.selector import GUARD_PCT, load_table, select  # noqa: E402

TABLE = ROOT / "config" / "selector_table.json"
CONFIRMS = [ROOT / "results" / "p800" / "confirm_b1024.json",
            ROOT / "results" / "p800" / "confirm_b256.json"]
BATCH_CONFIRM = ROOT / "results" / "p800" / "confirm_batch.json"


def test_selector_module_does_not_import_torch():
    src = (ROOT / "src" / "xpu" / "selector.py").read_text()
    assert "import torch" not in src


def test_table_loads_and_matches_profile():
    if not TABLE.exists():
        pytest.skip("selector table not built")
    t = load_table(str(TABLE))
    p = DeviceProfile.load()
    assert t["profile_hash"] == p.hash()
    assert t["precision"] == "fp32"
    assert t["guard_pct"] == GUARD_PCT
    keys = [(e["N"], e["batch"], e["direction"]) for e in t["entries"]]
    assert len(keys) == len(set(keys)), "duplicate table keys"


def _focus_sources():
    """Focus-run summaries recorded in the committed table's sources."""
    if not TABLE.exists():
        return []
    t = json.loads(TABLE.read_text())
    return [ROOT / s for s in t.get("sources", ())
            if "batch_gate" in s or "batch_release" in s]


def test_table_rebuild_is_deterministic(tmp_path):
    """Rebuilding from the committed confirmation + focus runs must
    reproduce the committed table (no hand edits)."""
    if not TABLE.exists() or not all(p.exists() for p in CONFIRMS):
        pytest.skip("confirmation artifacts not present")
    focus = [p for p in _focus_sources() if p.exists()]
    if focus != _focus_sources():
        pytest.skip("focus run artifacts not present")
    out = tmp_path / "selector_table.json"
    # relative paths keep table["sources"] stable (the rebuild must match
    # the committed file byte-for-byte in content, not depend on abs paths)
    cmd = [sys.executable, "scripts/build_selector_table.py",
           *[str(p.relative_to(ROOT)) for p in CONFIRMS], "--out", str(out)]
    if BATCH_CONFIRM.exists():
        cmd += ["--batch", str(BATCH_CONFIRM.relative_to(ROOT))]
    if focus:
        cmd += ["--focus", *[str(p.relative_to(ROOT)) for p in focus]]
    subprocess.run(cmd, cwd=ROOT, check=True)
    assert json.loads(out.read_text()) == json.loads(TABLE.read_text())


def test_entries_respect_guard_and_incumbent():
    """Every challenger entry must beat its M0 incumbent by the guard;
    M0-closure entries must be the ones where the guard did NOT clear."""
    if not TABLE.exists():
        pytest.skip("selector table not built")
    t = load_table(str(TABLE))
    guard = t["guard_pct"] / 100.0
    for e in t["entries"]:
        if e["batch"] != 1:
            continue
        inc, ch = e["incumbent_ms"], e["challenger_ms"]
        is_m0 = (e["flow_id"] == "baseline-global-stage"
                 and e["processing_unit"] == "radix2-scalar")
        if is_m0:
            assert ch > inc * (1 - guard) - 1e-12, (
                f"N={e['N']} {e['direction']}: M0 kept although challenger "
                f"{ch} clears the {t['guard_pct']}% guard vs {inc}"
            )
        else:
            assert ch <= inc * (1 - guard) + 1e-12, (
                f"N={e['N']} {e['direction']}: challenger {ch} does not clear "
                f"the {t['guard_pct']}% guard vs incumbent {inc}"
            )
            assert e["challenger_ms"] <= e["incumbent_ms"]


def test_select_returns_measured_winner():
    if not TABLE.exists():
        pytest.skip("selector table not built")
    t = load_table(str(TABLE))
    p = DeviceProfile.load()
    s = select("dit", 2048, profile_hash=p.hash(), table=t)
    assert s.source == "table"
    assert s.block in (256, 1024)
    assert s.flow_id and s.flow_realization and s.processing_unit
    assert s.params is not None


@pytest.mark.parametrize("kwargs", [
    {"direction": "dit", "n": 3000},            # not in the measured matrix
    {"direction": "dit", "n": 2048, "batch": 3},  # unmeasured batch
    {"direction": "dit", "n": 2048, "precision": "fp64"},
    {"direction": "r2c", "n": 2048},
])
def test_select_abstains_outside_matrix(kwargs):
    if not TABLE.exists():
        pytest.skip("selector table not built")
    t = load_table(str(TABLE))
    p = DeviceProfile.load()
    s = select(profile_hash=p.hash(), table=t, **kwargs)
    assert s.source == "abstain"
    assert not s.selected
    assert s.reason


def test_select_abstains_on_profile_mismatch():
    if not TABLE.exists():
        pytest.skip("selector table not built")
    t = load_table(str(TABLE))
    s = select("dit", 2048, profile_hash="0" * 16, table=t)
    assert s.source == "abstain"
    assert "profile hash" in s.reason


def test_select_abstains_on_missing_profile():
    """profile_hash=None must never bypass the table (review P0)."""
    if not TABLE.exists():
        pytest.skip("selector table not built")
    t = load_table(str(TABLE))
    s = select("dit", 2048, profile_hash=None, table=t)
    assert s.source == "abstain"
    assert "profile_hash is required" in s.reason


def test_table_rejects_cross_direction_m1():
    """M1 is DIT-only; a DIF entry may never claim fused-inplace."""
    if not TABLE.exists():
        pytest.skip("selector table not built")
    t = load_table(str(TABLE))
    for e in t["entries"]:
        if e["direction"] == "dif":
            assert e["flow_realization"] != "fused-inplace", (
                f"DIF entry N={e['N']} batch={e['batch']} claims "
                f"fused-inplace, which has no DIF lowering")


def test_missing_table_raises_config_error(tmp_path):
    with pytest.raises(ConfigError):
        load_table(str(tmp_path / "nope.json"))


# --- builder safety gates (review P0) -------------------------------------

def _fake_run(tmp_path, name, *, profile_hash=None,
              err=1e-7, ms=0.5, device="kunlunxin-p800",
              threshold=1e-4, dirty=False, warmup=10, trials=30,
              dtype="complex64", precision="fp32",
              timing_scope="api wall time (synchronized), ms",
              protocol_id="confirm-round:test:api_ms"):
    """Minimal confirmation-shaped run with one controllable row."""
    if profile_hash is None:
        profile_hash = DeviceProfile.load().hash()
    row = {"n": 2048, "dit_err": err, "dif_err": err,
           "m0_dit_ms": ms, "m0_dit_err": err,
           "m0_dif_ms": ms, "m0_dif_err": err,
           "m2_dit_t64_ms": ms / 2, "m2_dit_t64_err": err,
           "m2_dif_t64_ms": ms / 2, "m2_dif_t64_err": err,
           "m5_r4_dit_ms": ms / 2, "m5_r4_dit_err": err,
           "m5_r4_dif_ms": ms / 2, "m5_r4_dif_err": err}
    data = {"manifest": {
        "device": device, "profile_hash": profile_hash,
        "precision": precision,
        "correctness_threshold": threshold, "block": 1024, "dirty": dirty,
        "warmup": warmup, "trials": trials, "dtype": dtype,
        "timing_scope": timing_scope, "protocol_id": protocol_id,
        "extra": {"matrix_rows": 8, "glog": 2, "maxp": 4096, "cells": 1},
    }, "rows": [row]}
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def _build(tmp_path, *runs, batch=None):
    out = tmp_path / "t.json"
    cmd = [sys.executable, "scripts/build_selector_table.py",
           *[str(r) for r in runs], "--out", str(out)]
    if batch is not None:
        cmd += ["--batch", str(batch)]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True), out


def test_builder_rejects_failing_correctness(tmp_path):
    """A challenger whose error is at/above threshold must not enter."""
    run = _fake_run(tmp_path, "bad.json", err=2e-4)  # >= 1e-4
    proc, out = _build(tmp_path, run)
    assert proc.returncode == 0
    table = json.loads(out.read_text())
    assert table["rejections"], "expected logged rejections"
    for e in table["entries"]:
        assert e["flow_realization"] != "matrix-resident"
        # with every challenger rejected the M0 incumbent stays
        assert e["flow_realization"] == "global-stage"
        assert e["processing_unit"] == "radix2-scalar"


def test_builder_rejects_non_finite_ms(tmp_path):
    run = _fake_run(tmp_path, "nan.json", ms=float("nan"))
    proc, out = _build(tmp_path, run)
    assert proc.returncode == 0
    table = json.loads(out.read_text())
    assert table["rejections"]
    for e in table["entries"]:
        assert e["flow_realization"] == "global-stage"


def test_builder_rejects_mixed_profile(tmp_path):
    a = _fake_run(tmp_path, "a.json", profile_hash="aaaaaaaaaaaaaaaa")
    b = _fake_run(tmp_path, "b.json", profile_hash="bbbbbbbbbbbbbbbb")
    proc, _ = _build(tmp_path, a, b)
    assert proc.returncode != 0
    assert "mixed-profile" in (proc.stderr + proc.stdout)


def test_builder_rejects_incomplete_manifest(tmp_path):
    run = _fake_run(tmp_path, "inc.json")
    data = json.loads(run.read_text())
    del data["manifest"]["profile_hash"]
    run.write_text(json.dumps(data))
    proc, _ = _build(tmp_path, run)
    assert proc.returncode != 0
    assert "missing required keys" in (proc.stderr + proc.stdout)


def test_builder_rejects_profile_mismatch_vs_current(tmp_path):
    run = _fake_run(tmp_path, "other.json", profile_hash="0" * 16)
    proc, _ = _build(tmp_path, run)
    assert proc.returncode != 0
    assert "re-measure under the current profile" in (proc.stderr + proc.stdout)


@pytest.mark.parametrize("threshold", [2.0, 0.0, -1.0, float("nan")])
def test_builder_rejects_bad_threshold(tmp_path, threshold):
    run = _fake_run(tmp_path, "thr.json", threshold=threshold)
    proc, _ = _build(tmp_path, run)
    assert proc.returncode != 0
    assert "correctness_threshold" in (proc.stderr + proc.stdout)


def test_builder_rejects_mixed_protocol(tmp_path):
    """Same device/profile but different warmup is a mixed protocol."""
    a = _fake_run(tmp_path, "p1.json", warmup=10, trials=30)
    b = _fake_run(tmp_path, "p2.json", warmup=8, trials=30)
    proc, _ = _build(tmp_path, a, b)
    assert proc.returncode != 0
    assert "mixed-profile" in (proc.stderr + proc.stdout)


def test_builder_rejects_missing_protocol_fields(tmp_path):
    run = _fake_run(tmp_path, "np.json")
    data = json.loads(run.read_text())
    del data["manifest"]["warmup"]
    run.write_text(json.dumps(data))
    proc, _ = _build(tmp_path, run)
    assert proc.returncode != 0


def test_builder_rejects_mixed_timing_scope(tmp_path):
    """Same device/profile but different timing scope is mixed protocol."""
    a = _fake_run(tmp_path, "ts1.json",
                  timing_scope="api wall time (synchronized), ms")
    b = _fake_run(tmp_path, "ts2.json",
                  timing_scope="device_event only")
    proc, _ = _build(tmp_path, a, b)
    assert proc.returncode != 0
    assert "mixed-profile" in (proc.stderr + proc.stdout)


def test_builder_rejects_mixed_protocol_id(tmp_path):
    a = _fake_run(tmp_path, "pid1.json", protocol_id="round-a")
    b = _fake_run(tmp_path, "pid2.json", protocol_id="round-b")
    proc, _ = _build(tmp_path, a, b)
    assert proc.returncode != 0
    assert "mixed-profile" in (proc.stderr + proc.stdout)


def test_builder_rejects_non_fp32_precision(tmp_path):
    """Only FP32 tables are built; fp64 measurements are refused."""
    run = _fake_run(tmp_path, "fp64.json", precision="fp64",
                    dtype="complex128")
    proc, _ = _build(tmp_path, run)
    assert proc.returncode != 0
    assert "precision" in (proc.stderr + proc.stdout)


def test_builder_rejects_dtype_precision_mismatch(tmp_path):
    run = _fake_run(tmp_path, "mismatch.json", precision="fp32",
                    dtype="complex128")
    proc, _ = _build(tmp_path, run)
    assert proc.returncode != 0
    assert "does not match" in (proc.stderr + proc.stdout)
