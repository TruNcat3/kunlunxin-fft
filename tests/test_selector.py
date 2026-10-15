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


def test_table_rebuild_is_deterministic(tmp_path):
    """Rebuilding from the committed confirmation runs must reproduce the
    committed table (no hand edits)."""
    if not TABLE.exists() or not all(p.exists() for p in CONFIRMS):
        pytest.skip("confirmation artifacts not present")
    out = tmp_path / "selector_table.json"
    # relative paths keep table["sources"] stable (the rebuild must match
    # the committed file byte-for-byte in content, not depend on abs paths)
    cmd = [sys.executable, "scripts/build_selector_table.py",
           *[str(p.relative_to(ROOT)) for p in CONFIRMS], "--out", str(out)]
    if BATCH_CONFIRM.exists():
        cmd += ["--batch", str(BATCH_CONFIRM.relative_to(ROOT))]
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
    s = select(profile_hash=None, table=t, **kwargs)
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


def test_missing_table_raises_config_error(tmp_path):
    with pytest.raises(ConfigError):
        load_table(str(tmp_path / "nope.json"))
