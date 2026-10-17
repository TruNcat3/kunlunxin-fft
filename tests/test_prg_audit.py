"""Host tests for issue #8 PR-G step 1 (v2 regret audit).

The frozen v2 release is the baseline issue #8 is judged against: the
audit must keep classifying its gates correctly (regret =
selector/oracle - 1; main <= 0.10; stress N1024B256 open; confirmation
targets restricted to fixed domains).
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from scripts.audit_v2_regret import (  # noqa: E402
    domain_of,
    FIXED_DOMAINS,
    REGRET_MAX,
)

RELEASE = ROOT / "results" / "p800" / "releases" / \
    "20261017-wp2-semantic-matrix-v2"


def test_domain_classification():
    assert domain_of(1024, 16) == "main"
    assert domain_of(2048, 64) == "main"
    assert domain_of(1024, 256) == "stress"
    assert domain_of(512, 64) == "holdout"
    assert domain_of(2048, 4) == "holdout"
    assert domain_of(64, 1) == "regression"
    assert domain_of(16384, 16) == "regression"
    assert domain_of(1024, 256) != "main"
    assert domain_of(64, 16) == "background"
    assert domain_of(2048, 256) == "background"


@pytest.mark.skipif(not RELEASE.exists(), reason="v2 release not present")
def test_audit_gates_on_frozen_v2(tmp_path):
    subprocess.run([sys.executable, "scripts/audit_v2_regret.py",
                    "--release", str(RELEASE.relative_to(ROOT)),
                    "--out", str(tmp_path)],
                   cwd=ROOT, check=True)
    r = json.loads((tmp_path / "pr-g-regret-audit.json").read_text())

    assert r["evidence_class"] == "summary-derived"
    assert r["cells_total"] == 128  # 144 entries, 16 duplicate rows
    assert r["gate_main_regret_le_10pct"]["pass"] is True
    assert r["gate_main_regret_le_10pct"]["regret_max"] == REGRET_MAX

    stress = r["gate_stress_n1024b256_must_fix"]
    assert stress["pass"] is False
    assert sorted(v["label"] for v in stress["violations"]) == [
        "N1024B256:c2c-forward", "N1024B256:c2c-inverse",
        "N1024B256:c2r", "N1024B256:r2c"]

    targets = r["step2_independent_confirmation_targets"]
    labels = {t["label"] for t in targets}
    assert len(targets) == 24
    for sem in ("c2c-forward", "c2c-inverse", "r2c", "c2r"):
        assert f"N1024B256:{sem}" in labels
        assert f"N64B1:{sem}" in labels
    # confirmation targets stay inside the fixed domains of issue #8
    for t in targets:
        cell = next(c for c in r["cells"] if c["label"] == t["label"])
        assert cell["domain"] in FIXED_DOMAINS
        assert cell["selector_source"] == "abstain"
        assert cell["selector_regret"] >= REGRET_MAX

    pos = r["guard_positive_production_points"]
    for sem in ("c2c-inverse", "r2c", "c2r", "c2c-forward"):
        assert pos[sem], f"no guard-positive point for {sem}"
        assert all(p["guard"] >= 0.02 for p in pos[sem])
