"""Host-only tests: run with NO vendor stack, NO torch, NO device.

These cover the configuration schema, candidate enumeration / lowering
contract, and manifest generation — the pieces that must be verifiable in
ordinary CI (migration convergence plan Phase 0 deliverable 4).
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.xpu.candidates import (
    Candidate,
    LoweringStatus,
    check_lowering,
    enumerate_candidates,
    selected_candidate,
)
from src.xpu.config import (
    ConfigError,
    DesignSpace,
    DeviceProfile,
    SizeMatrix,
    build_manifest,
    git_commit,
)

# The pure-python modules must not import torch at module scope.
def test_config_modules_do_not_import_torch():
    import src.xpu.config as cfg
    import src.xpu.candidates as cand

    assert "torch" not in sys.modules or True  # torch may be imported elsewhere
    # stronger: the modules themselves must not reference torch
    src = Path(cfg.__file__).read_text()
    assert "import torch" not in src
    src2 = Path(cand.__file__).read_text()
    assert "import torch" not in src2


def test_device_profile_loads_and_hashes():
    p = DeviceProfile.load()
    assert p.device == "kunlunxin-p800"
    assert p.clusters == 12
    assert p.vendor_fft_max_n == 2048
    assert len(p.hash()) == 16


def test_size_matrix_all_powers_of_two():
    s = SizeMatrix.load()
    assert s.bench_sizes
    for n in s.correctness_sizes:
        assert n > 0 and (n & (n - 1)) == 0
    assert s.vendor_baseline_max_n == 2048


def test_design_space_exposes_consumed_axes():
    ds = DesignSpace.load()
    assert ds.radix == (2,)
    assert ds.fused_tile_candidates  # M1 axis present and consumed
    assert all((t & (t - 1)) == 0 for t in ds.fused_tile_candidates)
    assert 1 in ds.fused_tile_candidates  # M0 baseline must be an option


def test_manifest_pins_commit_and_profile():
    p = DeviceProfile.load()
    m = build_manifest(p, block=512, trials=7)
    assert m.commit == git_commit()
    assert m.profile_hash == p.hash()
    assert m.block == 512
    assert m.trials == 7
    assert m.device == p.device
    d = m.to_dict()
    assert "correctness_threshold" in d and d["correctness_threshold"] == 1e-4


def test_manifest_rejects_unknown_field():
    p = DeviceProfile.load()
    with pytest.raises(ConfigError):
        build_manifest(p, not_a_field=1)


def test_baseline_candidate_is_legal():
    p = DeviceProfile.load()
    c = Candidate(flow_id="baseline-global-stage", processing_unit="radix-2",
                  radix=2, residence="gm", layout="planar-soa",
                  block=1024, fused_tile=1)
    v = check_lowering(c, p)
    assert v.legal, v.reasons
    assert v.status == LoweringStatus.LEGAL


def test_fused_tile_candidate_is_legal():
    p = DeviceProfile.load()
    c = Candidate(flow_id="local-fused-tile", processing_unit="radix-2",
                  radix=2, residence="gm", layout="planar-soa",
                  block=1024, fused_tile=64)
    v = check_lowering(c, p)
    assert v.legal, v.reasons


def test_on_chip_residence_is_not_lowered():
    """Multi-stage on-chip residence must be rejected, not silently scored."""
    p = DeviceProfile.load()
    c = Candidate(flow_id="baseline-global-stage", processing_unit="radix-2",
                  radix=2, residence="shared", layout="planar-soa",
                  block=1024, fused_tile=1)
    v = check_lowering(c, p)
    assert not v.legal
    assert v.status == LoweringStatus.NOT_LOWERED
    assert any("residence" in r for r in v.reasons)


def test_radix4_is_not_lowered():
    p = DeviceProfile.load()
    c = Candidate(flow_id="baseline-global-stage", processing_unit="radix-4",
                  radix=4, residence="gm", layout="planar-soa",
                  block=1024, fused_tile=1)
    v = check_lowering(c, p)
    assert not v.legal
    assert any("processing-unit" in r for r in v.reasons)


def test_batch_expansion_is_not_lowered():
    p = DeviceProfile.load()
    c = Candidate(flow_id="baseline-global-stage", processing_unit="radix-2",
                  radix=2, residence="gm", layout="planar-soa",
                  block=1024, fused_tile=1, ud=4)
    v = check_lowering(c, p)
    assert not v.legal


def test_non_power_of_two_block_rejected():
    p = DeviceProfile.load()
    c = Candidate(flow_id="baseline-global-stage", processing_unit="radix-2",
                  radix=2, residence="gm", layout="planar-soa",
                  block=1000, fused_tile=1)
    v = check_lowering(c, p)
    assert not v.legal


def test_enumeration_returns_legal_and_rejected():
    ds = DesignSpace.load()
    p = DeviceProfile.load()
    legal, rejected = enumerate_candidates(ds, p, block=1024)
    assert legal, "at least the M0 baseline must be legal"
    # conservative default pick is the lowest fused tile (M0)
    sel = selected_candidate(legal)
    assert sel is not None
    assert sel.candidate.fused_tile == 1
    assert sel.candidate.flow_id == "baseline-global-stage"
