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
    arithmetic_cost,
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
    assert ds.radix == (2, 4)
    assert ds.fused_tile_candidates  # M1 axis present and consumed
    assert all((t & (t - 1)) == 0 for t in ds.fused_tile_candidates)
    assert 1 in ds.fused_tile_candidates  # M0 baseline must be an option
    # M4 batch axis present and consumed by dit/dif_stage_batch_kernel
    assert ds.batch_candidates and 1 in ds.batch_candidates
    assert all((t & (t - 1)) == 0 for t in ds.batch_candidates)
    assert set(ds.batch_variants) <= {"spread", "folded"}


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
    c = Candidate(flow_id="baseline-global-stage", processing_unit="radix2-scalar",
                  radix=2, boundary_residence="gm", layout="planar-soa",
                  block=1024, fused_tile=1)
    v = check_lowering(c, p)
    assert v.legal, v.reasons
    assert v.status == LoweringStatus.LEGAL


def test_fused_tile_candidate_is_legal():
    p = DeviceProfile.load()
    c = Candidate(flow_id="local-fused-tile", processing_unit="radix2-scalar",
                  radix=2, boundary_residence="gm", layout="planar-soa",
                  block=1024, fused_tile=64)
    v = check_lowering(c, p)
    assert v.legal, v.reasons


def test_on_chip_residence_is_not_lowered():
    """Multi-stage on-chip residence must be rejected, not silently scored."""
    p = DeviceProfile.load()
    c = Candidate(flow_id="baseline-global-stage", processing_unit="radix2-scalar",
                  radix=2, boundary_residence="shared", layout="planar-soa",
                  block=1024, fused_tile=1)
    v = check_lowering(c, p)
    assert not v.legal
    assert v.status == LoweringStatus.NOT_LOWERED
    assert any("residence" in r for r in v.reasons)


def test_radix4_is_legal_on_baseline_flow():
    """M5: radix-4 lowers on the same flow as radix-2; unit/radix must agree."""
    p = DeviceProfile.load()
    c = Candidate(flow_id="baseline-global-stage", processing_unit="radix4-scalar",
                  radix=4, boundary_residence="gm", layout="planar-soa",
                  block=1024, fused_tile=1)
    v = check_lowering(c, p)
    assert v.legal, v.reasons
    # unit/radix mismatch is still rejected
    mismatch = Candidate(flow_id="baseline-global-stage",
                         processing_unit="radix4-scalar", radix=2,
                         boundary_residence="gm", layout="planar-soa",
                         block=1024, fused_tile=1)
    v = check_lowering(mismatch, p)
    assert not v.legal
    assert any("processing-unit" in r for r in v.reasons)
    # unimplemented cores stay not-lowered
    other = Candidate(flow_id="baseline-global-stage",
                      processing_unit="radix-8", radix=8,
                      boundary_residence="gm", layout="planar-soa",
                      block=1024, fused_tile=1)
    v = check_lowering(other, p)
    assert not v.legal
    # M1 fusion remains radix-2-only (enumeration-level, see below)
    fused_r4 = Candidate(flow_id="local-fused-tile",
                         processing_unit="radix4-scalar", radix=4,
                         boundary_residence="gm", layout="planar-soa",
                         block=1024, fused_tile=64)
    assert check_lowering(fused_r4, p).legal  # contract allows tile>1...
    # ...but enumeration must not pair radix-4 with the fused flow
    ds = DesignSpace.load()
    legal, rejected = enumerate_candidates(ds, p, block=1024)
    r4_flows = {v.candidate.flow_id for v in legal
                if v.candidate.processing_unit == "radix4-scalar"}
    assert r4_flows == {"baseline-global-stage"}


def test_arithmetic_cost_radix4_vs_radix2():
    """M5: same flow, compare the arithmetic ledger (both directions)."""
    for n in (8, 64, 1024, 65536):
        r2 = arithmetic_cost(n, 2)
        r4 = arithmetic_cost(n, 4)
        logn = n.bit_length() - 1
        assert r2["launches"] == logn
        assert r4["launches"] == logn // 2 + logn % 2
        assert r4["launches"] < r2["launches"]
        assert r4["twiddle_mults"] < r2["twiddle_mults"]
        assert r4["real_adds"] > r2["real_adds"]  # reported, not hidden
        assert r4["butterflies"] < r2["butterflies"]
    with pytest.raises(ValueError):
        arithmetic_cost(12, 2)
    with pytest.raises(ValueError):
        arithmetic_cost(1024, 8)


def test_batch_expansion_is_not_lowered():
    p = DeviceProfile.load()
    c = Candidate(flow_id="baseline-global-stage", processing_unit="radix2-scalar",
                  radix=2, boundary_residence="gm", layout="planar-soa",
                  block=1024, fused_tile=1, ud=4)
    v = check_lowering(c, p)
    assert not v.legal


def test_non_power_of_two_block_rejected():
    p = DeviceProfile.load()
    c = Candidate(flow_id="baseline-global-stage", processing_unit="radix2-scalar",
                  radix=2, boundary_residence="gm", layout="planar-soa",
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


# --- M2 register-resident local matrices (host-only, numpy) ---


def test_matrices_module_does_not_import_torch():
    import src.xpu.matrices as matrices

    src = Path(matrices.__file__).read_text()
    assert "import torch" not in src
    assert "import triton" not in src


def test_local_matrix_verifies_against_scalar_loops():
    from src.xpu.matrices import verify_local_matrix

    for tile in (8, 16, 64, 128):
        for direction in ("dit", "dif"):
            err = verify_local_matrix(tile, direction)
            assert err < 1e-12, f"{direction} tile={tile}: err={err:.3e}"


def test_local_matrix_decomposition_matches_fft():
    """The local group + global tail must reproduce an N-dependent FFT.

    Pins the N-independence of the local twiddles against references whose
    twiddles do depend on N (np.fft.fft for DIT, full scalar DIF loops).
    """
    from src.xpu.matrices import verify_decomposition

    for n, tile in ((64, 64), (256, 64), (4096, 64), (4096, 8), (1024, 128)):
        for direction in ("dit", "dif"):
            rel = verify_decomposition(n, tile, direction)
            assert rel < 1e-12, f"{direction} n={n} tile={tile}: rel={rel:.3e}"


def test_local_matrix_shape_dtype_and_unitarity():
    import numpy as np

    from src.xpu.matrices import local_matrix, local_matrix_planar

    for tile in (8, 64):
        for direction in ("dit", "dif"):
            m = local_matrix(tile, direction)
            assert m.shape == (tile, tile)
            assert m.dtype == np.complex128
            # each butterfly stage is orthogonal up to scaling: after
            # log2(tile) stages the columns have norm sqrt(tile) and are
            # mutually orthogonal (M^H M = tile * I)
            gram = m.conj().T @ m
            assert np.allclose(gram, tile * np.eye(tile), atol=1e-10)
            mr, mi = local_matrix_planar(tile, direction)
            assert mr.shape == (tile, tile)
            assert mr.dtype == np.float32
            assert mr.flags["C_CONTIGUOUS"] and mi.flags["C_CONTIGUOUS"]


def test_tail_group_table_matches_scalar_stages():
    """M3 group coefficient table == independent per-stage scalar loops.

    Pins the N-independence of the group map (coefficients depend only on
    (n0, g, direction)) for both directions across sizes and group depths.
    """
    from src.xpu.matrices import verify_tail_group

    for n0 in (1, 2, 4, 8, 16, 64):
        for glog in (1, 2, 3):
            for direction in ("dit", "dif"):
                err = verify_tail_group(n0, glog, direction)
                assert err < 1e-10, (
                    f"tail group {direction} n0={n0} glog={glog}: err={err:.3e}"
                )


def test_grouped_decomposition_matches_fft():
    """Grouped tail + M2 local group == np.fft over the whole array.

    End-to-end host model of the M3 flow for both directions: verifies that
    the execution-group partition is a valid refactorisation of the stage
    sequence when the tail is exactly one group (n == tile << glog).
    """
    from src.xpu.matrices import verify_grouped_decomposition

    for tile, glog in ((8, 1), (8, 2), (8, 3), (16, 2), (64, 2), (64, 3)):
        n = tile << glog
        for direction in ("dit", "dif"):
            rel = verify_grouped_decomposition(n, tile, glog, direction)
            assert rel < 1e-12, (
                f"grouped {direction} n={n} tile={tile} glog={glog}: rel={rel:.3e}"
            )


def test_tail_groups_partition_is_complete():
    """The partition covers every tail stage exactly once, groups first.

    Groups must be a prefix of ascending half-spans (n0 doubling), singles a
    suffix, each group period n0<<g within maxp, every stage >= tile covered.
    """
    from src.xpu.matrices import tail_groups

    for tile, logn, glog, maxp in (
        (8, 16, 2, 4096),
        (8, 16, 1, 4096),
        (64, 16, 3, 8192),
        (64, 16, 3, 128),
        (16, 20, 2, 1024),
        (64, 16, 2, 8192),
        (8, 4, 2, 4096),
        (64, 16, 4, 4096),
    ):
        groups, singles = tail_groups(tile, logn, glog, maxp)
        # every stage >= tile up to logn covered exactly once, ascending
        covered = sorted(
            [n0 << s for n0, g in groups for s in range(g)] + list(singles)
        )
        expect = [1 << s for s in range(tile.bit_length() - 1, logn)]
        assert covered == expect, f"tile={tile} logn={logn}: {covered} != {expect}"
        # groups form a prefix: periods ascending, no group after a single
        periods = [n0 << g for n0, g in groups]
        assert periods == sorted(periods), f"groups not ascending: {groups}"
        if singles and groups:
            last_stage = groups[-1][0] << (groups[-1][1] - 1)
            assert all(s > last_stage for s in singles), (
                f"single at/below last group stage: groups={groups} singles={singles}"
            )
        for n0, g in groups:
            assert n0 << g <= maxp, f"period over maxp: n0={n0} g={g} maxp={maxp}"
            assert n0 in expect, f"group n0={n0} not a tail stage: tile={tile}"
        # singles are single stages, i.e. the leftover run above the groups
        for s in singles:
            assert s >= tile and s < (1 << logn)


def test_dot_free_kernels_do_not_trigger_sdnn_flag():
    """Vendor `ASTSource.parse_options` substring-scans `fn.src` for "tl.dot".

    The scan sets ``is_sdnn=True`` (SDNN matrix-unit lowering path) purely on
    the presence of the *text* ``tl.dot`` anywhere in the function source,
    docstrings included; that path breaks dot-free kernels with
    'linalg.generic' op expects region #0 to have 0 or 1 blocks. Keep the
    substring only in kernels that actually call the builtin.
    """
    import ast

    path = Path(__file__).resolve().parents[1] / "src" / "xpu" / "kernels.py"
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        seg = ast.get_source_segment(path.read_text(), node) or ""
        if "tl.dot" in seg or "tt.dot" in seg:
            assert node.name == "local_matrix_kernel", (
                f"{node.name} source contains a dot substring but does not call it"
            )


def test_tail_group_kernel_unrolls_k_loop():
    """The k loop must be ``tl.static_range``, never ``range``.

    The vendor TritonXPUUnrollControl pass (UnrollControl.cpp:79) exits the
    process with status 1 -- a native exit Python cannot catch, so a live
    canary would kill the whole test session -- whenever a ``range()`` loop
    coexists with BLOCK >= 256 in a kernel like this one. Compile-time
    unrolling removes the scf.for the pass chokes on. Guard the source text.
    """
    import ast

    path = Path(__file__).resolve().parents[1] / "src" / "xpu" / "kernels.py"
    text = path.read_text()
    tree = ast.parse(text)
    node = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "tail_group_kernel"
    )
    seg = ast.get_source_segment(text, node)
    assert "tl.static_range(1, NDEP)" in seg, "k loop lost its static_range"
    assert "for k in range" not in seg, "k loop regressed to range()"


# --- M4 batch expansion (host-only candidate contract) ---


def test_batch_expansion_lowers_with_variant():
    """spread (Ud=B) and folded (Td=B) are the two lowered batch points."""
    p = DeviceProfile.load()
    spread = Candidate(flow_id="batch-spread", processing_unit="radix2-scalar",
                       radix=2, boundary_residence="gm", layout="planar-soa",
                       block=1024, fused_tile=1, batch=4, batch_variant="spread",
                       ud=4, td=1)
    folded = Candidate(flow_id="batch-folded", processing_unit="radix2-scalar",
                       radix=2, boundary_residence="gm", layout="planar-soa",
                       block=1024, fused_tile=1, batch=4, batch_variant="folded",
                       ud=1, td=4)
    for c in (spread, folded):
        v = check_lowering(c, p)
        assert v.legal, v.reasons


def test_batch_variant_axis_mismatch_rejected():
    """Ud/Td, flow, batch, and fused_tile must all agree with the variant."""
    p = DeviceProfile.load()

    def cand(**kw):
        base = dict(flow_id="batch-spread", processing_unit="radix2-scalar", radix=2,
                    boundary_residence="gm", layout="planar-soa", block=1024,
                    fused_tile=1, batch=4, batch_variant="spread", ud=4, td=1)
        base.update(kw)
        return Candidate(**base)

    bad = [
        cand(td=4),                        # spread must keep Td=1
        cand(ud=1),                        # spread requires Ud=batch
        cand(batch=3),                     # batch must be a power of two
        cand(fused_tile=64),               # batched path is M0-stage-only
        cand(flow_id="batch-folded"),      # flow must match variant
        cand(batch_variant="none"),        # variant removed but Ud kept
        cand(flow_id="batch-spread", batch_variant="folded", ud=1, td=4),
        Candidate(flow_id="baseline-global-stage", processing_unit="radix2-scalar",
                  radix=2, boundary_residence="gm", layout="planar-soa", block=1024,
                  fused_tile=1, ud=8),     # expansion without any variant
    ]
    for c in bad:
        v = check_lowering(c, p)
        assert not v.legal, f"expected rejection: {c}"
        assert v.status == LoweringStatus.NOT_LOWERED


def test_enumeration_includes_batch_points():
    """The declared batch axis must surface as legal candidates, and the
    conservative selector must still default to the M0 baseline."""
    ds = DesignSpace.load()
    p = DeviceProfile.load()
    legal, rejected = enumerate_candidates(ds, p, block=1024)
    batch_flows = {v.candidate.flow_id for v in legal
                   if v.candidate.flow_id.startswith("batch-")}
    assert "batch-spread" in batch_flows and "batch-folded" in batch_flows
    batches = {v.candidate.batch for v in legal
               if v.candidate.flow_id.startswith("batch-")}
    assert batches == set(ds.batch_candidates)
    sel = selected_candidate(legal)
    assert sel.candidate.flow_id == "baseline-global-stage"
    assert sel.candidate.batch_variant == "none"
    assert sel.candidate.processing_unit == "radix2-scalar"  # incumbent stays radix-2


# --- schema v2 attribution (flow_realization / local_residence) ---


def test_enumeration_registers_m2_and_m3():
    """M2/M3 must surface as enumerated legal candidates, not ad-hoc points."""
    ds = DesignSpace.load()
    p = DeviceProfile.load()
    legal, _ = enumerate_candidates(ds, p, block=1024)
    realizations = {v.candidate.to_dict()["flow_realization"] for v in legal}
    assert {"global-stage", "fused-inplace", "matrix-resident",
            "grouped-boundary"} <= realizations
    units = {v.candidate.processing_unit for v in legal}
    assert {"radix2-scalar", "radix4-scalar", "dense-dot-fft64",
            "composite-linear-map"} <= units


def test_schema_v2_attribution_fields():
    """to_dict carries the split attribution axes with evidenced values."""
    p = DeviceProfile.load()
    cases = [
        (Candidate(flow_id="baseline-global-stage",
                   processing_unit="radix2-scalar", radix=2,
                   boundary_residence="gm", layout="planar-soa",
                   block=1024, fused_tile=1),
         "global-stage", "none"),
        (Candidate(flow_id="local-fused-tile",
                   processing_unit="radix2-scalar", radix=2,
                   boundary_residence="gm", layout="planar-soa",
                   block=1024, fused_tile=64),
         "fused-inplace", "cache"),
        (Candidate(flow_id="local-fused-tile",
                   processing_unit="dense-dot-fft64", radix=2,
                   boundary_residence="gm", layout="planar-soa",
                   block=1024, fused_tile=64, rows=8,
                   flow_realization="matrix-resident"),
         "matrix-resident", "register+implicit-dot"),
        (Candidate(flow_id="local-fused-tile",
                   processing_unit="composite-linear-map", radix=2,
                   boundary_residence="gm", layout="planar-soa",
                   block=1024, fused_tile=8, rows=8, glog=2, maxp=4096,
                   cells=1, flow_realization="grouped-boundary"),
         "grouped-boundary", "register"),
    ]
    for cand, realization, local_res in cases:
        v = check_lowering(cand, p)
        assert v.legal, v.reasons
        d = cand.to_dict()
        assert d["flow_realization"] == realization
        assert d["local_residence"] == local_res
        assert d["boundary_residence"] == "gm"


def test_local_residence_fabrication_rejected():
    """A flow may only claim residency the realization evidences."""
    p = DeviceProfile.load()
    c = Candidate(flow_id="local-fused-tile", processing_unit="radix2-scalar",
                  radix=2, boundary_residence="gm", layout="planar-soa",
                  block=1024, fused_tile=64, local_residence="register")
    v = check_lowering(c, p)
    assert not v.legal
    assert any("local_residence" in r for r in v.reasons)


def test_report_tail_groups_matches_matrices():
    """report._tail_groups must equal matrices.tail_groups on the grid.

    The report module clones the greedy partition to stay numpy-free; the
    clone and the original must never diverge.
    """
    import sys as _sys

    from src.xpu.matrices import tail_groups

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    _sys.path.insert(0, str(scripts))
    try:
        import report as report_mod
    finally:
        _sys.path.pop(0)
    for tile, logn, glog, maxp in (
        (8, 16, 2, 4096),
        (8, 16, 1, 4096),
        (64, 16, 3, 8192),
        (64, 16, 3, 128),
        (16, 20, 2, 1024),
        (64, 16, 2, 8192),
        (8, 4, 2, 4096),
        (64, 16, 4, 4096),
    ):
        assert report_mod._tail_groups(tile, logn, glog, maxp) == \
            tail_groups(tile, logn, glog, maxp), (tile, logn, glog, maxp)
