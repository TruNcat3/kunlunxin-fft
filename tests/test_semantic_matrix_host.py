"""WP2 semantic-matrix aggregation + snapshot contract (host-only).

Issue #6 PR-D: win/tie/loss, geomean, selector regret, vendor-error
preservation, manifest contract and snapshot immutability — all pure
host logic over synthetic rows.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_semantic_matrix as sm  # noqa: E402
from src.xpu.manifest_contract import (  # noqa: E402
    validate_benchmark_output,
    validate_manifest,
)


def _row(label, candidate, tier, api_ms=None, status="ok", **kw):
    return {
        "session": kw.get("session", 0),
        "label": label, "trial": kw.get("trial", 0),
        "candidate": candidate, "tier": tier,
        "api_ms": api_ms, "correct": status == "ok",
        "err": kw.get("err", 1e-8), "status": status,
        "protocol_id": sm.PROTOCOL_ID,
        "profile_hash": "p" * 16,
        "timed_scope": sm.TIMING_SCOPE,
        "reason": kw.get("reason"),
        "params": {"selection_source": kw.get("selection_source",
                                              "table")},
    }


def _write_raw(tmp_path, rows, sessions=1):
    for s in range(sessions):
        path = tmp_path / f"raw_session{s}.jsonl"
        with open(path, "w") as f:
            for r in rows:
                if r["session"] == s:
                    f.write(json.dumps(r) + "\n")
    return tmp_path


def _cases():
    return [(64, 1), (1024, 16)]


def _sem_rows(label, sel, ven, inc, oracle, vendor_status="ok"):
    rows = []
    for cand, ms in (("selector", sel), ("incumbent", inc),
                     ("vendor", ven), ("oracle:m0_dit", oracle)):
        if cand == "vendor" and vendor_status != "ok":
            rows.append(_row(label, cand, "correctness", None,
                             status=vendor_status,
                             reason="XDNN_UNIMPLEMENTED"))
            continue
        for i in range(3):
            rows.append(_row(label, cand, "steady", ms, trial=i))
    return rows


def test_aggregate_win_tie_loss_and_metrics(tmp_path):
    rows = []
    # forward: selector beats vendor by 3x -> win
    rows += _sem_rows("N64B1:c2c-forward", 0.3, 0.9, 1.2, 0.28)
    # inverse: near parity -> tie
    rows += _sem_rows("N64B1:c2c-inverse", 0.50, 0.50, 0.8, 0.45)
    # r2c: selector slower -> loss
    rows += _sem_rows("N64B1:r2c", 1.0, 0.5, 1.5, 0.6)
    # c2r: vendor unsupported -> excluded from geomean
    rows += _sem_rows("N1024B16:c2r", 0.4, None, 0.9, 0.35,
                      vendor_status="vendor-unsupported")
    _write_raw(tmp_path, rows)
    summary = sm._aggregate(tmp_path, _cases(), 3, 1, 1)
    m = summary["metrics"]
    assert m["win"] == 1
    assert m["tie"] == 1
    assert m["loss"] == 1
    assert m["vendor_supported_cases"] == 3  # c2r excluded
    import math
    expect = math.exp((math.log(3.0) + math.log(1.0)
                       + math.log(0.5)) / 3)
    assert abs(m["geomean_speedup_vs_vendor"] - expect) < 1e-9
    assert abs(m["worst_speedup_vs_vendor"] - 0.5) < 1e-9
    # regret: sel/oracle-1 per case
    reg = [c["selector_regret"] for c in summary["cases"]
           if c["semantic"] == "c2c-forward"]
    assert abs(reg[0] - (0.3 / 0.28 - 1)) < 1e-9
    # vendor failure preserved with original reason
    assert any(f["kind"] == "vendor-unsupported"
               and "XDNN" in f["reason"]
               for f in summary["failures"])


def test_guard_flag_vs_incumbent(tmp_path):
    rows = []
    # selector beats incumbent by >2% (guard pass) on forward
    rows += _sem_rows("N64B1:c2c-forward", 0.90, 2.0, 1.0, 0.88)
    # selector loses to incumbent on inverse (guard fail)
    rows += _sem_rows("N64B1:c2c-inverse", 1.05, 2.0, 1.0, 0.9)
    _write_raw(tmp_path, rows)
    summary = sm._aggregate(tmp_path, _cases(), 3, 1, 1)
    by = {c["label"]: c for c in summary["cases"]}
    assert by["N64B1:c2c-forward"]["guard_pass_2pct"] is True
    assert by["N64B1:c2c-inverse"]["guard_pass_2pct"] is False


def test_tiers_never_conflated(tmp_path):
    rows = _sem_rows("N64B1:c2c-forward", 0.3, 0.9, 1.2, 0.28)
    rows.append(_row("N64B1:c2c-forward", "selector:one_shot",
                     "one_shot", 0.8))
    rows.append(_row("N64B1:c2c-forward", "selector:e2e", "e2e", 1.5))
    rows.append(_row("N64B1:c2c-forward", "selector:plan_creation",
                     "plan_creation", 0.05))
    _write_raw(tmp_path, rows)
    summary = sm._aggregate(tmp_path, _cases(), 3, 1, 1)
    # steady selector median must ignore one_shot/e2e rows
    by = {c["label"]: c for c in summary["cases"]}
    assert by["N64B1:c2c-forward"]["selector_ms"] == 0.3


def test_manifest_fields_validate(tmp_path):
    man = {
        "commit": "a" * 40, "dirty": False, "device": "d",
        "driver": "515.58", "profile_hash": "p" * 16,
        "env_hash": "e" * 16, "toolchain": "python",
        "protocol_id": sm.PROTOCOL_ID,
        "timing_scope": sm.TIMING_SCOPE,
        "selection_semantics": "correctness-first",
        "warmup": 5, "trials": 30, "sessions": 3, "seed_base": 1234,
        "candidates": ["incumbent", "selector", "oracle:*", "vendor"],
        "shapes": ["N64B1:c2c-forward"], "error_threshold": 1e-4,
        "repro_command": "python scripts/run_semantic_matrix.py",
        "gm_bytes_kind": "analytical estimate only",
        "kernel_time_status": "not collected",
        "source_hash": "b" * 40,
    }
    assert validate_manifest(man, strict=False) == []


def test_snapshot_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "ROOT", tmp_path)
    # ROOT is used as snap parent: results/p800/releases under ROOT
    target = tmp_path / "results" / "p800" / "releases" / "rid1"
    target.mkdir(parents=True)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(SystemExit, match="immutable"):
        sm._write_snapshot(out_dir, "rid1", {}, {}, None, [])


def test_semantic_direction_mapping():
    assert sm._semantic_of("c2c-forward") == ("c2c", "forward")
    assert sm._semantic_of("c2c-inverse") == ("c2c", "inverse")
    assert sm._semantic_of("r2c") == ("r2c", "forward")
    assert sm._semantic_of("c2r") == ("c2r", "inverse")


def test_oracle_plan_names_bounded():
    plans = sm._oracle_plans(64, 1, "dit", "c2c-forward", block=1024)
    names = [p.params["oracle_name"] for p in plans]
    assert names == ["m0_dit", "m0_dif", "m1_fused_t64", "m2_dit_t64",
                     "m2_dif_t64", "m5_r4_dit"]
    plans = sm._oracle_plans(64, 16, "dit", "c2c-inverse", block=1024)
    names = [p.params["oracle_name"] for p in plans]
    assert "m4_spread" in names and "m1_fused_t64" not in names
    # semantics carried on every oracle plan
    for p in plans:
        assert p.transform_type == "c2c"
        assert p.transform_direction == "inverse"


def test_snapshot_manifest_passes_strict_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "ROOT", tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    with open(out / "raw_session0.jsonl", "w") as f:
        f.write(json.dumps(_row("N64B1:c2c-forward", "selector",
                                "steady", 0.1)) + "\n")
    man = {
        "commit": "a" * 40, "dirty": False, "device": "d",
        "driver": "515.58", "profile_hash": "p" * 16,
        "env_hash": "e" * 16, "toolchain": "python",
        "protocol_id": sm.PROTOCOL_ID,
        "timing_scope": sm.TIMING_SCOPE,
        "selection_semantics": "correctness-first",
        "warmup": 5, "trials": 30, "sessions": 1, "seed_base": 1234,
        "candidates": ["incumbent", "selector", "oracle:*", "vendor"],
        "shapes": ["N64B1:c2c-forward"], "error_threshold": 1e-4,
        "repro_command": "python scripts/run_semantic_matrix.py",
        "gm_bytes_kind": "analytical estimate only",
        "kernel_time_status": "not collected",
        "source_hash": "b" * 40,
    }
    # mirror main(): aggregate + stamp both manifests, then the writer
    # strict-validates the finished snapshot (issue #8 PR-J A2)
    summary = sm._aggregate(out, [(64, 1)], 3, 5, 1, manifest=man)
    res = sm.result_sha256(out)
    man["result_sha256"] = res
    summary["manifest"]["result_sha256"] = res
    man["cases_sha256"] = summary["manifest"]["cases_sha256"]
    (out / "manifest.json").write_text(json.dumps(man))
    (out / "summary.json").write_text(json.dumps(summary))
    snap = sm._write_snapshot(out, "rid2", man, {}, None, ["N64B1"])
    stamped = json.loads((snap / "manifest.json").read_text())
    assert validate_manifest(stamped, strict=True) == []
    assert stamped["result_sha256"] == sm.result_sha256(out)
    emb = json.loads((snap / "summary.json").read_text())["manifest"]
    assert stamped["cases_sha256"] == emb["cases_sha256"]
    assert validate_benchmark_output(snap, strict=True) == []
    # snapshot raw is gzip'd but byte-identical after decompression
    import gzip as _gz
    with _gz.open(snap / "raw_session0.jsonl.gz", "rb") as f:
        assert f.read() == (out / "raw_session0.jsonl").read_bytes()
    assert len(stamped["result_sha256"]) == 64


def test_dispersion_sessions_and_figure(tmp_path):
    rows = _sem_rows("N64B1:c2c-forward", 0.3, 0.9, 1.2, 0.28)
    rows += [_row("N64B1:c2c-forward", "selector", "steady", 0.4,
                  trial=3, session=1)]
    rows += [_row("N64B1:c2c-forward", "selector", "steady", 0.5,
                  trial=3, session=2)]
    _write_raw(tmp_path, rows, sessions=3)
    summary = sm._aggregate(tmp_path, _cases(), 3, 1, 3)
    c = [x for x in summary["cases"]
         if x["label"] == "N64B1:c2c-forward"][0]
    # MAD reported per contender (session0 selector: [0.3]*3)
    assert c["selector_mad_ms"] is not None
    assert c["vendor_mad_ms"] is not None
    # sessions_with_data counts distinct sessions, not trial rows
    assert c["sessions_with_data"] == 3
    # figure driven by the same summary
    fig = tmp_path / "speedup_vs_vendor.png"
    assert fig.is_file() and fig.stat().st_size > 1000


def test_oracle_plan_attribution_is_single_sourced():
    """issue #8: matrix winner rows must describe the lowering that ran
    (builder entries bind to this attribution, never re-derive it)."""
    assert sm._attribution("fft_dit") == (
        "baseline-global-stage", "global-stage", "radix2-scalar")
    assert sm._attribution("fft_dit_batch", {"variant": "spread"}) == (
        "batch-spread", "global-stage", "radix2-scalar")
    assert sm._attribution("fft_dit_batch", {"variant": "folded"}) == (
        "batch-folded", "global-stage", "radix2-scalar")
    assert sm._attribution("fft_dit_matrix_batch",
                           {"tile": 64, "rows": 8}) == (
        "local-fused-tile", "matrix-resident", "dense-dot-fft64")
    # batched radix-4 joins the batch flow family (issue #8 PR-I:
    # spread is the only lowerable radix4 batch variant; bare plans
    # fail closed rather than guessing an attribution).
    assert sm._attribution("fft_dit_radix4_batch",
                           {"variant": "spread"}) == (
        "batch-spread", "global-stage", "radix4-scalar")
    assert sm._attribution("fft_dit_radix4_batch") == (None, None, None)
    assert sm._attribution("fft_dit_fused", {"tile": 64}) == (
        "local-fused-tile", "fused-inplace", "radix2-scalar")
    # unknown executor never guesses attribution (fail closed upstream)
    assert sm._attribution("run_hybrid") == (None, None, None)


def test_oracle_plans_carry_attribution_and_executed():
    plans = sm._oracle_plans(1024, 256, "dit", "c2c-forward", block=1024)
    assert plans
    for plan in plans:
        assert plan.flow_id and plan.flow_realization and plan.processing_unit
        assert plan.executed
        assert plan.params["oracle_name"]
    names = {p.params["oracle_name"] for p in plans}
    assert names == {"m0_dit", "m4_spread", "m4_folded", "m2_dit_t64",
                     "m2_dif_t64", "m5_r4_dit"}
    flows = {p.flow_id for p in plans}
    assert flows <= {"batch-spread", "batch-folded",
                     "local-fused-tile", "baseline-global-stage"}


def test_scope_cells_all_dedupes_shared_shapes():
    """issue #8: ``--scope all`` must schedule each cell exactly once.

    main and apps both list the OFDM/radar proxy shapes; scheduling them
    twice emits duplicate raw trial keys and the strict manifest gate
    rejects the run (found by the post-promote full-domain run).
    """
    cells = sm._scope_cells("all")
    assert len(cells) == len(set(cells))
    # order-preserving first occurrence wins for the shared shapes
    shared = [(1024, 16), (1024, 64), (2048, 16), (2048, 64)]
    for shape in shared:
        assert cells.count(shape) == 1
        assert cells.index(shape) == (
            sm.SCOPES["main"] + sm.SCOPES["apps"]
            + sm.SCOPES["large"]).index(shape)
    # main + apps + large counted with overlap removed: 24+6+6-4
    assert len(cells) == 32
    # every scope schedules unique cells too
    for scope in sm.SCOPES:
        sc = sm._scope_cells(scope)
        assert len(sc) == len(set(sc))


def test_aggregate_stamp_and_strict_gate(tmp_path):
    """Issue #8 P1-C: a matrix run output passes the strict gate only
    when manifest.json and summary.json both carry result_sha256 bound
    with manifest_contract.result_sha256 over the raw sessions."""
    rows = _sem_rows("N64B1:c2c-forward", 0.3, 0.9, 1.2, 0.28)
    _write_raw(tmp_path, rows)
    man = {
        "commit": "a" * 40, "dirty": False, "device": "d",
        "driver": "515.58", "profile_hash": "p" * 16,
        "env_hash": "e" * 16, "toolchain": "python",
        "protocol_id": sm.PROTOCOL_ID,
        "timing_scope": sm.TIMING_SCOPE,
        "selection_semantics": "correctness-first",
        "warmup": 1, "trials": 3, "sessions": 1, "seed_base": 1234,
        "candidates": ["incumbent", "selector", "oracle:*", "vendor"],
        "shapes": ["N64B1:c2c-forward"], "error_threshold": 1e-4,
        "repro_command": "python scripts/run_semantic_matrix.py",
        "gm_bytes_kind": "analytical estimate only",
        "kernel_time_status": "not collected",
        "source_hash": "b" * 40,
    }
    summary = sm._aggregate(tmp_path, [(64, 1)], 3, 1, 1, manifest=man)
    # what run_semantic_matrix.main stamps after aggregation
    res = sm.result_sha256(tmp_path)
    man["result_sha256"] = res
    summary["manifest"]["result_sha256"] = res
    (tmp_path / "manifest.json").write_text(
        json.dumps(man, indent=2, sort_keys=True) + "\n")
    (tmp_path / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    assert validate_manifest(
        json.loads((tmp_path / "manifest.json").read_text()),
        strict=True) == []
    assert validate_benchmark_output(tmp_path, strict=True) == []
    # without the stamp the strict gate must fail (the bug this guards)
    summary["manifest"].pop("result_sha256")
    (tmp_path / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    assert any("result_sha256" in f
               for f in validate_benchmark_output(tmp_path, strict=True))


def _fake_winner_plan(block=1024, **params):
    from src.xpu.entry import ExecutionPlan
    return ExecutionPlan(
        n=4096, batch=16, direction="dit", precision="fp32",
        selection_source="table", reason="t", fallback=False,
        executed="fft_dit_radix4_batch", block=block,
        flow_id="batch-spread", flow_realization="global-stage",
        processing_unit="radix4-scalar",
        params={"variant": "spread", **params},
        transform_type="c2c", transform_direction="forward",
        norm="backward")


def test_winner_block_oracle_mode_sweeps_only_block():
    """issue #8 PR-I round 2: winner-block mode fixes the promoted
    winner (flow/unit/params carried verbatim) and varies ONLY the
    block over the fixed sweep -- a single-factor A/B."""
    base = _fake_winner_plan(block=1024, batch=16)
    plans = sm._winner_block_plans(base)
    assert [p.block for p in plans] == [256, 512, 1024, 2048]
    assert [p.params["oracle_name"] for p in plans] == [
        "winner-b256", "winner-b512", "winner-b1024", "winner-b2048"]
    for p in plans:
        assert p.executed == base.executed
        assert p.flow_id == base.flow_id
        assert p.flow_realization == base.flow_realization
        assert p.processing_unit == base.processing_unit
        assert p.params["variant"] == "spread"
    # a deployed block outside the sweep stays part of the A/B, so
    # the round can only tie or improve the deployed geometry
    outside = sm._winner_block_plans(_fake_winner_plan(block=4096))
    assert sorted(p.block for p in outside) == [256, 512, 1024, 2048, 4096]


def test_oracle_mode_defaults_and_calibrate_passthrough():
    import bootstrap_profile as bp
    assert sm.ORACLE_MODE == "full"
    cmds = bp.calibrate_commands(Path("/tmp/x"), "0", "pri")
    matrix = [c for c in cmds
              if any(str(x).endswith("run_semantic_matrix.py")
                     for x in c)][0]
    assert matrix[matrix.index("--oracle-mode") + 1] == "full"
    cmds = bp.calibrate_commands(Path("/tmp/x"), "0", "pri", "winner-block")
    matrix = [c for c in cmds
              if any(str(x).endswith("run_semantic_matrix.py")
                     for x in c)][0]
    assert matrix[matrix.index("--oracle-mode") + 1] == "winner-block"
    assert "--scope" in matrix and "pri" in matrix


def test_winner_block_plans_tolerate_composition_block_none():
    """composition plans carry block=None (the composed core picks its
    geometry); the fixed sweep must still build and measure."""
    plans = sm._winner_block_plans(_fake_winner_plan(block=None))
    assert [p.block for p in plans] == [256, 512, 1024, 2048]
    assert [p.params["oracle_name"] for p in plans] == [
        "winner-b256", "winner-b512", "winner-b1024", "winner-b2048"]


def test_previous_deployed_column_and_dispositions(tmp_path):
    """Issue #8 PR-J A1: same-session previous-deployed comparison."""
    rows = []
    for lbl, prev_ms, sel_ms in (
            ("N64B1:c2c-forward", 0.4, 0.4),   # parity
            ("N64B1:c2c-inverse", 0.4, 0.5),   # +25% -> regression
    ):
        rows += _sem_rows(lbl, sel_ms, 2.0, 1.2, sel_ms * 0.9)
        for i in range(3):
            rows.append(_row(lbl, "previous-deployed", "steady", prev_ms,
                             trial=i, selection_source="table"))
    _write_raw(tmp_path, rows)
    summary = sm._aggregate(tmp_path, _cases(), 3, 1, 1)
    by = {c["label"]: c for c in summary["cases"]}
    fwd = by["N64B1:c2c-forward"]
    assert fwd["previous_deployed_ms"] == 0.4
    assert abs(fwd["selector_vs_previous"]) < 1e-9
    assert fwd["previous_deployed_source"] == "table"
    assert fwd["regression_reason"] is None
    inv = by["N64B1:c2c-inverse"]
    assert abs(inv["selector_vs_previous"] - 0.25) < 1e-9
    assert "slower than legal previous" in inv["regression_reason"]
    m = summary["metrics"]
    assert m["previous_cells_measured"] == 2
    assert m["previous_regressions_gt_2pct"] == ["N64B1:c2c-inverse"]
    # full disposition table written for every cell
    import csv as _csv
    with open(tmp_path / "regression_vs_previous.csv") as f:
        out = list(_csv.reader(f))
    assert out[0][:3] == ["label", "semantic", "N"]
    assert len(out) == 1 + 8  # 2 shapes x 4 semantics + header
    # pre-registered intentional semantic-correction cells
    reason = sm._previous_regression_reason(
        "N2048B16", "c2c-inverse", {"selector_vs_previous": None})
    assert "intentional semantic correction" in reason
    assert sm._previous_regression_reason(
        "N64B1", "r2c", {"selector_vs_previous": 0.01}) is None


def test_previous_selector_source_is_pinned_commit():
    """A1: previous table comes from the fixed pre-issue commit, and
    its bytes hash to the reviewed digest (never the working tree)."""
    import hashlib
    import subprocess
    raw = subprocess.check_output(
        ["git", "show",
         f"{sm.PREVIOUS_SELECTOR_COMMIT}:config/selector_table.json"],
        cwd=ROOT)
    assert hashlib.sha256(raw).hexdigest() == (
        "acc1f21c9c93fe20589993f22589ccc24e5b08db05be29d63321d9b63a6b5449")
    assert sm.PREVIOUS_SELECTOR_COMMIT.startswith("54260cf")
