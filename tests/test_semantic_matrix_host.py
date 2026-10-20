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
    # complete steady evidence: manifest declares trials=30, so the
    # strict gate's trial census (issue #8 review4 P2-a) requires all
    # 30 timed rows for the group it sees
    with open(out / "raw_session0.jsonl", "w") as f:
        for trial in range(30):
            f.write(json.dumps(_row("N64B1:c2c-forward", "selector",
                                    "steady", 0.1, trial=trial)) + "\n")
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


# ------------- P2 aggregate replay over gzip snapshots (review4 一.2) -----

def test_aggregate_replays_released_gzip_snapshot(tmp_path):
    """--aggregate-only reads .jsonl.gz through the shared reader and
    regenerates the released analysis byte-for-byte (issue #8 review4
    P2-c). Runs against the committed v4 release in a temp copy --
    immutable evidence is never touched."""
    import shutil

    from src.xpu.manifest_contract import iter_raw_rows

    release = (ROOT / "results" / "p800" / "releases" /
               "20261019-wp2-semantic-matrix-v4")
    if not (release / "summary.json").is_file():
        pytest.skip("released v4 snapshot not present")
    copy = tmp_path / "snapshot"
    shutil.copytree(release, copy)
    orig = json.loads((copy / "summary.json").read_text())
    man = orig["manifest"]
    pairs = []
    for c in orig["cases"]:
        if (c["N"], c["batch"]) not in pairs:
            pairs.append((c["N"], c["batch"]))
    regen = sm._aggregate(copy, pairs, man["trials"], man["warmup"],
                          man["sessions"], manifest=man)
    # the released analysis reproduces from the gzip rows alone
    # (v4 predates the Option A fallback column: compare the shared keys)
    strip = lambda cases: [  # noqa: E731
        {k: v for k, v in c.items() if k != "selector_previous_fallback"}
        for c in cases]
    assert strip(regen["cases"]) == strip(orig["cases"])
    assert all(c["selector_previous_fallback"] is False
               for c in regen["cases"])   # no fallbacks deployed in v4
    assert regen["metrics"] == orig["metrics"]
    assert regen["failures"] == orig["failures"]
    assert regen["raw_sha256"] == orig["raw_sha256"]
    # the shared-keys analysis hashes to the released cases_sha256
    # (the full regen hash differs only by the new fallback column)
    from src.xpu.manifest_contract import cases_sha256 as _cases_hash
    assert _cases_hash(strip(regen["cases"])) == \
        orig["manifest"]["cases_sha256"]
    assert (regen["manifest"]["cases_sha256"]
            == _cases_hash(regen["cases"]))
    # P2-b: the replay now also records the actual oracle candidates
    observed = sorted(
        {r["candidate"] for _, _, r in iter_raw_rows(copy)
         if isinstance(r.get("candidate"), str)
         and r["candidate"].startswith("oracle:")})
    assert observed, "v4 evidence must contain oracle candidates"
    assert regen["manifest"]["oracle_candidates"] == observed


# ------------- issue #9 WP0-B: declared candidate inventory --------------

def _rows_from_inventory(inv, profile):
    """Healthy evidence exactly matching a declared inventory."""
    rows = []
    for cell in inv["cells"]:
        for cand, tiers in cell["candidates"].items():
            for s in range(inv["sessions"]):
                for tier, count in tiers.items():
                    for i in range(count):
                        rows.append({
                            "session": s, "label": cell["label"],
                            "trial": i, "candidate": cand, "tier": tier,
                            "api_ms": 1.0, "correct": True, "err": 1e-8,
                            "status": "ok", "protocol_id": sm.PROTOCOL_ID,
                            "profile_hash": profile,
                            "timed_scope": sm.TIMING_SCOPE,
                        })
    return rows


def _write_inv_rows(out_dir, rows, sessions):
    for s in range(sessions):
        lines = [json.dumps(r) + "\n" for r in rows if r["session"] == s]
        (out_dir / f"raw_session{s}.jsonl").write_text("".join(lines))


def _run_matrix_main(monkeypatch, out_dir):
    monkeypatch.setattr(sys, "argv", [
        "run_semantic_matrix.py", "--aggregate-only",
        "--scope", "stress", "--out", str(out_dir)])
    sm.main()


def test_declared_inventory_covers_the_runner_candidates():
    from src.xpu.config import DeviceProfile
    from src.xpu.selector import load_table

    table = load_table(str(ROOT / "config" / "selector_table.json"))
    prof = DeviceProfile.load().hash()
    inv = sm._declared_inventory([(64, 1)], table, "full",
                                 3, 30, 5, 10, 10, prof)
    assert inv["schema"] == "declared-inventory-v1"
    assert (inv["trials"], inv["sessions"], inv["warmup"]) == (30, 3, 5)
    assert set(inv["allowed_terminal_statuses"]) == {
        "ok", "wrong-value", "vendor-unsupported", "error"}
    assert len(inv["cells"]) == 4   # 4 semantics x 1 shape
    by = {c["label"]: c["candidates"] for c in inv["cells"]}
    cands = by["N64B1:c2c-forward"]
    # exactly the contenders run_case builds for this cell
    incumbent = sm._entry_plan(sm.DIRECTION, 64, 1, "c2c-forward",
                               None, table)
    plans = sm._oracle_plans(64, 1, "dit", "c2c-forward",
                             block=incumbent.block or 1024)
    assert ({c for c in cands if c.startswith("oracle:")}
            == {f"oracle:{p.params['oracle_name']}" for p in plans})
    assert cands["incumbent"] == {"steady": 30}
    assert cands["incumbent:one_shot"] == {"one_shot": 10}
    assert cands["incumbent:e2e"] == {"e2e": 10}
    assert cands["selector"] == {"steady": 30}
    assert cands["selector:one_shot"] == {"one_shot": 10}
    assert cands["selector:e2e"] == {"e2e": 10}
    assert cands["selector:plan_creation"] == {"plan_creation": 1}
    assert cands["vendor"] == {"steady": 30}
    assert cands["previous-deployed"] == {"steady": 30}
    # r2c declares the dedicated half-length pipeline candidates
    r2c = {c for c in by["N64B1:r2c"] if c.startswith("oracle:")}
    assert {"oracle:h_dit_t64", "oracle:h_dif_t64"} <= r2c


def test_declared_inventory_winner_block_mode_sweeps_declared_blocks():
    from src.xpu.config import DeviceProfile
    from src.xpu.selector import load_table

    table = load_table(str(ROOT / "config" / "selector_table.json"))
    prof = DeviceProfile.load().hash()
    inv = sm._declared_inventory([(1024, 256)], table, "winner-block",
                                 3, 30, 5, 10, 10, prof)
    cands = {c["label"]: c["candidates"] for c in inv["cells"]}[
        "N1024B256:c2c-forward"]
    oracles = {c for c in cands if c.startswith("oracle:")}
    assert oracles and all(
        o.startswith("oracle:winner-b") for o in oracles)
    assert len(oracles) in (4, 5)  # fixed sweep + deployed block


def test_aggregate_keeps_the_declared_oracle_registry(tmp_path):
    """WP0-B: observed oracle rows may never overwrite the declared
    registry -- a mismatch becomes a recorded failure."""
    rows = _sem_rows("N64B1:c2c-forward", 0.3, 0.9, 1.2, 0.28)
    rows.append(_row("N64B1:c2c-forward", "oracle:evil", "steady",
                     0.5, trial=0))
    _write_raw(tmp_path, rows)
    man = {"trials": 3, "sessions": 1, "warmup": 1,
           "shapes": ["N64B1:c2c-forward"],
           "oracle_candidates": ["oracle:m0_dit"]}
    summary = sm._aggregate(tmp_path, [(64, 1)], 3, 1, 1, manifest=man)
    assert summary["manifest"]["oracle_candidates"] == ["oracle:m0_dit"]
    assert any(f["kind"] == "oracle-candidates-mismatch"
               for f in summary["failures"]), summary["failures"]


def test_aggregate_stamps_observed_oracles_only_when_undeclared(
        tmp_path):
    """Legacy manifests without the key keep the observed stamping."""
    rows = _sem_rows("N64B1:c2c-forward", 0.3, 0.9, 1.2, 0.28)
    _write_raw(tmp_path, rows)
    man = {"trials": 3, "sessions": 1, "warmup": 1,
           "shapes": ["N64B1:c2c-forward"]}
    summary = sm._aggregate(tmp_path, [(64, 1)], 3, 1, 1, manifest=man)
    assert summary["manifest"]["oracle_candidates"] == ["oracle:m0_dit"]


def _stress_inventory_and_rows():
    from src.xpu.config import DeviceProfile
    from src.xpu.selector import load_table

    table = load_table(str(ROOT / "config" / "selector_table.json"))
    prof = DeviceProfile.load().hash()
    inv = sm._declared_inventory([(1024, 256)], table, "full",
                                 3, 30, 5, 10, 10, prof)
    return inv, prof


def test_main_declares_inventory_and_strict_validates(tmp_path,
                                                      monkeypatch):
    """--aggregate-only stamps the pre-measurement inventory into
    manifest.json and the finished output strict-validates against it."""
    from src.xpu.manifest_contract import declared_inventory_sha256

    inv, prof = _stress_inventory_and_rows()
    out = tmp_path / "out"
    out.mkdir()
    _write_inv_rows(out, _rows_from_inventory(inv, prof), 3)
    _run_matrix_main(monkeypatch, out)
    man = json.loads((out / "manifest.json").read_text())
    assert man["declared_inventory"]["cells"]
    assert man["declared_inventory_sha256"] == declared_inventory_sha256(
        man["declared_inventory"])
    assert man["oracle_candidates"] == sorted(
        {c for cell in inv["cells"] for c in cell["candidates"]
         if c.startswith("oracle:")})
    assert validate_benchmark_output(out, strict=True) == []
    summary = json.loads((out / "summary.json").read_text())
    assert not [f for f in summary["failures"]
                if f.get("kind") == "declared-inventory"]
    # the stamp survives into the embedded copy both gates read
    assert (summary["manifest"]["declared_inventory_sha256"]
            == man["declared_inventory_sha256"])


def test_main_fails_closed_when_a_declared_candidate_is_absent(
        tmp_path, monkeypatch):
    """Deleting every row of one declared candidate makes both the
    summary and the strict gate fail (all-absent is a hard fail)."""
    inv, prof = _stress_inventory_and_rows()
    out = tmp_path / "out"
    out.mkdir()
    rows = [r for r in _rows_from_inventory(inv, prof)
            if r["candidate"] != "oracle:m0_dit"]
    _write_inv_rows(out, rows, 3)
    _run_matrix_main(monkeypatch, out)
    summary = json.loads((out / "summary.json").read_text())
    inv_fails = [f for f in summary["failures"]
                 if f.get("kind") == "declared-inventory"]
    assert any("oracle:m0_dit" in f["detail"] and "absent" in f["detail"]
               for f in inv_fails), summary["failures"]
    gate = validate_benchmark_output(out, strict=True)
    assert any("oracle:m0_dit" in f and "absent" in f for f in gate), gate


# ------------- issue #9 WP0-C: recorded-run replay CLI -------------------

def _v5_copy(tmp_path, name):
    import shutil

    release = (ROOT / "results" / "p800" / "releases" /
               "20261019-wp2-semantic-matrix-v5")
    if not (release / "summary.json").is_file():
        pytest.skip("released v5 snapshot not present")
    dst = tmp_path / name
    shutil.copytree(release, dst)
    return dst


def _run_replay(monkeypatch, src, dst):
    monkeypatch.setattr(sys, "argv", [
        "run_semantic_matrix.py", "--aggregate-only",
        "--out", str(src), "--replay-out", str(dst)])
    sm.main()


def test_replay_regenerates_released_analysis_verbatim(tmp_path,
                                                       monkeypatch):
    """WP0-C: --aggregate-only over a recorded run replays the run's
    OWN manifest/cases/registry into a new directory -- cases, metrics,
    failures and hashes reproduce; the evidence identity stays the
    run's while the replay block names today's source commit."""
    import hashlib

    src = _v5_copy(tmp_path, "v5")
    orig = json.loads((src / "summary.json").read_text())
    src_summary = (src / "summary.json").read_bytes()
    src_manifest = (src / "manifest.json").read_bytes()
    replay = tmp_path / "replay"
    _run_replay(monkeypatch, src, replay)

    # the input evidence is never rewritten
    assert (src / "summary.json").read_bytes() == src_summary
    assert (src / "manifest.json").read_bytes() == src_manifest

    regen = json.loads((replay / "summary.json").read_text())
    assert regen["cases"] == orig["cases"]
    assert regen["metrics"] == orig["metrics"]
    assert regen["failures"] == orig["failures"]
    assert regen["raw_sha256"] == orig["raw_sha256"]
    # evidence identity (commit, hashes) is the RUN's, preserved
    for key in ("commit", "env_hash", "profile_hash", "result_sha256",
                "cases_sha256", "protocol_id", "oracle_candidates"):
        assert regen["manifest"][key] == orig["manifest"][key], key
    # source identity recorded separately from the evidence commit
    top = json.loads((replay / "manifest.json").read_text())
    assert top["replay"]["replayed_by_commit"] == sm._git(
        "rev-parse", "HEAD")
    assert top["replay"]["replayed_by_dirty"] in (True, False)
    assert (top["replay"]["replay_input_manifest_sha256"]
            == hashlib.sha256(src_manifest).hexdigest())
    # the replay dir is self-contained: copied bytes + strict-validates
    for gz in sorted(src.glob("raw_session*.jsonl.gz")):
        assert (replay / gz.name).read_bytes() == gz.read_bytes()
    assert validate_benchmark_output(replay, strict=True) == []


def test_replay_refuses_tampered_raw_evidence(tmp_path, monkeypatch):
    """Raw bytes edited after release: the strict input gate refuses
    before any analysis is regenerated."""
    import gzip as _gz

    src = _v5_copy(tmp_path, "v5raw")
    gz = sorted(src.glob("raw_session*.jsonl.gz"))[0]
    with _gz.open(gz, "rt") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        row = json.loads(line)
        if row.get("status") == "ok" and row.get("api_ms") is not None:
            row["api_ms"] = row["api_ms"] + 0.5
            lines[i] = json.dumps(row) + "\n"
            break
    else:
        pytest.fail("no timed row found to tamper")
    with _gz.open(gz, "wt") as f:
        f.writelines(lines)
    with pytest.raises(SystemExit, match="strict validation") as ei:
        _run_replay(monkeypatch, src, tmp_path / "replay")
    assert "result_sha256" in str(ei.value)


def test_replay_refuses_cases_schedule_drift(tmp_path, monkeypatch):
    """cases.json edited (cell dropped): the regenerated analysis no
    longer binds to the recorded cases_sha256 -> hard fail."""
    src = _v5_copy(tmp_path, "v5cases")
    cases = json.loads((src / "cases.json").read_text())
    cases.pop()
    (src / "cases.json").write_text(json.dumps(cases))
    with pytest.raises(SystemExit, match="cases drift"):
        _run_replay(monkeypatch, src, tmp_path / "replay")


def test_replay_refuses_provenance_drift(tmp_path, monkeypatch):
    """Embedded manifest provenance edited -> strict input gate fails;
    top manifest.json edited -> top-vs-summary binding check fails."""
    # (a) embedded summary manifest (what the raw rows are judged by)
    src = _v5_copy(tmp_path, "v5emb")
    summary = json.loads((src / "summary.json").read_text())
    summary["manifest"]["protocol_id"] = "semantic-matrix-v0:drifted"
    (src / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(SystemExit, match="strict validation"):
        _run_replay(monkeypatch, src, tmp_path / "replay1")
    # (b) top manifest.json (the identity the replay would inherit)
    src2 = _v5_copy(tmp_path, "v5top")
    man = json.loads((src2 / "manifest.json").read_text())
    man["protocol_id"] = "semantic-matrix-v0:drifted"
    (src2 / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(SystemExit, match="manifest drift"):
        _run_replay(monkeypatch, src2, tmp_path / "replay2")


def test_replay_requires_new_directory_and_aggregate_only(tmp_path,
                                                          monkeypatch):
    """Evidence is never rewritten in place: replay without
    --replay-out (existing manifest) and --replay-out while measuring
    are both refused."""
    src = _v5_copy(tmp_path, "v5inplace")
    monkeypatch.setattr(sys, "argv", [
        "run_semantic_matrix.py", "--aggregate-only", "--out", str(src)])
    with pytest.raises(SystemExit, match="replay-out"):
        sm.main()
    monkeypatch.setattr(sys, "argv", [
        "run_semantic_matrix.py", "--scope", "stress",
        "--out", str(tmp_path / "fresh"), "--replay-out",
        str(tmp_path / "r2")])
    with pytest.raises(SystemExit, match="replay-out"):
        sm.main()


# ------------- issue #9 WP1: exact cell schedule (--cells) ----------------

def test_parse_cells_spec_formats_and_merge():
    cells = sm.parse_cells_spec(
        "N64B4:c2c-forward+c2r, N64B4:r2c, N512B64")
    assert cells == [(64, 4, ["c2c-forward", "c2r", "r2c"]),
                     (512, 64, list(sm.SEMANTICS))]
    with pytest.raises(SystemExit, match="unknown semantic"):
        sm.parse_cells_spec("N64B4:c2c-sideways")
    with pytest.raises(SystemExit, match="bad cell"):
        sm.parse_cells_spec("64x4")
    with pytest.raises(SystemExit, match="empty"):
        sm.parse_cells_spec(" , ")


def test_schedule_flattening_respects_sem_map():
    cells = sm.parse_cells_spec("N64B4:c2c-forward,N512B64:r2c+c2r")
    cases = [((n, b)) for n, b, _ in cells]
    sem_map = {(n, b): s for n, b, s in cells}
    assert sm._schedule(cases, sem_map) == [
        (64, 4, "c2c-forward"),
        (512, 64, "r2c"), (512, 64, "c2r")]
    # without a sem_map every shape runs all four semantics
    assert len(sm._schedule([(64, 4)])) == 4


def test_main_cells_filtered_schedule_declares_and_validates(
        tmp_path, monkeypatch):
    """--cells measures exactly the declared cell+semantic set: one
    label in shapes/inventory/schedule, strict gate green."""
    from src.xpu.config import DeviceProfile
    from src.xpu.selector import load_table

    table = load_table(str(ROOT / "config" / "selector_table.json"))
    prof = DeviceProfile.load().hash()
    sem_map = {(1024, 256): ["r2c"]}
    inv = sm._declared_inventory([(1024, 256)], table, "full",
                                 3, 30, 5, 10, 10, prof,
                                 sem_map=sem_map)
    assert [c["label"] for c in inv["cells"]] == ["N1024B256:r2c"]
    out = tmp_path / "out"
    out.mkdir()
    _write_inv_rows(out, _rows_from_inventory(inv, prof), 3)
    monkeypatch.setattr(sys, "argv", [
        "run_semantic_matrix.py", "--aggregate-only",
        "--cells", "N1024B256:r2c", "--out", str(out)])
    sm.main()
    man = json.loads((out / "manifest.json").read_text())
    assert man["shapes"] == ["N1024B256:r2c"]
    sched = json.loads((out / "schedule.json").read_text())
    assert sched["cells"] == [[1024, 256, ["r2c"]]]
    summary = json.loads((out / "summary.json").read_text())
    assert [c["label"] for c in summary["cases"]] == ["N1024B256:r2c"]
    assert validate_benchmark_output(out, strict=True) == []


def test_filtered_schedule_snapshot_and_replay_roundtrip(tmp_path,
                                                         monkeypatch):
    """A filtered-schedule run snapshots and replays with its exact
    per-shape semantics (WP1 domains survive the WP0-C replay path)."""
    from src.xpu.config import DeviceProfile
    from src.xpu.selector import load_table

    table = load_table(str(ROOT / "config" / "selector_table.json"))
    prof = DeviceProfile.load().hash()
    sem_map = {(1024, 256): ["r2c"]}
    inv = sm._declared_inventory([(1024, 256)], table, "full",
                                 3, 30, 5, 10, 10, prof,
                                 sem_map=sem_map)
    out = tmp_path / "out"
    out.mkdir()
    _write_inv_rows(out, _rows_from_inventory(inv, prof), 3)
    monkeypatch.setattr(sys, "argv", [
        "run_semantic_matrix.py", "--aggregate-only",
        "--cells", "N1024B256:r2c", "--out", str(out)])
    sm.main()
    orig = json.loads((out / "summary.json").read_text())
    man = json.loads((out / "manifest.json").read_text())
    # snapshot under a temporary ROOT (immutable evidence never touched)
    monkeypatch.setattr(sm, "ROOT", tmp_path)
    snap = sm._write_snapshot(out, "rid-filtered", man, {}, None,
                              [(1024, 256)])
    monkeypatch.setattr(sm, "ROOT", ROOT)
    # replay reproduces the filtered analysis from the snapshot's own
    # manifest/shapes (single r2c label, no cross product)
    replay = tmp_path / "replay"
    monkeypatch.setattr(sys, "argv", [
        "run_semantic_matrix.py", "--aggregate-only",
        "--out", str(snap), "--replay-out", str(replay)])
    sm.main()
    regen = json.loads((replay / "summary.json").read_text())
    assert regen["cases"] == orig["cases"]
    assert [c["label"] for c in regen["cases"]] == ["N1024B256:r2c"]
    assert regen["metrics"] == orig["metrics"]
    assert validate_benchmark_output(replay, strict=True) == []
