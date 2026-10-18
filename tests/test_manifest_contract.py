"""WP0 manifest contract + bootstrap gate (issue #6 PR-A, host-only).

Covers the review acceptance:

* manifest missing field / duplicate trial / missing case aggregate
  into one failure report (never die on the first problem);
* bootstrap never touches production config/ without ``--promote``;
* a failed stage keeps staging for resume and publishes nothing;
* promote is an atomic replace;
* the verify bundle fails closed on env mismatch.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.xpu.manifest_contract import (  # noqa: E402
    REQUIRED_MANIFEST_KEYS,
    format_failures,
    result_sha256,
    rows_sha256,
    validate_benchmark_output,
    validate_manifest,
    validate_raw_sessions,
)


def _valid_manifest() -> dict:
    return {
        "commit": "a" * 40,
        "dirty": False,
        "device": "kunlunxin-p800",
        "driver": "515.58",
        "profile_hash": "a78b86793c7142a0",
        "env_hash": "6b50bd316032c0fe",
        "toolchain": "python 3.10 + torch 2.9",
        "protocol_id": "batch-release-v1:test",
        "timing_scope": "api wall time (sync), ms",
        "selection_semantics": "plan-reuse, interleaved, 2% guard",
        "warmup": 5,
        "trials": 30,
        "sessions": 1,
        "seed_base": 1234,
        "candidates": ["m0", "m2"],
        "shapes": ["N1024B16_dit"],
        "error_threshold": 1e-4,
        "repro_command": "python scripts/x.py",
        "gm_bytes_kind": "analytical estimate",
        "kernel_time_status": "not collected",
        "source_hash": "b" * 40,
        "result_sha256": "c" * 64,
    }


def _valid_row(session=0, label="N1024B16_dit", trial=0,
               candidate="m0") -> dict:
    return {
        "session": session, "label": label, "trial": trial,
        "candidate": candidate, "api_ms": 0.5, "correct": True,
        "err": 1e-7, "status": "ok",
        "protocol_id": "batch-release-v1:test",
        "profile_hash": "a78b86793c7142a0",
        "timed_scope": "api wall time (sync), ms",
    }


def _write_bench_dir(tmp_path, manifest, rows) -> Path:
    (tmp_path / "summary.json").write_text(
        json.dumps({"manifest": manifest, "shapes": []}))
    with open(tmp_path / "raw_session0.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return tmp_path / "summary.json"


# ---------------------------- schema ------------------------------------

def test_valid_manifest_passes():
    assert validate_manifest(_valid_manifest(), strict=True) == []


def test_missing_fields_aggregate():
    man = _valid_manifest()
    for key in ("env_hash", "protocol_id", "sessions", "trials"):
        del man[key]
    failures = validate_manifest(man)
    assert len(failures) == 4
    for key in ("env_hash", "protocol_id", "sessions", "trials"):
        assert any(key in f for f in failures)
    # one report, not first-failure abort
    text = format_failures(failures)
    assert text.startswith("manifest gate: 4 failure(s)")


def test_aliases_satisfy_canonical_keys():
    man = _valid_manifest()
    # legacy runner spellings still validate
    man["command"] = man.pop("repro_command")
    man["correctness_threshold"] = man.pop("error_threshold")
    man["config_hash"] = man.pop("source_hash")
    assert validate_manifest(man, strict=True) == []


def test_empty_field_reported():
    man = _valid_manifest()
    man["candidates"] = []
    failures = validate_manifest(man)
    assert any("empty: candidates" in f for f in failures)


def test_every_required_key_covered_by_schema():
    man = _valid_manifest()
    missing = [k for k in REQUIRED_MANIFEST_KEYS
               if validate_manifest(dict(man, **{k: None}))
               and not any(k in f for f in
                           validate_manifest(dict(man, **{k: None})))]
    # sanity: flipping any required key to None yields a failure
    # mentioning that key
    assert missing == []


# ------------------------- raw evidence ---------------------------------

def test_duplicate_trial_reported(tmp_path):
    man = _valid_manifest()
    rows = [_valid_row(trial=i) for i in range(3)]
    rows.append(_valid_row(trial=1))  # duplicate
    summary = _write_bench_dir(tmp_path, man, rows)
    failures = validate_benchmark_output(summary)
    assert any("duplicate trial" in f for f in failures)
    assert any("N1024B16_dit" in f or "trial" in f for f in failures)


def test_missing_case_reported(tmp_path):
    man = _valid_manifest()
    rows = [_valid_row(label="N1024B16_dit", trial=i) for i in range(3)]
    summary = _write_bench_dir(tmp_path, man, rows)
    # declared shape list includes one with no rows at all
    man2 = dict(man, shapes=["N1024B16_dit", "N2048B64_dit"])
    (tmp_path / "summary.json").write_text(
        json.dumps({"manifest": man2, "shapes": []}))
    failures = validate_benchmark_output(summary)
    assert any("missing case" in f for f in failures)
    assert any("N2048B64_dit" in f for f in failures)


def test_missing_row_field_reported(tmp_path):
    man = _valid_manifest()
    row = _valid_row()
    del row["api_ms"]
    summary = _write_bench_dir(tmp_path, man, [row])
    failures = validate_benchmark_output(summary)
    assert any("missing row field 'api_ms'" in f for f in failures)


def test_result_hash_mismatch_detected(tmp_path):
    man = _valid_manifest()
    rows = [_valid_row(trial=i) for i in range(2)]
    summary = _write_bench_dir(tmp_path, man, rows)
    # stamp correct hash first: strict passes
    payload = json.loads(summary.read_text())
    payload["manifest"]["result_sha256"] = result_sha256(tmp_path)
    summary.write_text(json.dumps(payload))
    assert validate_benchmark_output(summary, strict=True) == []
    # tamper with a row: hash mismatch
    with open(tmp_path / "raw_session0.jsonl", "a") as f:
        f.write(json.dumps(_valid_row(trial=99)) + "\n")
    failures = validate_benchmark_output(summary, strict=True)
    assert any("result_sha256 mismatch" in f for f in failures)


def test_shape_label_colon_spelling_matches_rows(tmp_path):
    # focus runners declare N1024B16:dit while raw rows use
    # N1024B16_dit — both spellings must satisfy the same case
    man = dict(_valid_manifest(), shapes=["N1024B16:dit"])
    rows = [_valid_row(label="N1024B16_dit", trial=i) for i in range(2)]
    summary = _write_bench_dir(tmp_path, man, rows)
    failures = validate_benchmark_output(summary, strict=False)
    assert not [f for f in failures if "missing case" in f], failures
    # and a genuinely absent direction is still reported
    man = dict(_valid_manifest(), shapes=["N1024B16:dif"])
    (tmp_path / "summary.json").write_text(
        json.dumps({"manifest": man, "shapes": []}))
    failures = validate_benchmark_output(summary, strict=False)
    assert any("missing case" in f for f in failures)


def test_strict_row_provenance_mismatch(tmp_path):
    man = _valid_manifest()
    rows = [_valid_row(trial=i) for i in range(2)]
    rows[1]["profile_hash"] = "local"  # legacy sentinel / mixed run
    summary = _write_bench_dir(tmp_path, man, rows)
    payload = json.loads(summary.read_text())
    payload["manifest"]["result_sha256"] = result_sha256(tmp_path)
    summary.write_text(json.dumps(payload))
    assert not [f for f in validate_benchmark_output(summary)
                if "mismatch" in f]  # lenient tier: archived
    failures = validate_benchmark_output(summary, strict=True)
    assert any("row profile_hash mismatch" in f for f in failures)


def test_strict_samples_count_matches_declared_trials(tmp_path):
    man = _valid_manifest()  # trials = 30
    payload = {"manifest": man,
               "rows": [{"n": 1024,
                         "m0_dit_ms": 0.5,
                         "m0_dit_samples": [0.5] * 30},
                        {"n": 2048,
                         "m0_dit_ms": 0.5,
                         "m0_dit_samples": [0.5] * 12}]}
    summary = tmp_path / "confirm_x.json"
    summary.write_text(json.dumps(payload))
    payload["manifest"] = dict(man,
                               result_sha256=rows_sha256(payload),
                               source_hash="b" * 40)
    summary.write_text(json.dumps(payload))
    failures = validate_benchmark_output(summary, strict=True)
    assert any("12 samples" in f and "30 trials" in f for f in failures)
    assert not [f for f in failures if "30 samples" in f]


def test_missing_session_files_reported_strict(tmp_path):
    man = dict(_valid_manifest(), sessions=2)
    rows = [_valid_row(session=0, trial=i) for i in range(2)]
    summary = _write_bench_dir(tmp_path, man, rows)
    failures = validate_benchmark_output(summary, strict=True)
    assert any("missing sessions" in f for f in failures)


def test_rows_sha256_stable_and_sensitive():
    payload = {"rows": [{"a": 1}, {"b": 2}]}
    assert rows_sha256(payload) == rows_sha256(dict(payload))
    assert rows_sha256(payload) != rows_sha256({"rows": [{"a": 1}]})


# --------------------------- bootstrap ----------------------------------

def test_bootstrap_module_exports_and_no_production_writes_on_import():
    import bootstrap_profile as bp
    assert callable(bp.promote)
    assert callable(bp.verify_bundle)
    # promote targets are the real config paths (only touched by
    # promote()); the module itself never writes on import
    assert bp.PRODUCTION_TABLE.name == "selector_table.json"


def test_state_roundtrip_and_resume(tmp_path):
    import bootstrap_profile as bp
    out = tmp_path / "out"
    out.mkdir()
    state = bp.load_state(out)
    assert state == {"completed": [], "failures": {}}
    bp.mark_complete(state, "probe")
    bp.save_state(out, state)
    reloaded = bp.load_state(out)
    assert "probe" in reloaded["completed"]
    assert bp.should_skip(reloaded, "probe", resume=True)
    assert not bp.should_skip(reloaded, "probe", resume=False)
    assert not bp.should_skip(reloaded, "check", resume=True)


def _minimal_staged_table():
    return {
        "profile_hash": "a78b86793c7142a0",
        "precision": "fp32",
        "guard_pct": 2.0,
        "env_hash": "e" * 16,
        "entries": [],
    }


def _promote_fixture(tmp_path, monkeypatch, *, table_text=None,
                     verify_failures=None):
    """Redirected production targets + a staged table + fresh state."""
    import bootstrap_profile as bp
    monkeypatch.setattr(bp, "verify_bundle",
                        lambda *a, **k: list(verify_failures or []))
    monkeypatch.setattr(bp, "PRODUCTION_TABLE",
                        tmp_path / "config" / "selector_table.json")
    monkeypatch.setattr(bp, "PRODUCTION_CALIBRATION",
                        tmp_path / "config" / "calibration.json")
    prod = bp.PRODUCTION_TABLE
    prod.parent.mkdir(parents=True, exist_ok=True)
    prod.write_text('{"published": "old"}')
    staging = tmp_path / "staging"
    staging.mkdir()
    staged = staging / "selector_table.json"
    staged.write_text(table_text if table_text is not None
                      else json.dumps(_minimal_staged_table()))
    state = {"verified": {"hashes": bp.staging_artifact_hashes(staging)}}
    return bp, prod, staging, state


def test_promote_only_touches_production_when_called(tmp_path, monkeypatch):
    bp, prod, staging, state = _promote_fixture(tmp_path, monkeypatch)
    old_bytes = prod.read_bytes()

    # no promote call: production bytes unchanged
    assert prod.read_bytes() == old_bytes

    failures = bp.promote(staging, state=state)
    assert failures == []
    assert json.loads(prod.read_text())["precision"] == "fp32"
    # atomic: no tmp residue
    assert not list(prod.parent.glob("*.promote-tmp"))


def test_promote_missing_staged_table_fails(tmp_path, monkeypatch):
    bp, prod, staging, state = _promote_fixture(
        tmp_path, monkeypatch, table_text="")
    # overwrite with nothing: remove the staged table
    (staging / "selector_table.json").unlink()
    state["verified"]["hashes"] = bp.staging_artifact_hashes(staging)
    failures = bp.promote(staging, state=state)
    assert failures and "staged table missing" in failures[0]
    assert prod.read_text() == '{"published": "old"}'


def test_promote_refuses_without_fresh_verify(tmp_path, monkeypatch):
    import bootstrap_profile as bp
    monkeypatch.setattr(bp, "PRODUCTION_TABLE",
                        tmp_path / "config" / "selector_table.json")
    monkeypatch.setattr(bp, "PRODUCTION_CALIBRATION",
                        tmp_path / "config" / "calibration.json")
    prod = bp.PRODUCTION_TABLE
    prod.parent.mkdir(parents=True, exist_ok=True)
    prod.write_text('{"published": "old"}')
    staging = tmp_path / "staging"
    staging.mkdir()
    failures = bp.promote(staging, state=None)
    assert failures and "no fresh verify" in failures[0]
    assert prod.read_text() == '{"published": "old"}'


def test_promote_refuses_on_post_verify_drift(tmp_path, monkeypatch):
    bp, prod, staging, state = _promote_fixture(tmp_path, monkeypatch)
    # simulate a --resume that modified staging after verify
    state["verified"]["hashes"] = {"selector_table.json": "0" * 64}
    failures = bp.promote(staging, state=state)
    assert failures and "changed after verify" in failures[0]
    assert prod.read_text() == '{"published": "old"}'


def test_promote_reruns_verify_and_refuses_on_failure(tmp_path, monkeypatch):
    # promote never trusts a prior verify: it re-runs the gate on the
    # current bytes and refuses on any failure (issue #8 P1-D)
    bp, prod, staging, state = _promote_fixture(
        tmp_path, monkeypatch, verify_failures=["verify: boom"])
    failures = bp.promote(staging, state=state)
    assert any("verify: boom" in f for f in failures)
    assert prod.read_text() == '{"published": "old"}'


def test_promote_refuses_dirty_evidence(tmp_path, monkeypatch):
    import run_semantic_matrix as sm
    from src.xpu.config import git_commit
    bp, prod, staging, state = _promote_fixture(tmp_path, monkeypatch)
    mdir = staging / "matrix_stress"
    mdir.mkdir()
    man = {"commit": git_commit(), "dirty": True}
    (mdir / "summary.json").write_text(json.dumps(
        {"manifest": man, "cases": [], "protocol_id": sm.PROTOCOL_ID}))
    state["verified"]["hashes"] = bp.staging_artifact_hashes(staging)
    failures = bp.promote(staging, state=state)
    assert any("dirty" in f for f in failures)
    assert prod.read_text() == '{"published": "old"}'


def test_promote_transactional_restore_on_replace_failure(tmp_path,
                                                          monkeypatch):
    bp, prod, staging, state = _promote_fixture(tmp_path, monkeypatch)
    (staging / "calibration.json").write_text('{"cal": "new"}')
    calib_prod = bp.PRODUCTION_CALIBRATION
    calib_prod.write_text('{"cal": "old"}')
    state["verified"]["hashes"] = bp.staging_artifact_hashes(staging)
    old_table = prod.read_bytes()
    old_calib = calib_prod.read_bytes()

    import os as _os
    real_replace = _os.replace
    calls = {"n": 0}

    def flaky(src, dst, *a, **k):
        calls["n"] += 1
        if calls["n"] == 2:   # fail the second publish replace only
            raise OSError("disk full")
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(_os, "replace", flaky)
    failures = bp.promote(staging, state=state)
    assert any("restored" in f for f in failures)
    # both production targets rolled back to their pre-promote bytes
    assert prod.read_bytes() == old_table
    assert calib_prod.read_bytes() == old_calib
    assert not list(prod.parent.glob("*.promote-tmp"))


def test_failed_stage_keeps_staging_and_records_state(tmp_path):
    import bootstrap_profile as bp
    out = tmp_path / "out"
    staging = out / bp.STAGING_DIR
    staging.mkdir(parents=True)
    (staging / "partial.log").write_text("partial evidence")
    state = bp.load_state(out)
    # simulate the failure path main() uses
    state.setdefault("failures", {})["calibrate"] = {"exit": 1}
    bp.save_state(out, state)
    reloaded = bp.load_state(out)
    assert reloaded["failures"]["calibrate"]["exit"] == 1
    assert "calibrate" not in reloaded["completed"]
    # staging retained for --resume
    assert (staging / "partial.log").read_text() == "partial evidence"


def test_verify_bundle_fails_closed_on_env_mismatch(tmp_path):
    import bootstrap_profile as bp
    # a minimal staged table whose profile/env is deliberately foreign
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "confirm_b1024.json").write_text(json.dumps({
        "manifest": _valid_manifest(), "rows": [{"n": 1024}],
        "candidates": {}}))
    table = {
        "profile_hash": "a78b86793c7142a0",
        "precision": "fp32",
        "guard_pct": 2.0,
        "env_hash": "0" * 16,          # foreign environment
        "entries": [],
    }
    (staging / "selector_table.json").write_text(json.dumps(table))
    failures = bp.verify_bundle(staging, env_hash="1" * 16)
    text = "\n".join(failures)
    # freshness gate refuses (unknown/mismatch) -> fail closed
    assert "freshness gate failed closed" in text or "env mismatch" in text
    assert failures  # never a silent pass


def test_non_ok_row_may_omit_timing_fields(tmp_path):
    """Issue #8 PR-G: unsupported/error rows keep no timing sample but
    stay in the raw evidence (status + reason), so the strict gate must
    not reject them for missing api_ms/err."""
    man = _valid_manifest()
    ok = _valid_row(trial=0)
    bad = _valid_row(trial=1, candidate="vendor")
    bad.update({"api_ms": None, "err": None, "correct": False,
                "status": "vendor-unsupported",
                "reason": "RuntimeError: unsupported size"})
    summary = _write_bench_dir(tmp_path, man, [ok, bad])
    assert validate_benchmark_output(summary, strict=False) == []
    payload = json.loads(summary.read_text())
    payload["manifest"]["result_sha256"] = result_sha256(tmp_path)
    summary.write_text(json.dumps(payload))
    assert validate_benchmark_output(summary, strict=True) == []
    # an "ok" row still requires its timing sample
    late = _valid_row(trial=2, candidate="m0")
    del late["api_ms"]
    with open(tmp_path / "raw_session0.jsonl", "a") as f:
        f.write(json.dumps(late) + "\n")
    failures = validate_benchmark_output(summary, strict=True)
    assert any("missing row field 'api_ms'" in f for f in failures)


def test_stamp_result_hashes_covers_matrix_runs(tmp_path):
    """Issue #8: matrix_*/summary.json is stamped in place with
    source_hash + result_sha256 over its raw sessions."""
    import bootstrap_profile as bp
    import run_semantic_matrix as sm
    out = tmp_path / "staging"
    mdir = out / "matrix_stress"
    mdir.mkdir(parents=True)
    rows = [_valid_row(trial=i) for i in range(2)]
    with open(mdir / "raw_session0.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    man = _valid_manifest()
    man["protocol_id"] = sm.PROTOCOL_ID
    man.pop("result_sha256", None)
    (mdir / "summary.json").write_text(json.dumps(
        {"manifest": man, "cases": [], "protocol_id": sm.PROTOCOL_ID}))
    failures = bp.stamp_result_hashes(out)
    assert failures == []
    stamped = json.loads((mdir / "summary.json").read_text())["manifest"]
    assert stamped.get("source_hash")
    assert stamped.get("result_sha256") == result_sha256(mdir)


# ------------- exact-key smoke vs the semantic/legacy tiers (PR-H) -----

def _smoke_table():
    legacy = {
        "N": 1024, "batch": 16, "direction": "dit", "block": 256,
        "flow_id": "local-fused-tile",
        "flow_realization": "matrix-resident",
        "processing_unit": "dense-dot-fft64",
        "params": {"tile": 128, "rows": 16, "variant": "spread",
                   "batch": 16},
        "reason": "legacy",
    }
    semantic = {
        "N": 1024, "batch": 16, "direction": "dit", "block": 256,
        "transform_type": "c2c", "transform_direction": "forward",
        "normalization": "backward",
        "flow_id": "local-fused-tile",
        "flow_realization": "matrix-resident",
        "processing_unit": "dense-dot-fft64",
        "params": {"tile": 64, "rows": 8, "variant": "spread",
                   "batch": 16},
        "reason": "semantic",
    }
    return {"profile_hash": "p" * 16, "precision": "fp32",
            "env_hash": "e" * 16, "entries": [legacy, semantic]}


def _run_smoke(table, select_impl, tmp_path, monkeypatch):
    """verify_bundle's exact-key smoke over a fabricated staged table."""
    import bootstrap_profile as bp
    from src.xpu.selector import Selection

    staging = tmp_path / "staging"
    (staging / "batch_dummy").mkdir(parents=True)
    (staging / "selector_table.json").write_text(json.dumps(table))
    monkeypatch.setattr(bp, "validate_benchmark_output",
                        lambda *a, **k: [])
    monkeypatch.setattr(
        "src.xpu.selector.load_table", lambda path: json.loads(
            (staging / "selector_table.json").read_text()))
    monkeypatch.setattr(
        "src.xpu.selector.assert_current_environment",
        lambda tbl: tbl.get("env_hash"))

    def fake_select(*, direction, n, batch, precision, profile_hash,
                    table, env_hash, transform_type="c2c",
                    transform_direction="forward", normalization="backward"):
        hit = select_impl(n, batch, transform_type, transform_direction,
                          normalization)
        if hit is None:
            return Selection(n=n, direction=direction, batch=batch,
                             precision=precision, source="abstain",
                             reason="none")
        entry, tier = hit
        return Selection(n=n, direction=direction, batch=batch,
                         precision=precision, source="table", reason="ok",
                         entry_tier=tier,
                         params=dict(entry.get("params") or {}),
                         block=entry.get("block"))

    monkeypatch.setattr("src.xpu.selector.select", fake_select)
    return [f for f in bp.verify_bundle(staging, "e" * 16)
            if "exact-key select" in f]


def test_exact_key_smoke_accepts_semantic_shadowing_legacy(tmp_path,
                                                           monkeypatch):
    # Legacy entry's default key resolves to the semantic-tier entry
    # (tier order by design); the legacy re-probe (c2c/ortho) answers
    # the legacy entry itself -> no smoke failure.
    table = _smoke_table()
    entries = table["entries"]
    legacy, semantic = entries[0], entries[1]

    def select_impl(n, b, ttype, tdir, norm):
        if (ttype, tdir, norm) == ("c2c", "forward", "ortho"):
            return (legacy, "legacy")
        return (semantic, "semantic")   # default key -> semantic tier

    failures = _run_smoke(table, select_impl, tmp_path, monkeypatch)
    assert failures == []


def test_exact_key_smoke_fails_when_legacy_is_actually_shadowed(
        tmp_path, monkeypatch):
    # Same shadow, but the legacy re-probe resolves a different entry:
    # that is a real shadowing bug and must fail closed.
    table = _smoke_table()
    entries = table["entries"]
    legacy, semantic = entries[0], entries[1]

    def select_impl(n, b, ttype, tdir, norm):
        if (ttype, tdir, norm) == ("c2c", "forward", "ortho"):
            return (semantic, "semantic")
        return (semantic, "semantic")

    failures = _run_smoke(table, select_impl, tmp_path, monkeypatch)
    assert any("resolved a different entry" in f for f in failures)


def test_exact_key_smoke_fails_legacy_tier_mismatch(tmp_path, monkeypatch):
    # No semantic shadow: a legacy-tier mismatch is a plain failure.
    table = _smoke_table()
    entries = table["entries"]
    legacy, semantic = entries[0], entries[1]

    def select_impl(n, b, ttype, tdir, norm):
        if "transform_type" in legacy:
            return (legacy, "legacy")
        return (semantic, "legacy")   # wrong legacy hit for the default

    failures = _run_smoke(table, select_impl, tmp_path, monkeypatch)
    assert any("resolved a different entry" in f for f in failures)
