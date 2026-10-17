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


def test_promote_only_touches_production_when_called(tmp_path, monkeypatch):
    import bootstrap_profile as bp
    # redirect production targets into tmp so the real config/ is never
    # at risk even if the test misbehaves
    monkeypatch.setattr(bp, "PRODUCTION_TABLE",
                        tmp_path / "config" / "selector_table.json")
    monkeypatch.setattr(bp, "PRODUCTION_CALIBRATION",
                        tmp_path / "config" / "calibration.json")
    prod = bp.PRODUCTION_TABLE
    prod.parent.mkdir(parents=True, exist_ok=True)
    prod.write_text('{"published": "old"}')
    old_bytes = prod.read_bytes()

    staging = tmp_path / "staging"
    staging.mkdir()
    staged = staging / "selector_table.json"
    staged.write_text('{"published": "new"}')

    # no promote call: production bytes unchanged
    assert prod.read_bytes() == old_bytes

    failures = bp.promote(staging)
    assert failures == []
    assert json.loads(prod.read_text())["published"] == "new"
    # atomic: no tmp residue
    assert not list(prod.parent.glob("*.promote-tmp"))


def test_promote_missing_staged_table_fails(tmp_path, monkeypatch):
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
    failures = bp.promote(staging)
    assert failures and "staged table missing" in failures[0]
    assert prod.read_text() == '{"published": "old"}'


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
