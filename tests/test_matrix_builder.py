"""Issue #8 PR-G: semantic-matrix ingestion gates (host-only).

The builder's ``--matrix-run`` path is fail-closed: a cell only
produces an exact-semantic entry when the confirmation artifact passes
the strict output gate AND the cell clears its own guard with complete,
correctness-clean, contract-legal evidence. Rejected cells land in
rejections and never touch the table.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_semantic_matrix as sm  # noqa: E402
from scripts.build_selector_table import (  # noqa: E402
    _load_matrix,
    build_matrix_entries,
)
from src.xpu.manifest_contract import result_sha256  # noqa: E402
from src.xpu.selector import entry_contract_violations  # noqa: E402

SEM = "c2c-forward"
LABEL = f"N1024B256:{SEM}"
WINNER = "oracle:m4_spread_b256_dit"


def _manifest(d: Path, shapes, **over):
    man = {
        "commit": "c" * 12,
        "dirty": False,
        "device": "kunlunxin-p800",
        "driver": "515.58",
        "profile_hash": "f" * 16,
        "env_hash": "e" * 16,
        "toolchain": "python fixture + semantic-matrix-v1",
        "protocol_id": sm.PROTOCOL_ID,
        "timing_scope": sm.TIMING_SCOPE,
        "selection_semantics": "correctness-first fixture",
        "warmup": 5,
        "trials": 30,
        "sessions": 3,
        "seed_base": 1234,
        "candidates": ["incumbent", "selector", "oracle:*", "vendor"],
        "shapes": list(shapes),
        "error_threshold": 1e-4,
        "correctness_threshold": 1e-4,
        "repro_command": "python scripts/run_semantic_matrix.py fixture",
        "gm_bytes_kind": "analytical estimate only",
        "kernel_time_status": "not collected",
        "source_hash": "d" * 12,
        "dtype": "complex64",
        "precision": "fp32",
        "env": {},
    }
    man.update(over)
    man.setdefault("result_sha256", None)
    return man


def _steady_row(session, trial, candidate="oracle:m4_spread_b256_dit",
                label=LABEL, **over):
    row = {
        "session": session,
        "label": label,
        "trial": trial,
        "candidate": candidate,
        "tier": sm.TIER_STEADY,
        "protocol_id": sm.PROTOCOL_ID,
        "profile_hash": "f" * 16,
        "timed_scope": sm.TIMING_SCOPE,
        "processing_unit": "radix2-scalar",
        "flow_id": "batch-spread",
        "flow_realization": "global-stage",
        "params": {"variant": "spread", "batch": 256,
                   "oracle_name": "m4_spread_b256_dit",
                   "selection_source": "oracle-candidate",
                   "fallback": False},
        "block": 1024,
        "api_ms": 1.0,
        "correct": True,
        "err": 1e-6,
        "status": "ok",
        "reason": None,
        "notes": None,
    }
    row.update(over)
    return row


def _case(label=LABEL, semantic=SEM, **over):
    case = {
        "label": label, "N": 1024, "batch": 256, "semantic": semantic,
        "direction": "dit",
        "incumbent_ms": 2.20, "oracle_ms": 1.00,
        "oracle_candidate": WINNER,
    }
    case.update(over)
    return case


def _write_matrix(tmp_path, *, cases=None, sessions=(0, 1, 2),
                  per_session_rows=3, extra_rows=(), extra_oracles=(),
                  omit_result_sha=False, name="matrix_stress",
                  label=LABEL, semantic=SEM):
    d = tmp_path / name
    d.mkdir()
    rows = []
    for s in sessions:
        for trial in range(per_session_rows):
            rows.append(_steady_row(s, trial, label=label))
        for trial, cand in enumerate(extra_oracles):
            rows.append(_steady_row(s, 100 + trial, candidate=cand,
                                    api_ms=1.5, label=label))
    rows.extend(extra_rows)
    by_session = {}
    for r in rows:
        by_session.setdefault(r["session"], []).append(r)
    for s, srows in sorted(by_session.items()):
        with open(d / f"raw_session{s}.jsonl", "w") as fh:
            for r in srows:
                fh.write(json.dumps(r) + "\n")
    man = _manifest(d, [label])
    if not omit_result_sha:
        man["result_sha256"] = result_sha256(d)
    payload = {"manifest": man,
               "cases": cases or [_case(label=label, semantic=semantic)],
               "protocol_id": sm.PROTOCOL_ID}
    (d / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    return d


# --- artifact gates -----------------------------------------------------

def _load(d):
    return _load_matrix(str(d), [])


def test_good_matrix_run_loads_and_builds_entry(tmp_path):
    d = _write_matrix(tmp_path)
    run = _load(d)
    entries = build_matrix_entries([run])
    assert len(entries) == 1
    e = entries[0]
    assert (e["N"], e["batch"], e["direction"]) == (1024, 256, "dit")
    assert (e["transform_type"], e["transform_direction"],
            e["normalization"]) == ("c2c", "forward", "backward")
    assert e["params"]["batch"] == 256
    assert e["block"] == 1024
    assert e["flow_id"] == "batch-spread"
    assert e["evidence"]["challenger"] == WINNER
    assert e["evidence"]["gain_pct"] == pytest.approx(54.545, abs=1e-3)
    assert e["evidence"]["sessions"] == 3
    assert entry_contract_violations(e) == []


def test_missing_result_sha_fails_the_strict_gate(tmp_path):
    d = _write_matrix(tmp_path, omit_result_sha=True)
    with pytest.raises(Exception) as ei:
        _load(d)
    assert "strict output gate" in str(ei.value)


def test_floor_sessions_enforced(tmp_path):
    d = _write_matrix(tmp_path)
    summary = json.loads((d / "summary.json").read_text())
    summary["manifest"]["sessions"] = 2
    (d / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(Exception) as ei:
        _load(d)
    assert "sessions=2" in str(ei.value)
    assert "issue floor" in str(ei.value)


def test_floor_trials_enforced(tmp_path):
    d = _write_matrix(tmp_path)
    summary = json.loads((d / "summary.json").read_text())
    summary["manifest"]["trials"] = 10
    (d / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(Exception) as ei:
        _load(d)
    assert "trials=10" in str(ei.value)


def test_wrong_protocol_refused(tmp_path):
    d = _write_matrix(tmp_path)
    summary = json.loads((d / "summary.json").read_text())
    summary["manifest"]["protocol_id"] = "bench-confirm:something-else"
    summary["manifest"]["result_sha256"] = result_sha256(d)
    (d / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(Exception) as ei:
        _load(d)
    assert "protocol_id" in str(ei.value)


# --- cell gates ---------------------------------------------------------

def test_guard_not_cleared_keeps_legacy_path(tmp_path):
    d = _write_matrix(tmp_path,
                      cases=[_case(incumbent_ms=2.20, oracle_ms=2.19)])
    rejections = []
    entries = build_matrix_entries([_load(d)], rejections=rejections)
    assert entries == []
    assert any("guard not cleared" in r for r in rejections)
    assert any("legacy/abstain path stays" in r for r in rejections)


def test_incomplete_winner_sessions_rejected(tmp_path):
    # all three session files exist (contract-clean) but the winner
    # only has timing in sessions 0-1: completeness must refuse it.
    d = _write_matrix(tmp_path, sessions=(0, 1))
    with open(d / "raw_session2.jsonl", "w") as fh:
        fh.write(json.dumps(_steady_row(
            2, 0, candidate="incumbent", api_ms=2.2)) + "\n")
    summary = json.loads((d / "summary.json").read_text())
    summary["manifest"]["result_sha256"] = result_sha256(d)
    (d / "summary.json").write_text(json.dumps(summary))
    rejections = []
    entries = build_matrix_entries([_load(d)], rejections=rejections)
    assert entries == []
    assert any("2/3 sessions" in r for r in rejections)


def test_winner_non_ok_sample_rejected(tmp_path):
    bad = _steady_row(2, 7, api_ms=None, correct=False, err=None,
                      status="wrong-value", reason="err=1.0e-03")
    d = _write_matrix(tmp_path, extra_rows=(bad,))
    rejections = []
    entries = build_matrix_entries([_load(d)], rejections=rejections)
    assert entries == []
    assert any("non-ok sample" in r for r in rejections)


def test_candidate_budget_enforced(tmp_path):
    oracle_names = [f"oracle:m{i}_x" for i in range(25)]
    d = _write_matrix(tmp_path, extra_oracles=oracle_names)
    rejections = []
    entries = build_matrix_entries([_load(d)], rejections=rejections)
    assert entries == []
    assert any("candidates exceed" in r for r in rejections)


def test_family_illegality_rejected(tmp_path):
    rows = [_steady_row(s, t, flow_id="dedicated-hermitian-pack-v1")
            for s in range(3) for t in range(3)]
    d = _write_matrix(tmp_path, extra_rows=())
    # rewrite all winner rows with the illegal flow
    for s in range(3):
        path = d / f"raw_session{s}.jsonl"
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        for r in lines:
            r["flow_id"] = "dedicated-hermitian-pack-v1"
        path.write_text("".join(json.dumps(r) + "\n" for r in lines))
    summary = json.loads((d / "summary.json").read_text())
    summary["manifest"]["result_sha256"] = result_sha256(d)
    (d / "summary.json").write_text(json.dumps(summary))
    rejections = []
    entries = build_matrix_entries([_load(d)], rejections=rejections)
    assert entries == []
    assert any("not in the c2c family" in r for r in rejections)


def test_inconsistent_block_geometry_rejected(tmp_path):
    d = _write_matrix(tmp_path)
    path = d / "raw_session1.jsonl"
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    for r in lines:
        r["block"] = 512
    path.write_text("".join(json.dumps(r) + "\n" for r in lines))
    summary = json.loads((d / "summary.json").read_text())
    summary["manifest"]["result_sha256"] = result_sha256(d)
    (d / "summary.json").write_text(json.dumps(summary))
    rejections = []
    entries = build_matrix_entries([_load(d)], rejections=rejections)
    assert entries == []
    assert any("block geometry" in r for r in rejections)


def test_cross_semantic_rows_cannot_answer_another_semantic(tmp_path):
    # a c2r case may only be answered from c2r-labeled rows; c2c-forward
    # evidence can never be reused for it (separate label key).
    d = _write_matrix(tmp_path,
                      cases=[_case(label="N1024B256:c2r", semantic="c2r")])
    run = _load(d)
    rejections = []
    entries = build_matrix_entries([run], rejections=rejections)
    assert entries == []
    assert any("no measured oracle candidates" in r
               for r in rejections)


@pytest.mark.parametrize("sem,tt,tdir", [
    ("c2c-forward", "c2c", "forward"),
    ("c2c-inverse", "c2c", "inverse"),
    ("r2c", "r2c", "forward"),
    ("c2r", "c2r", "inverse"),
])
def test_each_semantic_builds_its_own_exact_key(tmp_path, sem, tt, tdir):
    label = f"N1024B256:{sem}"
    d = _write_matrix(tmp_path, label=label, semantic=sem)
    entries = build_matrix_entries([_load(d)])
    assert len(entries) == 1
    e = entries[0]
    assert (e["transform_type"], e["transform_direction"],
            e["normalization"]) == (tt, tdir, "backward")
    assert e["evidence"]["semantic"] == sem
    # legal for its own family (core flows today, PR-H joins later)
    from src.xpu.semantic import family_allows
    assert family_allows(tt, e["flow_id"])
    assert entry_contract_violations(e) == []
