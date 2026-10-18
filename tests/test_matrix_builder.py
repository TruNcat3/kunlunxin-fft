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
from src.xpu.manifest_contract import (  # noqa: E402
    cases_sha256,
    result_sha256,
)
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


def _steady_row(session, trial, candidate=WINNER,
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
        # runner closure (issue #8 P1-A): every steady row records the
        # runner that produced the sample
        "executed": "fft_dit_batch",
        "api_ms": 1.0,
        "correct": True,
        "err": 1e-6,
        "status": "ok",
        "reason": None,
        "notes": None,
    }
    row.update(over)
    return row


def _derive_case(label, semantic, rows, n=1024, b=256):
    """Self-consistent summary case: the exact aggregation of the rows.

    Mirrors run_semantic_matrix._aggregate so a fixture summary always
    agrees with its raw evidence (issue #8 P1-C: the builder refuses a
    summary that its raw sessions do not produce).
    """
    steady = {}
    for r in rows:
        if (r.get("tier") == sm.TIER_STEADY and r.get("status") == "ok"
                and r.get("api_ms") is not None):
            steady.setdefault((r["label"], r["candidate"]), []).append(
                r["api_ms"])
    med, mad = {}, {}
    for (lab, cand), xs in steady.items():
        if lab == label:
            med[cand] = sm._median(xs)
            mad[cand] = sm._mad(xs)
    oracle_keys = [k for k in med if k.startswith("oracle:")]
    oracle = min((med[k] for k in oracle_keys), default=None)
    oracle_name = (min(oracle_keys, key=lambda k: med[k])
                   if oracle_keys else None)
    sel = med.get("selector")
    return {
        "label": label, "N": n, "batch": b, "semantic": semantic,
        "direction": "dit",
        "selector_ms": sel,
        "incumbent_ms": med.get("incumbent"),
        "vendor_ms": med.get("vendor"),
        "oracle_ms": oracle,
        "selector_mad_ms": mad.get("selector"),
        "incumbent_mad_ms": mad.get("incumbent"),
        "vendor_mad_ms": mad.get("vendor"),
        "oracle_mad_ms": (min((mad[k] for k in oracle_keys), default=None)
                          if oracle_keys else None),
        "oracle_candidate": oracle_name,
        "selector_regret": ((sel / oracle - 1.0) if (sel and oracle)
                            else None),
        "sessions_with_data": len({
            r["session"] for r in rows
            if r["label"] == label and r["candidate"] == "selector"
            and r.get("tier") == sm.TIER_STEADY
            and r.get("status") == "ok"}),
    }


def _no_rows_case(label, semantic, n=1024, b=256):
    """A case with zero measured rows: every median field is None."""
    case = _derive_case(label, semantic, [], n=n, b=b)
    return case


def _write_matrix(tmp_path, *, cases=None, sessions=(0, 1, 2),
                  per_session_rows=30, extra_rows=(), extra_oracles=(),
                  omit_result_sha=False, name="matrix_stress",
                  label=LABEL, semantic=SEM,
                  winner_ms=1.00, incumbent_ms=2.20, selector_ms=None,
                  vendor_ms=4.0):
    selector_ms = incumbent_ms if selector_ms is None else selector_ms
    d = tmp_path / name
    d.mkdir()
    rows = []
    for s in sessions:
        for trial in range(per_session_rows):
            rows.append(_steady_row(s, trial, candidate=WINNER,
                                    api_ms=winner_ms, label=label))
            rows.append(_steady_row(
                s, 100 + trial, candidate="incumbent",
                api_ms=incumbent_ms, label=label,
                flow_id="baseline-global-stage",
                executed="fft_dit_batch"))
            rows.append(_steady_row(
                s, 200 + trial, candidate="selector",
                api_ms=selector_ms, label=label,
                flow_id="baseline-global-stage",
                executed="fft_dit_batch",
                params={"variant": "folded", "batch": 256,
                        "selection_source": "abstain", "fallback": True}))
            rows.append(_steady_row(
                s, 300 + trial, candidate="vendor",
                api_ms=vendor_ms, label=label,
                flow_id="vendor", executed="vendor-torch-fft",
                processing_unit="vendor-torch-fft",
                params={"selection_source": "vendor", "fallback": False}))
        for cand in extra_oracles:
            for trial in range(per_session_rows):
                rows.append(_steady_row(s, 400 + trial, candidate=cand,
                                        api_ms=1.5, label=label))
    rows.extend(extra_rows)
    by_session = {}
    for r in rows:
        by_session.setdefault(r["session"], []).append(r)
    for s, srows in sorted(by_session.items()):
        with open(d / f"raw_session{s}.jsonl", "w") as fh:
            for r in srows:
                fh.write(json.dumps(r) + "\n")
    if cases is None:
        cases = [_derive_case(label, semantic, rows)]
    man = _manifest(d, [label])
    if not omit_result_sha:
        man["result_sha256"] = result_sha256(d)
    man["cases_sha256"] = cases_sha256(cases)
    payload = {"manifest": man,
               "cases": cases,
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
    # runner closure (issue #8 P1-A): the measured runner is recorded
    assert e["executed"] == "fft_dit_batch"
    assert e["evidence"]["executed"] == "fft_dit_batch"
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
    # winner median 2.19 vs incumbent 2.20: only 0.45% gain < 2% guard
    d = _write_matrix(tmp_path, winner_ms=2.19)
    rejections = []
    entries = build_matrix_entries([_load(d)], rejections=rejections)
    assert entries == []
    assert any("guard not cleared" in r for r in rejections)
    assert any("legacy/abstain path stays" in r for r in rejections)


def test_incomplete_winner_sessions_refused_at_load(tmp_path):
    # all three session files exist (contract-clean) but the winner
    # only has timing in sessions 0-1: the P1-C completeness gate
    # refuses the artifact before any cell gate runs.
    d = _write_matrix(tmp_path, sessions=(0, 1))
    s2 = []
    for trial in range(30):
        s2.append(_steady_row(2, 100 + trial, candidate="incumbent",
                              api_ms=2.20))
        s2.append(_steady_row(
            2, 200 + trial, candidate="selector", api_ms=2.20,
            flow_id="baseline-global-stage", executed="fft_dit_batch",
            params={"variant": "folded", "batch": 256,
                    "selection_source": "abstain", "fallback": True}))
        s2.append(_steady_row(
            2, 300 + trial, candidate="vendor", api_ms=4.0,
            flow_id="vendor", executed="vendor-torch-fft",
            processing_unit="vendor-torch-fft",
            params={"selection_source": "vendor", "fallback": False}))
    with open(d / "raw_session2.jsonl", "w") as fh:
        for r in s2:
            fh.write(json.dumps(r) + "\n")
    summary = json.loads((d / "summary.json").read_text())
    summary["manifest"]["result_sha256"] = result_sha256(d)
    (d / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(Exception) as ei:
        _load(d)
    assert "completeness" in str(ei.value)


def test_winner_non_ok_sample_rejected(tmp_path):
    bad = _steady_row(2, 999, api_ms=None, correct=False, err=None,
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
    # evidence can never be reused for it (separate label key). The
    # c2r case has zero rows, so its self-consistent summary is all
    # None and the cell gate refuses it.
    d = _write_matrix(
        tmp_path, cases=[_no_rows_case("N1024B256:c2r", "c2r")])
    run = _load(d)
    rejections = []
    entries = build_matrix_entries([run], rejections=rejections)
    assert entries == []
    assert any("non-finite incumbent/oracle median" in r
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
    assert e["executed"] == "fft_dit_batch"
    # legal for its own family (core flows today, PR-H joins later)
    from src.xpu.semantic import family_allows
    assert family_allows(tt, e["flow_id"])
    assert entry_contract_violations(e) == []


# --- runner closure (issue #8 P1-A) --------------------------------------

def test_inconsistent_executed_rejected(tmp_path):
    d = _write_matrix(tmp_path)
    path = d / "raw_session1.jsonl"
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    for r in lines:
        if r.get("candidate") == WINNER:
            r["executed"] = "fft_dif_matrix_batch"
    path.write_text("".join(json.dumps(r) + "\n" for r in lines))
    summary = json.loads((d / "summary.json").read_text())
    summary["manifest"]["result_sha256"] = result_sha256(d)
    (d / "summary.json").write_text(json.dumps(summary))
    rejections = []
    entries = build_matrix_entries([_load(d)], rejections=rejections)
    assert entries == []
    assert any("executed runner missing/inconsistent" in r
               for r in rejections)


# --- raw<->summary cross-check (issue #8 P1-C) ---------------------------

def test_summary_median_tamper_refused(tmp_path):
    d = _write_matrix(tmp_path)
    summary = json.loads((d / "summary.json").read_text())
    summary["cases"][0]["incumbent_ms"] = 9.99
    summary["manifest"]["cases_sha256"] = cases_sha256(summary["cases"])
    (d / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(Exception) as ei:
        _load(d)
    assert "disagree" in str(ei.value)


def test_summary_winner_tamper_refused(tmp_path):
    d = _write_matrix(tmp_path, extra_oracles=["oracle:other_m2"])
    summary = json.loads((d / "summary.json").read_text())
    assert summary["cases"][0]["oracle_candidate"] == WINNER
    summary["cases"][0]["oracle_candidate"] = "oracle:other_m2"
    summary["manifest"]["cases_sha256"] = cases_sha256(summary["cases"])
    (d / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(Exception) as ei:
        _load(d)
    assert "oracle_candidate" in str(ei.value)


def test_dirty_manifest_refused(tmp_path):
    d = _write_matrix(tmp_path)
    summary = json.loads((d / "summary.json").read_text())
    summary["manifest"]["dirty"] = True
    (d / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(Exception) as ei:
        _load(d)
    assert "dirty" in str(ei.value)


def test_dropped_trial_rows_break_completeness(tmp_path):
    d = _write_matrix(tmp_path)
    path = d / "raw_session0.jsonl"
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    kept = [r for r in lines if not (
        r.get("candidate") == WINNER and r.get("trial") == 29)]
    assert len(kept) == len(lines) - 1
    path.write_text("".join(json.dumps(r) + "\n" for r in kept))
    summary = json.loads((d / "summary.json").read_text())
    summary["manifest"]["result_sha256"] = result_sha256(d)
    (d / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(Exception) as ei:
        _load(d)
    assert "completeness" in str(ei.value)


def test_cases_sha256_gate_refuses_edited_cases(tmp_path):
    d = _write_matrix(tmp_path)
    summary = json.loads((d / "summary.json").read_text())
    summary["cases"][0]["vendor_ms"] = 1.0
    # deliberately NOT re-stamping cases_sha256: the strict gate must
    # catch the edit before the cross-check even runs
    (d / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(Exception) as ei:
        _load(d)
    assert "strict output gate" in str(ei.value)
    assert "cases_sha256" in str(ei.value)
