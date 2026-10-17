"""Host tests for scripts/diagnose_failing_cells.py (issue #6 PR-E).

Verdict ladder and summary/raw-row stages are pure host logic: no
device, no torch. The device timing stage is exercised separately on
device runs of the script itself.
"""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "diagnose_failing_cells", ROOT / "scripts" / "diagnose_failing_cells.py")
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def _case(label="N1024B64:c2c-forward", n=1024, b=64,
          sem="c2c-forward", sp=0.03, vs="ok", sel=3.0, ven=0.1,
          reg=1.05, src="abstain"):
    return {
        "label": label, "N": n, "batch": b, "semantic": sem,
        "speedup_vs_vendor": sp, "vendor_status": vs,
        "selector_ms": sel, "vendor_ms": ven, "incumbent_ms": sel,
        "selector_regret": reg, "selector_source": src,
        "oracle_ms": sel / reg if reg else sel,
        "oracle_candidate": "oracle:m2_dit_t64",
        "guard_pass_2pct": None, "sessions_with_data": 3,
    }


def test_classify_loses_tie_wins():
    assert diag.classify(_case(sp=0.5), [])["kind"] == "loses"
    assert diag.classify(_case(sp=1.0), [])["kind"] == "tie"
    assert diag.classify(_case(sp=2.0), [])["kind"] == "wins"
    assert diag.classify(_case(sp=1.021), [])["kind"] == "wins"
    assert diag.classify(_case(sp=0.981), [])["kind"] == "tie"


def test_classify_vendor_unsupported():
    c = _case(vs="unsupported", sp=None)
    failures = [{"label": c["label"], "kind": "vendor-unsupported",
                 "tier": "correctness", "candidate": "vendor",
                 "reason": "NOT IMPLEMENTED"}]
    assert diag.classify(c, failures)["kind"] == "vendor-unsupported"


def test_classify_wrong_value_first():
    c = _case()
    failures = [{"label": c["label"], "kind": "wrong-value",
                 "tier": "correctness", "candidate": "selector",
                 "reason": "err=1e-2"}]
    assert diag.classify(c, failures)["kind"] == "correctness"


def test_pick_cells_orders_and_filters():
    cases = [
        _case("N64B1:c2c-forward", n=64, b=1, sp=0.058),
        _case("N1024B256:c2c-forward", n=1024, b=256, sp=0.022),
        _case("N4096B1:c2c-forward", n=4096, b=1, sp=None,
              vs="unsupported"),
        _case("N2048B16:c2c-forward", n=2048, b=16, sp=2.4),
    ]
    picked = diag.pick_cells(cases, [], top_k=2)
    assert [c["label"] for c in picked] == [
        "N1024B256:c2c-forward", "N64B1:c2c-forward"]
    # unsupported vendor cell is never auto-picked
    assert all(c["vendor_status"] == "ok" for c in picked)


def test_pick_cells_correctness_priority():
    cases = [_case("N64B1:c2c-forward", n=64, b=1, sp=0.058),
             _case("N128B4:c2c-forward", n=128, b=4, sp=0.01)]
    failures = [{"label": "N64B1:c2c-forward", "kind": "wrong-value",
                 "tier": "correctness", "candidate": "selector",
                 "reason": "err"}]
    picked = diag.pick_cells(cases, failures, top_k=1)
    assert picked[0]["label"] == "N64B1:c2c-forward"


def test_flow_sensitivity_from_raw(tmp_path):
    label = "N512B16:c2c-forward"
    rows = []
    for cand, ms in (("selector", 2.0), ("incumbent", 2.0),
                     ("oracle:m2_dit_t64", 1.0), ("oracle:m5_dit", 1.5),
                     ("vendor", 0.2)):
        for i in range(5):
            rows.append({"label": label, "tier": "steady",
                         "candidate": cand, "api_ms": ms + i * 0.01,
                         "session": 0})
    rows.append({"label": "N512B16:c2c-inverse", "tier": "steady",
                 "candidate": "oracle:m2_dit_t64", "api_ms": 99.0})
    (tmp_path / "raw_session0.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n")
    fs = diag.flow_sensitivity(
        _case(label=label, n=512, b=16), tmp_path)
    assert fs["available"]
    assert fs["best"] == "oracle:m2_dit_t64"
    # medians of the sampled ladders: 2.02 / 1.02
    assert abs(fs["regret_vs_best"] - 2.02 / 1.02) < 1e-9
    # spread = (1.52 - 1.02) / 1.02
    assert abs(fs["spread_rel"] - 0.5 / 1.02) < 1e-9


def test_flow_sensitivity_absent_raw(tmp_path):
    fs = diag.flow_sensitivity(_case(), tmp_path / "nope")
    assert fs["available"] is False


def test_unit_sensitivity_with_composition_entry():
    cases = [_case(sel=1.5, ven=0.66)]
    table = {"entries": [
        {"N": 1024, "batch": 64, "direction": "dit",
         "precision": "fp32", "flow_id": "hybrid-A1P1",
         "candidate_ms": 0.2, "incumbent_ms": 1.5, "guard_pct": 86.0}]}
    us = diag.unit_sensitivity(cases[0], cases, table)
    assert us["available"]
    assert abs(us["selector_over_vendor"] - 1.5 / 0.66) < 1e-9
    assert us["composition_entry"]["flow_id"] == "hybrid-A1P1"
    us2 = diag.unit_sensitivity(cases[0], cases, None)
    assert us2["composition_entry"] is None


def test_batch_scaling_ratio():
    cases = [
        _case("N512B1:c2c-forward", n=512, b=1, sp=0.20),
        _case("N512B16:c2c-forward", n=512, b=16, sp=0.10),
        _case("N512B64:c2c-forward", n=512, b=64, sp=0.02),
        _case("N256B64:c2c-forward", n=256, b=64, sp=0.9),  # other N
    ]
    bs = diag.batch_scaling(cases[0], cases)
    assert bs["available"]
    assert bs["worst_B"] == 64 and bs["best_B"] == 1
    assert abs(bs["ratio_worst_over_best"] - 0.1) < 1e-9


def test_batch_scaling_single_point():
    bs = diag.batch_scaling(_case(), [_case()])
    assert bs["available"] is False


def test_ledger_formulas_and_unmodelled():
    m2 = diag.ledger({"flow_id": "oracle:m2_dit_t64", "n": 1024,
                      "batch": 64})
    # 2 + 10 - 6 = 6 launches, bytes = 6*2*1024*64*8
    assert m2["available"] and m2["launches"] == 6
    assert m2["gm_bytes"] == 6 * 2 * 1024 * 64 * 8
    m5 = diag.ledger({"flow_id": "oracle:m5_dit", "n": 1024,
                      "batch": 1})
    assert m5["launches"] == (10 + 1) // 2
    unk = diag.ledger({"flow_id": "hybrid-A1P1", "n": 2048,
                       "batch": 16})
    assert unk["available"] is False and "no launch model" in unk["why"]


def _stage(**over):
    base = {
        "classification": {"kind": "loses"},
        "dispatch_probe": {},
        "batch_scaling": {"available": False},
        "flow_sensitivity": {"available": False},
        "unit_sensitivity": {"available": False},
    }
    base.update(over)
    return base


def test_verdict_correctness_precedence():
    v = diag.verdict(_stage(
        classification={"kind": "correctness"},
        dispatch_probe={"dispatch_share": 0.5}))
    assert v["verdict"] == "correctness"


def test_verdict_external_capability():
    v = diag.verdict(_stage(classification={"kind": "vendor-unsupported"}))
    assert v["verdict"] == "external-capability(vendor)"


def test_verdict_dispatch_overhead():
    v = diag.verdict(_stage(
        dispatch_probe={"dispatch_share": 0.25}))
    assert v["verdict"] == "entry-dispatch-overhead"
    v2 = diag.verdict(_stage(
        dispatch_probe={"dispatch_share": 0.11}))
    assert v2["verdict"] == "entry-dispatch-overhead"


def test_verdict_sync_jitter():
    v = diag.verdict(_stage(
        dispatch_probe={"dispatch_share": 0.01, "p95_over_p50": 1.4}))
    assert v["verdict"] == "sync-handoff-jitter"


def test_verdict_worker_waves():
    v = diag.verdict(_stage(
        batch_scaling={"available": True,
                       "ratio_worst_over_best": 0.2}))
    assert v["verdict"] == "worker-waves-tail"


def test_verdict_flow_not_selected():
    v = diag.verdict(_stage(
        flow_sensitivity={"available": True, "regret_vs_best": 1.4,
                          "best": "oracle:m2_dit_t64"}))
    assert v["verdict"] == "flow-not-selected"


def test_verdict_processing_unit_bound():
    v = diag.verdict(_stage(
        flow_sensitivity={"available": True, "regret_vs_best": 1.05},
        unit_sensitivity={"available": True,
                          "selector_over_vendor": 15.0}))
    assert v["verdict"] == "processing-unit-bound"


def test_verdict_mixed_when_no_rungh_fires():
    v = diag.verdict(_stage(
        dispatch_probe={"dispatch_share": 0.02, "p95_over_p50": 1.05},
        flow_sensitivity={"available": True, "regret_vs_best": 1.02},
        unit_sensitivity={"available": True,
                          "selector_over_vendor": 1.1}))
    assert v["verdict"] == "mixed"


def test_verdict_ladder_order_is_contract():
    v = diag.verdict(_stage())
    assert v["ladder_order"] == list(diag.LADDER)
    labels = [r["label"] for r in v["rungs"]]
    assert labels == list(diag.LADDER[:-1])


def test_pick_cells_reads_real_release_summary():
    """Smoke against the real PR-D snapshot when present."""
    releases = sorted((ROOT / "results" / "p800" / "releases")
                      .glob("*/summary.json"))
    if not releases:
        import pytest
        pytest.skip("no release summaries")
    summary = json.loads(releases[-1].read_text())
    picked = diag.pick_cells(summary["cases"], summary["failures"], 4)
    assert 1 <= len(picked) <= 4
    for c in picked:
        assert c.get("vendor_status") == "ok"
        assert c.get("speedup_vs_vendor") is not None
    # gzip raw discovery works on the real snapshot when shipped
    gz = list(releases[-1].parent.glob("raw_session*.jsonl.gz"))
    if gz:
        stats = diag._load_raw_p50(releases[-1].parent, picked[0]["label"])
        assert "selector" in stats and stats["selector"]["n"] > 0
