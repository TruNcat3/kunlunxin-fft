"""Release-correction artifacts: gates + five-machine-reports (re-review P2).

Host-only. The gates test runs the canonical gate script against the
tracked immutable releases (v6-1 candidate vs the two-round stable
reference v6, plus the v5 identity/pair artifacts) and requires a
clean 30/30 pass. The reports test builds a synthetic release fixture
and checks every disclosure the re-review demanded.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import issue9_release_gates as gates  # noqa: E402
import issue9_release_reports as rep  # noqa: E402

REL_V61 = ROOT / "results/p800/releases/20261020-issue9-semantic-matrix-v6-1"
REL_V6 = ROOT / "results/p800/releases/20261020-issue9-semantic-matrix-v6"
V5_PAIR = ROOT / "results/p800/issue9_v61/v5_pair"
IDENTITY = ROOT / "results/p800/issue9_v61/v5_path_identity.json"


def test_release_gates_all_pass_v61(capsys):
    rc = gates.main([
        str(REL_V61),
        "--v5-pair", str(V5_PAIR),
        "--identity", str(IDENTITY),
        "--stable-ref", str(REL_V6),
    ])
    out = capsys.readouterr().out
    assert rc == 0, out[-4000:]
    assert "ALL HARD GATES PASSED" in out
    assert "HARD GATE FAILURES" not in out
    assert out.count("[PASS]") >= 30
    # two-round stability definition: single-round crossings land in
    # the disclosed noise band, only intentional cells stay >2%
    assert "noise-band (single-round)" in out
    assert "N2048B16:c2c-inverse" in out


def _write_synthetic_release(root: Path):
    manifest = {
        "sessions": 3, "trials": 30, "warmup": 5,
        "error_threshold": 1e-4,
        "protocol_id": "semantic-matrix-v1:plan-reuse:interleave:"
                       "correctness-first",
        "timing_scope": "api wall time (dispatch+alloc+layout+sync), ms; "
                        "tiers: plan_creation / steady / one_shot / e2e",
    }
    cases = [
        # closure + regression overlap
        {"label": "N64B4:c2c-forward",
         "incumbent_ms": 1.0, "selector_ms": 0.5,
         "previous_deployed_ms": 0.8, "oracle_ms": 0.48,
         "vendor_ms": 0.2, "selector_source": "table",
         "selector_regret": 0.0417, "guard_vs_incumbent": 0.5,
         "selector_vs_previous": -0.375, "vendor_status": "ok",
         "selector_previous_fallback": False},
        # scale domain
        {"label": "N8192B1:c2c-forward",
         "incumbent_ms": 5.0, "selector_ms": 4.4,
         "previous_deployed_ms": 4.5, "oracle_ms": 4.3,
         "vendor_ms": None, "selector_source": "abstain",
         "selector_regret": 0.0233, "guard_vs_incumbent": 0.88,
         "selector_vs_previous": -0.022, "vendor_status": "unsupported",
         "selector_previous_fallback": True},
        # apps domain, non-power-of-2 capability probe failure
        {"label": "N2048B14:c2c-forward",
         "incumbent_ms": 2.2, "selector_ms": None,
         "previous_deployed_ms": None, "oracle_ms": None,
         "vendor_ms": None, "selector_source": None,
         "selector_regret": None, "guard_vs_incumbent": None,
         "selector_vs_previous": None,
         "vendor_status": "unsupported",
         "selector_previous_fallback": None},
    ]
    metrics = {
        "previous_regressions_gt_2pct": ["N512B1:r2c"],
        "intentional_semantic_exceptions": ["N2048B16:c2c-inverse"],
        "mean_selector_regret": 0.0417, "max_selector_regret": 0.0417,
        "win": 4, "tie": 0, "loss": 102,
        "geomean_speedup_vs_vendor": 0.122,
    }
    (root / "summary.json").write_text(json.dumps({
        "manifest": manifest, "cases": cases, "metrics": metrics,
    }))
    rows = []
    for cand, ms, err in (("incumbent", 1.0, 1e-6), ("selector", 0.5, 1e-6),
                          ("previous_deployed", 0.8, 1e-6),
                          ("oracle:m2_dit_t64", 0.48, 1e-6),
                          ("vendor", 0.2, 1e-6)):
        rows.append({"label": "N64B4:c2c-forward", "tier": "steady",
                     "candidate": cand, "status": "ok", "api_ms": ms,
                     "err": err})
        rows.append({"label": "N64B4:c2c-forward", "tier": "e2e",
                     "candidate": cand if cand != "vendor" else "vendor",
                     "status": "ok", "api_ms": ms + 0.05, "err": err})
    rows.append({"label": "N64B4:c2c-forward", "tier": "one_shot",
                 "candidate": "selector", "status": "ok", "api_ms": 0.7,
                 "err": 1e-6})
    with open(root / "raw_session0.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _write_fake_tune(path: Path, oracle_mode: str, regrets):
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(json.dumps({
        "manifest": {"oracle_mode": oracle_mode},
        "cases": [{"label": lab, "selector_regret": reg}
                  for lab, reg in regrets.items()],
        "metrics": {},
    }))


def test_release_reports_five_summaries(tmp_path, capsys):
    rel = tmp_path / "release"
    rel.mkdir()
    _write_synthetic_release(rel)
    _write_fake_tune(tmp_path / "tune-p", "full",
                     {"N64B4:c2c-forward": 0.001})
    _write_fake_tune(tmp_path / "tune-a", "winner-block",
                     {"N64B4:c2c-forward": -0.0005})
    out = tmp_path / "reports"
    rc = rep.main([str(rel), "--out", str(out),
                   "--tune-p", str(tmp_path / "tune-p"),
                   "--tune-a", str(tmp_path / "tune-a")])
    assert rc == 0
    for name in ("steady_report", "e2e_report", "scale_report",
                 "shape_proxy_report", "ablation_scope_report"):
        assert (out / f"{name}.json").is_file(), name
    for name in ("steady_report", "e2e_report", "scale_report",
                 "shape_proxy_report"):
        assert (out / f"{name}.csv").is_file(), name
    assert (out / "ablation_scope_report.md").is_file()

    steady = json.loads((out / "steady_report.json").read_text())
    assert set(steady["comparisons"]) == {
        "current_vs_v5_production",
        "current_vs_A1_historical_previous",
        "current_vs_incumbent_M0",
        "current_vs_bounded_oracle",
        "current_vs_vendor"}
    # the five comparisons never mix: each block carries its own stats
    assert steady["comparisons"]["current_vs_incumbent_M0"]["cells"] == 2
    assert steady["comparisons"]["current_vs_vendor"][
        "geomean_speedup"] == 0.122
    assert "informational only" in steady["comparisons"][
        "current_vs_A1_historical_previous"]["note"]
    assert "N2048B14:c2c-forward" in steady["unsupported"][
        "capability_probe_failures"]
    assert steady["cells"] and all(
        "selector_ms" in c for c in steady["cells"])

    e2e = json.loads((out / "e2e_report.json").read_text())
    assert e2e["vendor_e2e"]["status"] == "unavailable"
    assert "not produced" in e2e["vendor_e2e"]["note"]
    assert e2e["coverage"]["candidates"] == ["incumbent", "selector"]
    labels = {c["label"] for c in e2e["cells"]}
    assert "N64B4:c2c-forward" in labels

    scale = json.loads((out / "scale_report.json").read_text())
    lab, = (c for c in scale["cells"]
            if c["label"] == "N8192B1:c2c-forward")
    assert lab["measured"] is True
    assert lab["vendor_status"] == "unsupported"
    assert "NOT a completed scale" in scale["interpretation"]

    proxy = json.loads((out / "shape_proxy_report.json").read_text())
    probe, = (c for c in proxy["cells"]
              if c["label"] == "N2048B14:c2c-forward")
    assert probe["capability_probe_failure"] is True
    assert "shape" in proxy["input_type"].lower()
    assert "padded-proxy" in proxy["input_type"]
    assert "no padded-proxy" in proxy["unsupported"]["note"] or \
        "NOT executed" in proxy["input_type"]
    assert any("non-power-of-2" in lim
               for lim in proxy["unsupported"]["note"].split(". "))

    ablation = json.loads((out / "ablation_scope_report.json").read_text())
    assert ablation["rounds"]["tune-p"]["oracle_mode"] == "full"
    assert ablation["rounds"]["tune-a"]["oracle_mode"] == "winner-block"
    assert ablation["rounds"]["tune-p"]["deployed_equals_oracle_cells"] == 1
    assert len(ablation["not_covered"]) >= 4
    assert any("batched M2" in x for x in ablation["not_covered"])
    assert "NOT a controlled A/P" in ablation["scope_statement"]
    md = (out / "ablation_scope_report.md").read_text()
    assert "ablation_scope" in md


def test_attribution_v2_vocabulary_on_profile_run(tmp_path):
    import issue9_attribute as attr
    rep_obj = attr.attribute(ROOT / "results/p800/issue9/profile")
    assert rep_obj["schema"] == "issue9-attribution-v2"
    assert "not direct causal" in rep_obj["evidence_status"]
    assert rep_obj["confidence_levels"] and set(
        rep_obj["confidence_levels"]) <= {"measured", "unverified"}
    for anchor in rep_obj["anchors"]:
        assert "hypothesis" in anchor and "confidence" in anchor
        assert anchor["hypothesis"] in rep_obj["hypotheses_used"]
        assert "attribution_class" not in anchor
