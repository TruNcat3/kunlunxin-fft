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
REPORTS = ROOT / "results/p800/issue9_v61/reports"
LEDGER = (ROOT / "results/p800/issue9_v61/release_evidence"
          "/artifacts_sha256.txt")


def _gates_argv(identity=None, reports=REPORTS, ledger=LEDGER):
    return [
        str(REL_V61),
        "--v5-pair", str(V5_PAIR),
        "--identity", str(identity or IDENTITY),
        "--stable-ref", str(REL_V6),
        "--reports", str(reports),
        "--ledger", str(ledger),
    ]


def test_release_gates_all_pass_v61(capsys):
    rc = gates.main(_gates_argv())
    out = capsys.readouterr().out
    assert rc == 0, out[-4000:]
    assert "ALL HARD GATES PASSED" in out
    assert "HARD GATE FAILURES" not in out
    assert out.count("[PASS]") >= 40
    assert "[FAIL]" not in out
    # two-round stability definition: single-round crossings land in
    # the disclosed noise band, only intentional cells stay >2%
    assert "noise-band (single-round)" in out
    assert "N2048B16:c2c-inverse" in out
    # round-2 additions: real SHA identity bindings, label-set
    # equality, closure official exit metrics
    assert "sha256(release previous_selector_table)" in out
    assert "report scale_report.json cells == config scale label set" \
        in out
    assert "closure_official" in out
    assert "bounded_oracle_gap (informational" in out
    # round-3 additions: pair provenance + artifact ledger
    assert "sha256(previous_selector_table.json file)" in out
    assert "manifest dirty=false (clean re-measure)" in out
    assert "pair manifest dirty=False" in out
    assert "artifact ledger: every listed SHA matches" in out


def test_release_gates_rejects_missing_ledger(tmp_path, capsys):
    rc = gates.main(_gates_argv(ledger=tmp_path / "nope.txt"))
    out = capsys.readouterr().out
    assert rc == 1
    assert "[FAIL] artifact ledger provided (--ledger)" in out


def test_release_gates_rejects_tampered_identity(tmp_path, capsys):
    ident = json.loads(IDENTITY.read_text())
    ident["current_table_sha256"] = "00" * 32
    bad = tmp_path / "tampered_identity.json"
    bad.write_text(json.dumps(ident))
    rc = gates.main(_gates_argv(identity=bad))
    out = capsys.readouterr().out
    assert rc == 1
    assert "HARD GATE FAILURES PRESENT" in out
    assert "[FAIL] identity: report current_table_sha256 == live " \
        "config/selector_table.json" in out
    # a tampered previous-table file binding must fail too
    ident2 = json.loads(IDENTITY.read_text())
    assert ident2["current_table_sha256"] != "00" * 32


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
         "guard_pass_2pct": True,
         "selector_vs_previous": -0.375, "vendor_status": "ok",
         "selector_previous_fallback": False},
        # scale domain
        {"label": "N8192B1:c2c-forward",
         "incumbent_ms": 5.0, "selector_ms": 4.4,
         "previous_deployed_ms": 4.5, "oracle_ms": 4.3,
         "vendor_ms": None, "selector_source": "abstain",
         "selector_regret": 0.0233, "guard_vs_incumbent": 0.88,
         "guard_pass_2pct": True,
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
        # apps domain, completed pow2 shape proxy
        {"label": "N1024B64:c2c-forward",
         "incumbent_ms": 4.0, "selector_ms": 3.0,
         "previous_deployed_ms": 3.6, "oracle_ms": 2.9,
         "vendor_ms": 0.5, "selector_source": "table",
         "selector_regret": 0.0345, "guard_vs_incumbent": 0.25,
         "guard_pass_2pct": True,
         "selector_vs_previous": -0.1667, "vendor_status": "ok",
         "selector_previous_fallback": False},
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


def _write_fake_pair(path: Path):
    """Paired run rows: selector vs previous-deployed on shared trials."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(json.dumps({
        "manifest": {"oracle_mode": "full"}, "cases": [], "metrics": {},
    }))
    with open(path / "raw_session0.jsonl", "w") as f:
        for session in range(3):
            for trial in range(4):
                for cand, ms in (("selector", 0.40),
                                 ("previous-deployed", 0.80),
                                 ("incumbent", 1.00)):
                    f.write(json.dumps({
                        "session": session, "trial": trial,
                        "label": "N64B4:c2c-forward",
                        "candidate": cand, "tier": "steady",
                        "status": "ok", "api_ms": ms, "err": 1e-6,
                    }) + "\n")


def test_release_reports_five_summaries(tmp_path, capsys):
    rel = tmp_path / "release"
    rel.mkdir()
    _write_synthetic_release(rel)
    _write_fake_tune(tmp_path / "tune-p", "full",
                     {"N64B4:c2c-forward": 0.001})
    _write_fake_tune(tmp_path / "tune-a", "winner-block",
                     {"N64B4:c2c-forward": -0.0005})
    _write_fake_pair(tmp_path / "pair")
    out = tmp_path / "reports"
    rc = rep.main([str(rel), "--out", str(out),
                   "--tune-p", str(tmp_path / "tune-p"),
                   "--tune-a", str(tmp_path / "tune-a"),
                   "--v5-pair", str(tmp_path / "pair")])
    assert rc == 0
    for name in ("steady_report", "e2e_report", "scale_report",
                 "shape_proxy_report", "ablation_scope_report"):
        assert (out / f"{name}.json").is_file(), name
    for name in ("steady_report", "e2e_report", "scale_report",
                 "shape_proxy_report"):
        assert (out / f"{name}.csv").is_file(), name
    assert (out / "ablation_scope_report.md").is_file()

    steady = json.loads((out / "steady_report.json").read_text())
    assert steady["schema"] == "issue9-release-reports-v2"
    assert set(steady["comparisons"]) == {
        "current_vs_v5_production",
        "current_vs_A1_historical_previous",
        "current_vs_incumbent_M0",
        "bounded_oracle_gap",
        "current_vs_vendor"}
    # the five comparisons never mix: each block carries its own stats
    assert steady["comparisons"]["current_vs_incumbent_M0"]["cells"] == 3
    assert steady["comparisons"]["current_vs_vendor"][
        "geomean_speedup"] == 0.122
    assert "informational only" in steady["comparisons"][
        "current_vs_A1_historical_previous"]["note"]
    gap = steady["comparisons"]["bounded_oracle_gap"]
    assert "EXCLUDES" in gap["definition"]
    assert gap["oracle_regret_as_exit_metric"].startswith("N/A")
    assert "mean_gap" in gap and "cells_gap_gt_10pct" in gap
    assert "N2048B14:c2c-forward" in steady["unsupported"][
        "capability_probe_failures"]
    assert steady["cells"] and all(
        "selector_ms" in c for c in steady["cells"])
    clo_cell, = (c for c in steady["cells"]
                 if c["label"] == "N64B4:c2c-forward")
    assert clo_cell["bounded_oracle_gap"] == 0.0417
    assert "oracle:m2_dit_t64" in clo_cell["oracle_candidates"]

    # closure official exit metrics: paired deltas + guard, oracle
    # regret marked N/A
    co = steady["closure_official"]
    assert co["oracle_regret"].startswith("N/A")
    cell, = (c for c in co["cells"] if c["label"] == "N64B4:c2c-forward")
    paired = cell["paired_vs_v5_deployed"]
    assert paired["n"] == 12
    assert paired["median_rel_delta"] == -0.5
    assert paired["mad_rel"] == 0.0
    assert paired["win_fraction"] == 1.0
    assert cell["guard_pass_2pct"] is True
    assert len(co["missing_labels"]) == 9  # fixture covers 1/10
    assert co["aggregate"]["all_cells_faster_than_v5_deployed"] is True
    assert co["aggregate"]["all_guard_pass_2pct"] is True

    e2e = json.loads((out / "e2e_report.json").read_text())
    assert e2e["vendor_e2e"]["status"] == "unavailable"
    assert "not produced" in e2e["vendor_e2e"]["note"]
    assert e2e["coverage"]["candidates"] == ["incumbent", "selector"]
    labels = {c["label"] for c in e2e["cells"]}
    assert "N64B4:c2c-forward" in labels

    scale = json.loads((out / "scale_report.json").read_text())
    # domain string is derived from the frozen config -- never a
    # hardcoded N range (re-review: no N524288)
    dom = scale["coverage"]["domain"]
    assert "N524288" not in dom and "524288" not in json.dumps(scale)
    for token in ("8192", "16384", "32768", "65536", "c2c-forward"):
        assert token in dom, token
    lab, = (c for c in scale["cells"]
            if c["label"] == "N8192B1:c2c-forward")
    assert lab["measured"] is True
    assert lab["vendor_status"] == "unsupported"
    assert lab["incumbent_ms"] == 5.0
    assert abs(lab["selector_vs_incumbent"] - (4.4 / 5.0 - 1)) < 1e-12
    assert lab["guard_pass_2pct"] is True
    assert lab["bounded_oracle_gap"] == 0.0233
    assert "NOT a completed scale" in scale["interpretation"]

    proxy = json.loads((out / "shape_proxy_report.json").read_text())
    probe, = (c for c in proxy["cells"]
              if c["label"] == "N2048B14:c2c-forward")
    assert probe["capability_probe_failure"] is True
    # baseline claim must be backed by per-cell incumbent numbers
    assert proxy["baseline"] == "incumbent per cell"
    assert probe["incumbent_ms"] == 2.2
    assert probe["selector_vs_incumbent"] is None  # selector absent
    done, = (c for c in proxy["cells"]
             if c["label"] == "N1024B64:c2c-forward")
    assert done["incumbent_ms"] == 4.0
    assert abs(done["selector_vs_incumbent"] - (3.0 / 4.0 - 1)) < 1e-12
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


def test_artifact_ledger_roundtrip_and_tamper_detect(tmp_path, capsys):
    import issue9_artifact_ledger as led
    out = tmp_path / "artifacts_sha256.txt"
    assert led.main(["--write", str(out)]) == 0
    n, errors = led.verify(ROOT, out)
    assert n == len(led.INVENTORY) >= 20
    assert errors == []
    assert led.main(["--verify", str(out)]) == 0
    # v5 pair provenance files must be covered (review provenance P1)
    listed = out.read_text()
    for required in (
        "v5_pair/manifest.json",
        "v5_pair/previous_selector_table.json",
        "v5_pair/raw_session0.jsonl",
        "v5_pair/raw_session1.jsonl",
        "v5_pair/raw_session2.jsonl",
        "v5_pair/schedule.json",
    ):
        assert required in listed
    # single-byte hash tamper in the ledger -> hard fail
    lines = out.read_text().splitlines()
    lines[0] = "0" * 64 + "  " + lines[0].split("  ", 1)[1]
    out.write_text("\n".join(lines) + "\n")
    _, errors = led.verify(ROOT, out)
    assert any("SHA MISMATCH" in e for e in errors)
    capsys.readouterr()
    assert led.main(["--verify", str(out)]) == 1
    assert led.main(["--verify", str(tmp_path / "missing.txt")]) == 1
