"""Issue #9 WP1: frozen config, unified launcher, dry-run budgets."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_issue9 as ri  # noqa: E402
import run_semantic_matrix as sm  # noqa: E402

CFG = ri.load_config()


def test_config_schema_and_frozen_domains():
    assert CFG["schema"] == "issue9-config-v1"
    assert CFG["issue"] == 9
    assert CFG["domain_lock"]["frozen"] is True
    assert CFG["device"]["env_hash"] == "6b50bd316032c0fe"
    assert CFG["device"]["profile_hash"] == "a78b86793c7142a0"
    assert (CFG["source"]["frozen_base_commit"]
            == "a94b8907f101792ed4afb08924e05e2c0bc74e94")
    m = CFG["measurement"]
    assert (m["sessions"], m["trials"], m["warmup"], m["seed"]) == (
        3, 30, 5, 1234)
    assert (m["one_shot_trials"], m["e2e_trials"]) == (10, 10)
    assert m["guard_pct"] == 2.0
    assert CFG["budgets"]["implementation_rounds"] == 2
    assert CFG["budgets"]["max_candidates_per_cell"] == 24
    assert CFG["budgets"]["device_hours_total"] == 16
    assert set(CFG["phases"]) == set(ri.PHASES)
    # every frozen domain parses and matches its declared cell count
    total = 0
    for name, dom in CFG["domains"].items():
        cells = sm.parse_cells_spec(",".join(dom["cells"]))
        n = sum(len(s) for _, _, s in cells)
        assert n == dom["cell_count"], name
        total += n
    assert total == 187


def test_frozen_domains_match_issue_tables():
    clo = sm.parse_cells_spec(
        ",".join(CFG["domains"]["closure"]["cells"]))
    assert [(n, b, tuple(s)) for n, b, s in clo] == [
        (64, 4, ("c2c-forward", "c2r")),
        (64, 16, ("c2c-forward", "c2r")),
        (512, 4, ("c2c-forward", "c2r")),
        (512, 16, ("c2c-forward", "c2r")),
        (512, 64, ("r2c", "c2r"))]
    assert CFG["domains"]["closure"]["cell_count"] == 10
    assert CFG["domains"]["tune_a"]["cell_count"] == 8
    assert CFG["domains"]["holdout"]["cell_count"] == 8
    assert CFG["domains"]["scale"]["cell_count"] == 12
    assert CFG["domains"]["apps"]["cell_count"] == 12
    # regression = the full v5 release domain, never narrowed
    reg = sm.parse_cells_spec(
        ",".join(CFG["domains"]["regression"]["cells"]))
    assert [(n, b) for n, b, _ in reg] == sm._scope_cells("all")
    assert len(reg) == 32 and all(len(s) == 4 for _, _, s in reg)
    assert CFG["domains"]["regression"]["cell_count"] == 128


def test_phase_cells_spec_and_schedule():
    assert ri.phase_cells_spec(CFG, "closure") == ",".join(
        CFG["domains"]["closure"]["cells"])
    assert ri.phase_cells_spec(CFG, "tune-a") == ri.phase_cells_spec(
        CFG, "tune-p")
    # validate = regression + holdout (dedup) + scale + apps;
    # cross-domain shape overlap merges (48 unique shapes)
    est_cells = sm.parse_cells_spec(ri.phase_cells_spec(CFG, "validate"))
    assert len(est_cells) == 48
    assert sum(len(s) for _, _, s in est_cells) == 144
    with pytest.raises(SystemExit, match="no measurement domains"):
        ri.phase_cells_spec({"phases": {"x": {"domains": []}}},
                            "x")


def test_estimate_phases_counts_and_anchors(capsys):
    est = ri.estimate_phase(CFG, "closure")
    assert est["shapes"] == 5 and est["semantic_cells"] == 10
    assert est["candidates"] > 0
    assert est["max_candidates_per_cell"] <= 24
    assert est["calls"] > 0
    assert est["device_hours"] > 0
    assert est["anchor_kinds"] == ["v5-exact"]
    assert est["anchor_missing_labels"] == 0

    # validate = regression (v5-exact) + scale + apps (scaled anchors)
    val = ri.estimate_phase(CFG, "validate")
    assert val["semantic_cells"] == 144
    assert "v5-exact" in val["anchor_kinds"]
    # N8192/N32768 are absent from v5 -> element-scaled;
    # OFDM B=14/140 and other unseen batches -> batch-scaled
    assert "v5-element-scaled" in val["anchor_kinds"]
    assert "v5-batch-scaled" in val["anchor_kinds"]
    assert val["anchor_missing_labels"] == 0
    assert val["device_hours"] > 0


def test_dry_run_measure_phase_reports_budget(capsys):
    rc = ri.main(["--phase", "closure", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "10 semantic cells" in out
    assert "candidates: " in out
    assert "calls: " in out
    assert "device-time:" in out
    assert "budget 16 h" in out
    assert "N64B4:c2c-forward+c2r" in out
    assert "run_state" not in out  # dry-run runs nothing


def test_dry_run_preflight_and_replay(capsys, tmp_path):
    rc = ri.main(["--phase", "preflight", "--dry-run",
                  "--output", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0 and "would check" in out

    rc = ri.main(["--phase", "replay", "--dry-run",
                  "--output", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "--aggregate-only" in out
    assert "--replay-out" in out
    assert str(tmp_path / "validate") in out


def test_measure_phase_spawns_with_device_and_state(tmp_path, capsys,
                                                    monkeypatch):
    calls = []

    def fake_spawn(cmd, env=None):
        calls.append((cmd, env))
        return 0

    monkeypatch.setattr(ri, "spawn", fake_spawn)
    out = str(tmp_path / "out")
    rc = ri.main(["--phase", "closure", "--device", "5",
                  "--output", out])
    assert rc == 0
    (cmd, env), = calls
    assert "--cells" in cmd
    cells_spec = cmd[cmd.index("--cells") + 1]
    assert cells_spec == ri.phase_cells_spec(CFG, "closure")
    assert "--sessions" in cmd and "3" in cmd
    assert "--out" in cmd
    assert env["CUDA_VISIBLE_DEVICES"] == "5"

    state = json.loads((tmp_path / "out" / "run_state.json").read_text())
    assert state["schema"] == ri.STATE_SCHEMA
    assert state["phases"]["closure"]["status"] == "done"
    assert state["phases"]["closure"]["returncode"] == 0
    assert state["phases"]["closure"]["device"] == "5"

    # completed phase is never re-run without --resume (state exists)
    with pytest.raises(SystemExit, match="--resume"):
        ri.main(["--phase", "closure", "--output", out])
    # --resume skips it without spawning
    rc = ri.main(["--phase", "closure", "--output", out, "--resume"])
    assert rc == 0
    assert len(calls) == 1
    assert "skipping closure" in capsys.readouterr().out


def test_failed_phase_records_and_resume_retries(tmp_path, capsys,
                                                 monkeypatch):
    rc_values = [7, 0]
    calls = []

    def fake_spawn(cmd, env=None):
        calls.append(cmd)
        return rc_values[len(calls) - 1]

    monkeypatch.setattr(ri, "spawn", fake_spawn)
    out = str(tmp_path / "out")
    rc = ri.main(["--phase", "profile", "--output", out])
    assert rc == 7
    state = json.loads((tmp_path / "out" / "run_state.json").read_text())
    assert state["phases"]["profile"]["status"] == "failed"
    assert state["phases"]["profile"]["returncode"] == 7

    rc = ri.main(["--phase", "profile", "--output", out, "--resume"])
    assert rc == 0
    assert len(calls) == 2
    state = json.loads((tmp_path / "out" / "run_state.json").read_text())
    assert state["phases"]["profile"]["status"] == "done"


def test_resume_refuses_when_config_changed(tmp_path, monkeypatch):
    monkeypatch.setattr(ri, "spawn", lambda cmd, env=None: 0)
    out = str(tmp_path / "out")
    ri.main(["--phase", "closure", "--output", out])
    state_path = tmp_path / "out" / "run_state.json"
    state = json.loads(state_path.read_text())
    state["config_sha256"] = "0" * 64
    state_path.write_text(json.dumps(state))
    with pytest.raises(SystemExit, match="config changed"):
        ri.main(["--phase", "profile", "--output", out, "--resume"])


def test_state_schema_fail_closed(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "run_state.json").write_text(
        json.dumps({"schema": "something-else"}))
    with pytest.raises(SystemExit, match="issue9-launch-state-v1"):
        ri.main(["--phase", "closure", "--output", str(out), "--resume"])


def test_replay_phase_spawns_new_replay_dir(tmp_path, monkeypatch):
    calls = []

    def fake_spawn(cmd, env=None):
        calls.append(cmd)
        return 0

    monkeypatch.setattr(ri, "spawn", fake_spawn)
    out = tmp_path / "out"
    src = out / "validate"
    src.mkdir(parents=True)
    (src / "manifest.json").write_text("{}")
    rc = ri.main(["--phase", "replay", "--output", str(out)])
    assert rc == 0
    cmd, = calls
    assert "--aggregate-only" in cmd
    assert cmd[cmd.index("--out") + 1] == str(src)
    assert cmd[cmd.index("--replay-out") + 1] == str(out / "replay")
    state = json.loads((out / "run_state.json").read_text())
    assert state["phases"]["replay"]["status"] == "done"


def test_replay_phase_requires_source(tmp_path, monkeypatch):
    monkeypatch.setattr(ri, "spawn",
                        lambda cmd, env=None: pytest.fail("must not run"))
    with pytest.raises(SystemExit, match="does not exist"):
        ri.main(["--phase", "replay", "--output", str(tmp_path / "out")])


def test_replay_phase_refuses_nonempty_destination(tmp_path,
                                                   monkeypatch):
    monkeypatch.setattr(ri, "spawn",
                        lambda cmd, env=None: pytest.fail("must not run"))
    out = tmp_path / "out"
    src = out / "validate"
    src.mkdir(parents=True)
    dst = out / "replay"
    dst.mkdir(parents=True)
    (dst / "summary.json").write_text("{}")
    with pytest.raises(SystemExit, match="not empty"):
        ri.main(["--phase", "replay", "--output", str(out)])


def test_preflight_env_hash_fail_closed(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(ri, "live_env_hash", lambda device: "deadbeef")
    rc = ri.main(["--phase", "preflight", "--device", "2",
                  "--output", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "[FAIL] env-hash" in out
    assert "preflight FAILED" in out
    state = json.loads((tmp_path / "run_state.json").read_text())
    assert state["phases"]["preflight"]["status"] == "failed"


def test_preflight_matches_config_identity(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(ri, "live_env_hash",
                        lambda device: CFG["device"]["env_hash"])
    rc = ri.main(["--phase", "preflight", "--device", "2",
                  "--output", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "preflight OK" in out
    assert "[ok] profile-hash" in out
    assert "[ok] fallback-binding" in out
    for dom in ("closure", "tune_a", "holdout", "regression", "scale",
                "apps", "profile_anchors"):
        assert f"[ok] domain:{dom}" in out
    state = json.loads((tmp_path / "run_state.json").read_text())
    assert state["phases"]["preflight"]["status"] == "done"
    assert state["phases"]["preflight"]["cmd"] is None


def test_load_config_rejects_wrong_schema(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "nope", "issue": 9}))
    with pytest.raises(SystemExit, match="issue9-config-v1"):
        ri.load_config(bad)
    bad.write_text(json.dumps({"schema": "issue9-config-v1", "issue": 8}))
    with pytest.raises(SystemExit, match="issue must be 9"):
        ri.load_config(bad)


def test_v5_anchor_model_rules():
    anchor = ri._v5_anchor(CFG)
    assert "N64B4:c2c-forward" in anchor
    ms, kind = ri._ms_for_call(anchor, 64, 4, "c2c-forward")
    assert kind == "v5-exact" and ms == anchor["N64B4:c2c-forward"]
    # same N, unseen batch -> linear batch scaling (4096 records
    # B=1/16 only in v5)
    ms_b, kind_b = ri._ms_for_call(anchor, 4096, 256, "c2c-forward")
    assert kind_b == "v5-batch-scaled"
    assert ms_b > 0
    # unseen N -> nearest recorded shape, linear in elements
    ms8, kind8 = ri._ms_for_call(anchor, 8192, 16, "c2c-forward")
    assert kind8 == "v5-element-scaled"
    assert ms8 > 0
    # no anchor data at all -> unavailable, never a crash
    ms, kind = ri._ms_for_call({}, 64, 4, "c2c-forward")
    assert ms is None and kind == "unavailable"


# ------------- issue #9 WP2: selection closure ----------------------------

_TRIPLE = {"c2c-forward": ("c2c", "forward", "backward"),
           "c2r": ("c2r", "inverse", "backward"),
           "r2c": ("r2c", "forward", "backward")}


def test_wp2_closure_domain_exact_entries_select_and_deploy():
    """WP2: every frozen closure cell is served by an exact semantic
    entry whose select() -> plan_fft() deployment matches the entry's
    recorded runner (declared -> entry -> plan -> deployed chain)."""
    from src.xpu.entry import plan_fft
    from src.xpu.selector import load_table, select
    import src.xpu.runner as runner_mod

    table = load_table(str(ROOT / "config" / "selector_table.json"))
    cells = sm.parse_cells_spec(ri.phase_cells_spec(CFG, "closure"))
    assert sum(len(s) for _, _, s in cells) == 10
    checked = 0
    for n, b, sems in cells:
        for sem in sems:
            tt, tdir, norm = _TRIPLE[sem]
            sel = select(direction="dit", n=n, batch=b,
                         precision=table["precision"],
                         profile_hash=table["profile_hash"], table=table,
                         env_hash=table["env_hash"], transform_type=tt,
                         transform_direction=tdir, normalization=norm)
            assert sel.entry_tier == "semantic", (n, b, sem, sel.entry_tier)
            assert sel.executed, (n, b, sem)
            plan = plan_fft(direction="dit", n=n, batch=b,
                            precision=table["precision"],
                            profile_hash=table["profile_hash"], table=table,
                            env_hash=table["env_hash"], transform_type=tt,
                            transform_direction=tdir, norm=norm)
            assert plan.executed == sel.executed
            assert plan.block == sel.block
            assert hasattr(runner_mod, plan.executed), plan.executed
            assert plan.fallback is False
            checked += 1
    assert checked == 10


def test_wp2_closure_entries_carry_guard_cleared_evidence():
    """Each promoted closure entry's evidence reason names its oracle
    winner and a >=2% guard win over the incumbent (issue #9 WP2)."""
    from src.xpu.selector import load_table, semantic_key_of

    table = load_table(str(ROOT / "config" / "selector_table.json"))
    cells = sm.parse_cells_spec(ri.phase_cells_spec(CFG, "closure"))
    guard = table["guard_pct"]
    assert guard == 2.0
    found = 0
    for n, b, sems in cells:
        for sem in sems:
            tt, tdir, norm = _TRIPLE[sem]
            entry = next(
                (e for e in table["entries"]
                 if e["N"] == n and e["batch"] == b and e["direction"] == "dit"
                 and semantic_key_of(e) == (tt, tdir, norm)),
                None)
            assert entry is not None, (n, b, sem, "missing exact entry")
            ev = entry["evidence"]
            assert ev["challenger"].startswith("oracle:"), ev
            assert ev["gain_pct"] >= guard, (n, b, sem, ev["gain_pct"])
            assert ev["sessions"] >= 3, ev
            assert ev["semantic"] == sem, ev
            assert ev["executed"] == entry["executed"], ev
            assert ev["protocol_id"] == CFG["protocol"]["protocol_id"]
            found += 1
    assert found == 10


# ------------- issue #9 WP3: same-source attribution -----------------------

def test_wp3_classify_rule_priority():
    """Disclosed classification rules pick exactly one class each."""
    import issue9_attribute as ia

    base = {"regret": 0.0, "steady_ms": 1.0, "dispatch_delta_ms": 0.1,
            "transfer_delta_ms": 0.1, "semantic": "c2c-forward",
            "flow_id": "local-fused-tile", "has_real_flow": False}
    # 1) selection still losing to the oracle wins first
    rec = dict(base, regret=0.42)
    assert ia.classify(rec) == ia.CLASS_SELECTOR
    # 2) per-call dispatch overhead >= 50% of steady
    rec = dict(base, dispatch_delta_ms=0.6)
    assert ia.classify(rec) == ia.CLASS_LAUNCH
    # 3) e2e transfer delta >= steady
    rec = dict(base, dispatch_delta_ms=0.0, transfer_delta_ms=1.5)
    assert ia.classify(rec) == ia.CLASS_GM
    # 4) dedicated real flow on r2c/c2r
    rec = dict(base, semantic="c2r", has_real_flow=True,
               flow_id="dedicated-c2r-hermitian-v1")
    assert ia.classify(rec) == ia.CLASS_REAL
    # 5) fused local tile core
    rec = dict(base)
    assert ia.classify(rec) == ia.CLASS_CORE
    # 6) worker-grid flows
    for flow in ("batch-spread", "batch-folded", "baseline-global-stage"):
        assert ia.classify(dict(base, flow_id=flow)) == ia.CLASS_TAIL
    # no counter-level evidence left -> honest fallback
    assert ia.classify(dict(base, flow_id=None)) == \
        "unclassified-need-counters"
    # vocabulary matches the issue's six classes
    assert set(ia.CLASSES) == {
        ia.CLASS_SELECTOR, ia.CLASS_LAUNCH, ia.CLASS_CORE,
        ia.CLASS_GM, ia.CLASS_TAIL, ia.CLASS_REAL}


def test_wp3_attribution_report_for_profile_run():
    """The recorded profile anchors produce a WP3 report: one class per
    anchor, measured tier medians, unavailable quantities disclosed."""
    import issue9_attribute as ia

    run = ROOT / "results" / "p800" / "issue9" / "profile"
    if not (run / "summary.json").is_file():
        pytest.skip("issue9 profile run not present")
    rep = ia.attribute(run)
    assert rep["schema"] == "issue9-attribution-v1"
    assert len(rep["anchors"]) == 9
    assert rep["unavailable_quantities"]
    by = {a["label"]: a for a in rep["anchors"]}
    # every anchor classified inside the issue's vocabulary
    for a in rep["anchors"]:
        assert a["attribution_class"] in set(ia.CLASSES) | {
            "unclassified-need-counters"}
        assert a["plan_creation_ms"] is not None, a["label"]
        assert a["steady_ms"] is not None, a["label"]
        assert a["geometry"]["executed"], a["label"]
    # measured classes on this run (same-source, WP2-closed selector)
    assert by["N64B16:c2c-forward"]["attribution_class"] == ia.CLASS_LAUNCH
    assert by["N1024B256:c2c-forward"]["attribution_class"] == ia.CLASS_CORE
    assert by["N512B64:r2c"]["attribution_class"] == ia.CLASS_REAL
    assert by["N16384B16:c2c-forward"]["attribution_class"] == ia.CLASS_TAIL
    # selection is no longer the bottleneck at any anchor (WP2 effect)
    assert all(a["regret"] is None or a["regret"] <= 0.10
               for a in rep["anchors"])
