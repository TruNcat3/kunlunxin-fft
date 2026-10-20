#!/usr/bin/env python3
"""Issue #9 unified launcher (WP1): one frozen config, one entry point.

Every phase takes its exact cell schedule from config/issue9_p800.json
and delegates measurement to scripts/run_semantic_matrix.py --cells, so
no round depends on hand-assembled command lines:

    scripts/run_issue9.py --phase preflight --device 2
    scripts/run_issue9.py --phase closure --device 2 --dry-run
    scripts/run_issue9.py --phase validate --device 2 --resume

--dry-run prints declared cases/candidates/call counts and an analytic
device-time budget (anchored on the v5 recorded medians; never measured
evidence for the phase itself).  Phase results are tracked in
<output>/run_state.json; --resume continues past completed phases and
refuses to start when the config changed since that state was written.
"""

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG_PATH = ROOT / "config" / "issue9_p800.json"
CONFIG_SCHEMA = "issue9-config-v1"
STATE_SCHEMA = "issue9-launch-state-v1"
PHASES = ("preflight", "closure", "profile", "tune-a", "tune-p",
          "validate", "replay")
MEASURE_PHASES = ("closure", "profile", "tune-a", "tune-p", "validate")


def load_config(path=CONFIG_PATH):
    cfg = json.loads(Path(path).read_text())
    if cfg.get("schema") != CONFIG_SCHEMA:
        raise SystemExit(f"{path}: expected schema {CONFIG_SCHEMA!r}, "
                         f"got {cfg.get('schema')!r}")
    if cfg.get("issue") != 9:
        raise SystemExit(f"{path}: issue must be 9")
    return cfg


def config_sha256(path=CONFIG_PATH):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def phase_cells_spec(cfg, phase):
    """Frozen ``--cells`` schedule for a phase, straight from config.

    Multiple domains may share shapes (holdout/regression, scale/
    apps); the canonical schedule is parsed once so duplicates merge
    and the emitted spec is deterministic.
    """
    import run_semantic_matrix as sm

    domains = cfg["phases"][phase]["domains"]
    if not domains:
        raise SystemExit(f"phase {phase!r} declares no measurement domains")
    parts = []
    for dom in domains:
        parts.extend(cfg["domains"][dom]["cells"])
    cells = sm.parse_cells_spec(",".join(parts))
    out = []
    for n, b, sems in cells:
        if len(sems) == len(sm.SEMANTICS):
            out.append(f"N{n}B{b}")
        else:
            out.append(f"N{n}B{b}:" + "+".join(sems))
    return ",".join(out)


def _parse_spec(spec):
    import run_semantic_matrix as sm
    cells = sm.parse_cells_spec(spec)
    cases = list(dict.fromkeys((n, b) for n, b, _ in cells))
    sem_map = {(n, b): list(s) for n, b, s in cells}
    return cells, cases, sem_map


_ANCHOR_CACHE = {}


def _v5_anchor(cfg):
    """Per-label worst-role steady median from the v5 release summary.

    Used only as the analytic per-call cost anchor for --dry-run device
    time; never as evidence for the phase being planned.
    """
    path = ROOT / cfg["source"]["v5_release"] / "summary.json"
    key = str(path)
    if key in _ANCHOR_CACHE:
        return _ANCHOR_CACHE[key]
    tbl = {}
    if path.is_file():
        for c in json.loads(path.read_text()).get("cases", []):
            vals = [c.get(k) for k in ("incumbent_ms", "selector_ms",
                                       "oracle_ms", "previous_deployed_ms",
                                       "vendor_ms")]
            vals = [v for v in vals
                    if isinstance(v, (int, float)) and v > 0]
            if vals:
                tbl[c["label"]] = max(vals)
    _ANCHOR_CACHE[key] = tbl
    return tbl


def _ms_for_call(anchor, n, b, semantic):
    """Disclosed analytic per-call ms for a label.

    Exact v5 label -> recorded worst-role median; same N with a
    different batch -> nearest recorded batch scaled linearly in batch;
    otherwise -> nearest recorded shape scaled linearly in total
    elements (N x B).
    """
    key = f"N{n}B{b}:{semantic}"
    if key in anchor:
        return anchor[key], "v5-exact"
    by_nb = {}
    for lbl, ms in anchor.items():
        m = re.match(r"^N(\d+)B(\d+):", lbl)
        if m:
            k = (int(m.group(1)), int(m.group(2)))
            by_nb[k] = max(by_nb.get(k, 0.0), ms)
    if not by_nb:
        return None, "unavailable"
    same_n = [(bb, ms) for (nn, bb), ms in by_nb.items() if nn == n]
    if same_n:
        b0, ms0 = min(same_n, key=lambda t: abs(t[0] - b))
        return ms0 * (b / float(b0)), "v5-batch-scaled"
    prod = n * b
    n0, b0, ms0 = min(
        ((nn, bb, ms) for (nn, bb), ms in by_nb.items()),
        key=lambda t: abs(math.log2(t[0] * t[1] / float(prod))))
    return ms0 * (prod / float(n0 * b0)), "v5-element-scaled"


def phase_oracle_mode(cfg, phase):
    """Per-phase oracle mode (tune rounds differ; default: protocol)."""
    return (cfg.get("phases", {}).get(phase, {})
            .get("oracle_mode", cfg["protocol"]["oracle_mode"]))


def estimate_phase(cfg, phase):
    """Declared cases/candidates/calls + analytic device-time for --dry-run."""
    from src.xpu.config import DeviceProfile
    from src.xpu.selector import load_table
    import run_semantic_matrix as sm

    spec = phase_cells_spec(cfg, phase)
    cells, cases, sem_map = _parse_spec(spec)
    m = cfg["measurement"]
    inv = sm._declared_inventory(
        cases, load_table(str(ROOT / "config" / "selector_table.json")),
        phase_oracle_mode(cfg, phase), m["sessions"], m["trials"],
        m["warmup"], m["one_shot_trials"], m["e2e_trials"],
        DeviceProfile.load().hash(), sem_map=sem_map)

    labels = [c["label"] for c in inv["cells"]]
    candidates = sum(len(c["candidates"]) for c in inv["cells"])
    max_per_cell = max(len(c["candidates"]) for c in inv["cells"])
    calls = m["sessions"] * sum(
        v for c in inv["cells"] for cd in c["candidates"].values()
        for v in cd.values())

    anchor = _v5_anchor(cfg)
    device_s = 0.0
    kinds = set()
    missing = 0
    for n, b, sems in cells:
        for sem in sems:
            ms, kind = _ms_for_call(anchor, n, b, sem)
            if ms is None:
                missing += 1
                continue
            kinds.add(kind)
            lbl_calls = m["sessions"] * sum(
                v for c in inv["cells"] if c["label"] == f"N{n}B{b}:{sem}"
                for cd in c["candidates"].values() for v in cd.values())
            device_s += lbl_calls * ms / 1000.0

    n_sem_cells = sum(len(sems) for _, _, sems in cells)
    budget = cfg["budgets"]["max_candidates_per_cell"]
    return {
        "phase": phase,
        "oracle_mode": phase_oracle_mode(cfg, phase),
        "domains": cfg["phases"][phase]["domains"],
        "shapes": len(cells),
        "semantic_cells": n_sem_cells,
        "labels": labels,
        "candidates": candidates,
        "max_candidates_per_cell": max_per_cell,
        "candidate_budget_per_cell": budget,
        "calls": calls,
        "sessions": m["sessions"],
        "device_hours": device_s / 3600.0,
        "anchor_kinds": sorted(kinds),
        "anchor_missing_labels": missing,
    }


def live_env_hash(device):
    """Live env fingerprint on the chosen device (fail-closed string)."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    code = ("import sys; sys.path.insert(0, %r); "
            "from src.xpu.envcheck import current_env_hash; "
            "print(current_env_hash() or 'unknown')" % str(ROOT))
    try:
        out = subprocess.run(
            [sys.executable, "-c", code], cwd=str(ROOT), env=env,
            capture_output=True, text=True, timeout=180)
    except Exception:  # noqa: BLE001
        return "error"
    if out.returncode != 0:
        return "error"
    lines = [x for x in out.stdout.strip().splitlines() if x]
    return lines[-1] if lines else "error"


def _git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=str(ROOT),
                             capture_output=True, text=True, timeout=30)
    except Exception:  # noqa: BLE001
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def preflight(cfg, args):
    """Hard identity checks + recorded source state (exit 1 on failure)."""
    from src.xpu.config import DeviceProfile
    from src.xpu.selector import load_table, fallback_binding_violations

    failures = []

    def hard(name, ok, detail):
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failures.append(name)

    print(f"preflight device={args.device} config={CONFIG_PATH.relative_to(ROOT)}")
    ph = DeviceProfile.load().hash()
    hard("profile-hash", ph == cfg["device"]["profile_hash"],
         f"live {ph} vs config {cfg['device']['profile_hash']}")
    eh = live_env_hash(args.device)
    hard("env-hash", eh == cfg["device"]["env_hash"],
         f"live {eh} vs config {cfg['device']['env_hash']}")
    v = fallback_binding_violations(
        load_table(str(ROOT / "config" / "selector_table.json")))
    hard("fallback-binding", not v,
         "0 violations" if not v else f"{len(v)} violation(s): {v}")

    for dom, d in cfg["domains"].items():
        try:
            cells = _parse_spec(",".join(d["cells"]))
            n = sum(len(sems) for _, _, sems in cells[0])
            hard(f"domain:{dom}", n == d["cell_count"],
                 f"{n} cells vs config {d['cell_count']}")
        except SystemExit as e:
            hard(f"domain:{dom}", False, str(e))

    sha = hashlib.sha256(
        (ROOT / "config" / "selector_table.json").read_bytes()).hexdigest()
    freeze = cfg["source"]["selector_table_sha256_at_freeze"]
    if sha == freeze:
        print(f"  [ok] selector-table-sha: matches freeze ({sha[:12]})")
    else:
        print("  [note] selector-table-sha: changed since freeze "
              f"({sha[:12]} vs {freeze[:12]}; expected once selector "
              "promotions land)")
    head = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain")
    print(f"  [note] git: HEAD {head or 'unknown'} "
          f"dirty={'yes' if dirty else 'no'}")
    print(f"  [note] device choice: CUDA_VISIBLE_DEVICES={args.device}")
    if failures:
        print(f"preflight FAILED: {', '.join(failures)}")
        return 1
    print("preflight OK")
    return 0


def matrix_cmd(cfg, out_dir, cells_spec, phase=None, release_id=None):
    m = cfg["measurement"]
    oracle_mode = (phase_oracle_mode(cfg, phase) if phase
                   else cfg["protocol"]["oracle_mode"])
    cmd = [
        sys.executable, str(ROOT / "scripts" / "run_semantic_matrix.py"),
        "--cells", cells_spec,
        "--sessions", str(m["sessions"]),
        "--trials", str(m["trials"]),
        "--warmup", str(m["warmup"]),
        "--one-shot-trials", str(m["one_shot_trials"]),
        "--e2e-trials", str(m["e2e_trials"]),
        "--seed", str(m["seed"]),
        "--oracle-mode", oracle_mode,
        "--out", str(out_dir),
    ]
    if release_id:
        # WP5: one measured run both validates the frozen table and
        # writes the immutable release snapshot (strict-validated).
        cmd += ["--snapshot", "--release-id", release_id]
    return cmd


def replay_cmd(src, dst):
    return [
        sys.executable, str(ROOT / "scripts" / "run_semantic_matrix.py"),
        "--aggregate-only", "--out", str(src), "--replay-out", str(dst),
    ]


def spawn(cmd, env=None):
    """Run a launcher child (tests monkeypatch this seam)."""
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


def child_env(device):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    return env


def dry_run(cfg, args, out_root):
    print(f"dry-run phase={args.phase} device={args.device} "
          f"output={out_root}"
          + (f" release={args.release_id}" if args.release_id else ""))
    print(f"config: {CONFIG_PATH.relative_to(ROOT)} "
          f"sha={config_sha256()[:12]}")
    print("device-time model: analytic, anchored on v5 recorded "
          "worst-role steady medians (v5-exact / batch-scaled / "
          "element-scaled); sensitivity x0.5 .. x2; never measured "
          "evidence for the phase itself")

    if args.phase == "preflight":
        print("would check: profile-hash, env-hash, fallback-binding, "
              "domain integrity, selector-table sha, git state")
        return 0
    if args.phase == "replay":
        src = Path(args.replay_src) if args.replay_src else out_root / "validate"
        print(f"replay: --aggregate-only --out {src} --replay-out "
              f"{out_root / 'replay'} (new directory, evidence untouched)")
        if not src.is_dir():
            print(f"note: source {src} does not exist yet "
                  "(run the validate phase first)")
        return 0

    spec = phase_cells_spec(cfg, args.phase)
    est = estimate_phase(cfg, args.phase)
    dh = est["device_hours"]
    print(f"[{est['phase']}] domains={','.join(est['domains'])} "
          f"mode={est['oracle_mode']}")
    print(f"  cases: {est['shapes']} shapes / {est['semantic_cells']} "
          "semantic cells / labels:")
    print(f"    {spec}")
    print(f"  candidates: {est['candidates']} declared "
          f"(max {est['max_candidates_per_cell']} per cell; "
          f"budget {est['candidate_budget_per_cell']} per cell)")
    print(f"  calls: {est['calls']} "
          f"({est['sessions']} sessions x declared tiers)")
    print(f"  device-time: {dh:.3f} h expected "
          f"({dh * 0.5:.3f} .. {dh * 2:.3f} h sensitivity; "
          f"anchors={','.join(est['anchor_kinds']) or 'none'}"
          + (f", missing={est['anchor_missing_labels']}"
             if est["anchor_missing_labels"] else "") + ")")

    # full-portfolio budget summary across all measuring phases
    total_s = 0.0
    total_calls = 0
    rows = []
    for ph in MEASURE_PHASES:
        try:
            e = estimate_phase(cfg, ph)
        except Exception as ex:  # noqa: BLE001
            rows.append(f"[{ph}] estimate unavailable: {ex}")
            continue
        total_s += e["device_hours"] * 3600.0
        total_calls += e["calls"]
        rows.append(f"[{ph}] mode={e['oracle_mode']} "
                    f"cells={e['semantic_cells']} "
                    f"candidates={e['candidates']} calls={e['calls']} "
                    f"device={e['device_hours']:.3f} h")
    for r in rows:
        print(r)
    total_h = total_s / 3600.0
    budget_h = cfg["budgets"]["device_hours_total"]
    print(f"total measuring phases: {total_calls} calls, "
          f"{total_h:.3f} h analytic (budget {budget_h} h; "
          f"remaining for tuning: {budget_h - total_h:.3f} h)")
    print(f"tuning budget: {cfg['budgets']['implementation_rounds']} "
          f"rounds, {cfg['budgets']['max_candidates_per_cell']} "
          "candidates per cell max")
    if total_h > budget_h:
        print("WARNING: analytic device-time exceeds the declared "
              "device-hours budget")
    return 0


def _new_state(cfg, args, out_root):
    return {
        "schema": STATE_SCHEMA,
        "config_sha256": config_sha256(),
        "device": str(args.device),
        "output": str(out_root),
        "config": str(CONFIG_PATH.relative_to(ROOT)),
        "phases": {},
    }


def _load_state(path):
    state = json.loads(path.read_text())
    if state.get("schema") != STATE_SCHEMA:
        raise SystemExit(f"{path}: expected schema {STATE_SCHEMA!r}, "
                         f"got {state.get('schema')!r}")
    return state


def _save_state(path, state):
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        epilog="example: scripts/run_issue9.py --phase closure "
               "--device 2 --dry-run")
    ap.add_argument("--phase", required=True, choices=PHASES)
    ap.add_argument("--device", default="2",
                    help="CUDA_VISIBLE_DEVICES value for all children "
                         "(healthy P800 cards: 2,3,4,5,7)")
    ap.add_argument("--output", default=None,
                    help="output root (default: config outputs.root)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <output>/run_state.json, "
                         "skipping completed phases")
    ap.add_argument("--dry-run", action="store_true",
                    help="print declared cases/candidates/calls and the "
                         "analytic device-time budget; run nothing")
    ap.add_argument("--replay-src", default=None,
                    help="recorded run to replay (default: "
                         "<output>/validate)")
    ap.add_argument("--release-id", default=None,
                    help="measure phase: pass --snapshot --release-id "
                         "to the matrix child (WP5 immutable release)")
    args = ap.parse_args(argv)
    if args.release_id and args.phase in ("preflight", "replay"):
        raise SystemExit("--release-id needs a measure phase "
                         "(closure/profile/tune-*/validate)")

    cfg = load_config()
    out_root = Path(args.output) if args.output else Path(cfg["outputs"]["root"])
    if not out_root.is_absolute():
        out_root = ROOT / out_root

    if args.dry_run:
        return dry_run(cfg, args, out_root)

    out_root.mkdir(parents=True, exist_ok=True)
    state_path = out_root / "run_state.json"
    if state_path.is_file():
        if not args.resume:
            raise SystemExit(
                f"{state_path} exists; pass --resume to continue or use "
                "a fresh --output")
        state = _load_state(state_path)
        if state.get("config_sha256") != config_sha256():
            raise SystemExit(
                f"{state_path}: config changed since this state was "
                "written (fail closed; use a fresh --output)")
        if state.get("device") != str(args.device):
            print(f"warning: resuming with device {args.device} "
                  f"(state was {state.get('device')})")
    else:
        state = _new_state(cfg, args, out_root)
    _save_state(state_path, state)

    done = state["phases"].get(args.phase, {})
    if done.get("status") == "done":
        print(f"skipping {args.phase}: already completed "
              f"(returncode {done.get('returncode')})")
        return 0

    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    rc = 1
    out_dir = None
    cmd = None
    if args.phase == "preflight":
        rc = preflight(cfg, args)
    elif args.phase == "replay":
        src = Path(args.replay_src) if args.replay_src else out_root / "validate"
        if not src.is_dir():
            raise SystemExit(f"replay source {src} does not exist; run "
                             "the validate phase first (or pass "
                             "--replay-src)")
        dst = out_root / "replay"
        if dst.exists() and any(dst.iterdir()):
            raise SystemExit(f"replay destination {dst} is not empty; "
                             "choose a fresh --output")
        cmd = replay_cmd(src, dst)
        print(f"replay: {' '.join(cmd)}")
        rc = spawn(cmd, child_env(args.device))
        out_dir = str(src)
    else:
        cells_spec = phase_cells_spec(cfg, args.phase)
        out_dir = out_root / args.phase
        cmd = matrix_cmd(cfg, out_dir, cells_spec, phase=args.phase,
                         release_id=args.release_id)
        est = None
        try:
            est = estimate_phase(cfg, args.phase)
        except Exception as ex:  # noqa: BLE001
            print(f"estimate unavailable: {ex}")
        if est:
            print(f"phase {args.phase}: {est['semantic_cells']} cells, "
                  f"{est['calls']} calls, "
                  f"{est['device_hours']:.3f} h analytic")
        print(f"measure: CUDA_VISIBLE_DEVICES={args.device} "
              f"{' '.join(cmd)}")
        rc = spawn(cmd, child_env(args.device))

    state["phases"][args.phase] = {
        "status": "done" if rc == 0 else "failed",
        "returncode": rc,
        "started": started,
        "ended": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": str(args.device),
        "cmd": cmd,
        "out": str(out_dir) if out_dir else None,
    }
    _save_state(state_path, state)
    if rc == 0:
        print(f"{args.phase}: OK")
    else:
        print(f"{args.phase}: FAILED (returncode {rc})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
