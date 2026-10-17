#!/usr/bin/env python3
"""WP0 one-click profile / model / selector bootstrap (issue #6, PR-A).

Chains the existing tools into a single resumable pipeline — no logic
is duplicated, every stage is a subprocess of the canonical script::

    probe  -> hw_probe.py                (calibration profile)
    check  -> smoke.py + test_correctness.py
    calibrate -> bench.py confirm rounds + run_batch_release focus
                 rounds + full-space holdout        (--calibrate)
    build  -> validate_cost_model.py + build_selector_table.py
    verify -> manifest contract (strict) + selector smoke + fail-closed
                                                             (--verify)

Modes (combinable):

* default            probe + check + build-from-existing (no measuring)
* ``--probe-only``   only the hardware probe, then exit
* ``--calibrate``    run the measuring rounds (hours on real hardware)
* ``--verify``       strict manifest contract + selector smoke
* ``--resume``       skip stages already recorded complete in state.json
* ``--promote``      atomically publish staged artifacts to config/
                     (WITHOUT this flag production config/ is never
                     touched, even if every stage succeeded)

Everything is written under ``--output`` staging first; a failed stage
keeps its logs and partial results for ``--resume`` but never publishes.
Promotion is ``os.replace`` (atomic within a filesystem).

Usage::

    CUDA_VISIBLE_DEVICES=6 python3 scripts/bootstrap_profile.py \
        --device 6 --output results/local-profile \
        --calibrate --verify --promote
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from src.xpu.manifest_contract import (  # noqa: E402
    format_failures,
    result_sha256,
    validate_benchmark_output,
    validate_manifest,
)

STATE_FILE = "state.json"
STAGING_DIR = ".staging"
PRODUCTION_TABLE = ROOT / "config" / "selector_table.json"
PRODUCTION_CALIBRATION = ROOT / "config" / "calibration.json"

# stage -> short human name; order matters.
ALL_STAGES = ("probe", "check", "calibrate", "build", "verify")

# The representative calibration recipe (issue #6 WP0): confirm rounds
# feed the table builder, focus+holdout feed the cost model — the same
# commands the frozen evidence was produced with.
def calibrate_commands(out: Path, device: str) -> list[list[str]]:
    py = sys.executable
    focus_a = out / "batch_focus_a"
    focus_b = out / "batch_focus_b"
    holdout = out / "batch_holdout"
    return [
        [py, "scripts/bench.py", "--sizes",
         "8,16,1024,2048,4096,65536,1048576", "--block", "1024",
         "--tiles", "64", "--matrix-tiles", "64", "--grouped-tiles", "8",
         "--batchs", "1", "--batch-variants", "spread",
         "--flows", "m0,m1,m2,m3,m5,vendor",
         "--trials", "30", "--warmup", "10",
         "--json", str(out / "confirm_b1024.json")],
        [py, "scripts/bench.py", "--sizes",
         "8,16,1024,2048,4096,65536,1048576", "--block", "256",
         "--tiles", "64", "--matrix-tiles", "64", "--grouped-tiles", "8",
         "--batchs", "1", "--batch-variants", "spread",
         "--flows", "m0,m1,m2,m3,m5,vendor",
         "--trials", "30", "--warmup", "10",
         "--json", str(out / "confirm_b256.json")],
        [py, "scripts/bench.py", "--sizes", "64,4096,65536",
         "--block", "1024", "--tiles", "64", "--matrix-tiles", "64",
         "--grouped-tiles", "8", "--batchs", "1,4,16,64",
         "--batch-variants", "spread,folded",
         "--flows", "m0,m4,vendor",
         "--trials", "30", "--warmup", "10",
         "--json", str(out / "confirm_batch.json")],
        # staging is this run's scratch: a --resume re-enters the
        # calibrate stage and must replace finished runs instead of
        # failing on run_batch_release's no-overwrite guard
        [py, "scripts/run_batch_release.py", "--overwrite",
         "--out", str(focus_a)],
        [py, "scripts/run_batch_release.py", "--overwrite",
         "--out", str(focus_b)],
        [py, "scripts/run_batch_release.py", "--overwrite",
         "--shapes", "1024x16,1024x64,2048x16", "--full-space",
         "--out", str(holdout)],
    ]


def stage_env(device: str) -> dict:
    env = dict(os.environ)
    if device:
        env["CUDA_VISIBLE_DEVICES"] = device
    return env


def load_state(out: Path) -> dict:
    p = out / STATE_FILE
    if p.is_file():
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            pass
    return {"completed": [], "failures": {}}


def save_state(out: Path, state: dict) -> None:
    tmp = out / (STATE_FILE + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    os.replace(tmp, out / STATE_FILE)


def mark_complete(state: dict, stage: str) -> None:
    if stage not in state["completed"]:
        state["completed"].append(stage)


def should_skip(state: dict, stage: str, resume: bool) -> bool:
    return bool(resume and stage in state["completed"])


def run_cmd(cmd: list[str], device: str, log_dir: Path,
            tag: str) -> int:
    """Run one stage command, tee-ing output to a log file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{tag}.log"
    env = stage_env(device)
    if cmd and cmd[0] == sys.executable and not any(
            a.startswith("scripts/") for a in cmd):
        pass  # plain python, cwd handles paths
    with open(log_path, "w") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(
            cmd, cwd=ROOT, env=env, stdout=log,
            stderr=subprocess.STDOUT, text=True)
        log.write(f"\n[exit {proc.returncode}]\n")
    return proc.returncode


def stamp_result_hashes(out: Path) -> list[str]:
    """Write source/result hashes into staged manifests (WP0)."""
    from src.xpu.config import git_commit
    failures = []
    source_hash = git_commit()
    targets: list[Path] = list(out.glob("batch_*/summary.json"))
    if (out / "summary.json").is_file():
        targets.append(out / "summary.json")
    # *.progress.json are bench resume checkpoints (pid/last_completed),
    # not result artifacts — the confirm_*.json glob would catch them
    # and fail "no manifest to stamp" (found by the WP0 full-chain run).
    targets += [q for q in out.glob("confirm_*.json")
                if not q.name.endswith(".progress.json")]
    for path in targets:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            failures.append(f"{path.name}: unreadable for stamping: {e}")
            continue
        man = payload.get("manifest")
        if not isinstance(man, dict):
            failures.append(f"{path.name}: no manifest to stamp")
            continue
        from src.xpu.manifest_contract import rows_sha256
        man.setdefault("source_hash", source_hash)
        if (path.name == "summary.json"
                and list(path.parent.glob("raw_session*.jsonl"))):
            man["result_sha256"] = result_sha256(path.parent)
        elif "rows" in payload:
            man["result_sha256"] = rows_sha256(payload)
        try:
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2) + "\n")
            os.replace(tmp, path)
        except OSError as e:
            failures.append(f"{path.name}: stamp write failed: {e}")
    return failures


def verify_bundle(out: Path, env_hash: str | None,
                  production_env_hash: str | None = None) -> list[str]:
    """Strict manifest contract + selector smoke + fail-closed gate.

    ``env_hash`` is the live fingerprint to test the staged table
    against; a mismatch (or ``None`` on a host without a device stack)
    must fail closed — never silently reuse.
    """
    from src.xpu.selector import (
        assert_current_environment,
        load_table,
        select,
    )

    failures: list[str] = []
    # 1. Strict manifest contract on every staged run output.
    outputs = sorted(out.glob("batch_*")) + sorted(
        p for p in out.glob("confirm_*.json")
        if not p.name.endswith(".progress.json"))
    if not outputs:
        failures.append("verify: no staged benchmark outputs found")
    for target in outputs:
        failures += validate_benchmark_output(target, strict=True)

    # 2. Selector smoke: staged table loads, env gate passes, exact-key
    #    select works and abstains are structural.
    table_path = out / "selector_table.json"
    if not table_path.is_file():
        failures.append("verify: staged selector_table.json missing")
        return failures
    try:
        table = load_table(str(table_path))
    except Exception as e:  # noqa: BLE001  (report, don't raise)
        failures.append(f"verify: staged table does not load: {e}")
        return failures
    try:
        live = assert_current_environment(table)
    except RuntimeError as e:
        failures.append(f"verify: freshness gate failed closed: {e}")
        live = None
    if live is not None and env_hash is not None and live != env_hash:
        failures.append(f"verify: env mismatch live={live} != {env_hash}")
    # Exact-key smoke: every table entry must select itself.
    for entry in table.get("entries", [])[:]:
        try:
            sel = select(direction=entry["direction"], n=entry["N"],
                         batch=entry["batch"],
                         precision=table["precision"],
                         profile_hash=table["profile_hash"],
                         table=table, env_hash=env_hash)
        except Exception as e:  # noqa: BLE001
            failures.append(f"verify: select({entry['N']},"
                            f"{entry['batch']}) raised {e}")
            continue
        if sel.source != "table":
            failures.append(
                f"verify: exact-key select abstained for "
                f"N{entry['N']}B{entry['batch']} ({sel.reason})")
    # Out-of-matrix abstention still works.
    try:
        sel = select(direction="dit", n=3000, batch=1,
                     precision=table["precision"],
                     profile_hash=table["profile_hash"],
                     table=table, env_hash=env_hash)
        if sel.source == "table":
            failures.append("verify: out-of-matrix key was not an "
                            "abstention")
    except Exception as e:  # noqa: BLE001
        failures.append(f"verify: fallback probe raised {e}: "
                        "expected structural abstention")
    return failures


def promote(staging: Path, table_src: Path | None = None,
            calib_src: Path | None = None) -> list[str]:
    """Atomically publish staged artifacts; production path only here."""
    failures: list[str] = []
    table_src = table_src or (staging / "selector_table.json")
    calib_src = calib_src or (staging / "calibration.json")
    if not table_src.is_file():
        failures.append(f"promote: staged table missing: {table_src}")
    else:
        tmp = PRODUCTION_TABLE.with_suffix(".json.promote-tmp")
        tmp.write_bytes(table_src.read_bytes())
        os.replace(tmp, PRODUCTION_TABLE)
    if calib_src.is_file():
        tmp = PRODUCTION_CALIBRATION.with_suffix(".json.promote-tmp")
        tmp.write_bytes(calib_src.read_bytes())
        os.replace(tmp, PRODUCTION_CALIBRATION)
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=os.environ.get("CUDA_VISIBLE_DEVICES",
                                                       "0"),
                    help="CUDA_VISIBLE_DEVICES value for stage commands")
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results" / "local-profile",
                    help="staging directory for this bootstrap run")
    ap.add_argument("--probe-only", action="store_true",
                    help="run only the hardware probe, then exit")
    ap.add_argument("--calibrate", action="store_true",
                    help="run the measuring rounds (confirm/focus/"
                         "holdout); without this flag no benchmarks "
                         "are executed")
    ap.add_argument("--verify", action="store_true",
                    help="strict manifest contract + selector smoke")
    ap.add_argument("--resume", action="store_true",
                    help="skip stages already complete in state.json")
    ap.add_argument("--promote", action="store_true",
                    help="atomically publish staged table+calibration "
                         "to config/ (only after verify passes)")
    args = ap.parse_args()

    out: Path = args.output
    staging = out / STAGING_DIR
    staging.mkdir(parents=True, exist_ok=True)
    log_dir = out / "logs"
    state = load_state(out)

    if args.probe_only:
        stages = ["probe"]
    else:
        stages = ["probe", "check"]
        if args.calibrate:
            stages += ["calibrate", "build"]
        if args.verify:
            stages += ["verify"]

    def _fail(stage: str, code: int, detail: str = "") -> int:
        state.setdefault("failures", {})[stage] = {
            "exit": code, "detail": detail,
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        save_state(out, state)
        print(format_failures([f"stage {stage} failed (exit {code}) "
                               f"{detail}".strip()],
                              title="bootstrap"), file=sys.stderr)
        print(f"staging kept for --resume: {staging}", file=sys.stderr)
        return 1

    for stage in stages:
        if should_skip(state, stage, args.resume):
            print(f"[skip] {stage} (already complete, --resume)")
            continue
        print(f"[stage] {stage}")
        if stage == "probe":
            code = run_cmd(
                [sys.executable, "scripts/hw_probe.py",
                 "--out", str(staging / "calibration.json")],
                args.device, log_dir, stage)
            if code != 0:
                return _fail(stage, code)
        elif stage == "check":
            for script, tag in (("scripts/smoke.py", "smoke"),
                                ("scripts/test_correctness.py",
                                 "correctness")):
                code = run_cmd([sys.executable, script], args.device,
                               log_dir, f"{stage}_{tag}")
                if code != 0:
                    return _fail(stage, code, tag)
        elif stage == "calibrate":
            for i, cmd in enumerate(calibrate_commands(staging,
                                                       args.device)):
                code = run_cmd(cmd, args.device, log_dir,
                               f"{stage}_{i:02d}")
                if code != 0:
                    return _fail(stage, code, " ".join(cmd[:4]))
            stamp_failures = stamp_result_hashes(staging)
            if stamp_failures:
                return _fail(stage, 1,
                             "; ".join(stamp_failures[:4]))
        elif stage == "build":
            focus_a = staging / "batch_focus_a" / "summary.json"
            focus_b = staging / "batch_focus_b" / "summary.json"
            holdout = staging / "batch_holdout" / "summary.json"
            confirms = [staging / "confirm_b1024.json",
                        staging / "confirm_b256.json"]
            have_focus = focus_a.is_file() and focus_b.is_file()
            have_holdout = holdout.is_file()
            have_confirms = all(p.is_file() for p in confirms)
            batch_confirm = staging / "confirm_batch.json"
            if not (have_focus and have_holdout and have_confirms
                    and batch_confirm.is_file()):
                # build without calibrate: fall back to nothing staged
                # — this is only legal when verifying pre-existing
                # artifacts is not the goal; fail clearly.
                return _fail(stage, 1,
                             "staged confirm/focus/holdout missing; "
                             "run with --calibrate first")
            code = run_cmd(
                [sys.executable, "scripts/validate_cost_model.py",
                 str(focus_a), str(holdout),
                 "--out", str(staging / "model_validation.json")],
                args.device, log_dir, stage + "_model")
            if code != 0:
                return _fail(stage, code, "validate_cost_model")
            code = run_cmd(
                [sys.executable, "scripts/build_selector_table.py",
                 str(confirms[0]), str(confirms[1]),
                 "--batch", str(batch_confirm),
                 "--focus", str(focus_a), str(focus_b),
                 "--out", str(staging / "selector_table.json"),
                 "--verify-live-env"],
                args.device, log_dir, stage + "_table")
            if code != 0:
                return _fail(stage, code, "build_selector_table")
        elif stage == "verify":
            from src.xpu.envcheck import current_env_hash
            failures = verify_bundle(staging, current_env_hash())
            if failures:
                return _fail(stage, 1,
                             "; ".join(failures[:6]))
        mark_complete(state, stage)
        state.pop("failures", None)
        save_state(out, state)

    if args.probe_only:
        print(f"probe-only done: {staging / 'calibration.json'}")
        return 0

    if args.promote:
        if args.verify and "verify" not in state["completed"]:
            print("promote refused: verify has not passed",
                  file=sys.stderr)
            return 1
        failures = promote(staging)
        if failures:
            print(format_failures(failures, "promote"), file=sys.stderr)
            return 1
        print(f"promoted staged table -> {PRODUCTION_TABLE}")
    else:
        print("staged only: production config/ untouched "
              "(rerun with --promote to publish)")
    print(format_failures([], "bootstrap"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
