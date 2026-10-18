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
                     (verify is never skipped)
* ``--promote``      publish staged artifacts to config/ — implies a
                     fresh verify, re-checks the current bytes against
                     the recorded hashes and refuses any post-verify
                     drift (WITHOUT this flag production config/ is
                     never touched, even if every stage succeeded)

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
import hashlib
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

# Issue #8: default semantic-matrix scope for the bootstrap chain is
# the stress domain (the cells whose high regret the issue requires
# fixing); --matrix-scope overrides for diagnostics runs.
MATRIX_SCOPE_DEFAULT = "stress"

# The representative calibration recipe (issue #6 WP0): confirm rounds
# feed the table builder, focus+holdout feed the cost model — the same
# commands the frozen evidence was produced with.
def calibrate_commands(out: Path, device: str,
                       matrix_scope: str = MATRIX_SCOPE_DEFAULT
                       ) -> list[list[str]]:
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
        # issue #8 WP1: the semantic-matrix confirmation run whose
        # cells feed --matrix-run (>= 3 sessions x >= 30 trials; the
        # builder enforces the same floor). Scope defaults to the
        # stress domain; production table entries come only from cells
        # that clear their own >= 2% guard.
        [py, "scripts/run_semantic_matrix.py",
         "--scope", matrix_scope, "--sessions", "3",
         "--trials", "30", "--warmup", "5",
         "--out", str(out / f"matrix_{matrix_scope}")],
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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def staging_artifact_hashes(staging: Path) -> dict:
    """SHA-256 of every staged artifact promote must see unchanged.

    The verify stage records this map; ``--promote`` re-computes it and
    refuses when any byte drifted after verify (issue #8 P1-D: a
    ``--resume`` that silently re-publishes a modified staging is a
    verification bypass).
    """
    hashes: dict = {}
    for p in sorted(staging.glob("confirm_*.json")):
        if p.name.endswith(".progress.json"):
            continue
        hashes[p.name] = _sha256_file(p)
    for pattern in ("batch_*/summary.json", "matrix_*/summary.json"):
        for p in sorted(staging.glob(pattern)):
            hashes[str(p.relative_to(staging))] = _sha256_file(p)
    for name in ("selector_table.json", "calibration.json"):
        p = staging / name
        if p.is_file():
            hashes[name] = _sha256_file(p)
    return hashes


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
    # matrix_* are directories: summary.json embeds the manifest and
    # raw_session siblings feed result_sha256(dir) below (issue #8)
    targets += list(out.glob("matrix_*/summary.json"))
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
        from src.xpu.manifest_contract import (
            SEMANTIC_MATRIX_PROTOCOL,
            cases_sha256,
            rows_sha256,
        )
        man.setdefault("source_hash", source_hash)
        if (path.name == "summary.json"
                and list(path.parent.glob("raw_session*.jsonl"))):
            man["result_sha256"] = result_sha256(path.parent)
        elif "rows" in payload:
            man["result_sha256"] = rows_sha256(payload)
        if (man.get("protocol_id") == SEMANTIC_MATRIX_PROTOCOL
                and isinstance(payload.get("cases"), list)):
            # matrix summaries bind their cases array (issue #8 P1-C)
            man["cases_sha256"] = cases_sha256(payload["cases"])
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
    outputs = (sorted(out.glob("batch_*"))
               + sorted(out.glob("matrix_*"))
               + sorted(
                   p for p in out.glob("confirm_*.json")
                   if not p.name.endswith(".progress.json")))
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
                         table=table, env_hash=env_hash,
                         transform_type=entry.get("transform_type",
                                                  "c2c"),
                         transform_direction=entry.get(
                             "transform_direction", "forward"),
                         normalization=entry.get("normalization",
                                                "backward"))
        except Exception as e:  # noqa: BLE001
            failures.append(f"verify: select({entry['N']},"
                            f"{entry['batch']}) raised {e}")
            continue
        if sel.source != "table":
            failures.append(
                f"verify: exact-key select abstained for "
                f"N{entry['N']}B{entry['batch']} ({sel.reason})")
            continue
        # The query must resolve to THIS entry (its params/geometry),
        # not merely to some table hit on the shape.
        if (dict(sel.params or {}) != dict(entry.get("params") or {})
                or sel.block != entry.get("block")):
            failures.append(
                f"verify: exact-key select for "
                f"N{entry['N']}B{entry['batch']} resolved a different "
                f"entry (params/block mismatch)")
    # Measured-vs-deployed runner closure (issue #8 P1-A): an entry
    # carrying ``executed`` must deploy exactly that runner through
    # plan_fft; legacy entries' challenger orientation token must
    # agree with the direction the entry claims (the runner direction
    # dispatch derives). Covers DIT/DIF and B=1/batched via the full
    # entry loop.
    from src.xpu.entry import plan_fft
    for entry in table.get("entries", []):
        try:
            plan = plan_fft(
                direction=entry["direction"], n=entry["N"],
                batch=entry["batch"], precision=table["precision"],
                profile_hash=table["profile_hash"], table=table,
                env_hash=env_hash,
                transform_type=entry.get("transform_type", "c2c"),
                transform_direction=entry.get(
                    "transform_direction", "forward"),
                norm=entry.get("normalization", "backward"))
        except Exception as e:  # noqa: BLE001
            failures.append(
                f"verify: plan_fft(N{entry['N']}B{entry['batch']}) "
                f"raised {e}")
            continue
        if plan.selection_source == "composition":
            # exact-shape composition outranks the butterfly table by
            # design; its runner is composition's own contract.
            continue
        if "executed" in entry:
            if plan.executed != entry["executed"]:
                failures.append(
                    f"verify: N{entry['N']}B{entry['batch']} "
                    f"direction={entry['direction']}: deployed runner "
                    f"{plan.executed!r} != measured "
                    f"{entry['executed']!r}")
        else:
            challenger = str(
                (entry.get("evidence") or {}).get("challenger") or "")
            token = ("dif" if "_dif" in challenger else
                     "dit" if "_dit" in challenger else None)
            if token is not None and (
                    token != entry["direction"]
                    or f"fft_{token}" not in (plan.executed or "")):
                failures.append(
                    f"verify: N{entry['N']}B{entry['batch']} legacy "
                    f"challenger {challenger!r} implies {token} but "
                    f"direction={entry['direction']!r} deploys "
                    f"{plan.executed!r}")
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


def promote(staging: Path, state: dict | None = None,
            table_src: Path | None = None,
            calib_src: Path | None = None) -> list[str]:
    """Publish staged artifacts only after a fresh, un-drifted verify.

    Fail-closed (issue #8 P1-D): promote never trusts a prior
    ``--verify`` from an earlier invocation — it re-runs the full
    host-side gate on the current bytes, re-checks
    profile/env/commit/dirty/contract, and only then replaces
    production in one rollback-capable transaction. ``state`` must
    carry the verify stage's recorded artifact hashes; a mismatch
    (any byte changed after verify) refuses the promote.
    """
    from src.xpu.config import DeviceProfile, git_commit
    from src.xpu.envcheck import current_env_hash
    from src.xpu.selector import load_table

    failures: list[str] = []
    if state is None or not isinstance(state.get("verified"), dict):
        return ["promote: no fresh verify result in state (run with "
                "--verify/--promote so the verify stage records its "
                "hashes)"]
    recorded = state["verified"].get("hashes")
    if not isinstance(recorded, dict):
        return ["promote: state verified record lacks artifact hashes"]
    current = staging_artifact_hashes(staging)
    if current != recorded:
        drift = sorted(
            set(current) ^ set(recorded)
            | {k for k in set(current) & set(recorded)
               if current[k] != recorded[k]})
        return ["promote: staged artifacts changed after verify "
                f"({', '.join(drift[:6]) or 'unknown drift'}); "
                "re-run verify"]
    failures += [f"promote: {f}"
                 for f in verify_bundle(staging, current_env_hash())]
    table_src = table_src or (staging / "selector_table.json")
    calib_src = calib_src or (staging / "calibration.json")
    if not table_src.is_file():
        failures.append(f"promote: staged table missing: {table_src}")
        return failures
    try:
        table = load_table(str(table_src))
    except Exception as e:  # noqa: BLE001
        failures.append(f"promote: staged table does not load: {e}")
        return failures
    try:
        live_profile = DeviceProfile.load().hash()
    except Exception as e:  # noqa: BLE001
        failures.append(f"promote: live device profile unreadable: {e}")
        live_profile = None
    if live_profile is not None and table.get("profile_hash") != live_profile:
        failures.append(
            f"promote: staged profile_hash {table.get('profile_hash')!r} "
            f"!= live {live_profile!r}")
    commit = git_commit()
    for p in (sorted(staging.glob("batch_*/summary.json"))
              + sorted(staging.glob("matrix_*/summary.json"))
              + [q for q in staging.glob("confirm_*.json")
                 if not q.name.endswith(".progress.json")]):
        try:
            man = json.loads(p.read_text()).get("manifest") or {}
        except (OSError, json.JSONDecodeError) as e:
            failures.append(f"promote: {p.name} unreadable: {e}")
            continue
        if man.get("dirty") is not False:
            failures.append(
                f"promote: {p.name} manifest.dirty={man.get('dirty')!r}; "
                "evidence must come from a clean tree")
        if commit != "unknown" and man.get("commit") != commit:
            failures.append(
                f"promote: {p.name} manifest.commit={man.get('commit')!r} "
                f"!= current {commit!r}; re-measure at HEAD")
    if failures:
        return failures
    # --- transactional publish ---------------------------------------
    table_bytes = table_src.read_bytes()
    calib_bytes = (calib_src.read_bytes() if calib_src.is_file()
                   else None)
    targets = [(PRODUCTION_TABLE, table_bytes)]
    if calib_bytes is not None:
        targets.append((PRODUCTION_CALIBRATION, calib_bytes))
    backups = {dst: dst.read_bytes() for dst, _ in targets
               if dst.is_file()}
    tmps = []
    try:
        for dst, data in targets:
            tmp = dst.with_suffix(dst.suffix + ".promote-tmp")
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(data)
            tmps.append((tmp, dst))
        for tmp, dst in tmps:
            os.replace(tmp, dst)
    except OSError as e:
        for dst, data in backups.items():
            try:
                restore = dst.with_suffix(dst.suffix + ".promote-restore")
                restore.write_bytes(data)
                os.replace(restore, dst)
            except OSError:
                pass  # best-effort rollback; originals kept in memory
        for tmp, _ in tmps:
            try:
                tmp.unlink()
            except OSError:
                pass
        failures.append(
            f"promote: publish failed, production restored: {e}")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=os.environ.get("CUDA_VISIBLE_DEVICES",
                                                       "0"),
                    help="CUDA_VISIBLE_DEVICES value for stage commands")
    ap.add_argument("--matrix-scope", default=MATRIX_SCOPE_DEFAULT,
                    help="semantic-matrix scope for calibrate + build "
                         "(default: %(default)s, the issue #8 stress "
                         "domain; pass 'all' for a full-domain run)")
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
                    help="publish staged table+calibration to config/ "
                         "(implies --verify; re-verifies the current "
                         "bytes and refuses any post-verify drift)")
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
        if args.verify or args.promote:
            # --promote implies a fresh verify (issue #8 P1-D): the
            # old "promote without --verify" bypass is gone.
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
        # verify is never skippable (issue #8 P1-D): a --resume that
        # skips it would let an unverified or drifted staging promote.
        if stage != "verify" and should_skip(state, stage, args.resume):
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
            for i, cmd in enumerate(calibrate_commands(
                    staging, args.device, args.matrix_scope)):
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
            matrix_dir = staging / f"matrix_{args.matrix_scope}"
            have_matrix = (matrix_dir / "summary.json").is_file()
            if not (have_focus and have_holdout and have_confirms
                    and batch_confirm.is_file() and have_matrix):
                # build without calibrate: fall back to nothing staged
                # — this is only legal when verifying pre-existing
                # artifacts is not the goal; fail clearly.
                return _fail(stage, 1,
                             "staged confirm/focus/holdout/matrix "
                             "missing; run with --calibrate first")
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
                 "--matrix-run", str(matrix_dir),
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
            # record the verified bytes: promote re-checks these hashes
            # and refuses on any post-verify drift (issue #8 P1-D)
            state["verified"] = {
                "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "hashes": staging_artifact_hashes(staging)}
        mark_complete(state, stage)
        state.pop("failures", None)
        save_state(out, state)

    if args.probe_only:
        print(f"probe-only done: {staging / 'calibration.json'}")
        return 0

    if args.promote:
        failures = promote(staging, state=state)
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
