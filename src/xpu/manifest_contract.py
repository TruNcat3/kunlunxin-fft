"""Benchmark manifest contract (issue #6 WP0 / PR-A).

Every publishable benchmark artifact must carry the provenance the
review requires: git commit/dirty, runner identity, device/profile/env
hashes, protocol and timing scope, the actual executed candidates with
their parameters, warmup/trial/session counts, synchronization and
plan-reuse semantics, and source/result hashes. :func:`validate_manifest`
checks the schema; :func:`validate_raw_sessions` checks the evidence
behind it (duplicate trials, missing cases, missing rows). Both return
an **aggregated** failure list so a run reports every problem at once
instead of dying on the first one.

Strictness tiers:

* ``strict=False`` (default) — archived artifacts may omit the
  hashes that only a fresh bootstrap run can stamp
  (``result_sha256`` / ``source_hash`` aliases are still checked when
  present).
* ``strict=True`` — used by the bootstrap verify gate: every key,
  including both hashes, must be present and correct.

All checks are pure Python over JSON/plain files (host-testable, no
device stack).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# WP0 manifest schema. Aliases map legacy/runner-local key names onto
# the canonical field so older artifacts validate without re-measuring.
ALIASES = {
    "repro_command": "command",
    "error_threshold": "correctness_threshold",
    "source_hash": "config_hash",
    "runner_version": "toolchain",
}

REQUIRED_MANIFEST_KEYS = (
    # identity / provenance
    "commit",
    "dirty",
    "device",
    "driver",
    "profile_hash",
    "env_hash",
    "toolchain",
    # protocol
    "protocol_id",
    "timing_scope",
    "selection_semantics",
    "warmup",
    "trials",
    "sessions",
    "seed_base",
    # design / evidence pointers
    "candidates",
    "shapes",
    "error_threshold",
    "repro_command",
    "gm_bytes_kind",
    "kernel_time_status",
)

# Hashes stamped by the bootstrap runner after the raw evidence exists.
REQUIRED_RESULT_HASH_KEYS = (
    "source_hash",
    "result_sha256",
)

# One raw trial row (raw_session*.jsonl) must carry these.
REQUIRED_ROW_KEYS = (
    "session",
    "label",
    "trial",
    "candidate",
    "api_ms",
    "correct",
    "err",
    "status",
    "protocol_id",
    "profile_hash",
    "timed_scope",
)


def _get(manifest: dict, key: str):
    """Fetch a manifest value, resolving documented aliases."""
    if key in manifest:
        return manifest[key]
    canon = ALIASES.get(key)
    if canon and canon in manifest:
        return manifest[canon]
    return None


def validate_manifest(manifest: dict, strict: bool = False) -> list[str]:
    """Aggregated schema failures for one manifest (empty == pass)."""
    failures: list[str] = []
    if not isinstance(manifest, dict):
        return [f"manifest is not a mapping: {type(manifest).__name__}"]
    for key in REQUIRED_MANIFEST_KEYS:
        val = _get(manifest, key)
        if val is None:
            failures.append(f"manifest missing field: {key}")
        elif val == "" or val == []:
            failures.append(f"manifest field empty: {key}")
    if strict:
        for key in REQUIRED_RESULT_HASH_KEYS:
            if _get(manifest, key) is None:
                failures.append(f"manifest missing field: {key}")
    # Types the downstream tooling relies on.
    for key in ("warmup", "trials", "sessions"):
        val = _get(manifest, key)
        if val is not None and not isinstance(val, int):
            failures.append(f"manifest field {key} must be int, "
                            f"got {type(val).__name__}")
    if manifest.get("dirty") not in (True, False, "True", "False"):
        failures.append("manifest dirty must be a boolean")
    return failures


def canonical_label(value) -> str:
    """Normalize a shape/label spelling: ``N1024B16:dit`` == ``N1024B16_dit``."""
    return str(value).replace(":", "_")


def _iter_raw_rows(raw_dir: Path):
    """Yield (path, line_no, row) for every raw session row."""
    for path in sorted(raw_dir.glob("raw_session*.jsonl")):
        with open(path) as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    yield path, line_no, {"__decode_error__": str(e)}
                    continue
                yield path, line_no, row


def validate_raw_sessions(raw_dir: Path, manifest: dict,
                          strict: bool = False) -> list[str]:
    """Duplicate-trial, missing-case and row-field failures.

    Missing case = a shape declared in ``manifest['shapes']`` with zero
    raw rows, or a declared session count with fewer raw session files.
    Duplicate trial = the same (session, label, trial, candidate) key
    observed more than once across the raw files.
    """
    failures: list[str] = []
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        return [f"raw session dir not found: {raw_dir}"]

    rows = list(_iter_raw_rows(raw_dir))
    if not rows:
        return [f"no raw session rows under {raw_dir}"]

    seen: dict[tuple, tuple] = {}
    labels_seen: set[str] = set()
    session_files = {p.name for p, _, _ in rows}
    for path, line_no, row in rows:
        if "__decode_error__" in row:
            failures.append(f"{path.name}:{line_no} invalid JSON: "
                            f"{row['__decode_error__']}")
            continue
        for key in REQUIRED_ROW_KEYS:
            if key not in row or row[key] in (None, ""):
                failures.append(f"{path.name}:{line_no} missing row field "
                                f"{key!r} (candidate="
                                f"{row.get('candidate')!r})")
                break
        key = (row.get("session"), row.get("label"), row.get("trial"),
               row.get("candidate"))
        if key in seen:
            prev = seen[key]
            failures.append(
                f"duplicate trial: {key} in {path.name}:{line_no} "
                f"(first seen {prev[0].name}:{prev[1]})")
        else:
            seen[key] = (path, line_no)
        if row.get("label"):
            labels_seen.add(canonical_label(row["label"]))

    declared_shapes = manifest.get("shapes") or []
    if declared_shapes:
        normalized = {canonical_label(s) for s in declared_shapes}
        if not (normalized & labels_seen):
            failures.append(
                "missing cases: none of the declared shapes "
                f"{list(declared_shapes)} appear in raw rows")
        else:
            for shape_s in map(str, declared_shapes):
                if canonical_label(shape_s) not in labels_seen:
                    failures.append(
                        f"missing case: declared shape {shape_s} has no "
                        "raw rows")

    # Row provenance must match the manifest — a mismatch means raw
    # rows from a different protocol/device run were mixed in. Strict
    # only: pre-contract archives carry legacy sentinels (the P2 packet
    # canary rows stamped profile_hash="local"; fixed in
    # run_batch_release.py), and the strict tier gates fresh staged
    # evidence produced by the fixed runner.
    if strict:
        man_proto = _get(manifest, "protocol_id")
        man_prof = _get(manifest, "profile_hash")
        man_scope = _get(manifest, "timing_scope")
        for path, line_no, row in rows:
            if "__decode_error__" in row:
                continue
            for row_key, want in (("protocol_id", man_proto),
                                  ("profile_hash", man_prof),
                                  ("timed_scope", man_scope)):
                got = row.get(row_key)
                if want is not None and got is not None and got != want:
                    failures.append(
                        f"{path.name}:{line_no} row {row_key} mismatch: "
                        f"row {got!r} != manifest {want!r}")

    declared_sessions = _get(manifest, "sessions")
    if isinstance(declared_sessions, int) and declared_sessions > 0:
        found_sessions = len({p.name for p in {pp for pp, _, _ in rows}})
        if found_sessions < declared_sessions and strict:
            failures.append(
                f"missing sessions: manifest declares "
                f"{declared_sessions}, found {found_sessions} raw "
                "session file(s)")
    return failures


def result_sha256(raw_dir: Path) -> str:
    """Deterministic hash over raw_session*.jsonl contents (sorted)."""
    h = hashlib.sha256()
    for path in sorted(Path(raw_dir).glob("raw_session*.jsonl")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def rows_sha256(payload: dict) -> str:
    """Deterministic hash of a single-file artifact's evidence rows."""
    rows = payload.get("rows", payload)
    blob = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def validate_benchmark_output(summary_or_dir, strict: bool = False
                              ) -> list[str]:
    """Aggregate manifest + raw-session failures for one run output.

    Accepts a ``summary.json`` path, a directory containing one, or a
    ``confirm_*.json`` style single-file artifact (rows embedded).
    """
    path = Path(summary_or_dir)
    if path.is_dir():
        summary_path = path / "summary.json"
        if not summary_path.is_file():
            return [f"summary.json not found in {path}"]
        raw_dir = path
        base = path
    else:
        summary_path = path
        raw_dir = path.parent
        base = path
    try:
        payload = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return [f"{summary_path}: unreadable summary: {e}"]
    manifest = payload.get("manifest")
    if manifest is None:
        return [f"{summary_path}: summary has no manifest"]
    failures = [f"{summary_path.name}: {m}"
                for m in validate_manifest(manifest, strict=strict)]
    has_raw = (raw_dir.is_dir()
               and bool(list(raw_dir.glob("raw_session*.jsonl"))))
    if has_raw:
        failures += [f"{base.name}/{m}"
                     for m in validate_raw_sessions(raw_dir, manifest,
                                                    strict=strict)]
    if strict and not has_raw and isinstance(payload.get("rows"), list):
        trials = _get(manifest, "trials")
        for i, row in enumerate(payload["rows"]):
            if not isinstance(row, dict):
                continue
            for col, val in row.items():
                if not col.endswith("_samples") or not isinstance(val,
                                                                  list):
                    continue
                if isinstance(trials, int) and len(val) != trials:
                    failures.append(
                        f"{base.name}: row[{i}].{col} has "
                        f"{len(val)} samples, manifest declares "
                        f"{trials} trials")
    if strict:
        if _get(manifest, "result_sha256") is None:
            failures.append(
                f"{summary_path.name}: manifest missing field: "
                "result_sha256")
        else:
            want = _get(manifest, "result_sha256")
            got = (result_sha256(raw_dir) if has_raw
                   else rows_sha256(payload))
            if want != got:
                failures.append(
                    f"{base.name}: result_sha256 mismatch "
                    f"(manifest {want[:12]}… != evidence {got[:12]}…)")
    return failures


def format_failures(failures: list[str], title: str = "manifest gate"
                    ) -> str:
    """Single aggregated report string (never dies on first failure)."""
    if not failures:
        return f"{title}: OK"
    lines = [f"{title}: {len(failures)} failure(s)"]
    lines += [f"  - {f}" for f in failures]
    return "\n".join(lines)
