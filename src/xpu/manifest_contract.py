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

import gzip
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

# One raw trial row (raw_session*.jsonl) must carry these. For rows
# recording a non-timed outcome (status != "ok": wrong-value / error /
# vendor-unsupported), ``api_ms`` and ``err`` are exempt: the row's
# evidence is the status+reason (issue #8: unsupported and negative
# points stay in the raw evidence, they just carry no timing sample).
TIMING_KEYS = ("api_ms", "err")
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


def _raw_paths(raw_dir: Path):
    """Raw session files under ``raw_dir``: plain ``.jsonl`` preferred,
    gzip ``.jsonl.gz`` when the plain session is absent (issue #8 PR-J
    A2: release snapshots store gzip'd raw as canonical evidence)."""
    plain = sorted(Path(raw_dir).glob("raw_session*.jsonl"))
    if plain:
        return plain
    return sorted(Path(raw_dir).glob("raw_session*.jsonl.gz"))


def _open_raw(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path)


def _logical_raw_name(path: Path) -> str:
    name = path.name
    return name[:-3] if name.endswith(".gz") else name


def _iter_raw_rows(raw_dir: Path):
    """Yield (path, line_no, row) for every raw session row.

    Reads both plain and gzip raw sessions; an unreadable/corrupt
    archive yields an ``__decode_error__`` row so strict validation
    fails closed instead of silently missing rows.
    """
    for path in _raw_paths(raw_dir):
        try:
            with _open_raw(path) as f:
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
        except (OSError, EOFError) as e:
            yield path, 0, {"__decode_error__": f"unreadable raw: {e}"}


def iter_raw_rows(raw_dir: Path):
    """Public plain/gzip raw reader shared by validators and aggregation.

    ``--aggregate-only`` replays exactly the bytes the strict gate
    reads (issue #8 review4 P2: same reader on both sides), yielding
    ``(path, line_no, row)`` triples.
    """
    return _iter_raw_rows(raw_dir)


# The manifest's timed tier carries exactly ``manifest['trials']``
# rows per (label, candidate, session). Legacy single-tier run outputs
# (focus/batch rounds) omit the row ``tier`` field entirely; those are
# still held to the cross-session census below but not to the
# steady-tier count (issue #8 review4 P2-a targets matrix evidence).
STEADY_TIERS = ("steady",)


def trial_completeness_failures(rows, manifest: dict) -> list[str]:
    """Session/label/candidate/tier census (issue #8 review4 P2-a).

    Every ``(label, candidate, tier)`` group of timed (``status=="ok"``)
    rows must appear in EVERY session with identical row counts -- a
    candidate truncated in one session, or present in fewer sessions,
    fails here. Steady groups must carry exactly ``manifest['trials']``
    timed rows per session. Non-ok rows (recorded correctness failures
    etc.) are exception evidence, not timings, and leave the census;
    row duplicates are already rejected by :func:`validate_raw_sessions`.
    """
    failures: list[str] = []
    trials = _get(manifest, "trials")
    groups: dict[tuple, dict] = {}
    for row in rows:
        if not isinstance(row, dict) or "__decode_error__" in row:
            continue
        if row.get("status", "ok") != "ok":
            continue
        key = (row.get("label"), row.get("candidate"), row.get("tier"))
        g = groups.setdefault(key, {})
        s = row.get("session")
        g[s] = g.get(s, 0) + 1
    if not groups:
        return failures

    def _skey(s):
        return (0, s, "") if isinstance(s, int) else (1, 0, str(s))

    sessions = sorted({s for g in groups.values() for s in g}, key=_skey)

    def _gkey(key):
        return tuple(str(part) for part in key)

    for key in sorted(groups, key=_gkey):
        g = groups[key]
        per_session = {s: g.get(s) for s in sessions}
        counts = [per_session[s] for s in sessions]
        if any(c is None for c in counts) or len(set(counts)) != 1:
            failures.append(
                "trial completeness census: group (label="
                f"{key[0]!r}, candidate={key[1]!r}, tier={key[2]!r}) "
                f"is not identical across sessions {sessions}: "
                f"per-session rows {per_session}")
        if (isinstance(trials, int) and trials > 0
                and key[2] in STEADY_TIERS):
            for s in sessions:
                if per_session[s] is not None and per_session[s] != trials:
                    failures.append(
                        "trial completeness census: group (label="
                        f"{key[0]!r}, candidate={key[1]!r}, tier="
                        f"{key[2]!r}) session {s} has "
                        f"{per_session[s]} rows, manifest declares "
                        f"trials={trials}")
    return failures


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
        timed = row.get("status", "ok") == "ok"
        for key in REQUIRED_ROW_KEYS:
            if not timed and key in TIMING_KEYS:
                continue
            if key not in row or row[key] in (None, ""):
                failures.append(f"{path.name}:{line_no} missing row field "
                                f"{key!r} (candidate="
                                f"{row.get('candidate')!r}, status="
                                f"{row.get('status')!r})")
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
    if strict:
        # Issue #8 review4 P2-a/P2-b: exact trial census per
        # session/label/candidate/tier, and the recorded oracle
        # candidate list must be the list the raw rows actually show.
        plain_rows = [r for _, _, r in rows]
        failures += trial_completeness_failures(plain_rows, manifest)
        declared_oracles = _get(manifest, "oracle_candidates")
        if isinstance(declared_oracles, list):
            observed = sorted(
                {r.get("candidate") for r in plain_rows
                 if isinstance(r.get("candidate"), str)
                 and r["candidate"].startswith("oracle:")})
            if sorted(str(c) for c in declared_oracles) != observed:
                failures.append(
                    "oracle_candidates mismatch: manifest "
                    f"{sorted(str(c) for c in declared_oracles)} != raw "
                    f"rows {observed}")
    return failures


def result_sha256(raw_dir: Path) -> str:
    """Deterministic hash over raw sessions (sorted).

    Canonical form is ``(logical filename, raw session bytes)``: a gzip
    snapshot hashes under its logical ``raw_sessionN.jsonl`` name over
    the decompressed bytes, so the plain source directory and the
    immutable ``.jsonl.gz`` snapshot produce the identical digest
    (issue #8 PR-J A2).
    """
    h = hashlib.sha256()
    for path in _raw_paths(raw_dir):
        h.update(_logical_raw_name(path).encode())
        if str(path).endswith(".gz"):
            with gzip.open(path, "rb") as f:
                h.update(f.read())
        else:
            h.update(path.read_bytes())
    return h.hexdigest()


def rows_sha256(payload: dict) -> str:
    """Deterministic hash of a single-file artifact's evidence rows."""
    rows = payload.get("rows", payload)
    blob = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


# Semantic-matrix protocol id (issue #8 PR-G): single-sourced here so
# the runner, the builder's strict gate and the manifest validator can
# never drift. The summary's ``cases`` array is bound into the
# manifest by :func:`cases_sha256` at stamp time.
SEMANTIC_MATRIX_PROTOCOL = ("semantic-matrix-v1:plan-reuse:interleave:"
                            "correctness-first")


def cases_sha256(cases) -> str:
    """Deterministic hash binding a matrix summary's ``cases`` array.

    Stamped into the manifest at calibration time and re-checked by
    the strict output gate: a summary whose cases array is edited
    after stamping (or replaced wholesale) fails closed (issue #8
    P1-C).
    """
    blob = json.dumps(cases, sort_keys=True, separators=(",", ":"))
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
    has_raw = raw_dir.is_dir() and bool(_raw_paths(raw_dir))
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
            try:
                got = (result_sha256(raw_dir) if has_raw
                       else rows_sha256(payload))
            except (OSError, EOFError) as e:
                # a corrupt gzip archive must fail closed as a
                # validation failure, never crash the gate
                failures.append(f"{base.name}: unreadable raw archive: {e}")
                got = None
            if got is not None and want != got:
                failures.append(
                    f"{base.name}: result_sha256 mismatch "
                    f"(manifest {want[:12]}… != evidence {got[:12]}…)")
        # Matrix summaries bind their cases array (issue #8 P1-C): the
        # manifest must carry a cases_sha256 that matches the payload.
        if _get(manifest, "protocol_id") == SEMANTIC_MATRIX_PROTOCOL:
            cases = payload.get("cases")
            stamp = _get(manifest, "cases_sha256")
            if not isinstance(cases, list):
                failures.append(
                    f"{summary_path.name}: matrix summary has no cases "
                    "array")
            elif stamp is None:
                failures.append(
                    f"{summary_path.name}: matrix manifest missing "
                    "cases_sha256")
            elif stamp != cases_sha256(cases):
                failures.append(
                    f"{summary_path.name}: cases_sha256 mismatch "
                    f"(manifest {stamp[:12]}… != payload "
                    f"{cases_sha256(cases)[:12]}…)")
    return failures


def format_failures(failures: list[str], title: str = "manifest gate"
                    ) -> str:
    """Single aggregated report string (never dies on first failure)."""
    if not failures:
        return f"{title}: OK"
    lines = [f"{title}: {len(failures)} failure(s)"]
    lines += [f"  - {f}" for f in failures]
    return "\n".join(lines)
