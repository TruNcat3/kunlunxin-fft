"""Exact-shape composition registry (issue #6 PR-E, WP3 external-core rules).

``A1P1`` — framework-owned DIT decomposition whose local stages are
computed by the vendor host-API FFT on materialized sub-block tensors —
is a **composition**, never a resident fused kernel: the vendor core is
``full-transform-only``, so it only ever runs on gather/scatter
sub-blocks the framework owns (``docs/benchmarks/external-cores.md``,
capability id ``a1p1_materialized_subtransform`` in
``config/external_cores.json``).

Promotion rules (issue #6 WP3):

- per **exact** shape only (``N``/``batch``/``direction`` all exact);
- the 2% incumbent guard must have passed on independent confirmation
  evidence (``scripts/promote_composition.py`` validates and stamps);
- composition entries live in ``config/composition_table.json`` and
  **never** enter the butterfly ``selector_table.json`` (separately
  named, separately validated);
- runtime selection fails closed: missing table -> no composition,
  any binding/contract mismatch -> no composition (the butterfly table
  path decides instead).
"""

from __future__ import annotations

import json
import os

from .config import ConfigError

TABLE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))),
    "config", "composition_table.json")
EXTERNAL_CORES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))),
    "config", "external_cores.json")

TABLE_REQUIRED_KEYS = (
    "schema_version", "kind", "profile_hash", "env_hash",
    "precision", "guard_pct", "contract_id", "entries",
    "source_hash", "result_sha256",
)
ENTRY_REQUIRED_KEYS = (
    "N", "batch", "direction", "precision", "tile", "core",
    "flow_id", "flow_realization", "processing_unit",
    "candidate_ms", "incumbent_ms", "guard_pct", "guard_pass",
    "reason", "evidence",
)
EVIDENCE_REQUIRED_KEYS = (
    "artifact", "artifact_sha256", "commit", "sessions", "trials",
    "confirmation_session", "err_max", "p50_ms", "mad_ms", "p95_ms",
)

KIND = "composition-promotion"


def guard_pct_of(candidate_ms: float, incumbent_ms: float) -> float:
    """Incumbent guard in percent: positive = candidate faster."""
    return (incumbent_ms - candidate_ms) / incumbent_ms * 100.0


def composition_entry_violations(entry: dict, table: dict) -> list[str]:
    """Contract/capability violations for one composition entry."""
    bad: list[str] = []
    for key in ENTRY_REQUIRED_KEYS:
        if key not in entry or entry[key] in (None, ""):
            bad.append(f"missing {key!r}")
    if bad:
        return bad
    if entry["direction"] != "dit":
        bad.append(f"direction {entry['direction']!r} (composition "
                   "flow is DIT-only)")
    if not isinstance(entry["N"], int) or entry["N"] <= 0:
        bad.append(f"N {entry['N']!r} not a positive int (exact keys "
                   "only)")
    if not isinstance(entry["batch"], int) or entry["batch"] <= 0:
        bad.append(f"batch {entry['batch']!r} not a positive int")
    if entry.get("precision") != table.get("precision"):
        bad.append(f"precision {entry.get('precision')!r} != table "
                   f"{table.get('precision')!r}")
    if entry.get("core") not in ("vendor", "triton"):
        bad.append(f"core {entry.get('core')!r} not in (vendor, triton)")
    if not isinstance(entry.get("tile"), int) or entry["tile"] < 16:
        bad.append(f"tile {entry.get('tile')!r} invalid")
    # Recompute the guard from the measured medians: the stamped
    # fields must agree with arithmetic (tamper-evidence).
    recomputed = guard_pct_of(float(entry["candidate_ms"]),
                              float(entry["incumbent_ms"]))
    if abs(recomputed - float(entry["guard_pct"])) > 1e-6:
        bad.append(f"guard_pct {entry['guard_pct']!r} != recomputed "
                   f"{recomputed:.6f} from candidate/incumbent ms")
    if float(entry["guard_pct"]) < float(table.get("guard_pct", 2.0)):
        bad.append(f"guard {entry['guard_pct']:.3f}% < required "
                   f"{table.get('guard_pct')}%")
    if entry.get("guard_pass") is not True:
        bad.append("guard_pass is not true")
    ev = entry.get("evidence")
    if not isinstance(ev, dict):
        bad.append("evidence not an object")
        return bad
    for key in EVIDENCE_REQUIRED_KEYS:
        if key not in ev or ev[key] in (None, ""):
            bad.append(f"evidence missing {key!r}")
    if isinstance(ev.get("err_max"), (int, float)) \
            and float(ev["err_max"]) > 1e-4:
        bad.append(f"evidence err_max {ev['err_max']} > 1e-4 "
                   "(correctness gate)")
    if not ev.get("confirmation_session"):
        bad.append("evidence has no independent confirmation session")
    if isinstance(ev.get("sessions"), int) and ev["sessions"] < 3:
        bad.append(f"evidence sessions {ev['sessions']} < 3")
    return bad


def table_violations(table: dict) -> list[str]:
    """Contract violations for a whole composition table."""
    bad: list[str] = []
    for key in TABLE_REQUIRED_KEYS:
        if key not in table or table[key] in (None, ""):
            bad.append(f"missing table key {key!r}")
    if bad:
        return bad
    if table["kind"] != KIND:
        bad.append(f"kind {table['kind']!r} != {KIND!r}")
    if table.get("precision") != "fp32":
        bad.append(f"precision {table.get('precision')!r} != 'fp32'")
    # The capability must exist in the external-core registry: every
    # composition candidate carries its capability/lowering contract.
    try:
        with open(EXTERNAL_CORES_PATH) as f:
            cores = json.load(f).get("cores", [])
        ids = {c.get("id") for c in cores}
    except (OSError, json.JSONDecodeError) as e:
        return bad + [f"external core registry unreadable: {e}"]
    if table["contract_id"] not in ids:
        bad.append(f"contract_id {table['contract_id']!r} not in "
                   "config/external_cores.json")
    entries = table.get("entries")
    if not isinstance(entries, list) or not entries:
        bad.append("entries empty or not a list")
        entries = []
    seen: set = set()
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            bad.append(f"entry {i} not an object")
            continue
        key = (e.get("N"), e.get("batch"), e.get("direction"))
        if key in seen:
            bad.append(f"duplicate exact-shape key {key}")
        seen.add(key)
        bad.extend(f"entry {key}: {msg}"
                   for msg in composition_entry_violations(e, table))
    # Tamper-evidence: result_sha256 covers the canonical entries.
    if entries:
        from .manifest_contract import rows_sha256
        recomputed = rows_sha256({"rows": entries})
        if recomputed != table["result_sha256"]:
            bad.append("result_sha256 mismatch (entries modified "
                       "after promotion)")
    return bad


def load_composition_table(path: str = TABLE_PATH) -> dict | None:
    """Load the composition table.

    ``None`` when no table has been promoted (absent file -> the
    composition path simply does not exist). ``ConfigError`` when a
    table exists but violates its contract — loud, never silent
    degradation.
    """
    try:
        with open(path) as f:
            table = json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        raise ConfigError(f"composition table unreadable: {path}: {e}") \
            from None
    if not isinstance(table, dict):
        raise ConfigError(f"composition table is not an object: {path}")
    violations = table_violations(table)
    if violations:
        raise ConfigError(
            f"composition table violates its contract "
            f"({len(violations)}): " + "; ".join(violations[:8])
            + (" ..." if len(violations) > 8 else "")
            + " -- re-run scripts/promote_composition.py from the "
              "confirmation evidence")
    return table


def composition_select(direction: str, n: int, batch: int = 1,
                       precision: str = "fp32",
                       profile_hash: str | None = None,
                       env_hash: str | None = "auto",
                       table: dict | None = None) -> dict | None:
    """Exact-shape composition verdict, or ``None`` (fail closed).

    Mirrors :func:`src.xpu.selector.select` fail-closed rules: no
    composition without a verifiable profile/env binding; every
    mismatch returns ``None`` so the butterfly table path decides.
    ``None`` also for DIF (the decomposition flow is DIT-only).
    """
    if direction != "dit":
        return None
    if table is None:
        table = load_composition_table()
        if table is None:
            return None
    if precision != table.get("precision"):
        return None
    if not profile_hash or profile_hash != table.get("profile_hash"):
        return None
    bound_env = table.get("env_hash") or ""
    if not bound_env:
        return None
    if env_hash == "auto":
        from .envcheck import current_env_hash
        env_hash = current_env_hash()
    if env_hash is None or env_hash != bound_env:
        return None
    for e in table["entries"]:
        if (e["N"] == n and e["batch"] == batch
                and e["direction"] == direction):
            return e
    return None
