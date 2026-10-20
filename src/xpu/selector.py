"""Per-size runtime selector: table-driven, M0-closed, abstains out of matrix.

Review round C of the migration convergence plan: the selector is a static
table built from measured confirmation sessions, never a model. Contract:

* **Keys**: ``(device, profile_hash, precision, transform_type,
  transform_direction, normalization, direction, N, batch)`` (issue #8
  PR-G). The three semantic fields form an exact-match tier: an entry
  carrying them only answers its own ``(transform_type,
  transform_direction, normalization)``; entries without them (the
  pre-issue-#8 table schema) are the legacy tier and answer any
  semantic as the old deployed path. Exact semantic tier wins over the
  legacy tier; both absent abstains.
* **Values**: ``(block, flow_id, flow_realization, processing_unit, params)``
  — the schema-v2 attribution of the measured winner.
* **M0 closure**: every table build compares challengers against the M0
  incumbent at the same shape; a challenger that does not beat the incumbent
  by the promotion guard (default 2%, an engineering guard — not a
  statistical claim) keeps M0. M0 itself is always a legal fallback.
* **Profile gate**: ``profile_hash`` is required. ``None`` abstains — a
  missing profile can never bypass the table, so a caller that has not
  resolved the current device profile falls back to M0 instead of guessing.
* **Environment gate**: production tables carry a non-empty ``env_hash``
  (bound at build time from the measurement manifests) and only select
  when the caller's environment fingerprint matches — ``env_hash="auto"``
  recomputes it live, ``None`` cannot verify and abstains (unknown or
  stale environment fails closed to M0; a table *without* ``env_hash`` is
  a pre-freshness artifact and never selects at runtime — offline
  inspection via ``load_table`` only).
* **Design-space contract**: every loaded entry's params must be drawn from
  ``config/butterfly_space.json`` (one shared candidate description) and
  satisfy the lowering contract (tile divides N, powers of two, legal
  variants); violations refuse the whole table at load time.
* **Abstain**: any key outside the measured matrix (unknown N, batch or
  precision not measured, profile-hash mismatch, missing profile) returns
  ``source="abstain"`` with the reason — the caller must fall back to the
  conservative M0 path rather than guess.

Host-only: no torch, no device. ``select()`` is pure table lookup; the
caller maps a Selection onto the runner entry points.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .config import (
    ConfigError,
    NORMALIZATIONS,
    TRANSFORM_DIRECTIONS,
    TRANSFORM_TYPES,
)

TABLE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "config", "selector_table.json")

# Promotion guard: a challenger must improve the search-stage median by at
# least this percentage to replace the incumbent (plan section 5/P2). The
# comparison is between aggregated medians of one run schedule, not paired
# per-trial medians.
GUARD_PCT = 2.0

# Runner names each (flow_realization, processing_unit) family may name
# via a table entry's optional ``executed`` field (issue #8 P1-A). The
# measured winner's runner is recorded end-to-end and dispatch honors
# it; these maps are the contract that the name belongs to the entry's
# own flow family, at the entry's own batch tier. Orientation (dit/dif
# may differ from the entry's locked ``direction`` — that is exactly
# the measured reality the field exists to close (a DIF winner inside
# a dit-locked matrix must deploy the DIF runner, never a renamed
# DIT). Static on purpose: src.xpu.runner imports torch, which the
# host-side contract must never pull in.
_SINGLE_RUNNERS = {
    ("global-stage", "radix2-scalar"): frozenset({"fft_dit", "fft_dif"}),
    ("global-stage", "radix4-scalar"): frozenset(
        {"fft_dit_radix4", "fft_dif_radix4"}),
    ("matrix-resident", "dense-dot-fft64"): frozenset(
        {"fft_dit_matrix", "fft_dif_matrix"}),
    ("fused-inplace", "radix2-scalar"): frozenset({"fft_dit_fused"}),
    ("grouped-boundary", "composite-linear-map"): frozenset(
        {"fft_dit_grouped", "fft_dif_grouped"}),
}
_BATCH_RUNNERS = {
    ("global-stage", "radix2-scalar"): frozenset(
        {"fft_dit_batch", "fft_dif_batch"}),
    ("global-stage", "radix4-scalar"): frozenset(
        {"fft_dit_radix4_batch", "fft_dif_radix4_batch"}),
    ("matrix-resident", "dense-dot-fft64"): frozenset(
        {"fft_dit_matrix_batch", "fft_dif_matrix_batch"}),
}


def _candidate_from_entry(entry: dict):
    """Shared schema-v2 ``Candidate`` for one frozen table entry.

    Entry attribution is the same candidate description the runner and the
    design-space enumerator use; params map onto the candidate knobs (tile
    -> fused_tile, variant -> batch_variant with the matching Ud/Td
    expansion). Optional entry-level ``layout``/``boundary_residence``/
    ``pipeline`` keys are honored so contract tests can probe them.
    """
    from .candidates import SPECIALIZED_ROUTING, UNIT_RADIX, Candidate

    params = dict(entry.get("params") or {})
    unit = entry.get("processing_unit")
    flow = entry.get("flow_id")
    variant = params.get("variant") or "none"
    batch = int(params.get("batch") or entry.get("batch") or 1)
    tile = params.get("tile")
    rows = params.get("rows")
    # Issue #8 PR-H: a specialized routing flow carries the CORE's
    # tile geometry, so it maps tile -> fused_tile like the local flow
    # does (contract view only; runners read params directly).
    local_tile = flow in (SPECIALIZED_ROUTING | {"local-fused-tile"})
    return Candidate(
        flow_id=flow,
        processing_unit=unit,
        radix=int(params.get("radix") or UNIT_RADIX.get(unit, 2)),
        block=int(entry.get("block") or 0),
        fused_tile=int(tile) if (
            local_tile and isinstance(tile, int)) else 1,
        batch=batch,
        batch_variant=variant if variant in ("spread", "folded") else "none",
        rows=int(rows) if isinstance(rows, int) else 1,
        glog=int(params.get("glog") or 0),
        maxp=int(params.get("maxp") or 0),
        cells=int(params.get("cells") or 1),
        flow_realization=entry.get("flow_realization") or "",
        boundary_residence=entry.get("boundary_residence")
        or params.get("boundary_residence") or "gm",
        pipeline=entry.get("pipeline") or params.get("pipeline") or "none",
        layout=entry.get("layout") or "planar-soa",
        ud=batch if variant == "spread" else 1,
        td=batch if variant == "folded" else 1,
    )


def entry_contract_violations(entry: dict) -> list[str]:
    """Lowering-contract + design-space violations of one table entry.

    Two gates, both single-sourced: published-space membership from
    ``config/butterfly_space.json`` (``DesignSpace``), then the runner's
    own lowering contract — the entry is mapped to a
    :class:`src.xpu.candidates.Candidate` and checked with
    ``candidates.check_lowering`` (issue #3 review: no second
    approximation of the rules). tile-must-divide-N is entry knowledge
    (Candidate carries no N) and stays here. Violations are illegal
    whether the entry was measured or not.
    """
    from .candidates import check_lowering
    from .config import ConfigError, DesignSpace, DeviceProfile

    try:
        sp = DesignSpace.load()
    except ConfigError as e:
        return [f"design space unreadable: {e}"]
    bad = []
    n = entry.get("N")
    params = dict(entry.get("params") or {})
    unit = entry.get("processing_unit")
    real = entry.get("flow_realization")
    block = entry.get("block")
    if not isinstance(block, int) or block <= 0 or (block & (block - 1)):
        # type/positivity guard stays here (check_lowering needs an int);
        # the power-of-two legality itself is check_lowering's rule.
        bad.append(f"block {block!r} not a positive power of two")
        return bad

    tile, rows = params.get("tile"), params.get("rows")
    if unit == "radix4-scalar" and 4 not in sp.radix:
        bad.append("radix-4 not in the published radix space")
    if unit == "dense-dot-fft64":
        if tile not in sp.matrix_tiles:
            bad.append(f"tile {tile!r} not in matrix tiles {sp.matrix_tiles}")
        if rows not in sp.matrix_rows:
            bad.append(f"rows {rows!r} not in matrix rows {sp.matrix_rows}")
        # Runner lowering contract: tile must divide N; because tile and N
        # are powers of two that means tile <= N, and the runner clamps
        # eff_tile to N when the declared tile is larger (documented in
        # butterfly_space.json P.register_matrix.note). A non-dividing tile
        # at tile <= N would violate lowering.
        if isinstance(tile, int) and isinstance(n, int) and tile > 0 \
                and tile <= n and n % tile:
            bad.append(f"tile {tile} does not divide N={n}")
    if unit == "radix2-scalar" and "tile" in params \
            and real == "fused-inplace":
        if params["tile"] not in sp.fused_tile_candidates:
            bad.append(
                f"fused tile {params['tile']!r} not in "
                f"{sp.fused_tile_candidates}")
    if unit == "composite-linear-map":
        for key, allowed in (("tile", sp.grouped_tiles),
                             ("glog", sp.grouped_glog),
                             ("maxp", sp.grouped_maxp),
                             ("cells", sp.grouped_cells)):
            if params.get(key) not in allowed:
                bad.append(f"{key}={params.get(key)!r} not in {allowed}")
    if "variant" in params and params["variant"] not in sp.batch_variants:
        bad.append(
            f"variant {params['variant']!r} not in {sp.batch_variants}")
    # Semantic-tier entries must carry a legal triple (issue #8); the
    # values come from the shared taxonomy, not from free text.
    if "transform_type" in entry:
        triple = (entry.get("transform_type"),
                  entry.get("transform_direction"),
                  entry.get("normalization"))
        if triple[0] not in TRANSFORM_TYPES:
            bad.append(f"transform_type {triple[0]!r} not in "
                       f"{TRANSFORM_TYPES}")
        if triple[1] not in TRANSFORM_DIRECTIONS:
            bad.append(f"transform_direction {triple[1]!r} not in "
                       f"{TRANSFORM_DIRECTIONS}")
        if triple[2] not in NORMALIZATIONS:
            bad.append(f"normalization {triple[2]!r} not in "
                       f"{NORMALIZATIONS}")
    if "batch" in params and params["batch"] not in sp.batch_candidates:
        bad.append(f"batch {params['batch']!r} not in {sp.batch_candidates}")
    if "radix" in params and params["radix"] not in sp.radix:
        bad.append(f"radix {params['radix']!r} not in {sp.radix}")
    # Runner closure (issue #8 P1-A): an entry carrying ``executed``
    # must name a legal runner of its own (flow_realization,
    # processing_unit) family at its own batch tier. Orientation may
    # differ from the locked ``direction`` (the measured winner is
    # authoritative); anything outside the family is a contract break.
    executed = entry.get("executed")
    if executed is not None:
        fam = (real, unit)
        if fam not in _SINGLE_RUNNERS:
            bad.append(
                f"executed {executed!r}: no known runner family for "
                f"(flow_realization={real!r}, processing_unit={unit!r})")
        else:
            b = int(params.get("batch") or entry.get("batch") or 1)
            allowed = (_BATCH_RUNNERS if b > 1 else _SINGLE_RUNNERS
                       ).get(fam, frozenset())
            if executed not in allowed:
                bad.append(
                    f"executed {executed!r} not a legal runner of "
                    f"{fam} at batch={b} (allowed: {sorted(allowed)})")
    if bad:
        return bad

    verdict = check_lowering(_candidate_from_entry(entry),
                             DeviceProfile.load())
    bad.extend(verdict.reasons)
    return bad


def semantic_key_of(entry: dict) -> tuple | None:
    """Exact semantic tier of an entry, or ``None`` for a legacy entry."""
    if "transform_type" not in entry:
        return None
    return (entry.get("transform_type"),
            entry.get("transform_direction"),
            entry.get("normalization"))


# Issue #9 WP0-A: a carried previous_fallback is only valid while the
# source environment it was justified in is STILL the environment of
# the table it ships in. The provenance must name that identity
# explicitly; anything else is a silent rebind and is rejected.
FALLBACK_CARRY_CONTRACT = "fallback-env-binding-v1"
_FALLBACK_BINDING_KEYS = (
    "carry_contract",        # schema of this contract
    "source_env_hash",       # env fingerprint of the paired confirmation
    "source_profile_hash",   # profile the fallback geometry was calibrated on
    "source_precision",      # precision contract of the source evidence
    "source_device",         # device identity (model/MP/smem family)
    "source_table_sha256",   # identity of the previous-deployed table
    "source_result_sha256",  # result hash of the paired-confirm artifact
    "paired_confirm",        # evidence path (artifact identity)
)
_DEVICE_IDENTITY_KEYS = ("device_model", "multiprocessor_count",
                         "max_shared_mem")


def fallback_binding_violations(table: dict) -> list[str]:
    """WP0-A shared contract: does every ``previous_fallbacks`` entry
    bind to THIS table's environment/profile/precision/device identity?

    One source of truth for load (runtime), rebuild/promote (builder +
    bootstrap verify) and tests: a profile, driver, runtime, toolchain
    or precision change turns every carried fallback into a violation,
    so callers either drop the fallbacks explicitly or refuse -- never
    silently rebind old evidence to a new environment.
    """
    violations: list[str] = []
    fbs = table.get("previous_fallbacks") or ()
    if not fbs:
        return violations
    ident = {k: (table.get("env") or {}).get(k)
             for k in _DEVICE_IDENTITY_KEYS}
    for i, e in enumerate(fbs):
        key = (e.get("N"), e.get("batch"), e.get("direction"))
        prov = e.get("previous_fallback_provenance")
        if not isinstance(prov, dict):
            violations.append(
                f"previous_fallbacks[{i}] {key}: missing "
                "previous_fallback_provenance (no source identity to "
                "bind against)")
            continue
        missing = [k for k in _FALLBACK_BINDING_KEYS if prov.get(k) in
                   (None, "")]
        if missing:
            violations.append(
                f"previous_fallbacks[{i}] {key}: provenance missing "
                f"binding key(s) {', '.join(missing)}")
            continue
        if prov["carry_contract"] != FALLBACK_CARRY_CONTRACT:
            violations.append(
                f"previous_fallbacks[{i}] {key}: carry_contract "
                f"{prov['carry_contract']!r} != {FALLBACK_CARRY_CONTRACT!r}")
        if prov["source_env_hash"] != table.get("env_hash"):
            violations.append(
                f"previous_fallbacks[{i}] {key}: source env "
                f"{prov['source_env_hash']} != table env "
                f"{table.get('env_hash')} (stale evidence; reconfirm "
                "under the current environment)")
        if prov["source_profile_hash"] != table.get("profile_hash"):
            violations.append(
                f"previous_fallbacks[{i}] {key}: source profile "
                f"{prov['source_profile_hash']} != table profile "
                f"{table.get('profile_hash')}")
        if prov["source_precision"] != table.get("precision"):
            violations.append(
                f"previous_fallbacks[{i}] {key}: source precision "
                f"{prov['source_precision']!r} != table precision "
                f"{table.get('precision')!r}")
        if any(ident[k] is None for k in _DEVICE_IDENTITY_KEYS):
            violations.append(
                f"previous_fallbacks[{i}] {key}: table carries no device "
                "identity (env: device_model/multiprocessor_count/"
                "max_shared_mem) to bind the fallback against")
        elif prov["source_device"] != ident:
            violations.append(
                f"previous_fallbacks[{i}] {key}: source device "
                f"{prov['source_device']} != table device {ident}")
    return violations


def is_previous_fallback(table: dict, entry: dict) -> bool:
    """True when ``entry`` is a declared previous-deployed fallback cell.

    Per-cell fallbacks (issue #8 review3 P1, Option A) live in their own
    table-level ``previous_fallbacks`` list: profile/env-bound entries
    carrying the full semantic triple of exactly one cell, outranking
    the current semantic entry there when the paired confirmation
    measured the previously deployed path faster. Identity, not key
    equality, so a look-alike entry in ``entries`` never inherits the
    tier.
    """
    return any(entry is e for e in (table.get("previous_fallbacks") or ()))


def find_entry(table: dict, n: int, batch: int, direction: str,
               transform_type: str = "c2c",
               transform_direction: str = "forward",
               normalization: str = "backward") -> dict | None:
    """Previous-deployed fallback cell first, exact semantic tier next,
    legacy (pre-semantic) fallback last.

    Single source of truth shared by :func:`select` and entry's
    incumbent-block lookup, so a plan and its M0 block resolve against
    the same tier of the table. A ``previous_fallbacks`` entry claims
    its (N, batch, direction, triple) exclusively -- that is the whole
    point of the per-cell fallback (issue #8): the old deployed path
    answers one cell's key instead of the current semantic entry that
    lost the paired confirmation there.
    """
    for e in table.get("previous_fallbacks") or ():
        if not (e["N"] == n and e["batch"] == batch
                and e["direction"] == direction):
            continue
        if semantic_key_of(e) == (transform_type, transform_direction,
                                  normalization):
            return e
    legacy = None
    for e in table.get("entries", ()):
        if not (e["N"] == n and e["batch"] == batch
                and e["direction"] == direction):
            continue
        key = semantic_key_of(e)
        if key is None:
            if legacy is None:
                legacy = e
        elif key == (transform_type, transform_direction, normalization):
            return e
    return legacy


@dataclass(frozen=True)
class Selection:
    """One selector verdict. ``source`` is ``table`` or ``abstain``."""

    n: int
    direction: str
    batch: int
    precision: str
    source: str
    reason: str
    transform_type: str = "c2c"
    transform_direction: str = "forward"
    normalization: str = "backward"
    entry_tier: str = "abstain"  # "semantic" | "previous-fallback" | "legacy" | "abstain"
    block: int | None = None
    flow_id: str | None = None
    flow_realization: str | None = None
    processing_unit: str | None = None
    params: dict = field(default_factory=dict)
    incumbent_ms: float | None = None
    challenger_ms: float | None = None
    # Measured winner's runner recorded end-to-end on the entry
    # (issue #8 P1-A); None for legacy entries, which dispatch by
    # direction via the runner table.
    executed: str | None = None

    @property
    def selected(self) -> bool:
        return self.source == "table"


def load_table(path: str = TABLE_PATH) -> dict:
    """Load the selector table (ConfigError when missing or malformed)."""
    try:
        with open(path) as f:
            table = json.load(f)
    except FileNotFoundError:
        raise ConfigError(
            f"selector table not found: {path} "
            "(run scripts/build_selector_table.py after a confirmation round)"
        ) from None
    except json.JSONDecodeError as e:
        raise ConfigError(f"selector table unreadable: {path}: {e}") from None
    for key in ("profile_hash", "precision", "guard_pct", "entries"):
        if key not in table:
            raise ConfigError(f"selector table missing required key {key!r}")
    violations = []
    for e in table["entries"]:
        for msg in entry_contract_violations(e):
            key = (e.get("N"), e.get("batch"), e.get("direction"))
            violations.append(f"entry {key}: {msg}")
    # Per-cell fallbacks (issue #8 review3 P1): same shared contract as
    # entries, plus two rules entries never need -- every fallback MUST
    # claim one exact semantic triple (a bare legacy entry here would
    # silently shadow every semantic of the shape) and no two may claim
    # the same cell key (first-wins would be an unnoticed duplicate).
    seen_fb: set[tuple] = set()
    for i, e in enumerate(table.get("previous_fallbacks") or []):
        key = (e.get("N"), e.get("batch"), e.get("direction"))
        sk = semantic_key_of(e)
        if sk is None:
            violations.append(
                f"previous_fallbacks[{i}] {key}: missing the exact "
                "semantic triple (transform_type/transform_direction/"
                "normalization) a per-cell fallback must claim")
        else:
            cell = key + sk
            if cell in seen_fb:
                violations.append(
                    f"previous_fallbacks[{i}] {key} {sk}: duplicate "
                    "previous_fallbacks cell key")
            seen_fb.add(cell)
        for msg in entry_contract_violations(e):
            violations.append(f"previous_fallbacks[{i}] {key}: {msg}")
    # Issue #9 WP0-A: the same env/profile/precision/device binding
    # contract the builder and verify stage use -- a carried fallback
    # from another environment must fail at load, not silently rebind.
    violations += fallback_binding_violations(table)
    if violations:
        raise ConfigError(
            "selector table violates the shared design space / lowering "
            f"contract ({len(violations)} entries): "
            + "; ".join(violations[:8])
            + (" ..." if len(violations) > 8 else "")
            + " -- rebuild the table from measurements under the current "
            "config/butterfly_space.json")
    return table


def assert_current_environment(table: dict) -> str:
    """Shared freshness gate for measurement scripts (issue #3 review).

    Computes the live environment fingerprint and compares it to the
    table's bound ``env_hash``. Raises ``RuntimeError`` on any mismatch
    or unknown fingerprint — callers MUST exit **before any kernel
    launch** (this function only imports torch/triton and queries the
    driver; it never creates tensors). Returns the live env hash on
    success so runs can record it in provenance.
    """
    bound = table.get("env_hash") or ""
    if not bound:
        raise RuntimeError(
            "selector table carries no env_hash (pre-freshness schema); "
            "refusing to measure against it -- rebuild the table from "
            "clean focus rounds under the current env")
    from .envcheck import current_env_hash
    cur = current_env_hash()
    if cur is None:
        raise RuntimeError(
            "live environment fingerprint unknown (device/runtime/"
            "compiler not probeable); cannot verify the table -- "
            "fail closed, recalibrate before measuring")
    if cur != bound:
        raise RuntimeError(
            f"environment hash {cur} != table {bound} (device/runtime/"
            "compiler changed since the table was built); re-run the "
            "probe and re-measure before producing new artifacts")
    return cur


def _abstain(n, direction, batch, precision, reason,
             transform_type: str = "c2c",
             transform_direction: str = "forward",
             normalization: str = "backward") -> Selection:
    return Selection(n=n, direction=direction, batch=batch,
                     precision=precision, source="abstain", reason=reason,
                     transform_type=transform_type,
                     transform_direction=transform_direction,
                     normalization=normalization, entry_tier="abstain")


def select(direction: str, n: int, batch: int = 1, precision: str = "fp32",
           profile_hash: str | None = None,
           table: dict | None = None,
           env_hash: str | None = "auto",
           transform_type: str = "c2c",
           transform_direction: str = "forward",
           normalization: str = "backward") -> Selection:
    """Return the measured winner for a shape, or abstain outside the matrix.

    ``env_hash`` verifies the table against the live environment: ``"auto"``
    recomputes it (fail closed when the environment is unknown), an explicit
    string is trusted as given, ``None`` cannot verify and abstains. A
    production-schema table MUST carry a non-empty ``env_hash``; unbound
    (pre-freshness) tables never select at runtime — inspect their entries
    offline via ``load_table`` only.

    Semantic tiering (issue #8 PR-G): an entry carrying
    ``(transform_type, transform_direction, normalization)`` answers
    only its exact triple; entries without the fields (legacy) answer
    any semantic as the old deployed path. :func:`find_entry` resolves
    a declared ``previous_fallbacks`` cell first (issue #8 Option A),
    then the exact semantic tier, then legacy, else abstains; a
    fallback hit reports ``entry_tier="previous-fallback"``.
    """
    table = table if table is not None else load_table()
    if transform_type not in TRANSFORM_TYPES:
        return _abstain(n, direction, batch, precision,
                        f"transform_type must be one of {TRANSFORM_TYPES}, "
                        f"got {transform_type!r}",
                        transform_type, transform_direction, normalization)
    if transform_direction not in TRANSFORM_DIRECTIONS:
        return _abstain(
            n, direction, batch, precision,
            f"transform_direction must be one of {TRANSFORM_DIRECTIONS}, "
            f"got {transform_direction!r}",
            transform_type, transform_direction, normalization)
    if normalization not in NORMALIZATIONS:
        return _abstain(n, direction, batch, precision,
                        f"normalization must be one of {NORMALIZATIONS}, "
                        f"got {normalization!r}",
                        transform_type, transform_direction, normalization)
    if direction not in ("dit", "dif"):
        return _abstain(n, direction, batch, precision,
                        f"direction must be dit/dif, got {direction!r}",
                        transform_type, transform_direction, normalization)
    if precision != table["precision"]:
        return _abstain(n, direction, batch, precision,
                        f"table measured {table['precision']}, got {precision!r}",
                        transform_type, transform_direction, normalization)
    if profile_hash is None:
        return _abstain(n, direction, batch, precision,
                        "profile_hash is required; None cannot bypass the "
                        "table (fall back to M0)",
                        transform_type, transform_direction, normalization)
    if profile_hash != table["profile_hash"]:
        return _abstain(n, direction, batch, precision,
                        f"profile hash {profile_hash} != table "
                        f"{table['profile_hash']}; re-measure before selecting",
                        transform_type, transform_direction, normalization)
    bound_env = table.get("env_hash") or ""
    if not bound_env:
        return _abstain(
            n, direction, batch, precision,
            "table carries no env_hash (pre-freshness schema); runtime "
            "selection fails closed to M0 -- inspect entries offline via "
            "load_table only",
            transform_type, transform_direction, normalization)
    cur = env_hash
    if cur == "auto":
        from .envcheck import current_env_hash
        cur = current_env_hash()
    if cur is None:
        return _abstain(
            n, direction, batch, precision,
            "environment fingerprint unknown (device/runtime/compiler "
            "not probeable); this table cannot be verified here and "
            "must fail closed to M0",
            transform_type, transform_direction, normalization)
    if cur != bound_env:
        return _abstain(
            n, direction, batch, precision,
            f"environment hash {cur} != table {bound_env} "
            "(device/runtime/compiler changed since the table was "
            "built); re-run the probe and re-measure before selecting",
            transform_type, transform_direction, normalization)
    e = find_entry(table, n, batch, direction, transform_type,
                   transform_direction, normalization)
    if e is not None:
        if is_previous_fallback(table, e):
            # Issue #8 Option A: the paired confirmation measured the
            # previously deployed path faster here -- the fallback cell
            # answers with the old geometry and says so in the reason,
            # so raw rows and plans are attributable without diffing
            # params.
            tier = "previous-fallback"
            reason = ("previous-deployed per-cell fallback (issue #8): "
                      + e.get("reason", "measured confirmation winner"))
        else:
            tier = ("semantic" if semantic_key_of(e) is not None
                    else "legacy")
            reason = e.get("reason", "measured confirmation winner")
        return Selection(
            n=n, direction=direction, batch=batch, precision=precision,
            source="table",
            reason=reason,
            transform_type=transform_type,
            transform_direction=transform_direction,
            normalization=normalization,
            entry_tier=tier,
            block=e["block"], flow_id=e["flow_id"],
            flow_realization=e["flow_realization"],
            processing_unit=e["processing_unit"],
            params=dict(e.get("params") or {}),
            incumbent_ms=e.get("incumbent_ms"),
            challenger_ms=e.get("challenger_ms"),
            executed=e.get("executed"),
        )
    return _abstain(n, direction, batch, precision,
                    f"no measured entry for N={n} batch={batch} "
                    f"direction={direction} semantic="
                    f"{transform_type}/{transform_direction}/"
                    f"{normalization}; fall back to M0",
                    transform_type, transform_direction, normalization)
