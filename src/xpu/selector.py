"""Per-size runtime selector: table-driven, M0-closed, abstains out of matrix.

Review round C of the migration convergence plan: the selector is a static
table built from measured confirmation sessions, never a model. Contract:

* **Keys**: ``(device, profile_hash, precision, direction, N, batch)``.
* **Values**: ``(block, flow_id, flow_realization, processing_unit, params)``
  — the schema-v2 attribution of the measured winner.
* **M0 closure**: every table build compares challengers against the M0
  incumbent at the same shape; a challenger that does not beat the incumbent
  by the promotion guard (default 2%, an engineering guard — not a
  statistical claim) keeps M0. M0 itself is always a legal fallback.
* **Profile gate**: ``profile_hash`` is required. ``None`` abstains — a
  missing profile can never bypass the table, so a caller that has not
  resolved the current device profile falls back to M0 instead of guessing.
* **Environment gate**: a table carrying ``env_hash`` (built by
  ``build_selector_table``) only selects when the caller's environment
  fingerprint matches — ``env_hash="auto"`` recomputes it live, ``None``
  cannot verify and abstains (unknown or stale environment fails closed to
  M0; tables without the field are pre-binding artifacts and unaffected).
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

from .config import ConfigError

TABLE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "config", "selector_table.json")

# Promotion guard: a challenger must improve the search-stage median by at
# least this percentage to replace the incumbent (plan section 5/P2). The
# comparison is between aggregated medians of one run schedule, not paired
# per-trial medians.
GUARD_PCT = 2.0


def entry_contract_violations(entry: dict) -> list[str]:
    """Lowering-contract + design-space violations of one table entry.

    The candidate description is single-sourced from
    ``config/butterfly_space.json`` (``DesignSpace``): a frozen entry whose
    params fall outside the published space, or that violates a runner
    lowering constraint (tile must divide N; powers of two), is illegal —
    whether it was measured or not.
    """
    from .config import ConfigError, DesignSpace

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
        bad.append(f"block {block!r} not a positive power of two")
    if unit == "radix4-scalar" and 4 not in sp.radix:
        bad.append("radix-4 not in the published radix space")
    if unit == "dense-dot-fft64":
        tile, rows = params.get("tile"), params.get("rows")
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
        if real != "matrix-resident":
            bad.append(f"realization {real!r} illegal for dense-dot-fft64")
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
        if real != "grouped-boundary":
            bad.append(f"realization {real!r} illegal for grouped boundary")
    if "variant" in params and params["variant"] not in sp.batch_variants:
        bad.append(
            f"variant {params['variant']!r} not in {sp.batch_variants}")
    if "batch" in params and params["batch"] not in sp.batch_candidates:
        bad.append(f"batch {params['batch']!r} not in {sp.batch_candidates}")
    if "radix" in params and params["radix"] not in sp.radix:
        bad.append(f"radix {params['radix']!r} not in {sp.radix}")
    return bad


@dataclass(frozen=True)
class Selection:
    """One selector verdict. ``source`` is ``table`` or ``abstain``."""

    n: int
    direction: str
    batch: int
    precision: str
    source: str
    reason: str
    block: int | None = None
    flow_id: str | None = None
    flow_realization: str | None = None
    processing_unit: str | None = None
    params: dict = field(default_factory=dict)
    incumbent_ms: float | None = None
    challenger_ms: float | None = None

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
    if violations:
        raise ConfigError(
            "selector table violates the shared design space / lowering "
            f"contract ({len(violations)} entries): "
            + "; ".join(violations[:8])
            + (" ..." if len(violations) > 8 else "")
            + " -- rebuild the table from measurements under the current "
            "config/butterfly_space.json")
    return table


def _abstain(n, direction, batch, precision, reason) -> Selection:
    return Selection(n=n, direction=direction, batch=batch,
                     precision=precision, source="abstain", reason=reason)


def select(direction: str, n: int, batch: int = 1, precision: str = "fp32",
           profile_hash: str | None = None,
           table: dict | None = None,
           env_hash: str | None = "auto") -> Selection:
    """Return the measured winner for a shape, or abstain outside the matrix.

    ``env_hash`` verifies the table against the live environment when the
    table carries one: ``"auto"`` recomputes it (fail closed when the
    environment is unknown), an explicit string is trusted as given, ``None``
    cannot verify and abstains. Tables without ``env_hash`` skip the gate.
    """
    table = table if table is not None else load_table()
    if direction not in ("dit", "dif"):
        return _abstain(n, direction, batch, precision,
                        f"direction must be dit/dif, got {direction!r}")
    if precision != table["precision"]:
        return _abstain(n, direction, batch, precision,
                        f"table measured {table['precision']}, got {precision!r}")
    if profile_hash is None:
        return _abstain(n, direction, batch, precision,
                        "profile_hash is required; None cannot bypass the "
                        "table (fall back to M0)")
    if profile_hash != table["profile_hash"]:
        return _abstain(n, direction, batch, precision,
                        f"profile hash {profile_hash} != table "
                        f"{table['profile_hash']}; re-measure before selecting")
    if table.get("env_hash"):
        cur = env_hash
        if cur == "auto":
            from .envcheck import current_env_hash
            cur = current_env_hash()
        if cur is None:
            return _abstain(
                n, direction, batch, precision,
                "environment fingerprint unknown (device/runtime/compiler "
                "not probeable); this table cannot be verified here and "
                "must fail closed to M0")
        if cur != table["env_hash"]:
            return _abstain(
                n, direction, batch, precision,
                f"environment hash {cur} != table {table['env_hash']} "
                "(device/runtime/compiler changed since the table was "
                "built); re-run the probe and re-measure before selecting")
    for e in table["entries"]:
        if (e["N"] == n and e["batch"] == batch and e["direction"] == direction):
            return Selection(
                n=n, direction=direction, batch=batch, precision=precision,
                source="table",
                reason=e.get("reason", "measured confirmation winner"),
                block=e["block"], flow_id=e["flow_id"],
                flow_realization=e["flow_realization"],
                processing_unit=e["processing_unit"],
                params=dict(e.get("params") or {}),
                incumbent_ms=e.get("incumbent_ms"),
                challenger_ms=e.get("challenger_ms"),
            )
    return _abstain(n, direction, batch, precision,
                    f"no measured entry for N={n} batch={batch} "
                    f"direction={direction}; fall back to M0")
