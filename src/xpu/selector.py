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
    return table


def _abstain(n, direction, batch, precision, reason) -> Selection:
    return Selection(n=n, direction=direction, batch=batch,
                     precision=precision, source="abstain", reason=reason)


def select(direction: str, n: int, batch: int = 1, precision: str = "fp32",
           profile_hash: str | None = None,
           table: dict | None = None) -> Selection:
    """Return the measured winner for a shape, or abstain outside the matrix."""
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
