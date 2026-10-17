"""Per-semantic candidate families and legality (issue #8 PR-G).

The selector's exact key carries ``(transform_type,
transform_direction, normalization)``; each semantic owns its own
candidate family, legality and promotion guard. Families are
membership rules only -- ordering always comes from that semantic's
own measured confirmation rows (a forward row can never promote an
inverse/R2C/C2R entry).

Current families share the measured butterfly core (every semantic
runs it through its own wrapper: conjugate in/out for inverse,
packed-compat pack/crop/epilogue for R2C/C2R -- those wrappers are
part of the semantic's measured path, verified in its own
confirmation rows). Semantic-specific cores land in
:data:`SPECIALIZED_FLOWS` (PR-H: dedicated R2C/C2R pack/Hermitian/
epilogue flows) and join only their own family.

Budget (issue #8): at most 24 pre-registered legal candidates per
semantic/shape; :func:`family_size` is the gate the builder checks
against a measured contender set.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import (
    NORMALIZATIONS,
    TRANSFORM_DIRECTIONS,
    TRANSFORM_TYPES,
)

# Label -> (transform_type, transform_direction) for the four semantics
# of the unified matrix. Normalization for these labels is "backward"
# (the matrix's fixed norm); labels with a norm suffix reuse the
# shared taxonomy.
SEMANTIC_SPECS: dict[str, tuple[str, str, str]] = {
    "c2c-forward": ("c2c", "forward", "backward"),
    "c2c-inverse": ("c2c", "inverse", "backward"),
    "r2c": ("r2c", "forward", "backward"),
    "c2r": ("c2r", "inverse", "backward"),
}

# Flow ids every semantic reaches through its own wrapper today.
CORE_FLOWS = frozenset({
    "baseline-global-stage",
    "batch-spread",
    "batch-folded",
    "local-fused-tile",
})

# Semantic-specific cores (PR-H hook): flow ids legal ONLY for the
# named transform_type. Empty until a dedicated real-transform flow
# lands; entries naming anything else here would be family-illegal.
SPECIALIZED_FLOWS: dict[str, frozenset] = {
    "c2c": frozenset(),
    "r2c": frozenset(),
    "c2r": frozenset(),
}


@dataclass(frozen=True)
class SemanticSpec:
    label: str
    transform_type: str
    transform_direction: str
    normalization: str

    @property
    def key(self) -> tuple:
        return (self.transform_type, self.transform_direction,
                self.normalization)


def spec(label: str) -> SemanticSpec:
    """Spec for one matrix semantic label (fail-closed on unknown)."""
    if label not in SEMANTIC_SPECS:
        raise KeyError(
            f"unknown semantic label {label!r}; known: "
            f"{sorted(SEMANTIC_SPECS)}")
    tt, tdir, norm = SEMANTIC_SPECS[label]
    return SemanticSpec(label, tt, tdir, norm)


def family(transform_type: str) -> frozenset:
    """Legal flow ids for one transform type (core + specialized)."""
    if transform_type not in TRANSFORM_TYPES:
        raise KeyError(
            f"unknown transform_type {transform_type!r}; "
            f"known: {TRANSFORM_TYPES}")
    return CORE_FLOWS | SPECIALIZED_FLOWS[transform_type]


def family_allows(transform_type: str, flow_id: str) -> bool:
    return flow_id in family(transform_type)


def family_size(transform_type: str) -> int:
    """Legal-candidate count bound (issue #8: <= 24 per semantic/shape).

    Families themselves are far smaller than the budget; the budget
    also bounds what a confirmation session may register per
    semantic/shape -- measured via the bounded oracle set, checked by
    the builder.
    """
    return len(family(transform_type))


def validate_taxonomy() -> list[str]:
    """Self-check: specs agree with the shared key taxonomy."""
    bad = []
    for label, (tt, tdir, norm) in SEMANTIC_SPECS.items():
        if tt not in TRANSFORM_TYPES:
            bad.append(f"{label}: transform_type {tt!r} not in taxonomy")
        if tdir not in TRANSFORM_DIRECTIONS:
            bad.append(f"{label}: transform_direction {tdir!r} not in "
                       f"taxonomy")
        if norm not in NORMALIZATIONS:
            bad.append(f"{label}: normalization {norm!r} not in taxonomy")
    for tt in TRANSFORM_TYPES:
        if tt not in SPECIALIZED_FLOWS:
            bad.append(f"SPECIALIZED_FLOWS missing {tt!r}")
    return bad
