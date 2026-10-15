"""Candidate enumeration and lowering-legality checking (host-only, no torch).

A candidate is one point in the design space ``a = (Us, Ts, Ud, Td, layout,
boundary_residence, pipeline, processing_unit)`` plus the flow-specific knobs
(radix, bit-reversal fold, local tile, M2 rows, M3 group parameters). The
migration convergence plan requires that every candidate pass a **lowering
contract** before it is timed: an unsupported candidate must be rejected as
``not-lowered`` or ``infeasible``, not silently scored by the performance
model.

Schema v2 (review round) separates three attribution axes that were
previously conflated:

* ``flow_id`` — the lowering identity (which kernel family consumes the
  point): ``baseline-global-stage`` | ``local-fused-tile`` | ``batch-spread``
  | ``batch-folded``.
* ``flow_realization`` — how the flow is realized on device:
  ``global-stage`` | ``fused-inplace`` | ``matrix-resident`` |
  ``grouped-boundary`` (M0/M4/M5, M1, M2, M3 respectively).
* ``processing_unit`` — the computational core:
  ``radix2-scalar`` | ``radix4-scalar`` | ``dense-dot-fft64`` |
  ``composite-linear-map``.
* ``local_residence`` — where intermediate values live inside one launch:
  ``none`` | ``cache`` | ``register`` | ``implicit-dot`` |
  ``register+implicit-dot``, derived from the realization unless explicitly
  evidenced otherwise; ``boundary_residence`` (formerly ``residence``)
  records that every implemented flow round-trips global memory at stage
  boundaries (``gm``).

M1 (``fused-inplace``), M2 (``matrix-resident``) and M3
(``grouped-boundary``) are enumerated here from the config axes, not only
timed ad hoc by the bench, so the lowering contract covers every flow the
experiment matrix reports.

This module only encodes the contract the *current* P800 lowering is known to
accept. It does not guess performance. The candidate state machine is:

    enumerated -> legal -> compiled -> correctness-passed -> measured

``legal`` is decided here; the later states are recorded by the runner/bench.
"""

from dataclasses import dataclass, asdict

from .config import DesignSpace, DeviceProfile

PROCESSING_UNITS = ("radix2-scalar", "radix4-scalar", "dense-dot-fft64",
                    "composite-linear-map")
# Butterfly radix of the flow's global (per-stage) tail for each unit.
UNIT_RADIX = {
    "radix2-scalar": 2,
    "radix4-scalar": 4,
    "dense-dot-fft64": 2,
    "composite-linear-map": 2,
}
# Which units each flow realization may lower to.
REALIZATION_UNITS = {
    "global-stage": ("radix2-scalar", "radix4-scalar"),
    "fused-inplace": ("radix2-scalar", "radix4-scalar"),
    "matrix-resident": ("dense-dot-fft64",),
    "grouped-boundary": ("composite-linear-map",),
}
FLOW_REALIZATIONS = ("global-stage", "fused-inplace", "matrix-resident",
                     "grouped-boundary")
LOCAL_RESIDENCES = ("none", "cache", "register", "implicit-dot",
                    "register+implicit-dot")
# Residency each realization may claim; anything else would be a fabricated
# attribution (the review's "when evidenced" constraint).
REALIZATION_LOCAL_RESIDENCES = {
    "global-stage": ("none",),
    "fused-inplace": ("none", "cache"),
    "matrix-resident": ("register", "implicit-dot", "register+implicit-dot"),
    "grouped-boundary": ("register",),
}
_DERIVED_LOCAL_RESIDENCE = {
    "global-stage": "none",
    "fused-inplace": "cache",
    "matrix-resident": "register+implicit-dot",
    "grouped-boundary": "register",
}


class LoweringStatus:
    LEGAL = "legal"
    NOT_LOWERED = "not-lowered"
    INFEASIBLE = "infeasible"


@dataclass(frozen=True)
class Candidate:
    """One point in the design space (schema v2)."""

    flow_id: str                 # "baseline-global-stage" | "local-fused-tile"
                                 # | "batch-spread" | "batch-folded"
    processing_unit: str         # PROCESSING_UNITS (schema v2 vocabulary)
    radix: int
    us: int = 1
    ts: int = 1
    ud: int = 1
    td: int = 1
    boundary_residence: str = "gm"   # stage-boundary memory: "gm" only today
    pipeline: str = "none"
    layout: str = "planar-soa"
    block: int = 1024
    fused_tile: int = 1          # 1 = no local tile; >1 = M1/M2/M3 local tile
    dit_bitrev_input_fold: bool = True
    dif_bitrev_output_fold: bool = True
    batch: int = 1               # M4: transforms per call (workload axis)
    batch_variant: str = "none"  # "none" | "spread" (Ud) | "folded" (Td)
    flow_realization: str = ""   # FLOW_REALIZATIONS; "" = derived (see module)
    local_residence: str = ""    # LOCAL_RESIDENCES; "" = derived from realization
    rows: int = 1                # M2/M3: tile-blocks per program (power of two)
    glog: int = 0                # M3: tail stages per execution group (0 = n/a)
    maxp: int = 0                # M3: cap on the group period (0 = runner default)
    cells: int = 1               # M3: BLOCK windows per group-kernel program

    def to_dict(self) -> dict:
        d = asdict(self)
        d["flow_realization"] = effective_flow_realization(self)
        d["local_residence"] = effective_local_residence(self)
        return d


def effective_flow_realization(cand: Candidate) -> str:
    """The candidate's flow realization (explicit, else derived from the flow)."""
    if cand.flow_realization:
        return cand.flow_realization
    if cand.flow_id in ("baseline-global-stage", "batch-spread", "batch-folded"):
        return "global-stage"
    if cand.glog >= 1:
        return "grouped-boundary"
    if cand.rows > 1:
        return "matrix-resident"
    return "fused-inplace"


def effective_local_residence(cand: Candidate) -> str:
    """The candidate's local residence (explicit, else the realized default)."""
    if cand.local_residence:
        return cand.local_residence
    return _DERIVED_LOCAL_RESIDENCE[effective_flow_realization(cand)]


@dataclass(frozen=True)
class Verdict:
    """Lowering-contract verdict for a candidate."""

    candidate: Candidate
    status: str                  # LoweringStatus.*
    reasons: tuple = ()          # empty when legal; else why it is rejected

    @property
    def legal(self) -> bool:
        return self.status == LoweringStatus.LEGAL

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate.to_dict(),
            "status": self.status,
            "reasons": list(self.reasons),
        }


def _pow2(name: str, val: int, reasons: list, minimum: int = 1) -> None:
    if val < minimum or (val & (val - 1)) != 0:
        reasons.append(f"{name}: must be a power of two >= {minimum}, got {val}")


def check_lowering(cand: Candidate, profile: DeviceProfile) -> Verdict:
    """Apply the current P800 lowering contract to a candidate.

    The contract encodes only what the tested Triton 3.0.0 XPU lowering is
    known to accept. Rejections carry a machine-readable reason so the runner
    can report ``not-lowered``/``infeasible`` before launch.
    """
    reasons = []
    realization = effective_flow_realization(cand)
    local_res = effective_local_residence(cand)

    # 1. Processing unit: implemented cores only, and the unit fixes the
    #    butterfly radix of the flow's global tail stages.
    if cand.processing_unit not in UNIT_RADIX:
        reasons.append(
            f"processing-unit: only {', '.join(PROCESSING_UNITS)} are "
            f"implemented; {cand.processing_unit!r} is not-lowered"
        )
    elif cand.radix not in (2, 4):
        reasons.append(f"radix: must be 2 or 4, got {cand.radix}")
    elif cand.radix != UNIT_RADIX[cand.processing_unit]:
        reasons.append(
            f"processing-unit: {cand.processing_unit!r} must match "
            f"radix={UNIT_RADIX[cand.processing_unit]} (got radix={cand.radix})"
        )

    # 2. The public layout is complex64; the internal lowering is planar SoA.
    if cand.layout not in ("planar-soa",):
        reasons.append(
            f"layout: internal lowering accepts 'planar-soa', got {cand.layout!r}"
        )

    # 3. Boundary residence beyond global memory requires a shared/register
    #    lowering the current Triton XPU build cannot express for multi-stage
    #    fusion (SSA gather is rejected). Every implemented flow round-trips
    #    GM at stage/group boundaries; local_residence covers what happens
    #    inside one launch.
    if cand.boundary_residence not in ("gm",):
        reasons.append(
            f"boundary_residence: multi-stage on-chip boundary residence is not-lowered "
            f"by the tested Triton XPU lowering (SSA gather x[idx] rejected); "
            f"got {cand.boundary_residence!r}"
        )

    # 4. Structural knobs must be powers of two (block/tile/rows/cells/maxp);
    #    glog is a stage count >= 0 (0 = grouping disabled).
    _pow2("block", cand.block, reasons)
    _pow2("fused_tile", cand.fused_tile, reasons)
    _pow2("rows", cand.rows, reasons)
    _pow2("cells", cand.cells, reasons)
    if cand.maxp != 0:
        _pow2("maxp", cand.maxp, reasons)
    if cand.glog < 0:
        reasons.append(f"glog: must be >= 0, got {cand.glog}")

    # 5. Flow identity and realization must agree, and the realization fixes
    #    the legal processing units (M2 only lowers as dense-dot-fft64, M3
    #    only as composite-linear-map, M0/M4/M5 and M1 as scalar radix cores).
    if realization not in FLOW_REALIZATIONS:
        reasons.append(
            f"flow_realization: must be one of {', '.join(FLOW_REALIZATIONS)}, "
            f"got {realization!r}"
        )
    else:
        wants_local_flow = realization != "global-stage"
        if wants_local_flow and cand.flow_id != "local-fused-tile":
            reasons.append(
                f"flow_realization: {realization!r} requires flow 'local-fused-tile', "
                f"got {cand.flow_id!r}"
            )
        if not wants_local_flow and cand.flow_id not in (
            "baseline-global-stage", "batch-spread", "batch-folded",
        ):
            reasons.append(
                f"flow_realization: 'global-stage' requires the baseline or a batch "
                f"flow, got {cand.flow_id!r}"
            )
        allowed_units = REALIZATION_UNITS.get(realization, ())
        if allowed_units and cand.processing_unit not in allowed_units:
            reasons.append(
                f"processing-unit: realization {realization!r} lowers only as "
                f"{', '.join(allowed_units)}; got {cand.processing_unit!r}"
            )
        allowed_res = REALIZATION_LOCAL_RESIDENCES.get(realization, ())
        if allowed_res and local_res not in allowed_res:
            reasons.append(
                f"local_residence: realization {realization!r} evidences only "
                f"{', '.join(allowed_res)}; got {local_res!r}"
            )
        if realization == "grouped-boundary" and cand.glog < 1:
            reasons.append(
                f"glog: realization 'grouped-boundary' needs glog >= 1, got {cand.glog}"
            )

    # 5b. Fused local tile legality: tile>1 pairs with the local-fused-tile
    #     flow (M1/M2/M3), tile=1 with the baseline or a batch flow (M0/M4/M5).
    if cand.fused_tile > 1 and cand.flow_id != "local-fused-tile":
        reasons.append(
            f"flow_id: fused_tile={cand.fused_tile} requires flow 'local-fused-tile', "
            f"got {cand.flow_id!r}"
        )
    if cand.fused_tile == 1 and cand.flow_id not in (
        "baseline-global-stage", "batch-spread", "batch-folded",
    ):
        reasons.append(
            f"flow_id: fused_tile=1 requires flow 'baseline-global-stage' or a "
            f"batch flow, got {cand.flow_id!r}"
        )

    # 6. Data-space/time expansion (Ud/Td) is lowered through batch variants:
    #    M4 spreads (Ud=B) or folds (Td=B) the M0 stage cascade; M2 batch
    #    spreads the matrix group (Ud=B, batch on grid axis 1); M5 batch
    #    spreads the radix-4 cascade (Ud=B). A folded Td packet loop on
    #    M2/M5 (one worker consuming multiple packets with residency) is
    #    **not-lowered** this round -- index flattening without worker
    #    residency would misattribute Td.
    if cand.batch_variant == "none":
        if cand.batch != 1:
            reasons.append(
                f"batch: batch={cand.batch} needs a batch variant ('spread'/'folded'); "
                f"got batch_variant='none'"
            )
        if cand.flow_id in ("batch-spread", "batch-folded"):
            reasons.append(
                f"flow_id: {cand.flow_id!r} requires batch_variant "
                f"'{cand.flow_id.removeprefix('batch-')}', got 'none'"
            )
        if cand.ud != 1 or cand.td != 1:
            reasons.append(
                f"data expansion: Ud/Td != 1 is lowered only through a batch "
                f"variant; got Ud={cand.ud}, Td={cand.td}"
            )
    else:
        if cand.batch_variant not in ("spread", "folded"):
            reasons.append(
                f"batch_variant: must be 'none'/'spread'/'folded', got {cand.batch_variant!r}"
            )
            return Verdict(candidate=cand, status=LoweringStatus.NOT_LOWERED,
                           reasons=tuple(reasons))
        if cand.batch <= 0 or (cand.batch & (cand.batch - 1)) != 0:
            reasons.append(f"batch: must be a power of two, got {cand.batch}")
        if cand.batch_variant == "spread":
            # M4/M5: the batched stage cascade is flow 'batch-spread'.
            # M2 batch: the matrix group keeps flow 'local-fused-tile'; the
            # batch is an expansion axis on that flow, not a new flow id.
            if realization == "matrix-resident":
                if cand.flow_id != "local-fused-tile":
                    reasons.append(
                        f"flow_id: matrix-resident batch keeps 'local-fused-tile', "
                        f"got {cand.flow_id!r}"
                    )
            elif cand.flow_id != "batch-spread":
                reasons.append(
                    f"flow_id: spread variant requires 'batch-spread' for "
                    f"{realization!r}, got {cand.flow_id!r}"
                )
        else:  # folded
            if realization != "global-stage" or cand.processing_unit != "radix2-scalar":
                reasons.append(
                    f"batch_variant: folded (Td) lowers the M4 radix-2 stage "
                    f"cascade only this round; {realization}/"
                    f"{cand.processing_unit} folded is not-lowered"
                )
            elif cand.flow_id != "batch-folded":
                reasons.append(
                    f"flow_id: folded variant requires 'batch-folded', "
                    f"got {cand.flow_id!r}"
                )
        if cand.fused_tile > 1 and not (
            realization == "matrix-resident" and cand.batch_variant == "spread"
        ):
            reasons.append(
                f"fused_tile: batched paths consume fused_tile only for the "
                f"matrix-resident spread lowering; got fused_tile={cand.fused_tile}"
            )
        if cand.batch_variant == "spread":
            if cand.ud != cand.batch or cand.td != 1:
                reasons.append(
                    f"data expansion: spread requires Ud=batch={cand.batch}, Td=1; "
                    f"got Ud={cand.ud}, Td={cand.td}"
                )
        else:
            if cand.ud != 1 or cand.td != cand.batch:
                reasons.append(
                    f"data expansion: folded requires Ud=1, Td=batch={cand.batch}; "
                    f"got Ud={cand.ud}, Td={cand.td}"
                )

    # 7. Pipeline beyond 'none' needs producer/consumer overlap evidence.
    if cand.pipeline not in ("none",):
        reasons.append(f"pipeline: {cand.pipeline!r} is not-lowered (no overlap evidence)")

    if reasons:
        status = LoweringStatus.NOT_LOWERED
    else:
        status = LoweringStatus.LEGAL
    return Verdict(candidate=cand, status=status, reasons=tuple(reasons))


_RADIX_UNIT = {2: "radix2-scalar", 4: "radix4-scalar"}


def enumerate_candidates(space: DesignSpace, profile: DeviceProfile, block: int = 1024):
    """Enumerate the current design space and return (legal, rejected) verdicts.

    Every flow the experiment matrix reports is enumerated here: M0/M5 on the
    baseline flow (radix-2 / radix-4 scalar units), M1 fused-inplace tiles,
    M2 matrix-resident dense-dot tiles, M3 grouped-boundary tails, and the M4
    batch expansion points. Only axes the kernel actually consumes are
    enumerated; retained-but-unused fields are not invented here (the config
    schema forbids ignored fields).
    """
    legal, rejected = [], []
    for radix in space.radix:
        for fused_tile in space.fused_tile_candidates:
            # M1's fused local micro-FFT is a radix-2-only kernel; the M5
            # radix-4 core pairs only with the baseline (tile=1) flow.
            if radix != 2 and fused_tile != 1:
                continue
            flow = "local-fused-tile" if fused_tile > 1 else "baseline-global-stage"
            for dit_fold in space.dit_bitrev_input_fold:
                cand = Candidate(
                    flow_id=flow,
                    processing_unit=_RADIX_UNIT[radix],
                    radix=radix,
                    boundary_residence="gm",
                    layout="planar-soa",
                    block=block,
                    fused_tile=fused_tile,
                    dit_bitrev_input_fold=bool(dit_fold),
                    dif_bitrev_output_fold=True,
                    flow_realization=("fused-inplace" if fused_tile > 1
                                      else "global-stage"),
                )
                v = check_lowering(cand, profile)
                (legal if v.legal else rejected).append(v)

    # M2: register-resident local matrix (dense-dot-fft64), one program owns
    # `rows` consecutive tile-blocks.
    for tile in space.matrix_tiles:
        for rows in space.matrix_rows:
            for dit_fold in space.dit_bitrev_input_fold:
                cand = Candidate(
                    flow_id="local-fused-tile",
                    processing_unit="dense-dot-fft64",
                    radix=2,
                    boundary_residence="gm",
                    layout="planar-soa",
                    block=block,
                    fused_tile=tile,
                    dit_bitrev_input_fold=bool(dit_fold),
                    dif_bitrev_output_fold=True,
                    flow_realization="matrix-resident",
                    rows=rows,
                )
                v = check_lowering(cand, profile)
                (legal if v.legal else rejected).append(v)

    # M3: execution-group boundary tail (composite-linear-map).
    for tile in space.grouped_tiles:
        for rows in space.grouped_rows:
            for glog in space.grouped_glog:
                for maxp in space.grouped_maxp:
                    for cells in space.grouped_cells:
                        for dit_fold in space.dit_bitrev_input_fold:
                            cand = Candidate(
                                flow_id="local-fused-tile",
                                processing_unit="composite-linear-map",
                                radix=2,
                                boundary_residence="gm",
                                layout="planar-soa",
                                block=block,
                                fused_tile=tile,
                                dit_bitrev_input_fold=bool(dit_fold),
                                dif_bitrev_output_fold=True,
                                flow_realization="grouped-boundary",
                                rows=rows,
                                glog=glog,
                                maxp=maxp,
                                cells=cells,
                            )
                            v = check_lowering(cand, profile)
                            (legal if v.legal else rejected).append(v)

    # M4 batch expansion points: variant x B x bitrev-fold, all M0-shaped.
    for variant in space.batch_variants:
        for batch in space.batch_candidates:
            for dit_fold in space.dit_bitrev_input_fold:
                cand = Candidate(
                    flow_id=f"batch-{variant}",
                    processing_unit="radix2-scalar",
                    radix=2,
                    boundary_residence="gm",
                    layout="planar-soa",
                    block=block,
                    fused_tile=1,
                    dit_bitrev_input_fold=bool(dit_fold),
                    dif_bitrev_output_fold=True,
                    batch=batch,
                    batch_variant=variant,
                    flow_realization="global-stage",
                    ud=batch if variant == "spread" else 1,
                    td=batch if variant == "folded" else 1,
                )
                v = check_lowering(cand, profile)
                (legal if v.legal else rejected).append(v)

    # M2 batch (issue #3 P0.1): the matrix group with the batch axis in the
    # address computation (spread only; folded packet loops not-lowered).
    for tile in space.matrix_tiles:
        for rows in space.matrix_rows:
            for batch in space.batch_candidates:
                for dit_fold in space.dit_bitrev_input_fold:
                    cand = Candidate(
                        flow_id="local-fused-tile",
                        processing_unit="dense-dot-fft64",
                        radix=2,
                        boundary_residence="gm",
                        layout="planar-soa",
                        block=block,
                        fused_tile=tile,
                        dit_bitrev_input_fold=bool(dit_fold),
                        dif_bitrev_output_fold=True,
                        batch=batch,
                        batch_variant="spread",
                        flow_realization="matrix-resident",
                        rows=rows,
                        ud=batch,
                        td=1,
                    )
                    v = check_lowering(cand, profile)
                    (legal if v.legal else rejected).append(v)

    # M5 batch (issue #3 P0.2): the radix-4 cascade over a batch (spread
    # only; folded packet loops not-lowered).
    for batch in space.batch_candidates:
        for dit_fold in space.dit_bitrev_input_fold:
            cand = Candidate(
                flow_id="batch-spread",
                processing_unit="radix4-scalar",
                radix=4,
                boundary_residence="gm",
                layout="planar-soa",
                block=block,
                fused_tile=1,
                dit_bitrev_input_fold=bool(dit_fold),
                dif_bitrev_output_fold=True,
                batch=batch,
                batch_variant="spread",
                flow_realization="global-stage",
                ud=batch,
                td=1,
            )
            v = check_lowering(cand, profile)
            (legal if v.legal else rejected).append(v)
    return legal, rejected


def selected_candidate(legal_verdicts):
    """Pick the measured winner: lowest fused_tile (most conservative flow).

    The convergence plan requires a new selector never to choose a candidate
    slower than the incumbent without explicit opt-in, so the default selection
    is the M0 baseline (fused_tile=1); M1 must prove itself by measurement.
    Stable sort keeps the radix-2 baseline ahead of same-tile radix-4/batch
    points, so the incumbent stays the default until measurement says
    otherwise. Per-size measured selection is layered on top by
    ``src/xpu/selector.py`` (review round: table-driven, abstains outside the
    measured matrix).
    """
    if not legal_verdicts:
        return None
    return sorted(legal_verdicts, key=lambda v: v.candidate.fused_tile)[0]


def arithmetic_cost(n: int, radix: int) -> dict:
    """Analytic per-transform op counts for the ``baseline-global-stage`` flow.

    The M5 comparison the plan asks for: same flow (one launch and one global
    round trip per stage), different processing unit. Counting conventions:
    one complex twiddle multiply = one ``twiddle_mults`` unit; butterfly
    adds count real additions only (sign flips and +/-i swaps are free).

    * **radix-2**: ``log2(N)`` stages of ``N/2`` butterflies; one twiddle per
      butterfly; each butterfly = complex add + complex sub = 4 real adds.
    * **radix-4**: ``floor(log2(N)/2)`` radix-4 stages of ``N/4`` butterflies
      (three twiddles each: d=1,2,3; four output columns x three complex adds
      = 24 real adds), plus one radix-2 tail stage when ``log2(N)`` is odd.

    Radix-4 therefore runs **fewer twiddle multiplies and fewer launches but
    more real adds** than radix-2 on the same flow; both sides of the ledger
    are reported rather than a single "better" claim.
    """
    if n < 2 or (n & (n - 1)) != 0:
        raise ValueError(f"n must be a power of two >= 2, got {n}")
    if radix not in (2, 4):
        raise ValueError(f"radix must be 2 or 4, got {radix}")
    logn = n.bit_length() - 1
    if radix == 2:
        launches = logn
        butterflies = (n >> 1) * logn
        twiddle_mults = butterflies
        real_adds = butterflies * 4
    else:
        r4_stages = logn // 2
        tail = logn % 2
        launches = r4_stages + tail
        butterflies = (n >> 2) * r4_stages + tail * (n >> 1)
        twiddle_mults = 3 * (n >> 2) * r4_stages + tail * (n >> 1)
        real_adds = 24 * (n >> 2) * r4_stages + tail * (n >> 1) * 4
    return {
        "launches": launches,
        "butterflies": butterflies,
        "twiddle_mults": twiddle_mults,
        "real_adds": real_adds,
    }
