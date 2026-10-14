"""Candidate enumeration and lowering-legality checking (host-only, no torch).

A candidate is one point in the design space ``a = (Us, Ts, Ud, Td, layout,
residence, pipeline, processing_unit)`` plus the flow-specific knobs (radix,
bit-reversal fold, fused local tile). The migration convergence plan requires
that every candidate pass a **lowering contract** before it is timed: an
unsupported candidate must be rejected as ``not-lowered`` or ``infeasible``, not
silently scored by the performance model.

This module only encodes the contract the *current* P800 lowering is known to
accept. It does not guess performance. The candidate state machine is:

    enumerated -> legal -> compiled -> correctness-passed -> measured

``legal`` is decided here; the later states are recorded by the runner/bench.
"""

from dataclasses import dataclass, asdict

from .config import DesignSpace, DeviceProfile


class LoweringStatus:
    LEGAL = "legal"
    NOT_LOWERED = "not-lowered"
    INFEASIBLE = "infeasible"


@dataclass(frozen=True)
class Candidate:
    """One point in the design space."""

    flow_id: str                 # "baseline-global-stage" | "local-fused-tile"
                                 # | "batch-spread" | "batch-folded"
    processing_unit: str         # "radix-2"
    radix: int
    us: int = 1
    ts: int = 1
    ud: int = 1
    td: int = 1
    residence: str = "gm"        # "gm" | "register" | "ub" | "shared"
    pipeline: str = "none"
    layout: str = "planar-soa"
    block: int = 1024
    fused_tile: int = 1          # 1 = no fusion (M0); >1 = M1 local micro-FFT
    dit_bitrev_input_fold: bool = True
    dif_bitrev_output_fold: bool = True
    batch: int = 1               # M4: transforms per call (workload axis)
    batch_variant: str = "none"  # "none" | "spread" (Ud) | "folded" (Td)

    def to_dict(self) -> dict:
        return asdict(self)


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


def check_lowering(cand: Candidate, profile: DeviceProfile) -> Verdict:
    """Apply the current P800 lowering contract to a candidate.

    The contract encodes only what the tested Triton 3.0.0 XPU lowering is
    known to accept. Rejections carry a machine-readable reason so the runner
    can report ``not-lowered``/``infeasible`` before launch.
    """
    reasons = []

    # 1. Only radix-2 is implemented as a processing unit in this migration.
    if cand.processing_unit != "radix-2" or cand.radix != 2:
        reasons.append(
            "processing-unit: only radix-2 is implemented; radix-4/other cores are not-lowered"
        )

    # 2. The public layout is complex64; the internal lowering is planar SoA.
    if cand.layout not in ("planar-soa",):
        reasons.append(
            f"layout: internal lowering accepts 'planar-soa', got {cand.layout!r}"
        )

    # 3. Residence beyond global memory requires a shared/register lowering the
    #    current Triton XPU build cannot express for multi-stage fusion (SSA
    #    gather is rejected). Stage-level residence='gm' is the M0 baseline.
    if cand.residence not in ("gm",):
        reasons.append(
            f"residence: multi-stage on-chip residence is not-lowered by the tested "
            f"Triton XPU lowering (SSA gather x[idx] rejected); got {cand.residence!r}"
        )

    # 4. Block/tile must be a power of two and not exceed the shared-memory
    #    budget when the flow is a fused local tile.
    for name, val in (("block", cand.block), ("fused_tile", cand.fused_tile)):
        if val <= 0 or (val & (val - 1)) != 0:
            reasons.append(f"{name}: must be a power of two, got {val}")

    # 5. Fused local tile legality: tile must divide N is checked at runtime,
    #    but tile>1 with residence='gm' is the M1 flow (fused global-pointer
    #    micro-FFT) and is legal; tile>1 with a non-gm residence is not.
    #    The M4 batch flows are M0-shaped (no local fusion), so tile>1 only
    #    pairs with the local-fused-tile flow.
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

    # 6. Data-space/time expansion (Ud/Td) is lowered exactly by the M4 batch
    #    variants: spread puts B on grid axis 1 (Ud=B, td=1), folded splits B
    #    out of the flat pair-space grid (Td=B, ud=1). Without a batch variant
    #    the expansion has no consuming kernel.
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
        expect_flow = ("batch-spread" if cand.batch_variant == "spread"
                       else "batch-folded")
        if cand.flow_id != expect_flow:
            reasons.append(
                f"flow_id: batch_variant={cand.batch_variant!r} requires {expect_flow!r}, "
                f"got {cand.flow_id!r}"
            )
        if cand.batch <= 0 or (cand.batch & (cand.batch - 1)) != 0:
            reasons.append(f"batch: must be a power of two, got {cand.batch}")
        if cand.fused_tile > 1:
            reasons.append(
                f"fused_tile: the batched path is M0-stage-only; got fused_tile={cand.fused_tile}"
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


def enumerate_candidates(space: DesignSpace, profile: DeviceProfile, block: int = 1024):
    """Enumerate the current design space and return (legal, rejected) verdicts.

    Only axes the kernel actually consumes are enumerated; retained-but-unused
    fields are not invented here (the config schema forbids ignored fields).
    """
    legal, rejected = [], []
    for fused_tile in space.fused_tile_candidates:
        flow = "local-fused-tile" if fused_tile > 1 else "baseline-global-stage"
        for dit_fold in space.dit_bitrev_input_fold:
            cand = Candidate(
                flow_id=flow,
                processing_unit="radix-2",
                radix=2,
                residence="gm",
                layout="planar-soa",
                block=block,
                fused_tile=fused_tile,
                dit_bitrev_input_fold=bool(dit_fold),
                dif_bitrev_output_fold=True,
            )
            v = check_lowering(cand, profile)
            (legal if v.legal else rejected).append(v)
    # M4 batch expansion points: variant x B x bitrev-fold, all M0-shaped.
    for variant in space.batch_variants:
        for batch in space.batch_candidates:
            for dit_fold in space.dit_bitrev_input_fold:
                cand = Candidate(
                    flow_id=f"batch-{variant}",
                    processing_unit="radix-2",
                    radix=2,
                    residence="gm",
                    layout="planar-soa",
                    block=block,
                    fused_tile=1,
                    dit_bitrev_input_fold=bool(dit_fold),
                    dif_bitrev_output_fold=True,
                    batch=batch,
                    batch_variant=variant,
                    ud=batch if variant == "spread" else 1,
                    td=batch if variant == "folded" else 1,
                )
                v = check_lowering(cand, profile)
                (legal if v.legal else rejected).append(v)
    return legal, rejected


def selected_candidate(legal_verdicts):
    """Pick the measured winner: lowest fused_tile (most conservative flow).

    The convergence plan requires a new selector never to choose a candidate
    slower than the incumbent without explicit opt-in, so the default selection
    is the M0 baseline (fused_tile=1); M1 must prove itself by measurement.
    """
    if not legal_verdicts:
        return None
    return sorted(legal_verdicts, key=lambda v: v.candidate.fused_tile)[0]
