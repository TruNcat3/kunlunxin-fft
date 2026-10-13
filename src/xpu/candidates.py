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
    if cand.fused_tile > 1 and cand.flow_id != "local-fused-tile":
        reasons.append(
            f"flow_id: fused_tile={cand.fused_tile} requires flow 'local-fused-tile', "
            f"got {cand.flow_id!r}"
        )
    if cand.fused_tile == 1 and cand.flow_id != "baseline-global-stage":
        reasons.append(
            f"flow_id: fused_tile=1 requires flow 'baseline-global-stage', got {cand.flow_id!r}"
        )

    # 6. Data-space/time expansion (Ud/Td) has no batched lowering yet.
    if cand.ud != 1 or cand.td != 1:
        reasons.append(
            f"data expansion: Ud/Td != 1 has no batched lowering yet; got Ud={cand.ud}, Td={cand.td}"
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
