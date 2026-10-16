"""Issue #3 P1.3 — structured cost model, bounded search, holdout validation.

The model screens candidates; **measurement always decides**. Terms are
physical and structured, not fitted to run data:

* ``t_launch`` — per-kernel launch + sync, from the calibration profile;
* ``t_gm``     — analytical GM bytes / device-internal bandwidth (``d2d``
  anchor from the calibration profile; no hardware counters exist, so the
  byte counts are declared estimates);
* ``t_compute`` — declared flop counts / (efficiency x measured fp32
  peak from the calibration profile).

Efficiency constants are declared here a priori and never fitted: every
measured shape is therefore out-of-sample (holdout), which
:func:`validate` quantifies — per-family error, incumbent-relative guard
decision agreement, and top-K hit rate for the bounded search.

Families use the protocol names of ``scripts/run_batch_release.py``.
"""

from __future__ import annotations

from .calibration import load as load_calibration
from .selector import GUARD_PCT  # single promotion guard for model+table

# a-priori compute efficiencies (declared; never fitted to run data)
EFFICIENCY = {
    "scalar-radix2": 0.12,   # M0 loop / M4 folded / M4 spread stages
    "scalar-radix4": 0.15,   # M5 radix-4 spans
    "dot-matrix": 0.30,      # M2 register-resident matrix group
}

def m2_space() -> list[dict]:
    """The published M2 search space -- read from the single shared
    candidate description (config/butterfly_space.json P.register_matrix),
    the same source the harness enumeration, lowering contract and
    selector validation use (issue #3 fix: one design space)."""
    from .config import DesignSpace

    sp = DesignSpace.load()
    return [{"tile": t, "rows": r}
            for t in sp.matrix_tiles for r in sp.matrix_rows]


def _log2(n: int) -> int:
    return n.bit_length() - 1


def _family_of(name: str) -> str | None:
    """Map a harness candidate name onto a model family."""
    if name in ("baseline_m0_rank1_loop", "incumbent_m4_folded",
                "m4_spread", "m2_batch", "m5_batch"):
        return name
    if name.startswith("m2_packet"):
        return "m2_packet"
    if name.startswith("m2_t"):
        return "m2_batch"
    return None


def axes(family: str, n: int, batch: int, params: dict | None = None) -> dict:
    """Structured axes: launches, analytical GM bytes, declared flops."""
    p = dict(params or {})
    logn = _log2(n)
    points = n * batch
    if family == "baseline_m0_rank1_loop":
        launches = batch * logn  # host loop: logn launches per rank-1 row
        gm = launches * 2 * n * 8  # each launch moves one row (N points)
        flops = 3 * n * batch * logn
        eff = "scalar-radix2"
    elif family in ("incumbent_m4_folded", "m4_spread"):
        launches = logn
        gm = launches * 2 * points * 8
        flops = 3 * points * logn
        eff = "scalar-radix2"
    elif family == "m5_batch":
        launches = (logn + 1) // 2
        gm = launches * 2 * points * 8
        flops = 3.75 * points * launches
        eff = "scalar-radix4"
    elif family in ("m2_batch", "m2_packet"):
        tile = int(p.get("tile", 64))
        rows = int(p.get("rows", 8))
        packets = int(p.get("packets", 1))
        if tile <= 0 or (tile & (tile - 1)) != 0:
            raise ValueError(f"tile must be a power of two, got {tile}")
        launches = 2 + logn - _log2(tile)  # bitrev + matrix group + tail
        gm = launches * 2 * points * 8
        # resident coefficient table: (2*tile, tile) f32, once per program
        programs = max(1, points // (rows * tile))
        gm += programs * 2 * tile * tile * 4
        # dot group: 2*tile flops per point (rows cancels), x packets
        flops = 2 * tile * points * packets
        flops += 3 * points * (logn - _log2(tile))  # tail stages
        eff = "dot-matrix"
    else:
        raise ValueError(f"unknown family: {family!r}")
    return {"launches": launches, "gm_bytes": int(gm), "flops": float(flops),
            "efficiency": eff}


def predict(family: str, n: int, batch: int, params: dict | None = None,
            calib: dict | None = None) -> float:
    """Predicted wall time in ms. Screening only — never a final decision."""
    if family == "vendor":
        raise ValueError("vendor cost is external; measure, do not model")
    cal = calib if calib is not None else load_calibration()
    costs = cal["baseline_costs"]
    # per-launch anchor = measured 1-block dispatch + sync. Validation
    # showed anchoring on the 1024-block dispatch (12.2 ms on this stack:
    # per-program cost is not amortized) overpredicts everything by ~27x;
    # that anchor stays recorded in the calibration profile as an
    # observation, not a model constant.
    t_launch = float(costs["triton_launch_ms"]) + float(
        costs["cuda_synchronize_ms"])
    bw = float(costs["d2d_gib_per_s"]) * (1024.0 ** 3)  # B/s
    peak = float(costs["torch_mm_gflops"]) * 1e9  # flop/s
    ax = axes(family, n, batch, params)
    ms = ax["launches"] * t_launch
    ms += ax["gm_bytes"] / bw * 1e3
    ms += ax["flops"] / (EFFICIENCY[ax["efficiency"]] * peak) * 1e3
    return ms


def bounded_search(family: str, n: int, batch: int, k: int = 3,
                   space=None, calib: dict | None = None) -> list[dict]:
    """Top-k parameter sets by predicted cost (model screens, run measures).

    The default space is the M2 local-stage structure (tile x rows); block
    inherits the frozen B=1 search and only enters through the guard.
    """
    if family != "m2_batch":
        raise ValueError(
            f"bounded search supports m2_batch (the only tunable family); "
            f"got {family!r}")
    space = list(space) if space is not None else m2_space()
    scored = []
    for params in space:
        try:
            ms = predict(family, n, batch, params, calib=calib)
        except (ValueError, KeyError, FileNotFoundError):
            continue
        scored.append((ms, dict(params)))
    scored.sort(key=lambda t: t[0])
    return [p for _, p in scored[:max(0, int(k))]]


def records_from_summary(summary: dict) -> list[dict]:
    """Measured decision rows from a run_batch_release summary.json dict.

    Skips non-finite rows, candidates that failed the correctness
    threshold, and candidates whose session/trial evidence is incomplete
    (``complete: false`` when the artifact records it) -- the same rows
    the selector would refuse to promote.
    """
    thr = float((summary.get("manifest") or {}).get("error_threshold")
                or 1e-4)
    out = []
    for sh in summary.get("shapes") or []:
        for name, c in (sh.get("candidates") or {}).items():
            import math
            ms = c.get("median_ms")
            if not isinstance(ms, (int, float)) or not math.isfinite(ms) \
                    or ms <= 0:
                continue
            err = c.get("err_max", c.get("err"))
            if not isinstance(err, (int, float)) or not math.isfinite(err) \
                    or err >= thr:
                continue
            if c.get("complete") is False:
                continue
            out.append({"name": name, "N": sh["N"], "batch": sh["batch"],
                        "direction": sh.get("direction"),
                        "measured_ms": ms, "params": c.get("params")})
    return out


def validate(records, incumbent="incumbent_m4_folded",
             guard_pct: float | None = None) -> dict:
    """Holdout metrics vs measured rows (never used to tune constants).

    ``records``: iterable of dicts with keys family-or-name, N, batch,
    measured_ms; optional params and name. Reports:

    * ``mdape`` — median absolute % error per family (model vs measured);
    * ``guard_agreement`` — fraction of (challenger, incumbent) pairs where
      model and measurement agree on whether the challenger **clears the
      formal promotion guard** (``GUARD_PCT``, default 2% -- the same line
      the selector uses; sign-of-gain agreement is deliberately NOT the
      metric); the stable incumbent participates explicitly;
    * ``topk_hit`` / ``topk_shapes`` -- legacy measured-subset metric (K =
      3), optimistic when only the model's own top-K was measured;
    * ``topk_full_space`` -- for shapes where **all nine** published m2
      (tile, rows) points were measured (the holdout protocol): ``recall``
      (model top-3 contains the measured best) and mean selection
      ``regret_pct`` over the full space.
    """
    import statistics

    guard = float(GUARD_PCT if guard_pct is None else guard_pct)

    rows = []
    for r in records:
        name = r.get("name") or r.get("family")
        fam = r.get("family") or _family_of(name)
        ms = r.get("measured_ms")
        if fam is None or not isinstance(ms, (int, float)) or ms <= 0:
            continue
        try:
            pred = predict(fam, int(r["N"]), int(r["batch"]),
                           r.get("params") or {})
        except (ValueError, KeyError, FileNotFoundError):
            continue
        rows.append({"name": name, "family": fam, "N": r["N"],
                     "batch": r["batch"], "direction": r.get("direction"),
                     "measured_ms": ms, "pred_ms": pred,
                     "params": r.get("params")})

    mdape = {}
    for fam in sorted({r["family"] for r in rows}):
        fr = [r for r in rows if r["family"] == fam]
        errs = [abs(r["pred_ms"] - r["measured_ms"]) / r["measured_ms"] * 100
                for r in fr]
        mdape[fam] = statistics.median(errs) if errs else None

    # guard agreement vs the stable incumbent, per (N, batch)
    agree, pairs = 0, 0
    by_shape = {}
    for r in rows:
        by_shape.setdefault(
            (r["N"], r["batch"], r["direction"]), []).append(r)
    for shape_rows in by_shape.values():
        inc = [r for r in shape_rows if r["name"] == incumbent]
        if not inc:
            continue
        inc_r = inc[0]
        for r in shape_rows:
            if r["name"] in (incumbent, "vendor", "selected"):
                continue
            # promotion decision, NOT sign-of-gain: both sides must
            # clear the same formal guard line the selector uses
            meas_promote = (
                (inc_r["measured_ms"] - r["measured_ms"])
                / inc_r["measured_ms"] * 100.0) >= guard
            pred_promote = (
                (inc_r["pred_ms"] - r["pred_ms"])
                / inc_r["pred_ms"] * 100.0) >= guard
            pairs += 1
            if meas_promote == pred_promote:
                agree += 1

    # top-K metrics per shape. `topk_hit` stays the measured-subset
    # metric (optimistic when only the model's own top-K was measured);
    # `topk_full_space` is computed ONLY where all nine published m2
    # points were measured (issue #3 review fix: full-space recall/regret).
    topk_hits, topk_shapes = 0, 0
    full_recall, full_total, regrets = 0, 0, []
    full_space = {(int(s["tile"]), int(s["rows"])) for s in m2_space()}
    for shape_rows in by_shape.values():
        m2 = [r for r in shape_rows if r["family"] == "m2_batch"]
        if len(m2) < 2:
            continue
        topk_shapes += 1
        measured_best = min(m2, key=lambda r: r["measured_ms"])
        ranked = sorted(m2, key=lambda r: r["pred_ms"])
        names_ranked = [r["name"] for r in ranked[:3]]
        if measured_best["name"] in names_ranked:
            topk_hits += 1
        measured_space = set()
        for r in m2:
            prm = r.get("params") or {}
            measured_space.add((int(prm.get("tile", 64)),
                                int(prm.get("rows", 8))))
        if measured_space >= full_space:
            full_total += 1
            pick = ranked[0]
            if measured_best["name"] in names_ranked:
                full_recall += 1
            regrets.append(
                (pick["measured_ms"] - measured_best["measured_ms"])
                / measured_best["measured_ms"] * 100.0)

    return {
        "n_records": len(rows),
        "mdape_pct": mdape,
        "guard_pct": guard,
        "guard_agreement": (agree / pairs) if pairs else None,
        "guard_pairs": pairs,
        "topk_hit": (topk_hits / topk_shapes) if topk_shapes else None,
        "topk_shapes": topk_shapes,
        "topk_full_space": {
            "recall": (full_recall / full_total) if full_total else None,
            "shapes": full_total,
            "mean_regret_pct": (sum(regrets) / len(regrets))
            if regrets else None,
            "regrets_pct": regrets,
            "space_size": len(full_space),
        },
        "constants_declared": True,
        "fitted": False,
    }
