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

# a-priori compute efficiencies (declared; never fitted to run data)
EFFICIENCY = {
    "scalar-radix2": 0.12,   # M0 loop / M4 folded / M4 spread stages
    "scalar-radix4": 0.15,   # M5 radix-4 spans
    "dot-matrix": 0.30,      # M2 register-resident matrix group
}

_M2_SPACE = tuple(
    {"tile": t, "rows": r}
    for t in (32, 64, 128)
    for r in (4, 8, 16)
)


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
    space = list(space) if space is not None else [dict(s) for s in _M2_SPACE]
    scored = []
    for params in space:
        try:
            ms = predict(family, n, batch, params, calib=calib)
        except (ValueError, KeyError, FileNotFoundError):
            continue
        scored.append((ms, dict(params)))
    scored.sort(key=lambda t: t[0])
    return [p for _, p in scored[:max(0, int(k))]]


def validate(records, incumbent="incumbent_m4_folded") -> dict:
    """Holdout metrics vs measured rows (never used to tune constants).

    ``records``: iterable of dicts with keys family-or-name, N, batch,
    measured_ms; optional params and name. Reports:

    * ``mdape`` — median absolute % error per family (model vs measured);
    * ``guard_agreement`` — fraction of (challenger, incumbent) pairs where
      model and measurement agree on which side of the guard line the
      challenger lands (the stable incumbent participates explicitly);
    * ``topk_hit`` — fraction of shapes where the model's predicted-best
      measured variant is within K of the measured-best (K = 3).
    """
    import statistics

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
            meas_gain = inc_r["measured_ms"] - r["measured_ms"]
            pred_gain = inc_r["pred_ms"] - r["pred_ms"]
            pairs += 1
            if (meas_gain > 0) == (pred_gain > 0):
                agree += 1

    # top-K hit among measured M2 variants (m2 family), per shape
    topk_hits, topk_shapes = 0, 0
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

    return {
        "n_records": len(rows),
        "mdape_pct": mdape,
        "guard_agreement": (agree / pairs) if pairs else None,
        "guard_pairs": pairs,
        "topk_hit": (topk_hits / topk_shapes) if topk_shapes else None,
        "topk_shapes": topk_shapes,
        "constants_declared": True,
        "fitted": False,
    }
