#!/usr/bin/env python3
"""Build config/selector_table.json from confirmation-round bench JSONs.

Safety gates (review round P0): a row may only enter the table when its
timing is a finite positive number, its recorded relative error is below
the run's correctness threshold, and every input run shares one
device/profile/precision/protocol manifest. Rejected rows and rejected runs
are logged with reasons; they never influence an entry.

Selection semantics: for each (N, direction) B=1 shape the builder compares
correctness-gated challengers against the correctness-gated M0 incumbent and
keeps the challenger only when its **search-stage median improvement**
(compare: aggregated medians measured in the same run schedule, not paired
per-trial medians) clears the promotion guard. Batch entries pick the faster
M4 variant the same way; the table itself — never a prose claim — is the
source of which variant wins where.

Usage::

    python scripts/build_selector_table.py \
        results/p800/confirm_b1024.json results/p800/confirm_b256.json \
        --batch results/p800/confirm_batch.json \
        --out config/selector_table.json

``--focus`` appends issue #3 batch entries from a run_batch_release
summary.json: the focus run's own medians (median of session medians,
all sessions correctness-gated) are compared against the stable batch
incumbent (best of M4 folded/spread) under the same 2% promotion guard.
A focus run with fewer than ``--min-focus-sessions`` sessions (default
3, the issue's protocol floor) is refused.

``--matrix-run`` appends issue #8 exact-semantic entries from
semantic-matrix run directories (``summary.json`` + raw sessions). A
cell only produces an entry when the artifact passes the strict output
gate (>= 3 sessions x >= 30 trials, stamped result_sha256), its oracle
winner has correctness-clean timing in every session, the plan geometry
is consistent and contract-legal, the flow belongs to that semantic's
own family, and the winner clears the promotion guard against the
matrix incumbent. Entries carry the full ``(transform_type,
transform_direction, normalization)`` key and answer only that triple;
legacy entries stay as the deployed old path for every other semantic.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_SCRIPT_DIR))
if _SCRIPT_DIR not in sys.path:
    # sibling scripts (run_semantic_matrix) for the matrix gates
    sys.path.insert(0, _SCRIPT_DIR)

from src.xpu import semantic as semantic_mod  # noqa: E402
from src.xpu.config import ConfigError, DeviceProfile  # noqa: E402
from src.xpu.envcheck import env_hash as fingerprint_env  # noqa: E402
from src.xpu.evidence import focus_candidate_verdict  # noqa: E402
from src.xpu.manifest_contract import (  # noqa: E402
    validate_benchmark_output,
)
from src.xpu.selector import GUARD_PCT  # noqa: E402
from src.xpu.selector import entry_contract_violations  # noqa: E402

import run_semantic_matrix as sm  # noqa: E402

# candidate key -> (flow_id, flow_realization, processing_unit, params-beyond-block)
_UNIT = {
    "m0": ("baseline-global-stage", "global-stage", "radix2-scalar", {}),
    "m5": ("baseline-global-stage", "global-stage", "radix4-scalar", {"radix": 4}),
}

# M1 is DIT-only in this tree (fft_dit_fused); there is no DIF fused lowering.
_M1_DIRECTIONS = ("dit",)

_MANIFEST_KEYS = ("device", "profile_hash", "precision",
                  "correctness_threshold", "warmup", "trials", "dtype",
                  "timing_scope", "protocol_id")

_SUPPORTED_PRECISIONS = {"fp32": "complex64"}


def _descriptor(key: str, block: int, extra: dict) -> dict:
    """Map a bench candidate key onto its schema-v2 attribution + params."""
    if key.startswith("m0_"):
        flow, real, unit, params = _UNIT["m0"]
    elif key.startswith("m1_t"):
        flow, real, unit = "local-fused-tile", "fused-inplace", "radix2-scalar"
        params = {"tile": int(key.split("_t")[-1])}
    elif key.startswith("m2_"):
        flow, real, unit = "local-fused-tile", "matrix-resident", "dense-dot-fft64"
        params = {"tile": int(key.split("_t")[-1]), "rows": int(extra["matrix_rows"])}
    elif key.startswith("m3_g_"):
        flow, real, unit = ("local-fused-tile", "grouped-boundary",
                            "composite-linear-map")
        params = {"tile": int(key.split("_t")[-1]), "glog": int(extra["glog"]),
                  "maxp": int(extra["maxp"]), "cells": int(extra["cells"])}
    elif key.startswith("m5_r4_"):
        flow, real, unit, params = _UNIT["m5"]
    elif key.startswith("m4_"):
        parts = key.split("_")           # m4, variant, bNN, dit|dif
        variant, bkey = parts[1], parts[2]
        flow = f"batch-{variant}"
        real, unit = "global-stage", "radix2-scalar"
        params = {"variant": variant, "batch": int(bkey[1:])}
    else:
        raise ConfigError(f"unmapped candidate key: {key}")
    return {"block": block, "flow_id": flow, "flow_realization": real,
            "processing_unit": unit, "params": params}


def _err_field(key: str) -> str:
    """Bench row field holding the relative error for a candidate key."""
    if key.startswith("m0_dit"):
        return "dit_err"
    if key.startswith("m0_dif"):
        return "dif_err"
    return f"{key}_err"


def _finite_positive(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(v) and v > 0


def _median(xs):
    s = sorted(xs)
    return s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1]
                                                    + s[len(s) // 2])


def _check_ms(row, key: str, threshold: float, path: str, n, rejections):
    """Return the timing when (ms, err) pass the safety gates, else None."""
    ms = row.get(f"{key}_ms")
    if not _finite_positive(ms):
        rejections.append(
            f"{path} N={n} key={key}: ms not finite positive ({ms!r})")
        return None
    err = row.get(_err_field(key))
    if not isinstance(err, (int, float)) or not math.isfinite(err):
        rejections.append(
            f"{path} N={n} key={key}: missing/non-finite error ({err!r})")
        return None
    if err >= threshold:
        rejections.append(
            f"{path} N={n} key={key}: max_rel_error {err:.3e} >= {threshold:g}")
        return None
    return ms


def _load_run(path: str, rejections) -> dict:
    """Load one confirmation run; reject it outright on manifest problems."""
    with open(path) as f:
        data = json.load(f)
    manifest = data.get("manifest") or {}
    missing = [k for k in _MANIFEST_KEYS if manifest.get(k) in (None, "")]
    if missing:
        raise ConfigError(
            f"run {path}: manifest missing required keys {missing}; "
            "refusing to build a table from an incomplete manifest")
    thr = manifest["correctness_threshold"]
    if (not isinstance(thr, (int, float)) or not math.isfinite(thr)
            or not (0 < float(thr) < 1)):
        raise ConfigError(
            f"run {path}: correctness_threshold {thr!r} must be a finite "
            "number in (0, 1); refusing to build")
    precision = manifest["precision"]
    want_dtype = _SUPPORTED_PRECISIONS.get(precision)
    if want_dtype is None:
        raise ConfigError(
            f"run {path}: precision {precision!r} is not a supported "
            f"table precision ({sorted(_SUPPORTED_PRECISIONS)}); "
            "refusing to merge mixed-precision measurements")
    if manifest["dtype"] != want_dtype:
        raise ConfigError(
            f"run {path}: dtype {manifest['dtype']!r} does not match "
            f"precision {precision!r} (expected {want_dtype}); "
            "refusing to build")
    return {
        "path": path,
        "rows": data["rows"],
        "manifest": manifest,
        "block": int(manifest["block"]),
        "extra": manifest.get("extra") or {},
    }


def _check_manifest_consistency(runs, table_profile_hash: str) -> None:
    """All runs (and the table itself) must share one device/profile/protocol."""
    ref = runs[0]
    for run in runs[1:]:
        for k in _MANIFEST_KEYS:
            if run["manifest"][k] != ref["manifest"][k]:
                raise ConfigError(
                    f"mixed-profile data: {run['path']} manifest {k}="
                    f"{run['manifest'][k]!r} != {ref['path']} "
                    f"{k}={ref['manifest'][k]!r}; refusing to build")
    if ref["manifest"]["profile_hash"] != table_profile_hash:
        raise ConfigError(
            f"run profile_hash {ref['manifest']['profile_hash']} != current "
            f"DeviceProfile hash {table_profile_hash}; re-measure under the "
            "current profile before building the table")


# Issue #8 PR-G: one semantic-matrix run dir must agree with the
# others and with the current profile on these keys.
_MATRIX_CONSISTENCY_KEYS = ("device", "profile_hash", "protocol_id",
                            "precision", "correctness_threshold", "dtype",
                            "timing_scope", "sessions", "trials", "warmup")
# Issue #8 confirmation floor for promotion-grade cells.
_MATRIX_MIN_SESSIONS = 3
_MATRIX_MIN_TRIALS = 30
# Issue #8 search budget: <= 24 pre-registered candidates per
# semantic/shape (the bounded oracle set in a matrix cell).
_MATRIX_MAX_CANDIDATES = 24
# Plan params that are provenance, not lowering input.
_PARAM_META_KEYS = ("oracle_name", "selection_source", "reason",
                    "fallback", "real_compat")

_FOCUS_MANIFEST_KEYS = ("device", "profile_hash", "protocol_id",
                        "error_threshold", "shapes", "candidates",
                        "timing_scope", "sessions", "trials")
_FOCUS_CONSISTENCY_KEYS = ("device", "profile_hash", "protocol_id",
                           "error_threshold")
# Focus challengers entering the frozen table (issue #3 P0): batch M2/M5.
# P2 packet canaries (m2_packet_*) stay experimental by design.
def _focus_challenger_names(candidates):
    return [c for c in candidates
            if c in ("m2_batch", "m5_batch") or c.startswith("m2_t")]


def _load_focus(path: str, rejections) -> dict:
    """Load one run_batch_release summary.json (its own protocol)."""
    with open(path) as f:
        data = json.load(f)
    man = data.get("manifest") or {}
    missing = [k for k in _FOCUS_MANIFEST_KEYS if man.get(k) in (None, "")]
    if missing:
        raise ConfigError(
            f"focus run {path}: manifest missing required keys {missing}; "
            "refusing to build from an incomplete focus manifest")
    thr = man["error_threshold"]
    if (not isinstance(thr, (int, float)) or not math.isfinite(thr)
            or not (0 < float(thr) < 1)):
        raise ConfigError(
            f"focus run {path}: error_threshold {thr!r} must be in (0, 1)")
    shapes = data.get("shapes") or []
    if not shapes:
        raise ConfigError(f"focus run {path}: no shape results")
    sessions = int(data.get("sessions") or 0)
    expected = int(man["sessions"])
    trials = int(man["trials"])
    if not isinstance(man.get("sessions"), int) or expected < 1:
        raise ConfigError(f"focus run {path}: manifest sessions invalid")
    if not isinstance(man.get("trials"), int) or trials < 1:
        raise ConfigError(f"focus run {path}: manifest trials invalid")
    if sessions < expected:
        raise ConfigError(
            f"focus run {path}: aggregated {sessions} session(s) but the "
            f"manifest declares {expected} (a session failed or was "
            "dropped); promotion requires the complete session set -- "
            "re-run the focus protocol, refusing to build")
    return {"path": path, "manifest": man, "shapes": shapes,
            "sessions": sessions, "expected_sessions": expected,
            "trials": trials, "threshold": float(thr)}


def _focus_ms(shape, name, run, threshold, path, rejections):
    """Gated (median_ms) for one candidate with COMPLETE evidence, else None.

    Delegates to the shared completeness verdict (src/xpu/evidence.py):
    the builder's promotion gates and the cost-model's validation records
    apply exactly one rule set, so a record the validator scores is a
    record the builder could have promoted (issue #3 review fix).
    """
    ms, why = focus_candidate_verdict(
        shape, name, threshold=threshold,
        expected_sessions=run["expected_sessions"], trials=run["trials"],
        label=f"{path} {shape.get('label')}")
    if ms is None:
        rejections.append(why)
    return ms


def _focus_descriptor(name, block, batch, cand):
    """Schema-v2 attribution for a focus challenger."""
    params = dict(cand.get("params") or {})
    params.setdefault("batch", batch)
    if name.startswith("m2_"):
        flow, real, unit = ("local-fused-tile", "matrix-resident",
                            "dense-dot-fft64")
    elif name.startswith("m5_"):
        flow, real, unit = "batch-spread", "global-stage", "radix4-scalar"
    else:
        raise ConfigError(f"unmapped focus challenger: {name}")
    return {"block": block, "flow_id": flow, "flow_realization": real,
            "processing_unit": unit, "params": params}


def _focus_incumbent_descriptor(name, block, batch):
    variant = "spread" if name == "m4_spread" else "folded"
    return {"block": block, "flow_id": f"batch-{variant}",
            "flow_realization": "global-stage",
            "processing_unit": "radix2-scalar",
            "params": {"variant": variant, "batch": batch}}


def build_focus_entries(focus_runs, guard=GUARD_PCT, min_sessions=3,
                        rejections=None):
    """(N, B, direction) entries from focus-run medians vs M4 incumbent.

    Each candidate's value is the median of its per-run medians (runs are
    independent schedules); the stable incumbent is the fastest gated M4
    variant, the winner the fastest gated challenger — one of them, only
    when the challenger clears the 2% guard.
    """
    rejections = rejections if rejections is not None else []
    usable = []
    for run in focus_runs:
        if run["sessions"] < min_sessions:
            rejections.append(
                f"{run['path']}: {run['sessions']} sessions < required "
                f"{min_sessions} (issue #3 protocol floor)")
            continue
        usable.append(run)
    grouped = {}
    for run in usable:
        for shape in run["shapes"]:
            key = (int(shape["N"]), int(shape["batch"]),
                   shape["direction"])
            grouped.setdefault(key, []).append((run, shape))

    INCUMBENTS = ("incumbent_m4_folded", "m4_spread")
    entries = []
    for (n, b, d), pairs in sorted(grouped.items()):
        thr = min(run["threshold"] for run, _ in pairs)
        sessions = min(run["sessions"] for run, _ in pairs)

        def _collect(names, kind):
            """name -> complete medians across EVERY usable run.

            A name expected anywhere but incomplete in any run (absent
            session, missing trial, recorded failure, wrong value,
            non-finite) is dropped with a rejection -- partial medians
            never enter the comparison.
            """
            expected = set()
            for _, shape in pairs:
                expected.update(shape["candidates"])
            expected &= set(names)
            per_name = {}
            blocks, cands, poisoned = {}, {}, set()
            for run, shape in pairs:
                blk = int(shape.get("incumbent_block") or 256)
                for name in expected:
                    ms = _focus_ms(shape, name, run, thr, run["path"],
                                   rejections)
                    if ms is None:
                        poisoned.add(name)
                        continue
                    per_name.setdefault(name, []).append(ms)
                    cands[name] = shape["candidates"][name]
                    blocks[name] = int(
                        shape["candidates"][name].get("block") or blk)
            n_runs = len(pairs)
            for name in list(per_name):
                if name in poisoned or len(per_name[name]) != n_runs:
                    rejections.append(
                        f"N={n} B={b} {d}: {kind} {name} incomplete "
                        f"({len(per_name.get(name, []))}/{n_runs} complete "
                        "runs after gates); excluded from promotion")
                    per_name.pop(name, None)
                    blocks.pop(name, None)
                    cands.pop(name, None)
            return per_name, blocks, cands, expected

        inc_by, inc_blocks, _, inc_expected = _collect(INCUMBENTS,
                                                       "incumbent")
        ch_by, ch_blocks, ch_cands, _ = _collect(
            _focus_challenger_names(
                set().union(*(s["candidates"] for _, s in pairs))),
            "challenger")

        if inc_expected - set(inc_by):
            # An expected incumbent control is incomplete anywhere: the
            # comparison would be biased, so the shape gets no focus entry
            # (the selector keeps whatever non-focus entry it has / M0).
            rejections.append(
                f"N={n} B={b} {d}: incomplete incumbent control "
                f"{sorted(inc_expected - set(inc_by))}; refusing a focus "
                "entry (fail closed)")
            continue
        if not inc_by:
            if ch_by:
                rejections.append(
                    f"{n} B={b} {d}: challenger(s) measured but no gated "
                    "incumbent control; refusing an entry without control")
            continue
        inc_vals = {k: float(_median(v)) for k, v in inc_by.items()}
        inc_best = min(inc_vals, key=inc_vals.get)
        inc_med = inc_vals[inc_best]

        ch_vals = {k: float(_median(v)) for k, v in ch_by.items()}
        if ch_vals:
            ch_best = min(ch_vals, key=ch_vals.get)
            ch_med = ch_vals[ch_best]
            gain_pct = (inc_med - ch_med) / inc_med * 100.0
        else:
            ch_best, ch_med, gain_pct = None, None, None

        if ch_best is not None and gain_pct >= guard:
            desc = _focus_descriptor(ch_best, ch_blocks[ch_best], b,
                                     ch_cands[ch_best])
            reason = (f"focus challenger {ch_best} beats stable incumbent "
                      f"{inc_best} by {gain_pct:.1f}% (>= {guard}%) across "
                      f"{sessions} sessions")
            win_block = ch_blocks[ch_best]
        else:
            desc = _focus_incumbent_descriptor(inc_best, inc_blocks[inc_best],
                                               b)
            win_block = inc_blocks[inc_best]
            if ch_best is None:
                reason = ("no focus challenger passed the correctness/"
                          f"timing gates; stable incumbent {inc_best} kept")
            else:
                reason = (f"no focus challenger clears the {guard}% guard "
                          f"(best {ch_best} gain {gain_pct:.1f}%); stable "
                          f"incumbent {inc_best} kept")
        entry = {"N": n, "batch": b, "direction": d,
                 "incumbent_ms": inc_med, "incumbent_block":
                 inc_blocks[inc_best],
                 "challenger_ms": ch_med, "block": win_block,
                 "evidence": {"incumbent": inc_best, "sessions": sessions,
                              "protocol_id": pairs[0][0]["manifest"][
                                  "protocol_id"],
                              "focus_paths": [r["path"] for r, _ in pairs]},
                 "reason": reason, **desc}
        if ch_best is not None:
            entry["evidence"]["challenger"] = ch_best
            entry["evidence"]["gain_pct"] = round(gain_pct, 2)
            entry["evidence"]["challenger_ms"] = ch_med
        entries.append(entry)
    return entries


def build_b1_entry(rows_by_path, direction, n, run_by_path, guard=GUARD_PCT,
                   threshold=1e-4, rejections=None):
    """One (N, direction) B=1 entry: best challenger vs best-M0 incumbent."""
    d = direction
    rejections = rejections if rejections is not None else []
    candidates = []   # (ms, cand_key, block, path)
    incumbents = []
    for path, rows in rows_by_path.items():
        row = next((r for r in rows if r["n"] == n), None)
        if row is None:
            continue
        run = run_by_path[path]
        block = run["block"]
        m0_key = f"m0_{d}_ms"
        if _check_ms(row, f"m0_{d}", threshold, path, n, rejections) is not None:
            incumbents.append((row[m0_key], f"m0_{d}", block, path))
        keys = [f"m2_{d}_t64", f"m3_g_{d}_t8", f"m5_r4_{d}"]
        if d in _M1_DIRECTIONS:
            keys.insert(0, "m1_t64")
        for key in keys:
            ms = _check_ms(row, key, threshold, path, n, rejections)
            if ms is not None:
                candidates.append((ms, key, block, path))
    if not incumbents or not candidates:
        return None
    inc_ms, inc_key, inc_block, inc_path = min(incumbents)
    ch_ms, ch_key, ch_block, ch_path = min(candidates)
    gain_pct = (inc_ms - ch_ms) / inc_ms * 100.0
    if gain_pct >= guard:
        desc = _descriptor(ch_key, ch_block, run_by_path[ch_path]["extra"])
        reason = (f"challenger {ch_key}@block{ch_block} beats M0 incumbent "
                  f"{inc_key}@block{inc_block} by {gain_pct:.1f}% (>= {guard}%)")
    else:
        desc = _descriptor(inc_key, inc_block, run_by_path[inc_path]["extra"])
        reason = (f"no challenger clears the {guard}% guard "
                  f"(best {ch_key} gain {gain_pct:.1f}%); M0 incumbent kept")
    return {"N": n, "batch": 1, "direction": d,
            "incumbent_ms": inc_ms, "incumbent_block": inc_block,
            "challenger_ms": ch_ms,
            "evidence": {"incumbent": f"{inc_key}@block{inc_block}",
                         "challenger": f"{ch_key}@block{ch_block}",
                         "gain_pct": round(gain_pct, 2)},
            "reason": reason, **desc}


def build_batch_entries(run, guard=GUARD_PCT, threshold=1e-4, rejections=None):
    """Per (N, B, direction): pick the faster M4 variant (table-derived)."""
    rejections = rejections if rejections is not None else []
    extra = run["extra"]
    block = run["block"]
    entries = []
    for row in run["rows"]:
        n = row["n"]
        for b in extra.get("batchs", ()):
            if b == 1:
                # B=1 selection is owned by the B=1 confirmation runs; the
                # batch probe's B=1 rows only sanity-check the M4 path.
                continue
            for d in ("dit", "dif"):
                ms = {}
                for v in ("spread", "folded"):
                    key = f"m4_{v}_b{b}_{d}"
                    ms[v] = _check_ms(row, key, threshold, run["path"], n,
                                      rejections)
                if not any(ms.values()):
                    continue
                if ms["spread"] and ms["folded"]:
                    gain = (ms["folded"] - ms["spread"]) / ms["folded"] * 100.0
                    if gain >= guard:
                        variant, why = "spread", (
                            f"spread beats folded by {gain:.1f}% (>= {guard}%)")
                    else:
                        variant, why = "folded", (
                            f"folded kept (spread gain {gain:.1f}% < {guard}%)")
                else:
                    variant = "spread" if ms["spread"] else "folded"
                    why = f"only {variant} measured"
                desc = _descriptor(f"m4_{variant}_b{b}_{d}", block, extra)
                entries.append({
                    "N": n, "batch": b, "direction": d,
                    "incumbent_ms": (ms["folded"] if variant == "spread"
                                     else ms["spread"]),
                    "incumbent_block": block,
                    "challenger_ms": ms[variant],
                    "evidence": {"variant_ms": {k: v for k, v in ms.items()
                                                if v}},
                    "reason": why, **desc,
                })
    return entries


def _load_matrix(path: str, rejections) -> dict:
    """Load one semantic-matrix run directory (issue #8 PR-G).

    Fail-closed artifact gates: the directory must pass the strict
    output gate (embedded manifest + raw sessions + result_sha256,
    stamped by the calibration stage), its protocol must be the matrix
    protocol this builder trusts, and it must meet the issue's
    confirmation floor (>= 3 sessions x >= 30 trials). Untrusted
    artifacts raise ConfigError; per-cell business gates live in
    :func:`build_matrix_entries`.
    """
    problems = validate_benchmark_output(path, strict=True)
    if problems:
        raise ConfigError(
            f"matrix run {path} failed the strict output gate: "
            + "; ".join(problems[:6])
            + (" ..." if len(problems) > 6 else "")
            + "; refusing to build")
    summary_path = os.path.join(path, "summary.json")
    try:
        with open(summary_path) as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        raise ConfigError(f"matrix run {path}: unreadable summary: {e}")
    man = payload.get("manifest")
    if not isinstance(man, dict):
        raise ConfigError(f"matrix run {path}: no embedded manifest")
    if man.get("protocol_id") != sm.PROTOCOL_ID:
        raise ConfigError(
            f"matrix run {path}: protocol_id {man.get('protocol_id')!r} != "
            f"trusted matrix protocol {sm.PROTOCOL_ID!r}")
    sessions = man.get("sessions")
    trials = man.get("trials")
    if not isinstance(sessions, int) or sessions < _MATRIX_MIN_SESSIONS:
        raise ConfigError(
            f"matrix run {path}: sessions={sessions!r} below the issue "
            f"floor {_MATRIX_MIN_SESSIONS}; refusing")
    if not isinstance(trials, int) or trials < _MATRIX_MIN_TRIALS:
        raise ConfigError(
            f"matrix run {path}: trials={trials!r} below the issue floor "
            f"{_MATRIX_MIN_TRIALS}; refusing")
    rows = []
    raw_dir = os.path.join(path)
    for name in sorted(os.listdir(raw_dir)):
        if not (name.startswith("raw_session") and name.endswith(".jsonl")):
            continue
        with open(os.path.join(raw_dir, name)) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    if not rows:
        raise ConfigError(f"matrix run {path}: no raw session rows")
    return {"path": path, "manifest": man,
            "cases": payload.get("cases") or [], "rows": rows}


def _check_matrix_consistency(matrix_runs, profile_hash) -> None:
    """All matrix runs share one protocol/profile with the build host."""
    ref = matrix_runs[0]["manifest"]
    for run in matrix_runs[1:]:
        for k in _MATRIX_CONSISTENCY_KEYS:
            if run["manifest"].get(k) != ref.get(k):
                raise ConfigError(
                    f"mixed matrix protocols: {run['path']} {k}="
                    f"{run['manifest'].get(k)!r} != {matrix_runs[0]['path']} "
                    f"{k}={ref.get(k)!r}; refusing to build")
    if ref.get("profile_hash") != profile_hash:
        raise ConfigError(
            f"matrix profile_hash {ref.get('profile_hash')!r} != current "
            f"DeviceProfile hash {profile_hash}; re-measure under the "
            "current profile before building")


def build_matrix_entries(matrix_runs, guard=GUARD_PCT, threshold=1e-4,
                         rejections=None) -> list:
    """Exact-semantic entries from confirmed matrix cells (issue #8).

    Per-cell promotion gates (issue #8 PR-G), all measured-source only:

    * the cell's oracle winner (bounded candidate set, <= 24 entries)
      beats the matrix incumbent by at least the semantic's own guard
      (default 2%);
    * the winner has correctness-clean steady timing in EVERY declared
      session of a >= 3 x >= 30 confirmation run, with err <= threshold;
    * its plan geometry (block/params/attribution) is identical across
      those samples, passes the design-space/lowering contract, and the
      flow belongs to the *semantic's own* family (no cross-semantic
      inheritance; PR-H specialized flows join only theirs).

    A cell that fails any gate is recorded in ``rejections`` and keeps
    whatever legacy/abstain path it already had -- the table is never
    touched by a rejected cell.
    """
    if rejections is None:
        rejections = []
    entries = []
    for run in matrix_runs:
        man = run["manifest"]
        path = run["path"]
        need_sessions = int(man.get("sessions") or 0)
        by_label: dict[str, list] = {}
        for row in run["rows"]:
            by_label.setdefault(row.get("label"), []).append(row)
        for case in run["cases"]:
            n, b = case.get("N"), case.get("batch")
            sem, label = case.get("semantic"), case.get("label")
            tag = f"matrix {path} {label}"

            def rej(msg, tag=tag):
                rejections.append(f"{tag}: {msg}")

            if not _finite_positive(case.get("incumbent_ms"))                     or not _finite_positive(case.get("oracle_ms")):
                rej("non-finite incumbent/oracle median; no entry")
                continue
            try:
                sp = semantic_mod.spec(sem)
            except KeyError:
                rej(f"unknown semantic {sem!r}; no entry")
                continue
            winner = case.get("oracle_candidate")
            if not winner:
                rej("no measured oracle winner; no entry")
                continue

            rows = by_label.get(label) or []
            oracle_names = {
                r.get("candidate") for r in rows
                if str(r.get("candidate") or "").startswith("oracle:")}
            if not oracle_names:
                rej("no measured oracle candidates in the raw rows; "
                    "no entry")
                continue
            if len(oracle_names) > _MATRIX_MAX_CANDIDATES:
                rej(f"{len(oracle_names)} candidates exceed the issue's "
                    f"{_MATRIX_MAX_CANDIDATES}-candidate budget; no entry")
                continue

            inc = float(case["incumbent_ms"])
            orc = float(case["oracle_ms"])
            gain_pct = (inc - orc) / inc * 100.0
            if gain_pct < guard:
                rej(f"guard not cleared: best oracle {winner} only "
                    f"{gain_pct:.2f}% vs incumbent (need {guard}%) "
                    f"-> legacy/abstain path stays")
                continue

            w_steady = [r for r in rows
                        if r.get("candidate") == winner
                        and r.get("tier") == sm.TIER_STEADY]
            ok_sessions = {r.get("session") for r in w_steady
                           if r.get("status") == "ok"}
            if need_sessions < _MATRIX_MIN_SESSIONS                     or len(ok_sessions) < need_sessions:
                rej(f"winner {winner} has timing in {len(ok_sessions)}/"
                    f"{need_sessions} sessions; no entry")
                continue
            bad = [r for r in rows
                   if r.get("candidate") == winner
                   and r.get("status") != "ok"]
            if bad:
                rej(f"winner {winner} has {len(bad)} non-ok sample(s) "
                    f"({bad[0].get('reason')}); no entry")
                continue
            errs = [r.get("err") for r in w_steady
                    if r.get("status") == "ok"
                    and isinstance(r.get("err"), (int, float))]
            if errs and max(errs) > threshold:
                rej(f"winner {winner} err {max(errs):.3e} > {threshold:g}; "
                    f"no entry")
                continue

            blocks = {r.get("block") for r in w_steady}
            if None in blocks or len(blocks) != 1:
                rej(f"winner {winner} block geometry missing/inconsistent "
                    f"across sessions {sorted(map(str, blocks))}; no entry")
                continue
            block = int(next(iter(blocks)))

            def _lower(p):
                return {k: v for k, v in (p or {}).items()
                        if k not in _PARAM_META_KEYS}
            param_variants = {
                json.dumps(_lower(r.get("params")), sort_keys=True)
                for r in w_steady}
            if len(param_variants) != 1:
                rej(f"winner {winner} plan params differ across samples "
                    f"({len(param_variants)} variants); no entry")
                continue
            params = json.loads(next(iter(param_variants)))
            params["batch"] = b

            flow_id = w_steady[0].get("flow_id")
            unit = w_steady[0].get("processing_unit")
            real = w_steady[0].get("flow_realization")
            if not flow_id or not unit or not real:
                rej(f"winner {winner} rows lack flow attribution "
                    f"(flow_id={flow_id!r}, unit={unit!r}, real={real!r}); "
                    f"no entry")
                continue
            if not semantic_mod.family_allows(sp.transform_type, flow_id):
                rej(f"flow {flow_id!r} not in the {sp.transform_type} "
                    f"family (per-semantic legality, issue #8); no entry")
                continue
            direction = case.get("direction") or sm.DIRECTION
            if direction != sm.DIRECTION:
                rej(f"matrix locks direction {sm.DIRECTION!r}, got "
                    f"{direction!r}; no entry")
                continue

            entries.append({
                "N": n, "batch": b, "direction": direction,
                "transform_type": sp.transform_type,
                "transform_direction": sp.transform_direction,
                "normalization": sp.normalization,
                "incumbent_ms": inc, "incumbent_block": block,
                "challenger_ms": orc, "block": block,
                "flow_id": flow_id, "flow_realization": real,
                "processing_unit": unit, "params": params,
                "evidence": {
                    "incumbent": "matrix-incumbent",
                    "challenger": winner,
                    "gain_pct": round(gain_pct, 3),
                    "challenger_ms": orc,
                    "sessions": len(ok_sessions),
                    "protocol_id": man.get("protocol_id"),
                    "semantic": sem,
                    "source": "semantic-matrix confirmation",
                },
                "reason": (
                    f"independent matrix confirmation ({sem}): {winner} "
                    f"beats incumbent by {gain_pct:.1f}% (>= {guard}%) "
                    f"over {len(ok_sessions)} sessions; exact semantic "
                    f"key {sp.key}"),
            })
    return entries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="B=1 confirmation JSONs")
    ap.add_argument("--batch", default=None, help="batch confirmation JSON")
    ap.add_argument("--focus", nargs="*", default=[],
                    help="run_batch_release summary.json files (issue #3)")
    ap.add_argument("--min-focus-sessions", type=int, default=3,
                    help="focus runs below this session count are refused")
    ap.add_argument("--matrix-run", nargs="*", default=[],
                    help="semantic-matrix run directories (issue #8): "
                         "summary.json + raw sessions; entries carry the "
                         "exact semantic key and only land when the cell "
                         "clears its own guard")
    ap.add_argument("--out", default="config/selector_table.json")
    ap.add_argument("--guard", type=float, default=GUARD_PCT)
    ap.add_argument("--verify-live-env", action="store_true",
                    help="device-host check: probe the live environment "
                         "and require it to equal the measurement env_hash")
    args = ap.parse_args()

    profile = DeviceProfile.load()
    profile_hash = profile.hash()
    rejections = []

    def _rel(path):
        try:
            return os.path.relpath(path)
        except ValueError:  # different drive (Windows); keep absolute
            return path

    runs = [_load_run(p, rejections) for p in args.runs]
    all_runs = list(runs)
    if args.batch:
        all_runs = all_runs + [_load_run(args.batch, rejections)]
    _check_manifest_consistency(all_runs, profile_hash)
    threshold = float(runs[0]["manifest"]["correctness_threshold"])

    focus_runs = [_load_focus(p, rejections) for p in args.focus]
    if focus_runs:
        ref = focus_runs[0]["manifest"]
        for run in focus_runs[1:]:
            for k in _FOCUS_CONSISTENCY_KEYS:
                if run["manifest"][k] != ref[k]:
                    raise ConfigError(
                        f"mixed focus protocols: {run['path']} {k}="
                        f"{run['manifest'][k]!r} != {focus_runs[0]['path']} "
                        f"{k}={ref[k]!r}; refusing to build")
        if ref["profile_hash"] != profile_hash:
            raise ConfigError(
                f"focus profile_hash {ref['profile_hash']} != current "
                f"DeviceProfile hash {profile_hash}; re-measure under the "
                "current profile before building")

    matrix_runs = [_load_matrix(p, rejections) for p in args.matrix_run]
    if matrix_runs:
        _check_matrix_consistency(matrix_runs, profile_hash)
        matrix_threshold = float(
            matrix_runs[0]["manifest"]["correctness_threshold"])
        if matrix_threshold != threshold:
            raise ConfigError(
                f"matrix correctness_threshold {matrix_threshold!r} != "
                f"confirm runs' {threshold!r}; refusing to mix bars")

    # Environment binding (issue #3 P0): taken from the MEASUREMENT
    # manifests, not from this build host -- the same clean artifact must
    # rebuild deterministically in plain CI. The focus campaign is the
    # binding source: every focus run must carry one identical non-empty
    # env_hash. Confirm runs may predate env binding and are covered by
    # the focus campaign; when they do carry env_hash it must agree.
    env_hashes = []  # (path, env_hash)
    for run in matrix_runs:
        h = (run["manifest"].get("env_hash") or "").strip()
        if not h:
            raise ConfigError(
                f"matrix run {run['path']} manifest has no env_hash; "
                "matrix runs come from the env-bound harness and are "
                "required to bind, refusing to build")
        env_hashes.append((run["path"], h))
    for run in focus_runs:
        h = (run["manifest"].get("env_hash") or "").strip()
        if not h:
            raise ConfigError(
                f"focus run {run['path']} manifest has no env_hash; the "
                "focus campaign is the binding source for the table "
                "(re-measure with the env-bound harness), refusing to build")
        env_hashes.append((run["path"], h))
    for run in all_runs:
        h = (run["manifest"].get("env_hash") or "").strip()
        if h:
            env_hashes.append((run["path"], h))
    if not env_hashes:
        raise ConfigError(
            "no input manifest carries an env_hash; refusing to build an "
            "unbound table (pass focus runs from the env-bound campaign)")
    distinct = sorted({h for _, h in env_hashes})
    if len(distinct) > 1:
        raise ConfigError(
            "input manifests disagree on env_hash "
            f"({', '.join(distinct[:4])}); refusing to build -- the inputs "
            "must come from one measurement environment")
    table_env_hash = distinct[0]
    table_env = None
    for run in list(all_runs) + list(focus_runs):
        man = run["manifest"]
        if man.get("env_hash") != table_env_hash:
            continue
        d = man.get("env")
        if isinstance(d, dict) and fingerprint_env(d) == table_env_hash:
            table_env = d
            break
        if isinstance(d, dict):
            raise ConfigError(
                f"run {run['path']}: manifest env dict hashes to "
                f"{fingerprint_env(d)}, not its declared env_hash "
                f"{table_env_hash}; refusing to build")
    if args.verify_live_env:
        from src.xpu.envcheck import collect as collect_env
        live = fingerprint_env(collect_env())
        if live != table_env_hash:
            raise ConfigError(
                f"--verify-live-env: live environment {live} != measurement "
                f"environment {table_env_hash}; re-measure before building")

    rows_by_path = {r["path"]: r["rows"] for r in runs}
    run_by_path = {r["path"]: r for r in runs}
    sizes = set()
    for r in runs:
        sizes.update(row["n"] for row in r["rows"])
    sources = [_rel(r["path"]) for r in runs]

    entries = []
    for n in sorted(sizes):
        for d in ("dit", "dif"):
            e = build_b1_entry(rows_by_path, d, n, run_by_path,
                               guard=args.guard, threshold=threshold,
                               rejections=rejections)
            if e:
                entries.append(e)
    if args.batch:
        batch_run = next(r for r in all_runs if r["path"] == args.batch)
        entries.extend(build_batch_entries(batch_run, guard=args.guard,
                                           threshold=threshold,
                                           rejections=rejections))
        sources.append(_rel(args.batch))
    if focus_runs:
        entries.extend(build_focus_entries(
            focus_runs, guard=args.guard,
            min_sessions=args.min_focus_sessions, rejections=rejections))
        sources.extend(_rel(r["path"]) for r in focus_runs)
    if matrix_runs:
        entries.extend(build_matrix_entries(
            matrix_runs, guard=args.guard, threshold=threshold,
            rejections=rejections))
        sources.extend(_rel(r["path"]) for r in matrix_runs)

    # Issue #8 PR-G: the duplicate key is the FULL selector key,
    # including the semantic triple. Legacy entries (no triple fields)
    # key as (None, None, None) and may coexist with exact-semantic
    # entries at the same shape: the semantic tier answers its own
    # triple, the legacy entry stays as the deployed old path for
    # every other semantic until it clears its own guard.
    seen = {}
    for e in entries:
        key = (e["N"], e["batch"], e["direction"],
               e.get("transform_type"), e.get("transform_direction"),
               e.get("normalization"))
        if key in seen:
            raise ConfigError(
                f"duplicate table entry for {key} ({seen[key]} and a "
                "second run claiming the same full key); refusing to "
                "build")
        seen[key] = e.get("reason", "?")

    # Entries must satisfy the shared design space / lowering contract
    # BEFORE they are written: a frozen item goes through the same
    # contract as any candidate (issue #3 review fix).
    illegal = []
    for e in entries:
        for msg in entry_contract_violations(e):
            key = (e["N"], e["batch"], e["direction"])
            illegal.append(f"entry {key}: {msg}")
    if illegal:
        raise ConfigError(
            f"built {len(entries)} entries but {len(illegal)} violate the "
            f"design-space/lowering contract: " + "; ".join(illegal[:8])
            + (" ..." if len(illegal) > 8 else "")
            + "; refusing to write the table")

    table = {
        "schema_version": 2,
        "device": profile.device,
        "profile_hash": profile_hash,
        "env_hash": table_env_hash,
        "env": table_env,
        "precision": "fp32",
        "guard_pct": args.guard,
        "selection_semantics": (
            "search-stage median improvement (aggregated medians within one "
            "run schedule), not paired per-trial medians"),
        "sources": sources,
        "rejections": rejections,
        "entries": entries,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(table, f, indent=2)
        f.write("\n")
    n_b1 = sum(1 for e in entries if e["batch"] == 1)
    n_m0 = sum(1 for e in entries if e["batch"] == 1
               and e["flow_id"] == "baseline-global-stage"
               and e["processing_unit"] == "radix2-scalar")
    print(f"wrote {args.out}: {len(entries)} entries "
          f"({n_b1} at B=1, {n_m0} M0-fallback) guard={args.guard}% "
          f"rejections={len(rejections)}")
    for r in rejections:
        print(f"  rejected: {r}", file=sys.stderr)


if __name__ == "__main__":
    main()
