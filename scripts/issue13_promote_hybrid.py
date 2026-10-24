#!/usr/bin/env python3
"""Promote WP1-stable issue #13 4-step winners into the composition table.

Reads ``results/p800/issue13_release/wp1_promotion.json`` (the 3+1
session stability-gate artifact, issue #13 WP1) and, for every
promoted exact key, stamps a composition entry bound to the MEASURED
implementation (``src/xpu/hybrid4`` — the holdout winner kernels,
deployed under the ``a1p1_materialized_subtransform`` capability):

- variant + executed runner recorded on the entry (plan/execute
  dispatches exactly that runner; ``composition_entry_violations``
  enforces the pairing at load time);
- per-entry semantic record (c2c / inverse / backward — the exact
  semantics the holdout measured against numpy ifft);
- 2% incumbent guard re-computed here from the stamped medians
  (all 4 sessions, the same basis the WP1 verdict used);
- full fail-closed re-verification: artifact sha, live env/profile,
  verdict.pass per key, entry_shape_uniform, correctness/err_max from
  the raw session rows, CI lows > 1.02 and spreads <= 10% re-checked
  from the stamped points (tamper-evidence), table_violations before
  atomic write.

Also emits selector_diff v3: the butterfly selector_table is
untouched (sha before == after, wiring=none); only composition
entries are added, with full provenance and the WP1 cells that did
not clear the stability gates recorded as still-uncovered.

Usage:
    python3 scripts/issue13_promote_hybrid.py [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.xpu.composition import (  # noqa: E402
    KIND, TABLE_PATH, entry_semantic, guard_pct_of, table_violations)
from src.xpu.config import DeviceProfile, git_dirty  # noqa: E402
from src.xpu.envcheck import current_env_hash  # noqa: E402
from src.xpu.hybrid4 import VARIANT_EXECUTED, factors  # noqa: E402
from src.xpu.manifest_contract import rows_sha256  # noqa: E402

CONTRACT_ID = "a1p1_materialized_subtransform"
FLOW_REALIZATION = "decomposition-composition"
PROCESSING_UNIT = "vendor-subfft-host-composition"
GUARD_PCT = 2.0
ERR_MAX = 1e-4
GATE_CI_LOW = 1.02
SPREAD_REL_MAX = 0.10
MIN_SESSIONS = 3
CONFIRM_SESSION = "4"
DEFAULT_EVIDENCE = ROOT / "results/p800/issue13_release/wp1_promotion.json"
DEFAULT_RAW = ROOT / "results/p800/issue13_release/wp1_sessions_raw.json"
DEFAULT_DIFF = ROOT / "results/p800/issue13_release/selector_diff.json"
SELECTOR_TABLE = ROOT / "config/selector_table.json"

# candidate path label -> (variant, flow_id, display)
CANDIDATE_IMPL = {
    "hybrid_vendor_p": ("4step-transpose", "hybrid-A1P1-4step",
                        "4-step transpose twiddle (measured "
                        "hybrid_vendor_p)"),
    "hybrid_fused_p": ("4step-fused", "hybrid-A1P1-4step-fused",
                       "4-step fused handoff twiddle (measured "
                       "hybrid_fused_p)"),
}


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT,
                                   text=True).strip()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _die(msg: str) -> None:
    print(f"refuse promotion: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _pctl(values: list[float], q: float) -> float:
    s = sorted(values)
    idx = min(len(s) - 1, int(q * len(s)))
    return float(s[idx])


def _spread(points: list[float]) -> float:
    if not points:
        return float("inf")
    lo, hi = min(points), max(points)
    return (hi - lo) / lo if lo > 0 else float("inf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", default=str(DEFAULT_EVIDENCE))
    ap.add_argument("--raw", default=str(DEFAULT_RAW),
                    help="wp1_sessions_raw.json (raw trial rows)")
    ap.add_argument("--out", default=str(TABLE_PATH))
    ap.add_argument("--diff-out", default=str(DEFAULT_DIFF))
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ev_path = Path(args.evidence)
    raw_path = Path(args.raw)
    if not ev_path.is_file():
        _die(f"evidence not found: {ev_path}")
    if not raw_path.is_file():
        _die(f"raw sessions not found: {raw_path}")
    ev = json.loads(ev_path.read_text())
    raw = json.loads(raw_path.read_text())
    if ev.get("schema") != "issue13-wp1-promotion-v1":
        _die(f"unexpected evidence schema {ev.get('schema')!r}")

    dirty = git_dirty()
    if dirty and not args.allow_dirty:
        _die("working tree dirty (commit first, or --allow-dirty)")

    # live bindings must match what WP1 measured against (fail closed)
    live_env = current_env_hash()
    if live_env is None:
        _die("live environment fingerprint unknown; cannot verify")
    env = ev.get("environment") or {}
    if env.get("env_hash_live") != live_env:
        _die(f"evidence live env {env.get('env_hash_live')} != current "
             f"{live_env}")
    profile = DeviceProfile.load()
    profile_hash = profile.hash()
    if env.get("profile_hash") != profile_hash:
        _die(f"evidence profile {env.get('profile_hash')} != live "
             f"{profile_hash}")
    ev_sha = _sha256_file(ev_path)

    # index raw rows: label -> session -> path -> row
    idx: dict = {}
    for r in raw.get("raw", []):
        key = f"N{r['n']}B{r['batch']}:{r['direction']}"
        idx.setdefault(key, {}).setdefault(str(r["session"]), {})[
            r["path"]] = r

    promoted = (ev.get("promotion_summary") or {}).get("promoted_keys") or []
    if not promoted:
        _die("evidence has no promoted_keys (nothing to stamp)")

    head = _git("rev-parse", "HEAD")
    new_entries = []
    for pk in promoted:
        key = pk["key"]
        n, batch = int(key["N"]), int(key["batch"])
        label = pk["label"]
        cand = pk.get("candidate")
        verdict = pk.get("verdict") or {}
        if cand is None or cand not in CANDIDATE_IMPL:
            _die(f"{label}: no/frozen-unrecognized candidate {cand!r}")
        if not verdict.get("pass"):
            _die(f"{label}: verdict.pass is not true for {cand}")
        variant, flow_id, impl_desc = CANDIDATE_IMPL[cand]
        executed = VARIANT_EXECUTED[variant]

        cell = next((c for c in ev["cells"] if c["label"] == label), None)
        if cell is None:
            _die(f"{label}: cell missing from evidence cells")
        if not cell.get("entry_shape_uniform"):
            _die(f"{label}: entry shapes not uniform across sessions")
        sessions = cell.get("sessions") or {}
        for s in sorted(sessions, key=int):
            flags = (sessions[s].get("flags") or {})
            if not (flags.get("all_paths_present")
                    and flags.get("no_fail_closed")):
                _die(f"{label} s{s}: session flags not clean "
                     f"({flags})")
        if str(CONFIRM_SESSION) not in sessions:
            _die(f"{label}: no confirmation session")
        roles = {sessions[s].get("role") for s in sessions}
        if "confirmation" not in roles or \
                len(sessions) < MIN_SESSIONS:
            _die(f"{label}: session roles/count wrong "
                 f"({roles}, {len(sessions)})")

        # tamper-evidence: re-check the WP1 gates from stamped points
        for side in ("vendor", "incumbent"):
            lows = (verdict.get("ci_low_by_session") or {}).get(side) or []
            if len(lows) < MIN_SESSIONS or min(lows) <= GATE_CI_LOW:
                _die(f"{label}/{cand}: {side} CI lows {lows} do not "
                     f"clear {GATE_CI_LOW} on all sessions")
            pts = (verdict.get("points_by_session") or {}).get(side) or []
            if _spread(pts) > SPREAD_REL_MAX:
                _die(f"{label}/{cand}: {side} spread "
                     f"{_spread(pts):.3f} > {SPREAD_REL_MAX}")

        # per-session medians (guard basis: all sessions — the same
        # basis the WP1 verdict used)
        cand_p50, inc_p50, vend_p50, mads = [], [], [], []
        rel_errs = []
        for s in sorted(sessions, key=int):
            paths = (sessions[s].get("paths") or {})
            for store, name in ((cand_p50, cand), (inc_p50,
                                                   "incumbent_entry"),
                                (vend_p50, "vendor")):
                if name not in paths:
                    _die(f"{label} s{s}: missing path {name}")
                store.append(float(paths[name]["warm_p50_ms"]))
            mads.append(float(paths[cand]["warm_mad_ms"]))
            rel_errs.append(float(paths[cand]["rel_err"]))
            if paths[cand].get("correct") is not True:
                _die(f"{label} s{s}: candidate not correct")
        err_max = max(rel_errs)
        if err_max > ERR_MAX:
            _die(f"{label}: err_max {err_max} > {ERR_MAX}")

        # pooled trials across all sessions for distribution stats
        pooled = []
        for s in sorted(sessions, key=int):
            row = (idx.get(label, {}).get(s, {}) or {}).get(cand)
            if row is None:
                _die(f"{label} s{s}: raw candidate row missing")
            pooled.extend(float(t) for t in row["trials"])
        if len(pooled) < 100:
            _die(f"{label}: only {len(pooled)} pooled trials")

        cand_ms = round(float(statistics.median(cand_p50)), 6)
        inc_ms = round(float(statistics.median(inc_p50)), 6)
        vendor_ms = round(float(statistics.median(vend_p50)), 6)
        mad_ms = round(float(statistics.median(mads)), 6)
        guard = guard_pct_of(cand_ms, inc_ms)
        if guard < GUARD_PCT:
            _die(f"{label}: guard {guard:.2f}% < {GUARD_PCT}% "
                 f"(cand {cand_ms} vs incumbent {inc_ms})")

        # incumbent attribution from the measured binding (what the
        # profile-bound incumbent actually executed in every session)
        s_first = sorted(sessions, key=int)[0]
        inc_bind = (sessions[s_first]["paths"]["incumbent_entry"]
                    .get("binding") or {})
        incumbent_flow = (f"{inc_bind.get('executed')} "
                          f"({inc_bind.get('flow_id')}; "
                          f"tier={inc_bind.get('entry_tier')})")

        t_sub, r_sub = factors(n)   # vendor sub-block lengths (stage1
                                    # r-FFT, stage2 t-FFT)
        entry = {
            "N": n, "batch": batch, "direction": "dit",
            "precision": "fp32",
            "tile": int(r_sub),
            "core": "vendor",
            "flow_id": flow_id,
            "flow_realization": FLOW_REALIZATION,
            "processing_unit": PROCESSING_UNIT,
            "candidate_ms": cand_ms,
            "incumbent_ms": inc_ms,
            "incumbent_flow": incumbent_flow,
            "guard_pct": guard,
            "guard_pass": True,
            "reason": (f"issue13 WP1 exact-key promotion: {impl_desc} "
                       f"{cand_ms:.4f}ms vs profile-bound incumbent "
                       f"{inc_ms:.4f}ms (+{guard:.1f}%; incumbent="
                       f"{incumbent_flow}); vendor {vendor_ms:.4f}ms "
                       f"({(vendor_ms - cand_ms) / vendor_ms * 100:.1f}% "
                       f"vendor delta, not a gate); bootstrap CI lows "
                       f"> {GATE_CI_LOW} both sides, spread <= "
                       f"{SPREAD_REL_MAX * 100:.0f}% over 4 sessions "
                       "(3 protocol + confirmation)"),
            "contract_id": CONTRACT_ID,
            "variant": variant,
            "executed": executed,
            "transform_type": key.get("transform_type", "c2c"),
            "transform_direction": key["transform_direction"],
            "normalization": key.get("normalization", "backward"),
            "entry_contract": (
                "rank-1 (N,) plan_fft->execute_fft_plan"
                if batch == 1 else
                f"rank-2 ({batch}, N) plan_fft->execute_fft_plan"),
            "evidence": {
                "artifact": str(ev_path.relative_to(ROOT)),
                "artifact_sha256": ev_sha,
                "commit": head,
                "sessions": len(sessions),
                "trials": len(pooled),
                "confirmation_session": True,
                "err_max": err_max,
                "p50_ms": cand_ms,
                "mad_ms": mad_ms,
                "p95_ms": round(_pctl(pooled, 0.95), 6),
            },
        }
        new_entries.append(entry)
        print(f"stamped {label}: {executed} guard +{guard:.2f}% "
              f"(cand {cand_ms:.4f}ms vs inc {inc_ms:.4f}ms) "
              f"semantic=({entry['transform_type']},"
              f"{entry['transform_direction']},"
              f"{entry['normalization']})")

    # --- merge + validate + atomic write --------------------------------
    out_path = Path(args.out)
    before_table_sha = (_sha256_file(out_path)
                        if out_path.is_file() else None)
    if out_path.is_file():
        table = json.loads(out_path.read_text())
        print(f"merging into existing {out_path}")
    else:
        table = {
            "schema_version": 1,
            "kind": KIND,
            "profile_hash": profile_hash,
            "env_hash": live_env,
            "precision": "fp32",
            "guard_pct": GUARD_PCT,
            "contract_id": CONTRACT_ID,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source_hash": head,
            "dirty": bool(dirty),
            "entries": [],
        }
    keep = []
    replaced = []
    for e in table.get("entries", []):
        dup = any((e.get("N") == ne["N"] and e.get("batch") == ne["batch"]
                   and e.get("direction") == ne["direction"]
                   and entry_semantic(e) == entry_semantic(ne))
                  for ne in new_entries)
        (replaced if dup else keep).append(e)
    entries = keep + new_entries
    table["entries"] = entries
    table["profile_hash"] = profile_hash
    table["env_hash"] = live_env
    table["source_hash"] = head
    table["dirty"] = bool(dirty)
    table["result_sha256"] = rows_sha256({"rows": entries})

    violations = table_violations(table)
    if violations:
        _die("resulting table violates contract: "
             + "; ".join(violations[:8]))

    if args.dry_run:
        print("dry-run: table valid, nothing written")
        for ne in new_entries:
            print(json.dumps(ne, sort_keys=True))
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(out_path.parent),
                               prefix=".composition_table.")
    with os.fdopen(fd, "w") as f:
        json.dump(table, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, out_path)
    after_sha = _sha256_file(out_path)
    print(f"promoted {len(new_entries)} entries -> {out_path}")
    print(f"result_sha256 {table['result_sha256']}")

    # --- selector_diff v3 ----------------------------------------------
    sel_sha = _sha256_file(SELECTOR_TABLE)
    diff = {
        "schema": "issue13-selector-diff-v3",
        "issue": "kunlunxin-fft#13",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": head,
        "selector_table": {
            "path": "config/selector_table.json",
            "sha256_before": sel_sha,
            "sha256_after": sel_sha,   # never written by this script
            "changed": False,
            "wiring": "none",
            "note": ("butterfly table untouched this round: composition "
                     "wins never enter selector_table.json (separately "
                     "named, separately validated)"),
        },
        "composition_table": {
            "path": str(out_path.relative_to(ROOT)),
            "sha256_before": before_table_sha,
            "sha256_after": after_sha,
            "result_sha256": table["result_sha256"],
            "added_entries": [
                {k: e[k] for k in ("N", "batch", "direction", "variant",
                                   "executed", "flow_id",
                                   "transform_direction",
                                   "candidate_ms", "incumbent_ms",
                                   "guard_pct") if k in e}
                for e in new_entries],
            "replaced_entries": [
                {k: e.get(k) for k in ("N", "batch", "direction",
                                       "flow_id", "executed")}
                for e in replaced],
        },
        "uncovered_points": {
            "policy": ("exact keys only; every query without a matching "
                       "entry (shape+semantic) falls through to the "
                       "butterfly table / M0 — composition_select fails "
                       "closed, no extrapolation"),
            "wp1_not_promoted": [
                {"label": c["label"],
                 "frozen_candidates": c["frozen_candidates"],
                 "reasons": {
                     ch: v["reasons"]
                     for ch, v in (c.get("verdicts") or {}).items()
                     if v.get("frozen_candidate") and not v["pass"]}}
                for c in ev["cells"]
                if not (c.get("promotion") or {}).get("selected")],
        },
        "dispatch": {
            "modules": ["src/xpu/composition.py", "src/xpu/entry.py",
                        "src/xpu/hybrid4.py"],
            "semantics": ("per-entry semantic matching on "
                          "(N, batch, direction, transform_type, "
                          "transform_direction, normalization); legacy "
                          "entries default c2c/forward/backward"),
            "plan_execute_consistency": (
                "plan.executed names the runner; entry.execute "
                "dispatches exactly it; composition_entry_violations "
                "enforces variant<->executed pairing at load"),
        },
    }
    diff_path = Path(args.diff_out)
    diff_path.parent.mkdir(parents=True, exist_ok=True)
    diff_path.write_text(json.dumps(diff, indent=2, sort_keys=True) + "\n")
    print(f"selector_diff v3 -> {diff_path}")


if __name__ == "__main__":
    main()
