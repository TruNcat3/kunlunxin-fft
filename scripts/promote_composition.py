#!/usr/bin/env python3
"""Promote an exact-shape composition cell (issue #6 PR-E / WP3).

Reads confirmation evidence (``scripts/run_hybrid_canary.py`` output)
and, only when every WP3 rule holds, writes
``config/composition_table.json`` atomically:

- exact-shape key (N, batch, direction) — never a range;
- 2% incumbent guard re-passed here from the pooled medians
  (incumbent = A0P0 production cell in the same evidence);
- >= 3 sessions, >= 1 independent confirmation session, distinct
  PIDs, same commit, correctness err_max <= 1e-4;
- live env/profile bindings match the evidence;
- negative points in the evidence are left untouched — a cell that
  fails any rule refuses promotion (they stay reported as-is).

Usage:
    CUDA_VISIBLE_DEVICES=6 python3 scripts/promote_composition.py \
        --evidence results/p800/composition_confirm.json \
        --shape 2048x16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.xpu.composition import (  # noqa: E402
    KIND, TABLE_PATH, guard_pct_of, table_violations)
from src.xpu.config import DeviceProfile  # noqa: E402
from src.xpu.envcheck import current_env_hash  # noqa: E402
from src.xpu.manifest_contract import rows_sha256  # noqa: E402

CONTRACT_ID = "a1p1_materialized_subtransform"
FLOW_ID = "hybrid-A1P1"
FLOW_REALIZATION = "decomposition-composition"
PROCESSING_UNIT = "vendor-subfft-host-composition"
GUARD_PCT = 2.0
ERR_MAX = 1e-4
MIN_SESSIONS = 3


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT,
                                   text=True).strip()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _die(msg: str) -> None:
    print(f"refuse promotion: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _find_cell(evidence: dict, n: int, batch: int) -> dict:
    cells = evidence.get("cells") or evidence.get("shapes") or []
    for cell in cells:
        if cell.get("N") == n and cell.get("batch") == batch:
            return cell
    _die(f"evidence has no cell N={n} batch={batch} "
         f"(cells: {[(c.get('N'), c.get('batch')) for c in cells]})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", required=True,
                    help="run_hybrid_canary.json confirmation artifact")
    ap.add_argument("--shape", required=True, help="NxB, e.g. 2048x16")
    ap.add_argument("--out", default=str(TABLE_PATH))
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    try:
        n_s, b_s = args.shape.lower().split("x")
        n, batch = int(n_s), int(b_s)
    except ValueError:
        _die(f"--shape must look like 2048x16, got {args.shape!r}")

    ev_path = Path(args.evidence)
    if not ev_path.is_file():
        _die(f"evidence not found: {ev_path}")
    with open(ev_path) as f:
        ev = json.load(f)

    dirty = _git("status", "--porcelain")
    if dirty and not args.allow_dirty:
        _die("working tree dirty (commit first, or --allow-dirty); "
             "promotion evidence must bind a known commit")

    # --- freshness bindings -------------------------------------------
    live_env = current_env_hash()
    if live_env is None:
        _die("live environment fingerprint unknown; cannot verify")
    if ev.get("live_env_hash") != live_env:
        _die(f"evidence env {ev.get('live_env_hash')} != live "
             f"{live_env}")
    profile = DeviceProfile.load()
    profile_hash = profile.hash()
    if ev.get("profile_hash") != profile_hash:
        _die(f"evidence profile {ev.get('profile_hash')} != live "
             f"{profile_hash}")
    if ev.get("dirty") and not args.allow_dirty:
        _die("evidence recorded a dirty tree")
    if not ev.get("all_sessions_same_commit"):
        _die("evidence sessions span multiple commits")
    if not ev.get("distinct_pids"):
        _die("evidence sessions are not distinct processes")

    # --- cell + guards --------------------------------------------------
    cell = _find_cell(ev, n, batch)
    if cell.get("direction") != "dit":
        _die("composition flow is DIT-only")
    sessions = cell.get("sessions", [])
    if len(sessions) < MIN_SESSIONS:
        _die(f"{len(sessions)} sessions < {MIN_SESSIONS}")
    roles = {s.get("role") for s in sessions}
    if "confirmation" not in roles:
        _die("no independent confirmation session (role=confirmation) "
             "in the cell")
    pids = {s.get("pid") for s in sessions}
    if len(pids) != len(sessions):
        _die("sessions share a pid (not independent)")
    if any(s.get("provenance", {}).get("commit")
           != ev.get("commit") for s in sessions):
        _die("session provenance commit != evidence commit")

    pooled = cell.get("pooled", {})
    for name in ("A1P1", "A0P0", "vendor_full"):
        if name not in pooled or "p50" not in pooled[name]:
            _die(f"evidence missing pooled stats for {name}")
    err_max = float(ev.get("err_max", ERR_MAX + 1))
    if err_max > ERR_MAX:
        _die(f"evidence err_max {err_max} > {ERR_MAX}")
    if cell.get("abstain"):
        _die(f"cell abstained in evidence: {cell['abstain']}")

    cand_ms = round(float(pooled["A1P1"]["p50"]), 6)
    inc_ms = round(float(pooled["A0P0"]["p50"]), 6)
    vendor_ms = float(pooled["vendor_full"]["p50"])
    # guard stamped from the stamped medians so load-time recomputation
    # matches the field exactly
    guard = guard_pct_of(cand_ms, inc_ms)
    if guard < GUARD_PCT:
        _die(f"A1P1 guard {guard:.2f}% < {GUARD_PCT}% vs A0P0 "
             f"({cand_ms:.4f} vs {inc_ms:.4f} ms) — negative point, "
             "kept as-is, not promoted")

    evidence_rel = str(ev_path.relative_to(ROOT)) \
        if ev_path.is_absolute() and ROOT in ev_path.parents \
        else str(ev_path)
    entry = {
        "N": n, "batch": batch, "direction": cell["direction"],
        "precision": "fp32",
        "tile": int(cell.get("tile", 64)),
        "core": "vendor",
        "flow_id": FLOW_ID,
        "flow_realization": FLOW_REALIZATION,
        "processing_unit": PROCESSING_UNIT,
        "candidate_ms": cand_ms,
        "incumbent_ms": inc_ms,
        "incumbent_flow": "A0P0-production",
        "guard_pct": guard,
        "guard_pass": True,
        "reason": (f"exact-shape composition: A1P1 {cand_ms:.4f}ms vs "
                   f"production A0P0 {inc_ms:.4f}ms (+{guard:.1f}%), "
                   f"vendor_full {vendor_ms:.4f}ms "
                   f"({'+' if vendor_ms > cand_ms else ''}"
                   f"{(vendor_ms - cand_ms) / vendor_ms * 100:.1f}% "
                   "vendor delta recorded, not a gate)"),
        "contract_id": CONTRACT_ID,
        "evidence": {
            "artifact": evidence_rel,
            "artifact_sha256": _sha256_file(ev_path),
            "commit": ev.get("commit"),
            "sessions": len(sessions),
            "trials": int(cell.get("trials")
                          or ev.get("trials") or 0),
            "confirmation_session": True,
            "err_max": err_max,
            "p50_ms": cand_ms,
            "mad_ms": round(float(pooled["A1P1"].get("mad", 0.0)), 6),
            "p95_ms": round(float(pooled["A1P1"].get("p95", 0.0)), 6),
        },
    }

    # --- merge + validate + atomic write --------------------------------
    table = None
    out_path = Path(args.out)
    if out_path.is_file():
        with open(out_path) as f:
            table = json.load(f)
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
            "created_at": __import__("time").strftime(
                "%Y-%m-%dT%H:%M:%S%z"),
            "source_hash": _git("rev-parse", "HEAD"),
            "dirty": bool(dirty),
            "entries": [],
        }
    entries = [e for e in table.get("entries", [])
               if not (e.get("N") == n and e.get("batch") == batch
                       and e.get("direction") == entry["direction"])]
    entries.append(entry)
    table["entries"] = entries
    table["profile_hash"] = profile_hash
    table["env_hash"] = live_env
    table["result_sha256"] = rows_sha256({"rows": entries})

    violations = table_violations(table)
    if violations:
        _die("resulting table violates contract: "
             + "; ".join(violations[:8]))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(out_path.parent),
                               prefix=".composition_table.")
    with os.fdopen(fd, "w") as f:
        json.dump(table, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, out_path)
    print(f"promoted {n}x{batch}: guard +{guard:.2f}% "
          f"(cand {cand_ms:.4f}ms vs incumbent {inc_ms:.4f}ms) "
          f"-> {out_path}")
    print(f"result_sha256 {table['result_sha256']}")


if __name__ == "__main__":
    main()
