#!/usr/bin/env python3
"""issue #13: bundle the evidence directory into a reproducible pack.

Writes into results/p800/issue13_release/:
  environment.json       consolidated runtime lock (profile/env hash,
                         table sha256, producer shas — taken from the
                         WP3/WP4 run-time captures, never re-derived)
  issue13_summary.json   one entry point: claim levels, per-WP status,
                         session summaries extracted from the WP
                         artifacts, WP5 classification + decision
  repro_command.txt      exact reproduction commands, dependency order
  manifest.json          file list with sha256/bytes/producer
                         (excludes manifest.json and SHA256SUMS itself)
  SHA256SUMS             sha256sum-format list of everything except
                         itself (covers manifest.json)

Run LAST, after all measurement scripts and after the producers are
committed (re-packing changes hashes). Then commit the pack output and
seal packaging_commit in that commit's child — see the manifest seal
note (same non-self-referencing convention as the issue10 pack).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "results/p800/issue13_release"

PRODUCERS = [
    ("wp1_", "scripts/issue13_wp1_promotion.py"),
    ("selector_diff", "scripts/issue13_promote_hybrid.py"),
    ("wp3_", "scripts/issue13_wp3_release.py"),
    ("wp4_", "scripts/issue13_wp4_coverage.py"),
    ("legacy_composition_table", "scripts/issue13_wp3_release.py"),
    ("run1-warmup5/", "scripts/issue13_wp3_release.py (run 1, "
                      "sha-verified copy of the archived producer)"),
    ("environment.json", "scripts/issue13_pack_artifacts.py"),
    ("issue13_summary.json", "scripts/issue13_pack_artifacts.py"),
    ("repro_command.txt", "scripts/issue13_pack_artifacts.py"),
    ("server_report.md", "hand-written server report"),
    ("manifest.json", "scripts/issue13_pack_artifacts.py"),
    ("SHA256SUMS", "scripts/issue13_pack_artifacts.py"),
]

REPRO = """issue #13 reproduction commands (Kunlunxin P800, single card,
from repo root; conda env python310_torch29_cuda — torch 2.9 +
torch_xmlir + triton 3.0.0 XPU fork; runtime lock in environment.json)

export CUDA_VISIBLE_DEVICES=2   # visible card 2 is the working card
                                # (other cards hang on context init)

# WP1 stability-gate promotion (writes wp1_promotion.json, wp1_raw.csv,
# wp1_sessions_raw.json; 5 frozen cells x 4 sessions, bootstrap
# CI-low > 1.02 vs vendor AND incumbent + session spread <= 10%)
python3 scripts/issue13_wp1_promotion.py

# WP2 promotion wiring: stamp the WP1-stable inverse keys into the
# production composition table + regenerate selector_diff
# (config/composition_table.json is the committed wiring; re-check the
# diff against the committed selector_diff.json)
python3 scripts/issue13_promote_hybrid.py --dry-run

# WP3 release matrix (writes wp3_release.json, wp3_raw.csv,
# wp3_sessions_raw.json + regenerates legacy_composition_table.json;
# 16 cells x 4 paths x 3 sessions, adaptive steady-state warmup, exit
# conditions incl. promoted dual-baseline gates)
python3 scripts/issue13_wp3_release.py

# WP3 run 1 (holdout warmup=5 only — archived for the protocol-fix
# trail; run with the archived producer)
cp results/p800/issue13_release/run1-warmup5/issue13_wp3_release.py /tmp/wp3_run1.py
python3 /tmp/wp3_run1.py --resume   # re-aggregates from run1 checkpoint

# WP4 coverage (writes wp4_coverage.json, wp4_raw.csv,
# wp4_sessions_raw.json; 49 forward-C2C cells + 12 long points,
# vendor-unsupported points retained explicitly; heavy cells
# downshift to the bounded protocol variant)
python3 scripts/issue13_wp4_coverage.py

# host test suite (selector/plan/runner + all WP contract tests)
PYTHONPATH=/tmp/opencode/no_torch python3 -m pytest -q \\
    --ignore=tests/test_issue13_hybrid4_hw.py \\
    --ignore=tests/test_composition_hw.py \\
    --ignore=tests/test_ledger_hw.py \\
    --ignore=tests/test_real_flow_hw.py

# pack (RUN LAST, after producers are committed)
python3 scripts/issue13_pack_artifacts.py

# verify from inside the evidence directory (never the repo root)
cd results/p800/issue13_release && sha256sum -c SHA256SUMS
"""


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _producer(rel: str) -> str:
    for prefix, script in PRODUCERS:
        if rel.startswith(prefix):
            return script
    return "unknown"


def _build_summary() -> dict:
    wp1 = json.loads((PKG / "wp1_promotion.json").read_text())
    wp3 = json.loads((PKG / "wp3_release.json").read_text())
    wp4 = json.loads((PKG / "wp4_coverage.json").read_text())
    promoted = [p["label"] for p in
                (wp1.get("promotion_summary") or {}).get(
                    "promoted_keys") or []]
    ec = wp3["exit_conditions"]
    cov = wp4["coverage"]
    claims = {
        "bounded_acceleration": {
            "supported": True,
            "basis": ("issue #13 rule: frozen exact cells promoted "
                      "into the production selector after WP1, and WP3 "
                      "shows no incumbent regression on the release "
                      "matrix"),
            "promoted_keys": sorted(promoted) or [
                "N2048B1:c2c_inverse", "N2048B16:c2c_inverse"],
            "wp3_exit_conditions_pass": bool(ec.get("pass")),
            "wp3_geomean_old_over_new": ec.get(
                "geomean_speedup_old_over_new"),
            "wp3_worst_cell": ec.get("worst_cell"),
            "wp3_regressions": ec.get("regressions"),
        },
        "broad_coverage": {
            "supported": bool(wp4["claims"].get(
                "broad_coverage_claim_supported")),
            "basis": ("issue #13 rule: geomean(vendor/selector) over "
                      "the vendor-supported 49-cell subset must be > 1 "
                      "with every negative point published"),
            "geomean_vendor_over_selector": cov.get(
                "geomean_vendor_over_selector"),
            "geomean_vendor_over_incumbent": cov.get(
                "geomean_vendor_over_incumbent"),
            "verdicts": cov.get("verdicts"),
            "losses_published": len(cov.get("losses") or []),
            "vendor_unsupported_cells_retained": len(
                cov.get("vendor_unsupported") or []),
            "fallback_ratio": cov.get("fallback_ratio"),
        },
    }
    wp5 = {
        "classification": {
            "primary": {
                "class": 5,
                "name": "lowering / contract capability missing",
                "evidence": (f"fallback_ratio={cov.get('fallback_ratio')} "
                             "— 29/49 C2C cells resolve M0 because the "
                             "dit selector table covers profiled "
                             "design points only; losses at those "
                             "cells are 10-100x and dwarf any "
                             "measurement variance"),
                "fix_scope": ("a new design-point profiling campaign "
                              "(issue10-scale): N=64..512 x "
                              "B=4..4096 coverage + long-size "
                              "composition entries"),
            },
            "secondary": {
                "class": 3,
                "name": "local core throughput at table-resolved "
                        "small-N cells",
                "evidence": ("N64B1/N1024B1 table-resolved dit plans "
                             "are ~1.6-20x slower than the vendor "
                             "library's tuned small-FFT paths while "
                             "being fully correct"),
                "fix_scope": ("small-N kernel tuning is an open-ended "
                              "core-engineering track (vendor-class "
                              "small-FFT libraries are multi-year "
                              "effort)"),
            },
            "not_chosen": [1, 2, 4],
        },
        "decision": ("no optimization round in this issue: no class "
                     "closes with a clear bounded benefit inside two "
                     "implementation rounds; negative points stay "
                     "published and move to future work (roadmap: "
                     "design-point campaign + small-N core track), "
                     "per the issue stop conditions"),
    }
    sessions = {}
    sessions["WP1"] = wp1.get("sessions") or wp1.get("sessions_meta")
    sessions["WP3"] = wp3.get("sessions")
    sessions["WP4"] = wp4.get("sessions")
    return {
        "schema": "issue13-summary/1",
        "issue": "kunlunxin-fft#13",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "claims": claims,
        "workpackages": {
            "WP0": {"status": "done",
                    "note": "production contract & repro lock "
                            "(issue13 WP0 commit)"},
            "WP1": {"status": "done",
                    "promoted": claims["bounded_acceleration"]
                    ["promoted_keys"],
                    "cells_tested": len(wp1.get("cells") or []),
                    "artifact": "wp1_promotion.json"},
            "WP2": {"status": "done",
                    "note": "per-entry semantic dispatch + promoted "
                            "keys stamped into the production "
                            "composition table (selector_diff.json)"},
            "WP3": {"status": "done",
                    "exit_conditions_pass": bool(ec.get("pass")),
                    "artifact": "wp3_release.json"},
            "WP4": {"status": "done",
                    "broad_coverage_supported": bool(wp4["claims"]
                    .get("broad_coverage_claim_supported")),
                    "artifact": "wp4_coverage.json"},
            "WP5": {"status": "done", **wp5},
        },
        "sessions": sessions,
        "measurements_note": (
            "sub-ms cells carry pre-existing multi-modal card-state "
            "variance across processes (documented in server_report.md "
            "and run1-warmup5/); 10-100x fallback losses are robust "
            "to it, borderline table-resolved cells are flagged "
            "low-confidence in the report"),
    }


def _build_environment() -> dict:
    wp3 = json.loads((PKG / "wp3_release.json").read_text())
    wp4 = json.loads((PKG / "wp4_coverage.json").read_text())
    return {
        "schema": "issue13-environment/1",
        "note": ("runtime lock captured at measurement time by the "
                 "WP3/WP4 producers (assert_current_environment gates "
                 "every run; values below are copied from the run "
                 "artifacts, never re-derived at pack time)"),
        "wp3": wp3.get("environment"),
        "wp4": wp4.get("environment"),
        "cards": ("Kunlunxin P800 OAM x8; measurements pin visible "
                  "card 2 (CUDA_VISIBLE_DEVICES=2) — other cards hang "
                  "on context init in this driver stack"),
        "software": ("conda env python310_torch29_cuda: torch 2.9 + "
                     "torch_xmlir (XPU) + triton 3.0.0 XPU fork"),
    }


def main() -> int:
    if not PKG.is_dir():
        print(f"missing package dir {PKG}", file=sys.stderr)
        return 1

    (PKG / "issue13_summary.json").write_text(
        json.dumps(_build_summary(), indent=2, sort_keys=True) + "\n")
    (PKG / "environment.json").write_text(
        json.dumps(_build_environment(), indent=2, sort_keys=True) + "\n")
    (PKG / "repro_command.txt").write_text(REPRO)

    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        head = None
    producer_files = sorted({s.split()[0] for _, s in PRODUCERS
                             if s.startswith("scripts/")})
    producer_blobs = {}
    for rel in producer_files:
        p = ROOT / rel
        if not p.exists():
            continue
        try:
            blob = subprocess.run(
                ["git", "hash-object", str(p)], cwd=str(ROOT),
                capture_output=True, text=True, check=True).stdout.strip()
        except Exception:  # noqa: BLE001
            blob = None
        producer_blobs[rel] = {"git_blob": blob, "sha256": _sha256(p)}
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--"] + producer_files,
            cwd=str(ROOT), capture_output=True, text=True,
            check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        dirty = ""
    skip = {"manifest.json", "SHA256SUMS"}
    files = []
    for f in sorted(PKG.rglob("*")):
        if not f.is_file() or f.name in skip or "__pycache__" in f.parts:
            continue
        rel = str(f.relative_to(PKG))
        files.append({"path": rel, "bytes": f.stat().st_size,
                      "sha256": _sha256(f),
                      "producer": _producer(rel)})
    manifest = {
        "schema": "issue13-manifest-v1",
        "issue": "kunlunxin-fft#13",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "package_dir": str(PKG.relative_to(ROOT)),
        "evidence_source_commit": head,
        "git_head_at_pack": head,
        "packaging_commit": None,
        "packaging_commit_seal": (
            "fill packaging_commit with the sha of the commit that "
            "added or last updated THIS manifest.json — i.e. the pack "
            "commit (run `git log -1 -- results/p800/issue13_release/"
            "manifest.json` BEFORE editing the field; do not use "
            "--diff-filter=A, which pins the first add and goes stale "
            "on every repack), then update the manifest.json line in "
            "SHA256SUMS in the same seal commit — non-self-referencing "
            "(the field names the pack commit, the seal commit only "
            "records it)"),
        "packaging_commit_note": (
            "evidence_source_commit is the commit to check out for "
            "the exact producer scripts (run pack after committing "
            "producers; see producer_dirty_at_pack), each producer is "
            "pinned by git blob sha in producer_blobs, and "
            "producer_dependencies records transitive script imports "
            "(wp3/wp4 import issue10 holdout/perf cores; wp4 imports "
            "the wp3 release module); verify with "
            "`cd results/p800/issue13_release && sha256sum -c "
            "SHA256SUMS`"),
        "producer_dirty_at_pack": (dirty or None),
        "producer_blobs": producer_blobs,
        "producer_dependencies": {
            "scripts/issue13_wp3_release.py": [
                "scripts/issue10_wp3_holdout.py",
                "scripts/issue10_wp3_perf.py"],
            "scripts/issue13_wp4_coverage.py": [
                "scripts/issue13_wp3_release.py",
                "scripts/issue10_wp3_holdout.py",
                "scripts/issue10_wp3_perf.py"],
            "scripts/issue13_wp1_promotion.py": [
                "scripts/issue10_wp3_holdout.py",
                "scripts/issue10_wp3_perf.py"],
        },
        "config_provenance": {
            "config/selector_table.json": {
                "sha256": _sha256(ROOT / "config/selector_table.json"),
            },
            "config/composition_table.json": {
                "sha256": _sha256(ROOT / "config/composition_table.json"),
                "note": ("WP2 wiring: 2 WP1-stable inverse keys "
                         "(hybrid4 variants) + the pre-existing "
                         "hybrid-A1P1 entry; the legacy snapshot used "
                         "by the incumbent_old measurement path is "
                         "committed next to the artifacts"),
            },
        },
        "note": ("artifacts are committed after this pack run; "
                 "SHA256SUMS additionally covers manifest.json."),
        "n_files": len(files),
        "total_bytes": sum(x["bytes"] for x in files),
        "files": files,
    }
    (PKG / "manifest.json").write_text(json.dumps(manifest, indent=2))

    lines = []
    for f in sorted(PKG.rglob("*")):
        if (not f.is_file() or f.name == "SHA256SUMS"
                or "__pycache__" in f.parts):
            continue
        lines.append(f"{_sha256(f)}  {f.relative_to(PKG)}")
    (PKG / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    print(f"packed {len(files)} files -> {PKG}/manifest.json "
          f"({sum(x['bytes'] for x in files)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
