#!/usr/bin/env python3
"""Issue #9 release artifact ledger: machine-checkable evidence list.

Writes/verifies SHA-256 for the fixed inventory of release-critical
artifacts (the five reports, the complete v5 paired-run evidence
including manifest / previous table / raw sessions / schedule, the
identity + attribution reports, the bound v5 table, the immutable
release identity files and the frozen configs).  Lines use the
``sha256sum`` format (``<hash>  <path>``), so the ledger can also be
checked with ``sha256sum -c`` outside this repo's tooling.

The ledger deliberately EXCLUDES run transcripts that are rewritten
after the fact (gates/host/hw logs) -- verifying them here would be
circular; ``issue9_release_gates.py --ledger`` executes the per-file
recompute as a hard gate.

Usage:
    scripts/issue9_artifact_ledger.py --write PATH
    scripts/issue9_artifact_ledger.py --verify PATH
"""

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

V61 = "results/p800/issue9_v61"
REL = "results/p800/releases/20261020-issue9-semantic-matrix-v6-1"

INVENTORY = [
    f"{V61}/reports/steady_report.json",
    f"{V61}/reports/steady_report.csv",
    f"{V61}/reports/e2e_report.json",
    f"{V61}/reports/e2e_report.csv",
    f"{V61}/reports/scale_report.json",
    f"{V61}/reports/scale_report.csv",
    f"{V61}/reports/shape_proxy_report.json",
    f"{V61}/reports/shape_proxy_report.csv",
    f"{V61}/reports/ablation_scope_report.json",
    f"{V61}/reports/ablation_scope_report.md",
    f"{V61}/v5_pair/manifest.json",
    f"{V61}/v5_pair/previous_selector_table.json",
    f"{V61}/v5_pair/schedule.json",
    f"{V61}/v5_pair/summary.json",
    f"{V61}/v5_pair/summary.csv",
    f"{V61}/v5_pair/regression_vs_previous.csv",
    f"{V61}/v5_pair/speedup_vs_vendor.png",
    f"{V61}/v5_pair/raw_session0.jsonl",
    f"{V61}/v5_pair/raw_session1.jsonl",
    f"{V61}/v5_pair/raw_session2.jsonl",
    f"{V61}/v5_path_identity.json",
    f"{V61}/attribution.json",
    f"{V61}/v5_production_selector_table_bound.json",
    f"{REL}/summary.json",
    f"{REL}/manifest.json",
    "config/issue9_p800.json",
    "config/selector_table.json",
]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(root: Path, out: Path):
    lines = []
    missing = []
    for rel in INVENTORY:
        p = root / rel
        if not p.is_file():
            missing.append(rel)
            continue
        lines.append(f"{_sha256(p)}  {rel}")
    if missing:
        for m in missing:
            print(f"MISSING: {m}", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"ledger: {out} ({len(lines)} files)")
    return 0


def verify(root: Path, ledger: Path):
    """Return (n_files, errors) -- errors is empty iff every listed
    hash matches the file on disk and every inventory file is listed."""
    if not ledger.is_file():
        return 0, [f"ledger not found: {ledger}"]
    listed = {}
    errors = []
    for i, line in enumerate(ledger.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            errors.append(f"line {i}: malformed")
            continue
        listed[parts[1]] = parts[0]
    for rel in INVENTORY:
        if rel not in listed:
            errors.append(f"not in ledger: {rel}")
    for rel, want in listed.items():
        p = root / rel
        if not p.is_file():
            errors.append(f"missing file: {rel}")
            continue
        got = _sha256(p)
        if got != want:
            errors.append(f"SHA MISMATCH: {rel} (ledger {want[:12]} "
                          f"!= actual {got[:12]})")
    return len(listed), errors


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", metavar="PATH")
    ap.add_argument("--verify", metavar="PATH")
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args(argv)
    if bool(args.write) == bool(args.verify):
        ap.error("exactly one of --write / --verify is required")
    root = Path(args.root)
    if args.write:
        return write(root, Path(args.write))
    n, errors = verify(root, Path(args.verify))
    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        print(f"ledger verify FAILED ({len(errors)} problems, "
              f"{n} listed)")
        return 1
    print(f"ledger verify OK ({n} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
