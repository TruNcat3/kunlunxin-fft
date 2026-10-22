"""issue #10 review item 1: bundle the evidence directory into a
reproducible artifact package.

Writes into results/p800/issue10_capability/:
  logs/            consolidated copies of wp1/wp2/wp3 step logs
  environment.json copy of wp0_environment.json (runtime lock)
  repro_command.txt exact reproduction commands, in dependency order
  manifest.json     file list with sha256/bytes/producer/script
                    (excludes manifest.json and SHA256SUMS itself)
  SHA256SUMS        sha256sum-format hash list of everything except
                    itself (covers manifest.json)

Run LAST, after all measurement scripts (re-packing changes hashes).
"""
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "results/p800/issue10_capability"
LOGS = PKG / "logs"

LOG_SOURCES = {
    "wp1": PKG / "wp1_probes/logs",
    "wp2": PKG / "wp2_logs",
    "wp3": PKG / "wp3_logs",
}

PRODUCERS = [
    ("wp0_", "scripts/issue10_wp0_env.py"),
    ("environment.json", "scripts/issue10_wp0_env.py"),
    ("capability_matrix.json", "scripts/issue10_wp1_capability.py"),
    ("wp1_", "scripts/issue10_wp1_capability.py"),
    ("wp2_", "scripts/issue10_wp2_minikernels.py"),
    ("ops_supported.json", "scripts/issue10_wp2_minikernels.py"),
    ("wp3_blockmax", "scripts/issue10_wp3_perf.py"),
    ("wp3_raw", "scripts/issue10_wp3_perf.py"),
    ("wp3_summary", "scripts/issue10_wp3_perf.py"),
    ("wp3_gonogo", "scripts/issue10_wp3_perf.py"),
    ("wp3_run_summary", "scripts/issue10_wp3_perf.py"),
    ("wp3_profile", "scripts/issue10_wp3_profile.py"),
    ("wp3_trace", "scripts/issue10_wp3_profile.py"),
    ("wp3_logs", "scripts/issue10_wp3_perf.py --step all"),
    ("stale_read_matrix", "scripts/issue10_stale_read_matrix.py"),
    ("wp3_holdout", "scripts/issue10_wp3_holdout.py"),
    ("selector_diff", "scripts/issue10_wp3_holdout.py"),
    ("wp3_ablation", "scripts/issue10_wp3_ablation.py"),
    ("logs/", "consolidated by scripts/issue10_pack_artifacts.py"),
    ("repro_command.txt", "scripts/issue10_pack_artifacts.py"),
    ("manifest.json", "scripts/issue10_pack_artifacts.py"),
    ("SHA256SUMS", "scripts/issue10_pack_artifacts.py"),
]

REPRO = """issue #10 reproduction commands (P800, single card, from repo root)

# runtime: conda env python310_torch29_cuda (torch 2.9 + torch_xmlir,
# triton 3.0.0 XPU fork), Kunlunxin P800; see environment.json
export CUDA_VISIBLE_DEVICES=2

# WP0/WP1 capability matrix (writes wp0_environment.json,
# capability_matrix.json, wp1_probes/)
python3 scripts/issue10_wp0_env.py
python3 scripts/issue10_wp1_capability.py

# WP2 mini-kernel gates (writes wp2_*.json, wp2_logs/)
python3 scripts/issue10_wp2_minikernels.py

# WP3 blockmax + perf + go/no-go (writes wp3_blockmax.json,
# wp3_raw.csv, wp3_summary.json, wp3_gonogo.json, wp3_logs/)
python3 scripts/issue10_wp3_perf.py --step all

# WP3 cupti profile + chrome traces (writes wp3_profile.json, wp3_trace/)
XPU_ENABLE_PROFILER_TRACING=1 XPU_CUPTI_ENABLE_DEVICE=0 python3 scripts/issue10_wp3_profile.py

# stale-read root-cause matrix (writes stale_read_matrix.json;
# 2 independent sessions, fresh TRITON_CACHE_DIR each)
python3 scripts/issue10_stale_read_matrix.py

# performance holdout (writes wp3_holdout.json + selector_diff.json;
# 16 cells x 4 paths [vendor, hybrid_vendor_p, hybrid_fused_p,
# incumbent_entry same-protocol] x 2 independent sessions; one
# subprocess per cell/path/session with fresh TRITON_CACHE_DIR,
# itemized context/setup/probe/first_call/steady segments, complex64
# non-zero-imag input, independent bootstrap CI)
python3 scripts/issue10_wp3_holdout.py

# twiddle ablation (writes wp3_ablation.json; 5 implementations x
# 10 cells x 2 sessions)
python3 scripts/issue10_wp3_ablation.py

# package: logs/ + environment.json + repro_command.txt + manifest.json
# + SHA256SUMS  (RUN LAST)
python3 scripts/issue10_pack_artifacts.py

# verify
sha256sum -c SHA256SUMS
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


def main() -> int:
    if not PKG.is_dir():
        print(f"missing package dir {PKG}", file=sys.stderr)
        return 1

    # 1) consolidate logs
    LOGS.mkdir(exist_ok=True)
    n_logs = 0
    for group, src in LOG_SOURCES.items():
        if not src.is_dir():
            continue
        for f in sorted(src.glob("*.log")):
            shutil.copy2(f, LOGS / f"{group}_{f.name}")
            n_logs += 1

    # 2) environment lock copy
    wp0 = PKG / "wp0_environment.json"
    if wp0.exists():
        shutil.copy2(wp0, PKG / "environment.json")

    # 3) repro commands
    (PKG / "repro_command.txt").write_text(REPRO)

    # 4) manifest (excludes itself + SHA256SUMS)
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT),
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        head = None
    skip = {"manifest.json", "SHA256SUMS"}
    files = []
    for f in sorted(PKG.rglob("*")):
        if not f.is_file() or f.name in skip or "__pycache__" in f.parts:
            continue
        rel = str(f.relative_to(PKG))
        files.append({
            "path": rel,
            "bytes": f.stat().st_size,
            "sha256": _sha256(f),
            "producer": _producer(rel),
        })
    manifest = {
        "schema": "issue10-manifest-v1",
        "issue": "kunlunxin-fft#10",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "package_dir": str(PKG.relative_to(ROOT)),
        "git_head_at_pack": head,
        "note": "git_head_at_pack is the parent of the packaging "
                "commit (the pack is committed together with the "
                "scripts); SHA256SUMS additionally covers "
                "manifest.json.",
        "n_files": len(files),
        "total_bytes": sum(x["bytes"] for x in files),
        "n_logs": n_logs,
        "files": files,
    }
    (PKG / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # 5) SHA256SUMS (covers everything except itself, incl manifest)
    lines = []
    for f in sorted(PKG.rglob("*")):
        if (not f.is_file() or f.name == "SHA256SUMS"
                or "__pycache__" in f.parts):
            continue
        lines.append(f"{_sha256(f)}  {f.relative_to(PKG)}")
    (PKG / "SHA256SUMS").write_text("\n".join(lines) + "\n")

    print(json.dumps({
        "manifest_files": len(manifest["n_files"] and files),
        "total_bytes": manifest["total_bytes"],
        "logs": n_logs,
        "sha256sums_lines": len(lines),
        "written": ["logs/", "environment.json", "repro_command.txt",
                    "manifest.json", "SHA256SUMS"],
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
