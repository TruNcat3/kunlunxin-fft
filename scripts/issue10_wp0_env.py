#!/usr/bin/env python3
"""WP0: environment & toolchain provenance for Issue #10.

Records, before any capability/performance claim:

* device model / card / xpu-smi driver / XPU runtime build ids;
* python, torch, triton (+ backend package origin), flag_gems,
  torch_xmlir versions with ``__file__`` origins and git/wheel sources;
* compiler and relevant environment variables;
* the repo's live ``env_hash`` / calibration ``profile_hash`` /
  ``device_profile_hash`` / git source commit.

Output: ``results/p800/issue10_capability/wp0_environment.json``
(self-hashed like config/calibration.json: ``wp0_hash`` covers every
other field, verified by issue10_gates).

Usage:
    python3 scripts/issue10_wp0_env.py [--out PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

# Import order matters on this stack: flag_gems' import can leave a
# half-initialized torch_xmlir in sys.modules when the xpytorch hook
# fails silently.  Warm torch + torch_xmlir FIRST (pattern used by
# scripts/warm_triton.py) so provenance reads the real module.
import torch  # noqa: E402,F401
import torch_xmlir  # noqa: E402,F401

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

DEFAULT_OUT = ROOT / "results/p800/issue10_capability/wp0_environment.json"

ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "PYTHONPATH",
    "TRITON_CACHE_DIR", "MLIR_ENABLE_DUMP", "XPU_VISIBLE_DEVICES",
    "XCCL_", "OMP_NUM_THREADS",
)


def _content_hash(obj: dict) -> str:
    blob = json.dumps({k: v for k, v in obj.items() if k != "wp0_hash"},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _pkg_record(name: str) -> dict:
    """Version + origin path + source kind (git checkout / wheel / dist)."""
    rec: dict = {"name": name}
    try:
        import importlib
        m = importlib.import_module(name)
        if name == "torch_xmlir":
            # no __version__ on this package; the build dict is the version
            vd = m.get_version_dict()
            rec["version"] = vd.get("XPYTORCH") or vd.get("XRE") or str(vd)
            rec["build_ids"] = {k: vd.get(k) for k in
                                ("XPYTORCH", "XRE", "XHPC", "XTDK",
                                 "compile_date")}
        else:
            rec["version"] = getattr(m, "__version__", "unknown")
        f = getattr(m, "__file__", None)
        rec["file"] = str(Path(f).resolve()) if f else None
    except Exception as e:  # noqa: BLE001  (provenance: record the error)
        rec.update(version="UNAVAILABLE", error=f"{type(e).__name__}: {e}")
        return rec
    # distribution metadata (wheel / editable install)
    try:
        rec["dist_version"] = metadata.version(name)
    except Exception:  # noqa: BLE001
        rec["dist_version"] = None
    # git source detection: walk up from __file__ for a .git dir
    try:
        p = Path(rec["file"]) if rec.get("file") else None
        git_dir = None
        while p is not None and p != p.parent:
            if (p / ".git").exists():
                git_dir = p
                break
            p = p.parent
        if git_dir is not None:
            rec["git_root"] = str(git_dir)
            rec["git_head"] = subprocess.run(
                ["git", "-C", str(git_dir), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10,
                check=False).stdout.strip() or None
            rec["git_dirty"] = bool(subprocess.run(
                ["git", "-C", str(git_dir), "status", "--porcelain",
                 "-uno"], capture_output=True, text=True, timeout=10,
                check=False).stdout.strip())
            rec["source_kind"] = "git-checkout"
        elif rec.get("file"):
            rec["source_kind"] = "site-packages (wheel/site)"
    except Exception as e:  # noqa: BLE001
        rec["source_detect_error"] = f"{type(e).__name__}: {e}"
    return rec


def _xpu_smi() -> dict:
    exe = shutil.which("xpu-smi")
    if not exe:
        return {"available": False}
    try:
        out = subprocess.run([exe], timeout=20, capture_output=True,
                             text=True, check=False).stdout
    except Exception as e:  # noqa: BLE001
        return {"available": True, "error": f"{type(e).__name__}: {e}"}
    rec = {"available": True, "raw_head": "\n".join(out.splitlines()[:6])}
    m = re.search(r"Driver Version:\s*([0-9][0-9A-Za-z._-]*)", out)
    if m:
        rec["driver_version"] = m.group(1)
    m = re.search(r"(\d+)\s+P800", out)
    if m:
        rec["model"] = "P800 OAM"
    rec["cards_seen"] = out.count("P800 OAM")
    return rec


def _device_props() -> dict:
    try:
        from triton.backends.xpu.driver import XPUUtils
        props = dict(XPUUtils().get_device_properties(0))
        return {k: (v if isinstance(v, (int, float, str, bool))
                    else str(v)) for k, v in sorted(props.items())}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def _repo_fingerprint() -> dict:
    rec: dict = {}
    try:
        from src.xpu.envcheck import collect as env_collect, env_hash
        rec["live_env_fingerprint"] = env_collect()
        rec["live_env_hash"] = env_hash()
    except Exception as e:  # noqa: BLE001
        rec["live_env_error"] = f"{type(e).__name__}: {e}"
    try:
        cal = json.loads((ROOT / "config/calibration.json").read_text())
        rec["calibration_env_hash"] = cal.get("env_hash")
        rec["calibration_profile_hash"] = cal.get("profile_hash")
        rec["calibration_device_profile_hash"] = (
            cal.get("versions", {}).get("device_profile_hash"))
        rec["calibration_git_rev"] = cal.get("versions", {}).get("git_rev")
    except Exception as e:  # noqa: BLE001
        rec["calibration_error"] = f"{type(e).__name__}: {e}"
    rec["git_rev"] = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=10,
        check=False).stdout.strip() or None
    rec["git_dirty"] = bool(subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "-uno"],
        capture_output=True, text=True, timeout=10,
        check=False).stdout.strip())
    rec["git_untracked"] = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain"],
        capture_output=True, text=True, timeout=10,
        check=False).stdout.splitlines()
    return rec


def _compiler() -> dict:
    rec = {}
    for exe in ("cc", "c++"):
        path = shutil.which(exe)
        if not path:
            rec[exe] = None
            continue
        try:
            out = subprocess.run([exe, "--version"], timeout=10,
                                 capture_output=True, text=True,
                                 check=False)
            rec[exe] = {"path": path,
                        "first_line": out.stdout.splitlines()[0]}
        except Exception as e:  # noqa: BLE001
            rec[exe] = {"path": path, "error": str(e)}
    return rec


def collect() -> dict:
    doc: dict = {
        "schema": "issue10-wp0-environment-v1",
        "issue": "kunlunxin-fft#10",
        "created_unix": int(time.time()),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "purpose": ("WP0 provenance: recorded BEFORE capability/perf "
                    "claims; gates verify wp0_hash and env binding"),
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": {name: _pkg_record(name)
                     for name in ("torch", "triton", "flag_gems",
                                  "torch_xmlir")},
        "triton_backend": {},
        "device": {},
        "xpu_smi": _xpu_smi(),
        "compiler": _compiler(),
        "environment_variables": {k: os.environ.get(k) for k in ENV_KEYS},
        "repo": _repo_fingerprint(),
    }
    # triton vendor extras + backend origin (WP1 probe surface provenance)
    try:
        import triton
        doc["triton_backend"]["triton_file"] = str(
            Path(triton.__file__).resolve())
        import triton.language.extra.xpu as xpu_extra
        doc["triton_backend"]["extra_xpu_file"] = str(
            Path(xpu_extra.__file__).resolve())
        try:
            import triton.backends.xpu as bx
            doc["triton_backend"]["backends_xpu_file"] = str(
                Path(bx.__file__).resolve())
        except Exception as e:  # noqa: BLE001
            doc["triton_backend"]["backends_xpu"] = f"UNAVAILABLE: {e}"
        import triton.language as tl
        doc["triton_backend"]["tl_surface_probe"] = {
            name: hasattr(tl, name)
            for name in ("async_copy", "static_alloc", "debug_barrier",
                         "dot", "inline_asm_elementwise", "permute",
                         "trans", "split", "join", "swizzle2d")}
    except Exception as e:  # noqa: BLE001
        doc["triton_backend"]["error"] = f"{type(e).__name__}: {e}"
    # torch_xmlir build ids
    try:
        import torch  # noqa: F401
        import torch_xmlir
        doc["device"]["torch_xmlir_versions"] = (
            torch_xmlir.get_version_dict())
    except Exception as e:  # noqa: BLE001
        doc["device"]["torch_xmlir_versions"] = f"UNAVAILABLE: {e}"
    doc["device"]["device_properties"] = _device_props()
    doc["wp0_hash"] = _content_hash(doc)
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)
    doc = collect()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n")
    print(f"wp0: {out}")
    print(f"  wp0_hash: {doc['wp0_hash'][:16]}")
    print(f"  live_env_hash: {doc['repo'].get('live_env_hash')}")
    print(f"  calibration_env_hash: "
          f"{doc['repo'].get('calibration_env_hash')}")
    print(f"  git: {str(doc['repo'].get('git_rev'))[:12]} "
          f"dirty={doc['repo'].get('git_dirty')}")
    pk = doc["packages"]
    for name in ("torch", "triton", "flag_gems", "torch_xmlir"):
        r = pk[name]
        print(f"  {name}: version={r.get('version')} "
              f"source={r.get('source_kind') or r.get('error', '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
