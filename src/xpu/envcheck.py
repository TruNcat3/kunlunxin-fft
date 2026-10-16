"""Live environment fingerprint (issue #3 fix: profile freshness).

A calibration profile or a selector table that only self-hashes proves it
was not edited; it does not prove it belongs to *this* device/runtime/
compiler. :func:`collect` gathers what this stack can actually obtain at
runtime — live driver (``xpu-smi``), torch/triton versions, the torch_xmlir
XPYTORCH/XRE/XHPC build ids, and device properties (model, cluster count,
shared memory). Anything missing or unreadable makes the whole fingerprint
**unknown** (:func:`collect` returns ``None``): callers must fail closed
(selector abstains to M0, calibration refuses reuse) instead of guessing.

Results are cached per process — the fingerprint cannot change while a
run is alive.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from functools import lru_cache

# Keys that MUST be present for a fingerprint to be usable at all.
_REQUIRED = ("driver", "torch", "triton", "xpytorch", "xre",
             "device_model", "multiprocessor_count", "max_shared_mem")


def _driver_version() -> str | None:
    """Live driver version from xpu-smi (None when unavailable)."""
    exe = shutil.which("xpu-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe], timeout=15, capture_output=True,
                             text=True, check=False).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"Driver Version:\s*([0-9][0-9A-Za-z._-]*)", out)
    return m.group(1) if m else None


@lru_cache(maxsize=4)
def collect() -> dict | None:
    """Best-effort live fingerprint; ``None`` means unknown (fail closed)."""
    try:
        import torch
        import triton

        env = {"torch": torch.__version__, "triton": triton.__version__}
        try:
            import torch_xmlir
            vd = torch_xmlir.get_version_dict()
            env["xpytorch"] = vd.get("XPYTORCH")
            env["xre"] = vd.get("XRE")
            env["xhpc"] = vd.get("XHPC")
        except Exception:  # noqa: BLE001  (no xmlir => unknown fingerprint)
            env["xpytorch"] = env["xre"] = env["xhpc"] = None
        env["driver"] = _driver_version()
        from triton.backends.xpu.driver import XPUUtils
        props = dict(XPUUtils().get_device_properties(0))
        env["device_model"] = int(props["device_model"])
        env["multiprocessor_count"] = int(props["multiprocessor_count"])
        env["max_shared_mem"] = int(props["max_shared_mem"])
    except Exception:  # noqa: BLE001
        return None
    if any(env.get(k) in (None, "") for k in _REQUIRED):
        return None
    return env


def env_hash(env: dict | None = None) -> str | None:
    """16-hex hash of a fingerprint dict; ``None`` when env is unknown."""
    if env is None:
        env = collect()
    if env is None:
        return None
    blob = json.dumps(env, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def current_env_hash() -> str | None:
    """Hash of the live fingerprint (cached); ``None`` when unknown."""
    return env_hash(collect())
