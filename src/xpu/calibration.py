"""Calibration profile (issue #3 P1.2): load-with-verification.

``scripts/hw_probe.py`` writes ``config/calibration.json`` on the device:
versions, verified execution paths, local resource constraints, and the
baseline launch/copy/compute/sync costs.  The file carries a self-hash
over its content; :func:`load` recomputes it and raises
:class:`CalibrationMismatch` on any tamper or partial write, so a stale
or edited profile is never silently reused.

Self-hash alone only proves the file was not edited.  The profile also
stores the live **environment fingerprint** the probe measured on the
device (driver, torch/triton, torch_xmlir build ids, device properties --
:mod:`src.xpu.envcheck`); with ``verify_env=True`` (the default) a load
on a different or unprobeable stack raises instead of reusing anchors
from the wrong environment (issue #3 review fix: profile freshness).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / "config" / "calibration.json"


class CalibrationMismatch(RuntimeError):
    """Stored ``profile_hash`` does not match the profile content."""


def compute_hash(profile: dict) -> str:
    """Hash the profile content (everything except ``profile_hash``)."""
    content = {k: v for k, v in profile.items() if k != "profile_hash"}
    blob = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def load(path: str | Path | None = None,
         verify_env: bool = True) -> dict:
    """Read the calibration profile, verifying self-hash and environment.

    Raises :class:`FileNotFoundError` when the probe has never run and
    :class:`CalibrationMismatch` when the stored hash differs from the
    recomputed one, when the profile carries no environment fingerprint,
    or (with ``verify_env=True``, the default) when the fingerprint does
    not match the live device/runtime/compiler — callers must not fall
    back silently.  ``verify_env=False`` is an explicit opt-out for
    offline inspection of a profile from a different host.
    """
    p = Path(path) if path is not None else DEFAULT_PATH
    with open(p) as f:
        profile = json.load(f)
    if not isinstance(profile, dict):
        raise CalibrationMismatch(f"{p}: calibration profile must be an object")
    stored = profile.get("profile_hash")
    if not isinstance(stored, str) or not stored:
        raise CalibrationMismatch(f"{p}: missing profile_hash")
    actual = compute_hash(profile)
    if actual != stored:
        raise CalibrationMismatch(
            f"{p}: profile_hash {stored} != recomputed {actual}; "
            "re-run scripts/hw_probe.py before reusing this profile")
    if verify_env:
        from .envcheck import collect, env_hash
        stored_env = profile.get("env_hash")
        if not stored_env or not profile.get("env"):
            raise CalibrationMismatch(
                f"{p}: no environment fingerprint bound to this profile "
                "(written before profile freshness); re-run "
                "scripts/hw_probe.py before reusing it")
        live = collect()
        live_hash = env_hash(live)
        if live_hash is None:
            raise CalibrationMismatch(
                f"{p}: live environment fingerprint unavailable "
                "(device/runtime/compiler not probeable); refusing to "
                "reuse this calibration profile here")
        if live_hash != stored_env:
            raise CalibrationMismatch(
                f"{p}: environment {live_hash} != profile {stored_env} "
                "(device/runtime/compiler changed since the probe); "
                "re-run scripts/hw_probe.py before reusing this profile")
    return profile
