"""Calibration profile (issue #3 P1.2): load-with-verification.

``scripts/hw_probe.py`` writes ``config/calibration.json`` on the device:
versions, verified execution paths, local resource constraints, and the
baseline launch/copy/compute/sync costs.  The file carries a self-hash
over its content; :func:`load` recomputes it and raises
:class:`CalibrationMismatch` on any tamper or partial write, so a stale
or edited profile is never silently reused.
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


def load(path: str | Path | None = None) -> dict:
    """Read the calibration profile, verifying its self-hash.

    Raises :class:`FileNotFoundError` when the probe has never run and
    :class:`CalibrationMismatch` when the stored hash differs from the
    recomputed one — callers must not fall back silently.
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
    return profile
