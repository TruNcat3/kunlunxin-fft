"""Shared max_rel_error definition for validation and release runners.

The historical correctness contract (do **not** change the norm without a
contract revision) is the peak-normalized component-wise deviation::

    max(max(|Re(got-ref)|, |Im(got-ref)|)) / max(|ref|)

Python's builtin ``max`` is not NaN-safe (``max(finite, nan)`` returns the
finite value), so an imaginary-part NaN used to be swallowed when the real
deviation was finite. This helper uses NumPy element-wise operations and
returns ``inf`` for any non-finite delta or reference, so no comparison
semantics can score a broken output as correct.
"""
from __future__ import annotations

import numpy as np


def max_rel_error(got, ref) -> float:
    """Contract error between ``got`` and ``ref``; ``inf`` if non-finite."""
    g = np.asarray(got)
    r = np.asarray(ref)
    real_err = np.abs(g.real - r.real)
    imag_err = np.abs(g.imag - r.imag)
    delta = np.maximum(real_err, imag_err)
    scale = np.abs(r).max()
    if not np.isfinite(delta).all() or not np.isfinite(scale):
        return float("inf")
    return float(delta.max() / max(float(scale), 1e-12))
