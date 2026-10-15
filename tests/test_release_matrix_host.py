"""Host-only release-matrix helpers: error metric, NaN gate, warmup, median.

Review follow-ups:
* the correctness gate must never let a non-finite error pass by
  comparison semantics — the shared ``max_rel_error`` returns ``inf`` for
  any non-finite delta/reference (an imaginary-part NaN used to be
  swallowed by Python's builtin ``max``);
* the interleave warmup must consume the declared round count
  (unit-tested by call counting);
* medians must be standard (even sample counts average the two centers).
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_release_matrix import (  # noqa: E402
    _err_gate, _interleave_warmup, _median)
from src.xpu.error_metric import max_rel_error  # noqa: E402

_NAN = float("nan")
_INF = float("inf")
_NINF = float("-inf")


def _pair(got, ref):
    return (np.asarray([got], dtype=np.complex128),
            np.asarray([ref], dtype=np.complex128))


@pytest.mark.parametrize("bad", [_NAN, _INF, _NINF], ids=["nan", "inf", "-inf"])
def test_max_rel_error_real_part_nonfinite_is_inf(bad):
    err, ref = _pair(complex(bad, 0.0), complex(1.0, 0.0))
    assert max_rel_error(err, ref) == _INF
    assert not _err_gate(max_rel_error(err, ref))


@pytest.mark.parametrize("bad", [_NAN, _INF, _NINF], ids=["nan", "inf", "-inf"])
def test_max_rel_error_imag_part_nonfinite_is_inf(bad):
    """An imaginary NaN must not be swallowed by a finite real deviation."""
    err, ref = _pair(complex(1.0, bad), complex(1.0, 0.0))
    assert max_rel_error(err, ref) == _INF
    assert not _err_gate(max_rel_error(err, ref))


@pytest.mark.parametrize("bad", [_NAN, _INF, _NINF], ids=["nan", "inf", "-inf"])
def test_max_rel_error_nonfinite_reference_is_inf(bad):
    err, ref = _pair(complex(1.0, 0.0), complex(bad, 0.0))
    assert max_rel_error(err, ref) == _INF
    err, ref = _pair(complex(1.0, 0.0), complex(0.0, bad))
    assert max_rel_error(err, ref) == _INF


def test_max_rel_error_zero_for_identical_finite():
    err, ref = _pair(complex(0.3, -0.7), complex(0.3, -0.7))
    assert max_rel_error(err, ref) == 0.0
    assert _err_gate(max_rel_error(err, ref))


def test_max_rel_error_at_threshold_is_rejected():
    """Error exactly equal to the threshold must not pass the gate."""
    step = 2.0 ** -13  # exactly representable in binary64 (~1.22e-4)
    err, ref = _pair(complex(1.0 + step, 0.0), complex(1.0, 0.0))
    got = max_rel_error(err, ref)
    assert got == step  # 1.0 + 2^-13 subtracts back exactly
    assert not _err_gate(got, threshold=step)
    assert not _err_gate(got)  # also above the default 1e-4 gate


def test_max_rel_error_peak_normalized_componentwise():
    """Contract: max(max(|dRe|, |dIm|)) / max(|ref|)."""
    err, ref = _pair(complex(1.05, 0.01), complex(1.0, 0.0))
    # delta = 0.05 (real), scale = 1.0
    assert max_rel_error(err, ref) == pytest.approx(0.05)


def test_err_gate_rejects_nan_and_inf():
    assert not _err_gate(float("nan"))
    assert not _err_gate(float("inf"))
    assert not _err_gate(float("-inf"))
    assert not _err_gate(None)
    assert not _err_gate("1e-8")


def test_err_gate_rejects_at_and_above_threshold():
    assert not _err_gate(1e-4)
    assert not _err_gate(2e-4)
    assert _err_gate(9.9e-5)
    assert _err_gate(0.0)


def test_interleave_warmup_consumes_round_count():
    calls = []

    def fn():
        calls.append(1)

    _interleave_warmup([fn, fn], 3)
    assert len(calls) == 6  # 3 rounds x 2 callables
    _interleave_warmup([fn], 0)
    assert len(calls) == 6
    _interleave_warmup([fn, fn, fn], 10)
    assert len(calls) == 36


def test_median_odd_sample_count():
    assert _median([3, 1, 2]) == 2
    assert _median([5]) == 5
    assert _median([9, 1, 5, 3, 7]) == 5


def test_median_even_sample_count_averages_centers():
    assert _median([4, 1, 2, 3]) == 2.5
    assert _median([10, 20]) == 15.0
    assert _median([1, 2, 3, 4, 5, 6]) == 3.5


def test_median_empty_is_none():
    assert _median([]) is None
