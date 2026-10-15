"""Host-only release-matrix runner helpers: NaN gate + warmup accounting.

Review follow-up: the correctness gate must never let a non-finite error
pass by comparison semantics, and the interleave warmup must consume the
declared round count (unit-tested by call counting).
"""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_release_matrix import _err_gate, _interleave_warmup  # noqa: E402


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
