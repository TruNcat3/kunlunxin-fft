"""Host regression: four_cell entry must not NameError after deferral.

Review P1 at b989e1d: module-level ``_timed_cell`` referenced
``max_rel_error`` / ``time_cell`` which the deferred-import patch had
moved into ``main()``'s locals — the first cell call would crash with
NameError. The fix lazy-imports them inside ``_timed_cell`` (still
after the gate). This test actually *calls* ``_timed_cell`` with a
mocked device stack, instead of only checking the AST import shape.
"""

import sys
import types
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import four_cell_ablation as fc  # noqa: E402


class _FakeTensor:
    device = "xpu:0"

    def __init__(self, arr):
        self._arr = np.asarray(arr)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr


class _FakeCuda:
    @staticmethod
    def synchronize(device=None):
        return None


def _install_fake_device_stack(monkeypatch):
    """Minimal torch + adapter_hybrid + error_metric for _timed_cell."""
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = _FakeCuda()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_em = types.ModuleType("src.xpu.error_metric")
    fake_em.max_rel_error = lambda y, ref: float(
        np.max(np.abs(y - ref)) / max(np.max(np.abs(ref)), 1e-12))
    monkeypatch.setitem(sys.modules, "src.xpu.error_metric", fake_em)

    calls = {"time_cell": 0}

    def fake_time_cell(fn, x, warmup, trials):
        calls["time_cell"] += 1
        # Exercise the fn once so a NameError inside the cell would
        # surface here too.
        fn(x)
        return None, {"p50": 0.1, "mad": 0.0, "p95": 0.2, "max": 0.2,
                      "outlier_count": 0, "outlier_values": []}

    fake_ah = types.ModuleType("src.xpu.adapter_hybrid")
    fake_ah.time_cell = fake_time_cell
    monkeypatch.setitem(sys.modules, "src.xpu.adapter_hybrid", fake_ah)
    return calls


def test_timed_cell_enters_correctness_and_timing_path(monkeypatch):
    calls = _install_fake_device_stack(monkeypatch)
    x = _FakeTensor(np.ones((2, 4), dtype=np.complex64))
    ref = np.ones((2, 4), dtype=np.complex64)

    def cell_fn(xx):
        return xx  # identity passes the correctness gate

    out = fc._timed_cell(cell_fn, x, ref, warmup=1, trials=2)
    assert "abstain" not in out
    assert out["first_ms"] >= 0.0
    assert out["steady"]["p50"] == 0.1
    assert calls["time_cell"] == 1


def test_timed_cell_correctness_gate_still_abstains(monkeypatch):
    _install_fake_device_stack(monkeypatch)
    x = _FakeTensor(np.zeros((2, 4), dtype=np.complex64))
    ref = np.ones((2, 4), dtype=np.complex64)  # very different

    def cell_fn(xx):
        return xx  # zeros vs ones -> fails the 1e-4 gate

    out = fc._timed_cell(cell_fn, x, ref, warmup=1, trials=1)
    assert "abstain" in out
    assert "correctness gate failed" in out["abstain"]


def test_module_scope_has_no_device_stack_imports():
    # The deferral contract: module scope must not import torch or the
    # adapter (which pulls torch/triton) — only pure-Python + git_dirty.
    import ast
    tree = ast.parse((ROOT / "scripts" / "four_cell_ablation.py")
                     .read_text())
    top = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.append(node.module)
    assert "src.xpu.adapter_hybrid" not in top
    assert "src.xpu.error_metric" not in top
    assert "torch" not in top
