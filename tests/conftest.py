"""Pytest configuration: register markers and provide device gating.

Host-only tests (``tests/test_host.py``) must run in ordinary CI with no
vendor stack. Hardware tests are marked ``hardware`` and skip cleanly when
torch/torch_xmlir are absent or no XPU is reachable (migration convergence
plan Phase 0 deliverable 4, P2).
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "hardware: requires the vendor xpytorch stack and a reachable P800 XPU",
    )


def _device_available() -> bool:
    try:
        import torch  # noqa: F401
        import torch_xmlir  # noqa: F401
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


@pytest.fixture(scope="session", autouse=True)
def _require_device_for_hardware_tests(request):
    """Skip the whole test module when it is a hardware test and no device."""
    if request.node.get_closest_marker("hardware") and not _device_available():
        pytest.skip("hardware test skipped: vendor stack / XPU not available")
