"""Explicit XPU device context (migration convergence plan Phase 0, P0).

``docs/device-stack.md`` documents that Triton device launches must run inside
an established device context (the ``flag_gems`` known-issues note #11: without
it a launch is a *silent no-op* — output untouched, and a checker that only
prints a max-error sees 0.0 and wrongly concludes success). The audit found the
launcher never entered that context. This module provides the single central
context manager every device launch must use.

``flag_gems.runtime.torch_device_fn.device`` is an alias of ``torch.cuda.device``
on this stack: a context manager that exchanges the current CUDA (== XPU) device
index and restores it on exit. We pair it with an explicit
``torch.cuda.set_device`` so the intent is unambiguous even if the alias moves.
"""

from contextlib import contextmanager

import torch

try:
    import torch_xmlir  # noqa: F401  # retargets device='cuda' to the XPU
except ImportError:  # pragma: no cover - only present in the P800 container
    torch_xmlir = None

try:
    from flag_gems.runtime import torch_device_fn

    _device_cm = torch_device_fn.device
except Exception:  # pragma: no cover - flag_gems optional at import time
    _device_cm = None


def device_ready() -> bool:
    """True when an XPU device is addressable through the torch 'cuda' API."""
    return torch_xmlir is not None and torch.cuda.is_available()


def current_device_index() -> int:
    """Index of the currently selected torch 'cuda' (XPU) device."""
    return torch.cuda.current_device()


@contextmanager
def xpu_context(device: int = 0):
    """Establish (and restore) the XPU device context around a device section.

    Every Triton launch and every torch XPU op must run inside this context.
    It is a no-op-ish exchange when ``device`` is already current, but it makes
    the context explicit and guarantees restoration, so an exception inside the
    body cannot leak a changed device to the caller.
    """
    if not device_ready():
        raise RuntimeError(
            "XPU device stack not available: import torch + torch_xmlir failed or "
            "torch.cuda.is_available() is False. Install the vendor xpytorch stack."
        )
    prev = torch.cuda.current_device()
    if _device_cm is not None:
        with _device_cm(device):
            yield device
    else:  # pragma: no cover - fallback when flag_gems is absent
        torch.cuda.set_device(device)
        try:
            yield device
        finally:
            torch.cuda.set_device(prev)
