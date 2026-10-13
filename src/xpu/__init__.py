"""Kunlunxin P800 FFT device path (Triton -> XMLIR -> XPU instructions).

Torch-dependent symbols (kernels, runner) are imported lazily so that
host-only modules (``config``, ``candidates``) can be imported without the
vendor xpytorch stack. This lets host-only tests and manifest generation run
in ordinary CI; only ``fft_dit``/``fft_dif``/``bit_reverse_permute`` and the
kernels require a device.
"""

import importlib

# Host-only symbols that never need torch.
from .config import (  # noqa: F401
    ConfigError,
    DesignSpace,
    DeviceProfile,
    Manifest,
    SizeMatrix,
    build_manifest,
    git_commit,
    git_dirty,
)

# Torch-dependent symbols, resolved lazily on first attribute access.
_TORCH_ATTRS = {
    "bit_reverse_permute": "runner",
    "bitrev_kernel": "kernels",
    "dif_stage_kernel": "kernels",
    "dit_stage_kernel": "kernels",
    "dit_fused_kernel": "kernels",
    "device_name": "runner",
    "device_ready": "runner",
    "fft_dif": "runner",
    "fft_dit": "runner",
    "fft_dit_fused": "runner",
    "log2_exact": "runner",
    "time_call": "runner",
    "time_call_detailed": "runner",
    "twiddle_tables": "runner",
    "xpu_context": "context",
    "validate_input": "validate",
}

__all__ = [
    # host-only
    "ConfigError",
    "DesignSpace",
    "DeviceProfile",
    "Manifest",
    "SizeMatrix",
    "build_manifest",
    "git_commit",
    "git_dirty",
    # torch-dependent
    "bit_reverse_permute",
    "bitrev_kernel",
    "dif_stage_kernel",
    "dit_stage_kernel",
    "dit_fused_kernel",
    "device_name",
    "device_ready",
    "fft_dif",
    "fft_dit",
    "fft_dit_fused",
    "log2_exact",
    "time_call",
    "time_call_detailed",
    "twiddle_tables",
    "xpu_context",
    "validate_input",
]


def __getattr__(name):
    if name in _TORCH_ATTRS:
        module = importlib.import_module(f".{_TORCH_ATTRS[name]}", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
