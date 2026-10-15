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
from .entry import (  # noqa: F401
    ExecutionPlan,
    ExecutionReport,
    execute_fft_plan,
    plan_fft,
)
from .selector import Selection, load_table, select  # noqa: F401

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
    "fft_dit_matrix": "runner",
    "fft_dif_matrix": "runner",
    "fft_dit_grouped": "runner",
    "fft_dif_grouped": "runner",
    "fft_dit_batch": "runner",
    "fft_dif_batch": "runner",
    "fft_dit_matrix_batch": "runner",
    "fft_dif_matrix_batch": "runner",
    "fft_dit_matrix_packet_batch": "runner",
    "fft_dit_radix4": "runner",
    "fft_dif_radix4": "runner",
    "fft_dit_radix4_batch": "runner",
    "fft_dif_radix4_batch": "runner",
    "local_matrix_kernel": "kernels",
    "local_matrix_batch_kernel": "kernels",
    "local_matrix_packet_kernel": "kernels",
    "tail_group_kernel": "kernels",
    "dit_stage_batch_kernel": "kernels",
    "dif_stage_batch_kernel": "kernels",
    "dif_stage2_batch_kernel": "kernels",
    "dit_stage4_kernel": "kernels",
    "dif_stage4_kernel": "kernels",
    "dit_stage4_batch_kernel": "kernels",
    "dif_stage4_batch_kernel": "kernels",
    "bitrev_batch_kernel": "kernels",
    "bitrev_interleaved_batch_kernel": "kernels",
    "log2_exact": "runner",
    "run_fft": "entry",
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
    "ExecutionPlan",
    "ExecutionReport",
    "Manifest",
    "SizeMatrix",
    "Selection",
    "build_manifest",
    "execute_fft_plan",
    "git_commit",
    "git_dirty",
    "load_table",
    "plan_fft",
    "select",
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
    "fft_dit_matrix",
    "fft_dif_matrix",
    "fft_dit_grouped",
    "fft_dif_grouped",
    "fft_dit_batch",
    "fft_dif_batch",
    "fft_dit_matrix_batch",
    "fft_dif_matrix_batch",
    "fft_dit_matrix_packet_batch",
    "fft_dit_radix4",
    "fft_dif_radix4",
    "fft_dit_radix4_batch",
    "fft_dif_radix4_batch",
    "local_matrix_kernel",
    "local_matrix_batch_kernel",
    "local_matrix_packet_kernel",
    "tail_group_kernel",
    "dit_stage_batch_kernel",
    "dif_stage_batch_kernel",
    "dif_stage2_batch_kernel",
    "dit_stage4_kernel",
    "dif_stage4_kernel",
    "dit_stage4_batch_kernel",
    "dif_stage4_batch_kernel",
    "bitrev_batch_kernel",
    "bitrev_interleaved_batch_kernel",
    "log2_exact",
    "run_fft",
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
