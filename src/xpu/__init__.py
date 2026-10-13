"""Kunlunxin P800 FFT device path (Triton -> XMLIR -> XPU instructions)."""

from .kernels import bitrev_kernel, dif_stage_kernel, dit_stage_kernel
from .runner import (
    bit_reverse_permute,
    device_name,
    device_ready,
    fft_dif,
    fft_dit,
    log2_exact,
    time_call,
    twiddle_tables,
)

__all__ = [
    "bit_reverse_permute",
    "bitrev_kernel",
    "dif_stage_kernel",
    "dit_stage_kernel",
    "device_name",
    "device_ready",
    "fft_dif",
    "fft_dit",
    "log2_exact",
    "time_call",
    "twiddle_tables",
]
