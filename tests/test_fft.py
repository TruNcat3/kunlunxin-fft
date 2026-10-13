"""Correctness tests: Triton XPU FFT vs torch.fft (vendor baseline)."""

import math
import sys

import numpy as np
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.xpu import device_ready, fft_dif, fft_dit  # noqa: E402


def max_err(a: torch.Tensor, np_ref: np.ndarray) -> float:
    """Max abs error against a numpy complex128 reference."""
    ar = a.real.cpu().numpy().astype(np.float64)
    ai = a.imag.cpu().numpy().astype(np.float64)
    return float(np.maximum(np.abs(ar - np_ref.real), np.abs(ai - np_ref.imag)).max())


def max_mag(np_ref: np.ndarray) -> float:
    return float(np.abs(np_ref).max())


def make_input(n: int, seed: int):
    torch.manual_seed(seed)
    x = torch.complex(
        torch.randn(n, dtype=torch.float32, device="cuda"),
        torch.randn(n, dtype=torch.float32, device="cuda"),
    )
    host = x.real.cpu().numpy() + 1j * x.imag.cpu().numpy()
    return x, np.fft.fft(host)


def test_bitrev_only():
    from src.xpu import bit_reverse_permute

    n = 64
    x = torch.complex(
        torch.arange(n, dtype=torch.float32, device="cuda"),
        torch.arange(n, dtype=torch.float32, device="cuda") * 0.5,
    )
    y = bit_reverse_permute(x)
    bits = n.bit_length() - 1
    expect = torch.tensor(
        [int(format(i, f"0{bits}b")[::-1], 2) for i in range(n)],
        dtype=torch.int64,
        device="cuda",
    )
    xr = x.real.to(torch.int64)
    yr = y.real.to(torch.int64)
    assert torch.equal(yr, xr[expect]), "bit-reversal mismatch"
    print("bitrev OK")


def test_dit_sizes(sizes=(8, 16, 64, 256, 1024, 4096, 65536, 1048576)):
    for n in sizes:
        x, ref = make_input(n, 0)
        got = fft_dit(x)
        e = max_err(got, ref)
        rel = e / max_mag(ref)
        status = "OK " if rel < 1e-4 else "FAIL"
        print(f"  dit  N={n:8d} maxerr={e:.3e} rel={rel:.3e} {status}")
        assert rel < 1e-4, f"DIT failed at N={n}"


def test_dif_sizes(sizes=(8, 16, 64, 256, 1024, 4096, 65536)):
    for n in sizes:
        x, ref = make_input(n, 1)
        got = fft_dif(x)
        e = max_err(got, ref)
        rel = e / max_mag(ref)
        status = "OK " if rel < 1e-4 else "FAIL"
        print(f"  dif  N={n:8d} maxerr={e:.3e} rel={rel:.3e} {status}")
        assert rel < 1e-4, f"DIF failed at N={n}"


def test_known_signals():
    """Impulse -> flat spectrum; single tone -> one bin."""
    n = 256
    impulse = torch.zeros(n, dtype=torch.complex64, device="cuda")
    impulse[0] = 1.0
    got = fft_dit(impulse)
    ones = np.ones(n, dtype=np.complex128)
    assert max_err(got, ones) < 1e-5, "impulse failed"

    k0 = 5
    ang = 2.0 * math.pi * k0 * torch.arange(n, device="cuda", dtype=torch.float32) / n
    tone = torch.complex(torch.cos(ang), torch.sin(ang))
    got = fft_dit(tone)
    host = tone.real.cpu().numpy() + 1j * tone.imag.cpu().numpy()
    ref = np.fft.fft(host)
    assert max_err(got, ref) < 1e-4, "tone failed"
    print("known signals OK")


if __name__ == "__main__":
    if not device_ready():
        print("no XPU device available, skipping")
        sys.exit(0)
    test_bitrev_only()
    test_dit_sizes()
    test_dif_sizes()
    test_known_signals()
    print("all correctness tests passed")
