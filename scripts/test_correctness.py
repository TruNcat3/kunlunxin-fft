#!/usr/bin/env python3
"""Run the declared correctness suite against numpy (CPU, complex128).

Reads config/test_matrix.json. The vendor torch.fft.fft is not used as the
reference because it only supports N <= 2048 on this device.

Usage:
    CUDA_VISIBLE_DEVICES=6 python3 scripts/test_correctness.py [--profile full]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch_xmlir  # noqa: E402,F401

from src.xpu import device_name, fft_dif, fft_dit  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_input(n: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.complex(
        torch.randn(n, dtype=torch.float32, device="cuda"),
        torch.randn(n, dtype=torch.float32, device="cuda"),
    )


def max_err(got: torch.Tensor, ref: np.ndarray) -> float:
    gr = got.real.cpu().numpy().astype(np.float64)
    gi = got.imag.cpu().numpy().astype(np.float64)
    return float(np.maximum(np.abs(gr - ref.real), np.abs(gi - ref.imag)).max())


def run_profile(profile: dict, block: int) -> bool:
    ok = True
    for n in profile["c2c"]["ns"]:
        x = make_input(n, seed=0)
        host = x.real.cpu().numpy() + 1j * x.imag.cpu().numpy()
        ref = np.fft.fft(host)
        mag = float(np.abs(ref).max())
        for name, fn in (("dit", fft_dit), ("dif", fft_dif)):
            got = fn(x, block=block)
            err = max_err(got, ref)
            rel = err / mag
            status = "OK" if rel < 1e-4 else "FAIL"
            if status == "FAIL":
                ok = False
            print(f"  {name:4s} N={n:>8} maxerr={err:.3e} rel={rel:.3e} {status}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="smoke", help="profile id from config/test_matrix.json")
    ap.add_argument("--block", type=int, default=1024)
    args = ap.parse_args()

    with open(os.path.join(ROOT, "config", "test_matrix.json")) as f:
        matrix = json.load(f)

    profile = matrix["profiles"].get(args.profile)
    if profile is None:
        sys.exit(f"unknown profile {args.profile!r}; known: {list(matrix['profiles'])}")

    print(f"device: {device_name()}")
    print(f"profile: {args.profile} (block={args.block})")
    all_ok = run_profile(profile, args.block)
    print("ALL PASSED" if all_ok else "FAILURES PRESENT")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
