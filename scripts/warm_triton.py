#!/usr/bin/env python3
"""Pre-warm the Triton device-code cache with a small transform.

Called by scripts/build.sh. The first Triton launch compiles the kernel to
XMLIR and then to XPU instructions, which can take minutes; subsequent runs
read the cache.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch_xmlir  # noqa: E402,F401

from src.xpu import fft_dif, fft_dit  # noqa: E402


def main() -> None:
    n = 8
    torch.manual_seed(0)
    x = torch.complex(
        torch.randn(n, dtype=torch.float32, device="cuda"),
        torch.randn(n, dtype=torch.float32, device="cuda"),
    )
    fft_dit(x)
    fft_dif(x)
    print("triton cache warm")


if __name__ == "__main__":
    main()
