#!/usr/bin/env python3
"""Phase-0 smoke gate: verify the launch path cannot silently no-op.

The audit (migration convergence plan, P0) found that a Triton launch outside a
device context is a *silent no-op* and that a checker printing a max-error on an
untouched (all-zero) buffer would see 0.0 and wrongly report success. This gate:

1. Requires the device stack; exits with a clear prerequisite error otherwise.
2. Writes a **non-zero sentinel** input, runs the FFT, and asserts the output
   **changed** from the input and is **finite** — a zero error on an untouched
   buffer is not accepted.
3. Checks the output has non-trivial energy (not all-zero, not a copy of input).
4. Exercises negative input paths (wrong rank, wrong dtype, wrong device,
   non-contiguous, non-power-of-two) and asserts each raises a clear error.
5. Prints the target device info.

Exit codes: 0 = verified launch; 2 = prerequisite missing (no device stack);
1 = smoke/negative-test failure.

Usage: CUDA_VISIBLE_DEVICES=6 python3 scripts/smoke.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- prerequisite gate (exit 2, never silently pass) ---
try:
    import torch
    import torch_xmlir  # noqa: F401
except Exception as e:
    print(f"PREREQUISITE MISSING: cannot import torch/torch_xmlir: {e}", file=sys.stderr)
    sys.exit(2)

from src.xpu import device_name, fft_dif, fft_dit, fft_dit_fused, xpu_context  # noqa: E402
from src.xpu.context import device_ready  # noqa: E402

FAILURES = []


def check(cond, msg):
    if cond:
        print(f"  ok   {msg}")
    else:
        print(f"  FAIL {msg}")
        FAILURES.append(msg)


def expect_raises(fn, exc_types, label):
    try:
        fn()
    except exc_types as e:
        print(f"  ok   {label} -> {type(e).__name__}: {str(e)[:70]}")
        return
    except Exception as e:  # wrong exception type is still a failure
        print(f"  FAIL {label} raised {type(e).__name__}: {e}")
        FAILURES.append(f"{label}: wrong exception {type(e).__name__}")
        return
    print(f"  FAIL {label} did not raise")
    FAILURES.append(f"{label}: no exception")


def main() -> int:
    if not device_ready():
        print("PREREQUISITE MISSING: device_ready() is False "
              "(torch_xmlir not importable or torch.cuda unavailable)", file=sys.stderr)
        return 2

    print(f"target device: {device_name()}")

    # --- 1. non-zero sentinel: output must CHANGE and be FINITE ---
    print("== sentinel launch (non-zero input, output must change) ==")
    n = 1024
    with xpu_context(0):
        # deterministic non-zero sentinel: distinct value per element
        base = torch.arange(1.0, n + 1.0, device="cuda")
        x = torch.complex(base, -base)
        x_before = x.clone()
        y = fft_dit(x, block=1024)

        check(y.shape == x.shape, "output shape preserved")
        check(bool(torch.isfinite(y.real).all()) and bool(torch.isfinite(y.imag).all()),
              "output is finite")
        changed = not torch.equal(y.real, x_before.real) or not torch.equal(y.imag, x_before.imag)
        check(changed, "output differs from input (kernel actually ran)")
        energy = float(y.real.abs().sum() + y.imag.abs().sum())
        check(energy > 0.0, f"output has non-zero energy ({energy:.3e})")
        # a real FFT of a non-constant input concentrates energy; the DC bin
        # must equal the sum of the inputs
        dc = y[0]
        want = complex(float(base.sum()), float((-base).sum()))
        dc_ok = abs(complex(dc.real.item(), dc.imag.item()) - want) < 1e-2 * max(1.0, abs(want))
        check(dc_ok, "DC bin == sum of inputs (numeric spot check)")

        # DIF and fused paths also produce a changed, finite output
        y2 = fft_dif(x, block=1024)
        check(bool(torch.isfinite(y2.real).all()), "DIF output finite")
        check(not torch.equal(y2.real, x_before.real), "DIF output differs from input")
        y3 = fft_dit_fused(x, block=1024, tile=64)
        check(bool(torch.isfinite(y3.real).all()), "fused output finite")
        check(not torch.equal(y3.real, x_before.real), "fused output differs from input")

    # --- 2. negative input paths must raise clearly ---
    print("== negative input paths (must raise) ==")
    good = torch.complex(torch.ones(16, device="cuda"), torch.zeros(16, device="cuda"))
    expect_raises(lambda: fft_dit(good.reshape(4, 4)), ValueError, "wrong rank (2-D)")
    expect_raises(lambda: fft_dit(good.real), TypeError, "wrong dtype (float32)")
    # non-contiguous: take a strided view
    big = torch.complex(torch.ones(32, device="cuda"), torch.zeros(32, device="cuda"))
    expect_raises(lambda: fft_dit(big[::2]), ValueError, "non-contiguous view")
    # non-power-of-two length
    nonpot = torch.complex(torch.ones(12, device="cuda"), torch.zeros(12, device="cuda"))
    expect_raises(lambda: fft_dit(nonpot), ValueError, "non-power-of-two length")
    # wrong device (CPU tensor)
    cpu = torch.complex(torch.ones(16), torch.zeros(16))
    expect_raises(lambda: fft_dit(cpu), ValueError, "wrong device (cpu)")

    # --- summary ---
    if FAILURES:
        print(f"\nSMOKE FAILED ({len(FAILURES)}):")
        for m in FAILURES:
            print(f"  - {m}")
        return 1
    print("\nSMOKE PASSED: launch path verified, negative inputs rejected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
