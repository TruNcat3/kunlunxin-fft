# Kunlunxin-FFT

**Kunlunxin-FFT is a Triton-mapped FFT operator library for the Kunlunxin P800.**
It is a migration instance of the [cuButterfly](https://github.com/TruNcat3/cuButterfly)
mixed space-time dataflow method onto a new class of hardware, in the same family as
[Ascend-FFT](https://github.com/TruNcat3/ascend-fft). This repository's primary goal is
to establish, in a public and reproducible experiment range, the *executable* device-code
path on the P800, record the paths that are not executable and why, and ship a
correctness-verified FFT with a measured baseline against the vendor stack.

The first milestone of this repository is a **correctness-first baseline**, not a
performance win. See [Results](results.md) for the measured numbers and an honest
reading of them.

## Relationship to cuButterfly

This project is a migration instance of the [cuButterfly](https://github.com/TruNcat3/cuButterfly)
method: it keeps the architectural paradigm of splitting both the *stage* dimension and
the *data* dimension across space and time, but it does not reuse CUDA kernels. Computation,
layout, storage, and synchronization are re-mapped to the P800's device model. Method
provenance and differences are credited to [cuButterfly](https://github.com/TruNcat3/cuButterfly);
this repository does not re-claim those methodological positions as its own.

Two orthogonal layers must be distinguished. **Compute-flow optimization** decides how
stages and data are blocked, reused, handed off, reordered, and pipelined. **Compute-core
optimization** decides which verified butterfly algorithm (radix-2, radix-4, ...) completes
the arithmetic inside each local block. **Hardware lowering** then decides the concrete
instructions and data paths. cuButterfly's subject is the compute flow; the radix-2 core used
here is one executable physical instance, not a fixed definition of the method.

## What this device stack actually allows

The single most important finding documented here is which device-code paths are
*executable* on the P800 in the reference container, and which are not. In short:

| Device-code path | Executable on P800? | Why |
|---|---|---|
| **Triton -> XMLIR -> XPU instructions** | **Yes** | The only working path. `triton` 3.0.0 with an `xpu` backend compiles at runtime and launches. |
| nvcc -> PTX -> ... | No | The P800 does not execute PTX. |
| clang-written CUDA device code -> xpubin | No | Compiles, but the XPU ELF loader needs torch's in-process runtime state; a bare C++ host cannot reproduce it (silent no-op). |
| XHPC / xpucc SDK | No | Absent from the reference container. |
| On-chip multi-stage fusion via gather/scatter | No | The Triton 3.0.0 XPU build rejects `x[idx]` gather, so an in-place shared-memory FFT cannot be expressed in SSA form. |

Full evidence, including the experiments that rule out each dead end, is in
[Device Stack](device-stack.md).

## Current capability

The table below is the current implementation's support contract and the experiment
range of this milestone; results outside the range should not be extrapolated.

| Transform | Precision | Length | Layout | Status |
|---|---|---|---|---|
| C2C forward (DIT) | fp32 complex | powers of two, `8..1048576` | planar split real/imag | Stable experimental path |
| C2C forward (DIF) | fp32 complex | powers of two, `8..65536` | planar split real/imag | Stable experimental path |

Complex data is carried **planar** (separate float32 real and imag buffers) because the
Triton XPU backend has no `complex64` argument type. There is no inverse transform, no
R2C/C2R, no batched API, and no stable C ABI in this milestone.

The device code is Triton; the host driver is Python. The C++ under `include/` and
`src/host/` is a **host-side reference** for the index and twiddle math, verified by a
g++ unit test with no device present — it is not the device path.

## Performance overview

Measured on one P800 OAM (12 clusters x 64 cores, 96 GiB HBM), `block=1024`, device-only
timings (kernel launches + device sync, excluding host-side twiddle construction). The
vendor baseline is `torch.fft.fft` (xdnn), which **only supports N <= 2048** on this
device. Full table in [Results](results.md).

| Regime | Baseline | Reading |
|---|---|---|
| N <= 2048 (vendor reachable) | `torch.fft.fft` | Naive Triton radix-2 is **slower** (0.11x–0.90x); launch and global-memory cost dominate. |
| N >= 4096 (vendor unreachable) | none | This library is the **only** path; it runs correctly to 1048576 at rel err ~6e-7. |

The value proposition of this milestone is **coverage, correctness, and a documented
executable path** — not speed. Closing the performance gap is exactly what the cuButterfly
mixed space-time dataflow (blocking, register residency, fused stages) is for, and the
specific blocker (no gather/scatter) is recorded in [Device Stack](device-stack.md).

## Getting started

```bash
# 1. environment (needs the vendor xpytorch stack: torch + torch_xmlir)
bash scripts/init.sh

# 2. build host-side C++ reference + unit test, warm Triton cache
bash scripts/build.sh

# 3. correctness on the device
CUDA_VISIBLE_DEVICES=6 python3 scripts/test_correctness.py --profile smoke

# 4. benchmark
CUDA_VISIBLE_DEVICES=6 python3 scripts/bench.py
```

See [Architecture](architecture.md) for how the pieces fit together, and
[Results](results.md) for the measured data. The implementation and measurement
roadmap is documented in the [Migration convergence plan](development/migration-convergence-plan.md).
