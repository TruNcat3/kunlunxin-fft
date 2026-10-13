# Architecture

Kunlunxin-FFT is split into a **device path** (Triton, runs on the P800) and a
**host-side reference** (C++, runs on the CPU and is unit-tested with no device). The
device path is the implementation; the C++ is the specification the index/twiddle math is
checked against.

```
kunlunxin-fft/
├── config/                 # device profile, sizes, design space, test matrix
├── include/butterfly/      # host-side C++ reference (header-only)
│   ├── bit_reverse.hpp     #   bit-reversal + power-of-two helpers
│   ├── twiddle.hpp         #   exp(-2*pi*i*idx/N) addressing
│   └── fft.hpp             #   radix-2 DIT/DIF stage + full FFT
├── src/
│   ├── xpu/                # DEVICE PATH (Triton -> XMLIR -> XPU)
│   │   ├── kernels.py      #   bitrev / dit_stage / dif_stage Triton kernels
│   │   └── runner.py       #   Python host driver, twiddles, timing
│   └── host/               # HOST-SIDE REFERENCE (CPU, g++ unit test)
│       └── fft_check.cpp   #   verifies DIT/DIF vs naive DFT
├── scripts/
│   ├── init.sh             #   environment check
│   ├── build.sh            #   g++ unit test + Triton cache warm
│   ├── warm_triton.py      #   pre-compile device code
│   ├── test_correctness.py #   device correctness vs numpy
│   └── bench.py            #   benchmark vs torch.fft.fft
├── tests/test_fft.py       # pytest-style correctness entry point
└── results/p800/           # published benchmark data
```

## Device path (`src/xpu/`)

### Public layout vs internal lowering (planar SoA)

The **public API accepts and returns `complex64`** (interleaved, the usual torch
layout). **Planar SoA is an internal kernel ABI only**, not the public contract: the
Triton XPU backend has **no `complex64` argument type** (it raises
`KeyError: 'complex64'` in `type_canonicalisation_dict`), so every kernel takes two
`float32` pointers for data (`Xr`, `Xi`) and two for the twiddle table (`Wr`, `Wi`).
`fft_dit`/`fft_dif` convert at the boundary (`_as_so` / `_from_so`). Callers never see
planar. This is the same SoA convention real GPU FFTs use.

### Input validation and device context (Phase 0)

Every public entry point runs `validate_input` (rank-1, complex64, contiguous, on the
XPU device, power-of-two length) and `validate_twiddle` before any launch, so a wrong
semantic input raises instead of silently entering an incorrect path. Every device
launch runs inside `xpu_context()`, the single central context manager that establishes
(and restores) the device context — a launch outside it is a silent no-op on this stack
(see [Device Stack](device-stack.md)).

### The kernels (`kernels.py`)

All butterfly kernels share one index mapping, which is also the specification the C++
reference implements. For a stage with butterfly half-span `n` and twiddle stride
`M = N / (2n)`, for `k` in `[0, N/2)`:

```
j = k % n
i = ((k // n) * 2n) + j
pair (i, i+n), twiddle W[j * M]
```

* **`bitrev_kernel`** — out-of-place bit-reversal permutation.
  `Y[i] = X[bit_reverse(i)]`. `_bit_reverse` reverses the low `BITS` bits in an unrolled
  loop (`BITS` is a `constexpr`, so the loop is compile-time).

* **`dit_stage_kernel`** — one radix-2 decimation-in-time butterfly stage.
  `a = X[i]; b = X[i+n] * W[j*M]; Y[i] = a + b; Y[i+n] = a - b`.
  With `BITREV_INPUT` set, the **first** stage loads `X[bit_reverse(i)]` **and**
  `X[bit_reverse(i+n)]`, folding the input permutation into stage 0. Both indices must be
  bit-reversed — reversing only the first index is incorrect.

* **`dif_stage_kernel`** — one radix-2 decimation-in-frequency butterfly stage.
  `a = X[i]; b = X[i+n]; Y[i] = a + b; Y[i+n] = (a - b) * W[j*M]`.
  With `BITREV_OUTPUT` set, the **last** stage writes through `bit_reverse(i)` and
  `bit_reverse(i+n)`, folding the output permutation into the final stage.

* **`dit_fused_kernel`** (M1) — fuses the first `log2(TILE)` block-local DIT stages.
  For `n < TILE` a pair `(i, i+n)` never crosses a tile boundary, so each program owns one
  tile and applies those stages **in place** with only global-pointer `tl.load`/`tl.store`
  (no SSA gather). A `tl.debug_barrier()` between stages is required for correctness.
  Used by `fft_dit_fused`.

For the M0 baseline (`fft_dit`/`fft_dif`), a full FFT is `log2(N)` stage launches with
the permutation folded into the first (DIT) or last (DIF) stage — no separate pass.

### Driver (`runner.py`): the two flow candidates

| Flow id | Entry point | Launches | Description |
|---|---|---|---|
| `baseline-global-stage` (M0) | `fft_dit` / `fft_dif` | `log2(N)` | One global round trip + one launch per stage. The **compatibility baseline**, not a performance claim. |
| `local-fused-tile` (M1) | `fft_dit_fused` | `1 + 1 + log2(N) - log2(TILE)` | Explicit bit-reversal, then a fused local micro-FFT over `TILE`, then the remaining M0 stages. |

Both split a complex64 tensor into planar views, build the twiddle tables once per size
(`twiddle_tables`), and run inside `xpu_context()`. `time_call` reports median
synchronized wall time; `time_call_detailed` adds randomized trials and a separate
device-event kernel-time estimate.

### Why radix-2, and the baseline's limits

This milestone is correctness-first. Each M0 stage reads global memory, writes global
memory, and is its own kernel launch — memory- and launch-bound (see
[Results](results.md)). The cuButterfly method's contribution (blocking, register
residency, fused stages) is the identified path forward. M1 implements the *first* step
(static local micro-FFT) and is measured below; register/UB residence (M2) is the next
step and is currently limited by one indexing form the tested Triton lowering does not
generate — scoped precisely in [Device Stack](device-stack.md), not a claim that the
method is impossible.

## Host-side reference (`include/`, `src/host/`)

`include/butterfly/` is a header-only, dependency-free C++17 mirror of the index and
twiddle math: `bit_reverse`, `twiddle_table`, `dit_stage`, `dif_stage`, `fft_dit`,
`fft_dif`. `src/host/fft_check.cpp` verifies both FFTs against a naive `O(N^2)` DFT at
lengths up to 4096 (rel err ~1e-13) and is run by `scripts/build.sh` with **no device
present**.

This C++ is **not** the device path. It exists so the butterfly layout is auditable and
unit-testable independently of the P800 and of Triton. The device path is Triton only.
