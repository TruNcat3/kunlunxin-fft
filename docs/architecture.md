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

### Split real/imaginary (planar) layout

The Triton XPU backend has **no `complex64` argument type** (it raises
`KeyError: 'complex64'` in `type_canonicalisation_dict`). Every kernel therefore takes
two `float32` pointers for data (`Xr`, `Xi`) and two for the twiddle table (`Wr`, `Wi`).
This is the same planar (SoA) convention real GPU FFTs use.

### The three building blocks (`kernels.py`)

All three kernels share one index mapping, which is also the specification the C++
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

A full FFT is `log2(N)` stage launches. There is no separate permutation pass; it is
folded into the first (DIT) or last (DIF) stage.

### Driver (`runner.py`)

`fft_dit(x)` / `fft_dif(x)` split a complex64 tensor into planar float32 views, build the
twiddle tables once per size (`twiddle_tables`, `exp(-2*pi*i*idx/N)` for `idx` in
`[0, N/2)`), then launch `log2(N)` stages ping-ponging between two buffers. `time_call`
reports the median of device-synchronized wall-clock samples.

The XPU device context is established by importing `torch` and `torch_xmlir`; after that
`device='cuda'` addresses the P800 and Triton's `xpu` backend compiles and launches.

### Why radix-2, and why one global round trip per stage

This milestone is correctness-first. Each stage reads global memory, writes global
memory, and is its own kernel launch. That is memory- and launch-bound — see
[Results](results.md). The cuButterfly method's contribution (blocking, register
residency, fused stages) is the identified path to close the gap, but it is **blocked on
this toolchain**: the Triton 3.0.0 XPU build rejects gather/scatter (`x[idx]`), so an
in-place shared-memory FFT cannot be expressed in SSA form. See
[Device Stack](device-stack.md).

## Host-side reference (`include/`, `src/host/`)

`include/butterfly/` is a header-only, dependency-free C++17 mirror of the index and
twiddle math: `bit_reverse`, `twiddle_table`, `dit_stage`, `dif_stage`, `fft_dit`,
`fft_dif`. `src/host/fft_check.cpp` verifies both FFTs against a naive `O(N^2)` DFT at
lengths up to 4096 (rel err ~1e-13) and is run by `scripts/build.sh` with **no device
present**.

This C++ is **not** the device path. It exists so the butterfly layout is auditable and
unit-testable independently of the P800 and of Triton. The device path is Triton only.
