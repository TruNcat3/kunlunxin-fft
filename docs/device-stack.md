# Device Stack

This page records, with the experiments that support them, which device-code paths are
**executable** on the Kunlunxin P800 in the reference container and which are **not**.
The reference environment is the vendor "xpytorch" stack: `torch` 2.9.0 + `torch_xmlir`
+ `flag_gems`, inside the container that also ships the XRE runtime
(`libxpurt.so`, `libxpucuda.so`, `libcudart.so`) and a `triton` 3.0.0 with an `xpu`
backend.

Hardware under test: 8x Kunlunxin **P800 OAM**, driver 515.58, 12 clusters x 64 cores,
256 KiB shared memory per cluster, 96 GiB HBM per card. `CUDA_VISIBLE_DEVICES=6` is used
for every measurement (the environment default is `1,2`, and the device is left in a
recovered, idle state).

## The one executable path

**Triton -> XMLIR -> XPU instructions.** Importing `torch` and `torch_xmlir` retargets
`device='cuda'` to the P800 (`torch.xpu` is not enabled). After that:

```python
import triton, triton.language as tl
from triton.runtime.driver import driver
driver.active.get_current_target()   # GPUTarget(backend='xpu', arch=3, warp_size=1)
driver.active.utils.get_device_properties(0)
# {'max_shared_mem': 262144, 'multiprocessor_count': 12, 'device_model': 3, ...}
```

A `@triton.jit` kernel compiles to an XPU ELF (`CompiledKernel.asm['xpubin']`, md5-identical
to the cached `k.xpubin`) and launches. `get_xpu_spec(3, False)` returns `(12, 64)` — the
launch grid/cluster geometry must use `12, 64`.

**A critical runtime requirement.** Triton kernels that touch the device must run inside
`flag_gems.runtime.torch_device_fn.device(...)` (known-issues #11). Without that context a
launch is a silent no-op: the kernel never runs, output is untouched, and a checker that
only prints a max-error will see `0.0` and wrongly conclude success.

**`torch.cuda` properties are stubbed.** `torch.cuda.get_device_properties(0)` reports
`name='GPU'` and `multi_processor_count=8`, which is wrong (the real topology is 12
clusters). The Triton driver reports the true values; `device_name()` uses the Triton
driver, not torch.

## Dead end 1: nvcc / PTX

The P800 does not execute PTX. `nvcc` is not present in the reference container, and
even a PTX-carrying cubin cannot be run. This is consistent with the container's own
known-issues notes (#12: CUDA is not executable on this device). Ruled out.

## Dead end 2: clang-written CUDA device code -> xpubin

`triton/backends/xpu/` ships a `clang` (`xpu3/bin`) and an ELF loader, so it is
*possible* to write CUDA-like device code, compile it with the bundled clang to an
`xpubin`, and load it. This was pursued to the point of a working host-side launcher:

* The `xpubin` format is a flat binary: a 64-byte `struct xpu_kernel`
  (`code_addr`, `code_byte_size`, `code_pc`, `param_dword_size`, `hash`, `name`,
  `rt_private`, `printf_buffer_offset`, with 16-bit `type`/`place` bit-fields) plus a
  `checksum` (sum of the payload as **signed** char).
* The launch sequence is `xpu_launch_config(12, 64, NULL)` ->
  `xpu_launch_argument_set(...)` -> `xpu_launch_async(...)` -> `xpu_wait(stream)`.
  `xpu_launch_argument_set` lays out `{ptr x@0, ptr y@8, n@16, gridX@20, gridY@24, gridZ@28}`.
* Early bugs were found and fixed: passing host *variable addresses* instead of *values*
  in `kernel_params` (caused a device hang; `xpu_wait` blocked forever), and
  `kernel_consts` needing the BLOCK constexpr triple `{3,4,256}`.
* With those fixed, `xpuLaunchKernel` (mangled `_Z15xpuLaunchKernelPKcPviiiiiS1_PS1_S2_S2_`,
  only present in `xpu3/lib/liblaunch.a`) returns **rc=0** in a bare C++ process, but the
  kernel is a **silent no-op** — the output buffer stays zero. `xpu_wait` also hangs if no
  stream was pre-created; with a pre-created stream it returns 0 immediately.
* `xpu_create_cl_func` / `xpu_create_sd_func` / `xpu_pointer_get_attrs` are **declared in
  `xpu3/include/xpu/runtime.h` but not exported by any available library** — the header
  is ahead of the shipped binaries.

**The decisive experiment.** The *identical* C++ call sequence (`xpu_launch_config` /
`argument_set` / `async`), byte-for-byte, was run in two processes:

1. A **bare** C++ process — `rc=0`, kernel silent no-op.
2. A process with **torch already initialized** (the sequence loaded via `librunseq.so`
   + `ctypes` after `import torch, torch_xmlir`) — **maxerr 0.0, correct**.

An `LD_PRELOAD` shim confirmed the two processes issue the same calls with the same
arguments, so the difference is **not** the parameter or call sequence. It is torch's
in-process runtime state. Neither `cuInit` + `cuDevicePrimaryCtxRetain` + `cuCtxSetCurrent`
nor pre-loading XRE's `libxpucuda.so.1.0` restores it in the bare process. `dlopen`-ing
the libraries in torch's load order with `RTLD_GLOBAL` makes `xpu_malloc` /
`xpu_current_device` / `xpu_device_count` return rc=0 but leave the out-params unwritten
(nil / -1) — they resolve to stubs.

**Conclusion.** A bare C++ host cannot reliably launch P800 device code; the loader needs
torch's initialized runtime. The device path is therefore Python (Triton), and C++ is kept
only as a host-side reference. This is the finding that determines the repository's shape.

## Dead end 3: XHPC / xpucc SDK

The Kunlunxin XHPC / `xpucc` compiler SDK is not present in the reference container, so
the vendor's own device-code toolchain is unavailable. Ruled out by absence.

## Limitation 4: SSA gather/scatter form is not lowered (scoped, not "impossible")

The performance baseline is one global-memory round trip and one kernel launch per
stage (see [Results](results.md)). Register/UB residence (holding intermediate values
for several stages without touching memory) would express the butterfly as, per stage,

```text
new_tensor = where(is_low, a+b, (a-b)*w)   # built from tensor[idx] gathers
```

**Finding (scoped precisely, both canaries in `tests/test_fft.py`).** Two forms were
probed separately on the Triton 3.0.0 XPU build:

* **Fails — tensor subscript:** gathering from an SSA tensor via a tensor index,
  `Xt[idx_tensor]`, raises `CompilationError ... ValueError: Did you forget to add
  @triton.jit?` (`_builder` argument must be provided outside of JIT functions). Canary:
  `test_tensor_subscript_gather_blocked`, which **fails the moment a future toolchain
  accepts it** — the signal that this limitation no longer applies.
* **Works — pointer arithmetic:** `tl.load(X_ptr + idx_tensor)` with runtime integer
  tensor offsets lowers and produces correct values. Canary:
  `test_pointer_form_gather_lowers`.

So the blocker is **narrower than "gather is impossible"**: dynamic offsets through
pointers lower fine; only the *tensor-subscript SSA form* is rejected. This is a
statement about the current lowering of one indexing form on one shape, not a proof
that the architecture or the algorithm is impossible.

**What is distinguished:**

```text
algorithmically possible          yes — the block-local DIT stages are well defined
hardware-lowerable in this compiler  partial — pointer-form gather and pointer
                                   tl.load/tl.store: yes (canary-tested);
                                   tensor-subscript Xt[idx]: no (canary-tested)
implemented and measured          M1 local-fused-tile: yes (pointer form);
                                   register/UB residence: not yet
```

**Alternative legal forms to try before declaring a toolchain blocker** (ladder from
the migration convergence plan; step 1 is measured, steps 2–3 now look reachable given
the pointer-form result):

1. compile-time / static index tables for small tiles — **M1 `local-fused-tile` does
   this**: it fuses the first `log2(TILE)` block-local stages using only global-pointer
   `tl.load`/`tl.store`, and is correct at every tested size.
2. contiguous tile loads followed by register permutation — the load half is proven
   (pointer form works); the register-side pairing may be expressible with
   static reshape/slice/concat instead of `Xt[idx]` (untested, next round);
3. ping-pong registers or UB with explicit stage-local permutations;
4. multiple legal subgraphs connected by explicit boundaries (execution groups);
5. only then classify the remaining form as a toolchain blocker.

**Why M1 still regresses at large N (measured cause).** The M1 fused kernel updates a
tile **in place** across stages, which requires a `tl.debug_barrier()` between stages —
removing it produces races at every size (rel err ~0.9, confirmed). The mandatory
barrier plus the in-place store→load dependency serializes the memory pipeline that the
M0 out-of-place stages keep fully independent, so M1 wins where launches dominate
(`N <= 1024`, geometric mean ~1.6x) and loses where memory time dominates
(`N >= 4096`, geometric mean ~0.7x). Register/UB residence (M2) avoids this entirely —
values would never round-trip through memory, so no inter-stage fence exists. The only
part of M2 the current lowering rejects is the tensor-subscript SSA shape above; how
much of the register-side permutation can be rebuilt from the proven pointer form and
static reshape/slice ops is the concrete open question for the next round. This is the
specific, recorded gap between the current baseline and the full cuButterfly mixed
space-time dataflow.

## Dangerous-operation notes

* A kernel launched with wrong parameters can **hang the device** (`xpu-smi` shows 100%
  util). Recovery is `kill -9` on the offending process.
* Do **not** `pkill -f <script-name>` to kill such a process — the pattern matches your
  own shell command line and kills the shell. (This happened during this investigation.)
* `CUDA_VISIBLE_DEVICES` is honored by both the XRE runtime and `torch_xmlir`.
  `torch.cuda.set_device(3)` fails with "invalid device ordinal" when `3` is not in the
  visible list; select the device with the env var, not `set_device`.
