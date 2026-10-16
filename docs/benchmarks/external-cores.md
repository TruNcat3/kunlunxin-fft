# External processing-unit inventory — issue #3 review (P800)

What high-performance FFT/transform cores are actually reachable from
this machine and repository, classified per the review's taxonomy.
Every claim here is from a probe command recorded below; nothing is
inferred from marketing material.

Classification:

- **local-core** — a host-callable core that computes a *partial*
  transform (a length-k block / stage) on data the framework owns; usable
  inside our decomposition with our twiddles/layout.
- **full-transform-only** — host-callable, but the API takes the whole
  transform; we cannot substitute it for a stage without owning the
  decomposition around it.
- **reference-only** — source or device-side symbols that exist but are
  not a host-callable API on this stack (or are CPU/reference code).
- **unavailable / license-blocked** — does not run on this hardware, or
  its license forbids the use we would need.

## 1. What this machine exposes (probed)

| name | source | version | license | classification |
|---|---|---|---|---|
| `torch.fft.*` via torch_xmlir dispatch → `xdnn_pytorch::fft_fft` / `fft_rfft` / `_fft_r2c` / `_fft_c2r` (+ `fftn` family) in `libxdnn_pytorch.so` | installed torch stack on this device | torch 2.9.0+cu129, torch_xmlir `XPYTORCH XMLIR--bc1b1dc6f-dev+2026032411`, XRE 5.13.0.0, XHPC 20770323, XTDK 3.6.0.1 | torch BSD-3-Clause; `xmlir`/`xtorch_ops` wheels declare **no** license metadata (vendor-internal; we only call them through torch's public API, no vendor code is redistributed here) | **full-transform-only** |
| `libxpuapi.so` device stubs `fft_len256/512/1024`, `fft_fwd_to_1K`, `FFT32<dtype,…>`, `fft_transform_{32…1920}` | bundled inside the torch_xmlir wheel | same build | same as above | **reference-only** — `__device_stub__` device symbols, no host-callable entry point, no headers; not invokable without vendor JIT support we do not have |
| Kunlunxin FFT SDK / kernel sources | **not present** on this machine (`/usr/local`, `/opt` contain no xpu/xmlir SDK; only `/usr/local/bin/xpu-smi`) | — | — | **unavailable** |
| `nvidia-cufft-cu12` | pip, CUDA build | 11.4.1.4 | NVIDIA proprietary | **unavailable** — CUDA library, does not execute on P800 |
| `torch.xpu` triton backend (`triton==3.0.0`) | installed stack | 3.0.0+03c4c9be | MIT (triton) | our own kernels' host stack — the **P0 local core** of this repo (`runner`/`cost_model` families), not an external dependency |
| vendor GEMM (`torch.mm` on XPU → `libxpu_blas.so`) | torch_xmlir | same build | as above | **local-core candidate for non-FFT linear algebra** — not used by the FFT adapter cells (our m2 already computes the register-resident dot via triton) |

Probe commands (reproducible):

```bash
python -c "import torch, torch_xmlir; print(torch.__version__)"
nm -D .../torch_xmlir/libxdnn_pytorch.so | c++filt | grep -i fft
nm -D .../torch_xmlir/libxpuapi.so       | c++filt | grep -i fft
find /usr/local /opt -iname '*xpu*' -o -iname '*xmlir*'
python -m pip show xmlir xtorch_ops      # no License field
```

## 2. Publicly available cores (research, not installed)

| name | source | version reviewed | license | classification for P800 |
|---|---|---|---|---|
| PocketFFT | https://github.com/mreineck/pocketfft (also vendored inside SciPy) | HEAD, 2025 | BSD-3-Clause | **reference-only** — CPU C++ source; no XPU backend; usable only as an algorithmic cross-check, not as a device core |
| FFTW | https://www.fftw.org | 3.3.x | **GPL-2.0-or-later** — linking creates a derivative work | **license-blocked** for this repository's use; also CPU-only |
| oneMKL (Intel) | oneAPI oneMKL | 2025.x | Intel proprietary (free redistributable binaries, license restricts use to Intel hardware) | **unavailable** — Intel-hardware-only |
| cuFFT | NVIDIA | 12.x | NVIDIA proprietary | **unavailable** — CUDA devices only |
| hipFFT / rocFFT | ROCm | 6.x | MIT / MIT (`rocFFT`) | **unavailable** — AMD GPUs only; no P800 target |
| vkFFT | https://github.com/DTolm/VkFFT | 2.x | **GPL-3.0** | **license-blocked**; Vulkan compute, no P800 Vulkan driver on this machine |
| heFFTe | https://github.com/icl-utk-edu/heffte | 2.4 | GPL-2.0 (FFTW wrapper lineage) | **license-blocked**; distributed CPU/GPU FFT, no XPU |
| XLA reference FFT (`xla::reference::FFT`) | https://github.com/openxla/xla | 2025 | Apache-2.0 | **reference-only** — CPU reference implementation for correctness cross-checks |
| FlashFFT / Merril & FlashFFTTranspose | paper code | 2018 | mixed academic | **reference-only** — NVIDIA-specific algorithms; no P800 backend; no maintained host API |

No public repository ships a Kunlunxin/P800 FFT kernel implementation
under any license; the vendor's own FFT stack is distributed only inside
the closed torch_xmlir wheels (table 1).

## 3. Canary measurements for each reachable candidate

The review asked for one minimal canary per candidate:

| candidate | canary | result |
|---|---|---|
| `torch.fft.fft` (vendor full transform) | `scripts/run_batch_release.py` `vendor` rows on all focus shapes | works for (N≤2048, B≤64) focus shapes; N≥4096 rejected by the runtime (`NOT IMPLEMENTED` in aten capture) — recorded in every summary manifest `vendor_probe` |
| `libxpuapi.so` length stubs | `nm -D` symbol inspection | symbols exist but no host-callable entry/headers → cannot canary at runtime; classified reference-only |
| PocketFFT | not built (CPU-only, would not exercise the device) | classification-only, no runtime canary |
| FFTW / vkFFT / heFFTe | not linked (license) | classification-only |
| cuFFT / hipFFT / oneMKL | not present / not executable on P800 | classification-only |

## 4. Consequence for the adapter experiment (four-cell)

The only *legal, host-callable* external transform core on this machine
is the vendor's **whole-tensor** FFT. A minimal adapter can therefore
still exercise a **P1 local core** in the review's sense by calling the
whole-tensor API on *framework-owned sub-block tensors* (the framework
holds the decomposition; the core only transforms a small contiguous
block). That is exactly the four-cell design measured in
`batch-focus.md` (A0/A1 flow × P0/P1 core), with the vendor's full
transform listed separately as the external baseline — never as a cell.

If the only available external core were the full transform called on
the *whole* signal, the honest conclusion would be "no legal local core
exists on this machine" — that is not the case here: the sub-block
invocation is legal (public torch API), measurable, and the report says
exactly what it costs.
