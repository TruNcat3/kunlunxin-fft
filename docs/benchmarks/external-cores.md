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
- **unavailable** — does not run on this hardware (wrong device/API
  backend), regardless of license.
- **license-blocked** — technically usable elsewhere, but its license
  forbids the use we would need *here* (e.g. copyleft linking into this
  repository). Hardware unavailability and license blocking are recorded
  separately; an item can be both.

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
| vkFFT | https://github.com/DTolm/VkFFT | 2.x | **MIT** (license is fine for this repo) | **unavailable** — Vulkan compute; no P800 Vulkan backend/driver on this machine |
| heFFTe | https://github.com/icl-utk-edu/heffte | 2.4 | **BSD-3-Clause** | **unavailable** — distributed CPU/CUDA/HIP backends; no XPU/P800 target |
| XLA reference FFT (`xla::reference::FFT`) | https://github.com/openxla/xla | 2025 | Apache-2.0 | **reference-only** — CPU reference implementation for correctness cross-checks |
| FlashFFT / Merril & FlashFFTTranspose | paper code | 2018 | mixed academic | **reference-only** — NVIDIA-specific algorithms; no P800 backend; no maintained host API |

Within the recorded search scope (the probe commands in table 1 plus a
review of the public repositories listed in table 2, as of 2026-10-16)
we found **no** public Kunlunxin/P800 FFT kernel implementation under
any license; this is a bounded statement about that search, not a claim
that none can exist. The vendor's own FFT stack is distributed only
inside the closed torch_xmlir wheels (table 1).

## 3. Canary measurements for each reachable candidate

The review asked for one minimal canary per candidate:

| candidate | canary | result |
|---|---|---|
| `torch.fft.fft` (vendor full transform) | `scripts/run_batch_release.py` `vendor` rows on all focus shapes | works for (N≤2048, B≤64) focus shapes; N≥4096 rejected by the runtime (`NOT IMPLEMENTED` in aten capture) — recorded in every summary manifest `vendor_probe` |
| `libxpuapi.so` length stubs | `nm -D` symbol inspection | symbols exist but no host-callable entry/headers → cannot canary at runtime; classified reference-only |
| PocketFFT | not built (CPU-only, would not exercise the device) | classification-only, no runtime canary |
| FFTW | not linked (GPL-2.0-or-later copyleft) | classification-only |
| vkFFT / heFFTe | not built (no P800 Vulkan/XPU backend; licenses are MIT/BSD-3 and fine) | classification-only |
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

## 5. 从 canary 到 exact-shape 晋升（composition table）

`config/external_cores.json` 中 `a1p1_materialized_subtransform` 的 `selector_status`
现在记录两段事实：canary 结论与**精确 shape 晋升**。晋升不改
`config/selector_table.json`，而是写入独立的 `config/composition_table.json`，由
`scripts/promote_composition.py` 执行。契约（运行时与晋升时各校验一遍）：

- **精确 key**：`N × batch × direction=dit × precision=fp32`，当前仅 `2048×16`；
  非 c2c 或非该 shape 一律走原 selector 路径（fail-closed，不是通配扩展）；
- **2% incumbent guard**：A1P1 对 A0P0 的提升从 evidence 的 protocol-session
  pooled p50 盖章，load 时按同一数字重算（误差 ≤1e-6）；负点（`N=1024,B=16/64`）
  拒绝晋升并原样保留为证据；
- **session 独立性**：≥3 session 且含 `role=confirmation` 独立确认、不同 pid、
  同一 commit、`all_sessions_same_commit`；
- **正确性**：evidence `err_max ≤ 1e-4`，abstain 拒绝；
- **绑定**：evidence `live_env_hash`/`profile_hash` 与晋升时 live 环境一致；
- **capability 注册**：`contract_id` 必须在本文件的 registry 中存在；
- **防篡改**：表带 `result_sha256`（entries 规范化 JSON 求哈希），坏表在 load 时
  抛 `ConfigError`（loud，不静默降级）。

归因不升格：entry 的 `executed=run_hybrid`、`flow_realization=decomposition-composition`、
`processing_unit` 含 composition —— 这是 host-API 组合（框架分解 + vendor 子块整变换
API），**不是**片上驻留单核，也不代表 vendor 内核 counter。

下一页：[工程结构](../development/repository-layout.md)。
