# Results

All numbers below were measured on a single **Kunlunxin P800 OAM** (12 clusters x 64
cores, 96 GiB HBM, driver 515.58) with `block=1024`, device-only timings (kernel launches
+ `torch.cuda.synchronize`, excluding host-side twiddle-table construction). The
reference is **numpy** `np.fft.fft` (CPU, complex128) — not the vendor `torch.fft.fft`,
which **only supports N <= 2048** on this device and therefore cannot validate the larger
sizes.

Correctness threshold: `rel_err < 1e-4`. Every size below passes for both `fft_dit` and
`fft_dif`.

## Correctness and timing

| N | dit_ms | dif_ms | vendor_ms (`torch.fft.fft`) | speedup (DIT vs vendor) | rel_err |
|---:|---:|---:|---:|---:|---:|
| 8 | 0.1854 | 0.1881 | 0.0202 | 0.11x | 6.5e-08 |
| 16 | 0.2286 | 0.2276 | 0.0218 | 0.10x | 8.0e-08 |
| 32 | 0.3783 | 0.2657 | 0.0138 | 0.04x | 2.2e-07 |
| 64 | 0.3497 | 0.3525 | 0.0174 | 0.05x | 1.3e-07 |
| 128 | 0.4128 | 0.4167 | 0.0228 | 0.06x | 1.7e-07 |
| 256 | 0.4693 | 0.4759 | 0.0400 | 0.09x | 2.1e-07 |
| 512 | 0.5389 | 0.5362 | 0.0815 | 0.15x | 2.1e-07 |
| 1024 | 0.6018 | 0.6058 | 0.2394 | 0.40x | 2.9e-07 |
| 2048 | 0.7184 | 0.7198 | 0.6468 | 0.90x | 3.2e-07 |
| 4096 | 0.9531 | 0.9548 | n/a | n/a | 3.4e-07 |
| 8192 | 1.4172 | 1.3691 | n/a | n/a | 3.9e-07 |
| 16384 | 2.2850 | 2.2168 | n/a | n/a | 3.7e-07 |
| 32768 | 4.7132 | 4.5661 | n/a | n/a | 4.6e-07 |
| 65536 | 9.4302 | 9.1163 | n/a | n/a | 4.2e-07 |
| 131072 | 19.4358 | 18.8277 | n/a | n/a | 4.4e-07 |
| 262144 | 40.2743 | 39.0281 | n/a | n/a | 5.5e-07 |
| 524288 | 82.0812 | 79.6325 | n/a | n/a | 5.4e-07 |
| 1048576 | 177.1069 | 172.7782 | n/a | n/a | 5.8e-07 |

Raw data: [`results/p800/bench_c2c.json`](../results/p800/bench_c2c.json).

## Honest reading

**This milestone is a correctness-first baseline, not a performance win.** Where the
vendor baseline is reachable (N <= 2048), the naive Triton radix-2 is **slower** —
0.04x–0.90x, geometric mean ~0.12x over the reachable sizes. The vendor `torch.fft.fft`
(xdnn) is a highly tuned native implementation; a per-stage Triton kernel with one
global-memory round trip and one launch per stage cannot compete with it on latency.

Two regimes tell different stories:

* **N <= 2048 (vendor reachable).** Launch overhead and global-memory traffic dominate
  our implementation. The gap narrows as N grows (0.04x at N=32 to 0.90x at N=2048),
  because fixed launch cost amortizes, but we never overtake the vendor in this range.
* **N >= 4096 (vendor unreachable).** `torch.fft.fft` fails outright
  (`[NOT IMPLEMENTED]` from `xdnn_pytorch/fft_fft.cpp`). This library is the **only**
  path, and it runs correctly to N=1048576 at rel err ~6e-7. **Coverage is the value
  proposition of this milestone**, not speed.

## Why it is slow, and the identified fix

The implementation is memory-bound: each of the `log2(N)` stages reads N complex values
from global memory, writes N back, and is a separate kernel launch. At N=1048576 that is
20 stages x (read + write) of 8 MB of complex data, i.e. hundreds of MB of global traffic
per transform — orders of magnitude more than a blocked implementation would move.

The fix is the cuButterfly mixed space-time dataflow: fuse stages into one kernel running
against on-chip storage (register/shared residency), so a transform touches global memory
twice instead of `2*log2(N)` times, and block the data dimension so the 12 clusters are
all occupied. **This is currently blocked on the toolchain**: the Triton 3.0.0 XPU build
rejects gather/scatter (`x[idx]`), which an in-place on-chip FFT needs. See
[Device Stack](device-stack.md) for the evidence and
[Architecture](architecture.md) for the design space that would consume the fix.

The benchmark is reproducible:

```bash
CUDA_VISIBLE_DEVICES=6 python3 scripts/bench.py --json results/p800/bench_c2c.json
```
