# Results

## Measurement protocol (Phase 2)

All rows come from `scripts/bench.py`, which records a **manifest** (commit, dirty
state, profile hash, device, block, warmup, trials) and applies the protocol required by
the [migration convergence plan](development/migration-convergence-plan.md):

* **Randomized interleaved trials.** For each length, every candidate (M0 DIT, M0 DIF,
  M1 fused, vendor) is timed in a shuffled order each round, so drift across the run does
  not bias one candidate. Reported value is the **median** of 40 trials (`warmup=8`).
* **Two timing metrics, reported separately.** `api_ms` is synchronized wall time and
  includes Python dispatch, planar conversion, allocation, and sync. `kernel_ms` is
  device-event elapsed time (pure device). `api_ms` is the fair cross-candidate metric;
  `kernel_ms` confirms the same ordering.
* **DIT and DIF error columns are separate**, and both are checked against numpy
  (`np.fft.fft`, complex128) — not the vendor, which caps at `N <= 2048` here.
* **Vendor-unsupported rows are retained with the actual error string** (shown inline
  below), never dropped.

Hardware: single Kunlunxin **P800 OAM** (12 clusters x 64 cores, 96 GiB HBM, driver
515.58), `block=1024`, `CUDA_VISIBLE_DEVICES=6`. Manifest commit `ace8524`, profile hash
`a78b8679`. Raw rows: [`bench_phase2.json`](https://github.com/TruNcat3/kunlunxin-fft/blob/master/results/p800/bench_phase2.json)
/ [`bench_phase2.csv`](https://github.com/TruNcat3/kunlunxin-fft/blob/master/results/p800/bench_phase2.csv).

## Candidates

| Flow id | Entry | Description |
|---|---|---|
| `baseline-global-stage` (M0) | `fft_dit` / `fft_dif` | One global round trip + one launch per stage. **Compatibility baseline.** |
| `local-fused-tile` (M1) | `fft_dit_fused(tile=64)` | Explicit bit-reversal, then the first `log2(64)=6` block-local stages fused in one kernel, then the remaining M0 stages. |

The lowering-contract check ran before timing: both candidates are `legal` for
`residence=gm, layout=planar-soa, radix=2`; `selected_candidate` defaults to the
conservative M0 (`lowest fused_tile`) so M1 must prove itself by measurement.

## Timing (median api_ms; DIT and DIF errors separate)

The table below is **generated from the raw rows** by
`scripts/bench.py --markdown` (Phase 2 acceptance: regenerable without hand-editing);
the JSON/CSV are the source of truth.

| N | M0 dit | M0 dif | M1 fused | vendor | M0 vs vendor | M1 vs M0 | dit_err | dif_err |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.2103 | 0.2106 | 0.1368 | 0.0393 | 0.19x | **1.54x** | 6.5e-08 | 6.5e-08 |
| 16 | 0.2501 | 0.2493 | 0.1396 | 0.0389 | 0.16x | **1.79x** | 8.0e-08 | 5.7e-08 |
| 32 | 0.6317 | 0.6363 | 0.3758 | 0.0791 | 0.13x | **1.68x** | 2.2e-07 | 1.4e-07 |
| 64 | 0.3825 | 0.3792 | 0.1979 | 0.0365 | 0.10x | **1.93x** | 1.3e-07 | 1.6e-07 |
| 128 | 0.7491 | 0.7506 | 0.4355 | 0.0727 | 0.10x | **1.72x** | 1.6e-07 | 1.9e-07 |
| 256 | 0.5030 | 0.4988 | 0.3122 | 0.0518 | 0.10x | **1.61x** | 2.1e-07 | 2.0e-07 |
| 512 | 0.6793 | 0.6670 | 0.5258 | 0.1150 | 0.17x | **1.29x** | 2.1e-07 | 2.3e-07 |
| 1024 | 0.6355 | 0.6373 | 0.5228 | 0.2514 | 0.40x | **1.22x** | 2.9e-07 | 3.7e-07 |
| 2048 | 0.8462 | 0.8527 | 0.8647 | 0.6580 | 0.78x | 0.98x | 3.2e-07 | 2.8e-07 |
| 4096 | 1.0271 | 1.0124 | 1.1142 | n/a | n/a | 0.92x | 3.4e-07 | 3.3e-07 |
| 8192 | 1.5509 | 1.5114 | 2.0400 | n/a | n/a | 0.76x | 3.9e-07 | 3.4e-07 |
| 16384 | 2.3451 | 2.2272 | 3.5932 | n/a | n/a | 0.65x | 3.7e-07 | 3.3e-07 |
| 32768 | 4.6835 | 4.5431 | 7.0239 | n/a | n/a | 0.67x | 4.6e-07 | 4.6e-07 |
| 65536 | 9.2227 | 8.9556 | 14.7228 | n/a | n/a | 0.63x | 4.2e-07 | 3.9e-07 |
| 131072 | 19.8717 | 19.2159 | 29.5284 | n/a | n/a | 0.67x | 4.4e-07 | 3.8e-07 |
| 262144 | 39.3062 | 38.3923 | 59.1930 | n/a | n/a | 0.66x | 5.5e-07 | 4.5e-07 |
| 524288 | 82.5392 | 80.1628 | 121.6680 | n/a | n/a | 0.68x | 5.4e-07 | 4.9e-07 |
| 1048576 | 168.4914 | 164.8123 | 252.4688 | n/a | n/a | 0.67x | 5.8e-07 | 5.6e-07 |

Vendor unsupported rows (`N >= 4096`) report the actual runtime error, retained verbatim
in `vendor_error` in the raw JSON:

```text
RuntimeError: [NOT IMPLEMENTED]: error code=4 (at xdnn_pytorch/fft_fft.cpp:69)
```

`kernel_ms` (device events) tracks `api_ms` at every size — e.g. at `N=1048576`,
M0 dit `kernel_ms=168.40` vs `api_ms=168.49`; M1 `kernel_ms=252.39` vs `api_ms=252.47`.
The ordering is identical, so the conclusions do not depend on the timing metric.

## M0 vs M1 (paired, same protocol)

| Regime | geometric-mean M1 speedup vs M0 | Reading |
|---|---:|---|
| `N <= 1024` (launch-bound) | **1.58x** | Fewer launches (6 vs 10 at `N=1024`) and tile-resident intermediates win. |
| `N >= 4096` (memory-bound) | **0.70x** | Regression — cause below. |
| all 18 sizes | 1.02x | Roughly neutral overall. |

**Cause of the large-`N` regression (measured, not inferred).** The M1 fused kernel
updates a tile **in place** across its 6 stages, which requires a `tl.debug_barrier()`
between stages. Removing that barrier produces races at every size (rel err ~0.9,
confirmed) — it is mandatory. The barrier plus the in-place store→load dependency
serializes the memory pipeline that M0's out-of-place stages keep fully independent.
At small `N` the launch reduction dominates and M1 wins; at large `N` barrier cost
dominates and M1 loses. This is a **resource/dependency bottleneck of the pointer-based
fusion form**, not an algorithmic limit. Overcoming it is the job of register/UB
residence (M2), which never round-trips through memory — but M2 depends on the SSA
gather indexing form the current Triton lowering does not generate. Scoped precisely in
[Device Stack](device-stack.md).

## Honest reading

1. **M0 is a correctness-first baseline, not an acceleration.** Where the vendor is
   reachable (`N <= 2048`) the naive Triton radix-2 is slower (0.10x–0.78x). The vendor
   `torch.fft.fft` is heavily tuned; a per-stage global kernel cannot beat it on latency.
2. **M1 is a genuine, measured win in the launch-bound regime** (`N <= 1024`,
   geometric mean 1.58x vs M0) and a **measured regression in the memory-bound regime**
   (`N >= 4096`, 0.70x). Both are reported; neither is hidden.
3. **Coverage is the milestone's other value.** The vendor fails outright at `N >= 4096`;
   both M0 and M1 run correctly to `N = 1048576` at rel err ~6e-7.
4. **A selector must choose per size.** M1 should be selected only where it wins; the
   conservative default stays M0 (the plan's "never pick slower than the incumbent
   without opt-in" rule).

Reproduce (regenerates JSON, CSV, and this page's table):

```bash
CUDA_VISIBLE_DEVICES=6 python3 scripts/bench.py --trials 40 --tiles 64 \
    --json results/p800/bench_phase2.json --csv results/p800/bench_phase2.csv \
    --markdown /tmp/table.md
```
