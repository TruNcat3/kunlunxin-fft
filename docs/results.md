# Results

The current headline is the **Phase 3 experiment matrix** (plan §7) further down this
page; the Phase 2 sections below are the historical record of the M0/M1 milestones.

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
| `local-fused-tile` (M2) | `fft_dit_matrix` / `fft_dif_matrix` | Register-resident tile: the tile stage group as a dense linear map through `tl.dot` (one load/store pair, no barrier). |
| `local-fused-tile` (M3) | `fft_dit_grouped` / `fft_dif_grouped` | Execution-group tail over the post-tile stages. |
| `batch-spread` / `batch-folded` (M4) | `fft_dit_batch` / `fft_dif_batch` | Batched M0 stages, `B` on the grid (`spread`) or folded into the pair space (`folded`). Workload axis — reported as Gpts/s, never as B-vs-1 speedup. |
| `baseline-global-stage` (M5) | `fft_dit_radix4` / `fft_dif_radix4` | Same flow, radix-4 core: digit-reversal folded into the boundary stage, radix-2 tail for odd `log2(N)`. |

The lowering-contract check runs before timing: candidates are checked for
`boundary_residence=gm, layout=planar-soa` with unit/radix consistency
(`radix2-scalar` and `radix4-scalar` both legal on the baseline flow; M1
fusion stays radix-2-only; the schema-v2 split records `flow_realization`
and `local_residence` separately from the processing unit);
`selected_candidate` defaults to the conservative M0 (`lowest fused_tile`, stable
sort keeps the incumbent first) so every challenger must prove itself by
measurement. The analytic arithmetic ledger (`candidates.arithmetic_cost`) reports
M5's twiddle-multiply saving **and** its real-add cost — both directions, no
single-number claim.

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
fusion form**, not an algorithmic limit. Residency that never round-trips
through memory across stages (hence no inter-stage fence) is the fix — and it is
**landed as M2** via a dense matrix linear map through `tl.dot`, a form the
lowering accepts without the rejected tensor-subscript gather. The residency
attribution (M1 intermediates cache-resident per launch; M2
register+dot-implicit) is **supported by the correctness ablation above and
structural analysis of the lowering**; this stack exposes no cache/occupancy
counters, so it is not counter-confirmed. The tensor-subscript limitation stays
canaried and scoped in [Device Stack](device-stack.md); no shipped kernel
depends on it.

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

---

## Experiment matrix (Phase 3, plan §7)

Commit `12965bc`, profile `a78b8679`, `trials=50`, `warmup=10`, card 6,
`block=1024` (flagship). Regenerable report:
[`bench_phase3.md`](https://github.com/TruNcat3/kunlunxin-fft/blob/master/results/p800/bench_phase3.md)
from raw rows
[`bench_phase3.json`](https://github.com/TruNcat3/kunlunxin-fft/blob/master/results/p800/bench_phase3.json)
/ [`bench_phase3.csv`](https://github.com/TruNcat3/kunlunxin-fft/blob/master/results/p800/bench_phase3.csv)
/ [`bench_phase3_long.csv`](https://github.com/TruNcat3/kunlunxin-fft/blob/master/results/p800/bench_phase3_long.csv)
(section-9 columns). Every table below has an adjacent plot under
[`plots/`](https://github.com/TruNcat3/kunlunxin-fft/tree/master/results/p800/plots).

### Coverage (every axis the plan requires)

| Axis | Covered |
|---|---|
| direction | DIT **and** DIF, all flows |
| precision | fp32 (complex64 API, planar SoA kernel ABI) |
| length | `2^3..2^20`, 18 sizes; `2^12, 2^14, 2^16, 2^18, 2^20` highlighted in plots |
| batch | `1, 4, 16, 64` x `{spread, folded}` |
| block/tile | every declared legal block `256, 512, 1024, 2048` (separate runs, `B=1`) |
| flow | baseline, local-fused, matrix, grouped, batch, radix-4 |
| core | radix-2 baseline + legal radix-4 (analytic cost ledger both directions) |
| layout | public complex64 API vs internal planar lowering (conversion in `api_ms`) |

Vendor (`torch.fft.fft`) is the external baseline where it supports the size
(`N <= 2048`); beyond that, rows are **coverage/correctness results, not speedups**,
and the vendor error string is retained in the raw JSON.

### Latency by flow

![api_ms vs N](https://github.com/TruNcat3/kunlunxin-fft/blob/master/results/p800/plots/api_ms_vs_n.png)
![speedup vs M0](https://github.com/TruNcat3/kunlunxin-fft/blob/master/results/p800/plots/speedup_vs_m0.png)

Geometric-mean speedup vs the M0 baseline, DIT (same for DIF within noise):

| Flow | `N <= 1024` | `N >= 4096` | all 18 |
|---|---:|---:|---:|
| M1 fused tile=64 | 1.62x | 0.67x | 1.01x |
| M2 matrix tile=64 (DIT) | 1.80x | 1.53x | 1.65x |
| M2 matrix tile=64 (DIF) | 1.91x | 1.57x | 1.72x |
| M3 grouped tile=8 (DIT) | 1.58x | 1.57x | 1.58x |
| M5 radix-4 (DIT) | 1.03x | **1.69x** | 1.33x |
| M5 radix-4 (DIF) | 1.01x | **1.71x** | 1.32x |

At the highlighted large sizes (median api_ms, DIT):

| N | M0 | M1 | M2 | M3 | M5 radix-4 |
|---:|---:|---:|---:|---:|---:|
| 4096 | 1.05 | 1.31 | **0.72** | 0.78 | 0.86 |
| 16384 | 2.32 | 3.69 | 1.50 | 1.68 | **1.49** |
| 65536 | 9.15 | 13.85 | 6.06 | 5.50 | **4.73** |
| 262144 | 39.00 | 61.12 | 26.19 | 23.37 | **20.29** |
| 1048576 | 174.57 | 254.11 | 112.94 | 103.27 | **91.42** |

Reading: **no single winner across sizes.** M2/M3 lead from mid sizes; M5 radix-4
becomes the fastest flow from `N=65536` up (fewer launches and twiddle multiplies
finally outweigh its extra real adds once global traffic dominates). M1 remains a
small-`N`-only win (its mandatory inter-stage barrier — measured cause in the M1
section above). M2 vs M1 at `N >= 4096`: 2.29x; M3 vs M2 there: 1.03x.

Against the vendor (reachable only at `N <= 2048`): M0 is 1.22x slower than the
vendor at `N=2048`, but **M2 is the first flow to beat the vendor where it runs** —
0.53ms vs 0.65ms (**1.22x faster**) at `N=2048` (M1/M3/M5 sit between). The vendor
fails outright at `N >= 4096`; every Triton flow runs correctly to `2^20` at worst
err 1.1e-5 (M2 DIT at `2^20`; all other flows < 1e-6).

### M5 arithmetic ledger (analytic, same flow)

Radix-4 on the baseline flow replaces `log2(N)` radix-2 stages with
`ceil(log2(N)/2)` radix-4 stages (radix-2 tail when odd). Per transform:

| N | launches r2->r4 | twiddle mults r2->r4 | real adds r2->r4 |
|---:|---:|---:|---:|
| 8 | 3 -> 2 | 12 -> 10 | 48 -> 64 |
| 4096 | 12 -> 6 | 24576 -> 18432 | 98304 -> 147456 |
| 1048576 | 20 -> 10 | 10485760 -> 7864320 | 41943040 -> 62914560 |

Both directions are reported (`candidates.arithmetic_cost`): radix-4 buys **25% fewer
twiddle multiplies and half the launches at 50% more real adds**. The measured
crossover (M5 wins at large N, neutral below) matches this ledger — fewer round
trips matter when memory dominates; the add cost matters when launches do.

### Batch workload axis (M4)

Batch is a workload axis, never a B-vs-1 speedup: throughput is amortized
`B * M0_ms / M4_ms` (higher = better packing of the same stage grid).

![batch throughput](https://github.com/TruNcat3/kunlunxin-fft/blob/master/results/p800/plots/batch_throughput.png)

| B (folded, DIT) | `N <= 1024` | `N >= 4096` |
|---|---:|---:|
| 1 | 0.99x | 1.00x |
| 4 | 3.67x | 1.17x |
| 16 | 11.52x | 1.25x |
| 64 | 28.22x | 1.27x |

`spread` tracks `folded` within ~5% **from `N >= 2048` up** (raw rows carry
both); in the launch-bound regime `N <= 1024` the folded variant wins — up to
**4.11x at `N=8, B=64`** — so the two variants must not be pooled below
`N=2048`. Amortization is
strong in the launch-bound regime and converges to ~1.2-1.3x once memory
dominates; per-element correctness holds at every (B, N) (max err 7.2e-7).

### Block sweep (every declared legal block)

![block sweep](https://github.com/TruNcat3/kunlunxin-fft/blob/master/results/p800/plots/block_sweep.png)

Runs at `B=1`, sizes `{2^3, 2^6, 2^10, 2^12, 2^14, 2^16, 2^18, 2^20}` (full range,
all highlighted sizes), raw rows:
[block256](https://github.com/TruNcat3/kunlunxin-fft/blob/master/results/p800/bench_block256_long.csv),
[block512](https://github.com/TruNcat3/kunlunxin-fft/blob/master/results/p800/bench_block512_long.csv),
[block1024=phase3](https://github.com/TruNcat3/kunlunxin-fft/blob/master/results/p800/bench_phase3_long.csv),
[block2048](https://github.com/TruNcat3/kunlunxin-fft/blob/master/results/p800/bench_block2048_long.csv).

Geometric-mean api_ms ratio vs the default block=1024 (lower = faster):

| block | M0 DIT | M5 DIT | M2 DIT |
|---:|---:|---:|---:|
| 256 | **0.80x** | **0.70x** | **0.86x** |
| 512 | 0.87x | 0.80x | 0.91x |
| 1024 (default) | 1.00x | 1.00x | 1.00x |
| 2048 | 1.24x | 1.29x | 1.10x |

The declared default is **not optimal**: block=256 is ~20% faster than 1024
geometric-mean across sizes; block=2048 is ~25% slower. At `N >= 65536` the
differences collapse to <5% (traffic-dominated). Correctness holds at every
block (worst err 1.1e-5). This is a measured tuning axis, not a knob to leave
at its default.

### Honest reading (phase 3 deltas)

1. **The ladder pays off and changes the winner.** Against the M0 baseline,
   M2/M3 deliver 1.5-1.9x and M5 delivers 1.7-1.8x at large N — the first flows
   that beat the memory-bound M0 regime at every large size.
2. **The vendor comparison is unchanged in kind**: still ahead where reachable,
   except near parity at `N=2048`; still absent where we have coverage.
   Nothing here is a vendor speedup claim.
3. **M5's ledger is honest in both directions**: fewer twiddles/launches, more
   adds, and a measured size crossover — not a free win.
4. **Defaults are measured, not assumed**: block=1024 is ~20% off the best
   declared block; the conservative selector (M0) is far from the per-size best
   (M2/M3/M5). A per-size selector remains the obvious next step, gated by the
   plan's "never slower than the incumbent without opt-in" rule.
5. **Operational note**: the `B=64` phases at `N >= 2^19` are legitimately
   silent for tens of minutes (seconds per call x 50 trials x 16 entries);
   run watchdogs must not treat that as a hang (a real hang looks identical —
   `kill -9` by PID and re-run; see
   [Device Stack](device-stack.md) dangerous-operation notes).

## Round B: confirmation matrix (commit `da251f9`)

Re-run of the winner question on a clean commit with the review-round
protocol: vendor **inside the randomized interleave** at its supported sizes,
per-trial samples stored (`*_samples`, `*_p10`, `*_p90` in the JSON),
`trials=30`, schema-v2 raw rows. Sizes `8,16,1024,2048,4096,65536,1048576`;
flows M0/M1/M2/M3/M5 (+ interleaved vendor at `N<=2048`); blocks 1024 and 256;
batch probe `B=1,4,16,64` x `{spread,folded}` at `N=64,4096,65536`.

Artifacts (manifest `dirty=false` for the block-1024 run; the block-256 and
batch manifests record `dirty=true` only because the preceding run's output
files were untracked in the same tree — same commit, no code change):
[block1024](https://github.com/TruNcat3/kunlunxin-fft/blob/master/results/p800/confirm_b1024_long.csv),
[block256](https://github.com/TruNcat3/kunlunxin-fft/blob/master/results/p800/confirm_b256_long.csv),
[batch](https://github.com/TruNcat3/kunlunxin-fft/blob/master/results/p800/confirm_batch_long.csv);
session stdout traces + offline-reproduction steps:
[diagnostics/2026-10-15](https://github.com/TruNcat3/kunlunxin-fft/tree/master/results/p800/diagnostics/2026-10-15).

DIT speedup vs M0 (median api_ms, block-1024 run; max per-element rel err
5.8e-7 across all flows):

| Flow | `N<=1024` | `N>=4096` | all 7 |
|---|---:|---:|---:|
| M1 fused tile=64 | 1.51x | 0.75x | 1.06x |
| M2 matrix tile=64 | 1.57x | 1.55x | 1.58x |
| M3 grouped tile=8 | 1.35x | 1.57x | 1.45x |
| M5 radix-4 | 1.00x | 1.64x | 1.27x |

The phase-3 ordering reproduces on the confirmation sizes: M1 small-only, M2
flat 1.5x, M3 tracks M2, M5 wins the long transforms. Nothing in the
confirmation matrix reverses a phase-3 conclusion.

## Round C: per-size selector (table-driven)

`config/selector_table.json` (built by
[`scripts/build_selector_table.py`](https://github.com/TruNcat3/kunlunxin-fft/blob/master/scripts/build_selector_table.py)
from the Round B runs) maps
`(profile_hash, precision=fp32, direction, N, batch)` to
`(block, flow_realization, processing_unit, params)`. Promotion guard: a
challenger must beat the M0 incumbent by **>=2%** paired-median improvement
(engineering guard, not a significance claim); otherwise M0 stays. Any key
outside the measured matrix **abstains** (`src/xpu/selector.py::select`)
— unknown N, unmeasured batch/precision, or a profile-hash mismatch return
`source="abstain"` and the caller must fall back to M0, never guess.

B=1 winners (14 entries, `N x {dit,dif}`):

| N regime | dit | dif |
|---|---|---|
| 8, 16 | M1 fused (block 1024) | M1 fused (block 1024) |
| 1024, 2048, 4096 | M2 matrix (block 256) | M2 at 2048/4096; **M5 radix-4 at 1024** |
| 65536, 1048576 | M5 radix-4 (block 256) | M5 radix-4 (block 256) |

Batch entries (18): the folded M4 variant at every probed `(N,B)` — the
spread variant only wins where it clears the 2% guard (never on this probe).
Every table entry records its incumbent, challenger, and evidence keys, and
`tests/test_selector.py` pins that the committed table rebuilds byte-identically
from the committed confirmation JSONs and that every challenger clears the
guard. The conservative `selected_candidate` (M0) remains the launch default;
the table is the measured opt-in layered on top, per the plan's
"never slower than the incumbent without opt-in" rule.
