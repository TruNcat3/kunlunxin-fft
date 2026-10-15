# Batch focus report — issue #3 (P800)

Evidence for the issue #3 deliverables: batch-aware M2/M5 lowering, selector
closure with the 2% guard, execution metadata, hardware probe and calibration,
bounded cost-model search, flow/core ablations, and the P2 Td canary.

Two independent formal runs (each 3 process sessions x 30 interleaved trials x
8 shapes x both directions, correctness-first, `CUDA_VISIBLE_DEVICES=6`,
sync API wall time in ms, median of session medians):

* gate run: [`results/p800/batch_gate/`](https://github.com/TruNcat3/kunlunxin-fft/tree/master/results/p800/batch_gate)
* final run: [`results/p800/batch_release/`](https://github.com/TruNcat3/kunlunxin-fft/tree/master/results/p800/batch_release)

Raw per-trial rows, `summary.{json,csv}`, `manifest.json` (versions, seeds,
protocol, vendor re-probe, repro command), and `server_report.md` are in each
directory.

## Headline (final run, plan-reuse `selected` from the frozen table)

| shape | selected | incumbent M4 | speedup | vendor | vendor ahead |
|---|---:|---:|---:|---:|---:|
| N1024 B16 dit | 0.710 | 1.761 | 2.48x | 0.252 | 2.8x |
| N1024 B16 dif | 0.731 | 1.685 | 2.30x | 0.252 | 2.9x |
| N1024 B64 dit | 2.268 | 6.092 | 2.69x | 0.262 | 8.7x |
| N1024 B64 dif | 2.324 | 5.903 | 2.54x | 0.262 | 8.9x |
| N2048 B16 dit | 1.569 | 3.492 | 2.23x | 0.663 | 2.4x |
| N2048 B16 dif | 1.583 | 3.406 | 2.15x | 0.663 | 2.4x |
| N2048 B64 dit | 5.009 | 12.344 | 2.46x | 0.672 | 7.5x |
| N2048 B64 dif | 5.051 | 11.940 | 2.36x | 0.672 | 7.5x |

Against the literal M0 host loop the same rows are 4.7-12.1x faster.
Session-medians stdev <= 0.006 ms; all-trial stdevs are recorded per candidate
(`stdev_ms_all_trials`, `n_trials=90`).

## 1. Which shapes gained

All four issue shapes (N=1024/2048 x B=16/64), in both directions: 2.15-2.69x
over the stable batch incumbent (best of M4 folded/spread), each entry clearing
the 2% promotion guard in two independent 3-session runs.

## 2. Where the gain comes from (one axis per contrast)

Ratios from the same interleaved schedule (control/variant, >1 = variant
faster); full table in `server_report.md`:

| contrast (axis varied, others held) | ratio range | attribution |
|---|---:|---|
| m0 host loop / m4 spread — batch lowering, radix-2 held | 1.98-4.61x | flow: Ud=B on the grid axis vs no batch lowering |
| m4 spread / m5 — core radix, global-stage flow held | 1.75-2.20x | core: radix-4 spans beat radix-2 stages |
| m4 spread / m2 default — local-matrix vs scalar stages | 1.99-2.17x | joint flow+unit: register-resident dot group (local-fused-tile + dense-dot-fft64) |
| m2 default / best screened m2 — parameters only | 1.17-1.29x | cost-model bounded search: tile=128 variants |
| m4 spread / m2 packet Td=2 — P2 canary | 0.92 | true packet Td costs ~8% here; not promoted |

Reading: the incumbent-relative win is the M2 lowering (joint flow + processing
unit, ~2.0-2.2x) multiplied by a ~1.2x parameter-screening gain from the
bounded search; radix-4 (M5) is the faster global-stage core but still loses
to M2's register residency.

## 3. Where we still lose to the vendor

Everywhere: `torch.fft.fft` is 2.4-8.9x faster than our best path on these
shapes (median 5.2x). The vendor gap widens with batch at fixed N (B=64 rows:
7.5-8.9x). Vendor support was **re-probed this round** (manifest
`vendor_probe`): N<=2048 ok, N>=4096 returns
`NOT IMPLEMENTED error code= 4` — so no vendor row exists beyond N=2048 on
this stack, and none is claimed.

## 4. What the evidence does not prove

* **API wall time only.** This stack exposes reliable synchronized wall time;
  kernel/device time was not collected this round (recorded as such in the
  manifest).
* **GM bytes are analytical estimates** (`launches x 2 x N x B x 8`), no
  hardware counters — labeled `analytical-estimate` in every row.
* **The cost model does not predict absolute time.** Holdout validation
  (constants declared a priori, never fitted): median absolute % error is
  89-92% for stage-based families (unmodeled multi-block launch latency),
  17% for the m0 loop. What it does provide: incumbent-relative guard-decision
  agreement 57/58 pairs, top-K hit 8/8 shapes, mean selection regret 0.1%.
  The model screens top-K only; every final decision is measured.
* **Single healthy card** (card 6; physical card 1 is broken — XID, reset
  refused), one container, one driver. No multi-card or cross-stack claim.
* **Td is bounded to 2 on this backend**: `packets>2` is rejected by the
  runner because programs beyond four dot operations *silently miscompile*
  on this Triton XPU fork (probe evidence, kernel docstrings).
  DIF packet lowering is `not-lowered` this round (no interleaved DIF path).
* No claim for shapes outside the measured matrix — the selector abstains to
  M0 there (existing contract).

## 5. Frozen selector vs experimental

**Frozen** (in `config/selector_table.json`, both runs reproduced the same
winner descriptors 8/8, guard=2%, deterministic rebuild test):

* the 8 focus entries: batch M2 with per-shape screened parameters —
  tile=128 with rows in {4, 8, 16} depending on shape, block=256;
* all 32 pre-existing B=1 / batch entries (byte-identical, unchanged).

**Experimental only** (measured, recorded, not promoted):

* packet Td=1/Td=2 canary (slower than m2 spread at the canary shape);
* m2 default tile=64 (kept as a challenger, lost the guard to tile=128);
* the bounded-search space outside the measured top-K;
* DIF packet loop and packets>2: `not-lowered` / backend-limited.

## Probe, calibration, P1.1 metadata

* `scripts/hw_probe.py` wrote `config/calibration.json` (self-hash
  `profile_hash`, verified by `src/xpu/calibration.py`; tampering raises
  `CalibrationMismatch` — no silent reuse). Anchors: launch 0.0258 ms
  (1-block), observed 1024-block dispatch 12.2 ms (recorded, excluded from
  the model after validation), sync 0.0093 ms, d2d 867 GiB/s, H2D 9.5 GiB/s,
  mm1024 fp32 ~52 TFLOP/s, vendor fft(64x2048) 0.61 ms. Six verified
  execution paths re-checked each probe run.
* Every raw row carries the P1.1 axes: Us/Ts/Ud/Td (folded indexes are
  honestly `not-lowered` for Td — flattening is not a packet loop), logical
  vs physical groups, launch count, GM-boundary estimates, tile/block/rows,
  packets, profile hash, timing scope.
* `src/xpu/cost_model.py` — structured launch/GM/compute terms, bounded
  search over tile x rows, `validate()` holdout metrics (above).

## P2 negative results (kept, not expanded)

Probe evidence compiled under `/tmp` this round and summarized in the runner
and kernel docstrings of this commit:

* >4 `tl.dot` per program: **silent wrong codegen** (no error, rel err ~1.0);
* non-affine row offsets in pointer arithmetic: `SetVector::front()` crash;
* 3-D broadcast offsets: `OffsetAnalysis.cpp:580` assertion;
* `tl.reshape`: `GlobalEncodingAttr` UNREACHABLE; `tl.cat`: 1-D only;
* Td=2 design that stays within these limits: sequential `static_range`
  packet loop, interleaved re/im loads, 2-dot fused complex pair-product,
  table loaded once — measured 0.938 ms vs 0.866 ms for plain spread at the
  canary shape, i.e. no gain, kept as data.

## Reproduce

```bash
# probe (writes config/calibration.json; needs a healthy card)
CUDA_VISIBLE_DEVICES=6 python scripts/hw_probe.py

# formal run (3 sessions x 30 trials, both directions)
CUDA_VISIBLE_DEVICES=6 timeout -s KILL 3600 \
  python scripts/run_batch_release.py --sessions 3 --trials 30 --warmup 3 \
  --out results/p800/batch_release

# table rebuild (must reproduce config/selector_table.json byte-for-byte)
python scripts/build_selector_table.py \
  results/p800/confirm_b1024.json results/p800/confirm_b256.json \
  --batch results/p800/confirm_batch.json \
  --focus results/p800/batch_gate/summary.json \
          results/p800/batch_release/summary.json \
  --out config/selector_table.json

# tests (host; hardware on a healthy card)
python -m pytest tests/ -m "not hardware"
CUDA_VISIBLE_DEVICES=6 python -m pytest tests/ -m hardware
```
