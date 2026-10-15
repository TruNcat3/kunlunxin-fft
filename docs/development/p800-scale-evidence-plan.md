# P800 Scale and Evidence Completion Plan

## 1. Purpose and scope

This round turns the implementation completed under PR #1 into a compact,
reproducible migration case study. It has four goals:

1. show that the cuButterfly compute-flow method transfers to the P800;
2. expose the length, batch, and total-work breakpoints of the transferred
   implementation;
3. deploy measured winners without losing the mature M0 path; and
4. preserve enough raw evidence to explain both gains and regressions.

This is **not** an open-ended search for the fastest possible FFT at every
shape. The current milestone remains FP32 C2C forward. DIT and DIF are two
algorithm organizations for that forward transform; they are not forward and
inverse directions. Inverse FFT, R2C/C2R, non-power-of-two lengths, and 2D FFT
remain future work and do not block this round.

The method and optimization reasoning come from
[cuButterfly](https://github.com/TruNcat3/cuButterfly). Ascend-FFT is used as a
reference for experiment layering and evidence organization. CUDA and Ascend
parameters must not be copied as presumed P800 optima.

## 2. What is already covered

Phase 3 already covers all 18 powers of two from `N=8` through `N=2^20` and
the M4 batch values `1, 4, 16, 64`. The largest measured workload contains
`2^20 * 64 = 67,108,864` complex points. The single-transform length range is
therefore not small.

The remaining gap is the **shape and evidence structure**:

- four batch values do not identify saturation and tail breakpoints;
- M4 batches the M0 stage flow and does not prove that M2/M3/M5 have efficient
  batched lowerings;
- no equal-work experiment separates length dependence from data-volume
  dependence;
- application-proxy and long-running stability evidence is missing;
- the vendor overlap ends at `N=2048`, so longer rows are coverage and internal
  comparisons, not external speedups;
- stored Phase 3 rows contain aggregates rather than every independent trial;
  and
- the runtime selector still needs a measured, independently validated
  `winner -> table -> fallback` deployment loop.

The next experiment must fill these gaps without evaluating every candidate at
every large shape.

## 3. Feature support matrix

The levels below are the single source of truth for what this round claims.
Every level refers to something enforceable in code (validation or lowering
contract), not to intent. Any change to validation or the lowering contract
updates this table in the same change.

| Level | Meaning |
|---|---|
| supported+tested | implemented, correctness-gated in the current suites, and present in the published matrix |
| supported+pending-evidence | implemented and correctness-gated; scale/session evidence still owed this round (E-group noted) |
| planned-this-round | committed by this plan (P0/P1/P2); not implemented yet |
| roadmap | future rounds; explicitly non-blocking |
| rejected-before-launch | rejected at the validation/lowering layer with a structured reason; never silently scored |

| Dimension | Value | Level | Enforced / observed at |
|---|---|---|---|
| Transform | C2C forward | supported+tested | `src/xpu/validate.py`, hardware suite, phase-3 (DIT and DIF) |
| Transform | C2C inverse | roadmap | — |
| Precision | fp32 (`complex64`) | supported+tested | dtype check in `validate_input`; the only precision this round |
| Precision | fp16/bf16/fp64 | roadmap | — |
| Length | power-of-two `8..2^20` | supported+tested | `log2_exact`; 18 published sizes |
| Length | non-power-of-two | rejected-before-launch; support roadmap | `log2_exact` raises before any launch |
| Rank | 1-D | supported+tested | rank-1 check in `validate_input` |
| Rank | 2-D FFT | roadmap | — |
| Batch | `(B,N)`, `B` power of two, **M0-stage flow only** (M4 spread/folded) | supported+tested at `B=1,4,16,64`; `B=256..4096` supported+pending-evidence (E2) | `_validate_batch_input`; batch stage kernels |
| Batch | non-power-of-two `B` | rejected-before-launch; E5 uses labeled padding; support roadmap | `log2_exact(b)` |
| Batch | batched M1/M2/M3/M5 | planned-this-round (P2) or remains single-transform by explicit decision | lowering contract must reject batch for rank-1-only flows before launch |
| Algorithm flows | M0..M5 (see [architecture](../architecture.md)) | supported+tested | `tests/test_fft.py`; bench correctness phase; phase-3 rows |
| Vendor comparison | `N<=2048` (including batched shapes) | supported+tested | phase-3 vendor rows; interleaved again in E1 |
| Vendor comparison | `N>2048` | rejected-by-vendor, recorded as `unsupported`; internal comparisons only | vendor `NOT IMPLEMENTED` error retained in the raw row |
| External speedup claims | matched vendor-supported shapes only | acceptance rule (section 8) | — |
| Runtime selector | winner table, M0 closure, fallback with reason | planned-this-round (P2) | — |
| Structured failure outcomes | `unsupported/infeasible/compile-failed/incorrect/timeout/oom` | planned-this-round (P0) | — |

DIT and DIF are two algorithm organizations of the **same forward transform**;
they are not forward versus inverse. Every published comparison must match on
operator, precision, direction, normalization, layout, placement, and timing
scope (section 6).

E5 proxies use non-power-of-two application batch counts (`14`, `140`). This
round pads those to the next power of two and labels the row `padded-proxy`;
a native non-power-of-two batch lowering is roadmap. A padded row must never
be presented as a native non-pow2 measurement.

## 4. Test scope and gates

Four layers, one gate each. A layer never substitutes for another: host
legality precedes every launch, hardware correctness precedes every timed row,
and scale sessions (E1--E6) run only on shapes that passed the layer below
them.

| Layer | Entry point | Environment | What it gates | What it covers today |
|---|---|---|---|---|
| Host-only unit tests | `pytest -q` (`tests/test_host.py`, `tests/test_report.py`) | ordinary CI; no vendor stack, no torch | lowering contract for every enumerated candidate, config schema, report byte-identity | M0--M5 candidate enumeration and rejection reasons, batch legality, matrices math, regenerated-report byte identity |
| Hardware correctness | `pytest -q -m hardware` (`tests/test_fft.py`) | P800 + xpytorch | correctness admission per flow/size/axis | M0 DIT 8 sizes, M0 DIF 7; M1 6; M2 DIT 6/DIF 5; M3 DIT 6/DIF 5; M5 DIT 9/DIF 8; M4 3 sizes x `B={1,4,16,64}` x `{spread,folded}`; axis-neutrality (tile/rows/glog/maxp/cells); negative-input rejection |
| Declared matrix script | `scripts/test_correctness.py --profile smoke\|full` | P800 | release gate over `config/test_matrix.json` | **M0 only** (DIT/DIF at the declared sizes); per-flow gates live in the hardware suite |
| Scale evidence sessions | experiment runner built in P0 of this plan | P800 | E1--E6 rows with structured outcomes | not built yet |

Unification items this round:

- [ ] Give "the correctness suite" one unambiguous scope statement: either
  extend `scripts/test_correctness.py` to every flow, or explicitly declare
  `tests/test_fft.py` the per-flow gate and keep the script as the M0
  published-matrix gate. Do not leave the two scopes implied.
- [ ] Every E-group run executes its hardware correctness gate (table below)
  before admitting a shape to timing, and stores the gate result next to the
  timing rows.
- [ ] Raw rows, tests, and reports use the schema-v2 vocabulary of
  `src/xpu/candidates.py` (`flow_realization`, `processing_unit`,
  `boundary_residence`, `local_residence`) so the layers cannot drift apart.

E-group correctness gates:

| Group | Host pre-gate | Hardware gate before timing |
|---|---|---|
| E1 | batch legality (M4 candidates) | every timed flow at `N<=2048` for each probed `B`; vendor rows only where `torch.fft.fft` succeeds |
| E2 | batch legality at each `B` | `fft_dit_batch`/`fft_dif_batch` at each `(N,B)`; `oom`/`timeout` rows recorded, not dropped |
| E3 | legality at each derived `(N,B)` | the size and batch gates at every derived shape |
| E4 | — | hardware-suite sizes `2^16..2^20` for the timed flows |
| E5 | padded-proxy labeling for non-pow2 `B` | proxy-shape correctness; padding documented in the row |
| E6 | — | correctness re-checked periodically and at the end of the run, not only at admission |

One error contract for all layers: the reference is numpy `complex128`;
`max_rel_error = max(|got-ref|) / max(|ref|)` over the peak-normalized
real/imag deviation (the `rel < 1e-4` gate in `config/test_matrix.json` and
`err_vs` in the bench). The vendor `torch.fft.fft` is a performance baseline
only (`N<=2048`), never the numeric reference. Seeded-random input is primary;
impulse/constant/single-tone patterns isolate anomalies (section 7).

## 5. Execution order

### P0: runner and evidence safety

- [ ] Add named experiment groups, resume support, and a separate output
  directory per run.
- [ ] Print `group/N/batch/candidate/session/trial` at case start and completion,
  with estimated and actual elapsed time.
- [ ] Preserve `unsupported`, `infeasible`, `compile-failed`, `incorrect`,
  `timeout`, and `oom` as structured outcomes with the original error.
- [ ] Record the selected device, active processes, toolchain, repository state,
  and configuration hashes before timing.
- [ ] Randomize and interleave M0, selected, oracle, and vendor calls within a
  shape. Do not time the vendor in a separate, fixed-order pass.
- [ ] Store each independent trial. Aggregates must be regenerated from raw
  rows rather than replacing them.
- [ ] Add a dry-run mode that prints case count, estimated bytes, and expected
  runtime before allocation or compilation.
- [ ] On a suspected hang, collect a process/device snapshot before terminating
  the exact PID. Silence or high utilization alone is not a hang verdict.

Read the existing
[P800 incident record](https://github.com/TruNcat3/kunlunxin-fft/blob/master/docs/development/incidents/2026-10-14-p800-bench-hang.md)
before changing watchdog behavior.

### P1: layered scale experiments

These are planned measurements, not current results.

| Group | Question | Shapes | Timed implementations |
|---|---|---|---|
| E1: matched vendor overlap | Where is the transferred library competitive under identical semantics? | `N=8..2048`; begin with `B=1,4,16,64`, then probe `256,1024` where the vendor supports the batched shape | M0, runtime-selected, limited measured oracle, vendor |
| E2: batch saturation | Where do data-space expansion, saturation, and tails change the winner? | `N=256,1024,4096,16384`; `B=1,4,16,64,256,1024,4096` | batched M0, selected, one relevant flow/layout ablation |
| E3: equal work | How much of the cost comes from stage depth rather than total points? | total points `2^20`, `2^24`, `2^26`; derive integer `B=points/N` for representative `N=256..2^20` | M0, selected, spread/folded where both are legal |
| E4: long transform | Does the selected path retain its advantage as stage depth grows? | `N=2^16,2^18,2^20`; `B=1,4,16`; reuse existing `B=64` only after the protocol matches | M0, selected, one causal ablation |
| E5: application proxies | Do realistic shape families preserve the migration benefit? | OFDM-like: `(2048,14),(4096,140)`; radar-like: `(1024,64),(2048,256)`; DL-frequency-like: `(1024,32),(4096,128)` | M0, selected, vendor where supported |
| E6: stability | Does repeated plan/cache reuse remain correct and bounded? | one small, one medium, and one long selected shape | selected plus the minimum diagnostic baseline |

E5 measures FFT application **proxies**, not complete OFDM, radar, or learning
applications. P800 cluster/core counts are hints for where a breakpoint may
occur, not proof that a particular batch maps one-to-one to hardware cores.
After the power-of-two pilot, add `B-1/B/B+1` only around observed knees.

Do not run a full Cartesian product of all M0--M5 candidates over E1--E6. Use
the following funnel:

1. low-repeat pilot on a small set of anchor shapes;
2. legality and correctness pruning;
3. limited search among plausible candidates;
4. freeze the winner table; and
5. measure M0, the frozen selector, and a measured oracle on independent
   validation sessions.

Capacity-only cases may use fewer timing repetitions, but they must be labeled
as capacity/correctness evidence and excluded from performance aggregates.

### P2: batched lowering and deployment selection

- [ ] Today only the M0-stage flow has a batched lowering (M4 spread/folded);
  M1/M2/M3/M5 accept rank-1 inputs only. State explicitly which of M2/M3/M5
  gain batched lowerings this round and which stay single-transform by
  decision. Reject unsupported combinations before launch — the lowering
  contract already rejects batch points outside M4.
- [ ] Extend only the promising processing flows to representative batches;
  repeating the M4/M0 flow at more batches is not sufficient.
- [ ] Keep flow organization and processing-unit selection as separate fields
  (`flow_realization` vs `processing_unit` in `src/xpu/candidates.py`
  schema v2). Compare cores under the same flow and flows under the same
  core.
- [ ] Preserve candidate transitions:
  `enumerated -> legal -> compiled -> correct -> measured -> validated`.
- [ ] Build the selection table from training/search shapes and test it on
  independent validation sessions.
- [ ] Keep M0 in the candidate closure. A selected plan must fall back to M0
  for an uncalibrated or regressing shape and emit the reason.
- [ ] Report M0, a fixed optimized point, search oracle, runtime selector, and
  vendor as distinct columns. Never label a hand-picked point as selector
  performance.

Use a conservative promotion guard: a challenger must pass all correctness
checks and improve the paired median by at least 2% before replacing the
incumbent. A result inside the guard band is a tie and retains the incumbent;
the threshold is an engineering guard, not a statistical-significance claim.

## 6. Measurement contract

Every published speedup must match:

```text
operator, precision, transform direction, normalization,
N, batch, input/output layout, placement, and timing scope
```

Increasing batch changes the workload. Compare batched configurations using
latency per transform and transforms/second or points/second; do not call
`B=64` versus `B=1` a same-workload speedup.

For formal representative points:

- use at least five independent sessions;
- randomize/interleave contender order within each shape;
- run correctness before admitting each trial to timing;
- report median, p10, p90, coefficient of variation, and all raw values; and
- keep unsupported and failed rows out of geometric means while retaining
  their structured records.

Store the following artifacts:

| Artifact | Required content |
|---|---|
| `manifest.json` | commit and dirty state, device/toolchain, command, seeds, timing protocol, config/profile hashes |
| `raw.jsonl` or `raw.csv` | one row per session/trial, including execution order and outcome |
| `summary.csv` | generated per-shape/per-candidate statistics |
| `selection.json` | chosen plan, applicability, confidence/reason, and fallback |
| `profile/` | raw traces/counters and exact collection commands for anchor points |
| `report.md` | question, result, causal evidence, limitations, and unresolved cases |

Minimum raw schema:

```text
run_id, session, trial, order, commit, dirty, device, toolchain,
profile_hash, config_hash, operator, precision, transform_direction,
flow_id, flow_realization, N, batch, total_points, input_bytes, candidate_id,
selection_source, Us, Ts, Ud, Td, processing_unit, layout,
boundary_residence, local_residence,
block, tile, rows, glog, maxp, cells, execution_groups, stages_per_group,
materialized_boundaries, launches,
warmup, repeat, seed, timing_scope, api_ms, device_event_ms,
max_abs_error, max_rel_error, correct, status, error,
predicted_ms, selection_reason
```

Vocabulary matches `src/xpu/candidates.py` schema v2: `flow_realization` is
`global-stage | fused-inplace | matrix-resident | grouped-boundary`;
`processing_unit` is `radix2-scalar | radix4-scalar | dense-dot-fft64 |
composite-linear-map`; `boundary_residence` is the stage/group-boundary
memory (`gm` today) and `local_residence` is the in-launch residency
(`none | cache | register | implicit-dot | register+implicit-dot`). `status`
carries the structured outcomes (`ok`, `unsupported`, `infeasible`,
`compile-failed`, `incorrect`, `timeout`, `oom`). `max_rel_error` uses the
single error definition from section 4.

Analytical and measured quantities must not share an unlabeled field. For
example, use `gm_read_bytes_estimate` plus `gm_read_bytes_counter` and record
the source. Batch must be included in every traffic calculation.

The server-side task is implementation, measurement, raw-data preservation,
and a concise engineering report. Repository-wide prose and site restructuring
are handled separately after the evidence is accepted.

## 7. Optimization and troubleshooting map

| Observation | First checks and response | Read before changing code |
|---|---|---|
| Fusion helps small `N` but hurts large `N` | Separate launch reduction from actual GM-boundary reduction; inspect barriers, store-load chains, spill, and local-resource pressure | [Kunlun architecture](../architecture.md), [cuButterfly performance model](https://github.com/TruNcat3/cuButterfly/blob/main/docs/performance_model.md) |
| Throughput stops improving with batch | Check grid mapping, active programs, bandwidth, tails, and per-point materialized traffic; more work cannot amortize steady-state bytes | [Ascend scale experiment](https://github.com/TruNcat3/ascend-fft/blob/master/docs/benchmarks/scale.md), [cuButterfly general shapes](https://github.com/TruNcat3/cuButterfly/blob/main/docs/general_shapes.md) |
| A higher radix is neutral or slower | Under the same flow, compare launches, twiddles, real operations, register/UB use, occupancy, and traffic | [cuButterfly processing-unit space](https://github.com/TruNcat3/cuButterfly/blob/main/docs/processing_unit_design_space.md) |
| Gather, transpose, or a tile does not compile | Identify the exact rejected form; try static indices, pointer-form loads, register permutation, small tiles, then explicit boundaries. Do not infer architectural impossibility from one lowering failure | [P800 device stack](../device-stack.md), [Ascend adding design points](https://github.com/TruNcat3/ascend-fft/blob/master/docs/development/adding-design-points.md) |
| Runtime selector loses to a hand-picked point | Verify candidate closure, legality, cache/config keys, model envelope, and that fallback actually executes; independently remeasure the frozen plan | [cuButterfly runtime selector](https://github.com/TruNcat3/cuButterfly/blob/main/docs/runtime_selector.md), [architecture guardrails](https://github.com/TruNcat3/cuButterfly/blob/main/docs/architecture_guardrails.md) |
| API and device timings disagree | Isolate allocation, conversion, compilation, plan setup, dispatch, auxiliary kernels, and synchronization; keep both scopes | [Ascend methodology](https://github.com/TruNcat3/ascend-fft/blob/master/docs/benchmarks/methodology.md), [Ascend profiling](https://github.com/TruNcat3/ascend-fft/blob/master/docs/development/profiling.md) |
| Vendor fails for a long or batched shape | Preserve the error and mark `unsupported`; use CPU/reference correctness and internal M0 comparison, but publish no external speedup | [Ascend scale evidence rules](https://github.com/TruNcat3/ascend-fft/blob/master/docs/benchmarks/scale.md) |
| Wrong output, zeros, or order-dependent error | Confirm device context and real writes; isolate bit/digit reversal, twiddles, normalization, and ownership using impulse, constant, single-tone, and seeded-random inputs | [Kunlun architecture](../architecture.md), [migration convergence plan](migration-convergence-plan.md) |
| OOM at a large shape | Estimate all live input/output/scratch/cache buffers before allocation; record `oom` and reduce batch within the same experiment tier rather than silently changing the shape | [Ascend methodology](https://github.com/TruNcat3/ascend-fft/blob/master/docs/benchmarks/methodology.md) |
| Long silence or high utilization | Check per-case progress and expected duration, collect process/device snapshots, then terminate only the target PID if the evidence indicates a hang | [P800 hang incident](incidents/2026-10-14-p800-bench-hang.md) |
| A full search takes too long | Use anchor shapes, model/legal pruning, and independent validation; do not initialize and time every candidate at every capacity point | [cuButterfly design overview](https://github.com/TruNcat3/cuButterfly/blob/main/docs/design_overview.md), [runtime selector](https://github.com/TruNcat3/cuButterfly/blob/main/docs/runtime_selector.md) |

## 8. Acceptance and stopping criteria

- [ ] P0 runner, dry-run, progress, resume, and structured failure paths work.
- [ ] E1--E6 each produce accepted data or an explicit
  `unsupported/infeasible/timeout/oom/budget-limited` record.
- [ ] Correct rows satisfy the declared `max_rel_error < 1e-4` contract, with
  input pattern and error definition recorded.
- [ ] Raw independent trials regenerate all summaries without hand edits.
- [ ] The selector preserves the M0 closure and shows no repeatable regression
  outside the 2% guard band on validation points.
- [ ] At least one representative result connects a flow change to a changed
  materialized-boundary/profile term and then to a measured benefit.
- [ ] Each workload regime has an observed configuration preference or
  breakpoint; no universal P800 configuration is assumed.
- [ ] External speedups are reported only in matched vendor-supported shapes.
- [ ] Host-only tests and P800 tests remain separate and record their
  environments.
- [ ] The feature support matrix (section 3) is updated in the same change as
  any new feature, validation check, or lowering-contract edit.
- [ ] "The correctness suite" has one declared scope: the script/hardware-suite
  unification item from section 4 is resolved, not left implied.
- [ ] Every E-group timing row carries the result of its hardware correctness
  gate (section 4 table).
- [ ] Raw rows use the schema-v2 vocabulary (`flow_realization`,
  `processing_unit`, `boundary_residence`, `local_residence`) everywhere —
  bench output, selector tables, and reports.

This round is complete when the repository can support the statement:

> The cuButterfly compute-flow method was transferred to P800, improves
> representative supported workloads under a reproducible protocol, and uses a
> validated selector to retain the mature path elsewhere.

It is not necessary to implement every FFT form or exhaust every physical-core
optimization before making that migration claim. Remaining transforms, 2D
support, and exhaustive physical-core tuning move to the roadmap.

## 9. Reference reading order

1. [Current Kunlun migration convergence plan](migration-convergence-plan.md)
2. [Kunlun architecture](../architecture.md) and
   [device stack](../device-stack.md)
3. [cuButterfly design overview](https://github.com/TruNcat3/cuButterfly/blob/main/docs/design_overview.md)
4. [cuButterfly performance model](https://github.com/TruNcat3/cuButterfly/blob/main/docs/performance_model.md)
5. [cuButterfly runtime selector](https://github.com/TruNcat3/cuButterfly/blob/main/docs/runtime_selector.md)
6. [Ascend measurement methodology](https://github.com/TruNcat3/ascend-fft/blob/master/docs/benchmarks/methodology.md)
7. [Ascend migration convergence contract](https://github.com/TruNcat3/ascend-fft/blob/master/docs/benchmarks/pr7-migration-convergence.md)

