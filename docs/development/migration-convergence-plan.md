# 迁移收敛计划（已完成归档）

!!! warning "历史计划"
    计划中的“下一步”和当时能力只描述该开发轮次。当前状态见
    [结果汇总](../benchmarks/results.md)、[支持范围](../reference/support.md)和
    [未来计划](../roadmap.md)。

# Original migration convergence plan

This document is the execution contract for the next Kunlunxin-FFT iteration. It
is intentionally written as a plan and an evidence checklist: the current
repository is a correctness-first P800 baseline, not yet a complete mixed
space-time dataflow implementation.

The architectural method and the analysis below originate from
[cuButterfly](https://github.com/TruNcat3/cuButterfly). The Kunlunxin repository
is a hardware-mapping instance, and [Ascend-FFT](https://github.com/TruNcat3/ascend-fft)
is the reference for the documentation, profiling, and release workflow. This
plan does not claim a new algorithmic method for Kunlunxin-FFT.

## 1. Current Status

The initial P800 snapshot establishes useful facts:

- Triton -> XMLIR -> XPU is the only device-code path demonstrated in the
  reference container.
- Radix-2 C2C forward DIT/DIF is correct for the declared power-of-two sizes.
- The host-side C++ butterfly reference passes its independent tests.
- The published DIT comparison is a baseline, not an acceleration claim: over
  the vendor-reachable sizes the geometric mean is about `0.119x`.
- For larger sizes the vendor path is unavailable in the tested environment;
  those rows demonstrate coverage and correctness, not speedup.

The current device implementation launches one global-memory kernel per FFT
stage. It therefore provides a **compatibility baseline** for the migration,
not the final cuButterfly execution pattern.

## 2. Historical Design Lessons

The following order is important and should be preserved during implementation.

### 2.1 Roofline first, then dependency analysis

For an execution organization with `G` materialized stage boundaries, use the
first-order payload model

```text
Q_GM ~= 2 * B * N * q * G
I_GM = W / Q_GM
throughput <= min(P_peak, BW_GM * I_GM)
```

`G` is the number of physically materialized execution-group boundaries, not
the number of logical labels in a decomposition. This model is only a bound;
actual counters also include twiddle/index traffic, padding, transaction
granularity, and cache effects.

The first optimization target is therefore repeated GM traffic. Increasing the
radix or adding arithmetic before reducing materialized boundaries can make the
kernel slower.

### 2.2 Stage and data dimensions are separate axes

The design point must keep the four axes explicit:

```text
a = (Us, Ts, Ud, Td, layout, residence, pipeline, processing_unit)
```

- `Us/Ts`: stage-space and stage-time organization;
- `Ud/Td`: data-space and data-time organization;
- `layout/residence/pipeline`: storage and handoff choices;
- `processing_unit`: the local radix/FFT arithmetic core.

The flow organization and the arithmetic core are orthogonal. A radix-2 core
can be used in several flow organizations; a different core does not replace
the need to choose residence, boundaries, and synchronization correctly.

### 2.3 Lowering is a legality layer, not a performance guess

Every candidate must pass a hardware-lowering contract before it is timed:

- supported Triton/XPU operations;
- pointer/indexing form accepted by the current compiler;
- alignment and layout requirements;
- UB/register capacity;
- ownership and synchronization semantics;
- launch/grid limits.

An unsupported candidate must be rejected as `not-lowered` or `infeasible`,
not silently scored by the performance model.

### 2.4 Search and measurement are separate evidence

The model ranks legal candidates; it does not prove performance. Report these
states separately:

```text
enumerated -> legal -> compiled -> correctness-passed -> measured
```

If a new configuration is slower than the established incumbent, retain the
row and explain the cause. Never hide a regression by deleting the point.

## 3. Audit Findings to Resolve

| Priority | Finding | Evidence | Required action |
|---|---|---|---|
| P0 | Device context is documented but not wired into the launcher | `docs/device-stack.md` requires `flag_gems.runtime.torch_device_fn.device(...)`; `src/xpu/runner.py` never enters it | Add one central context manager and a non-zero-output smoke gate |
| P0 | Documentation build is not publishable | `docs/results.md` links to a root `results/` file that MkDocs does not copy; `mkdocs build --strict` fails | Move/copy published JSON under `docs/data/` or use a GitHub blob URL; add CI |
| P1 | Public API silently accepts the wrong semantic input | `runner.py` uses `x.numel()` without rank/dtype/device checks | Reject non-1D/non-complex64/non-contiguous/wrong-device inputs; add explicit batch API later |
| P1 | Profile and design-space JSON are not consumed | `device.json`, `butterfly_space.json`, and `sizes.json` have no runtime loader | Add schema validation and record the effective candidate/profile in every run |
| P1 | Current kernel is stage-by-stage global-memory execution | DIT/DIF allocate ping-pong buffers and launch once per stage | Label it as baseline; implement and measure a real local-resident candidate before claiming mixed dataflow |
| P1 | Timing is synchronized API wall time, not pure kernel time | Timing includes Python dispatch, planar copies, allocations, output assembly, and synchronization | Rename the metric and add a separate device/kernel timing path where the runtime permits |
| P1 | Test matrix and implementation drift | Declared input patterns/block sizes are not all consumed; DIF ranges differ across config/tests/results | Generate tests from one manifest and record DIT/DIF errors independently |
| P1 | Vendor support boundary is hard-coded | `N <= 2048` is written into several files but not probed per run | Record the actual unsupported error for each runtime/shape/dtype |
| P2 | Hardware tests are not portable to ordinary CI | `pytest` imports torch at collection and cannot skip without the vendor stack | Separate host-only tests from an explicitly marked P800 test group |
| P2 | Device-stack conclusion is too broad | One rejected `x[idx]` form is used to conclude all fusion is impossible | Say “the tested Triton lowering is blocked” and enumerate alternative legal forms |
| P2 | Public layout wording is ambiguous | API accepts/returns `complex64`; planar is an internal kernel ABI | Document public layout and internal planar lowering separately |

## 4. Target Architecture Contract

Before optimizing, add a table like the following to the generated run
metadata. `current` values describe the baseline, not a claimed final design.

| Axis | Meaning | Current P800 baseline | Target evidence |
|---|---|---|---|
| `Us/Ts` | stage-space/time expansion | `1/1` | number of stages retained per execution group and reuse distance |
| `Ud/Td` | data-space/time expansion | one Triton block, no batch axis | active data tiles, batch folding, and per-tile reuse |
| `residence` | where intermediate state lives | GM between stages | register/UB/shared residence and boundary count |
| `pipeline` | overlap of producer/consumer work | none | trace or counter evidence of overlap |
| `P` | processing unit | radix-2 | radix-2/radix-4/other core legality and timing |
| `L` | layout | internal planar SoA | alignment, transpose/reorder, and public API layout |
| `H` | hardware lowering | Triton -> XMLIR -> XPU | compiler/runtime version and accepted instruction forms |
| `Q` | acceptance | relative error `< 1e-4` | correctness, timing protocol, and provenance |

The configuration file must not contain a parameter that the kernel ignores.
If a field is retained for future exploration, mark it explicitly as
`not-lowered` and exclude it from the measured winner set.

## 5. Implementation Phases

### Phase 0: correctness and safety gates

Deliverables:

1. Add `xpu_context()` around every device launch, including the Triton warm-up.
2. Make the smoke test write a sentinel input and verify a changed, finite
   output after synchronization; a zero error on an untouched buffer is not
   accepted as success.
3. Validate rank, dtype, device, contiguity, `block`, twiddle shape/device, and
   power-of-two length in `fft_dit`, `fft_dif`, and `bit_reverse_permute`.
4. Add host-only tests that run without torch/XPU and marked hardware tests that
   skip cleanly when the vendor stack is absent.
5. Make `mkdocs build --strict` pass.

Acceptance: the smoke command either executes a verified kernel or exits with a
clear prerequisite error; it never silently reports success.

### Phase 1: profile, candidate, and manifest plumbing

Deliverables:

1. Add a P800 hardware profile with evidence fields: device model, clusters,
   cores, UB/shared capacity, alignment, compiler/runtime versions, and device
   selection.
2. Add a loader for `H/G/A/P/L/F/Q` and a legal-candidate checker.
3. Emit a manifest for every benchmark containing commit, dirty state, profile
   hash, command, versions, shape, direction, block, warmup, trials, and
   correctness threshold.
4. Generate the correctness and benchmark matrix from the same manifest.

Acceptance: changing a profile or candidate changes the recorded effective
configuration, and an unsupported candidate is rejected before launch.

### Phase 2: fair baseline measurement

Keep the current stage-by-stage implementation as `baseline-global-stage` and
measure it under a declared contract:

- API wall time and (if available) device/kernel time as separate metrics;
- identical input semantics and precision for self and vendor paths;
- at least three independently ordered trials;
- warmup and repeat counts in the manifest;
- median/min-of-means rule stated in the report;
- DIT and DIF correctness/error columns separately;
- vendor unsupported rows retained with the actual error string.

Acceptance: the report can be regenerated from raw rows without hand-editing
the table, and no baseline comparison mixes unsupported or non-equivalent rows.

### Phase 3: physical dataflow optimization ladder

Do not start by changing radix. Use the following ladder and keep each stage as
an independently measured candidate:

1. **M0: baseline-global-stage** — current one-stage/one-GM-round-trip path.
2. **M1: static local micro-FFT** — fuse only a small, compile-time-indexed
   contiguous butterfly tile that the current Triton lowering accepts.
3. **M2: register/UB residence** — retain intermediate values for multiple
   stages; measure register pressure, spills, occupancy, and local capacity.
4. **M3: execution groups** — group several stages, materialize only group
   boundaries, and compare `G` and actual GM transactions with M0.
5. **M4: data-space/batch expansion** — map independent transforms or tiles
   across clusters and add `Td` only after the single-tile path is correct.
6. **M5: processing-core variants** — radix-4 or another local core only after
   the flow candidate is legal; compare arithmetic cost against the same flow.

For each step, report:

```text
G, Us, Ts, Ud, Td, local storage, boundary bytes,
kernel launches, GM read/write transactions, occupancy,
register/UB use, compile status, correctness, timing
```

### If dynamic gather/scatter remains blocked

Do not conclude that the architecture is impossible from one `x[idx]` failure.
Try the following in order:

1. compile-time/static index tables for small tiles;
2. contiguous tile loads followed by register permutation;
3. ping-pong registers or UB with explicit stage-local permutations;
4. multiple legal subgraphs connected by explicit boundaries;
5. only then classify the remaining form as a toolchain blocker.

The final report must distinguish:

```text
algorithmically possible
hardware-lowerable in this compiler
implemented and measured
```

## 6. Optimization Decision Rules

- Reduce high-level GM materialization before increasing arithmetic intensity
  through a more complicated butterfly core.
- Keep `flow organization` and `processing-unit selection` as separate columns
  in the candidate manifest.
- Use NCU-like counters or the closest P800 profiler evidence to explain every
  speed change; do not infer a memory benefit from elapsed time alone.
- Treat larger batch as a workload axis, not an automatic speedup. It can
  amortize launch overhead but can also reduce active independent tiles or
  expose the slowest role.
- Preserve the v0 baseline as an incumbent. A new selector must never choose a
  candidate slower than the incumbent for the same semantic point without an
  explicit opt-in.
- Keep unsupported, correctness-failing, and unmeasured candidates out of the
  production winner table, but retain them in the evidence archive.

## 7. Required Experiment Matrix

The first publishable convergence matrix should cover:

| Axis | Minimum range |
|---|---|
| direction | DIT and DIF |
| precision | fp32 first; other types only after explicit lowering support |
| length | `2^3..2^20`, with `2^12`, `2^14`, `2^16`, `2^18`, `2^20` highlighted |
| batch | `1, 4, 16, 64` once a batched path exists |
| block/tile | every declared legal block, not only the default |
| flow | baseline, local fused, grouped-boundary candidates |
| core | radix-2 baseline, then legal radix-4/other candidates |
| layout | public complex API vs internal planar lowering |

Every published table should have an adjacent plot and a link to raw rows. A
point without an external baseline is a coverage/correctness result, not a
speedup result.

## 8. Where to Read Before Implementing

Use these sections as the design references rather than guessing from the
current kernel:

- cuButterfly: [motivation](https://github.com/TruNcat3/cuButterfly/blob/master/docs/design/motivation.md),
  [architecture](https://github.com/TruNcat3/cuButterfly/blob/master/docs/design/architecture.md),
  [performance model](https://github.com/TruNcat3/cuButterfly/blob/master/docs/design/performance-model.md),
  and [profiling](https://github.com/TruNcat3/cuButterfly/blob/master/docs/development/profiling.md).
- Ascend-FFT: [motivation](https://github.com/TruNcat3/ascend-fft/blob/master/docs/design/motivation.md),
  [architecture](https://github.com/TruNcat3/ascend-fft/blob/master/docs/design/architecture.md),
  [measurement methodology](https://github.com/TruNcat3/ascend-fft/blob/master/docs/benchmarks/methodology.md),
  [results](https://github.com/TruNcat3/ascend-fft/blob/master/docs/benchmarks/results.md),
  and [configuration](https://github.com/TruNcat3/ascend-fft/blob/master/docs/reference/configuration.md).
- Kunlunxin-specific facts remain in
  [device-stack.md](../device-stack.md); do not copy Ascend resource values to
  P800. Only the reasoning and evidence structure is portable.

## 9. Server Reporting Template

Each exploration round should attach one Markdown summary and raw CSV/JSON
with the following fields:

```text
commit, device, runtime, profile_hash, N, batch, direction, precision,
flow_id, processing_unit, Us, Ts, Ud, Td, residence, layout,
compiled, correct, max_err, kernel_ms, api_ms, launches,
gm_read_bytes, gm_write_bytes, local_bytes, occupancy, notes
```

For every regression, write one sentence answering:

```text
Which term changed: launch cost, GM traffic, local capacity,
dependency/scoreboard wait, arithmetic core cost, or measurement noise?
```

## 10. Definition of Done

This migration round is complete only when:

- the documented launch path cannot silently no-op;
- strict docs build and host-only tests pass without a P800 installation;
- hardware tests are reproducible from a manifest;
- the baseline and at least one local-resident candidate have matched
  correctness and timing rows;
- the report clearly separates flow optimization, processing-unit selection, and
  hardware lowering;
- any remaining compiler limitation is backed by a minimal reproducer and is
  listed as a future/toolchain item rather than presented as a measured speed
  result.
