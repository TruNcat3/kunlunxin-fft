# P1 fix + post-promote full-domain confirmation acceptance (issue #8)

- date: 2026-10-18 (Asia/Shanghai, device `CUDA_VISIBLE_DEVICES=0`)
- commits (clean-tree gated, dirty=false in every stamped manifest):
  - `1dc7775` — review fixes P1-A..D: runner closure (`executed`
    end-to-end), composition semantic gate, raw<->summary
    cross-check + `cases_sha256`, un-bypassable promote re-verify
  - `c04fd9f` — promote: P1-fixed chain table (production
    `selector_table.json` `f172147dc2c3e875`, calibration
    `a3d98cd7d0312d91`, gate 2 hash match against staged artifacts)
  - `7401160` — this confirmation run itself exposed two defects and
    they are fixed in the run code (see "Defects found by this run"):
    scope=all double-scheduling, and a divergent/unstamped
    `result_sha256`
- command:

  ```
  CUDA_VISIBLE_DEVICES=0 python3 scripts/run_semantic_matrix.py \
      --sessions 3 --trials 30 --scope all \
      --out results/p800/prg-fullconfirm-20261018 --snapshot
  ```

- host suite after the fix: **312 passed**, 3 skipped, 336 deselected
  (hardware tests deselected), `py_compile` clean (no ruff/flake8 on
  this box)

## Device smoke — runner closure on hardware (fresh measured)

Production path at the stress shape, all four semantics, real XPU
tensors (`plan_fft` -> `execute_fft_plan`, device alias `cuda`):

| semantic | plan source | plan.executed | execution |
|---|---|---|---|
| c2c-forward | table | `fft_dif_matrix_batch` | ok, (256,1024) complex64 |
| c2c-inverse | table | `fft_dif_matrix_batch` | ok, (256,1024) complex64 |
| r2c | table | `fft_dif_matrix_batch` | ok, (256,513) complex64 from float32 (256,1024) |
| c2r | table | `fft_dif_matrix_batch` | ok, (256,1024) float32 from complex64 (256,513) |

P1-A is closed end to end: the DIF batch winner stays named
`fft_dif_matrix_batch` and the deployed runner equals the measured one.

## Strict manifest gates (all `[]`, this run)

| gate | result |
|---|---|
| `validate_manifest(manifest.json, strict=True)` | `[]` |
| `validate_raw_sessions(dir, manifest, strict=True)` | `[]` (0 duplicate keys over 128 labels x 3 sessions) |
| `validate_benchmark_output(dir, strict=True)` | `[]` |
| snapshot `manifest.json` strict | `[]` |
| snapshot embedded summary manifest strict | `[]` (`result_sha256 cd8310c256de…`, `cases_sha256 1975d53a1bdf…`) |

## Defects found by this run (fixed in `7401160`)

1. **scope=all double-scheduling**: `main + apps + large` concatenated
   without dedupe, so the four OFDM/radar proxy shapes shared by both
   scopes were measured twice per session — 14928 duplicate
   `(session,label,trial,candidate)` raw keys over 16 labels, and 144
   case entries for 128 unique cells. `_scope_cells()` now
   order-preserving-dedupes (32 unique cells -> 128 cases). The
   superseded pre-fix run lives on as immutable snapshot
   `releases/20261018-150616-c04fd9f3-semantic-matrix` (failed the
   strict gate; never promoted).
2. **`result_sha256` never reached the source manifest**, and
   `_write_snapshot` stamped a local bytes-only variant that diverged
   from `manifest_contract.result_sha256` (filenames not hashed).
   `main()` now stamps both `manifest.json` and the summary's embedded
   manifest with the validator's own hash right after aggregation;
   `_write_snapshot` uses the same function.

## Issue gates (fresh measured summary, joined against the frozen
audit `results/p800/diagnostics/2026-10-17/pr-g-regret-audit.json`)

| domain | cells | over 10% | max regret | mean regret (audit -> now) | status |
|---|---|---|---|---|---|
| 主域 main | 16 | **0** | **-10.38%** | -24.00% -> -19.31% | **PASS** (target <= 10%) |
| 压力域 stress (N1024B256) | 4 | **0** | **2.35%** | 111.42% -> 1.40% | **PASS** (must-fix, issue line 163) |
| 留出域 holdout | 8 | 8 | 75.19% | 65.07% -> 66.95% | negative result recorded; abstain on holdout allowed (issue line 61) |
| 回归域 regression | 28 | 21 | 193.34% | 62.93% -> 62.29% | negative result recorded (issue line 162) |
| background | 72 | 64 | 222.10% | 66.86% -> 67.22% | negative result recorded |

Stress detail (the issue's must-fix cells, both sessions-derived from
this run's raw):

| cell | selector ms | oracle ms | regret | selector source | guard 2% | selector/incumbent |
|---|---|---|---|---|---|---|
| N1024B256:c2c-forward | 10.968 | 10.717 | 2.35% | table | pass | 0.467 (2.14x faster) |
| N1024B256:c2c-inverse | 11.022 | 10.826 | 1.80% | table | pass | 0.469 (2.13x faster) |
| N1024B256:r2c | 11.661 | 11.563 | 0.85% | table | pass | 0.479 (2.09x faster) |
| N1024B256:c2r | 12.113 | 12.040 | 0.61% | table | pass | 0.505 (1.98x faster) |

Audit baseline for the same four cells: 103.0–117.8% regret, all
`selector_source=abstain`. Delta: **-110 pp average**, all four guards
pass, all winners `oracle:m2_dif_t64` deployed as
`fft_dif_matrix_batch`.

### P1-B in action (composition semantic gate)

`N2048B16:c2c-inverse` moved `composition -> table`
(regret -76.31% -> -16.94%, still passing): the old table let inverse
ride a forward+backward composition entry; the P1-B gate restricts
composition to `c2c+forward+backward` exactly. Its forward sibling
stays `composition` (-83.34%) — both cells pass.

## Overall matrix metrics (fresh measured, 128 unique cells)

- win/tie/loss vs vendor: **4 / 0 / 100** (104 vendor-supported;
  72 vendor-unsupported preserved as negative points)
- geomean speedup vs vendor 0.0883, worst 0.0231 — the vendor
  `torch.fft` path still dominates most sizes; this is the honest
  current state and is exactly what PR-H (dedicated real flows) and
  PR-I (big-batch targeted optimization) are scoped to attack
- selector regret mean 0.5325 / max 2.221 (issue baseline: 0.470 /
  2.34 — same audit cell set per-domain deltas are the comparison
  table above)

## Immutable snapshot

`results/p800/releases/20261018-154943-7401160e-semantic-matrix/`

- commit `7401160ee331`, `dirty=false`, `result_sha256 cd8310c256de…`
- gzipped raw sessions (canonical evidence), summary/csv/figure,
  repro command; `_write_snapshot` refuses overwrite (immutability)
- supersedes `releases/20261018-150616-c04fd9f3-semantic-matrix`
  (pre-fix, strict-gate failure, kept for traceability)

## Evidence classes

- fresh measured: this run's summaries, stress/main gate numbers,
  device smoke
- summary-derived: per-domain audit joins (audit file is marked
  summary-derived at its source)
- analytic estimate: none used in any gated number (`gm_bytes_kind`
  recorded as estimate-only in the manifest)
