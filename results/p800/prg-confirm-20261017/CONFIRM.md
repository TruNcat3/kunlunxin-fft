# PR-G step 2: independent confirmation of the fast-oracle abstain points

- date: 2026-10-17, device `CUDA_VISIBLE_DEVICES=6`
- code: clean tree `864f6d2` (`manifest.json`: `source_hash=864f6d2…`,
  `dirty=false`, `env_hash=6b50bd316032c0fe`, `profile_hash=a78b86793c7142a0`)
- command:

  ```
  CUDA_VISIBLE_DEVICES=6 python3 scripts/run_semantic_matrix.py \
      --scope regret --sessions 3 --trials 30 \
      --out results/p800/prg-confirm-20261017
  ```

- protocol: `semantic-matrix-v1:plan-reuse:interleave:correctness-first`,
  identical to the frozen v2 release — 3 independent serial-process
  sessions, 30 interleaved trials/case, correctness-first (1e-4),
  incumbent/selector/bounded-oracle/vendor in one interleave
- evidence class: **fresh measured** (rows in `raw_session*.jsonl`,
  aggregates in `summary.json`)
- scope: 7 shapes x 4 semantics = 28 cells — the fixed-domain abstain
  cells whose v2 oracle measured >= 1.10x vs the incumbent (N64B1,
  N512B64, N2048B4, N1024B256, N16384B1, N16384B16) plus N2048B256 as
  background evidence (same 2x-class gap, no gate)

## Result: every v2 gap reproduces under independent sessions

selector == incumbent (source=abstain) on all 28 cells; the bounded
oracle stays clearly faster; per-cell regret (selector/oracle - 1) moves
by <= 0.06 vs the frozen v2 values, typically <= 0.03:

| cell group | v2 oracle/incumbent | confirm oracle/incumbent | verdict |
|---|---|---|---|
| N1024B256 (stress, must-fix) | 2.03-2.17x | **2.03-2.20x** | reproduced, all 4 semantics |
| N2048B256 (background) | 2.34-2.41x | 2.38-2.46x | reproduced |
| N512B64 / N2048B4 (holdout) | 1.44-1.75x | 1.44-1.80x | reproduced — evidence only, never used to fit candidates or rules |
| N64B1 (regression) | 1.38-3.08x | 1.41-2.93x | reproduced |
| N16384B1/B16 (regression) | 1.49-1.88x | 1.47-1.89x | reproduced; vendor unsupported (as in v2) |

`summary.json` failures = 24, all `vendor-unsupported` N16384 records
(8 cells x 3 sessions); zero correctness failures for any measured
candidate.

## Consequences

- The N1024B256 high-regret gap (~2.0-2.2x) is measured fact, not
  search noise: PR-G may proceed to per-semantic candidate families and
  exact-key promotion against this same-protocol incumbent.
- Holdout cells are confirmed for the record only; their numbers must
  not shape candidate rules (issue #8 holdout clause).
- Next: semantic-aware candidate/selector key (step 3).
