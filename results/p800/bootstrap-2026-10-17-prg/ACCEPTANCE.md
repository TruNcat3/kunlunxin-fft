# PR-G full-chain acceptance — semantic key + stress-domain promotion (issue #8)

- date: 2026-10-17 (Asia/Shanghai, device `CUDA_VISIBLE_DEVICES=0`)
- device note: the pool now exposes 2 P800 devices (was 8); the
  measured environment hash is unchanged (`6b50bd316032c0fe` — the env
  fingerprint covers driver/toolchain/device model, not slot count).
- code: clean tree at `746261d` (committed before the run; `dirty=false`
  in every stamped manifest of this chain). `746261d` is the attribution
  fix that followed the first attempt of this chain (see "Development
  note" below).
- command (round 1, no promote):

  ```
  CUDA_VISIBLE_DEVICES=0 python3 scripts/bootstrap_profile.py \
      --output results/p800/bootstrap-2026-10-17-prg \
      --device 0 --calibrate --verify
  ```

- command (round 2, explicit promote; `--resume` re-executes nothing but
  the publish):

  ```
  CUDA_VISIBLE_DEVICES=0 python3 scripts/bootstrap_profile.py \
      --output results/p800/bootstrap-2026-10-17-prg \
      --device 0 --verify --resume --promote
  ```

## Stage timeline (log mtimes, +0800)

| stage | end | notes |
|---|---|---|
| probe | 23:00:57 | |
| check | 23:01:14 | correctness + smoke |
| calibrate | 23:10:29 | 7 rounds: bench b1024 63s, bench b256 60s, batch 115s, focus_a 65s, focus_b 66s, holdout 53s, **matrix_stress 110s** (3 sessions x 30 trials, 4 cells) |
| build | 23:10:38 | model 4s + table 4s: **44 entries** (40 legacy + 4 exact-semantic), `rejections=[]` |
| verify | 23:10:42 | strict manifest contract on 6 outputs + semantic exact-key smoke on the staged table |

Total wall: **~9 min 45 s**, exit 0, `state.json` =
`["probe","check","calibrate","build","verify"]`.

## Gate 1 — no `--promote` leaves production byte-identical

Pre-run baseline (`/tmp/opencode/pre_prg_sha.txt`):

```
config/selector_table.json: OK (eb1e5f388a92f2f2e268db49bc2d64571009f2fc6002706a6b13452dc212e10f)
config/calibration.json:    OK (cab0777ff984929f63b2ccf758d81d31d96c7e0fc422633531274c7626eed3e8)
```

## Gate 2 — explicit promote publishes exactly the verified artifacts

After `--resume --promote`: `config/selector_table.json` sha256 ==
`.staging/selector_table.json` (`d048ffe1df99623…`), production now
carries **44 entries incl. 4 semantic**; `config/calibration.json`
`c1c1ce4f25a0876…`. Environment binding unchanged
(`env_hash=6b50bd316032c0fe`, `profile_hash=a78b86793c7142a0`).

## Issue #8 PR-G gates (fresh measured unless marked summary-derived)

### Stress domain N=1024,B=256 fixed (the issue's must-fix cells)

Promotion-grade matrix cells (`.staging/matrix_stress`, 3 sessions x
30 trials, stamped `result_sha256`, embedded manifest, strict output
gate `[]`):

| cell | incumbent ms | oracle ms | winner | gain vs incumbent |
|---|---|---|---|---|
| N1024B256:c2c-forward | 23.452 | 10.567 | `oracle:m2_dif_t64` | 54.9% (2.22x) |
| N1024B256:c2c-inverse | 23.603 | 10.644 | `oracle:m2_dif_t64` | 54.9% (2.22x) |
| N1024B256:r2c | 23.854 | 11.183 | `oracle:m2_dif_t64` | 53.1% (2.14x) |
| N1024B256:c2r | 24.272 | 11.638 | `oracle:m2_dif_t64` | 52.1% (2.10x) |

All four clear the >= 2% guard; every winner row is
correctness-clean in all 3 sessions; `entry_contract_violations` on
the four built entries: none (batch=256 joins the published axis in
`config/butterfly_space.json`).

Post-promote independent re-measurement against the promoted table
(`results/p800/prg-postpromote-stress`, same protocol, summary-derived
from its raw sessions; `source_hash=746261d`, `dirty=true` because it
ran while the promote commit was pending, `result_sha256` stamped,
strict output gate `[]`):

| cell | selector ms | oracle ms | regret | selector source | guard 2% |
|---|---|---|---|---|---|
| N1024B256:c2c-forward | 10.671 | 10.552 | 1.13% | table | pass |
| N1024B256:c2c-inverse | 10.785 | 10.635 | 1.41% | table | pass |
| N1024B256:r2c | 11.553 | 11.136 | 3.75% | table | pass |
| N1024B256:c2r | 11.995 | 11.576 | 3.63% | table | pass |

**max stress regret 3.75% (gate <= 10%), mean 2.48%, failures=0** —
vs the frozen v2 baseline's 2.03–2.46x incumbent-oracle gap (regret
audit `prg-regret-audit.json`).

### Exact semantic key + tiers (fresh measured selection, host tests)

- Table entries carry `(transform_type, transform_direction,
  normalization)`; a semantic entry answers only its own triple, a
  legacy entry (no triple fields) answers every semantic as the old
  deployed path. Duplicate keys are the FULL key, so legacy and
  semantic entries coexist per shape (`tests/test_semantic_key.py`,
  `tests/test_matrix_builder.py`).
- Post-promote device smoke (fresh measured): all four semantics at
  N1024,B256 select `source=table`, `entry_tier=semantic`,
  `m2_dif_t64` geometry (`tile=64, rows=8, variant=spread`); the main
  cell N2048,B16 resolves via `entry_tier=legacy` with its own
  incumbent; unmeasured `dif` at the stress shape and out-of-matrix
  keys abstain; `plan_fft(r2c)` returns the table plan with the
  packed-compat marker.
- Per-semantic legality: `src/xpu/semantic.py` families (core flows
  + `SPECIALIZED_FLOWS` hook for PR-H) — the builder refuses any
  winner whose flow is not in that semantic's family; inverse/R2C/C2R
  entries are never produced from forward ordering (separate cells,
  separate guards, separate keys).

### Search budget (issue: <= 2 rounds, <= 24 candidates per semantic/shape)

- Measured oracle set per matrix cell: 6 bounded candidates
  (gate in `build_matrix_entries` refuses > 24).
- PR-G promotion happened in a single confirmation round beyond the
  audit (round 1 = `prg-confirm-20261017` reproduction, round 2 =
  this chain's promotion matrix); no rule fitting on holdout data
  (holdout feeds the cost model only, unchanged).

### Production set contains the old incumbents

The 40 legacy entries remain in the table (legacy tier); nothing in
the main/holdout/regression domains was re-keyed. Re-measurement in
this round changed winner identity or geometry on **7 of 40** legacy
entries vs the previous production table (block 1024→256 at N8/N16,
`rows` changes at B16, m5→m2 at N1024B1, etc.) — these are ordinary
fresh-round outcomes of the unchanged confirm protocol (every entry
still guard-gated and contract-checked in this run's build log
`rejections=[]`), not semantic-tier effects; the semantic tier only
added the 4 B256 entries.

## Development note (not evidence)

The first attempt of this chain (same command, code `7dab03b`) refused
all four stress cells with `rows lack flow attribution (flow_id=None)`;
the oracle plans never recorded flow/unit attribution. Fixed in
`746261d` (attribution now single-sourced in
`run_semantic_matrix._attribution`, rows record `plan.executed`, host
tests lock the mapping). The attempt's staging was discarded and
re-measured from scratch by this run.

## Reproduction

```
CUDA_VISIBLE_DEVICES=0 python3 scripts/bootstrap_profile.py \
    --output results/p800/bootstrap-2026-10-17-prg --device 0 \
    --calibrate --verify            # gate 1: production untouched
CUDA_VISIBLE_DEVICES=0 python3 scripts/bootstrap_profile.py \
    --output results/p800/bootstrap-2026-10-17-prg --device 0 \
    --verify --resume --promote     # gate 2: publish verified artifacts
CUDA_VISIBLE_DEVICES=0 python3 scripts/run_semantic_matrix.py \
    --scope stress --sessions 3 --trials 30 --warmup 5 \
    --out results/p800/prg-postpromote-stress
```
