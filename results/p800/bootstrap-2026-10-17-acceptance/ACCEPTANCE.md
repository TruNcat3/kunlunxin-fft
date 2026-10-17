# WP0 full-chain bootstrap acceptance (issue #8 PR-F)

- date: 2026-10-17 (Asia/Shanghai, device `CUDA_VISIBLE_DEVICES=6`)
- code: clean tree at `dadd1d7` (committed before the run; `dirty=false` in every stamped manifest). A `docs/legal` commit (`b17a4ee`) landed on the remote mid-session; this branch was rebased on top of it (`697308a` = same content for every file this run produced or verified — only LICENSE/docs/gen_focus_report differ, none of them touch the measurement path). The stamped manifests keep the original `dadd1d7` id because that is the tree that was actually measured.
- command (round 1, no promote):

  ```
  CUDA_VISIBLE_DEVICES=6 python3 scripts/bootstrap_profile.py \
      --output results/p800/bootstrap-2026-10-17-acceptance --calibrate --verify
  ```

- command (round 2, explicit promote; `--resume` re-executes nothing but the publish):

  ```
  CUDA_VISIBLE_DEVICES=6 python3 scripts/bootstrap_profile.py \
      --output results/p800/bootstrap-2026-10-17-acceptance \
      --calibrate --verify --resume --promote
  ```

## Stage timeline (log mtimes, +0800)

| stage | end | notes |
|---|---|---|
| probe | 20:51:55 | started ~20:51:45 |
| check | 20:52:13 | correctness + smoke |
| calibrate | 20:59:38 | 6 rounds: bench b1024 63s, bench b256 60s, focus_a 137s, focus_b 65s, batch 65s, holdout 53s |
| build | 20:59:46 | model 4s + table 4s |
| verify | 20:59:50 | strict manifest contract on all 6 outputs + selector smoke on the staged table |

Total wall: **~8 min 10 s** (20:51:40 → 20:59:50), exit 0, `state.json` =
`["probe","check","calibrate","build","verify"]`, no failure entries.
Prompt: `staged only: production config/ untouched`.

## Gate 1 — no `--promote` leaves production byte-identical

`sha256sum -c` against the pre-run baseline (`/tmp/opencode/pre_promote_sha.txt`,
recorded before any promote, and equal to the committed `HEAD` files):

```
config/selector_table.json:    OK   (acc1f21c9c93fe20589993f22589ccc24e5b08db05be29d63321d9b63a6b5449)
config/composition_table.json: OK   (49e5f41b4a3a88b06d3e674cddecf579216a3539a8d774aa3d25afa4ae70a725)
```

## Gate 2 — explicit promote publishes exactly the verified artifacts

- `config/selector_table.json` sha256 == `.staging/selector_table.json`
  `eb1e5f388a92f2f2e268db49bc2d64571009f2fc6002706a6b13452dc212e10f`
- `config/calibration.json` sha256 == `.staging/calibration.json`
  `cab0777ff984929f63b2ccf758d81d31d96c7e0fc422633531274c7626eed3e8`
- `config/composition_table.json` untouched: `49e5f41b…` (unchanged since PR-E)
- diff vs previous production: `selector_table.json` 230 insertions / 231
  deletions (same 40-entry shape, re-measured medians), `calibration.json`
  30 lines — no structural drift
- post-promote smoke on the production file (device):
  env gate `6b50bd316032c0fe` passes, **40/40 exact-key selects return
  `source=table` (0 abstain)**, out-of-matrix `N=3000,B=1` abstains with
  the structural reason

Every stamped manifest binds `commit=dadd1d7`, `dirty=false`, `env_hash=6b50bd316032c0fe`,
`profile_hash=a78b86793c7142a0`; `verify` additionally checks
`result_sha256 == rows_sha256(payload)` per output.

## Failure-recovery path exercised (development rounds, same chain)

Three real full-chain failures were found and fixed before this clean run
(all fixes in `dadd1d7`):

1. `calibrate` → `stamp_result_hashes` aborted on bench
   `confirm_*.progress.json` checkpoints ("no manifest to stamp"); the
   stamp/verify globs now exclude `*.progress.json`.
2. `--resume` re-ran `run_batch_release` over a finished staging run
   ("already holds a finished run"); calibrate re-runs now pass `--overwrite`.
3. `build_selector_table` refused confirm manifests missing
   `timing_scope`/`protocol_id` (bench pre-dated the contract); `bench.py`
   now fills all `REQUIRED_MANIFEST_KEYS`.

Each round used `--resume` against `results/p800/bootstrap-2026-10-17/`
(dev output, gitignored); completed stages were skipped, only the failing
stage re-ran.

## Companion evidence

- WP2 phase/launch ledger: `../diagnostics/2026-10-17/phase_ledger.{json,md}`
  (device attribution; headline timings run with the ledger off)
- hardware tests: `tests/test_ledger_hw.py -m hardware` 6/6 passed
- host suite at this commit: 254 passed, 13 skipped, 11 deselected
