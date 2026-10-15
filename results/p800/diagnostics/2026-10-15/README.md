# Diagnostics: 2026-10-15 review-round confirmation runs (Round B/C)

Raw session traces for the PR #1 review-round fixes and the confirmation
matrix. The measured data lives in `results/p800/confirm_*.json` (per-trial
`*_samples`, `*_p10`/`*_p90`, schema-v2 `rows_long`); these logs are the
unaggregated stdout of the sessions that produced it.

| File | What it is |
|---|---|
| `roundb-b1024.log` | Round B block-1024 session (commit `da251f9`, card 6): lowering-contract verdicts, per-case interleaved medians/kernel_ms/launches, M1–M5 and M5-ledger summaries, vendor `NOT IMPLEMENTED` stderr at `N>2048` (expected), `EXIT=0` |
| `roundb-b256.log` | Round B block-256 session, same sizes/protocol |
| `roundb-batch.log` | batch probe session (`B=1,4,16,64` x `{spread,folded}` at `N=64,4096,65536`, flows m0/m4/vendor) |
| `hw-suite.log` | `pytest -q -m hardware` (card 6) over the phase-3 size grid — `106 passed, 36 deselected, EXIT=0` |
| `selected-sessions.log` | stdout of the 3-session selector validation (`scripts/run_selected_sessions.py`, commit `bbafd7f`) — correctness gates, fallback probes, `rejected-before-launch` record, `EXIT=0` |
| `confirm_*.progress.json` | atomic per-case breadcrumbs written by bench during the sessions (`last_started`/`last_completed`, pid, timestamps) — evidence of the hang-mitigation path, not timing data |

## How to check the numbers against the logs

* Per-size medians in each `roundb-*.log` line must match the
  `<key>_ms` fields of the corresponding `results/p800/confirm_*.json` rows
  (the log prints the same values the JSON stores).
* `confirm_b1024.json` manifest: `commit=da251f9`, `dirty=false`,
  `trials=30`, `schema_version=2`. The b256/batch manifests record
  `dirty=true` only because the preceding run's output files were untracked
  in the same tree (same commit, no code difference).
* Regenerate any report offline (no device, no torch):

  ```bash
  python3 scripts/regen_markdown.py results/p800/confirm_b1024.json /tmp/out.md
  diff /tmp/out.md results/p800/confirm_b1024.md   # byte-identical
  ```

* Rebuild the selector table offline and diff against the committed one:

  ```bash
  python3 scripts/build_selector_table.py \
      results/p800/confirm_b1024.json results/p800/confirm_b256.json \
      --batch results/p800/confirm_batch.json --out /tmp/selector_table.json
  diff /tmp/selector_table.json config/selector_table.json
  ```

  (also pinned by `tests/test_selector.py::test_table_rebuild_is_deterministic`).
