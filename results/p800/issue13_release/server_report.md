# issue #13 server report — production selector wiring & coverage

Scope: production selector/plan/runner wiring for the WP1-promoted
exact cells (AscendFFT-aligned forward-C2C coverage organization),
plus release-matrix and coverage evidence. Hardware: Kunlunxin P800
(visible card 2), torch 2.9 + torch_xmlir + triton 3.0.0 XPU fork
(runtime lock: `environment.json`).

## Claim levels (per the issue's stop conditions)

| Claim | Status | Basis |
|---|---|---|
| **Bounded acceleration** | **supported** | 2 frozen exact cells passed the WP1 stability gate and are wired into the production composition table (`N2048B1:c2c_inverse` → `run_hybrid4`, `N2048B16:c2c_inverse` → `run_hybrid4_fused`); the WP3 16-cell release matrix passes ALL exit conditions: 192/192 correct, 0 silent fallback at the promoted keys, no incumbent regression anywhere, geomean(old/new) = **1.199**, worst cell `N1024B1:c2c_forward` at 0.989 (tie band). |
| **Broad coverage acceleration** | **NOT supported** | On the vendor-supported subset of the 49-cell forward-C2C grid: geomean(vendor/selector) = **0.071** (incumbent: 0.067), 40 loss / 2 win / 0 tie. The issue requires geomean > 1 for this wording; every negative point is published in `wp4_coverage.json`. |

## Work packages

- **WP0** — production contract & repro lock (all-path contract gate,
  real rank-1 plan→execute dispatch tests, evidence-dir sha256sum
  verify).
- **WP1** — 5 frozen cells × 4 sessions, bootstrap CI-low > 1.02 vs
  vendor AND incumbent + session spread ≤ 10%: **2/5 promoted**
  (both inverse at N=2048; +66.3% / +83.1% vs incumbent). 3 cells not
  promoted (pre-existing bimodal environment at tiny sizes, spread
  26–51% — retained, not promoted).
- **WP2** — per-entry semantic dispatch in the composition registry
  (no implicit batch/transform matching), `variant`/`executed`
  contract through `plan_fft` → `execute_fft_plan`, hybrid4 4-step
  production runners; 2 promoted keys stamped into
  `config/composition_table.json` (`selector_diff.json` v3).
- **WP3** — 16-cell release matrix (N∈{1024,2048} × B∈{1,16,64,256} ×
  both directions × 4 paths × 3 sessions). Exit conditions PASS.
  Promoted dual-baseline (vendor + incumbent, CI-low > 1.02) holds in
  every session; wins: N2048B1:inv **2.72×**, N2048B16:inv **6.15×**;
  all other cells tie; no regression (retained list: empty).
- **WP4** — coverage per the AscendFFT grid: 49 forward-C2C cells
  (N∈{64…4096} × B∈{1…4096}) + 12 long points
  (N∈{8192…65536} × B∈{1,16,256}). 492 measured rows, 0 incorrect,
  selector 0 fail-closed. Long slice: 12/12 correct in every session
  with resolved paths + memory telemetry (vendor cannot run any
  N ≥ 4096 point on this stack). Verdicts: 2 win (N64B16,
  N2048B16-composition), 40 loss, 7 vendor-unsupported (retained
  explicitly with per-session reasons). Fallback ratio **0.59**
  (29/49 cells resolve M0).
- **WP5** — see classification below; no optimization round.

## WP5: classification of the systematic negatives

| Class | Verdict | Evidence |
|---|---|---|
| **5. lowering / contract capability missing** | **primary** | 29/49 C2C cells abstain → M0 because the dit selector table only covers profiled design points; losses there are 10–100× vs vendor. Fix = a new design-point profiling campaign (issue10-scale), not a two-round change. |
| 3. local core throughput | secondary | Table-resolved small-N cells (e.g. N64B1: dit 0.41 ms vs vendor 0.021 ms) lose to the vendor library's tuned small-FFT paths. Vendor-class small-FFT tuning is an open-ended core track. |
| 1/2/4. dispatch/GM/tail | not selected | No dominant signature: dispatch overhead is bounded (WP3 old-vs-new identical below ~0.5% on identical-plan cells), and heavy-cell traces show compute-bound M0 rows, not layout/tail artifacts. |

**Decision (per the issue's stop conditions):** no class closes with a
clear bounded benefit inside ≤ 2 implementation rounds. Negative
points stay published; capability coverage (N=64…512 × B=4…4096 +
long-size composition entries) and the small-N core track move to
future work. The issue converges on the bounded-acceleration claim.

## Measurement notes (read before reusing the numbers)

1. **Protocol fix run 1 → run 2 (WP3).** The holdout warmup (5 calls)
   left a 2-step ramp inside the timed window on 42/144 plan rows
   (steady state only after up to ~144 timed calls). With one
   subprocess per (cell, path), the ramp landed asymmetrically and
   fabricated a "confirmed regression" at `N2048B1:forward` — a cell
   whose old/new dispatch is byte-identical. Run 2 warms adaptively
   (16-call batches until the last 3 means are within 2% of the
   running minimum, min 320 / cap 800 calls): 0/192 rows show a
   residual ramp (worst half-window ratio 1.010). **Both runs are
   archived** (`run1-warmup5/` includes the exact run-1 producer,
   sha256-verified against its recorded producer_sha256).
2. **Vendor capability map (XMLIR).** `torch.fft` last-dim ops work
   for N ≤ 2048 at all measured batches and return
   `[NOT IMPLEMENTED]` for N ≥ 4096 (all 7 N4096 cells + all 12 long
   points, in every session). Vendor-unsupported points are kept
   explicitly in the verdict table — never dropped from the
   denominator.
3. **Sub-ms bimodality.** Cells with per-call times ≲ 1 ms carry
   pre-existing multi-modal card-state variance across processes
   (e.g. vendor at N64B1 measured 0.219 ms in one session and
   0.021 ms in another). The 10–100× fallback losses dwarf this
   noise and are robust; borderline table-resolved cells (e.g.
   N2048B1) can flip verdict band under different card state and are
   flagged low-confidence here. WP3 gates are unaffected: they are
   old-vs-new comparisons inside the same process/session.
4. **Heavy-cell downshift (WP4 only).** Cells with ≥ 30 ms/call
   (M0-fallback long rows, e.g. 4096×4096 at ~1.6 s/call) use a
   bounded protocol variant (64–96 warmup calls, 3×32 sync-per-trial
   trials), recorded per row as `protocol_variant`. At that scale the
   launch/clock ramp cannot contaminate the median.

## Artifacts

- `issue13_summary.json` — one entry point (claims, per-WP status,
  session summaries, WP5 record).
- `wp1_promotion.json` / `wp1_raw.csv` / `wp1_sessions_raw.json`
- `selector_diff.json` — selector state before/after WP2 wiring
- `wp3_release.json` / `wp3_raw.csv` / `wp3_sessions_raw.json` +
  `run1-warmup5/` (protocol-fix trail)
- `wp4_coverage.json` / `wp4_raw.csv` / `wp4_sessions_raw.json`
- `legacy_composition_table.json` — pre-issue13 snapshot used by the
  incumbent_old measurement path
- `manifest.json` / `SHA256SUMS` / `environment.json` /
  `repro_command.txt` — verify with
  `cd results/p800/issue13_release && sha256sum -c SHA256SUMS`
