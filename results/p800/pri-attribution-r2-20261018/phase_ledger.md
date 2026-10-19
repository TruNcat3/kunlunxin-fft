# WP2 phase/launch ledger

- generated: 2026-10-19T00:31:14.051539+00:00
- source summary: results/p800/pri-postpromote-20261018/summary.json
- commit: 78308014ab9e3d71394a18fba07b68906ca4cf37
- evidence kinds: `fresh measured` (phase timers with extra synchronize — attribution only, headline protocol timings run with the ledger off), `analytic estimate` (cost_model launch families + real-compat op bytes), `summary-derived` (release medians). None of these are hardware performance counters or resident-kernel claims.

## N64B64:c2c-forward
- summary: selector 0.19552605226635933 ms / incumbent 0.6610259879380465 ms / vendor 0.031139003112912178 ms (source=table)
- phases[selector] (fresh measured (phase timers; extra synchronize added)): trial wall p50 0.2268 ms; plan-validate=0.0096ms, core-invoke=0.1738ms, epilogue-scale=0.0075ms
  - inter-phase host overhead (unattributed): 0.0379 ms
- phases[incumbent] (fresh measured (phase timers; extra synchronize added)): trial wall p50 0.6658 ms; plan-validate=0.0106ms, core-invoke=0.6038ms, epilogue-scale=0.0078ms
  - inter-phase host overhead (unattributed): 0.0412 ms
- vendor: wall p50 0.0196 ms (fresh measured (vendor host API wall; single call, extra synchronize added))
- plan creation: mean 0.1516 ms (fresh measured (host plan resolution, no device))
- launches[selector] (analytic estimate (cost_model.axes + real-compat op list)): family=m2_batch launches=2 gm_bytes=393216 flops=524288.0

## N64B64:r2c
- summary: selector 0.25718496181070805 ms / incumbent 0.7008849643170834 ms / vendor 0.031569041311740875 ms (source=table)
- phases[selector] (fresh measured (phase timers; extra synchronize added)): trial wall p50 0.3324 ms; plan-validate=0.0097ms, real-input-pack=0.0289ms, core-invoke=0.1760ms, epilogue-scale=0.0074ms, half-spectrum-crop=0.0554ms
  - inter-phase host overhead (unattributed): 0.0566 ms
- phases[incumbent] (fresh measured (phase timers; extra synchronize added)): trial wall p50 0.7582 ms; plan-validate=0.0105ms, real-input-pack=0.0300ms, core-invoke=0.5911ms, epilogue-scale=0.0080ms, half-spectrum-crop=0.0594ms
  - inter-phase host overhead (unattributed): 0.0579 ms
- vendor: wall p50 0.0184 ms (fresh measured (vendor host API wall; single call, extra synchronize added))
- plan creation: mean 0.0336 ms (fresh measured (host plan resolution, no device))
- launches[selector] (analytic estimate (cost_model.axes + real-compat op list)): family=m2_batch launches=2 gm_bytes=393216 flops=524288.0
  - real-compat op: real-input-pack (torch.complex) (in 32768 B, out 32768 B, analytic estimate)
  - real-compat op: half-spectrum crop (slice+contiguous) (in 32768 B, out 8448 B, analytic estimate)
  - real-compat op: normalization epilogue (scale) (in 32768 B, out 32768 B, analytic estimate)

## N128B64:c2c-forward
- summary: selector 0.6291971076279879 ms / incumbent 1.1218579020351171 ms / vendor 0.03674905747175217 ms (source=table)
- phases[selector] (fresh measured (phase timers; extra synchronize added)): trial wall p50 0.6290 ms; plan-validate=0.0099ms, core-invoke=0.5705ms, epilogue-scale=0.0078ms
  - inter-phase host overhead (unattributed): 0.0423 ms
- phases[incumbent] (fresh measured (phase timers; extra synchronize added)): trial wall p50 1.1231 ms; plan-validate=0.0106ms, core-invoke=1.1379ms, epilogue-scale=0.0083ms
  - inter-phase host overhead (unattributed): 0.0466 ms
- vendor: wall p50 0.0247 ms (fresh measured (vendor host API wall; single call, extra synchronize added))
- plan creation: mean 0.1637 ms (fresh measured (host plan resolution, no device))
- launches[selector] (analytic estimate (cost_model.axes + real-compat op list)): family=m2_batch launches=3 gm_bytes=917504 flops=1073152.0

## N2048B256:c2c-forward
- summary: selector 21.914728917181492 ms / incumbent 50.02492805942893 ms / vendor 1.4663790352642536 ms (source=table)
- phases[selector] (fresh measured (phase timers; extra synchronize added)): trial wall p50 21.2834 ms; plan-validate=0.0142ms, core-invoke=21.1872ms, epilogue-scale=0.0094ms
  - inter-phase host overhead (unattributed): 0.0802 ms
- phases[incumbent] (fresh measured (phase timers; extra synchronize added)): trial wall p50 49.7033 ms; plan-validate=0.0196ms, core-invoke=49.5086ms, epilogue-scale=0.0119ms
  - inter-phase host overhead (unattributed): 0.1156 ms
- vendor: wall p50 1.4057 ms (fresh measured (vendor host API wall; single call, extra synchronize added))
- plan creation: mean 0.2204 ms (fresh measured (host plan resolution, no device))
- launches[selector] (analytic estimate (cost_model.axes + real-compat op list)): family=m2_batch launches=7 gm_bytes=92274688 flops=74973184.0
- batch_fill N=2048 c2c-forward (summary-derived (release summary medians)): per-point growth 4.51x smallB->largeB, poor_batch_fill=True; per-point wall time from summary medians; fill proxy only — no worker/wave hardware counters exist on this stack

