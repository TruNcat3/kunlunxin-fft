# WP2 phase/launch ledger

- generated: 2026-10-17T12:46:50.163785+00:00
- source summary: /workspace/kunlunxin-fft/results/p800/releases/20261017-wp2-semantic-matrix-v2/summary.json
- commit: f536c7176c8e1f7ac7e28b72d58ed23fdec4f3ca
- evidence kinds: `fresh measured` (phase timers with extra synchronize — attribution only, headline protocol timings run with the ledger off), `analytic estimate` (cost_model launch families + real-compat op bytes), `summary-derived` (release medians). None of these are hardware performance counters or resident-kernel claims.

## N1024B256:r2c
- summary: selector 23.917904938571155 ms / incumbent 23.919534985907376 ms / vendor 0.3947810037061572 ms (source=abstain)
- phases[selector] (fresh measured (phase timers; extra synchronize added)): trial wall p50 24.1681 ms; plan-validate=0.0147ms, real-input-pack=0.0452ms, core-invoke=23.8708ms, epilogue-scale=0.0096ms, half-spectrum-crop=0.1478ms
  - inter-phase host overhead (unattributed): 0.0957 ms
- phases[incumbent] (fresh measured (phase timers; extra synchronize added)): trial wall p50 24.1867 ms; plan-validate=0.0146ms, real-input-pack=0.0485ms, core-invoke=23.9259ms, epilogue-scale=0.0100ms, half-spectrum-crop=0.1558ms
  - inter-phase host overhead (unattributed): 0.0971 ms
- vendor: wall p50 0.3471 ms (fresh measured (vendor host API wall; single call, extra synchronize added))
- plan creation: mean 0.0815 ms (fresh measured (host plan resolution, no device))
- launches[selector] (analytic estimate (cost_model.axes + real-compat op list)): family=incumbent_m4_folded launches=10 gm_bytes=41943040 flops=7864320.0
  - real-compat op: real-input-pack (torch.complex) (in 2097152 B, out 2097152 B, analytic estimate)
  - real-compat op: half-spectrum crop (slice+contiguous) (in 2097152 B, out 525312 B, analytic estimate)
  - real-compat op: normalization epilogue (scale) (in 2097152 B, out 2097152 B, analytic estimate)
- batch_fill N=1024 r2c (summary-derived (release summary medians)): per-point growth 0.23x smallB->largeB, poor_batch_fill=False; per-point wall time from summary medians; fill proxy only — no worker/wave hardware counters exist on this stack

## N1024B256:c2r
- summary: selector 24.286748026497662 ms / incumbent 24.28027696441859 ms / vendor 0.4744089674204588 ms (source=abstain)
- phases[selector] (fresh measured (phase timers; extra synchronize added)): trial wall p50 24.2020 ms; plan-validate=0.0182ms, hermitian-rebuild=0.4972ms, inverse-conj=0.0638ms, core-invoke=23.2480ms, epilogue-scale=0.2091ms, real-output-cast=0.0300ms
  - inter-phase host overhead (unattributed): 0.1251 ms
- phases[incumbent] (fresh measured (phase timers; extra synchronize added)): trial wall p50 24.0802 ms; plan-validate=0.0164ms, hermitian-rebuild=0.4664ms, inverse-conj=0.0577ms, core-invoke=23.2430ms, epilogue-scale=0.1793ms, real-output-cast=0.0262ms
  - inter-phase host overhead (unattributed): 0.1099 ms
- vendor: wall p50 0.4224 ms (fresh measured (vendor host API wall; single call, extra synchronize added))
- plan creation: mean 0.0920 ms (fresh measured (host plan resolution, no device))
- launches[selector] (analytic estimate (cost_model.axes + real-compat op list)): family=incumbent_m4_folded launches=10 gm_bytes=41943040 flops=7864320.0
  - real-compat op: hermitian-rebuild (empty+flip+write) (in 525312 B, out 2097152 B, analytic estimate)
  - real-compat op: conjugate materialize (in 2097152 B, out 2097152 B, analytic estimate)
  - real-compat op: real-output cast (y.real) (in 2097152 B, out 1048576 B, analytic estimate)
  - real-compat op: normalization epilogue (scale) (in 2097152 B, out 2097152 B, analytic estimate)
- batch_fill N=1024 c2r (summary-derived (release summary medians)): per-point growth 0.12x smallB->largeB, poor_batch_fill=False; per-point wall time from summary medians; fill proxy only — no worker/wave hardware counters exist on this stack

## N1024B256:c2c-inverse
- summary: selector 23.513294057920575 ms / incumbent 23.43298599589616 ms / vendor 0.5192189710214734 ms (source=abstain)
- phases[selector] (fresh measured (phase timers; extra synchronize added)): trial wall p50 23.5063 ms; plan-validate=0.0152ms, inverse-conj=0.0592ms, core-invoke=23.1814ms, epilogue-scale=0.1737ms
  - inter-phase host overhead (unattributed): 0.0852 ms
- phases[incumbent] (fresh measured (phase timers; extra synchronize added)): trial wall p50 23.5618 ms; plan-validate=0.0151ms, inverse-conj=0.0599ms, core-invoke=23.1818ms, epilogue-scale=0.1745ms
  - inter-phase host overhead (unattributed): 0.0861 ms
- vendor: wall p50 0.4699 ms (fresh measured (vendor host API wall; single call, extra synchronize added))
- plan creation: mean 0.1671 ms (fresh measured (host plan resolution, no device))
- launches[selector] (analytic estimate (cost_model.axes + real-compat op list)): family=incumbent_m4_folded launches=10 gm_bytes=41943040 flops=7864320.0
  - real-compat op: inverse conjugate materialize (in 2097152 B, out 2097152 B, analytic estimate)
  - real-compat op: normalization epilogue (scale) (in 2097152 B, out 2097152 B, analytic estimate)
- batch_fill N=1024 c2c-inverse (summary-derived (release summary medians)): per-point growth 0.18x smallB->largeB, poor_batch_fill=False; per-point wall time from summary medians; fill proxy only — no worker/wave hardware counters exist on this stack

## N1024B256:c2c-forward
- summary: selector 23.380086990073323 ms / incumbent 23.339277016930282 ms / vendor 0.5185490008443594 ms (source=abstain)
- phases[selector] (fresh measured (phase timers; extra synchronize added)): trial wall p50 23.3377 ms; plan-validate=0.0157ms, core-invoke=23.2280ms, epilogue-scale=0.0100ms
  - inter-phase host overhead (unattributed): 0.0786 ms
- phases[incumbent] (fresh measured (phase timers; extra synchronize added)): trial wall p50 23.3932 ms; plan-validate=0.0150ms, core-invoke=23.2735ms, epilogue-scale=0.0099ms
  - inter-phase host overhead (unattributed): 0.0737 ms
- vendor: wall p50 0.4698 ms (fresh measured (vendor host API wall; single call, extra synchronize added))
- plan creation: mean 0.1663 ms (fresh measured (host plan resolution, no device))
- launches[selector] (analytic estimate (cost_model.axes + real-compat op list)): family=incumbent_m4_folded launches=10 gm_bytes=41943040 flops=7864320.0
- batch_fill N=1024 c2c-forward (summary-derived (release summary medians)): per-point growth 0.25x smallB->largeB, poor_batch_fill=False; per-point wall time from summary medians; fill proxy only — no worker/wave hardware counters exist on this stack

