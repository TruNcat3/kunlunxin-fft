# WP2 phase/launch ledger

- generated: 2026-10-18T11:54:54.065985+00:00
- source summary: results/p800/prg-fullconfirm-20261018/summary.json
- commit: c61350996dbcdcfecc1450d23e3c4a32ac689058
- evidence kinds: `fresh measured` (phase timers with extra synchronize — attribution only, headline protocol timings run with the ledger off), `analytic estimate` (cost_model launch families + real-compat op bytes), `summary-derived` (release medians). None of these are hardware performance counters or resident-kernel claims.

## N64B64:c2c-forward
- summary: selector 0.6349770119413733 ms / incumbent 0.6382070714607835 ms / vendor 0.029699993319809437 ms (source=table)
- phases[selector] (fresh measured (phase timers; extra synchronize added)): trial wall p50 0.6858 ms; plan-validate=0.0109ms, core-invoke=0.6305ms, epilogue-scale=0.0082ms
  - inter-phase host overhead (unattributed): 0.0434 ms
- phases[incumbent] (fresh measured (phase timers; extra synchronize added)): trial wall p50 0.6829 ms; plan-validate=0.0142ms, core-invoke=0.6243ms, epilogue-scale=0.0083ms
  - inter-phase host overhead (unattributed): 0.0426 ms
- vendor: wall p50 0.0196 ms (fresh measured (vendor host API wall; single call, extra synchronize added))
- plan creation: mean 0.1653 ms (fresh measured (host plan resolution, no device))
- launches[selector] (analytic estimate (cost_model.axes + real-compat op list)): family=incumbent_m4_folded launches=6 gm_bytes=393216 flops=73728.0
- batch_fill N=64 c2c-forward (summary-derived (release summary medians)): per-point growth 0.02x smallB->largeB, poor_batch_fill=False; per-point wall time from summary medians; fill proxy only — no worker/wave hardware counters exist on this stack

## N128B64:c2c-forward
- summary: selector 1.1171159567311406 ms / incumbent 1.1148559860885143 ms / vendor 0.03709900192916393 ms (source=abstain)
- phases[selector] (fresh measured (phase timers; extra synchronize added)): trial wall p50 1.2018 ms; plan-validate=0.0112ms, core-invoke=1.1413ms, epilogue-scale=0.0089ms
  - inter-phase host overhead (unattributed): 0.0441 ms
- phases[incumbent] (fresh measured (phase timers; extra synchronize added)): trial wall p50 1.2007 ms; plan-validate=0.0106ms, core-invoke=1.1387ms, epilogue-scale=0.0083ms
  - inter-phase host overhead (unattributed): 0.0419 ms
- vendor: wall p50 0.0250 ms (fresh measured (vendor host API wall; single call, extra synchronize added))
- plan creation: mean 0.1584 ms (fresh measured (host plan resolution, no device))
- launches[selector] (analytic estimate (cost_model.axes + real-compat op list)): family=incumbent_m4_folded launches=7 gm_bytes=917504 flops=172032.0
- batch_fill N=128 c2c-forward (summary-derived (release summary medians)): per-point growth 0.03x smallB->largeB, poor_batch_fill=False; per-point wall time from summary medians; fill proxy only — no worker/wave hardware counters exist on this stack

## N2048B256:c2c-forward
- summary: selector 49.30968303233385 ms / incumbent 49.341271980665624 ms / vendor 1.4647298958152533 ms (source=abstain)
- phases[selector] (fresh measured (phase timers; extra synchronize added)): trial wall p50 49.7918 ms; plan-validate=0.0191ms, core-invoke=49.6309ms, epilogue-scale=0.0113ms
  - inter-phase host overhead (unattributed): 0.1030 ms
- phases[incumbent] (fresh measured (phase timers; extra synchronize added)): trial wall p50 49.7489 ms; plan-validate=0.0198ms, core-invoke=49.5940ms, epilogue-scale=0.0120ms
  - inter-phase host overhead (unattributed): 0.1128 ms
- vendor: wall p50 1.4050 ms (fresh measured (vendor host API wall; single call, extra synchronize added))
- plan creation: mean 0.2166 ms (fresh measured (host plan resolution, no device))
- launches[selector] (analytic estimate (cost_model.axes + real-compat op list)): family=incumbent_m4_folded launches=11 gm_bytes=92274688 flops=17301504.0
- batch_fill N=2048 c2c-forward (summary-derived (release summary medians)): per-point growth 0.47x smallB->largeB, poor_batch_fill=False; per-point wall time from summary medians; fill proxy only — no worker/wave hardware counters exist on this stack

## N4096B16:c2c-forward
- summary: selector 7.252955925650895 ms / incumbent 7.257665973156691 ms / vendor None ms (source=table)
- phases[selector] (fresh measured (phase timers; extra synchronize added)): trial wall p50 7.3670 ms; plan-validate=0.0124ms, core-invoke=7.2768ms, epilogue-scale=0.0087ms
  - inter-phase host overhead (unattributed): 0.0539 ms
- phases[incumbent] (fresh measured (phase timers; extra synchronize added)): trial wall p50 7.3381 ms; plan-validate=0.0118ms, core-invoke=7.2788ms, epilogue-scale=0.0087ms
  - inter-phase host overhead (unattributed): 0.0537 ms
- vendor: unavailable (RuntimeError: [NOT IMPLEMENTED]: error code= 4(at /home/agent/xpytorch/workspace/8ba19c15-fc11-4377-85f3-84daf8eaab09/XMLIR/python/torch_xmlir/csrc/aten_capture/eager_customized/fft_fft.cpp:40))
- plan creation: mean 0.1943 ms (fresh measured (host plan resolution, no device))
- launches[selector] (analytic estimate (cost_model.axes + real-compat op list)): family=incumbent_m4_folded launches=12 gm_bytes=12582912 flops=2359296.0

## N65536B16:c2c-forward
- summary: selector 136.77618990186602 ms / incumbent 138.8867850182578 ms / vendor None ms (source=table)
- phases[selector] (fresh measured (phase timers; extra synchronize added)): trial wall p50 135.9234 ms; plan-validate=0.0261ms, core-invoke=135.7164ms, epilogue-scale=0.0153ms
  - inter-phase host overhead (unattributed): 0.1701 ms
- phases[incumbent] (fresh measured (phase timers; extra synchronize added)): trial wall p50 137.4570 ms; plan-validate=0.0253ms, core-invoke=137.2402ms, epilogue-scale=0.0142ms
  - inter-phase host overhead (unattributed): 0.1548 ms
- vendor: unavailable (RuntimeError: [NOT IMPLEMENTED]: error code= 4(at /home/agent/xpytorch/workspace/8ba19c15-fc11-4377-85f3-84daf8eaab09/XMLIR/python/torch_xmlir/csrc/aten_capture/eager_customized/fft_fft.cpp:40))
- plan creation: mean 0.2189 ms (fresh measured (host plan resolution, no device))
- launches[selector] (analytic estimate (cost_model.axes + real-compat op list)): family=incumbent_m4_folded launches=16 gm_bytes=268435456 flops=50331648.0

