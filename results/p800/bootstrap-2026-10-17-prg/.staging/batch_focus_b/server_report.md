# Batch release focus report (issue #3 P0 acceptance)

commit `746261d30f46` dirty=False device=kunlunxin-p800 driver=515.58 protocol=`batch-release-v1:focus:plan-reuse:interleave:correctness-first:3sess:p11`
profile_hash=`a78b86793c7142a0` table_sha256=`eb1e5f388a92f2f2...` sessions=3 warmup=5 trials=30 seed=1234

Timing = api wall time (dispatch+alloc+layout+sync), ms; median of session medians over independent process sessions. Verdict vs the incumbent radix-2 folded control: >=1.02 speedup, <=0.98 regression, else parity (matches the 2% promotion gate). GM bytes in raw rows are analytical estimates (no hardware counters).

| label | N | B | blk | m0 loop | inc fold | m4 spr | m2 b | m5 b | selected | vendor | m2>=1.02? | m5>=1.02? | sel | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| N1024B16_dif | 1024 | 16 | 256 | 7.0214 | 1.6902 parity (1.000x) | 1.7629 regression (0.959x) | 0.8682 speedup (1.947x) | 0.8947 speedup (1.889x) | 0.7463 speedup (2.265x) | 0.2522 | speedup (1.947x) | speedup (1.889x) | speedup (2.265x) | True |
| N1024B16_dit | 1024 | 16 | 256 | 7.0381 | 1.7724 parity (1.000x) | 1.7715 parity (1.001x) | 0.8728 speedup (2.031x) | 0.9489 speedup (1.868x) | 0.7335 speedup (2.416x) | 0.2526 | speedup (2.031x) | speedup (1.868x) | speedup (2.416x) | True |
| N1024B64_dif | 1024 | 64 | 256 | 27.6798 | 5.9044 parity (1.000x) | 5.9505 parity (0.992x) | 2.9403 speedup (2.008x) | 2.7111 speedup (2.178x) | 2.3503 speedup (2.512x) | 0.2635 | speedup (2.008x) | speedup (2.178x) | speedup (2.512x) | True |
| N1024B64_dit | 1024 | 64 | 256 | 27.5459 | 6.0911 parity (1.000x) | 6.2449 regression (0.975x) | 2.8884 speedup (2.109x) | 3.0159 speedup (2.020x) | 2.2924 speedup (2.657x) | 0.2637 | speedup (2.109x) | speedup (2.020x) | speedup (2.657x) | True |
| N2048B16_dif | 2048 | 16 | 256 | 7.5163 | 3.3847 parity (1.000x) | 3.7323 regression (0.907x) | 1.8736 speedup (1.807x) | 1.8297 speedup (1.850x) | 1.6072 speedup (2.106x) | 0.6629 | speedup (1.807x) | speedup (1.850x) | speedup (2.106x) | True |
| N2048B16_dit | 2048 | 16 | 256 | 7.4731 | 3.4861 parity (1.000x) | 3.8297 regression (0.910x) | 1.8490 speedup (1.885x) | 1.9980 speedup (1.745x) | 0.3104 speedup (11.230x) | 0.6610 | speedup (1.885x) | speedup (1.745x) | speedup (11.230x) | True |
| N2048B64_dif | 2048 | 64 | 256 | 29.3627 | 12.0503 parity (1.000x) | 12.3247 regression (0.978x) | 6.1777 speedup (1.951x) | 6.5876 speedup (1.829x) | 5.1112 speedup (2.358x) | 0.6713 | speedup (1.951x) | speedup (1.829x) | speedup (2.358x) | True |
| N2048B64_dit | 2048 | 64 | 256 | 29.4443 | 12.4463 parity (1.000x) | 12.8642 regression (0.968x) | 5.8478 speedup (2.128x) | 7.1046 speedup (1.752x) | 5.0114 speedup (2.484x) | 0.6731 | speedup (2.128x) | speedup (1.752x) | speedup (2.484x) | True |

## P2 Td canary (representative shape N=1024 B=16)

| candidate | median ms | speedup vs inc | td | verdict |
|---|---:|---:|---|---|
| incumbent_m4_folded | 1.7724 | 1.000x | not-lowered | parity |
| m2_batch | 0.8728 | 2.031x | 1 | speedup |
| m2_packet_td1 | 0.9445 | 1.877x | 1 | speedup |
| m2_packet_td2 | 0.9404 | 1.885x | 2 | speedup |

`m2_packet_td2` is the true worker packet loop (tables loaded once, consumed by a static_range loop over 2 packets; DIT only; packets>2 rejected by the runner: this backend silently miscompiles programs beyond four dot operations).

## Notes

- P1.1 axes on every raw row: `us`/`ts`/`ud`/`td` are `not-lowered`, a real fixed value, or `external` -- never an aspirational name; folded batch rows carry the explicit flattened-index caveat.
- `baseline_m0_rank1_loop` is literal M0 at batch shapes (host row loop); the batch incumbent control is radix-2 folded.
- Vendor support was re-probed this round (see `manifest.vendor_probe`), not hardcoded from old conclusions.
- 2% promotion gate: candidates are NOT promoted by this harness; promotion is a separate gated step against the incumbent.


## Ablation contrasts (median ms ratios, >1 = variant faster)

- **flow / batch lowering** (radix-2 held): m0 host loop vs m4 spread (Ud=B on grid axis) and m4 folded (flattened index); the spread/folded pair isolates the Td-indexing form.
- **core radix** (global-stage flow held): m4 spread radix-2 vs m5 radix-4 spans.
- **local-matrix core** (batch spread held): m4 spread scalar stages vs m2 register-resident dot group.
- **bounded-search delta** (same m2 flow): default tile=64 vs the fastest cost-model top-K variant measured this round.
- **P2 Td canary** (dit canary shape only): m2 spread vs packet Td=1 vs packet Td=2.

| label | m0/m4spr | m4spr/m5 | m4spr/m2 | m2/t2best | m2/m2pkt_td2 |
|---|---:|---:|---:|---:|---:|
| N1024B16_dif | 3.983 | 1.970 | 2.031 | 1.211 | - |
| N1024B16_dit | 3.973 | 1.867 | 2.030 | 1.256 | 0.928 |
| N1024B64_dif | 4.652 | 2.195 | 2.024 | 1.276 | - |
| N1024B64_dit | 4.411 | 2.071 | 2.162 | 1.285 | - |
| N2048B16_dif | 2.014 | 2.040 | 1.992 | 1.196 | - |
| N2048B16_dit | 1.951 | 1.917 | 2.071 | 1.190 | - |
| N2048B64_dif | 2.382 | 1.871 | 1.995 | 1.218 | - |
| N2048B64_dit | 2.289 | 1.811 | 2.200 | 1.178 | - |