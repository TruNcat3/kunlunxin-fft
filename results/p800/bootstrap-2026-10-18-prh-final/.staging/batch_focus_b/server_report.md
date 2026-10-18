# Batch release focus report (issue #3 P0 acceptance)

commit `0fb91f9a8d2f` dirty=False device=kunlunxin-p800 driver=515.58 protocol=`batch-release-v1:focus:plan-reuse:interleave:correctness-first:3sess:p11`
profile_hash=`a78b86793c7142a0` table_sha256=`f172147dc2c3e875...` sessions=3 warmup=5 trials=30 seed=1234

Timing = api wall time (dispatch+alloc+layout+sync), ms; median of session medians over independent process sessions. Verdict vs the incumbent radix-2 folded control: >=1.02 speedup, <=0.98 regression, else parity (matches the 2% promotion gate). GM bytes in raw rows are analytical estimates (no hardware counters).

| label | N | B | blk | m0 loop | inc fold | m4 spr | m2 b | m5 b | selected | vendor | m2>=1.02? | m5>=1.02? | sel | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| N1024B16_dif | 1024 | 16 | 256 | 7.0260 | 1.6864 parity (1.000x) | 1.7654 regression (0.955x) | 0.8635 speedup (1.953x) | 0.8708 speedup (1.937x) | 0.7436 speedup (2.268x) | 0.2524 | speedup (1.953x) | speedup (1.937x) | speedup (2.268x) | True |
| N1024B16_dit | 1024 | 16 | 256 | 6.9942 | 1.7668 parity (1.000x) | 1.7619 parity (1.003x) | 0.8670 speedup (2.038x) | 0.9327 speedup (1.894x) | 0.7324 speedup (2.412x) | 0.2523 | speedup (2.038x) | speedup (1.894x) | speedup (2.412x) | True |
| N1024B64_dif | 1024 | 64 | 256 | 28.0124 | 5.9133 parity (1.000x) | 5.9560 parity (0.993x) | 2.9427 speedup (2.009x) | 2.7199 speedup (2.174x) | 2.3521 speedup (2.514x) | 0.2626 | speedup (2.009x) | speedup (2.174x) | speedup (2.514x) | True |
| N1024B64_dit | 1024 | 64 | 256 | 27.5596 | 6.0951 parity (1.000x) | 6.2462 regression (0.976x) | 2.8738 speedup (2.121x) | 3.0239 speedup (2.016x) | 2.2894 speedup (2.662x) | 0.2623 | speedup (2.121x) | speedup (2.016x) | speedup (2.662x) | True |
| N2048B16_dif | 2048 | 16 | 256 | 7.5651 | 3.3811 parity (1.000x) | 3.7574 regression (0.900x) | 1.8793 speedup (1.799x) | 1.8296 speedup (1.848x) | 1.6110 speedup (2.099x) | 0.6625 | speedup (1.799x) | speedup (1.848x) | speedup (2.099x) | True |
| N2048B16_dit | 2048 | 16 | 256 | 7.4784 | 3.4782 parity (1.000x) | 3.8246 regression (0.909x) | 1.8496 speedup (1.880x) | 2.0177 speedup (1.724x) | 0.3105 speedup (11.201x) | 0.6590 | speedup (1.880x) | speedup (1.724x) | speedup (11.201x) | True |
| N2048B64_dif | 2048 | 64 | 256 | 29.4342 | 12.0449 parity (1.000x) | 12.3326 regression (0.977x) | 6.1846 speedup (1.948x) | 6.5741 speedup (1.832x) | 5.1233 speedup (2.351x) | 0.6735 | speedup (1.948x) | speedup (1.832x) | speedup (2.351x) | True |
| N2048B64_dit | 2048 | 64 | 256 | 29.5741 | 12.4648 parity (1.000x) | 12.8863 regression (0.967x) | 5.8415 speedup (2.134x) | 7.1271 speedup (1.749x) | 5.0099 speedup (2.488x) | 0.6738 | speedup (2.134x) | speedup (1.749x) | speedup (2.488x) | True |

## P2 Td canary (representative shape N=1024 B=16)

| candidate | median ms | speedup vs inc | td | verdict |
|---|---:|---:|---|---|
| incumbent_m4_folded | 1.7668 | 1.000x | not-lowered | parity |
| m2_batch | 0.8670 | 2.038x | 1 | speedup |
| m2_packet_td1 | 0.9419 | 1.876x | 1 | speedup |
| m2_packet_td2 | 0.9446 | 1.870x | 2 | speedup |

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
| N1024B16_dif | 3.980 | 2.027 | 2.045 | 1.212 | - |
| N1024B16_dit | 3.970 | 1.889 | 2.032 | 1.246 | 0.918 |
| N1024B64_dif | 4.703 | 2.190 | 2.024 | 1.276 | - |
| N1024B64_dit | 4.412 | 2.066 | 2.173 | 1.279 | - |
| N2048B16_dif | 2.013 | 2.054 | 1.999 | 1.197 | - |
| N2048B16_dit | 1.955 | 1.896 | 2.068 | 1.191 | - |
| N2048B64_dif | 2.387 | 1.876 | 1.994 | 1.219 | - |
| N2048B64_dit | 2.295 | 1.808 | 2.206 | 1.175 | - |