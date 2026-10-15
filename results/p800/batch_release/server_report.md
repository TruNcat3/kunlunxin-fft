# Batch release focus report (issue #3 P0 acceptance)

commit `3882e86cf7cc` dirty=True device=kunlunxin-p800 driver=515.58 protocol=`batch-release-v1:focus:plan-reuse:interleave:correctness-first:3sess:p11`
profile_hash=`a78b86793c7142a0` table_sha256=`2e5eac926069d290...` sessions=3 warmup=3 trials=30 seed=1234

Timing = api wall time (dispatch+alloc+layout+sync), ms; median of session medians over independent process sessions. Verdict vs the incumbent radix-2 folded control: >=1.02 speedup, <=0.98 regression, else parity (matches the 2% promotion gate). GM bytes in raw rows are analytical estimates (no hardware counters).

| label | N | B | blk | m0 loop | inc fold | m4 spr | m2 b | m5 b | selected | vendor | m2>=1.02? | m5>=1.02? | sel | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| N1024B16_dif | 1024 | 16 | 256 | 6.9305 | 1.6849 parity (1.000x) | 1.7604 regression (0.957x) | 0.8638 speedup (1.951x) | 0.8658 speedup (1.946x) | 0.7312 speedup (2.304x) | 0.2516 | speedup (1.951x) | speedup (1.946x) | speedup (2.304x) | True |
| N1024B16_dit | 1024 | 16 | 256 | 6.9736 | 1.7614 parity (1.000x) | 1.7481 parity (1.008x) | 0.8661 speedup (2.034x) | 0.9289 speedup (1.896x) | 0.7099 speedup (2.481x) | 0.2517 | speedup (2.034x) | speedup (1.896x) | speedup (2.481x) | True |
| N1024B64_dif | 1024 | 64 | 256 | 27.3925 | 5.9034 parity (1.000x) | 5.9394 parity (0.994x) | 2.9354 speedup (2.011x) | 2.6954 speedup (2.190x) | 2.3240 speedup (2.540x) | 0.2617 | speedup (2.011x) | speedup (2.190x) | speedup (2.540x) | True |
| N1024B64_dit | 1024 | 64 | 256 | 27.3839 | 6.0918 parity (1.000x) | 6.2395 regression (0.976x) | 2.8830 speedup (2.113x) | 3.0057 speedup (2.027x) | 2.2678 speedup (2.686x) | 0.2620 | speedup (2.113x) | speedup (2.027x) | speedup (2.686x) | True |
| N2048B16_dif | 2048 | 16 | 256 | 7.4050 | 3.4063 parity (1.000x) | 3.7328 regression (0.913x) | 1.8733 speedup (1.818x) | 1.8248 speedup (1.867x) | 1.5827 speedup (2.152x) | 0.6625 | speedup (1.818x) | speedup (1.867x) | speedup (2.152x) | True |
| N2048B16_dit | 2048 | 16 | 256 | 7.3579 | 3.4915 parity (1.000x) | 3.8939 regression (0.897x) | 1.9175 speedup (1.821x) | 1.9267 speedup (1.812x) | 1.5685 speedup (2.226x) | 0.6628 | speedup (1.821x) | speedup (1.812x) | speedup (2.226x) | True |
| N2048B64_dif | 2048 | 64 | 256 | 29.1143 | 11.9395 parity (1.000x) | 12.7734 regression (0.935x) | 5.9005 speedup (2.023x) | 7.2914 speedup (1.637x) | 5.0513 speedup (2.364x) | 0.6722 | speedup (2.023x) | speedup (1.637x) | speedup (2.364x) | True |
| N2048B64_dit | 2048 | 64 | 256 | 29.0164 | 12.3442 parity (1.000x) | 13.4334 regression (0.919x) | 6.1084 speedup (2.021x) | 7.9362 speedup (1.555x) | 5.0086 speedup (2.465x) | 0.6724 | speedup (2.021x) | speedup (1.555x) | speedup (2.465x) | True |

## P2 Td canary (representative shape N=1024 B=16)

| candidate | median ms | speedup vs inc | td | verdict |
|---|---:|---:|---|---|
| incumbent_m4_folded | 1.7614 | 1.000x | not-lowered | parity |
| m2_batch | 0.8661 | 2.034x | 1 | speedup |
| m2_packet_td1 | 0.9410 | 1.872x | 1 | speedup |
| m2_packet_td2 | 0.9380 | 1.878x | 2 | speedup |

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
| N1024B16_dif | 3.937 | 2.033 | 2.038 | 1.211 | - |
| N1024B16_dit | 3.989 | 1.882 | 2.018 | 1.250 | 0.923 |
| N1024B64_dif | 4.612 | 2.204 | 2.023 | 1.279 | - |
| N1024B64_dit | 4.389 | 2.076 | 2.164 | 1.288 | - |
| N2048B16_dif | 1.984 | 2.046 | 1.993 | 1.193 | - |
| N2048B16_dit | 1.890 | 2.021 | 2.031 | 1.231 | - |
| N2048B64_dif | 2.279 | 1.752 | 2.165 | 1.171 | - |
| N2048B64_dit | 2.160 | 1.693 | 2.199 | 1.227 | - |