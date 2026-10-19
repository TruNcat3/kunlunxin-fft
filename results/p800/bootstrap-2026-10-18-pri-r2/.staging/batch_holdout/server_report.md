# Batch release focus report (issue #3 P0 acceptance)

commit `3a57d1a69bee` dirty=False device=kunlunxin-p800 driver=515.58 protocol=`batch-release-v1:focus:plan-reuse:interleave:correctness-first:3sess:p11:m2-fullspace-holdout`
profile_hash=`a78b86793c7142a0` table_sha256=`25818154a3ab06dc...` sessions=3 warmup=5 trials=30 seed=1234

Timing = api wall time (dispatch+alloc+layout+sync), ms; median of session medians over independent process sessions. Verdict vs the incumbent radix-2 folded control: >=1.02 speedup, <=0.98 regression, else parity (matches the 2% promotion gate). GM bytes in raw rows are analytical estimates (no hardware counters).

| label | N | B | blk | m0 loop | inc fold | m4 spr | m2 b | m5 b | selected | vendor | m2>=1.02? | m5>=1.02? | sel | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| N1024B16_dif | 1024 | 16 | 256 | 7.0839 | 1.7482 parity (1.000x) | 1.8520 regression (0.944x) | 0.8872 speedup (1.970x) | 0.9075 speedup (1.926x) | 0.7690 speedup (2.273x) | 0.2536 | speedup (1.970x) | speedup (1.926x) | speedup (2.273x) | True |
| N1024B16_dit | 1024 | 16 | 256 | 7.0299 | 1.8753 parity (1.000x) | 1.8528 parity (1.012x) | 0.8941 speedup (2.098x) | 0.9817 speedup (1.910x) | 0.8918 speedup (2.103x) | 0.2539 | speedup (2.098x) | speedup (1.910x) | speedup (2.103x) | True |
| N1024B64_dif | 1024 | 64 | 256 | 27.8851 | 6.2581 parity (1.000x) | 6.3855 parity (0.980x) | 3.0062 speedup (2.082x) | 2.7534 speedup (2.273x) | 2.3902 speedup (2.618x) | 0.2657 | speedup (2.082x) | speedup (2.273x) | speedup (2.618x) | True |
| N1024B64_dit | 1024 | 64 | 256 | 27.9093 | 6.0994 parity (1.000x) | 6.3571 regression (0.959x) | 2.9274 speedup (2.084x) | 3.0151 speedup (2.023x) | 2.9121 speedup (2.094x) | 0.2655 | speedup (2.084x) | speedup (2.023x) | speedup (2.094x) | True |
| N2048B16_dif | 2048 | 16 | 256 | 7.5044 | 3.4849 parity (1.000x) | 3.9497 regression (0.882x) | 1.9100 speedup (1.825x) | 1.8991 speedup (1.835x) | 1.6514 speedup (2.110x) | 0.6656 | speedup (1.825x) | speedup (1.835x) | speedup (2.110x) | True |
| N2048B16_dit | 2048 | 16 | 256 | 7.4895 | 3.7256 parity (1.000x) | 3.9714 regression (0.938x) | 1.8416 speedup (2.023x) | 1.9984 speedup (1.864x) | 0.3257 speedup (11.438x) | 0.6625 | speedup (2.023x) | speedup (1.864x) | speedup (11.438x) | True |

## P2 Td canary (representative shape N=1024 B=16)

| candidate | median ms | speedup vs inc | td | verdict |
|---|---:|---:|---|---|
| incumbent_m4_folded | 1.8753 | 1.000x | not-lowered | parity |
| m2_batch | 0.8941 | 2.098x | 1 | speedup |
| m2_packet_td1 | nan | nanx | None | None |
| m2_packet_td2 | nan | nanx | None | None |

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
| N1024B16_dif | 3.825 | 2.041 | 2.087 | 1.207 | - |
| N1024B16_dit | 3.794 | 1.887 | 2.072 | 1.222 | - |
| N1024B64_dif | 4.367 | 2.319 | 2.124 | 1.284 | - |
| N1024B64_dit | 4.390 | 2.108 | 2.172 | 1.292 | - |
| N2048B16_dif | 1.900 | 2.080 | 2.068 | 1.183 | - |
| N2048B16_dit | 1.886 | 1.987 | 2.157 | 1.187 | - |