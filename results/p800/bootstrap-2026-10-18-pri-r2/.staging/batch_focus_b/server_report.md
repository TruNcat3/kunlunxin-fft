# Batch release focus report (issue #3 P0 acceptance)

commit `3a57d1a69bee` dirty=False device=kunlunxin-p800 driver=515.58 protocol=`batch-release-v1:focus:plan-reuse:interleave:correctness-first:3sess:p11`
profile_hash=`a78b86793c7142a0` table_sha256=`25818154a3ab06dc...` sessions=3 warmup=5 trials=30 seed=1234

Timing = api wall time (dispatch+alloc+layout+sync), ms; median of session medians over independent process sessions. Verdict vs the incumbent radix-2 folded control: >=1.02 speedup, <=0.98 regression, else parity (matches the 2% promotion gate). GM bytes in raw rows are analytical estimates (no hardware counters).

| label | N | B | blk | m0 loop | inc fold | m4 spr | m2 b | m5 b | selected | vendor | m2>=1.02? | m5>=1.02? | sel | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| N1024B16_dif | 1024 | 16 | 256 | 7.0487 | 1.8208 parity (1.000x) | 1.9343 regression (0.941x) | 0.9511 speedup (1.914x) | 0.9237 speedup (1.971x) | 0.8172 speedup (2.228x) | 0.2524 | speedup (1.914x) | speedup (1.971x) | speedup (2.228x) | True |
| N1024B16_dit | 1024 | 16 | 256 | 7.0740 | 1.6684 parity (1.000x) | 1.7857 regression (0.934x) | 0.8974 speedup (1.859x) | 0.9203 speedup (1.813x) | 0.9478 speedup (1.760x) | 0.2528 | speedup (1.859x) | speedup (1.813x) | speedup (1.760x) | True |
| N1024B64_dif | 1024 | 64 | 256 | 27.8900 | 5.8826 parity (1.000x) | 5.9593 parity (0.987x) | 2.8568 speedup (2.059x) | 2.7274 speedup (2.157x) | 2.2864 speedup (2.573x) | 0.2620 | speedup (2.059x) | speedup (2.157x) | speedup (2.573x) | True |
| N1024B64_dit | 1024 | 64 | 256 | 28.3717 | 6.0558 parity (1.000x) | 6.2735 regression (0.965x) | 2.9041 speedup (2.085x) | 2.9716 speedup (2.038x) | 2.9154 speedup (2.077x) | 0.2646 | speedup (2.085x) | speedup (2.038x) | speedup (2.077x) | True |
| N2048B16_dif | 2048 | 16 | 256 | 7.4920 | 3.4595 parity (1.000x) | 3.6855 regression (0.939x) | 1.8126 speedup (1.909x) | 1.9042 speedup (1.817x) | 1.5705 speedup (2.203x) | 0.6632 | speedup (1.909x) | speedup (1.817x) | speedup (2.203x) | True |
| N2048B16_dit | 2048 | 16 | 256 | 7.4722 | 3.5977 parity (1.000x) | 3.9160 regression (0.919x) | 1.7941 speedup (2.005x) | 2.0252 speedup (1.776x) | 0.3066 speedup (11.736x) | 0.6605 | speedup (2.005x) | speedup (1.776x) | speedup (11.736x) | True |
| N2048B64_dif | 2048 | 64 | 256 | 29.5568 | 11.7568 parity (1.000x) | 12.4007 regression (0.948x) | 6.1952 speedup (1.898x) | 6.8395 speedup (1.719x) | 5.1380 speedup (2.288x) | 0.6715 | speedup (1.898x) | speedup (1.719x) | speedup (2.288x) | True |
| N2048B64_dit | 2048 | 64 | 256 | 29.7212 | 12.2282 parity (1.000x) | 13.1596 regression (0.929x) | 5.9146 speedup (2.067x) | 7.5959 speedup (1.610x) | 5.9564 speedup (2.053x) | 0.6734 | speedup (2.067x) | speedup (1.610x) | speedup (2.053x) | True |

## P2 Td canary (representative shape N=1024 B=16)

| candidate | median ms | speedup vs inc | td | verdict |
|---|---:|---:|---|---|
| incumbent_m4_folded | 1.6684 | 1.000x | not-lowered | parity |
| m2_batch | 0.8974 | 1.859x | 1 | speedup |
| m2_packet_td1 | 0.9446 | 1.766x | 1 | speedup |
| m2_packet_td2 | 0.9446 | 1.766x | 2 | speedup |

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
| N1024B16_dif | 3.644 | 2.094 | 2.034 | 1.215 | - |
| N1024B16_dit | 3.962 | 1.940 | 1.990 | 1.214 | 0.950 |
| N1024B64_dif | 4.680 | 2.185 | 2.086 | 1.277 | - |
| N1024B64_dit | 4.522 | 2.111 | 2.160 | 1.283 | - |
| N2048B16_dif | 2.033 | 1.935 | 2.033 | 1.180 | - |
| N2048B16_dit | 1.908 | 1.934 | 2.183 | 1.185 | - |
| N2048B64_dif | 2.383 | 1.813 | 2.002 | 1.221 | - |
| N2048B64_dit | 2.259 | 1.732 | 2.225 | 1.180 | - |