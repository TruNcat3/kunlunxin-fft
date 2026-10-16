# Batch release focus report (issue #3 P0 acceptance)

commit `70bd4718b38b` dirty=False device=kunlunxin-p800 driver=515.58 protocol=`batch-release-v1:focus:plan-reuse:interleave:correctness-first:3sess:p11:m2-fullspace-holdout`
profile_hash=`a78b86793c7142a0` table_sha256=`93be5746f70863c6...` sessions=3 warmup=5 trials=30 seed=1234

Timing = api wall time (dispatch+alloc+layout+sync), ms; median of session medians over independent process sessions. Verdict vs the incumbent radix-2 folded control: >=1.02 speedup, <=0.98 regression, else parity (matches the 2% promotion gate). GM bytes in raw rows are analytical estimates (no hardware counters).

| label | N | B | blk | m0 loop | inc fold | m4 spr | m2 b | m5 b | selected | vendor | m2>=1.02? | m5>=1.02? | sel | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| N1024B16_dif | 1024 | 16 | 256 | 6.9775 | 1.7186 parity (1.000x) | 1.7998 regression (0.955x) | 0.8840 speedup (1.944x) | 0.9029 speedup (1.904x) | 0.7513 speedup (2.287x) | 0.2538 | speedup (1.944x) | speedup (1.904x) | speedup (2.287x) | True |
| N1024B16_dit | 1024 | 16 | 256 | 7.0060 | 1.7651 parity (1.000x) | 1.7391 parity (1.015x) | 0.8375 speedup (2.108x) | 0.9450 speedup (1.868x) | 0.7145 speedup (2.471x) | 0.2533 | speedup (2.108x) | speedup (1.868x) | speedup (2.471x) | True |
| N1024B64_dif | 1024 | 64 | 256 | 27.9784 | 6.3466 parity (1.000x) | 6.3773 parity (0.995x) | 3.0197 speedup (2.102x) | 2.8575 speedup (2.221x) | 2.3917 speedup (2.654x) | 0.2659 | speedup (2.102x) | speedup (2.221x) | speedup (2.654x) | True |
| N1024B64_dit | 1024 | 64 | 256 | 27.6171 | 6.0970 parity (1.000x) | 6.3782 regression (0.956x) | 2.9440 speedup (2.071x) | 3.1162 speedup (1.957x) | 2.3021 speedup (2.649x) | 0.2660 | speedup (2.071x) | speedup (1.957x) | speedup (2.649x) | True |
| N2048B16_dif | 2048 | 16 | 256 | 7.5464 | 3.3130 parity (1.000x) | 3.9902 regression (0.830x) | 1.9216 speedup (1.724x) | 1.8840 speedup (1.758x) | 1.6124 speedup (2.055x) | 0.6658 | speedup (1.724x) | speedup (1.758x) | speedup (2.055x) | True |
| N2048B16_dit | 2048 | 16 | 256 | 7.5109 | 3.4285 parity (1.000x) | 4.2181 regression (0.813x) | 1.9854 speedup (1.727x) | 2.0923 speedup (1.639x) | 1.6231 speedup (2.112x) | 0.6671 | speedup (1.727x) | speedup (1.639x) | speedup (2.112x) | True |

## P2 Td canary (representative shape N=1024 B=16)

| candidate | median ms | speedup vs inc | td | verdict |
|---|---:|---:|---|---|
| incumbent_m4_folded | 1.7651 | 1.000x | not-lowered | parity |
| m2_batch | 0.8375 | 2.108x | 1 | speedup |
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
| N1024B16_dif | 3.877 | 1.993 | 2.036 | 1.211 | - |
| N1024B16_dit | 4.028 | 1.840 | 2.077 | 1.202 | - |
| N1024B64_dif | 4.387 | 2.232 | 2.112 | 1.272 | - |
| N1024B64_dit | 4.330 | 2.047 | 2.166 | 1.289 | - |
| N2048B16_dif | 1.891 | 2.118 | 2.077 | 1.209 | - |
| N2048B16_dit | 1.781 | 2.016 | 2.125 | 1.234 | - |