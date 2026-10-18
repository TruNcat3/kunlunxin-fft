# Batch release focus report (issue #3 P0 acceptance)

commit `1dc7775aaebd` dirty=False device=kunlunxin-p800 driver=515.58 protocol=`batch-release-v1:focus:plan-reuse:interleave:correctness-first:3sess:p11`
profile_hash=`a78b86793c7142a0` table_sha256=`3f64e677379a232d...` sessions=3 warmup=5 trials=30 seed=1234

Timing = api wall time (dispatch+alloc+layout+sync), ms; median of session medians over independent process sessions. Verdict vs the incumbent radix-2 folded control: >=1.02 speedup, <=0.98 regression, else parity (matches the 2% promotion gate). GM bytes in raw rows are analytical estimates (no hardware counters).

| label | N | B | blk | m0 loop | inc fold | m4 spr | m2 b | m5 b | selected | vendor | m2>=1.02? | m5>=1.02? | sel | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| N1024B16_dif | 1024 | 16 | 256 | 6.9984 | 1.6872 parity (1.000x) | 1.7542 regression (0.962x) | 0.8641 speedup (1.952x) | 0.8890 speedup (1.898x) | 0.7416 speedup (2.275x) | 0.2517 | speedup (1.952x) | speedup (1.898x) | speedup (2.275x) | True |
| N1024B16_dit | 1024 | 16 | 256 | 7.0013 | 1.7562 parity (1.000x) | 1.7464 parity (1.006x) | 0.8659 speedup (2.028x) | 0.9012 speedup (1.949x) | 0.7322 speedup (2.398x) | 0.2521 | speedup (2.028x) | speedup (1.949x) | speedup (2.398x) | True |
| N1024B64_dif | 1024 | 64 | 256 | 27.6313 | 5.9059 parity (1.000x) | 5.9679 parity (0.990x) | 2.9396 speedup (2.009x) | 2.7112 speedup (2.178x) | 2.3308 speedup (2.534x) | 0.2624 | speedup (2.009x) | speedup (2.178x) | speedup (2.534x) | True |
| N1024B64_dit | 1024 | 64 | 256 | 27.5516 | 6.0759 parity (1.000x) | 6.2753 regression (0.968x) | 2.8803 speedup (2.109x) | 2.9984 speedup (2.026x) | 2.2764 speedup (2.669x) | 0.2613 | speedup (2.109x) | speedup (2.026x) | speedup (2.669x) | True |
| N2048B16_dif | 2048 | 16 | 256 | 7.4741 | 3.3917 parity (1.000x) | 3.7304 regression (0.909x) | 1.8712 speedup (1.813x) | 1.8284 speedup (1.855x) | 1.6060 speedup (2.112x) | 0.6631 | speedup (1.813x) | speedup (1.855x) | speedup (2.112x) | True |
| N2048B16_dit | 2048 | 16 | 256 | 7.4375 | 3.4762 parity (1.000x) | 3.8234 regression (0.909x) | 1.8425 speedup (1.887x) | 1.9929 speedup (1.744x) | 0.3095 speedup (11.230x) | 0.6601 | speedup (1.887x) | speedup (1.744x) | speedup (11.230x) | True |
| N2048B64_dif | 2048 | 64 | 256 | 29.3260 | 12.0258 parity (1.000x) | 12.3512 regression (0.974x) | 6.1854 speedup (1.944x) | 6.5605 speedup (1.833x) | 5.1087 speedup (2.354x) | 0.6718 | speedup (1.944x) | speedup (1.833x) | speedup (2.354x) | True |
| N2048B64_dit | 2048 | 64 | 256 | 29.4064 | 12.4458 parity (1.000x) | 12.8982 regression (0.965x) | 5.8436 speedup (2.130x) | 7.1090 speedup (1.751x) | 5.0102 speedup (2.484x) | 0.6720 | speedup (2.130x) | speedup (1.751x) | speedup (2.484x) | True |

## P2 Td canary (representative shape N=1024 B=16)

| candidate | median ms | speedup vs inc | td | verdict |
|---|---:|---:|---|---|
| incumbent_m4_folded | 1.7562 | 1.000x | not-lowered | parity |
| m2_batch | 0.8659 | 2.028x | 1 | speedup |
| m2_packet_td1 | 0.9408 | 1.867x | 1 | speedup |
| m2_packet_td2 | 0.9418 | 1.865x | 2 | speedup |

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
| N1024B16_dif | 3.989 | 1.973 | 2.030 | 1.210 | - |
| N1024B16_dit | 4.009 | 1.938 | 2.017 | 1.247 | 0.919 |
| N1024B64_dif | 4.630 | 2.201 | 2.030 | 1.278 | - |
| N1024B64_dit | 4.390 | 2.093 | 2.179 | 1.288 | - |
| N2048B16_dif | 2.004 | 2.040 | 1.994 | 1.194 | - |
| N2048B16_dit | 1.945 | 1.919 | 2.075 | 1.191 | - |
| N2048B64_dif | 2.374 | 1.883 | 1.997 | 1.220 | - |
| N2048B64_dit | 2.280 | 1.814 | 2.207 | 1.176 | - |