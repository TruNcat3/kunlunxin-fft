# Batch release focus report (issue #3 P0 acceptance)

commit `1dc7775aaebd` dirty=False device=kunlunxin-p800 driver=515.58 protocol=`batch-release-v1:focus:plan-reuse:interleave:correctness-first:3sess:p11`
profile_hash=`a78b86793c7142a0` table_sha256=`3f64e677379a232d...` sessions=3 warmup=5 trials=30 seed=1234

Timing = api wall time (dispatch+alloc+layout+sync), ms; median of session medians over independent process sessions. Verdict vs the incumbent radix-2 folded control: >=1.02 speedup, <=0.98 regression, else parity (matches the 2% promotion gate). GM bytes in raw rows are analytical estimates (no hardware counters).

| label | N | B | blk | m0 loop | inc fold | m4 spr | m2 b | m5 b | selected | vendor | m2>=1.02? | m5>=1.02? | sel | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| N1024B16_dif | 1024 | 16 | 256 | 7.0135 | 1.6830 parity (1.000x) | 1.7639 regression (0.954x) | 0.8636 speedup (1.949x) | 0.8726 speedup (1.929x) | 0.7460 speedup (2.256x) | 0.2522 | speedup (1.949x) | speedup (1.929x) | speedup (2.256x) | True |
| N1024B16_dit | 1024 | 16 | 256 | 7.0291 | 1.7610 parity (1.000x) | 1.7510 parity (1.006x) | 0.8719 speedup (2.020x) | 0.9377 speedup (1.878x) | 0.7313 speedup (2.408x) | 0.2524 | speedup (2.020x) | speedup (1.878x) | speedup (2.408x) | True |
| N1024B64_dif | 1024 | 64 | 256 | 28.2393 | 5.9225 parity (1.000x) | 5.9419 parity (0.997x) | 2.9320 speedup (2.020x) | 2.7189 speedup (2.178x) | 2.3460 speedup (2.525x) | 0.2622 | speedup (2.020x) | speedup (2.178x) | speedup (2.525x) | True |
| N1024B64_dit | 1024 | 64 | 256 | 28.8039 | 6.1000 parity (1.000x) | 6.2662 regression (0.973x) | 2.8806 speedup (2.118x) | 3.0285 speedup (2.014x) | 2.2861 speedup (2.668x) | 0.2625 | speedup (2.118x) | speedup (2.014x) | speedup (2.668x) | True |
| N2048B16_dif | 2048 | 16 | 256 | 7.5509 | 3.3762 parity (1.000x) | 3.7359 regression (0.904x) | 1.8719 speedup (1.804x) | 1.8188 speedup (1.856x) | 1.6040 speedup (2.105x) | 0.6622 | speedup (1.804x) | speedup (1.856x) | speedup (2.105x) | True |
| N2048B16_dit | 2048 | 16 | 256 | 7.5477 | 3.4850 parity (1.000x) | 3.8285 regression (0.910x) | 1.8460 speedup (1.888x) | 2.0142 speedup (1.730x) | 0.3120 speedup (11.171x) | 0.6611 | speedup (1.888x) | speedup (1.730x) | speedup (11.171x) | True |
| N2048B64_dif | 2048 | 64 | 256 | 29.6947 | 12.0397 parity (1.000x) | 12.3413 regression (0.976x) | 6.1871 speedup (1.946x) | 6.5686 speedup (1.833x) | 5.1174 speedup (2.353x) | 0.6720 | speedup (1.946x) | speedup (1.833x) | speedup (2.353x) | True |
| N2048B64_dit | 2048 | 64 | 256 | 29.6735 | 12.4453 parity (1.000x) | 12.8908 regression (0.965x) | 5.8497 speedup (2.128x) | 7.1186 speedup (1.748x) | 5.0074 speedup (2.485x) | 0.6733 | speedup (2.128x) | speedup (1.748x) | speedup (2.485x) | True |

## P2 Td canary (representative shape N=1024 B=16)

| candidate | median ms | speedup vs inc | td | verdict |
|---|---:|---:|---|---|
| incumbent_m4_folded | 1.7610 | 1.000x | not-lowered | parity |
| m2_batch | 0.8719 | 2.020x | 1 | speedup |
| m2_packet_td1 | 0.9466 | 1.860x | 1 | speedup |
| m2_packet_td2 | 0.9446 | 1.864x | 2 | speedup |

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
| N1024B16_dif | 3.976 | 2.021 | 2.043 | 1.210 | - |
| N1024B16_dit | 4.014 | 1.867 | 2.008 | 1.251 | 0.923 |
| N1024B64_dif | 4.753 | 2.185 | 2.027 | 1.272 | - |
| N1024B64_dit | 4.597 | 2.069 | 2.175 | 1.280 | - |
| N2048B16_dif | 2.021 | 2.054 | 1.996 | 1.190 | - |
| N2048B16_dit | 1.971 | 1.901 | 2.074 | 1.189 | - |
| N2048B64_dif | 2.406 | 1.879 | 1.995 | 1.220 | - |
| N2048B64_dit | 2.302 | 1.811 | 2.204 | 1.179 | - |