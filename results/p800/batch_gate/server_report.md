# Batch release focus report (issue #3 P0 acceptance)

commit `3882e86cf7cc` dirty=True device=kunlunxin-p800 driver=515.58 protocol=`batch-release-v1:focus:plan-reuse:interleave:correctness-first:3sess:p11`
profile_hash=`a78b86793c7142a0` table_sha256=`309483cca022b624...` sessions=3 warmup=3 trials=30 seed=1234

Timing = api wall time (dispatch+alloc+layout+sync), ms; median of session medians over independent process sessions. Verdict vs the incumbent radix-2 folded control: >=1.02 speedup, <=0.98 regression, else parity (matches the 2% promotion gate). GM bytes in raw rows are analytical estimates (no hardware counters).

| label | N | B | blk | m0 loop | inc fold | m4 spr | m2 b | m5 b | selected | vendor | m2>=1.02? | m5>=1.02? | sel | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| N1024B16_dif | 1024 | 16 | 256 | 7.0032 | 1.6799 parity (1.000x) | 1.6928 parity (0.992x) | 0.8202 speedup (2.048x) | 0.8915 speedup (1.884x) | 1.7281 regression (0.972x) | 0.2518 | speedup (2.048x) | speedup (1.884x) | regression (0.972x) | True |
| N1024B16_dit | 1024 | 16 | 256 | 7.0062 | 1.7499 parity (1.000x) | 1.7479 parity (1.001x) | 0.8632 speedup (2.027x) | 0.9031 speedup (1.938x) | 1.7688 parity (0.989x) | 0.2528 | speedup (2.027x) | speedup (1.938x) | parity (0.989x) | True |
| N1024B64_dif | 1024 | 64 | 256 | 27.6003 | 5.8603 parity (1.000x) | 5.9347 parity (0.987x) | 2.9397 speedup (1.993x) | 2.7189 speedup (2.155x) | 5.7764 parity (1.015x) | 0.2643 | speedup (1.993x) | speedup (2.155x) | parity (1.015x) | True |
| N1024B64_dit | 1024 | 64 | 256 | 27.4305 | 6.0125 parity (1.000x) | 6.2078 regression (0.969x) | 2.8743 speedup (2.092x) | 2.9882 speedup (2.012x) | 5.9425 parity (1.012x) | 0.2641 | speedup (2.092x) | speedup (2.012x) | parity (1.012x) | True |
| N2048B16_dif | 2048 | 16 | 256 | 7.4640 | 3.3649 parity (1.000x) | 3.4617 regression (0.972x) | 1.7870 speedup (1.883x) | 1.7972 speedup (1.872x) | 3.4606 regression (0.972x) | 0.6641 | speedup (1.883x) | speedup (1.872x) | regression (0.972x) | True |
| N2048B16_dit | 2048 | 16 | 256 | 7.4752 | 3.4856 parity (1.000x) | 3.6403 regression (0.958x) | 1.9326 speedup (1.804x) | 1.9583 speedup (1.780x) | 3.5912 regression (0.971x) | 0.6649 | speedup (1.804x) | speedup (1.780x) | regression (0.971x) | True |
| N2048B64_dif | 2048 | 64 | 256 | 29.3341 | 12.2215 parity (1.000x) | 12.9824 regression (0.941x) | 6.0312 speedup (2.026x) | 7.3906 speedup (1.654x) | 12.6611 regression (0.965x) | 0.6728 | speedup (2.026x) | speedup (1.654x) | regression (0.965x) | True |
| N2048B64_dit | 2048 | 64 | 256 | 29.2963 | 12.6787 parity (1.000x) | 13.7023 regression (0.925x) | 6.2234 speedup (2.037x) | 8.0876 speedup (1.568x) | 13.2270 regression (0.959x) | 0.6749 | speedup (2.037x) | speedup (1.568x) | regression (0.959x) | True |

## P2 Td canary (representative shape N=1024 B=16)

| candidate | median ms | speedup vs inc | td | verdict |
|---|---:|---:|---|---|
| incumbent_m4_folded | 1.7499 | 1.000x | not-lowered | parity |
| m2_batch | 0.8632 | 2.027x | 1 | speedup |
| m2_packet_td1 | 0.9418 | 1.858x | 1 | speedup |
| m2_packet_td2 | 0.9421 | 1.857x | 2 | speedup |

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
| N1024B16_dif | 4.137 | 1.899 | 2.064 | 1.205 | - |
| N1024B16_dit | 4.008 | 1.935 | 2.025 | 1.240 | 0.916 |
| N1024B64_dif | 4.651 | 2.183 | 2.019 | 1.271 | - |
| N1024B64_dit | 4.419 | 2.077 | 2.160 | 1.280 | - |
| N2048B16_dif | 2.156 | 1.926 | 1.937 | 1.195 | - |
| N2048B16_dit | 2.053 | 1.859 | 1.884 | 1.238 | - |
| N2048B64_dif | 2.260 | 1.757 | 2.153 | 1.167 | - |
| N2048B64_dit | 2.138 | 1.694 | 2.202 | 1.215 | - |