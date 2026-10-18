# Batch release focus report (issue #3 P0 acceptance)

commit `39bd915c054b` dirty=False device=kunlunxin-p800 driver=515.58 protocol=`batch-release-v1:focus:plan-reuse:interleave:correctness-first:3sess:p11`
profile_hash=`a78b86793c7142a0` table_sha256=`f172147dc2c3e875...` sessions=3 warmup=5 trials=30 seed=1234

Timing = api wall time (dispatch+alloc+layout+sync), ms; median of session medians over independent process sessions. Verdict vs the incumbent radix-2 folded control: >=1.02 speedup, <=0.98 regression, else parity (matches the 2% promotion gate). GM bytes in raw rows are analytical estimates (no hardware counters).

| label | N | B | blk | m0 loop | inc fold | m4 spr | m2 b | m5 b | selected | vendor | m2>=1.02? | m5>=1.02? | sel | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| N1024B16_dif | 1024 | 16 | 256 | 7.0078 | 1.6897 parity (1.000x) | 1.7582 regression (0.961x) | 0.8673 speedup (1.948x) | 0.8816 speedup (1.917x) | 0.7498 speedup (2.254x) | 0.2524 | speedup (1.948x) | speedup (1.917x) | speedup (2.254x) | True |
| N1024B16_dit | 1024 | 16 | 256 | 7.3480 | 1.7834 parity (1.000x) | 1.7497 parity (1.019x) | 0.8726 speedup (2.044x) | 0.9049 speedup (1.971x) | 0.7462 speedup (2.390x) | 0.2533 | speedup (2.044x) | speedup (1.971x) | speedup (2.390x) | True |
| N1024B64_dif | 1024 | 64 | 256 | 27.8538 | 5.9022 parity (1.000x) | 5.9549 parity (0.991x) | 2.9375 speedup (2.009x) | 2.6916 speedup (2.193x) | 2.3379 speedup (2.525x) | 0.2617 | speedup (2.009x) | speedup (2.193x) | speedup (2.525x) | True |
| N1024B64_dit | 1024 | 64 | 256 | 27.5263 | 6.0927 parity (1.000x) | 6.2733 regression (0.971x) | 2.8790 speedup (2.116x) | 3.0190 speedup (2.018x) | 2.2844 speedup (2.667x) | 0.2629 | speedup (2.116x) | speedup (2.018x) | speedup (2.667x) | True |
| N2048B16_dif | 2048 | 16 | 256 | 7.5596 | 3.3862 parity (1.000x) | 3.7353 regression (0.907x) | 1.8772 speedup (1.804x) | 1.8264 speedup (1.854x) | 1.6011 speedup (2.115x) | 0.6622 | speedup (1.804x) | speedup (1.854x) | speedup (2.115x) | True |
| N2048B16_dit | 2048 | 16 | 256 | 7.5320 | 3.4758 parity (1.000x) | 3.8256 regression (0.909x) | 1.8398 speedup (1.889x) | 2.0084 speedup (1.731x) | 0.3055 speedup (11.377x) | 0.6605 | speedup (1.889x) | speedup (1.731x) | speedup (11.377x) | True |
| N2048B64_dif | 2048 | 64 | 256 | 29.7971 | 12.0458 parity (1.000x) | 12.3518 regression (0.975x) | 6.1764 speedup (1.950x) | 6.6007 speedup (1.825x) | 5.1030 speedup (2.361x) | 0.6739 | speedup (1.950x) | speedup (1.825x) | speedup (2.361x) | True |
| N2048B64_dit | 2048 | 64 | 256 | 29.7549 | 12.4579 parity (1.000x) | 12.8937 regression (0.966x) | 5.8517 speedup (2.129x) | 7.1583 speedup (1.740x) | 5.0251 speedup (2.479x) | 0.6754 | speedup (2.129x) | speedup (1.740x) | speedup (2.479x) | True |

## P2 Td canary (representative shape N=1024 B=16)

| candidate | median ms | speedup vs inc | td | verdict |
|---|---:|---:|---|---|
| incumbent_m4_folded | 1.7834 | 1.000x | not-lowered | parity |
| m2_batch | 0.8726 | 2.044x | 1 | speedup |
| m2_packet_td1 | 0.9581 | 1.861x | 1 | speedup |
| m2_packet_td2 | 0.9627 | 1.853x | 2 | speedup |

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
| N1024B16_dif | 3.986 | 1.994 | 2.027 | 1.222 | - |
| N1024B16_dit | 4.200 | 1.934 | 2.005 | 1.233 | 0.906 |
| N1024B64_dif | 4.677 | 2.212 | 2.027 | 1.283 | - |
| N1024B64_dit | 4.388 | 2.078 | 2.179 | 1.286 | - |
| N2048B16_dif | 2.024 | 2.045 | 1.990 | 1.198 | - |
| N2048B16_dit | 1.969 | 1.905 | 2.079 | 1.185 | - |
| N2048B64_dif | 2.412 | 1.871 | 2.000 | 1.220 | - |
| N2048B64_dit | 2.308 | 1.801 | 2.203 | 1.178 | - |