# Batch release focus report (issue #3 P0 acceptance)

commit `70bd4718b38b` dirty=False device=kunlunxin-p800 driver=515.58 protocol=`batch-release-v1:focus:plan-reuse:interleave:correctness-first:3sess:p11`
profile_hash=`a78b86793c7142a0` table_sha256=`93be5746f70863c6...` sessions=3 warmup=5 trials=30 seed=1234

Timing = api wall time (dispatch+alloc+layout+sync), ms; median of session medians over independent process sessions. Verdict vs the incumbent radix-2 folded control: >=1.02 speedup, <=0.98 regression, else parity (matches the 2% promotion gate). GM bytes in raw rows are analytical estimates (no hardware counters).

| label | N | B | blk | m0 loop | inc fold | m4 spr | m2 b | m5 b | selected | vendor | m2>=1.02? | m5>=1.02? | sel | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| N1024B16_dif | 1024 | 16 | 256 | 7.0017 | 1.6801 parity (1.000x) | 1.7494 regression (0.960x) | 0.8617 speedup (1.950x) | 0.8932 speedup (1.881x) | 0.7293 speedup (2.304x) | 0.2517 | speedup (1.950x) | speedup (1.881x) | speedup (2.304x) | True |
| N1024B16_dit | 1024 | 16 | 256 | 7.0649 | 1.7563 parity (1.000x) | 1.7472 parity (1.005x) | 0.8641 speedup (2.033x) | 0.9011 speedup (1.949x) | 0.7115 speedup (2.469x) | 0.2530 | speedup (2.033x) | speedup (1.949x) | speedup (2.469x) | True |
| N1024B64_dif | 1024 | 64 | 256 | 27.6945 | 5.9304 parity (1.000x) | 5.9529 parity (0.996x) | 2.9229 speedup (2.029x) | 2.7038 speedup (2.193x) | 2.3195 speedup (2.557x) | 0.2636 | speedup (2.029x) | speedup (2.193x) | speedup (2.557x) | True |
| N1024B64_dit | 1024 | 64 | 256 | 27.6647 | 6.0915 parity (1.000x) | 6.2695 regression (0.972x) | 2.8735 speedup (2.120x) | 3.0036 speedup (2.028x) | 2.2640 speedup (2.691x) | 0.2633 | speedup (2.120x) | speedup (2.028x) | speedup (2.691x) | True |
| N2048B16_dif | 2048 | 16 | 256 | 7.5497 | 3.4207 parity (1.000x) | 3.7325 regression (0.916x) | 1.8749 speedup (1.824x) | 1.8232 speedup (1.876x) | 1.5784 speedup (2.167x) | 0.6627 | speedup (1.824x) | speedup (1.876x) | speedup (2.167x) | True |
| N2048B16_dit | 2048 | 16 | 256 | 7.5631 | 3.4995 parity (1.000x) | 3.8904 regression (0.900x) | 1.9387 speedup (1.805x) | 1.9398 speedup (1.804x) | 1.5768 speedup (2.219x) | 0.6644 | speedup (1.805x) | speedup (1.804x) | speedup (2.219x) | True |
| N2048B64_dif | 2048 | 64 | 256 | 29.7136 | 11.9589 parity (1.000x) | 12.7838 regression (0.935x) | 5.8993 speedup (2.027x) | 7.2983 speedup (1.639x) | 5.0465 speedup (2.370x) | 0.6731 | speedup (2.027x) | speedup (1.639x) | speedup (2.370x) | True |
| N2048B64_dit | 2048 | 64 | 256 | 29.6488 | 12.3411 parity (1.000x) | 13.4517 regression (0.917x) | 6.0928 speedup (2.026x) | 7.9304 speedup (1.556x) | 4.9969 speedup (2.470x) | 0.6744 | speedup (2.026x) | speedup (1.556x) | speedup (2.470x) | True |

## P2 Td canary (representative shape N=1024 B=16)

| candidate | median ms | speedup vs inc | td | verdict |
|---|---:|---:|---|---|
| incumbent_m4_folded | 1.7563 | 1.000x | not-lowered | parity |
| m2_batch | 0.8641 | 2.033x | 1 | speedup |
| m2_packet_td1 | 0.9402 | 1.868x | 1 | speedup |
| m2_packet_td2 | 0.9419 | 1.865x | 2 | speedup |

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
| N1024B16_dif | 4.002 | 1.959 | 2.030 | 1.207 | - |
| N1024B16_dit | 4.044 | 1.939 | 2.022 | 1.241 | 0.917 |
| N1024B64_dif | 4.652 | 2.202 | 2.037 | 1.271 | - |
| N1024B64_dit | 4.413 | 2.087 | 2.182 | 1.288 | - |
| N2048B16_dif | 2.023 | 2.047 | 1.991 | 1.193 | - |
| N2048B16_dit | 1.944 | 2.006 | 2.007 | 1.241 | - |
| N2048B64_dif | 2.324 | 1.752 | 2.167 | 1.172 | - |
| N2048B64_dit | 2.204 | 1.696 | 2.208 | 1.227 | - |