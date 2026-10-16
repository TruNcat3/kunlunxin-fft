# Batch release focus report (issue #3 P0 acceptance)

commit `ec19fd3e8a60` dirty=False device=kunlunxin-p800 driver=515.58 protocol=`batch-release-v1:focus:plan-reuse:interleave:correctness-first:3sess:p11`
profile_hash=`a78b86793c7142a0` table_sha256=`93be5746f70863c6...` sessions=3 warmup=5 trials=30 seed=1234

Timing = api wall time (dispatch+alloc+layout+sync), ms; median of session medians over independent process sessions. Verdict vs the incumbent radix-2 folded control: >=1.02 speedup, <=0.98 regression, else parity (matches the 2% promotion gate). GM bytes in raw rows are analytical estimates (no hardware counters).

| label | N | B | blk | m0 loop | inc fold | m4 spr | m2 b | m5 b | selected | vendor | m2>=1.02? | m5>=1.02? | sel | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| N1024B16_dif | 1024 | 16 | 256 | 7.0001 | 1.6982 parity (1.000x) | 1.7683 regression (0.960x) | 0.8627 speedup (1.968x) | 0.8777 speedup (1.935x) | 0.7296 speedup (2.328x) | 0.2529 | speedup (1.968x) | speedup (1.935x) | speedup (2.328x) | True |
| N1024B16_dit | 1024 | 16 | 256 | 7.0094 | 1.7680 parity (1.000x) | 1.7437 parity (1.014x) | 0.8709 speedup (2.030x) | 0.9338 speedup (1.893x) | 0.7128 speedup (2.480x) | 0.2532 | speedup (2.030x) | speedup (1.893x) | speedup (2.480x) | True |
| N1024B64_dif | 1024 | 64 | 256 | 27.5738 | 5.9017 parity (1.000x) | 5.9508 parity (0.992x) | 2.9284 speedup (2.015x) | 2.7111 speedup (2.177x) | 2.3306 speedup (2.532x) | 0.2628 | speedup (2.015x) | speedup (2.177x) | speedup (2.532x) | True |
| N1024B64_dit | 1024 | 64 | 256 | 27.6030 | 6.1050 parity (1.000x) | 6.2335 regression (0.979x) | 2.8784 speedup (2.121x) | 3.0064 speedup (2.031x) | 2.2666 speedup (2.693x) | 0.2640 | speedup (2.121x) | speedup (2.031x) | speedup (2.693x) | True |
| N2048B16_dif | 2048 | 16 | 256 | 7.4309 | 3.4201 parity (1.000x) | 3.7340 regression (0.916x) | 1.8783 speedup (1.821x) | 1.8299 speedup (1.869x) | 1.5892 speedup (2.152x) | 0.6635 | speedup (1.821x) | speedup (1.869x) | speedup (2.152x) | True |
| N2048B16_dit | 2048 | 16 | 256 | 7.4511 | 3.5001 parity (1.000x) | 3.9034 regression (0.897x) | 1.9222 speedup (1.821x) | 1.9294 speedup (1.814x) | 1.5734 speedup (2.225x) | 0.6636 | speedup (1.821x) | speedup (1.814x) | speedup (2.225x) | True |
| N2048B64_dif | 2048 | 64 | 256 | 29.3398 | 11.9281 parity (1.000x) | 12.7810 regression (0.933x) | 5.8870 speedup (2.026x) | 7.3015 speedup (1.634x) | 5.0597 speedup (2.357x) | 0.6735 | speedup (2.026x) | speedup (1.634x) | speedup (2.357x) | True |
| N2048B64_dit | 2048 | 64 | 256 | 29.4144 | 12.3381 parity (1.000x) | 13.4221 regression (0.919x) | 6.0962 speedup (2.024x) | 7.9213 speedup (1.558x) | 5.0045 speedup (2.465x) | 0.6751 | speedup (2.024x) | speedup (1.558x) | speedup (2.465x) | True |

## P2 Td canary (representative shape N=1024 B=16)

| candidate | median ms | speedup vs inc | td | verdict |
|---|---:|---:|---|---|
| incumbent_m4_folded | 1.7680 | 1.000x | not-lowered | parity |
| m2_batch | 0.8709 | 2.030x | 1 | speedup |
| m2_packet_td1 | 0.9434 | 1.874x | 1 | speedup |
| m2_packet_td2 | 0.9449 | 1.871x | 2 | speedup |

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
| N1024B16_dif | 3.959 | 2.015 | 2.050 | 1.206 | - |
| N1024B16_dit | 4.020 | 1.867 | 2.002 | 1.245 | 0.922 |
| N1024B64_dif | 4.634 | 2.195 | 2.032 | 1.271 | - |
| N1024B64_dit | 4.428 | 2.073 | 2.166 | 1.283 | - |
| N2048B16_dif | 1.990 | 2.041 | 1.988 | 1.193 | - |
| N2048B16_dit | 1.909 | 2.023 | 2.031 | 1.233 | - |
| N2048B64_dif | 2.296 | 1.750 | 2.171 | 1.170 | - |
| N2048B64_dit | 2.191 | 1.694 | 2.202 | 1.224 | - |