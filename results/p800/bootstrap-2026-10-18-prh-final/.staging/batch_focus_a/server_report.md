# Batch release focus report (issue #3 P0 acceptance)

commit `0fb91f9a8d2f` dirty=False device=kunlunxin-p800 driver=515.58 protocol=`batch-release-v1:focus:plan-reuse:interleave:correctness-first:3sess:p11`
profile_hash=`a78b86793c7142a0` table_sha256=`f172147dc2c3e875...` sessions=3 warmup=5 trials=30 seed=1234

Timing = api wall time (dispatch+alloc+layout+sync), ms; median of session medians over independent process sessions. Verdict vs the incumbent radix-2 folded control: >=1.02 speedup, <=0.98 regression, else parity (matches the 2% promotion gate). GM bytes in raw rows are analytical estimates (no hardware counters).

| label | N | B | blk | m0 loop | inc fold | m4 spr | m2 b | m5 b | selected | vendor | m2>=1.02? | m5>=1.02? | sel | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| N1024B16_dif | 1024 | 16 | 256 | 7.0875 | 1.6883 parity (1.000x) | 1.7465 regression (0.967x) | 0.8654 speedup (1.951x) | 0.8932 speedup (1.890x) | 0.7445 speedup (2.268x) | 0.2522 | speedup (1.951x) | speedup (1.890x) | speedup (2.268x) | True |
| N1024B16_dit | 1024 | 16 | 256 | 7.1230 | 1.7523 parity (1.000x) | 1.7528 parity (1.000x) | 0.8654 speedup (2.025x) | 0.9039 speedup (1.939x) | 0.7326 speedup (2.392x) | 0.2527 | speedup (2.025x) | speedup (1.939x) | speedup (2.392x) | True |
| N1024B64_dif | 1024 | 64 | 256 | 27.8701 | 5.9239 parity (1.000x) | 5.9559 parity (0.995x) | 2.9321 speedup (2.020x) | 2.7177 speedup (2.180x) | 2.3402 speedup (2.531x) | 0.2636 | speedup (2.020x) | speedup (2.180x) | speedup (2.531x) | True |
| N1024B64_dit | 1024 | 64 | 256 | 29.1874 | 6.0876 parity (1.000x) | 6.2869 regression (0.968x) | 2.8905 speedup (2.106x) | 3.0222 speedup (2.014x) | 2.2969 speedup (2.650x) | 0.2632 | speedup (2.106x) | speedup (2.014x) | speedup (2.650x) | True |
| N2048B16_dif | 2048 | 16 | 256 | 7.7088 | 3.4013 parity (1.000x) | 3.7386 regression (0.910x) | 1.8769 speedup (1.812x) | 1.8385 speedup (1.850x) | 1.6078 speedup (2.115x) | 0.6640 | speedup (1.812x) | speedup (1.850x) | speedup (2.115x) | True |
| N2048B16_dit | 2048 | 16 | 256 | 7.5736 | 3.4890 parity (1.000x) | 3.8329 regression (0.910x) | 1.8495 speedup (1.886x) | 2.0050 speedup (1.740x) | 0.3077 speedup (11.340x) | 0.6609 | speedup (1.886x) | speedup (1.740x) | speedup (11.340x) | True |
| N2048B64_dif | 2048 | 64 | 256 | 29.2804 | 12.0507 parity (1.000x) | 12.3542 regression (0.975x) | 6.1869 speedup (1.948x) | 6.5808 speedup (1.831x) | 5.1153 speedup (2.356x) | 0.6721 | speedup (1.948x) | speedup (1.831x) | speedup (2.356x) | True |
| N2048B64_dit | 2048 | 64 | 256 | 29.7210 | 12.4722 parity (1.000x) | 12.8878 regression (0.968x) | 5.8413 speedup (2.135x) | 7.0947 speedup (1.758x) | 5.0070 speedup (2.491x) | 0.6723 | speedup (2.135x) | speedup (1.758x) | speedup (2.491x) | True |

## P2 Td canary (representative shape N=1024 B=16)

| candidate | median ms | speedup vs inc | td | verdict |
|---|---:|---:|---|---|
| incumbent_m4_folded | 1.7523 | 1.000x | not-lowered | parity |
| m2_batch | 0.8654 | 2.025x | 1 | speedup |
| m2_packet_td1 | 0.9423 | 1.860x | 1 | speedup |
| m2_packet_td2 | 0.9478 | 1.849x | 2 | speedup |

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
| N1024B16_dif | 4.058 | 1.955 | 2.018 | 1.215 | - |
| N1024B16_dit | 4.064 | 1.939 | 2.026 | 1.242 | 0.913 |
| N1024B64_dif | 4.679 | 2.192 | 2.031 | 1.277 | - |
| N1024B64_dit | 4.643 | 2.080 | 2.175 | 1.288 | - |
| N2048B16_dif | 2.062 | 2.034 | 1.992 | 1.198 | - |
| N2048B16_dit | 1.976 | 1.912 | 2.072 | 1.191 | - |
| N2048B64_dif | 2.370 | 1.877 | 1.997 | 1.222 | - |
| N2048B64_dit | 2.306 | 1.817 | 2.206 | 1.177 | - |