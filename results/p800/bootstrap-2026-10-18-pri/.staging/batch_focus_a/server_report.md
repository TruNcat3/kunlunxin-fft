# Batch release focus report (issue #3 P0 acceptance)

commit `dbb8c5b7432d` dirty=False device=kunlunxin-p800 driver=515.58 protocol=`batch-release-v1:focus:plan-reuse:interleave:correctness-first:3sess:p11`
profile_hash=`a78b86793c7142a0` table_sha256=`602782a3ad785fa0...` sessions=3 warmup=5 trials=30 seed=1234

Timing = api wall time (dispatch+alloc+layout+sync), ms; median of session medians over independent process sessions. Verdict vs the incumbent radix-2 folded control: >=1.02 speedup, <=0.98 regression, else parity (matches the 2% promotion gate). GM bytes in raw rows are analytical estimates (no hardware counters).

| label | N | B | blk | m0 loop | inc fold | m4 spr | m2 b | m5 b | selected | vendor | m2>=1.02? | m5>=1.02? | sel | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| N1024B16_dif | 1024 | 16 | 256 | 7.1100 | 1.8327 parity (1.000x) | 1.9494 regression (0.940x) | 0.9594 speedup (1.910x) | 0.9354 speedup (1.959x) | 0.8299 speedup (2.208x) | 0.2537 | speedup (1.910x) | speedup (1.959x) | speedup (2.208x) | True |
| N1024B16_dit | 1024 | 16 | 256 | 7.0580 | 1.6681 parity (1.000x) | 1.7912 regression (0.931x) | 0.9029 speedup (1.848x) | 0.9360 speedup (1.782x) | 0.9434 speedup (1.768x) | 0.2533 | speedup (1.848x) | speedup (1.782x) | speedup (1.768x) | True |
| N1024B64_dif | 1024 | 64 | 256 | 27.7578 | 5.8843 parity (1.000x) | 5.9730 parity (0.985x) | 2.8659 speedup (2.053x) | 2.7520 speedup (2.138x) | 2.2978 speedup (2.561x) | 0.2617 | speedup (2.053x) | speedup (2.138x) | speedup (2.561x) | True |
| N1024B64_dit | 1024 | 64 | 256 | 27.7910 | 6.0530 parity (1.000x) | 6.2615 regression (0.967x) | 2.8840 speedup (2.099x) | 2.9868 speedup (2.027x) | 2.9389 speedup (2.060x) | 0.2633 | speedup (2.099x) | speedup (2.027x) | speedup (2.060x) | True |
| N2048B16_dif | 2048 | 16 | 256 | 7.5122 | 3.4535 parity (1.000x) | 3.6784 regression (0.939x) | 1.8117 speedup (1.906x) | 1.8790 speedup (1.838x) | 1.5668 speedup (2.204x) | 0.6634 | speedup (1.906x) | speedup (1.838x) | speedup (2.204x) | True |
| N2048B16_dit | 2048 | 16 | 256 | 7.4644 | 3.5788 parity (1.000x) | 3.9097 regression (0.915x) | 1.8022 speedup (1.986x) | 2.0279 speedup (1.765x) | 0.3092 speedup (11.574x) | 0.6605 | speedup (1.986x) | speedup (1.765x) | speedup (11.574x) | True |
| N2048B64_dif | 2048 | 64 | 256 | 29.6307 | 11.7707 parity (1.000x) | 12.4049 regression (0.949x) | 6.1966 speedup (1.900x) | 6.8240 speedup (1.725x) | 5.1137 speedup (2.302x) | 0.6719 | speedup (1.900x) | speedup (1.725x) | speedup (2.302x) | True |
| N2048B64_dit | 2048 | 64 | 256 | 29.8619 | 12.2516 parity (1.000x) | 13.1419 regression (0.932x) | 5.9196 speedup (2.070x) | 7.6033 speedup (1.611x) | 5.9578 speedup (2.056x) | 0.6725 | speedup (2.070x) | speedup (1.611x) | speedup (2.056x) | True |

## P2 Td canary (representative shape N=1024 B=16)

| candidate | median ms | speedup vs inc | td | verdict |
|---|---:|---:|---|---|
| incumbent_m4_folded | 1.6681 | 1.000x | not-lowered | parity |
| m2_batch | 0.9029 | 1.848x | 1 | speedup |
| m2_packet_td1 | 0.9461 | 1.763x | 1 | speedup |
| m2_packet_td2 | 0.9466 | 1.762x | 2 | speedup |

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
| N1024B16_dif | 3.647 | 2.084 | 2.032 | 1.210 | - |
| N1024B16_dit | 3.940 | 1.914 | 1.984 | 1.211 | 0.954 |
| N1024B64_dif | 4.647 | 2.170 | 2.084 | 1.271 | - |
| N1024B64_dit | 4.438 | 2.096 | 2.171 | 1.275 | - |
| N2048B16_dif | 2.042 | 1.958 | 2.030 | 1.180 | - |
| N2048B16_dit | 1.909 | 1.928 | 2.169 | 1.183 | - |
| N2048B64_dif | 2.389 | 1.818 | 2.002 | 1.220 | - |
| N2048B64_dit | 2.272 | 1.728 | 2.220 | 1.177 | - |