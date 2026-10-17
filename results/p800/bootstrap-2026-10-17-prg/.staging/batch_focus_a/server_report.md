# Batch release focus report (issue #3 P0 acceptance)

commit `746261d30f46` dirty=False device=kunlunxin-p800 driver=515.58 protocol=`batch-release-v1:focus:plan-reuse:interleave:correctness-first:3sess:p11`
profile_hash=`a78b86793c7142a0` table_sha256=`eb1e5f388a92f2f2...` sessions=3 warmup=5 trials=30 seed=1234

Timing = api wall time (dispatch+alloc+layout+sync), ms; median of session medians over independent process sessions. Verdict vs the incumbent radix-2 folded control: >=1.02 speedup, <=0.98 regression, else parity (matches the 2% promotion gate). GM bytes in raw rows are analytical estimates (no hardware counters).

| label | N | B | blk | m0 loop | inc fold | m4 spr | m2 b | m5 b | selected | vendor | m2>=1.02? | m5>=1.02? | sel | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| N1024B16_dif | 1024 | 16 | 256 | 7.3834 | 1.7102 parity (1.000x) | 1.7654 regression (0.969x) | 0.8757 speedup (1.953x) | 0.8950 speedup (1.911x) | 0.7594 speedup (2.252x) | 0.2549 | speedup (1.953x) | speedup (1.911x) | speedup (2.252x) | True |
| N1024B16_dit | 1024 | 16 | 256 | 7.4376 | 1.7841 parity (1.000x) | 1.7596 parity (1.014x) | 0.8803 speedup (2.027x) | 0.9115 speedup (1.957x) | 0.7503 speedup (2.378x) | 0.2548 | speedup (2.027x) | speedup (1.957x) | speedup (2.378x) | True |
| N1024B64_dif | 1024 | 64 | 256 | 27.6590 | 5.9047 parity (1.000x) | 5.9678 parity (0.989x) | 2.9286 speedup (2.016x) | 2.6919 speedup (2.193x) | 2.3426 speedup (2.521x) | 0.2637 | speedup (2.016x) | speedup (2.193x) | speedup (2.521x) | True |
| N1024B64_dit | 1024 | 64 | 256 | 28.0310 | 6.0680 parity (1.000x) | 6.2657 regression (0.968x) | 2.8898 speedup (2.100x) | 3.0011 speedup (2.022x) | 2.2877 speedup (2.652x) | 0.2641 | speedup (2.100x) | speedup (2.022x) | speedup (2.652x) | True |
| N2048B16_dif | 2048 | 16 | 256 | 7.5555 | 3.3954 parity (1.000x) | 3.7387 regression (0.908x) | 1.8784 speedup (1.808x) | 1.8319 speedup (1.854x) | 1.6089 speedup (2.110x) | 0.6627 | speedup (1.808x) | speedup (1.854x) | speedup (2.110x) | True |
| N2048B16_dit | 2048 | 16 | 256 | 7.5935 | 3.4774 parity (1.000x) | 3.8459 regression (0.904x) | 1.8417 speedup (1.888x) | 2.0106 speedup (1.730x) | 0.3160 speedup (11.004x) | 0.6626 | speedup (1.888x) | speedup (1.730x) | speedup (11.004x) | True |
| N2048B64_dif | 2048 | 64 | 256 | 29.7654 | 12.0502 parity (1.000x) | 12.3394 regression (0.977x) | 6.1847 speedup (1.948x) | 6.5846 speedup (1.830x) | 5.1246 speedup (2.351x) | 0.6731 | speedup (1.948x) | speedup (1.830x) | speedup (2.351x) | True |
| N2048B64_dit | 2048 | 64 | 256 | 29.8501 | 12.4593 parity (1.000x) | 12.9053 regression (0.965x) | 5.8427 speedup (2.132x) | 7.1469 speedup (1.743x) | 5.0148 speedup (2.484x) | 0.6727 | speedup (2.132x) | speedup (1.743x) | speedup (2.484x) | True |

## P2 Td canary (representative shape N=1024 B=16)

| candidate | median ms | speedup vs inc | td | verdict |
|---|---:|---:|---|---|
| incumbent_m4_folded | 1.7841 | 1.000x | not-lowered | parity |
| m2_batch | 0.8803 | 2.027x | 1 | speedup |
| m2_packet_td1 | 0.9689 | 1.841x | 1 | speedup |
| m2_packet_td2 | 0.9711 | 1.837x | 2 | speedup |

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
| N1024B16_dif | 4.182 | 1.972 | 2.016 | 1.205 | - |
| N1024B16_dit | 4.227 | 1.930 | 1.999 | 1.241 | 0.907 |
| N1024B64_dif | 4.635 | 2.217 | 2.038 | 1.276 | - |
| N1024B64_dit | 4.474 | 2.088 | 2.168 | 1.290 | - |
| N2048B16_dif | 2.021 | 2.041 | 1.990 | 1.195 | - |
| N2048B16_dit | 1.974 | 1.913 | 2.088 | 1.184 | - |
| N2048B64_dif | 2.412 | 1.874 | 1.995 | 1.221 | - |
| N2048B64_dit | 2.313 | 1.806 | 2.209 | 1.176 | - |