# Batch release focus report (issue #3 P0 acceptance)

commit `3a57d1a69bee` dirty=False device=kunlunxin-p800 driver=515.58 protocol=`batch-release-v1:focus:plan-reuse:interleave:correctness-first:3sess:p11`
profile_hash=`a78b86793c7142a0` table_sha256=`25818154a3ab06dc...` sessions=3 warmup=5 trials=30 seed=1234

Timing = api wall time (dispatch+alloc+layout+sync), ms; median of session medians over independent process sessions. Verdict vs the incumbent radix-2 folded control: >=1.02 speedup, <=0.98 regression, else parity (matches the 2% promotion gate). GM bytes in raw rows are analytical estimates (no hardware counters).

| label | N | B | blk | m0 loop | inc fold | m4 spr | m2 b | m5 b | selected | vendor | m2>=1.02? | m5>=1.02? | sel | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| N1024B16_dif | 1024 | 16 | 256 | 7.1367 | 1.8269 parity (1.000x) | 1.9391 regression (0.942x) | 0.9511 speedup (1.921x) | 0.9301 speedup (1.964x) | 0.8143 speedup (2.244x) | 0.2529 | speedup (1.921x) | speedup (1.964x) | speedup (2.244x) | True |
| N1024B16_dit | 1024 | 16 | 256 | 7.4296 | 1.6772 parity (1.000x) | 1.7930 regression (0.935x) | 0.9133 speedup (1.836x) | 0.9497 speedup (1.766x) | 0.9587 speedup (1.750x) | 0.2556 | speedup (1.836x) | speedup (1.766x) | speedup (1.750x) | True |
| N1024B64_dif | 1024 | 64 | 256 | 27.9569 | 5.8850 parity (1.000x) | 5.9579 parity (0.988x) | 2.8677 speedup (2.052x) | 2.7404 speedup (2.148x) | 2.2980 speedup (2.561x) | 0.2625 | speedup (2.052x) | speedup (2.148x) | speedup (2.561x) | True |
| N1024B64_dit | 1024 | 64 | 256 | 27.9547 | 6.0492 parity (1.000x) | 6.2655 regression (0.965x) | 2.8926 speedup (2.091x) | 2.9850 speedup (2.027x) | 2.9218 speedup (2.070x) | 0.2626 | speedup (2.091x) | speedup (2.027x) | speedup (2.070x) | True |
| N2048B16_dif | 2048 | 16 | 256 | 7.4495 | 3.4457 parity (1.000x) | 3.6841 regression (0.935x) | 1.8115 speedup (1.902x) | 1.8748 speedup (1.838x) | 1.5679 speedup (2.198x) | 0.6630 | speedup (1.902x) | speedup (1.838x) | speedup (2.198x) | True |
| N2048B16_dit | 2048 | 16 | 256 | 7.4248 | 3.5894 parity (1.000x) | 3.9064 regression (0.919x) | 1.8018 speedup (1.992x) | 2.0304 speedup (1.768x) | 0.3093 speedup (11.607x) | 0.6613 | speedup (1.992x) | speedup (1.768x) | speedup (11.607x) | True |
| N2048B64_dif | 2048 | 64 | 256 | 29.5577 | 11.7650 parity (1.000x) | 12.4066 regression (0.948x) | 6.1911 speedup (1.900x) | 6.8304 speedup (1.722x) | 5.1376 speedup (2.290x) | 0.6724 | speedup (1.900x) | speedup (1.722x) | speedup (2.290x) | True |
| N2048B64_dit | 2048 | 64 | 256 | 29.5823 | 12.2481 parity (1.000x) | 13.1465 regression (0.932x) | 5.9212 speedup (2.069x) | 7.6121 speedup (1.609x) | 5.9666 speedup (2.053x) | 0.6723 | speedup (2.069x) | speedup (1.609x) | speedup (2.053x) | True |

## P2 Td canary (representative shape N=1024 B=16)

| candidate | median ms | speedup vs inc | td | verdict |
|---|---:|---:|---|---|
| incumbent_m4_folded | 1.6772 | 1.000x | not-lowered | parity |
| m2_batch | 0.9133 | 1.836x | 1 | speedup |
| m2_packet_td1 | 0.9643 | 1.739x | 1 | speedup |
| m2_packet_td2 | 0.9654 | 1.737x | 2 | speedup |

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
| N1024B16_dif | 3.680 | 2.085 | 2.039 | 1.211 | - |
| N1024B16_dit | 4.144 | 1.888 | 1.963 | 1.202 | 0.946 |
| N1024B64_dif | 4.692 | 2.174 | 2.078 | 1.268 | - |
| N1024B64_dit | 4.462 | 2.099 | 2.166 | 1.280 | - |
| N2048B16_dif | 2.022 | 1.965 | 2.034 | 1.182 | - |
| N2048B16_dit | 1.901 | 1.924 | 2.168 | 1.184 | - |
| N2048B64_dif | 2.382 | 1.816 | 2.004 | 1.221 | - |
| N2048B64_dit | 2.250 | 1.727 | 2.220 | 1.180 | - |