# Batch release focus report (issue #3 P0 acceptance)

commit `70bd4718b38b` dirty=False device=kunlunxin-p800 driver=515.58 protocol=`batch-release-v1:focus:plan-reuse:interleave:correctness-first:3sess:p11`
profile_hash=`a78b86793c7142a0` table_sha256=`93be5746f70863c6...` sessions=3 warmup=5 trials=30 seed=1234

Timing = api wall time (dispatch+alloc+layout+sync), ms; median of session medians over independent process sessions. Verdict vs the incumbent radix-2 folded control: >=1.02 speedup, <=0.98 regression, else parity (matches the 2% promotion gate). GM bytes in raw rows are analytical estimates (no hardware counters).

| label | N | B | blk | m0 loop | inc fold | m4 spr | m2 b | m5 b | selected | vendor | m2>=1.02? | m5>=1.02? | sel | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| N1024B16_dif | 1024 | 16 | 256 | 6.9708 | 1.6921 parity (1.000x) | 1.7624 regression (0.960x) | 0.8625 speedup (1.962x) | 0.8667 speedup (1.952x) | 0.7302 speedup (2.317x) | 0.2518 | speedup (1.962x) | speedup (1.952x) | speedup (2.317x) | True |
| N1024B16_dit | 1024 | 16 | 256 | 6.9717 | 1.7628 parity (1.000x) | 1.7506 parity (1.007x) | 0.8650 speedup (2.038x) | 0.9321 speedup (1.891x) | 0.7124 speedup (2.474x) | 0.2522 | speedup (2.038x) | speedup (1.891x) | speedup (2.474x) | True |
| N1024B64_dif | 1024 | 64 | 256 | 27.4524 | 5.9069 parity (1.000x) | 5.9425 parity (0.994x) | 2.9362 speedup (2.012x) | 2.7016 speedup (2.186x) | 2.3262 speedup (2.539x) | 0.2636 | speedup (2.012x) | speedup (2.186x) | speedup (2.539x) | True |
| N1024B64_dit | 1024 | 64 | 256 | 27.4199 | 6.0942 parity (1.000x) | 6.2441 regression (0.976x) | 2.8744 speedup (2.120x) | 3.0043 speedup (2.028x) | 2.2647 speedup (2.691x) | 0.2635 | speedup (2.120x) | speedup (2.028x) | speedup (2.691x) | True |
| N2048B16_dif | 2048 | 16 | 256 | 7.4048 | 3.4135 parity (1.000x) | 3.7387 regression (0.913x) | 1.8750 speedup (1.821x) | 1.8236 speedup (1.872x) | 1.5880 speedup (2.150x) | 0.6633 | speedup (1.821x) | speedup (1.872x) | speedup (2.150x) | True |
| N2048B16_dit | 2048 | 16 | 256 | 7.4174 | 3.4902 parity (1.000x) | 3.9049 regression (0.894x) | 1.9281 speedup (1.810x) | 1.9264 speedup (1.812x) | 1.5750 speedup (2.216x) | 0.6629 | speedup (1.810x) | speedup (1.812x) | speedup (2.216x) | True |
| N2048B64_dif | 2048 | 64 | 256 | 29.1574 | 11.9439 parity (1.000x) | 12.7767 regression (0.935x) | 5.8897 speedup (2.028x) | 7.2906 speedup (1.638x) | 5.0506 speedup (2.365x) | 0.6715 | speedup (2.028x) | speedup (1.638x) | speedup (2.365x) | True |
| N2048B64_dit | 2048 | 64 | 256 | 29.3177 | 12.3558 parity (1.000x) | 13.4299 regression (0.920x) | 6.0966 speedup (2.027x) | 7.9267 speedup (1.559x) | 5.0020 speedup (2.470x) | 0.6730 | speedup (2.027x) | speedup (1.559x) | speedup (2.470x) | True |

## P2 Td canary (representative shape N=1024 B=16)

| candidate | median ms | speedup vs inc | td | verdict |
|---|---:|---:|---|---|
| incumbent_m4_folded | 1.7628 | 1.000x | not-lowered | parity |
| m2_batch | 0.8650 | 2.038x | 1 | speedup |
| m2_packet_td1 | 0.9408 | 1.874x | 1 | speedup |
| m2_packet_td2 | 0.9432 | 1.869x | 2 | speedup |

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
| N1024B16_dif | 3.955 | 2.033 | 2.043 | 1.209 | - |
| N1024B16_dit | 3.982 | 1.878 | 2.024 | 1.245 | 0.917 |
| N1024B64_dif | 4.620 | 2.200 | 2.024 | 1.272 | - |
| N1024B64_dit | 4.391 | 2.078 | 2.172 | 1.280 | - |
| N2048B16_dif | 1.981 | 2.050 | 1.994 | 1.194 | - |
| N2048B16_dit | 1.900 | 2.027 | 2.025 | 1.240 | - |
| N2048B64_dif | 2.282 | 1.752 | 2.169 | 1.169 | - |
| N2048B64_dit | 2.183 | 1.694 | 2.203 | 1.225 | - |