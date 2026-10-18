# Batch release focus report (issue #3 P0 acceptance)

commit `39bd915c054b` dirty=False device=kunlunxin-p800 driver=515.58 protocol=`batch-release-v1:focus:plan-reuse:interleave:correctness-first:3sess:p11`
profile_hash=`a78b86793c7142a0` table_sha256=`f172147dc2c3e875...` sessions=3 warmup=5 trials=30 seed=1234

Timing = api wall time (dispatch+alloc+layout+sync), ms; median of session medians over independent process sessions. Verdict vs the incumbent radix-2 folded control: >=1.02 speedup, <=0.98 regression, else parity (matches the 2% promotion gate). GM bytes in raw rows are analytical estimates (no hardware counters).

| label | N | B | blk | m0 loop | inc fold | m4 spr | m2 b | m5 b | selected | vendor | m2>=1.02? | m5>=1.02? | sel | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| N1024B16_dif | 1024 | 16 | 256 | 7.0328 | 1.6819 parity (1.000x) | 1.7662 regression (0.952x) | 0.8662 speedup (1.942x) | 0.8852 speedup (1.900x) | 0.7485 speedup (2.247x) | 0.2524 | speedup (1.942x) | speedup (1.900x) | speedup (2.247x) | True |
| N1024B16_dit | 1024 | 16 | 256 | 7.0708 | 1.7737 parity (1.000x) | 1.7687 parity (1.003x) | 0.8669 speedup (2.046x) | 0.9440 speedup (1.879x) | 0.7300 speedup (2.430x) | 0.2530 | speedup (2.046x) | speedup (1.879x) | speedup (2.430x) | True |
| N1024B64_dif | 1024 | 64 | 256 | 27.7147 | 5.9061 parity (1.000x) | 5.9398 parity (0.994x) | 2.9386 speedup (2.010x) | 2.7137 speedup (2.176x) | 2.3455 speedup (2.518x) | 0.2625 | speedup (2.010x) | speedup (2.176x) | speedup (2.518x) | True |
| N1024B64_dit | 1024 | 64 | 256 | 27.7434 | 6.0794 parity (1.000x) | 6.2411 regression (0.974x) | 2.8842 speedup (2.108x) | 3.0268 speedup (2.008x) | 2.2845 speedup (2.661x) | 0.2625 | speedup (2.108x) | speedup (2.008x) | speedup (2.661x) | True |
| N2048B16_dif | 2048 | 16 | 256 | 7.4882 | 3.3860 parity (1.000x) | 3.7310 regression (0.908x) | 1.8767 speedup (1.804x) | 1.8355 speedup (1.845x) | 1.6041 speedup (2.111x) | 0.6637 | speedup (1.804x) | speedup (1.845x) | speedup (2.111x) | True |
| N2048B16_dit | 2048 | 16 | 256 | 7.4840 | 3.4887 parity (1.000x) | 3.8252 regression (0.912x) | 1.8493 speedup (1.887x) | 1.9905 speedup (1.753x) | 0.3092 speedup (11.282x) | 0.6609 | speedup (1.887x) | speedup (1.753x) | speedup (11.282x) | True |
| N2048B64_dif | 2048 | 64 | 256 | 29.4784 | 12.0454 parity (1.000x) | 12.3199 regression (0.978x) | 6.1887 speedup (1.946x) | 6.5675 speedup (1.834x) | 5.1190 speedup (2.353x) | 0.6736 | speedup (1.946x) | speedup (1.834x) | speedup (2.353x) | True |
| N2048B64_dit | 2048 | 64 | 256 | 29.5657 | 12.4557 parity (1.000x) | 12.8633 regression (0.968x) | 5.8363 speedup (2.134x) | 7.1164 speedup (1.750x) | 5.0018 speedup (2.490x) | 0.6732 | speedup (2.134x) | speedup (1.750x) | speedup (2.490x) | True |

## P2 Td canary (representative shape N=1024 B=16)

| candidate | median ms | speedup vs inc | td | verdict |
|---|---:|---:|---|---|
| incumbent_m4_folded | 1.7737 | 1.000x | not-lowered | parity |
| m2_batch | 0.8669 | 2.046x | 1 | speedup |
| m2_packet_td1 | 0.9445 | 1.878x | 1 | speedup |
| m2_packet_td2 | 0.9427 | 1.882x | 2 | speedup |

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
| N1024B16_dif | 3.982 | 1.995 | 2.039 | 1.213 | - |
| N1024B16_dit | 3.998 | 1.874 | 2.040 | 1.245 | 0.920 |
| N1024B64_dif | 4.666 | 2.189 | 2.021 | 1.280 | - |
| N1024B64_dit | 4.445 | 2.062 | 2.164 | 1.289 | - |
| N2048B16_dif | 2.007 | 2.033 | 1.988 | 1.196 | - |
| N2048B16_dit | 1.956 | 1.922 | 2.068 | 1.192 | - |
| N2048B64_dif | 2.393 | 1.876 | 1.991 | 1.219 | - |
| N2048B64_dit | 2.298 | 1.808 | 2.204 | 1.173 | - |