# Batch release focus report (issue #3 P0 acceptance)

commit `dadd1d7daf5f` dirty=False device=kunlunxin-p800 driver=515.58 protocol=`batch-release-v1:focus:plan-reuse:interleave:correctness-first:3sess:p11`
profile_hash=`a78b86793c7142a0` table_sha256=`acc1f21c9c93fe20...` sessions=3 warmup=5 trials=30 seed=1234

Timing = api wall time (dispatch+alloc+layout+sync), ms; median of session medians over independent process sessions. Verdict vs the incumbent radix-2 folded control: >=1.02 speedup, <=0.98 regression, else parity (matches the 2% promotion gate). GM bytes in raw rows are analytical estimates (no hardware counters).

| label | N | B | blk | m0 loop | inc fold | m4 spr | m2 b | m5 b | selected | vendor | m2>=1.02? | m5>=1.02? | sel | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| N1024B16_dif | 1024 | 16 | 256 | 6.9427 | 1.6901 parity (1.000x) | 1.7619 regression (0.959x) | 0.8644 speedup (1.955x) | 0.8866 speedup (1.906x) | 0.7453 speedup (2.268x) | 0.2514 | speedup (1.955x) | speedup (1.906x) | speedup (2.268x) | True |
| N1024B16_dit | 1024 | 16 | 256 | 6.9841 | 1.7649 parity (1.000x) | 1.7673 parity (0.999x) | 0.8655 speedup (2.039x) | 0.9457 speedup (1.866x) | 0.7292 speedup (2.420x) | 0.2521 | speedup (2.039x) | speedup (1.866x) | speedup (2.420x) | True |
| N1024B64_dif | 1024 | 64 | 256 | 27.4279 | 5.9216 parity (1.000x) | 5.9468 parity (0.996x) | 2.9466 speedup (2.010x) | 2.7148 speedup (2.181x) | 2.3572 speedup (2.512x) | 0.2626 | speedup (2.010x) | speedup (2.181x) | speedup (2.512x) | True |
| N1024B64_dit | 1024 | 64 | 256 | 27.4437 | 6.0883 parity (1.000x) | 6.2317 regression (0.977x) | 2.8780 speedup (2.115x) | 3.0157 speedup (2.019x) | 2.2835 speedup (2.666x) | 0.2619 | speedup (2.115x) | speedup (2.019x) | speedup (2.666x) | True |
| N2048B16_dif | 2048 | 16 | 256 | 7.4125 | 3.3875 parity (1.000x) | 3.7303 regression (0.908x) | 1.8744 speedup (1.807x) | 1.8363 speedup (1.845x) | 1.5961 speedup (2.122x) | 0.6621 | speedup (1.807x) | speedup (1.845x) | speedup (2.122x) | True |
| N2048B16_dit | 2048 | 16 | 256 | 7.4170 | 3.4811 parity (1.000x) | 3.8264 regression (0.910x) | 1.8405 speedup (1.891x) | 1.9892 speedup (1.750x) | 0.3070 speedup (11.339x) | 0.6604 | speedup (1.891x) | speedup (1.750x) | speedup (11.339x) | True |
| N2048B64_dif | 2048 | 64 | 256 | 29.3085 | 12.0426 parity (1.000x) | 12.3339 regression (0.976x) | 6.1831 speedup (1.948x) | 6.5503 speedup (1.838x) | 5.1240 speedup (2.350x) | 0.6723 | speedup (1.948x) | speedup (1.838x) | speedup (2.350x) | True |
| N2048B64_dit | 2048 | 64 | 256 | 29.4572 | 12.4389 parity (1.000x) | 12.8497 regression (0.968x) | 5.8472 speedup (2.127x) | 7.1137 speedup (1.749x) | 5.0126 speedup (2.482x) | 0.6724 | speedup (2.127x) | speedup (1.749x) | speedup (2.482x) | True |

## P2 Td canary (representative shape N=1024 B=16)

| candidate | median ms | speedup vs inc | td | verdict |
|---|---:|---:|---|---|
| incumbent_m4_folded | 1.7649 | 1.000x | not-lowered | parity |
| m2_batch | 0.8655 | 2.039x | 1 | speedup |
| m2_packet_td1 | 0.9342 | 1.889x | 1 | speedup |
| m2_packet_td2 | 0.9389 | 1.880x | 2 | speedup |

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
| N1024B16_dif | 3.940 | 1.987 | 2.038 | 1.210 | - |
| N1024B16_dit | 3.952 | 1.869 | 2.042 | 1.243 | 0.922 |
| N1024B64_dif | 4.612 | 2.190 | 2.018 | 1.276 | - |
| N1024B64_dit | 4.404 | 2.066 | 2.165 | 1.284 | - |
| N2048B16_dif | 1.987 | 2.031 | 1.990 | 1.195 | - |
| N2048B16_dit | 1.938 | 1.924 | 2.079 | 1.187 | - |
| N2048B64_dif | 2.376 | 1.883 | 1.995 | 1.220 | - |
| N2048B64_dit | 2.292 | 1.806 | 2.198 | 1.177 | - |