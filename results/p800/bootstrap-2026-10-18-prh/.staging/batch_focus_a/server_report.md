# Batch release focus report (issue #3 P0 acceptance)

commit `b3b48b9b4082` dirty=False device=kunlunxin-p800 driver=515.58 protocol=`batch-release-v1:focus:plan-reuse:interleave:correctness-first:3sess:p11`
profile_hash=`a78b86793c7142a0` table_sha256=`f172147dc2c3e875...` sessions=3 warmup=5 trials=30 seed=1234

Timing = api wall time (dispatch+alloc+layout+sync), ms; median of session medians over independent process sessions. Verdict vs the incumbent radix-2 folded control: >=1.02 speedup, <=0.98 regression, else parity (matches the 2% promotion gate). GM bytes in raw rows are analytical estimates (no hardware counters).

| label | N | B | blk | m0 loop | inc fold | m4 spr | m2 b | m5 b | selected | vendor | m2>=1.02? | m5>=1.02? | sel | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| N1024B16_dif | 1024 | 16 | 256 | 7.0471 | 1.6958 parity (1.000x) | 1.7564 regression (0.966x) | 0.8636 speedup (1.964x) | 0.8843 speedup (1.918x) | 0.7417 speedup (2.287x) | 0.2518 | speedup (1.964x) | speedup (1.918x) | speedup (2.287x) | True |
| N1024B16_dit | 1024 | 16 | 256 | 7.0920 | 1.7851 parity (1.000x) | 1.7392 speedup (1.026x) | 0.8671 speedup (2.059x) | 0.9019 speedup (1.979x) | 0.7295 speedup (2.447x) | 0.2521 | speedup (2.059x) | speedup (1.979x) | speedup (2.447x) | True |
| N1024B64_dif | 1024 | 64 | 256 | 27.7571 | 5.9016 parity (1.000x) | 5.9652 parity (0.989x) | 2.9306 speedup (2.014x) | 2.6854 speedup (2.198x) | 2.3411 speedup (2.521x) | 0.2625 | speedup (2.014x) | speedup (2.198x) | speedup (2.521x) | True |
| N1024B64_dit | 1024 | 64 | 256 | 27.7975 | 6.0904 parity (1.000x) | 6.2707 regression (0.971x) | 2.8721 speedup (2.121x) | 3.0010 speedup (2.029x) | 2.2832 speedup (2.668x) | 0.2620 | speedup (2.121x) | speedup (2.029x) | speedup (2.668x) | True |
| N2048B16_dif | 2048 | 16 | 256 | 7.5201 | 3.3981 parity (1.000x) | 3.7444 regression (0.908x) | 1.8745 speedup (1.813x) | 1.8303 speedup (1.857x) | 1.6021 speedup (2.121x) | 0.6635 | speedup (1.813x) | speedup (1.857x) | speedup (2.121x) | True |
| N2048B16_dit | 2048 | 16 | 256 | 7.4168 | 3.4757 parity (1.000x) | 3.8304 regression (0.907x) | 1.8349 speedup (1.894x) | 2.0067 speedup (1.732x) | 0.3076 speedup (11.299x) | 0.6601 | speedup (1.894x) | speedup (1.732x) | speedup (11.299x) | True |
| N2048B64_dif | 2048 | 64 | 256 | 29.3546 | 12.0526 parity (1.000x) | 12.3365 regression (0.977x) | 6.1791 speedup (1.951x) | 6.5731 speedup (1.834x) | 5.1049 speedup (2.361x) | 0.6711 | speedup (1.951x) | speedup (1.834x) | speedup (2.361x) | True |
| N2048B64_dit | 2048 | 64 | 256 | 29.2418 | 12.4548 parity (1.000x) | 12.8927 regression (0.966x) | 5.8405 speedup (2.132x) | 7.1199 speedup (1.749x) | 5.0181 speedup (2.482x) | 0.6725 | speedup (2.132x) | speedup (1.749x) | speedup (2.482x) | True |

## P2 Td canary (representative shape N=1024 B=16)

| candidate | median ms | speedup vs inc | td | verdict |
|---|---:|---:|---|---|
| incumbent_m4_folded | 1.7851 | 1.000x | not-lowered | parity |
| m2_batch | 0.8671 | 2.059x | 1 | speedup |
| m2_packet_td1 | 0.9493 | 1.880x | 1 | speedup |
| m2_packet_td2 | 0.9505 | 1.878x | 2 | speedup |

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
| N1024B16_dif | 4.012 | 1.986 | 2.034 | 1.209 | - |
| N1024B16_dit | 4.078 | 1.928 | 2.006 | 1.241 | 0.912 |
| N1024B64_dif | 4.653 | 2.221 | 2.036 | 1.284 | - |
| N1024B64_dit | 4.433 | 2.090 | 2.183 | 1.283 | - |
| N2048B16_dif | 2.008 | 2.046 | 1.998 | 1.198 | - |
| N2048B16_dit | 1.936 | 1.909 | 2.088 | 1.185 | - |
| N2048B64_dif | 2.379 | 1.877 | 1.996 | 1.219 | - |
| N2048B64_dit | 2.268 | 1.811 | 2.207 | 1.178 | - |