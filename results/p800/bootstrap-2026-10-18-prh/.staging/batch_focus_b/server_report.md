# Batch release focus report (issue #3 P0 acceptance)

commit `b3b48b9b4082` dirty=False device=kunlunxin-p800 driver=515.58 protocol=`batch-release-v1:focus:plan-reuse:interleave:correctness-first:3sess:p11`
profile_hash=`a78b86793c7142a0` table_sha256=`f172147dc2c3e875...` sessions=3 warmup=5 trials=30 seed=1234

Timing = api wall time (dispatch+alloc+layout+sync), ms; median of session medians over independent process sessions. Verdict vs the incumbent radix-2 folded control: >=1.02 speedup, <=0.98 regression, else parity (matches the 2% promotion gate). GM bytes in raw rows are analytical estimates (no hardware counters).

| label | N | B | blk | m0 loop | inc fold | m4 spr | m2 b | m5 b | selected | vendor | m2>=1.02? | m5>=1.02? | sel | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| N1024B16_dif | 1024 | 16 | 256 | 6.9860 | 1.6831 parity (1.000x) | 1.7627 regression (0.955x) | 0.8671 speedup (1.941x) | 0.8841 speedup (1.904x) | 0.7471 speedup (2.253x) | 0.2522 | speedup (1.941x) | speedup (1.904x) | speedup (2.253x) | True |
| N1024B16_dit | 1024 | 16 | 256 | 7.0193 | 1.7764 parity (1.000x) | 1.7727 parity (1.002x) | 0.8693 speedup (2.044x) | 0.9432 speedup (1.883x) | 0.7315 speedup (2.428x) | 0.2525 | speedup (2.044x) | speedup (1.883x) | speedup (2.428x) | True |
| N1024B64_dif | 1024 | 64 | 256 | 27.4764 | 5.8867 parity (1.000x) | 5.9442 parity (0.990x) | 2.9283 speedup (2.010x) | 2.7171 speedup (2.167x) | 2.3529 speedup (2.502x) | 0.2632 | speedup (2.010x) | speedup (2.167x) | speedup (2.502x) | True |
| N1024B64_dit | 1024 | 64 | 256 | 27.4793 | 6.0903 parity (1.000x) | 6.2326 regression (0.977x) | 2.8724 speedup (2.120x) | 3.0378 speedup (2.005x) | 2.2891 speedup (2.661x) | 0.2628 | speedup (2.120x) | speedup (2.005x) | speedup (2.661x) | True |
| N2048B16_dif | 2048 | 16 | 256 | 7.4646 | 3.3811 parity (1.000x) | 3.7266 regression (0.907x) | 1.8745 speedup (1.804x) | 1.8344 speedup (1.843x) | 1.6070 speedup (2.104x) | 0.6645 | speedup (1.804x) | speedup (1.843x) | speedup (2.104x) | True |
| N2048B16_dit | 2048 | 16 | 256 | 7.4474 | 3.4898 parity (1.000x) | 3.8369 regression (0.910x) | 1.8464 speedup (1.890x) | 1.9904 speedup (1.753x) | 0.3158 speedup (11.052x) | 0.6615 | speedup (1.890x) | speedup (1.753x) | speedup (11.052x) | True |
| N2048B64_dif | 2048 | 64 | 256 | 29.1152 | 12.0271 parity (1.000x) | 12.3333 regression (0.975x) | 6.1806 speedup (1.946x) | 6.5861 speedup (1.826x) | 5.1215 speedup (2.348x) | 0.6728 | speedup (1.946x) | speedup (1.826x) | speedup (2.348x) | True |
| N2048B64_dit | 2048 | 64 | 256 | 29.3148 | 12.4506 parity (1.000x) | 12.8672 regression (0.968x) | 5.8409 speedup (2.132x) | 7.1069 speedup (1.752x) | 5.0062 speedup (2.487x) | 0.6735 | speedup (2.132x) | speedup (1.752x) | speedup (2.487x) | True |

## P2 Td canary (representative shape N=1024 B=16)

| candidate | median ms | speedup vs inc | td | verdict |
|---|---:|---:|---|---|
| incumbent_m4_folded | 1.7764 | 1.000x | not-lowered | parity |
| m2_batch | 0.8693 | 2.044x | 1 | speedup |
| m2_packet_td1 | 0.9420 | 1.886x | 1 | speedup |
| m2_packet_td2 | 0.9429 | 1.884x | 2 | speedup |

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
| N1024B16_dif | 3.963 | 1.994 | 2.033 | 1.210 | - |
| N1024B16_dit | 3.960 | 1.880 | 2.039 | 1.252 | 0.922 |
| N1024B64_dif | 4.622 | 2.188 | 2.030 | 1.275 | - |
| N1024B64_dit | 4.409 | 2.052 | 2.170 | 1.278 | - |
| N2048B16_dif | 2.003 | 2.031 | 1.988 | 1.197 | - |
| N2048B16_dit | 1.941 | 1.928 | 2.078 | 1.191 | - |
| N2048B64_dif | 2.361 | 1.873 | 1.996 | 1.218 | - |
| N2048B64_dit | 2.278 | 1.811 | 2.203 | 1.177 | - |