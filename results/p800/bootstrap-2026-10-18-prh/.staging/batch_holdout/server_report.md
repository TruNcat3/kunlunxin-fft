# Batch release focus report (issue #3 P0 acceptance)

commit `b3b48b9b4082` dirty=False device=kunlunxin-p800 driver=515.58 protocol=`batch-release-v1:focus:plan-reuse:interleave:correctness-first:3sess:p11:m2-fullspace-holdout`
profile_hash=`a78b86793c7142a0` table_sha256=`f172147dc2c3e875...` sessions=3 warmup=5 trials=30 seed=1234

Timing = api wall time (dispatch+alloc+layout+sync), ms; median of session medians over independent process sessions. Verdict vs the incumbent radix-2 folded control: >=1.02 speedup, <=0.98 regression, else parity (matches the 2% promotion gate). GM bytes in raw rows are analytical estimates (no hardware counters).

| label | N | B | blk | m0 loop | inc fold | m4 spr | m2 b | m5 b | selected | vendor | m2>=1.02? | m5>=1.02? | sel | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| N1024B16_dif | 1024 | 16 | 256 | 7.1489 | 1.7294 parity (1.000x) | 1.8088 regression (0.956x) | 0.8899 speedup (1.943x) | 0.8773 speedup (1.971x) | 0.7806 speedup (2.215x) | 0.2541 | speedup (1.943x) | speedup (1.971x) | speedup (2.215x) | True |
| N1024B16_dit | 1024 | 16 | 256 | 7.3939 | 1.7880 parity (1.000x) | 1.7549 parity (1.019x) | 0.8447 speedup (2.117x) | 0.9188 speedup (1.946x) | 0.7481 speedup (2.390x) | 0.2547 | speedup (2.117x) | speedup (1.946x) | speedup (2.390x) | True |
| N1024B64_dif | 1024 | 64 | 256 | 27.6565 | 6.3321 parity (1.000x) | 6.3877 parity (0.991x) | 3.0308 speedup (2.089x) | 2.8615 speedup (2.213x) | 2.4241 speedup (2.612x) | 0.2655 | speedup (2.089x) | speedup (2.213x) | speedup (2.612x) | True |
| N1024B64_dit | 1024 | 64 | 256 | 27.9766 | 6.1197 parity (1.000x) | 6.3874 regression (0.958x) | 2.9528 speedup (2.072x) | 3.1290 speedup (1.956x) | 2.3291 speedup (2.627x) | 0.2644 | speedup (2.072x) | speedup (1.956x) | speedup (2.627x) | True |
| N2048B16_dif | 2048 | 16 | 256 | 7.4405 | 3.4130 parity (1.000x) | 4.0367 regression (0.845x) | 1.9873 speedup (1.717x) | 1.9999 speedup (1.707x) | 1.6878 speedup (2.022x) | 0.6648 | speedup (1.717x) | speedup (1.707x) | speedup (2.022x) | True |
| N2048B16_dit | 2048 | 16 | 256 | 7.4940 | 3.5301 parity (1.000x) | 4.2151 regression (0.837x) | 1.8169 speedup (1.943x) | 2.1975 speedup (1.606x) | 0.3209 speedup (11.002x) | 0.6615 | speedup (1.943x) | speedup (1.606x) | speedup (11.002x) | True |

## P2 Td canary (representative shape N=1024 B=16)

| candidate | median ms | speedup vs inc | td | verdict |
|---|---:|---:|---|---|
| incumbent_m4_folded | 1.7880 | 1.000x | not-lowered | parity |
| m2_batch | 0.8447 | 2.117x | 1 | speedup |
| m2_packet_td1 | nan | nanx | None | None |
| m2_packet_td2 | nan | nanx | None | None |

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
| N1024B16_dif | 3.952 | 2.062 | 2.033 | 1.203 | - |
| N1024B16_dit | 4.213 | 1.910 | 2.078 | 1.196 | - |
| N1024B64_dif | 4.330 | 2.232 | 2.108 | 1.282 | - |
| N1024B64_dit | 4.380 | 2.041 | 2.163 | 1.294 | - |
| N2048B16_dif | 1.843 | 2.019 | 2.031 | 1.202 | - |
| N2048B16_dit | 1.778 | 1.918 | 2.320 | 1.190 | - |