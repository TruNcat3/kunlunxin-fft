# Batch release focus report (issue #3 P0 acceptance)

commit `746261d30f46` dirty=False device=kunlunxin-p800 driver=515.58 protocol=`batch-release-v1:focus:plan-reuse:interleave:correctness-first:3sess:p11:m2-fullspace-holdout`
profile_hash=`a78b86793c7142a0` table_sha256=`eb1e5f388a92f2f2...` sessions=3 warmup=5 trials=30 seed=1234

Timing = api wall time (dispatch+alloc+layout+sync), ms; median of session medians over independent process sessions. Verdict vs the incumbent radix-2 folded control: >=1.02 speedup, <=0.98 regression, else parity (matches the 2% promotion gate). GM bytes in raw rows are analytical estimates (no hardware counters).

| label | N | B | blk | m0 loop | inc fold | m4 spr | m2 b | m5 b | selected | vendor | m2>=1.02? | m5>=1.02? | sel | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| N1024B16_dif | 1024 | 16 | 256 | 7.4776 | 1.7318 parity (1.000x) | 1.8102 regression (0.957x) | 0.9048 speedup (1.914x) | 0.8843 speedup (1.958x) | 0.7858 speedup (2.204x) | 0.2556 | speedup (1.914x) | speedup (1.958x) | speedup (2.204x) | True |
| N1024B16_dit | 1024 | 16 | 256 | 7.5126 | 1.8032 parity (1.000x) | 1.7501 speedup (1.030x) | 0.8545 speedup (2.110x) | 0.9231 speedup (1.953x) | 0.7499 speedup (2.404x) | 0.2554 | speedup (2.110x) | speedup (1.953x) | speedup (2.404x) | True |
| N1024B64_dif | 1024 | 64 | 256 | 27.9377 | 6.3319 parity (1.000x) | 6.3819 parity (0.992x) | 3.0372 speedup (2.085x) | 2.8617 speedup (2.213x) | 2.4120 speedup (2.625x) | 0.2646 | speedup (2.085x) | speedup (2.213x) | speedup (2.625x) | True |
| N1024B64_dit | 1024 | 64 | 256 | 27.8933 | 6.1295 parity (1.000x) | 6.3977 regression (0.958x) | 2.9547 speedup (2.075x) | 3.1265 speedup (1.960x) | 2.3254 speedup (2.636x) | 0.2658 | speedup (2.075x) | speedup (1.960x) | speedup (2.636x) | True |
| N2048B16_dif | 2048 | 16 | 256 | 7.5963 | 3.4151 parity (1.000x) | 4.0452 regression (0.844x) | 1.9932 speedup (1.713x) | 2.0045 speedup (1.704x) | 1.7000 speedup (2.009x) | 0.6645 | speedup (1.713x) | speedup (1.704x) | speedup (2.009x) | True |
| N2048B16_dit | 2048 | 16 | 256 | 7.5455 | 3.5269 parity (1.000x) | 4.2026 regression (0.839x) | 1.8208 speedup (1.937x) | 2.2119 speedup (1.595x) | 0.3230 speedup (10.919x) | 0.6630 | speedup (1.937x) | speedup (1.595x) | speedup (10.919x) | True |

## P2 Td canary (representative shape N=1024 B=16)

| candidate | median ms | speedup vs inc | td | verdict |
|---|---:|---:|---|---|
| incumbent_m4_folded | 1.8032 | 1.000x | not-lowered | parity |
| m2_batch | 0.8545 | 2.110x | 1 | speedup |
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
| N1024B16_dif | 4.131 | 2.047 | 2.001 | 1.211 | - |
| N1024B16_dit | 4.293 | 1.896 | 2.048 | 1.206 | - |
| N1024B64_dif | 4.378 | 2.230 | 2.101 | 1.283 | - |
| N1024B64_dit | 4.360 | 2.046 | 2.165 | 1.296 | - |
| N2048B16_dif | 1.878 | 2.018 | 2.030 | 1.203 | - |
| N2048B16_dit | 1.795 | 1.900 | 2.308 | 1.191 | - |