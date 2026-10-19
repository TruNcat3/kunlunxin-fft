# Batch release focus report (issue #3 P0 acceptance)

commit `dbb8c5b7432d` dirty=False device=kunlunxin-p800 driver=515.58 protocol=`batch-release-v1:focus:plan-reuse:interleave:correctness-first:3sess:p11:m2-fullspace-holdout`
profile_hash=`a78b86793c7142a0` table_sha256=`602782a3ad785fa0...` sessions=3 warmup=5 trials=30 seed=1234

Timing = api wall time (dispatch+alloc+layout+sync), ms; median of session medians over independent process sessions. Verdict vs the incumbent radix-2 folded control: >=1.02 speedup, <=0.98 regression, else parity (matches the 2% promotion gate). GM bytes in raw rows are analytical estimates (no hardware counters).

| label | N | B | blk | m0 loop | inc fold | m4 spr | m2 b | m5 b | selected | vendor | m2>=1.02? | m5>=1.02? | sel | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| N1024B16_dif | 1024 | 16 | 256 | 7.0385 | 1.7500 parity (1.000x) | 1.8551 regression (0.943x) | 0.8871 speedup (1.973x) | 0.9109 speedup (1.921x) | 0.7625 speedup (2.295x) | 0.2533 | speedup (1.973x) | speedup (1.921x) | speedup (2.295x) | True |
| N1024B16_dit | 1024 | 16 | 256 | 7.0525 | 1.8663 parity (1.000x) | 1.8575 parity (1.005x) | 0.8922 speedup (2.092x) | 0.9907 speedup (1.884x) | 0.8914 speedup (2.094x) | 0.2532 | speedup (2.092x) | speedup (1.884x) | speedup (2.094x) | True |
| N1024B64_dif | 1024 | 64 | 256 | 27.7418 | 6.2540 parity (1.000x) | 6.3920 regression (0.978x) | 3.0016 speedup (2.084x) | 2.7624 speedup (2.264x) | 2.3874 speedup (2.620x) | 0.2643 | speedup (2.084x) | speedup (2.264x) | speedup (2.620x) | True |
| N1024B64_dit | 1024 | 64 | 256 | 27.7315 | 6.0948 parity (1.000x) | 6.3690 regression (0.957x) | 2.9223 speedup (2.086x) | 3.0063 speedup (2.027x) | 2.9674 speedup (2.054x) | 0.2644 | speedup (2.086x) | speedup (2.027x) | speedup (2.054x) | True |
| N2048B16_dif | 2048 | 16 | 256 | 7.5004 | 3.4667 parity (1.000x) | 3.9405 regression (0.880x) | 1.9023 speedup (1.822x) | 1.9038 speedup (1.821x) | 1.6491 speedup (2.102x) | 0.6649 | speedup (1.822x) | speedup (1.821x) | speedup (2.102x) | True |
| N2048B16_dit | 2048 | 16 | 256 | 7.4940 | 3.7289 parity (1.000x) | 3.9661 regression (0.940x) | 1.8340 speedup (2.033x) | 1.9985 speedup (1.866x) | 0.3186 speedup (11.706x) | 0.6623 | speedup (2.033x) | speedup (1.866x) | speedup (11.706x) | True |

## P2 Td canary (representative shape N=1024 B=16)

| candidate | median ms | speedup vs inc | td | verdict |
|---|---:|---:|---|---|
| incumbent_m4_folded | 1.8663 | 1.000x | not-lowered | parity |
| m2_batch | 0.8922 | 2.092x | 1 | speedup |
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
| N1024B16_dif | 3.794 | 2.037 | 2.091 | 1.212 | - |
| N1024B16_dit | 3.797 | 1.875 | 2.082 | 1.221 | - |
| N1024B64_dif | 4.340 | 2.314 | 2.130 | 1.282 | - |
| N1024B64_dit | 4.354 | 2.119 | 2.179 | 1.293 | - |
| N2048B16_dif | 1.903 | 2.070 | 2.071 | 1.179 | - |
| N2048B16_dit | 1.890 | 1.985 | 2.163 | 1.182 | - |