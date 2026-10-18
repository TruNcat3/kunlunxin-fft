# Batch release focus report (issue #3 P0 acceptance)

commit `39bd915c054b` dirty=False device=kunlunxin-p800 driver=515.58 protocol=`batch-release-v1:focus:plan-reuse:interleave:correctness-first:3sess:p11:m2-fullspace-holdout`
profile_hash=`a78b86793c7142a0` table_sha256=`f172147dc2c3e875...` sessions=3 warmup=5 trials=30 seed=1234

Timing = api wall time (dispatch+alloc+layout+sync), ms; median of session medians over independent process sessions. Verdict vs the incumbent radix-2 folded control: >=1.02 speedup, <=0.98 regression, else parity (matches the 2% promotion gate). GM bytes in raw rows are analytical estimates (no hardware counters).

| label | N | B | blk | m0 loop | inc fold | m4 spr | m2 b | m5 b | selected | vendor | m2>=1.02? | m5>=1.02? | sel | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| N1024B16_dif | 1024 | 16 | 256 | 7.0325 | 1.7118 parity (1.000x) | 1.8093 regression (0.946x) | 0.8901 speedup (1.923x) | 0.8930 speedup (1.917x) | 0.7684 speedup (2.228x) | 0.2534 | speedup (1.923x) | speedup (1.917x) | speedup (2.228x) | True |
| N1024B16_dit | 1024 | 16 | 256 | 7.0357 | 1.7678 parity (1.000x) | 1.7461 parity (1.012x) | 0.8377 speedup (2.110x) | 0.9391 speedup (1.882x) | 0.7241 speedup (2.441x) | 0.2533 | speedup (2.110x) | speedup (1.882x) | speedup (2.441x) | True |
| N1024B64_dif | 1024 | 64 | 256 | 27.7146 | 6.3212 parity (1.000x) | 6.3572 parity (0.994x) | 3.0172 speedup (2.095x) | 2.8649 speedup (2.206x) | 2.4071 speedup (2.626x) | 0.2649 | speedup (2.095x) | speedup (2.206x) | speedup (2.626x) | True |
| N1024B64_dit | 1024 | 64 | 256 | 27.7854 | 6.1093 parity (1.000x) | 6.3833 regression (0.957x) | 2.9516 speedup (2.070x) | 3.1598 speedup (1.933x) | 2.3296 speedup (2.622x) | 0.2649 | speedup (2.070x) | speedup (1.933x) | speedup (2.622x) | True |
| N2048B16_dif | 2048 | 16 | 256 | 7.5032 | 3.4188 parity (1.000x) | 4.0231 regression (0.850x) | 1.9905 speedup (1.718x) | 2.0071 speedup (1.703x) | 1.6879 speedup (2.026x) | 0.6642 | speedup (1.718x) | speedup (1.703x) | speedup (2.026x) | True |
| N2048B16_dit | 2048 | 16 | 256 | 7.5221 | 3.5249 parity (1.000x) | 4.1832 regression (0.843x) | 1.8240 speedup (1.932x) | 2.2070 speedup (1.597x) | 0.3240 speedup (10.878x) | 0.6637 | speedup (1.932x) | speedup (1.597x) | speedup (10.878x) | True |

## P2 Td canary (representative shape N=1024 B=16)

| candidate | median ms | speedup vs inc | td | verdict |
|---|---:|---:|---|---|
| incumbent_m4_folded | 1.7678 | 1.000x | not-lowered | parity |
| m2_batch | 0.8377 | 2.110x | 1 | speedup |
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
| N1024B16_dif | 3.887 | 2.026 | 2.033 | 1.210 | - |
| N1024B16_dit | 4.029 | 1.859 | 2.084 | 1.209 | - |
| N1024B64_dif | 4.360 | 2.219 | 2.107 | 1.271 | - |
| N1024B64_dit | 4.353 | 2.020 | 2.163 | 1.297 | - |
| N2048B16_dif | 1.865 | 2.004 | 2.021 | 1.204 | - |
| N2048B16_dit | 1.798 | 1.895 | 2.293 | 1.192 | - |