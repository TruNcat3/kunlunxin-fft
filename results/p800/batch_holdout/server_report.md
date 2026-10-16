# Batch release focus report (issue #3 P0 acceptance)

commit `ec19fd3e8a60` dirty=True device=kunlunxin-p800 driver=515.58 protocol=`batch-release-v1:focus:plan-reuse:interleave:correctness-first:3sess:p11:m2-fullspace-holdout`
profile_hash=`a78b86793c7142a0` table_sha256=`93be5746f70863c6...` sessions=3 warmup=5 trials=30 seed=1234

Timing = api wall time (dispatch+alloc+layout+sync), ms; median of session medians over independent process sessions. Verdict vs the incumbent radix-2 folded control: >=1.02 speedup, <=0.98 regression, else parity (matches the 2% promotion gate). GM bytes in raw rows are analytical estimates (no hardware counters).

| label | N | B | blk | m0 loop | inc fold | m4 spr | m2 b | m5 b | selected | vendor | m2>=1.02? | m5>=1.02? | sel | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| N1024B16_dif | 1024 | 16 | 256 | 7.0909 | 1.7232 parity (1.000x) | 1.8007 regression (0.957x) | 0.8852 speedup (1.947x) | 0.9037 speedup (1.907x) | 0.7502 speedup (2.297x) | 0.2532 | speedup (1.947x) | speedup (1.907x) | speedup (2.297x) | True |
| N1024B16_dit | 1024 | 16 | 256 | 7.1254 | 1.7624 parity (1.000x) | 1.7383 parity (1.014x) | 0.8330 speedup (2.116x) | 0.9484 speedup (1.858x) | 0.7166 speedup (2.459x) | 0.2535 | speedup (2.116x) | speedup (1.858x) | speedup (2.459x) | True |
| N1024B64_dif | 1024 | 64 | 256 | 28.1172 | 6.3370 parity (1.000x) | 6.4046 parity (0.989x) | 3.0299 speedup (2.091x) | 2.8561 speedup (2.219x) | 2.3939 speedup (2.647x) | 0.2658 | speedup (2.091x) | speedup (2.219x) | speedup (2.647x) | True |
| N1024B64_dit | 1024 | 64 | 256 | 28.3820 | 6.1212 parity (1.000x) | 6.3866 regression (0.958x) | 2.9453 speedup (2.078x) | 3.1288 speedup (1.956x) | 2.3071 speedup (2.653x) | 0.2677 | speedup (2.078x) | speedup (1.956x) | speedup (2.653x) | True |
| N2048B16_dif | 2048 | 16 | 256 | 7.5698 | 3.3230 parity (1.000x) | 3.9917 regression (0.832x) | 1.9181 speedup (1.732x) | 1.8893 speedup (1.759x) | 1.6116 speedup (2.062x) | 0.6651 | speedup (1.732x) | speedup (1.759x) | speedup (2.062x) | True |
| N2048B16_dit | 2048 | 16 | 256 | 7.5552 | 3.4240 parity (1.000x) | 4.2116 regression (0.813x) | 1.9859 speedup (1.724x) | 2.0863 speedup (1.641x) | 1.6315 speedup (2.099x) | 0.6666 | speedup (1.724x) | speedup (1.641x) | speedup (2.099x) | True |

## P2 Td canary (representative shape N=1024 B=16)

| candidate | median ms | speedup vs inc | td | verdict |
|---|---:|---:|---|---|
| incumbent_m4_folded | 1.7624 | 1.000x | not-lowered | parity |
| m2_batch | 0.8330 | 2.116x | 1 | speedup |
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
| N1024B16_dif | 3.938 | 1.993 | 2.034 | 1.212 | - |
| N1024B16_dit | 4.099 | 1.833 | 2.087 | 1.195 | - |
| N1024B64_dif | 4.390 | 2.242 | 2.114 | 1.277 | - |
| N1024B64_dit | 4.444 | 2.041 | 2.168 | 1.287 | - |
| N2048B16_dif | 1.896 | 2.113 | 2.081 | 1.202 | - |
| N2048B16_dit | 1.794 | 2.019 | 2.121 | 1.235 | - |