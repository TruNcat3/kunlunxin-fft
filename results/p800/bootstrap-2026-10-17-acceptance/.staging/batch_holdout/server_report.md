# Batch release focus report (issue #3 P0 acceptance)

commit `dadd1d7daf5f` dirty=False device=kunlunxin-p800 driver=515.58 protocol=`batch-release-v1:focus:plan-reuse:interleave:correctness-first:3sess:p11:m2-fullspace-holdout`
profile_hash=`a78b86793c7142a0` table_sha256=`acc1f21c9c93fe20...` sessions=3 warmup=5 trials=30 seed=1234

Timing = api wall time (dispatch+alloc+layout+sync), ms; median of session medians over independent process sessions. Verdict vs the incumbent radix-2 folded control: >=1.02 speedup, <=0.98 regression, else parity (matches the 2% promotion gate). GM bytes in raw rows are analytical estimates (no hardware counters).

| label | N | B | blk | m0 loop | inc fold | m4 spr | m2 b | m5 b | selected | vendor | m2>=1.02? | m5>=1.02? | sel | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| N1024B16_dif | 1024 | 16 | 256 | 6.9684 | 1.7255 parity (1.000x) | 1.7935 regression (0.962x) | 0.8986 speedup (1.920x) | 0.8676 speedup (1.989x) | 0.7645 speedup (2.257x) | 0.2527 | speedup (1.920x) | speedup (1.989x) | speedup (2.257x) | True |
| N1024B16_dit | 1024 | 16 | 256 | 7.0150 | 1.7923 parity (1.000x) | 1.7444 speedup (1.027x) | 0.8377 speedup (2.139x) | 0.9087 speedup (1.972x) | 0.7300 speedup (2.455x) | 0.2524 | speedup (2.139x) | speedup (1.972x) | speedup (2.455x) | True |
| N1024B64_dif | 1024 | 64 | 256 | 28.0174 | 6.3490 parity (1.000x) | 6.3831 parity (0.995x) | 3.0192 speedup (2.103x) | 2.8649 speedup (2.216x) | 2.4144 speedup (2.630x) | 0.2647 | speedup (2.103x) | speedup (2.216x) | speedup (2.630x) | True |
| N1024B64_dit | 1024 | 64 | 256 | 27.4974 | 6.1205 parity (1.000x) | 6.3793 regression (0.959x) | 2.9473 speedup (2.077x) | 3.1313 speedup (1.955x) | 2.3322 speedup (2.624x) | 0.2650 | speedup (2.077x) | speedup (1.955x) | speedup (2.624x) | True |
| N2048B16_dif | 2048 | 16 | 256 | 7.4259 | 3.4228 parity (1.000x) | 4.0424 regression (0.847x) | 1.9907 speedup (1.719x) | 2.0042 speedup (1.708x) | 1.7003 speedup (2.013x) | 0.6654 | speedup (1.719x) | speedup (1.708x) | speedup (2.013x) | True |
| N2048B16_dit | 2048 | 16 | 256 | 7.6302 | 3.5350 parity (1.000x) | 4.2149 regression (0.839x) | 1.8233 speedup (1.939x) | 2.2268 speedup (1.588x) | 0.3270 speedup (10.811x) | 0.6642 | speedup (1.939x) | speedup (1.588x) | speedup (10.811x) | True |

## P2 Td canary (representative shape N=1024 B=16)

| candidate | median ms | speedup vs inc | td | verdict |
|---|---:|---:|---|---|
| incumbent_m4_folded | 1.7923 | 1.000x | not-lowered | parity |
| m2_batch | 0.8377 | 2.139x | 1 | speedup |
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
| N1024B16_dif | 3.885 | 2.067 | 1.996 | 1.226 | - |
| N1024B16_dit | 4.021 | 1.920 | 2.082 | 1.212 | - |
| N1024B64_dif | 4.389 | 2.228 | 2.114 | 1.273 | - |
| N1024B64_dit | 4.310 | 2.037 | 2.164 | 1.293 | - |
| N2048B16_dif | 1.837 | 2.017 | 2.031 | 1.199 | - |
| N2048B16_dit | 1.810 | 1.893 | 2.312 | 1.193 | - |