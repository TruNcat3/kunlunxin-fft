# Batch release focus report (issue #3 P0 acceptance)

commit `0fb91f9a8d2f` dirty=False device=kunlunxin-p800 driver=515.58 protocol=`batch-release-v1:focus:plan-reuse:interleave:correctness-first:3sess:p11:m2-fullspace-holdout`
profile_hash=`a78b86793c7142a0` table_sha256=`f172147dc2c3e875...` sessions=3 warmup=5 trials=30 seed=1234

Timing = api wall time (dispatch+alloc+layout+sync), ms; median of session medians over independent process sessions. Verdict vs the incumbent radix-2 folded control: >=1.02 speedup, <=0.98 regression, else parity (matches the 2% promotion gate). GM bytes in raw rows are analytical estimates (no hardware counters).

| label | N | B | blk | m0 loop | inc fold | m4 spr | m2 b | m5 b | selected | vendor | m2>=1.02? | m5>=1.02? | sel | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| N1024B16_dif | 1024 | 16 | 256 | 7.0015 | 1.7121 parity (1.000x) | 1.7952 regression (0.954x) | 0.8933 speedup (1.917x) | 0.8589 speedup (1.993x) | 0.7717 speedup (2.219x) | 0.2531 | speedup (1.917x) | speedup (1.993x) | speedup (2.219x) | True |
| N1024B16_dit | 1024 | 16 | 256 | 7.0060 | 1.7640 parity (1.000x) | 1.7276 speedup (1.021x) | 0.8362 speedup (2.109x) | 0.9128 speedup (1.932x) | 0.7238 speedup (2.437x) | 0.2528 | speedup (2.109x) | speedup (1.932x) | speedup (2.437x) | True |
| N1024B64_dif | 1024 | 64 | 256 | 27.6603 | 6.3338 parity (1.000x) | 6.3624 parity (0.996x) | 3.0218 speedup (2.096x) | 2.8607 speedup (2.214x) | 2.4117 speedup (2.626x) | 0.2644 | speedup (2.096x) | speedup (2.214x) | speedup (2.626x) | True |
| N1024B64_dit | 1024 | 64 | 256 | 27.6560 | 6.1043 parity (1.000x) | 6.3909 regression (0.955x) | 2.9382 speedup (2.078x) | 3.1691 speedup (1.926x) | 2.3190 speedup (2.632x) | 0.2633 | speedup (2.078x) | speedup (1.926x) | speedup (2.632x) | True |
| N2048B16_dif | 2048 | 16 | 256 | 7.5279 | 3.4253 parity (1.000x) | 4.0354 regression (0.849x) | 1.9934 speedup (1.718x) | 2.0029 speedup (1.710x) | 1.6862 speedup (2.031x) | 0.6655 | speedup (1.718x) | speedup (1.710x) | speedup (2.031x) | True |
| N2048B16_dit | 2048 | 16 | 256 | 7.5205 | 3.5271 parity (1.000x) | 4.2049 regression (0.839x) | 1.8305 speedup (1.927x) | 2.2005 speedup (1.603x) | 0.3203 speedup (11.010x) | 0.6633 | speedup (1.927x) | speedup (1.603x) | speedup (11.010x) | True |

## P2 Td canary (representative shape N=1024 B=16)

| candidate | median ms | speedup vs inc | td | verdict |
|---|---:|---:|---|---|
| incumbent_m4_folded | 1.7640 | 1.000x | not-lowered | parity |
| m2_batch | 0.8362 | 2.109x | 1 | speedup |
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
| N1024B16_dif | 3.900 | 2.090 | 2.010 | 1.214 | - |
| N1024B16_dit | 4.055 | 1.893 | 2.066 | 1.207 | - |
| N1024B64_dif | 4.347 | 2.224 | 2.106 | 1.280 | - |
| N1024B64_dit | 4.327 | 2.017 | 2.175 | 1.289 | - |
| N2048B16_dif | 1.865 | 2.015 | 2.024 | 1.208 | - |
| N2048B16_dit | 1.788 | 1.911 | 2.297 | 1.193 | - |