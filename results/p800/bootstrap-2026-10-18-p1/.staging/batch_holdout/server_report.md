# Batch release focus report (issue #3 P0 acceptance)

commit `1dc7775aaebd` dirty=False device=kunlunxin-p800 driver=515.58 protocol=`batch-release-v1:focus:plan-reuse:interleave:correctness-first:3sess:p11:m2-fullspace-holdout`
profile_hash=`a78b86793c7142a0` table_sha256=`3f64e677379a232d...` sessions=3 warmup=5 trials=30 seed=1234

Timing = api wall time (dispatch+alloc+layout+sync), ms; median of session medians over independent process sessions. Verdict vs the incumbent radix-2 folded control: >=1.02 speedup, <=0.98 regression, else parity (matches the 2% promotion gate). GM bytes in raw rows are analytical estimates (no hardware counters).

| label | N | B | blk | m0 loop | inc fold | m4 spr | m2 b | m5 b | selected | vendor | m2>=1.02? | m5>=1.02? | sel | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| N1024B16_dif | 1024 | 16 | 256 | 7.0121 | 1.7154 parity (1.000x) | 1.7957 regression (0.955x) | 0.8889 speedup (1.930x) | 0.8612 speedup (1.992x) | 0.7697 speedup (2.229x) | 0.2530 | speedup (1.930x) | speedup (1.992x) | speedup (2.229x) | True |
| N1024B16_dit | 1024 | 16 | 256 | 7.0116 | 1.7696 parity (1.000x) | 1.7338 speedup (1.021x) | 0.8334 speedup (2.123x) | 0.9120 speedup (1.940x) | 0.7295 speedup (2.426x) | 0.2526 | speedup (2.123x) | speedup (1.940x) | speedup (2.426x) | True |
| N1024B64_dif | 1024 | 64 | 256 | 27.9405 | 6.3457 parity (1.000x) | 6.3612 parity (0.998x) | 3.0132 speedup (2.106x) | 2.8750 speedup (2.207x) | 2.4086 speedup (2.635x) | 0.2642 | speedup (2.106x) | speedup (2.207x) | speedup (2.635x) | True |
| N1024B64_dit | 1024 | 64 | 256 | 28.6610 | 6.0962 parity (1.000x) | 6.4001 regression (0.953x) | 2.9532 speedup (2.064x) | 3.1795 speedup (1.917x) | 2.3225 speedup (2.625x) | 0.2657 | speedup (2.064x) | speedup (1.917x) | speedup (2.625x) | True |
| N2048B16_dif | 2048 | 16 | 256 | 7.6483 | 3.4176 parity (1.000x) | 4.0512 regression (0.844x) | 2.0005 speedup (1.708x) | 2.0002 speedup (1.709x) | 1.6924 speedup (2.019x) | 0.6668 | speedup (1.708x) | speedup (1.709x) | speedup (2.019x) | True |
| N2048B16_dit | 2048 | 16 | 256 | 7.5777 | 3.5269 parity (1.000x) | 4.1991 regression (0.840x) | 1.8246 speedup (1.933x) | 2.1945 speedup (1.607x) | 0.3212 speedup (10.980x) | 0.6628 | speedup (1.933x) | speedup (1.607x) | speedup (10.980x) | True |

## P2 Td canary (representative shape N=1024 B=16)

| candidate | median ms | speedup vs inc | td | verdict |
|---|---:|---:|---|---|
| incumbent_m4_folded | 1.7696 | 1.000x | not-lowered | parity |
| m2_batch | 0.8334 | 2.123x | 1 | speedup |
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
| N1024B16_dif | 3.905 | 2.085 | 2.020 | 1.208 | - |
| N1024B16_dit | 4.044 | 1.901 | 2.080 | 1.200 | - |
| N1024B64_dif | 4.392 | 2.213 | 2.111 | 1.278 | - |
| N1024B64_dit | 4.478 | 2.013 | 2.167 | 1.297 | - |
| N2048B16_dif | 1.888 | 2.025 | 2.025 | 1.212 | - |
| N2048B16_dit | 1.805 | 1.913 | 2.301 | 1.192 | - |