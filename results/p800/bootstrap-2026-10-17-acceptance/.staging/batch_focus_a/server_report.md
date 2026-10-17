# Batch release focus report (issue #3 P0 acceptance)

commit `dadd1d7daf5f` dirty=False device=kunlunxin-p800 driver=515.58 protocol=`batch-release-v1:focus:plan-reuse:interleave:correctness-first:3sess:p11`
profile_hash=`a78b86793c7142a0` table_sha256=`acc1f21c9c93fe20...` sessions=3 warmup=5 trials=30 seed=1234

Timing = api wall time (dispatch+alloc+layout+sync), ms; median of session medians over independent process sessions. Verdict vs the incumbent radix-2 folded control: >=1.02 speedup, <=0.98 regression, else parity (matches the 2% promotion gate). GM bytes in raw rows are analytical estimates (no hardware counters).

| label | N | B | blk | m0 loop | inc fold | m4 spr | m2 b | m5 b | selected | vendor | m2>=1.02? | m5>=1.02? | sel | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| N1024B16_dif | 1024 | 16 | 256 | 7.0043 | 1.6934 parity (1.000x) | 1.7583 regression (0.963x) | 0.8627 speedup (1.963x) | 0.8743 speedup (1.937x) | 0.7436 speedup (2.277x) | 0.2517 | speedup (1.963x) | speedup (1.937x) | speedup (2.277x) | True |
| N1024B16_dit | 1024 | 16 | 256 | 6.9782 | 1.7682 parity (1.000x) | 1.7424 parity (1.015x) | 0.8646 speedup (2.045x) | 0.8973 speedup (1.971x) | 0.7276 speedup (2.430x) | 0.2523 | speedup (2.045x) | speedup (1.971x) | speedup (2.430x) | True |
| N1024B64_dif | 1024 | 64 | 256 | 29.2370 | 5.9079 parity (1.000x) | 5.9821 parity (0.988x) | 2.9409 speedup (2.009x) | 2.7093 speedup (2.181x) | 2.3616 speedup (2.502x) | 0.2652 | speedup (2.009x) | speedup (2.181x) | speedup (2.502x) | True |
| N1024B64_dit | 1024 | 64 | 256 | 27.4010 | 6.0792 parity (1.000x) | 6.2759 regression (0.969x) | 2.8746 speedup (2.115x) | 3.0081 speedup (2.021x) | 2.2828 speedup (2.663x) | 0.2611 | speedup (2.115x) | speedup (2.021x) | speedup (2.663x) | True |
| N2048B16_dif | 2048 | 16 | 256 | 7.5289 | 3.3960 parity (1.000x) | 3.7470 regression (0.906x) | 1.8745 speedup (1.812x) | 1.8302 speedup (1.856x) | 1.6083 speedup (2.112x) | 0.6626 | speedup (1.812x) | speedup (1.856x) | speedup (2.112x) | True |
| N2048B16_dit | 2048 | 16 | 256 | 7.5615 | 3.4715 parity (1.000x) | 3.8250 regression (0.908x) | 1.8466 speedup (1.880x) | 2.0163 speedup (1.722x) | 0.3113 speedup (11.151x) | 0.6605 | speedup (1.880x) | speedup (1.722x) | speedup (11.151x) | True |
| N2048B64_dif | 2048 | 64 | 256 | 29.4971 | 12.0545 parity (1.000x) | 12.3328 regression (0.977x) | 6.1826 speedup (1.950x) | 6.6014 speedup (1.826x) | 5.1198 speedup (2.355x) | 0.6725 | speedup (1.950x) | speedup (1.826x) | speedup (2.355x) | True |
| N2048B64_dit | 2048 | 64 | 256 | 29.6107 | 12.4446 parity (1.000x) | 12.8969 regression (0.965x) | 5.8423 speedup (2.130x) | 7.1432 speedup (1.742x) | 5.0132 speedup (2.482x) | 0.6723 | speedup (2.130x) | speedup (1.742x) | speedup (2.482x) | True |

## P2 Td canary (representative shape N=1024 B=16)

| candidate | median ms | speedup vs inc | td | verdict |
|---|---:|---:|---|---|
| incumbent_m4_folded | 1.7682 | 1.000x | not-lowered | parity |
| m2_batch | 0.8646 | 2.045x | 1 | speedup |
| m2_packet_td1 | 0.9449 | 1.871x | 1 | speedup |
| m2_packet_td2 | 0.9435 | 1.874x | 2 | speedup |

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
| N1024B16_dif | 3.984 | 2.011 | 2.038 | 1.213 | - |
| N1024B16_dit | 4.005 | 1.942 | 2.015 | 1.247 | 0.916 |
| N1024B64_dif | 4.887 | 2.208 | 2.034 | 1.279 | - |
| N1024B64_dit | 4.366 | 2.086 | 2.183 | 1.286 | - |
| N2048B16_dif | 2.009 | 2.047 | 1.999 | 1.198 | - |
| N2048B16_dit | 1.977 | 1.897 | 2.071 | 1.191 | - |
| N2048B64_dif | 2.392 | 1.868 | 1.995 | 1.222 | - |
| N2048B64_dit | 2.296 | 1.805 | 2.208 | 1.176 | - |