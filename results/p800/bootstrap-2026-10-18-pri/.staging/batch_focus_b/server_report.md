# Batch release focus report (issue #3 P0 acceptance)

commit `dbb8c5b7432d` dirty=False device=kunlunxin-p800 driver=515.58 protocol=`batch-release-v1:focus:plan-reuse:interleave:correctness-first:3sess:p11`
profile_hash=`a78b86793c7142a0` table_sha256=`602782a3ad785fa0...` sessions=3 warmup=5 trials=30 seed=1234

Timing = api wall time (dispatch+alloc+layout+sync), ms; median of session medians over independent process sessions. Verdict vs the incumbent radix-2 folded control: >=1.02 speedup, <=0.98 regression, else parity (matches the 2% promotion gate). GM bytes in raw rows are analytical estimates (no hardware counters).

| label | N | B | blk | m0 loop | inc fold | m4 spr | m2 b | m5 b | selected | vendor | m2>=1.02? | m5>=1.02? | sel | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| N1024B16_dif | 1024 | 16 | 256 | 7.0379 | 1.8177 parity (1.000x) | 1.9409 regression (0.937x) | 0.9575 speedup (1.898x) | 0.9299 speedup (1.955x) | 0.8254 speedup (2.202x) | 0.2530 | speedup (1.898x) | speedup (1.955x) | speedup (2.202x) | True |
| N1024B16_dit | 1024 | 16 | 256 | 7.0439 | 1.6683 parity (1.000x) | 1.7857 regression (0.934x) | 0.8962 speedup (1.862x) | 0.9254 speedup (1.803x) | 0.9473 speedup (1.761x) | 0.2528 | speedup (1.862x) | speedup (1.803x) | speedup (1.761x) | True |
| N1024B64_dif | 1024 | 64 | 256 | 27.6283 | 5.8747 parity (1.000x) | 5.9665 parity (0.985x) | 2.8512 speedup (2.060x) | 2.7389 speedup (2.145x) | 2.2837 speedup (2.572x) | 0.2625 | speedup (2.060x) | speedup (2.145x) | speedup (2.572x) | True |
| N1024B64_dit | 1024 | 64 | 256 | 27.8239 | 6.0431 parity (1.000x) | 6.2617 regression (0.965x) | 2.8935 speedup (2.088x) | 2.9606 speedup (2.041x) | 2.9322 speedup (2.061x) | 0.2624 | speedup (2.088x) | speedup (2.041x) | speedup (2.061x) | True |
| N2048B16_dif | 2048 | 16 | 256 | 7.4787 | 3.4592 parity (1.000x) | 3.6860 regression (0.938x) | 1.8092 speedup (1.912x) | 1.8975 speedup (1.823x) | 1.5723 speedup (2.200x) | 0.6628 | speedup (1.912x) | speedup (1.823x) | speedup (2.200x) | True |
| N2048B16_dit | 2048 | 16 | 256 | 7.4740 | 3.5967 parity (1.000x) | 3.9205 regression (0.917x) | 1.7974 speedup (2.001x) | 2.0317 speedup (1.770x) | 0.3074 speedup (11.702x) | 0.6610 | speedup (2.001x) | speedup (1.770x) | speedup (11.702x) | True |
| N2048B64_dif | 2048 | 64 | 256 | 29.6055 | 11.7660 parity (1.000x) | 12.4153 regression (0.948x) | 6.1837 speedup (1.903x) | 6.8253 speedup (1.724x) | 5.1181 speedup (2.299x) | 0.6716 | speedup (1.903x) | speedup (1.724x) | speedup (2.299x) | True |
| N2048B64_dit | 2048 | 64 | 256 | 29.6260 | 12.2192 parity (1.000x) | 13.1551 regression (0.929x) | 5.9102 speedup (2.067x) | 7.6114 speedup (1.605x) | 5.9590 speedup (2.051x) | 0.6746 | speedup (2.067x) | speedup (1.605x) | speedup (2.051x) | True |

## P2 Td canary (representative shape N=1024 B=16)

| candidate | median ms | speedup vs inc | td | verdict |
|---|---:|---:|---|---|
| incumbent_m4_folded | 1.6683 | 1.000x | not-lowered | parity |
| m2_batch | 0.8962 | 1.862x | 1 | speedup |
| m2_packet_td1 | 0.9475 | 1.761x | 1 | speedup |
| m2_packet_td2 | 0.9480 | 1.760x | 2 | speedup |

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
| N1024B16_dif | 3.626 | 2.087 | 2.027 | 1.207 | - |
| N1024B16_dit | 3.945 | 1.930 | 1.993 | 1.207 | 0.945 |
| N1024B64_dif | 4.631 | 2.178 | 2.093 | 1.271 | - |
| N1024B64_dit | 4.443 | 2.115 | 2.164 | 1.283 | - |
| N2048B16_dif | 2.029 | 1.943 | 2.037 | 1.176 | - |
| N2048B16_dit | 1.906 | 1.930 | 2.181 | 1.187 | - |
| N2048B64_dif | 2.385 | 1.819 | 2.008 | 1.218 | - |
| N2048B64_dit | 2.252 | 1.728 | 2.226 | 1.177 | - |