# Batch release focus report (issue #3 P0 acceptance)

commit `ec19fd3e8a60` dirty=True device=kunlunxin-p800 driver=515.58 protocol=`batch-release-v1:focus:plan-reuse:interleave:correctness-first:3sess:p11`
profile_hash=`a78b86793c7142a0` table_sha256=`93be5746f70863c6...` sessions=3 warmup=5 trials=30 seed=1234

Timing = api wall time (dispatch+alloc+layout+sync), ms; median of session medians over independent process sessions. Verdict vs the incumbent radix-2 folded control: >=1.02 speedup, <=0.98 regression, else parity (matches the 2% promotion gate). GM bytes in raw rows are analytical estimates (no hardware counters).

| label | N | B | blk | m0 loop | inc fold | m4 spr | m2 b | m5 b | selected | vendor | m2>=1.02? | m5>=1.02? | sel | correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| N1024B16_dif | 1024 | 16 | 256 | 7.3240 | 1.6997 parity (1.000x) | 1.7591 regression (0.966x) | 0.8731 speedup (1.947x) | 0.9061 speedup (1.876x) | 0.7389 speedup (2.300x) | 0.2534 | speedup (1.947x) | speedup (1.876x) | speedup (2.300x) | True |
| N1024B16_dit | 1024 | 16 | 256 | 7.3560 | 1.7654 parity (1.000x) | 1.7542 parity (1.006x) | 0.8771 speedup (2.013x) | 0.9133 speedup (1.933x) | 0.7278 speedup (2.426x) | 0.2537 | speedup (2.013x) | speedup (1.933x) | speedup (2.426x) | True |
| N1024B64_dif | 1024 | 64 | 256 | 27.4790 | 5.9124 parity (1.000x) | 5.9538 parity (0.993x) | 2.9319 speedup (2.017x) | 2.6978 speedup (2.192x) | 2.3143 speedup (2.555x) | 0.2638 | speedup (2.017x) | speedup (2.192x) | speedup (2.555x) | True |
| N1024B64_dit | 1024 | 64 | 256 | 27.6508 | 6.0683 parity (1.000x) | 6.2669 regression (0.968x) | 2.8758 speedup (2.110x) | 3.0089 speedup (2.017x) | 2.2639 speedup (2.680x) | 0.2642 | speedup (2.110x) | speedup (2.017x) | speedup (2.680x) | True |
| N2048B16_dif | 2048 | 16 | 256 | 7.4182 | 3.4244 parity (1.000x) | 3.7267 regression (0.919x) | 1.8777 speedup (1.824x) | 1.8245 speedup (1.877x) | 1.5796 speedup (2.168x) | 0.6646 | speedup (1.824x) | speedup (1.877x) | speedup (2.168x) | True |
| N2048B16_dit | 2048 | 16 | 256 | 7.4143 | 3.4888 parity (1.000x) | 3.8820 regression (0.899x) | 1.9277 speedup (1.810x) | 1.9391 speedup (1.799x) | 1.5712 speedup (2.221x) | 0.6639 | speedup (1.810x) | speedup (1.799x) | speedup (2.221x) | True |
| N2048B64_dif | 2048 | 64 | 256 | 29.0156 | 11.9526 parity (1.000x) | 12.7757 regression (0.936x) | 5.8877 speedup (2.030x) | 7.3049 speedup (1.636x) | 5.0480 speedup (2.368x) | 0.6740 | speedup (2.030x) | speedup (1.636x) | speedup (2.368x) | True |
| N2048B64_dit | 2048 | 64 | 256 | 29.3200 | 12.3195 parity (1.000x) | 13.4382 regression (0.917x) | 6.0847 speedup (2.025x) | 7.9434 speedup (1.551x) | 5.0055 speedup (2.461x) | 0.6741 | speedup (2.025x) | speedup (1.551x) | speedup (2.461x) | True |

## P2 Td canary (representative shape N=1024 B=16)

| candidate | median ms | speedup vs inc | td | verdict |
|---|---:|---:|---|---|
| incumbent_m4_folded | 1.7654 | 1.000x | not-lowered | parity |
| m2_batch | 0.8771 | 2.013x | 1 | speedup |
| m2_packet_td1 | 0.9562 | 1.846x | 1 | speedup |
| m2_packet_td2 | 0.9556 | 1.847x | 2 | speedup |

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
| N1024B16_dif | 4.164 | 1.941 | 2.015 | 1.207 | - |
| N1024B16_dit | 4.193 | 1.921 | 2.000 | 1.237 | 0.918 |
| N1024B64_dif | 4.615 | 2.207 | 2.031 | 1.281 | - |
| N1024B64_dit | 4.412 | 2.083 | 2.179 | 1.288 | - |
| N2048B16_dif | 1.991 | 2.043 | 1.985 | 1.201 | - |
| N2048B16_dit | 1.910 | 2.002 | 2.014 | 1.240 | - |
| N2048B64_dif | 2.271 | 1.749 | 2.170 | 1.169 | - |
| N2048B64_dit | 2.182 | 1.692 | 2.209 | 1.224 | - |