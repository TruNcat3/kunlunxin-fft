# Selector validation sessions

commit `bbafd7fff78c` dirty=True device=kunlunxin-p800 sessions=3

| N | B | dir | executed | fallback | source | flow/realization | unit | speedup (m0/selected, median over sessions) | correct |
|---:|---:|---|---|---|---|---|---|---:|---|
| 16 | 1 | dit | fft_dit_fused | False | table | fused-inplace | radix2-scalar | 1.347x | True |
| 32 | 1 | dit | fft_dit | True | abstain | global-stage | radix2-scalar | 0.951x | True |
| 2048 | 1 | dif | fft_dif | True | abstain | global-stage | radix2-scalar | 1.615x | True |
| 2048 | 1 | dit | fft_dit_matrix | False | table | matrix-resident | dense-dot-fft64 | 1.971x | True |
| 4096 | 16 | dit | fft_dit_batch | False | table | global-stage | radix2-scalar | 0.996x | True |
| 4096 | 64 | dit | fft_dit_batch | False | table | global-stage | radix2-scalar | 0.998x | True |
| 65536 | 1 | dit | fft_dit_radix4 | False | table | global-stage | radix4-scalar | 2.150x | True |

## Ablation (N=4096, DIT)

| candidate | flow | unit | launches | median ms | speedup vs M0 | err |
|---|---|---|---:|---:|---:|---:|
| m0 | global-stage | radix2-scalar | 12 | 1.0760 | 1.000x | 3.6e-07 |
| m1_t64 | fused-inplace | radix2-scalar | 8 | 1.4888 | 0.723x | 4.2e-07 |
| m2_dit_t64 | matrix-resident | dense-dot-fft64 | 8 | 0.7438 | 1.447x | 9.7e-07 |
| m5_r4_dit | global-stage | radix4-scalar | 6 | 0.9296 | 1.158x | 1.7e-07 |

Speedups are session-level medians of interleaved M0/selected API wall times from independent process sessions; the 2% guard band is an engineering protection, not a significance claim. Raw per-trial rows: `raw_session*.jsonl`.
