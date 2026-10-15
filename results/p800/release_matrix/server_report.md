# Release matrix server report

commit `410d9ae9cdb3` dirty=False device=kunlunxin-p800 driver=515.58 protocol=`release-matrix-v2:plan-reuse:interleave:correctness-first:3sess`
profile_hash=`a78b86793c7142a0` table_sha256=`309483cca022b624…` warmup=3 trials=30 seed=20261015

| label | N | B | incumbent block | selected | sel block | flow | unit | fb | m0 ms | sel ms | speedup | vendor | correct |
|---|---:|---:|---:|---|---:|---|---|---|---:|---:|---:|---|---|
| N1024B1 | 1024 | 1 | 256 | fft_dit_matrix | 256 | matrix-resident | dense-dot-fft64 | False | 0.4497 | 0.3591 | 1.252x | ok (0.2448 ms) | True |
| N1024B16 | 1024 | 16 | 1024 | fft_dit_batch | 1024 | global-stage | radix2-scalar | True | 1.7736 | 1.7723 | 0.998x | ok (0.2494 ms) | True |
| N1024B64 | 1024 | 64 | 1024 | fft_dit_batch | 1024 | global-stage | radix2-scalar | True | 6.1552 | 6.1717 | 0.999x | ok (0.2569 ms) | True |
| N1048576B1 | 1048576 | 1 | 256 | fft_dit_radix4 | 256 | global-stage | radix4-scalar | False | 160.6676 | 86.0158 | 1.868x | n/a (N>2048) | True |
| N16B1 | 16 | 1 | 1024 | fft_dit_fused | 1024 | fused-inplace | radix2-scalar | False | 0.3262 | 0.2181 | 1.494x | ok (0.0331 ms) | True |
| N2048B1 | 2048 | 1 | 256 | fft_dit_matrix | 256 | matrix-resident | dense-dot-fft64 | False | 0.4769 | 0.3872 | 1.232x | ok (0.6426 ms) | True |
| N2048B16 | 2048 | 16 | 1024 | fft_dit_batch | 1024 | global-stage | radix2-scalar | True | 3.5845 | 3.6048 | 1.001x | ok (0.6595 ms) | True |
| N2048B64 | 2048 | 64 | 1024 | fft_dit_batch | 1024 | global-stage | radix2-scalar | True | 13.1054 | 13.1717 | 0.996x | ok (0.6728 ms) | True |
| N32B1 | 32 | 1 | 1024 | fft_dit | 1024 | global-stage | radix2-scalar | True | 0.3712 | 0.3687 | 0.997x | ok (0.0205 ms) | True |
| N4096B1 | 4096 | 1 | 256 | fft_dit_matrix | 256 | matrix-resident | dense-dot-fft64 | False | 0.6139 | 0.4694 | 1.308x | n/a (N>2048) | True |
| N4096B16 | 4096 | 16 | 1024 | fft_dit_batch | 1024 | global-stage | radix2-scalar | False | 7.1682 | 7.1564 | 1.002x | n/a (N>2048) | True |
| N4096B64 | 4096 | 64 | 1024 | fft_dit_batch | 1024 | global-stage | radix2-scalar | False | 27.4772 | 27.5797 | 0.998x | n/a (N>2048) | True |
| N65536B1 | 65536 | 1 | 256 | fft_dit_radix4 | 256 | global-stage | radix4-scalar | False | 9.2029 | 4.4455 | 2.072x | n/a (N>2048) | True |
| N65536B4 | 65536 | 4 | 1024 | fft_dit_batch | 1024 | global-stage | radix2-scalar | False | 35.1713 | 35.1632 | 1.001x | n/a (N>2048) | True |
| fallback-profile-mismatch | 2048 | 1 | 256 | fft_dit | 256 | global-stage | radix2-scalar | True | 0.4768 | 0.4781 | 0.990x | ok (0.6442 ms) | True |

## Plan overhead (one-shot run_fft vs plan-reuse)

| N | plan-reuse ms | one-shot ms | overhead ms | % |
|---:|---:|---:|---:|---:|
| 16 | 0.2015 | 0.2173 | 0.0158 | 7.9% |
| 32 | 0.3605 | 0.3726 | 0.0121 | 3.4% |

The 2% guard applies to plan-reuse `selected` vs the same-block `m0_incumbent`; one-shot plan overhead is a separate convenience-API column, never a kernel regression. Vendor rows enter speedups only when `torch.fft.fft` executed the identical (B, N) fp32 forward C2C tensor successfully; failures stay as `unsupported + error`. Speedups, plan-overhead, and ablation columns are session-level medians over 3 independent process sessions, rebuilt offline from summary_session*.json (raw rows keep every trial).
