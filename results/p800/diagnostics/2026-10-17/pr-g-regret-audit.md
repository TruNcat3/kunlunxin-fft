# PR-G step 1: v2 regret audit

- evidence: summary-derived from `results/p800/releases/20261017-wp2-semantic-matrix-v2`
- protocol: semantic-matrix-v1:plan-reuse:interleave:correctness-first
- cells: 128 (deduplicated)

## Gates on the frozen v2 baseline

- main-domain regret <= 10%: PASS 
- stress N1024B256 (must fix): OPEN
  - N1024B256:c2c-forward: regret 1.18 (2.174x oracle vs incumbent, source=abstain)
  - N1024B256:c2c-inverse: regret 1.17 (2.161x oracle vs incumbent, source=abstain)
  - N1024B256:c2r: regret 1.03 (2.029x oracle vs incumbent, source=abstain)
  - N1024B256:r2c: regret 1.08 (2.081x oracle vs incumbent, source=abstain)

## PR-G step 2: fixed-domain abstain points whose measured oracle is clearly faster (regret >= 0.10, i.e. oracle >= 1.10x the incumbent fallback)

- N64B1:c2c-forward: regret 2.08, oracle oracle:m2_dif_t64, 3.078x vs incumbent
- N64B1:c2c-inverse: regret 0.82, oracle oracle:m2_dif_t64, 1.825x vs incumbent
- N64B1:c2r: regret 0.38, oracle oracle:m2_dit_t64, 1.379x vs incumbent
- N64B1:r2c: regret 1.67, oracle oracle:m2_dit_t64, 2.67x vs incumbent
- N512B64:c2c-forward: regret 0.72, oracle oracle:m2_dif_t64, 1.722x vs incumbent
- N512B64:c2c-inverse: regret 0.66, oracle oracle:m2_dif_t64, 1.657x vs incumbent
- N512B64:c2r: regret 0.58, oracle oracle:m2_dit_t64, 1.582x vs incumbent
- N512B64:r2c: regret 0.69, oracle oracle:m2_dit_t64, 1.691x vs incumbent
- N1024B256:c2c-forward: regret 1.18, oracle oracle:m2_dit_t64, 2.174x vs incumbent
- N1024B256:c2c-inverse: regret 1.17, oracle oracle:m2_dit_t64, 2.161x vs incumbent
- N1024B256:c2r: regret 1.03, oracle oracle:m2_dit_t64, 2.029x vs incumbent
- N1024B256:r2c: regret 1.08, oracle oracle:m2_dit_t64, 2.081x vs incumbent
- N2048B4:c2c-forward: regret 0.75, oracle oracle:m2_dif_t64, 1.747x vs incumbent
- N2048B4:c2c-inverse: regret 0.68, oracle oracle:m2_dif_t64, 1.685x vs incumbent
- N2048B4:c2r: regret 0.44, oracle oracle:m2_dif_t64, 1.44x vs incumbent
- N2048B4:r2c: regret 0.68, oracle oracle:m2_dif_t64, 1.68x vs incumbent
- N16384B1:c2c-forward: regret 0.59, oracle oracle:m2_dif_t64, 1.596x vs incumbent
- N16384B1:c2c-inverse: regret 0.54, oracle oracle:m2_dit_t64, 1.536x vs incumbent
- N16384B1:c2r: regret 0.49, oracle oracle:m2_dif_t64, 1.494x vs incumbent
- N16384B1:r2c: regret 0.58, oracle oracle:m2_dif_t64, 1.584x vs incumbent
- N16384B16:c2c-forward: regret 0.88, oracle oracle:m5_r4_dit, 1.881x vs incumbent
- N16384B16:c2c-inverse: regret 0.88, oracle oracle:m5_r4_dit, 1.878x vs incumbent
- N16384B16:c2r: regret 0.80, oracle oracle:m5_r4_dit, 1.806x vs incumbent
- N16384B16:r2c: regret 0.84, oracle oracle:m5_r4_dit, 1.84x vs incumbent

## Guard-positive production points (>=2%) per semantic

- c2c-forward: 9 point(s): N1024B1:c2c-forward +0.213, N1024B16:c2c-forward +0.598, N1024B64:c2c-forward +0.635, N2048B1:c2c-forward +0.199
- c2c-inverse: 8 point(s): N1024B1:c2c-inverse +0.159, N1024B16:c2c-inverse +0.594, N1024B64:c2c-inverse +0.630, N2048B1:c2c-inverse +0.145
- r2c: 8 point(s): N1024B1:r2c +0.188, N1024B16:r2c +0.572, N1024B64:r2c +0.631, N2048B1:r2c +0.183
- c2r: 9 point(s): N1024B1:c2r +0.100, N1024B16:c2r +0.422, N1024B64:c2r +0.567, N2048B1:c2r +0.097
