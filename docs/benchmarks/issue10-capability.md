# Issue #10：能力门禁与有界加速证据

本页记录 Issue #10 和 PR #12 的冻结结论。它回答两个问题：P800 的 Triton/XMLIR 路径能够合法
表达哪些 cuButterfly 架构映射，以及在严格匹配的范围内是否已经观察到相对厂商 FFT 和当前部署
路径的加速。它不替代 [Issue #9 v6-1](issue9-v61.md) 的全语义发布矩阵，两种协议不能拼接平均。

## 结论

| 项目 | 冻结结果 |
|---|---|
| Capability gate | 完成；PASS、FAIL、UNKNOWN 均保留原始证据 |
| same-buffer in-kernel permutation read | `not-lowered`；不同 buffer 和跨 launch 控制组正确 |
| 完整 resident/persistent pipeline | NO-GO，不作为近期发布阻塞项 |
| Restricted A-axis + vendor-P | GO/PARTIAL |
| Holdout | 16 cells × 5 paths × 2 sessions = 160 条，全部正确 |
| 最大相对误差 | `1.202284e-6` |
| Promotion candidates | `hybrid_fused_p` 5 cells；`hybrid_vendor_p` 3 cells |
| 默认 selector | 未修改；候选只达到 evidence gate |

这里的“双基线门槛”要求候选在每个独立 session 中，相对 vendor 和实际 profile-bound incumbent
的 bootstrap speedup 置信区间下界都大于 `1.02`。M0 和 Issue #9 历史标量只作描述，不参与门禁。

## 已验证的 fused exact cells

下表给出两个 session 的 point speedup；大于 1 表示 `hybrid_fused_p` 更快。

| Cell | 相对 vendor | 相对 profile-bound incumbent |
|---|---:|---:|
| `N=1024, B=1, inverse` | `1.581x / 1.567x` | `3.635x / 3.349x` |
| `N=2048, B=1, forward` | `2.465x / 2.311x` | `2.921x / 2.649x` |
| `N=2048, B=1, inverse` | `4.041x / 4.144x` | `3.434x / 3.743x` |
| `N=2048, B=16, forward` | `2.126x / 2.123x` | `1.447x / 1.423x` |
| `N=2048, B=16, inverse` | `2.087x / 2.008x` | `5.962x / 5.701x` |

这 5 个 cell 足以证明受限混合数据流迁移可以产生实际加速，但不足以声称整个 FFT 网格已经追平
厂商库。`B=64/256` 和其他未过门点完整保留；`hybrid_vendor_p` 的 3 个候选也只作为后续接线的
备选，不与 fused 结果重复计数。

## 为什么仍未进入默认路径

本轮最大跨 session 相对 spread 约为 `0.504`。虽然上述 exact cells 在两个 session 都通过当前
门槛，但生产接线还需要：

1. 修正 B=1 公共 contract 元数据，并补真实 rank-1 runner 回归；
2. 增加第三个独立 session，或定义明确的 spread guard；
3. 使用 exact key 绑定 `profile_hash + env_hash + evidence_sha256`；
4. 验证真实 `plan_fft -> execute_fft_plan` dispatch，未覆盖点继续走 incumbent/fallback；
5. 发布完整预声明矩阵，不只展示正点。

## 与 AscendFFT 的对齐边界

AscendFFT 的迁移展示按 C2C 主矩阵、E2E/应用、实数变换和长尺寸能力切片分别报告。Kunlunxin
后续沿用同一组织方式，但不会假设两个平台具有相同的 lowering 能力：

- 先在本页 16-cell holdout 范围完成生产接线和稳定性闭环；
- 再用预声明的 C2C 覆盖矩阵报告全部正负点，判断总体几何平均是否大于 1；
- 长尺寸只要求正确性和延迟切片，不强行要求本轮相对 vendor 加速；
- R2C/C2R 继续独立报告，不把 compatibility path 混入 C2C headline；
- resident/persistent、更多精度、非二次幂和跨设备留作未来工作。

## 证据入口

- [PR #12](https://github.com/TruNcat3/kunlunxin-fft/pull/12)：代码、审阅和合并记录；
- [Issue #10](https://github.com/TruNcat3/kunlunxin-fft/issues/10)：能力门禁目标与冻结结论；
- `results/p800/issue10_capability/wp3_holdout.json`：逐 cell、逐 session 原始 trials 与门禁；
- `results/p800/issue10_capability/selector_diff.json`：未接线证明和两张配置表的 provenance；
- `results/p800/issue10_capability/manifest.json` 与 `SHA256SUMS`：78 个校验条目的可复现打包；
- `results/p800/issue10_capability/repro_command.txt`：实验入口。

下一页：[Issue #9 v6-1 冻结基线](issue9-v61.md)。
