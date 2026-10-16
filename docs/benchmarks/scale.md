# 规模切片

Phase 3 扫描覆盖 `N=2^3..2^20` 的 18 个二次幂长度，并包含 M0/M1/M2/M3/M5 以及 M4 的
`B=1/4/16/64` spread/folded。该矩阵用于观察交叉点、正确性和候选偏好，不替代正式 release
快照：Phase 3 的 commit、block 和计时协议与 release 不同。

## 已观察趋势

| 路径 | 研究矩阵趋势 | 解释 |
|---|---|---|
| M1 radix-2 fused | `N<=1024` 约 `1.62x`，`N>=4096` 约 `0.67x` | barrier 与 store-load 依赖在长尺寸占主导 |
| M2 dense-dot | 全范围约 `1.64x` | 核心与 residence 的联合效果 |
| M3 grouped | 全范围约 `1.56x` | tail grouping 减少部分边界 |
| M5 radix-4 | `N>=4096` DIT 约 `1.69x` | 固定 flow 下核心收益随长度放大 |

450 条 long rows 通过当前误差门禁；最坏误差和完整原始数据见 `results/p800` 归档。没有同语义
外部 baseline 的尺寸只证明可运行和正确性，不能计入 vendor speedup。

## 规模实验的边界

大长度不自动意味着更高加速：必要 GM 边界、同步、局部容量、波次和 processing unit 成本仍会
随 shape 改变。大 batch 也只增加同构 packet，不能自动把 M4 的 global-stage radix-2 变成驻留
batch pipeline。关闭的 [Issue #3](https://github.com/TruNcat3/kunlunxin-fft/issues/3) 已在
`N=1024/2048, B=16/64` 上接入 batch-aware M2/M5/local-fused 候选，相对旧 incumbent 获得
`2.2x--2.7x`；这验证了 batch 映射可以继续优化，但并不等同于完整跨 worker persistent 流水。

下一页：[消融与交叉点](ablations.md)。
