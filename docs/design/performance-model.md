# 性能模型与选择证据

## 与冻结基线和最新证据的关系

Issue #9 v6-1 的 140-cell current/M0 几何平均延迟比为 `0.6086x`，说明模型筛选和已实现的
lowering 能够产生稳定的迁移收益；这不是模型对 vendor 的性能承诺。相同快照中 106 个
vendor-supported cell 的 current/vendor 几何平均约 `0.122x`，说明 processing unit、局部存储、
GM 边界和 dispatch 能力必须作为显式成本项，而不能只拟合 shape。

后续 Issue #10 capability gate 已将 same-buffer in-kernel permutation read 判为 `not-lowered`，
将完整 resident/persistent pipeline 判为 NO-GO，同时验证 restricted A-axis + vendor-P 为
GO/PARTIAL，并得到 5 个 fused exact cell 的双基线正证据。模型可以纳入这些已观测候选的成本，
但不能为未实现的 persistent overlap 添加假设收益；只有 trace/计时观测到的重叠才能进入方程。

本页回答“为什么某个候选在某个 P800 shape 上更快”，而不是把冻结 selector 重新描述成模型。
部署选择仍是 profile-bound 的实测精确表；结构化成本模型、probe 校准与留出验证**已实现**
（`src/xpu/cost_model.py`、`scripts/hw_probe.py`、`scripts/validate_cost_model.py`），模型只做
screening，永不覆盖表。性能模型的分析框架来自 [cuButterfly](https://github.com/TruNcat3/cuButterfly)，并参考
[Ascend-FFT 的模型组织](https://github.com/TruNcat3/ascend-fft/blob/master/docs/design/performance-model.md)。

## 当前选择闭环

```text
候选生成
   → lowering 合法性
   → correctness gate
   → 同 shape/block 的 M0 incumbent
   → 多次交错测量
   → 2% 工程晋升门限
   → profile-bound selector table
   → runtime exact-key lookup 或 M0 fallback
```

`config/selector_table.json` 是已确认点的冻结表，不外推到未测长度、batch 或硬件。它保证部署
闭包和 provenance，但不能回答表外 shape 的最优参数，也不能替代硬件计数器。

## Roofline 到结构账本

对 radix-2 FFT，若每个 stage 都独立读写 GM，则边界字节随 `N log2(N)` 增长。当前可可靠获得的
结构量是：

| 结构量 | 含义 | 当前来源 |
|---|---|---|
| `launch_count` | kernel/阶段发射次数 | runner 逻辑推导 |
| `materialized_boundaries` | 必须跨 GM 的 stage/group 边界 | flow realization |
| `tile/group` | 局部融合和 tail 分组规模 | candidate 参数 |
| `processing_unit` | radix-2/radix-4/dense-dot/composite | lowering 描述 |
| `batch_variant` | M4 spread 或 folded | batch runner |
| `sync_count` | barrier 或显式交接次数 | kernel/runner 记录 |

没有等价 NCU 的通用 XPU counter 时，GM bytes、带宽和占用率只能标为**解析估算**，不能写成硬件
实测。性能模型的第一版应解释以下成本：

```text
T ≈ T_dispatch
  + T_global_boundary(bytes, layout)
  + T_core(processing_unit, stages, points)
  + T_sync(barriers, handoff)
  + T_wave(tail, occupancy proxy)
```

只有已经实现并被 trace/计时证明的 load/compute/store 重叠，才允许加入 overlap 项；不能为了
拟合数据引入未实现的流水收益。

## 当前证据怎样归因

- M0 vs M1：同为 radix-2，主要用于观察局部阶段融合、启动和 barrier 变化。
- M0 vs M5：同为 global-stage，主要用于观察 radix-4 processing unit 变化。
- M2：dense-dot 核心与 matrix-resident lowering 同时变化，只能报告联合收益。
- M3：tail grouping 改变边界数量和局部线性映射，属于分组/核心联合路径。
- M4：固定 radix-2，只比较 batch 的 spread/folded 数据轴组织。

因此正式结果中的 `1.23x--2.07x` 是候选路径相对 M0 的实测收益，不是单独由某一个抽象轴贡献的
定量证明。完整消融见[消融与交叉点](../benchmarks/ablations.md)。

## 模型闭环（已实现）

按少量典型变量校准，而不是为每个 shape 单独拟合（Issue #3 轮次已落地）：

1. `cost_model.axes()` 从真实 lowering 生成 execution groups、GM 边界、核心操作量和同步计数；
2. 固定开销和硬件锚点来自 `config/calibration.json`，核心效率常数显式声明在
   `src/xpu/cost_model.py`，**从不**对验证 run 回拟合；
3. `bounded_search` 用模型筛选 top-K，再以 correctness-first 的实测决定 winner；
4. 留出验证已跑：focus 轮的 2% guard 晋升决策一致率、full-space holdout 的 top-3 命中与选择
   regret 见 [batch-focus 报告](../benchmarks/batch-focus.md)与
   `results/p800/model_validation.json`（focus 与 holdout 统计隔离、holdout-only 口径）；
5. M0 闭包与环境绑定同时生效：profile/环境指纹缺失或失配时拒绝静默复用旧表。

模型不声称绝对时间精度（MDAPE 结构性偏高：只有 launch + 带宽 + 峰值 flops 三段账本，无硬件
计数器）；它的声明范围是 guard 决策一致性与 bounded-search 的低 regret。Issue #3 已完成并
关闭，其验收数据见[批量与模型验证](../benchmarks/batch-focus.md)；未完成能力见
[未来计划](../roadmap.md)。

下一页：[实数变换边界](real-transforms.md)。
