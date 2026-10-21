# Issue #9 v6-1：迁移验收与性能边界

本页是当前 P800 迁移证据的最新入口。数据来自不可变 release
[`20261020-issue9-semantic-matrix-v6-1`](https://github.com/TruNcat3/kunlunxin-fft/tree/master/results/p800/releases/20261020-issue9-semantic-matrix-v6-1)，并与
[Issue #9 的最终复审](https://github.com/TruNcat3/kunlunxin-fft/issues/9)保持一致。
旧的 release matrix、Issue #3 和 Issue #6 快照仍保留作历史对照，但不能与本页的数字混合平均。

## 先看结论

Issue #9 的目标是证明 cuButterfly 的方法能够在 P800 上形成可复现的迁移实例，而不是在这一轮
一次性追平厂商 FFT。这个目标已经完成；厂商级物理性能目标仍未完成。

| 证据层 | 当前结果 | 应如何解读 |
|---|---:|---|
| 发布门禁 | `41/41 PASS` | 版本、manifest、artifact ledger 和路径身份闭合 |
| Host 验证 | `469 passed` | 解析、选择器、报告和 provenance 测试通过 |
| Hardware 验证 | `370 passed` | P800 上的 smoke、correctness、scale 和应用字段通过 |
| v6 可比较 cell | `140` | current/M0 的同语义比较集合 |
| current/M0 几何平均延迟比 | `0.6086x` | 相对旧 incumbent/M0，几何平均延迟约降低 `39%` |
| 10-cell paired closure | `37.0%--75.2%` | paired median 改善，中位数约 `60.7%`，全部通过 `2%` guard |
| vendor-supported 子集 | `106` | 仅统计厂商调用成功且语义匹配的 cell |
| 相对 vendor | `4 / 0 / 102` | win / tie / loss；几何平均 `cuButterfly/vendor ≈ 0.122x` |

这里的比值方向必须明确：`current/M0 < 1` 表示迁移实现比旧 incumbent 延迟更低；
`cuButterfly/vendor > 1` 才表示相对厂商基线更快。因而当前证据同时说明“相对旧实现有明显收益”和
“相对厂商 FFT 仍有物理性能差距”，两者并不矛盾。

## 迁移闭环实际覆盖了什么

v6-1 的发布门禁确认了以下链路可以重放：

```text
cuButterfly 方法分层
    -> A/P 候选与合法 lowering contract
    -> P800 环境/profile 绑定
    -> correctness-first selector
    -> 交错多 session 测量
    -> immutable release artifact
    -> 报告与回归门禁
```

其中架构轴 `A` 描述阶段/数据的空间-时间展开、分段、layout、residence 和 pipeline；处理轴 `P`
描述 radix、局部 FFT 或外部核心。候选必须满足：

```text
candidate = legal_lowering(A, P, device_profile)
```

因此当前发布结果证明的是方法迁移、选择闭包和实验可复现性。它不把尚未实现的 persistent
pipeline 或第三方局部核心算作已完成能力。

## 与厂商基线的边界

最新 v6 vendor-supported 子集为 106 个 cell：4 个胜出、0 个持平、102 个落后，
几何平均 `cuButterfly/vendor ≈ 0.122x`。这不是 selector 没有搜索到某个神奇参数的证据；Issue #9
明确记录了当前实现还没有加入：

- 真正的 batched `M2/hybrid/h_dif` 物理 kernel；
- 跨 worker 的 resident/persistent pipeline；
- 可替换的第三方或厂商局部 FFT processing unit；
- 非二次幂 batch 的合法 lowering。

因此下一轮的性能差距应归因到 **processing unit、GM 边界/dispatch 和 lowering 能力**，而不是把
当前 selector 结果描述成已经搜索完整架构空间。应用 shape proxy 也只用于验证流程，不能外推为
完整应用性能。

## 下一轮的最小可证伪计划

Issue #9 已关闭，后续工作转入独立的能力门禁与性能 issue（见
[Issue #10](https://github.com/TruNcat3/kunlunxin-fft/issues/10)）：

1. **先做 capability gate**：确认 Triton/XPU 工具链能否表达局部存储、静态重排、跨 role 同步、
   多 role residence、批内复用和可观测计时；不能表达的组合必须在候选生成阶段 abstain。
2. **固定一轴扫描另一轴**：固定同一个 P800 lowering 时扫描 `A`；固定一个合法 `A` 时扫描
   `P`，并保留 holdout，避免把联合收益误归因给某一轴。
3. **按瓶颈分类**：用 Roofline/trace 分开 `T_core`、global boundary、dispatch/plan、sync 和
   tail；只有实际观测到的 load/compute/store overlap 才能写入模型。
4. **先实现最小正闭环**：优先 batched local residence、边界/dispatch 减少和一个可测的高性能
   局部 FFT core；每个候选必须通过 correctness 和同语义 vendor 对照。
5. **使用稳定晋升门槛**：相对同 shape incumbent 至少 `2%` 且在独立 session/holdout 上稳定，
   才能写入 selector；负结果继续保留在 evidence archive。

### 推荐阅读

- [设计动机](../design/motivation.md)：Roofline、依赖和混合空间-时间展开的推导；
- [架构与数据流](../design/architecture.md)：A 轴的 stage/data mapping；
- [计算核心与 lowering](../design/kernels.md)：P 轴与合法 contract；
- [性能模型](../design/performance-model.md)：结构成本、筛选和 holdout；
- [Issue #10：FlagOS/Kunlunxin Triton 能力核验](https://github.com/TruNcat3/kunlunxin-fft/issues/10)。

## 证据与复现入口

- release manifest、环境和摘要：[v6-1 release directory](https://github.com/TruNcat3/kunlunxin-fft/tree/master/results/p800/releases/20261020-issue9-semantic-matrix-v6-1)；
- 路径身份与 selector 证据：`results/p800/issue9_v61/`；
- 复现命令和门禁：[`repro_command.txt`](https://github.com/TruNcat3/kunlunxin-fft/blob/master/results/p800/releases/20261020-issue9-semantic-matrix-v6-1/repro_command.txt)、[`gates_v61.txt`](https://github.com/TruNcat3/kunlunxin-fft/blob/master/results/p800/issue9_v61/release_evidence/gates_v61.txt)、[`host_tests.txt`](https://github.com/TruNcat3/kunlunxin-fft/blob/master/results/p800/issue9_v61/release_evidence/host_tests.txt) 和 [`hw_tests.txt`](https://github.com/TruNcat3/kunlunxin-fft/blob/master/results/p800/issue9_v61/release_evidence/hw_tests.txt)；
- 迁移方法来源：[cuButterfly](https://github.com/TruNcat3/cuButterfly)；本仓库独立贡献是 P800
  lowering、候选实现、profile 绑定和实测证据。

下一页：[测量协议](methodology.md)。
