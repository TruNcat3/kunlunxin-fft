# Kunlunxin-FFT 文档

Kunlunxin-FFT 是一个面向昆仑芯平台的 FFT 算子库，也是
[cuButterfly](https://github.com/TruNcat3/cuButterfly) 混合空间-时间数据流方法在 P800 上的
迁移实例。它继承方法层的阶段/数据展开、局部驻留和计算核心分层，但针对 Triton/XPU 工具链
重新实现 lowering、候选执行和硬件绑定选择。当前版本完成了可执行设备路径、M0--M5 候选、冻结
selector、plan reuse、batch 局部核心、手动硬件 probe、结构成本模型、留出验证、C2C
inverse/normalization、R2C/C2R compatibility path 和正式 release matrix；安装期一键校准与
跨 worker persistent pipeline 仍在路线图中。

## 最新验收状态

Issue #9 v6-1 已完成迁移验收：`41/41` release gate 通过，`469` 个 host 测试和 `370` 个
hardware 测试通过。v6 的 140 个可比较 cell 中，current/M0 几何平均延迟比为 `0.6086x`，即
相对旧 incumbent/M0 约降低 39%；10-cell paired closure 的中位改善约 `60.7%`，全部通过 2%
guard。这个结果证明 cuButterfly 的 A/P 分层、合法 lowering、环境绑定、选择器和可重放实验已在
P800 上形成迁移闭环。

这不等于已经追平厂商 FFT：在 106 个 vendor-supported、语义匹配的 cell 中，结果为 `4/0/102`
（win/tie/loss），cuButterfly/P800 相对 vendor 的几何平均约为 `0.122x`。当前版本尚未计入真正
的 batched M2/hybrid/h_dif、跨 worker persistent pipeline 或第三方局部 FFT core。完整定义、证据
链接和下一轮能力门禁见[Issue #9 v6-1](benchmarks/issue9-v61.md)。

<div class="project-affiliation">
  <div class="project-affiliation__logo">
    <a href="https://www.ustc.edu.cn/"><img src="assets/ustc-logo.png" alt="University of Science and Technology of China"></a>
  </div>
  <div class="project-affiliation__text">
    <strong>Teng Wang</strong><br>
    High Efficient Intelligent Computing Lab<br>
    Suzhou Institute for Advanced Research, USTC<br>
    <a href="mailto:wangt635@ustc.edu.cn">wangt635@ustc.edu.cn</a>
  </div>
</div>

## 一张图看当前结果

<figure class="doc-figure">
  <a href="assets/figures/release-overview.svg"><img src="assets/figures/release-overview.svg" alt="P800 release matrix: selected versus M0 and six primary matched vendor points"></a>
  <figcaption>正式 release matrix。左侧与同 block M0 比较；右侧为 N=1024/2048、B=1/16/64 的六个主厂商对比点。数值大于 1 表示 Kunlunxin-FFT 对应 selected path 更快。</figcaption>
</figure>

当前证据支持以下结论：

- selector 在单变换代表点相对 M0 获得 `1.23x--2.07x`；没有可靠收益时回退 M0。
- 厂商基线不是整体被超越：六个主 matched shape 中只有 `N=2048, B=1` 的 selected path 更快；`N=16/32, B=1` 另作为小尺寸补充点记录。
- 发布矩阵全部通过 NumPy complex128 参考检查；完整误差、设备和协议见[结果汇总](benchmarks/results.md)。
- 本项目证明方法能够迁移并产生可复现收益，不声称已达到 P800 的全局最优。

Issue #6 的历史语义矩阵包含 144 个 cell，其中 120 个 vendor-supported；selector 相对 vendor
为 `7/0/113`（win/tie/loss），几何平均 speedup `0.0986x`。这项快照证明语义迁移、正确性
门禁和 fail-closed 选择已贯通，同时明确暴露了兼容 real path 的性能缺口；它不能与上面的 release
matrix 混合平均。

### Issue #3 验收增量

本轮新增证据与正式 release 分层保存，不把实验性结果改写成发布 headline：

| 证据层 | 结果 | 状态 |
|---|---|---|
| Batch focus | 四个 `N=1024/2048 × B=16/64` shape 相对旧 incumbent 提升 `2.2x--2.7x` | 通过 guard 的项已冻结 |
| 模型留出 | top-3/full-space recall `6/6`；平均 top-1 regret `0.0816%` | 用于筛选，不替代实测 |
| 外部核心组合 | Issue #3 canary 的 `N=2048,B=16` 为 `2.87x` vendor；Issue #6 仅晋升一个 exact-shape composition 条目 | `N=1024` 两点为负；composition 不进通用 selector |

详见[实验设计与结果汇总](benchmarks/results.md)和[批量与模型验证](benchmarks/batch-focus.md)。

## 从哪里开始

| 目标 | 第一步 | 然后阅读 |
|---|---|---|
| 在 P800 上运行 | [安装与环境](getting-started/installation.md) | [首次运行](getting-started/quickstart.md)、[Plan 与选型](user-guide/plans.md) |
| 理解为什么这样设计 | [为什么混合展开](design/motivation.md) | [架构与数据流](design/architecture.md)、[计算核心](design/kernels.md)、[性能模型](design/performance-model.md) |
| 判断性能结论 | [实验设计与结果汇总](benchmarks/results.md) | [C2C](benchmarks/c2c.md)、[端到端](benchmarks/end-to-end.md)、[规模切片](benchmarks/scale.md) |
| 复现或扩展 | [原始数据与复现](benchmarks/reproducibility.md) | [增加设计点](development/adding-design-points.md)、[测试策略](testing.md)、[验证与发布](development/validation-and-release.md) |

最新证据的阅读顺序是：[v6-1 验收结论](benchmarks/issue9-v61.md) ->
[测量协议](benchmarks/methodology.md) -> [原始数据与复现](benchmarks/reproducibility.md)。

## 当前接口边界

公开 Python 入口接受连续 XPU tensor，支持 1D 二次幂长度、FP32/complex64 C2C forward/inverse，
以及首轮 R2C/C2R compatibility path；rank-1 单变换或 `(B, N)` batch 均有独立语义检查。DIT/DIF
都是 forward FFT 的分解。内部 planar lowering 不是用户输入格式。未覆盖能力见[支持范围](reference/support.md)
和[未来计划](roadmap.md)。

下一页：[安装](getting-started/installation.md)。