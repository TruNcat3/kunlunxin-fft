# 为什么混合展开

本页解释 Kunlunxin-FFT 的设计起点。Roofline、依赖对纯展开的限制，以及同时沿阶段和数据维度做
空间/时间映射的分析来自 [cuButterfly](https://github.com/TruNcat3/cuButterfly)；本仓库在 P800
上重新实现并验证其中一组合法映射。

## Roofline 给出的不是“算得更快”，而是“少搬、能流水”

一个 radix-2 FFT 有 `log2(N)` 个阶段。M0 每个阶段都从 GM 读取复数数据和 twiddle，再写回下一
阶段。仓库的解析账本给出每个阶段约 `12N` 读取字节和 `8N` 写入字节，因此总 GM 流量随
`20N log2(N)` 增长。算术量也按 `N logN` 增长，于是单纯增加长度不会自动提高 arithmetic
intensity：大问题仍可能沿着带宽 roof 运行。

Roofline 对本项目的直接启发有三点：

1. **减少物化边界**：多个阶段在寄存器/局部存储中连续完成，才能从分母中移除 GM 字节；
2. **隐藏必要搬运**：当边界不能消除时，用独立数据 packet 的流水覆盖 load/store 与算术；
3. **核心峰值不是充分条件**：更快的 radix 或矩阵核心若引入更多重排、同步或降低驻留度，端到端
   仍可能更慢。

因此我们不把工作限定为某一种蝶形公式。计算核心可以继续采用已有高性能实现，研究重点是计算流
如何让核心持续得到数据，并避免把每个阶段都暴露为片外边界。

## 为什么纯阶段展开不够

沿阶段做空间展开可以让不同 stage role 同时存在，但同一个数据 packet 必须按依赖顺序经过这些
role。若每一级都独立启动并写 GM，依赖被满足了，却付出 `O(logN)` 次边界流量和启动；若全部在一个
worker 内时间展开，live set、同步和寄存器/局部容量又会限制 occupancy。

换言之：

- 阶段空间并行 `Us` 增加角色并行，但需要合法交接；
- 阶段时间复用 `Ts` 减少边界，但增加局部容量与依赖链；
- 二者的最佳比例由硬件资源决定，不是固定 tile 常数。

## 为什么还要展开数据维度

不同 transform、batch item 或独立 tile 之间没有阶段依赖，可以作为同构 packet：

- 数据空间并行 `Ud` 用更多 worker 同时处理 packet；
- 数据时间复用 `Td` 让一个驻留 worker 连续遍历 packet，摊销准备和启动；
- 当一个 packet 等待边界或同步时，另一个 packet 可以占用计算/搬运通路。

大 batch 只增加可用 packet 数，并不自动提高单 packet 效率。硬件饱和后，吞吐仍由最慢 role、
边界流量、bank conflict 和核心效率决定。这也是为什么必须同时搜索长度、batch、布局、驻留和核心。

## 混合空间-时间数据流

<figure class="doc-figure doc-figure--wide">
  <a href="../../assets/figures/mixed-unfolding.svg"><img src="../../assets/figures/mixed-unfolding.svg" alt="Stage and data dimensions are each partitioned across space and time"></a>
  <figcaption>示例只用 6 个阶段和 4 个 tile 说明映射关系。生产参数不是固定的 2+2 分段或 8 点单元；模板根据硬件资源实例化 Us、Ts、Ud、Td、布局、驻留和计算核心。</figcaption>
</figure>

一个设计点可写成：

```text
a = (Us, Ts, Ud, Td, segmentation, layout, residence, pipeline, processing_unit)
```

它不是“时间复用一个子 kernel”的别名。完整目标是让局部子图在片上连续计算、让同构 packet 形成
稳态流水，并只在必要边界重排/物化。当前 P800 版本分别实现了局部融合、矩阵驻留、grouped tail、
batch spread/folded 和 radix-4 等切片；尚未把所有阶段角色实现成跨 worker persistent pipeline，
因此不能把当前结果称为方法理论上限。

## 流和核心为什么正交但耦合

架构映射回答“数据如何经过计算”，计算核心回答“局部块怎么算”。同一个流可以装 radix-2 或
radix-4；同一个核心也可放进不同分段与 layout。两者概念上正交，但会共同占用寄存器、局部存储、
指令和同步资源，所以必须经过硬件 lowering 后实测。

这正是本项目不把“发明新 FFT 核心”作为唯一目标的原因：成熟局部核心可以复用为设计空间选项，
而跨阶段/数据的系统编排规律才是需要抽象和迁移的部分。下一页用 P800 的 M0--M5 具体说明。

下一页：[架构与数据流](architecture.md)。
