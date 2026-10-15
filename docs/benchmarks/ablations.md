# 消融与交叉点

消融的目标不是寻找一条在所有尺寸都快的固定配置，而是解释为什么配置偏好随长度、batch 和硬件
资源变化。

## 固定 radix-2 看流组织：M1 vs M0

M1 将早期 stage 融合在静态 tile 中。小长度由 launch 主导，减少 launch/boundary 得到
`1.62x` 几何均值；大长度中每级 `tl.debug_barrier()` 和 in-place store-load 依赖限制流水，降至
`0.67x`。这是一条重要负结果：阶段时间复用只有在节省的边界大于新增同步时才有效。

## 固定 global-stage 看核心：M5 vs M0

M5 保持全局阶段 flow，将相邻 radix-2 组合为 radix-4。Phase 3 的 DIT 结果从小长度约 `1.03x`
过渡到大长度约 `1.69x`；正式 release 的 `N=65536/1048576` 为 `2.072x/1.868x`。这说明
物理核心必须进入 selector，不能从一个长度的结论固定整个空间。

## 联合候选：M2 与 M3

M2 的 dense-dot FFT64 与 matrix-resident lowering 同时变化，所以 `1.64x` 全范围几何均值不能
全部归因于混合数据流。M3 把 tail 组成 execution groups，减少部分边界；它用于探索 grouping，
但当前 release 表会在更合适的长度选择 M2 或 M5。

## Batch：Ud 与 Td

<figure class="doc-figure doc-figure--wide">
  <a href="../../assets/figures/phase3-batch-throughput.png"><img src="../../assets/figures/phase3-batch-throughput.png" alt="Phase 3 batch throughput for spread and folded variants"></a>
  <figcaption>M4 将 batch 映射为 spread（数据空间展开 Ud）或 folded（数据时间展开 Td）。图为 Phase 3 协议，不能与正式 release latency 直接合并。</figcaption>
</figure>

大 batch 提供更多同构 packet，但不会消除每个 packet 的阶段 GM 流量。当前 release 中 batch
selector 多数回退或与 M0 持平，并明显落后 matched vendor。这说明数据维度模板已经存在，但高效
batched local core / residence 尚未接入，不应把“负载更多”直接等价为“相对专业库更快”。

## Block 也是硬件参数

<figure class="doc-figure doc-figure--wide">
  <a href="../../assets/figures/phase3-block-sweep.png"><img src="../../assets/figures/phase3-block-sweep.png" alt="M0 and M5 latency across block sizes"></a>
  <figcaption>不同 processing unit 对 block 的偏好不同。release selector 保存 incumbent 与 selected block，避免只比较某个候选的有利 block。</figcaption>
</figure>

这些交叉点支持项目的核心主张：范式不固定 tile、分段数量或核心，而是固定参数化方法和合法性
合同，由每台硬件的 profile/测量决定设计点。

下一页：[原始数据与复现](reproducibility.md)。
