# 架构与数据流

## 三层而不是一条轴

<figure class="doc-figure doc-figure--wide">
  <a href="../../assets/figures/method-layers.svg"><img src="../../assets/figures/method-layers.svg" alt="Architecture mapping and processing-unit selection converge through a P800 lowering contract"></a>
  <figcaption>架构映射与计算核心是两个输入维度；只有通过 P800 lowering contract 的组合才成为可执行 plan。冻结 selector 在合法候选中选择，没有可信条目时回退 M0。</figcaption>
</figure>

这张图刻意避免把“两个正交轴”画成抽象坐标系。实际过程是：

1. **架构映射**组织 stage、packet、layout、边界和驻留；
2. **计算核心选型**决定局部 radix 或矩阵 FFT；
3. **硬件 lowering**检查 XPU 指令形态、对齐、shape、memory 和同步；
4. 合法候选经过正确性与测量后进入冻结表。

## 四个基本展开量

| 参数 | 含义 | 增大时的主要收益 | 主要代价 |
|---|---|---|---|
| `Us` | 阶段维度空间展开 | 更多 stage role/子图并行 | 交接、同步、驻留资源 |
| `Ts` | 阶段维度时间复用 | 减少阶段边界和启动 | live set、依赖链、局部容量 |
| `Ud` | 数据维度空间展开 | 更多独立 packet 并发 | worker/带宽竞争 |
| `Td` | 数据维度时间复用 | 摊销准备并形成稳态流 | 单 worker 工作量、尾部 |

`segmentation` 决定多少阶段属于一个执行组；`layout` 决定边界排列；`residence` 决定中间状态是否
留在寄存器/局部存储；`pipeline` 决定 packet/role 的交叠。它们共同约束可行性，不应把某个
`tile=64` 或“分三段”写成方法定义。

## N=16 的直观例子

四级 FFT 若按 M0 执行，每级都形成一个 launch 和 GM 边界。混合映射可以把前两级作为局部 group，
对多个 4 点 tile 时间复用；后两级再按更大配对距离执行。边界输出可同时完成必要置换，从而避免
单独 transpose。实际是否更快取决于：局部 group 是否降低 GM 流量、barrier 是否形成新瓶颈、
worker 数是否足够，以及所选核心是否适合该 tile。

## P800 上的 M0--M5

| ID | Flow realization | Processing unit | 主要改变 | 归因边界 |
|---|---|---|---|---|
| M0 | global-stage | radix2-scalar | 每阶段一个全局边界 | 稳定 incumbent/fallback |
| M1 | fused-inplace | radix2-scalar | 静态 tile 内融合多级 | 固定核心看 flow；barrier 可限制大 N |
| M2 | matrix-resident | dense-dot-fft64 | 一次读写的局部矩阵 FFT | 同时改变核心与驻留，不能归因于 flow 单项 |
| M3 | grouped-boundary | composite-linear-map | 将 tail stage 组织成 execution groups | 探索边界数量与 group 大小 |
| M4 | global-stage, spread/folded | radix2-scalar | Batch 映射为 Ud 或 Td | 比较数据空间/时间组织，不保证 batch 自动加速 |
| M5 | global-stage | radix4-scalar | 保持 global-stage flow 更换物理核心 | 较干净的固定 flow 核心比较 |

M0--M5 是当前 P800 可执行 envelope，不是 cuButterfly 理论空间的全部。特别是 M2 的收益包含
dense-dot 核心和 matrix-resident lowering；M5 更适合回答“固定流时更换核心会怎样”。

## 边界与执行闭包

候选元数据不等于实现。每个组合必须有 runner lowering，且 direction、batch、tile/block 和设备
profile 都满足合同。`plan_fft` 会在表命中后再次验证 runner 映射；无法执行的组合回退 M0，而非把
未消费参数静默忽略。

当前跨 role persistent 子图、显式 load/compute/store 重叠和自动新硬件校准仍未形成生产路径。
这些是[未来计划](../roadmap.md)，不是当前结果的隐藏前提。

下一页：[计算核心与 lowering](kernels.md)。
