# 架构与硬件映射

Kunlunxin-FFT 继承 [cuButterfly](https://github.com/TruNcat3/cuButterfly) 的阶段/数据二维空间-
时间映射方法，并在 Kunlunxin P800 上重新实现物理 lowering。它不是 CUDA kernel 的直接移植：
阶段组织、数据所有权、局部驻留和计算核心先形成概念候选，再由 Triton/XPU 工具链和设备 profile
判定哪些组合能够执行。

本页说明当前 P800 的可执行 envelope。方法层的 Roofline、依赖、展开和正交性分析来自
cuButterfly；P800 的 kernel、编译器限制、选择表和测量结果属于本仓库。不要把本页的 P800 数值
外推到其他 Kunlunxin 设备。

## 从方法到执行

```text
FFT 依赖图 + N/batch/数值语义
                 ↓
阶段/数据空间展开与时间复用
                 + layout / residence / pipeline
                 ↓
局部 processing unit + P800 lowering contract
                 ↓
correctness gate + profile-bound measured selector
                 ↓
可执行 plan：runner、边界、回退与 provenance
```

架构映射回答“数据怎样经过阶段、packet 和边界”；计算核心回答“局部块怎样计算”；lowering
回答“该组合是否满足指针形式、布局、局部容量、同步和 launch 合同”。这三层必须分开记录，
但资源成本最终共同决定性能。

## 流编排与核心是两个集合

```text
A_i = (segmentation, Us, Ts, Ud, Td, layout, residence, pipeline)
P_j = (radix-2, radix-4, dense-dot, composite map, ...)
C_ij = (A_i, P_j)
legal_P800(C_ij) = lowering(C_ij, profile)
```

只有通过 lowering 的 `C_ij` 才能进入正确性门禁和实测排序。当前字段完整性不代表所有轴都有
生产 kernel：`Us/Ts` 仍未形成通用 lowering，`Ud/Td` 主要由 M4 batch 映射消费，跨组
`boundary_residence` 目前只允许 GM，`pipeline` 只允许 `none`。

这决定了消融口径：固定 radix-2 比较 M0/M1 是流组织实验；固定 global-stage 比较 M0/M5 是
核心实验；M2 同时改变 dense-dot 核心和局部驻留，只能报告为联合效果。

<figure class="doc-figure doc-figure--wide">
  <a href="../../assets/figures/method-layers.svg"><img src="../../assets/figures/method-layers.svg" alt="架构映射、计算核心和 P800 lowering contract 组成可执行 plan"></a>
  <figcaption>架构映射与计算核心是两个输入维度；只有通过 P800 lowering contract 的组合才进入正确性和测量闭环。</figcaption>
</figure>

## 框架对象不是六套独立算法

| 对象 | 职责 | 当前实现入口 |
|---|---|---|
| H / Hardware | 设备、工具链和局部资源约束 | `config/device.json`、`src/xpu/config.py` |
| A / Mapping | `Us/Ts/Ud/Td`、分段、数据轴和边界语义 | `src/xpu/candidates.py` |
| P / Processing unit | radix-2、radix-4、dense-dot、组合映射 | `src/xpu/kernels.py` |
| L / Layout | 公开 complex64 与内部 planar SoA | `src/xpu/validate.py`、runner |
| F / Flow realization | M0--M5 的 launch 和局部组织 | `src/xpu/runner.py` |
| Q / Evidence | 正确性、计时、profile hash 和选择原因 | `scripts/build_selector_table.py` |

配置或 dataclass 中出现的字段不等于已有独立 kernel。当前 `Us/Ts` 和跨组 residence 仍是
未 lowering 维度，必须以 runner 消费和正确性/性能证据为准。

## 当前 P800 数据通路

```text
complex64 输入
   → planar SoA lowering
   → M0/M1/M2/M3/M5 的局部阶段或核心
   → 必要的 GM group boundary
   → 自然顺序 complex64 输出
```

M2 在一个 local tile 内完成一次 load、局部 dense-dot 计算和一次 store；M3 将 tail stages
组合成 execution groups；M1 把早期 radix-2 stages 放进一个 fused kernel，但各级仍有指针
load/store 和 barrier；M5 保持 global-stage flow，只替换为 radix-4 核心。它们都是真实执行路径，
但都不是完整的跨 worker persistent 子图。

## M0--M5 可执行切片

| ID | Flow realization | Processing unit | 当前改变 | 归因边界 |
|---|---|---|---|---|
| M0 | `global-stage` | `radix2-scalar` | 每级一个 GM 边界 | 正确性 incumbent/fallback |
| M1 | `fused-inplace` | `radix2-scalar` | tile 内融合早期 stages | 固定核心看 flow；大 N 受 barrier 影响 |
| M2 | `matrix-resident` | `dense-dot-fft64` | 局部矩阵一次读写 | 核心与 residence 联合效果 |
| M3 | `grouped-boundary` | `composite-linear-map` | tail stages 组成 group | 减少部分 group boundary |
| M4 | `global-stage` | `radix2-scalar` | batch spread/folded | 数据轴组织，不是驻留 batch |
| M5 | `global-stage` | `radix4-scalar` | 更换局部核心 | 固定 flow 的 core A/B |

当前正式 release 中，M1/M2/M5 在单变换代表点相对同 block M0 达到约 `1.23x--2.07x`；batch
路径多数仍接近 M0。这是当前 lowering 覆盖的直接结果，不应被解释成混合数据流方法本身无效。
逐点数据见[实验结果](../benchmarks/results.md)。

## P800 lowering 约束

当前设备路径是 `Triton → XMLIR → XPU instructions`。已验证 pointer-form 的动态地址形式，
但记录版本不接受 tensor-subscript SSA gather；因此 M1 使用静态索引，M2 使用 dense linear map，
M3 使用合法的 execution group。这个限制描述的是工具链 lowering 形式，不是对所有 P800 硬件
能力的永久判断，详见[P800 设备路径](../development/device-stack.md)。

更换设备或软件栈时，必须重新验证设备、驱动、编译器、局部容量和候选合同。静态
`config/device.json` 的 hash 只防止旧表静默命中，并不代替硬件探测。

## P800 约束如何影响映射

以下是当前发布 profile 的事实，不是所有 Kunlunxin 型号的通用常量：

| 约束 | 对映射的影响 |
|---|---|
| 12 clusters、每 cluster 64 cores | grid 和 batch 展开必须检查 worker 波次 |
| 256 KiB shared/local capacity（profile 字段） | tile、矩阵和 group 的 live set 不能无限增大 |
| 设备代码路径为 Triton → XMLIR → XPU instructions | 不能用 nvcc/PTX 或 bare C++ host 代替设备路径 |
| `torch.device("cuda")` 是 xpytorch 兼容键 | 不表示 NVIDIA CUDA；context 必须由参考 runtime 建立 |
| tensor-subscript SSA gather 在记录工具链中失败 | M1 使用 pointer/static form，M2 使用 dense map，M3 使用合法 group |
| profile hash 与 selector 绑定 | 换驱动、工具链或设备必须重新验证和建表 |

这些限制来自当前[设备路径调查](../archive/device-path-investigation.md)和
[支持范围](../reference/support.md)。其中“某种 lowering 形式失败”不能推导为“硬件永远不能实现
该算法”。

## 当前边界

| 能力 | 状态 |
|---|---|
| 单 transform 的局部融合/矩阵驻留 | 已实现并实测 |
| radix-4 processing unit | 已实现并实测 |
| batch spread/folded | 已实现，但仍是 global-stage radix-2 |
| 跨组片上边界驻留 | 未 lowering，当前为 GM |
| 真正 `Td` worker token 遍历 | 未实现；folded 目前是索引组织 |
| producer/consumer persistent pipeline | 未实现 |
| 自动硬件 probe、参数化性能模型 | 路线图中 |

下一页：[计算核心与 lowering](kernels.md)；随后阅读[性能模型](performance-model.md)和
[实验设计与结果](../benchmarks/results.md)。早期实现快照保存在[历史归档](../archive/legacy-architecture.md)。
