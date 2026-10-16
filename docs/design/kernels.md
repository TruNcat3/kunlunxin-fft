# 计算核心与 lowering

计算核心是混合数据流设计空间中的可替换项，不是方法本身。一个核心只有满足当前 flow 的 shape、
布局、存储和同步合同，才是可运行候选。

## 当前核心

| 核心 | 当前实现 | 适用角色 | 备注 |
|---|---|---|---|
| `radix2-scalar` | M0/M1/M4 | global stage、fused tile、spread/folded batch | 基础正确性闭包 |
| `radix4-scalar` | M5 | 单变换与 batch global stage | 相邻阶段组合；奇数 logN 带 radix-2 tail |
| `dense-dot-fft64` | M2 | 单变换与 batch matrix-resident local tile | host 预计算矩阵，`tl.dot(..., input_precision="ieee")` |
| `composite-linear-map` | M3 | grouped tail | 多 stage group 的组合映射 |
| vendor sub-transform adapter | A1P1 experimental | 框架分解后的物化子块 | host API 组合；不是 device-resident 单核，不进入默认 selector |

## P800 lowering 的现实约束

当前设备路径经 Triton/XLIR。动态 pointer offset 的 `tl.load/tl.store` 已验证可用，但
`Xt[idx_tensor]` 的 tensor-subscript SSA gather 在记录版本中不能 lowering。M1 采用静态索引和
pointer form；M2 用 dense linear map 避开跨 stage gather，并使局部中间值隐式驻留。

这不表示硬件“不能融合 FFT”，也不表示任何 gather 都不可用；它只描述特定编译器版本的一种 SSA
形式。完整 canary 与排查记录见[P800 设备路径](../development/device-stack.md)。

## 如何正确归因

- 比较 M0 与 M5：flow 都是 global-stage，更接近纯核心 A/B；
- 比较 M0 与 M1：核心保持 radix-2，主要变化是局部融合与 barrier；
- M2 同时换核心和 residence，必须标为联合效果；
- M4 固定 radix-2，比较 batch 的 spread (`Ud`) 与 folded (`Td`)；
- selector 的最终最优值可以来自联合搜索，但论文消融不能把联合收益全部记到某一个轴。

当前结果显示 M1 在小长度能减少启动主导成本，却在大长度受 barrier/store-load 链影响；M5 在
`N=65536` 和 `1048576` 显著优于 M0；M2 在 `N=1024..4096` 的单变换 release 点胜出。这正说明
核心和流都需要成为参数，而不是固定选择。Issue #3 进一步验证 batch M2/M5 与 tile/rows 搜索：
focus winner 相对旧 M4 incumbent 提升 `2.2x--2.7x`。外部 adapter 只在 `N=2048,B=16`
形成稳定正向 canary，说明外部核心也能进入 `P` 轴，但仍需逐 shape guard。

下一页：[性能模型](performance-model.md)。
