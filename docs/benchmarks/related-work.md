# 相关工作与基线

Kunlunxin-FFT 的实验首先回答“cuButterfly 方法能否在 P800 上合法迁移并相对内部 M0 获益”，
其次才在语义匹配点比较厂商库。不同硬件或不同接口的数字不能直接组成排名。

## 方法来源与平台迁移

- [cuButterfly](https://github.com/TruNcat3/cuButterfly)：混合空间-时间蝶形数据流、架构映射与
  计算核心分层的方法来源；本项目应同时引用它。
- [Ascend-FFT](https://github.com/TruNcat3/ascend-fft)：同一方法在 Huawei Ascend 上的迁移
  实例，可用于对照文档、lowering 和实验组织，但不是 P800 性能基线；其发布数字见
  [结果页](https://truncat3.github.io/ascend-fft/benchmarks/results/)。

## 当前直接基线

- [PyTorch `torch.fft.fft`](https://docs.pytorch.org/docs/stable/generated/torch.fft.fft.html)：
  通过记录 xpytorch/xdnn 栈在同一 P800 上执行，是当前唯一进入 release matrix 的外部 baseline。
  只统计成功的同 shape、dtype 和 forward C2C 调用。
- M0 radix-2 global-stage：同仓库、同设备、同 shape/block 的内部 incumbent，用于回答新映射或
  核心是否带来净收益。

## 与 Ascend-FFT 迁移实例的成熟度

两个仓库已经在同一层次证明 cuButterfly 范式可以跨平台迁移：都把架构映射、计算核心和硬件
lowering 分层，并提供正确性门禁、硬件绑定选择和可复现实验。但“迁移证明同层次”不等于功能与
总体性能覆盖相同：

| 维度 | Kunlunxin-FFT | Ascend-FFT |
|---|---|---|
| 方法落地 | M0--M5、batch local core、外部 sub-transform 组合 | AIV/UB/MTE/GM 映射、K-plane、batch 折叠与边界融合 |
| 模型证据 | focus guard 100%；full-space top-3 `6/6`，平均 regret `0.0816%` | 结构计数、实测系数、候选合法性与 Plan 选择闭环 |
| 正确性/长度 | 研究网格 `2^3..2^20`，450 long rows；公共 forward C2C | C2C、R2C、C2R；公共 C2C 范围 64--4096，研究长度到 65536 |
| 外部性能 | 正式主匹配 6 点 1 胜；实验 A1P1 在 `2048×16` 为 `2.87x`，两个 1024 点为负 | C2C 49 点 `1.555x`（47/49 严格领先）；E2E `1.442x`；R2C/C2R 为 `1.180x/1.868x` |
| 工程接口 | Python one-shot/plan reuse；无稳定 C ABI/stream | C++ Plan/同步主机指针接口与应用/E2E 测试更完整；同样没有公开 device-pointer/用户 stream API |

因此当前严谨结论是：**Kunlunxin-FFT 已经达到完整“迁移实例”的研究闭环，证据纪律与 Ascend-FFT
接近；但算子语义、API、应用/E2E 矩阵和对厂商库的广泛领先仍明显少于 Ascend-FFT。**

## 参考但未直接比较的专业库

[NVIDIA cuFFT](https://docs.nvidia.com/cuda/cufft/)、
[cuFFTDx](https://docs.nvidia.com/cuda/cufftdx/)、
[VkFFT](https://github.com/DTolm/VkFFT) 和
[AMD rocFFT](https://github.com/ROCm/rocFFT) 展示了成熟 FFT 库的 API、plan、核心与多硬件工程
实践。它们运行在不同硬件/软件栈，本仓库没有把公开数字换算为 P800 speedup，也没有声称当前接口
完整度与这些产品库相同。

## 如何陈述结果

当前可支持的表述是：方法已在 P800 上形成合法实现、冻结选择和可复现的 M0 加速；正式 release
有一个 matched shape 超过记录厂商基线，Issue #3 的 experimental A1P1 又在 `N=2048,B=16`
形成稳定 `2.87x` 正证据，但两个 1024 点仍为负。不能表述为“全面超过专业库”或“P800 上唯一
可用 FFT”。

下一页：[批量与模型验证](batch-focus.md)。
