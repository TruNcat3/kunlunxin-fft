# 相关工作与基线

Kunlunxin-FFT 的实验首先回答“cuButterfly 方法能否在 P800 上合法迁移并相对内部 M0 获益”，
其次才在语义匹配点比较厂商库。不同硬件或不同接口的数字不能直接组成排名。

## 方法来源与平台迁移

- [cuButterfly](https://github.com/TruNcat3/cuButterfly)：混合空间-时间蝶形数据流、架构映射与
  计算核心分层的方法来源；本项目应同时引用它。
- [Ascend-FFT](https://github.com/TruNcat3/ascend-fft)：同一方法在 Huawei Ascend 上的迁移
  实例，可用于对照文档、lowering 和实验组织，但不是 P800 性能基线。

## 当前直接基线

- [PyTorch `torch.fft.fft`](https://docs.pytorch.org/docs/stable/generated/torch.fft.fft.html)：
  通过记录 xpytorch/xdnn 栈在同一 P800 上执行，是当前唯一进入 release matrix 的外部 baseline。
  只统计成功的同 shape、dtype 和 forward C2C 调用。
- M0 radix-2 global-stage：同仓库、同设备、同 shape/block 的内部 incumbent，用于回答新映射或
  核心是否带来净收益。

## 参考但未直接比较的专业库

[NVIDIA cuFFT](https://docs.nvidia.com/cuda/cufft/)、
[cuFFTDx](https://docs.nvidia.com/cuda/cufftdx/)、
[VkFFT](https://github.com/DTolm/VkFFT) 和
[AMD rocFFT](https://github.com/ROCm/rocFFT) 展示了成熟 FFT 库的 API、plan、核心与多硬件工程
实践。它们运行在不同硬件/软件栈，本仓库没有把公开数字换算为 P800 speedup，也没有声称当前接口
完整度与这些产品库相同。

## 如何陈述结果

当前可支持的表述是：方法已在 P800 上形成合法实现、冻结选择和可复现的 M0 加速；一个正式 matched
shape 超过记录的厂商基线，其余尤其 batch 仍落后。不能表述为“全面超过专业库”或“P800 上唯一
可用 FFT”。

下一页：[工程结构](../development/repository-layout.md)。
