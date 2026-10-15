# Kunlunxin-FFT

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Citation](https://img.shields.io/badge/citation-CFF-orange.svg)](CITATION.cff)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-2F80ED.svg)](https://truncat3.github.io/kunlunxin-fft/)
[![Hardware](https://img.shields.io/badge/tested-Kunlunxin%20P800-8A2BE2.svg)](docs/reference/support.md)

**Kunlunxin-FFT 是一个面向昆仑芯平台的 FFT 算子库，也是
[cuButterfly](https://github.com/TruNcat3/cuButterfly) 混合空间-时间数据流方法的迁移实例。**
它把阶段/数据展开、局部驻留、边界组织和计算核心选择拆成可组合的设计轴，再根据硬件 profile
选择可执行实现；当前 P800 版本提供 forward C2C、单变换与 batch 路径、M0--M5 候选、冻结
selector 和保守 fallback。

<p align="center">
  <a href="https://www.ustc.edu.cn/">
    <img src="docs/assets/ustc-logo.png" width="520" alt="University of Science and Technology of China">
  </a>
</p>

## 当前结果

![P800 release overview](docs/assets/figures/release-overview.png)

图中左侧是冻结 selector 相对同 block M0 的加速，右侧是与当前记录的 `torch.fft.fft`
同语义点对比。正式 P800 release matrix 的结论是：

- 单变换代表点相对 M0 达到 `1.23x--2.07x`；`N=32` 和若干 batch 点由 fallback 保持约持平。
- 在六个主厂商对比点中，`N=2048, B=1` 达到 `1.66x`；其余主矩阵点仍落后。发布数据还记录了 `N=16/32, B=1` 两个小尺寸补充点。
- 全部正式发布点通过正确性检查，selected/M0 最坏相对误差为 `9.91e-7`。
- 这些数字来自单张 P800 OAM、FP32 complex64、forward C2C、API wall time、3 个独立进程
  session；不代表其他软件栈、硬件或未测语义。

完整逐点数据、负点和计时边界见[实验结果](docs/benchmarks/results.md)与
[测量协议](docs/benchmarks/methodology.md)。按实验目的继续阅读[C2C 性能](docs/benchmarks/c2c.md)、
[端到端口径](docs/benchmarks/end-to-end.md)和[规模切片](docs/benchmarks/scale.md)。

## 快速开始

需要厂商 xpytorch 环境（`torch + torch_xmlir + Triton XPU backend`）。脚本检查现有环境，
不会安装驱动或自动生成新硬件模型。

```bash
git clone https://github.com/TruNcat3/kunlunxin-fft.git
cd kunlunxin-fft

export CUDA_VISIBLE_DEVICES=6  # 改为当前可用的 P800 设备号
bash scripts/init.sh
bash scripts/build.sh
python3 scripts/smoke.py
python3 scripts/test_correctness.py --profile smoke
```

最小 plan-reuse 调用：

```python
import torch
import torch_xmlir  # 激活参考环境中的 XPU runtime
from src.xpu import DeviceProfile, execute_fft_plan, plan_fft

x = torch.randn(2048, device="cuda", dtype=torch.complex64)
profile_hash = DeviceProfile.load().hash()
plan = plan_fft("dit", 2048, profile_hash=profile_hash)
y = execute_fft_plan(plan, x)
print(plan.executed, plan.fallback, plan.reason)
```

这里的 `"cuda"` 是该 xpytorch 栈暴露 P800 的设备键，不表示代码运行在 NVIDIA GPU 上。
安装细节见[安装](docs/getting-started/installation.md)，接口语义见
[Plan 与选型](docs/user-guide/plans.md)。

## 为什么这样设计

FFT 的算术量固定，但硬件时间不只由蝶形 FLOPs 决定。纯阶段并行会反复物化阶段边界，纯数据
并行又会受局部容量和依赖链限制。cuButterfly 的核心观察是：同时在阶段维度和数据维度做空间/
时间划分，用局部驻留与边界重排降低数据搬运，再把 radix 或矩阵小 FFT 作为可替换计算核心。

Kunlunxin-FFT 保留这套方法层抽象，但所有 kernel、布局和 lowering 都针对 P800 重新实现：

```text
架构映射 A (Us, Ts, Ud, Td, layout, residence, pipeline)
        x 计算核心 P (radix-2, radix-4, dense-dot FFT64, ...)
        -> P800 lowering contract 过滤
        -> profile-bound selector
        -> 可执行 plan 或 M0 fallback
```

因此计算流优化与计算核心优化在概念上正交、在资源上耦合，并不互斥。详细推导见
[设计动机](docs/design/motivation.md)和[架构与数据流](docs/design/architecture.md)。本观点和
方法论来源于 cuButterfly；本仓库的独立内容是 P800 lowering、实现、选择器和实测证据。

### 当前完成度

| 层次 | 当前状态 |
|---|---|
| 方法抽象与合法 lowering | 已迁移并实测 M0--M5 |
| 单变换局部融合/核心选择 | 已完成首轮优化，代表点相对 M0 为 `1.23x--2.07x` |
| Batch 高效局部核心 | 尚未接入 M2/M5；当前主要是 M4 global-stage radix-2 |
| 自动硬件 profile/成本模型 | 路线图中，当前使用静态 profile 与冻结表 |
| 跨 worker persistent pipeline | 路线图中，不是当前发布路径 |

新增实现应先阅读[增加设计点](docs/development/adding-design-points.md)和[测试策略](docs/testing.md)，
再更新 selector 与发布证据。当前收敛任务集中在
[Issue #3](https://github.com/TruNcat3/kunlunxin-fft/issues/3)。

## 阅读顺序

| 读者 | 建议路径 |
|---|---|
| 首次使用 | [安装](docs/getting-started/installation.md) -> [首次运行](docs/getting-started/quickstart.md) -> [支持范围](docs/reference/support.md) |
| 了解方法 | [设计动机](docs/design/motivation.md) -> [架构与数据流](docs/design/architecture.md) -> [计算核心](docs/design/kernels.md) -> [性能模型](docs/design/performance-model.md) |
| 阅读实验 | [结果汇总](docs/benchmarks/results.md) -> [测量协议](docs/benchmarks/methodology.md) -> [C2C](docs/benchmarks/c2c.md) -> [端到端](docs/benchmarks/end-to-end.md) -> [规模切片](docs/benchmarks/scale.md) |
| 复现/开发 | [数据与复现](docs/benchmarks/reproducibility.md) -> [增加设计点](docs/development/adding-design-points.md) -> [测试策略](docs/testing.md) -> [验证与发布](docs/development/validation-and-release.md) |

在线文档：<https://truncat3.github.io/kunlunxin-fft/>。

## 支持边界

当前已验证接口是 1D、二次幂长度、FP32/complex64、forward C2C。DIT 与 DIF 是同一 forward
FFT 的两种分解，不是正/逆变换。inverse、R2C/C2R、非二次幂、多维、其他精度、稳定 C ABI 和
用户 stream 接口仍在路线图中。内部 kernel 使用 planar 实虚分离 lowering，但公开 Python 输入
是 `torch.complex64` tensor。

## 引用与许可

软件采用 [Apache-2.0](LICENSE)，允许修改和再发布，并须遵守许可证中的通知要求。引用本实现和
P800 实验时请引用 Kunlunxin-FFT；引用混合空间-时间数据流方法时请同时引用 cuButterfly。机器可读
信息见 [CITATION.cff](CITATION.cff)，作者与机构信息见[项目与作者](docs/about.md)。

```bibtex
@software{wang2026kunlunxinfft,
  author       = {Teng Wang},
  title        = {Kunlunxin-FFT: A Hardware-Mapped FFT Operator Library for Kunlunxin P800},
  year         = {2026},
  url          = {https://github.com/TruNcat3/kunlunxin-fft},
  organization = {High Efficient Intelligent Computing Lab,
                  Suzhou Institute for Advanced Research,
                  University of Science and Technology of China}
}
```
