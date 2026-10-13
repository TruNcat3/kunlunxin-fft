# Kunlunxin-FFT

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Citation](https://img.shields.io/badge/citation-CFF-orange.svg)](CITATION.cff)
[![Hardware](https://img.shields.io/badge/SoC-Kunlunxin__P800__OAM-8A2BE2.svg)](docs/device-stack.md)
[![Method](https://img.shields.io/badge/method-cuButterfly-2F80ED.svg)](https://github.com/TruNcat3/cuButterfly)

**Kunlunxin-FFT 是 cuButterfly 混合空间-时间数据流方法在昆仑芯 P800 上的迁移实例。**
本仓库的首要目标是在一个公开、可复现的实验范围内，确定 P800 上**可执行**的设备代码路径
（Triton -> XMLIR -> XPU 指令），记录并论证哪些路径不可执行及其原因，并交付一个经过正确性
验证、与厂商栈对比的 radix-2 FFT 基线。

本里程碑是**正确性优先的基线**，不是性能收益。实测数据与诚实解读见 [Results](docs/results.md)。

<p align="center">
  <a href="https://www.ustc.edu.cn/">
    <img src="https://img.shields.io/badge/USTC-High%20Efficient%20Intelligent%20Computing%20Lab-8A2BE2.svg" alt="USTC HEIC Lab">
  </a>
</p>

## 与 cuButterfly 的关系

本项目是 [cuButterfly](https://github.com/TruNcat3/cuButterfly) 方法在昆仑芯 P800 上的迁移实例：
保留“阶段维度与数据维度同时进行空间/时间划分”的架构范式，但不复用 CUDA kernel；计算、布局、
存储和同步均重新映射到 P800 的设备模型。方法来源与差异归属于
[cuButterfly](https://github.com/TruNcat3/cuButterfly)；本项目不将这些方法论观点重新宣称独立提出。

**计算流优化**决定阶段和数据如何分块、复用、交接、重排和流水；**计算核心优化**决定每个局部块
内部用 radix-2 或其他已验证蝶形算法完成算术；**硬件 lowering** 再决定具体指令与数据通路。
cuButterfly/本项目的主要研究对象是计算流；本项目当前使用的 radix-2 核只是一个可执行的物理实例，
不是方法的固定定义。

## 设备栈的关键结论

本仓库记录的最重要发现，是 P800 上哪些设备代码路径**可执行**、哪些**不可执行**及其证据：

| 设备代码路径 | P800 可执行？ | 原因 |
|---|---|---|
| **Triton -> XMLIR -> XPU 指令** | **是** | 唯一可用路径。`triton` 3.0.0 带 `xpu` 后端，运行期编译并启动。 |
| nvcc -> PTX | 否 | P800 不执行 PTX。 |
| clang 手写 CUDA 设备码 -> xpubin | 否 | 可编译，但 XPU ELF loader 需要 torch 进程内运行时状态，裸 C++ 宿主无法复现（静默 no-op）。 |
| XHPC / xpucc SDK | 否 | 参考容器内不存在。 |
| 片上多阶段融合（gather/scatter） | 否 | Triton 3.0.0 XPU 构建拒绝 `x[idx]` gather，SSA 形式无法表达原地 shared-memory FFT。 |

完整证据（含排除每条死路的实验）见 [Device Stack](docs/device-stack.md)。

## 当前能力

| 变换 | 精度 | 长度 | 布局 | 状态 |
|---|---|---|---|---|
| C2C forward (DIT) | fp32 complex | 2 的幂，`8..1048576` | planar 分离实虚部 | 稳定实验路径 |
| C2C forward (DIF) | fp32 complex | 2 的幂，`8..65536` | planar 分离实虚部 | 稳定实验路径 |

复数采用 **planar**（分离 float32 实部/虚部缓冲）布局，因为 Triton XPU 后端没有 `complex64`
参数类型。本里程碑不含逆变换、R2C/C2R、batched API 或稳定 C ABI。设备代码是 Triton，宿主驱动
是 Python；`include/` 与 `src/host/` 下的 C++ 是索引/twiddle 数学的**宿主侧参考实现**，由 g++
单元测试在无设备时验证，不是设备路径。

## 性能概览

在单张 P800 OAM 上实测（`block=1024`，device-only）。厂商基线 `torch.fft.fft`（xdnn）在本设备
**仅支持 N <= 2048**。完整表格与诚实解读见 [Results](docs/results.md)。

| 场景 | 基线 | 解读 |
|---|---|---|
| N <= 2048（厂商可达） | `torch.fft.fft` | 朴素 Triton radix-2 **更慢**（0.04x–0.90x），启动与全局访存成本主导。 |
| N >= 4096（厂商不可达） | 无 | 本库是**唯一**路径；正确运行到 1048576，rel err ~6e-7。 |

本里程碑的价值主张是**覆盖范围、正确性与一条有据可查的可执行路径**，不是速度。缩小性能差距正是
cuButterfly 混合空间-时间数据流（分块、寄存器驻留、阶段融合）的目标；具体阻塞点（无 gather/scatter）
记录在 [Device Stack](docs/device-stack.md)。

## 快速开始

```bash
# 1. 环境（需要厂商 xpytorch 栈：torch + torch_xmlir）
bash scripts/init.sh

# 2. 构建宿主侧 C++ 参考实现 + 单元测试，并预热 Triton 缓存
bash scripts/build.sh

# 3. 设备端正确性
CUDA_VISIBLE_DEVICES=6 python3 scripts/test_correctness.py --profile smoke

# 4. 基准测试
CUDA_VISIBLE_DEVICES=6 python3 scripts/bench.py
```

架构见 [Architecture](docs/architecture.md)，实测数据见 [Results](docs/results.md)。下一阶段的修复、
性能建模和混合数据流实现要求见
[Migration Convergence Plan](docs/development/migration-convergence-plan.md)。

## 目录结构

```
config/                 设备 profile、长度、设计空间、测试矩阵
include/butterfly/      宿主侧 C++ 参考实现（header-only）
src/xpu/                设备路径（Triton -> XMLIR -> XPU）
src/host/               宿主侧参考（CPU，g++ 单元测试）
scripts/                init / build / 测试 / 基准
tests/                  正确性入口
results/p800/           公开的基准数据
docs/                   index / architecture / device-stack / results
```

## 许可

[Apache-2.0](LICENSE)。引用见 [CITATION.cff](CITATION.cff)。
