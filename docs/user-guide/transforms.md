# 变换契约

## 当前语义

| 属性 | 当前公开行为 |
|---|---|
| 变换 | 1D C2C forward/inverse；R2C/C2R 为 **compatibility path**（`transform_type="r2c"/"c2r"`，复用 C2C 核心，见下） |
| 数据类型 | C2C：`torch.complex64`；R2C 输入 `float32` 实数；C2R 输出 `float32` 实数 |
| 长度 | 正二次幂；C2C 发布正确性网格 `8..1048576`；R2C/C2R 首轮 `64..2048` |
| Batch | `(B, N)`，B 与 N 均为正二次幂 |
| 布局 | 用户为连续 tensor；planar 仅为内部 lowering |
| 输出 | 自然顺序；normalization 由 `norm` 显式给出（默认 `backward`，与 torch.fft 一致） |
| DIT/DIF | 同一 C2C 的两种阶段组织（decomposition），与 forward/inverse 正交 |

### R2C/C2R compatibility path

- R2C：输入实数 `(N,)`/`(B,N)`，输出半谱 complex64 `(N//2+1,)`/`(B,N//2+1)`；
  实现 = 显式打包成 complex（imag=0）→ C2C 核心 → 半谱裁剪，全部工作在
  `execute_fft_plan` 内完成（计入 one-shot 与 plan-reuse 计时），plan 在
  `params["real_compat"]` 标记为 compatibility path，不冒充专用实数 FFT 核心。
- C2R：输入半谱 complex64，输出实数全长；实现 = Hermitian 重建（`X[n-k]=conj(X[k])`，
  DC 与 Nyquist 的虚部按 torch.fft 约定忽略）→ inverse C2C → 取实部。
  one-shot 推断全长为偶数 `N = 2*(m-1)`（与 torch 一致；奇数 N 需显式 plan `n`）。
- 首轮范围：N ∈ {64,128,256,512,1024,2048}，B ∈ {1,16,64}，FP32/complex64。
  后续厂商/第三方/自研 real-FFT 核心可作为 processing unit 接入同一合同。

### Normalization

与 `torch.fft`/`numpy.fft` 完全一致（`n = N`）：

| norm | forward | inverse |
|---|---|---|
| `none` | 1 | 1 |
| `forward` | 1/N | 1 |
| `backward`（默认） | 1 | 1/N |
| `ortho` | 1/sqrt(N) | 1/sqrt(N) |

Inverse C2C 复用 forward lowering：对共轭输入执行同一选中 flow，输出再共轭并做 norm 的
epilogue scale；不建立平行架构，selector 表仍复用 forward incumbent（共轭技巧下候选结构
一致，实测排序可迁移）。不支持的 `transform_type`/`norm` 在查表前 `ValueError`，绝不回退到
不同变换。

单变换输入是 `(N,)`。one-shot `run_fft` 也接受 `(1, N)` 并自动 squeeze/restore；已解析的 B=1
plan 则要求精确的 rank-1 `(N,)`。Batch plan 要求精确匹配 `(B, N)`。

输入必须位于 XPU 设备、连续且为 complex64。非二次幂、CPU tensor、非连续视图、错误 rank 或
错误 dtype 会在 launch 前拒绝。

## DIT 与 DIF

DIT 先按位反转解释输入，再按低到高阶段完成蝶形；DIF 按高到低阶段计算并在输出端完成位反转。
两条路径最终都返回与 `numpy.fft.fft` 相同语义的自然顺序 forward FFT。它们用于探索不同阶段组织
与硬件 lowering，不代表 API 方向。

## 尚未支持

专用（非兼容路径）real-FFT 核心、fp16/bf16/fp64、非二次幂、任意 stride、2D/多维、
异步用户 stream、device-pointer C ABI 均不属于当前发布接口。配置空间或文档中出现研究
目标不等于已有可执行 kernel。

下一页：[Plan 与选型](plans.md)。
