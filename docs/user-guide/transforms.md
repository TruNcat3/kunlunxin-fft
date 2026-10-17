# 变换契约

## 当前语义

| 属性 | 当前公开行为 |
|---|---|
| 变换 | 1D C2C，`transform_type="c2c"`，`transform_direction` 为 `forward`/`inverse` |
| 数据类型 | 输入/输出 `torch.complex64`，内部 fp32 实虚分离 |
| 长度 | 正二次幂；发布正确性网格为 `8..1048576` |
| Batch | `(B, N)`，B 与 N 均为正二次幂 |
| 布局 | 用户为连续 complex64；planar 仅为内部 lowering |
| 输出 | 自然顺序；normalization 由 `norm` 显式给出（默认 `backward`，与 torch.fft 一致） |
| DIT/DIF | 同一 C2C 的两种阶段组织（decomposition），与 forward/inverse 正交 |

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

R2C/C2R、fp16/bf16/fp64、非二次幂、任意 stride、2D/多维、异步用户 stream、
device-pointer C ABI 均不属于当前发布接口。配置空间或文档中出现研究目标不等于已有可执行 kernel。

下一页：[Plan 与选型](plans.md)。
