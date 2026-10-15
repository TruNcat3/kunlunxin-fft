# 支持范围

本页描述当前发布接口，不定义项目最终目标。

| 能力 | 当前支持 | 限制 |
|---|---|---|
| C2C | FP32/complex64，forward，1D | 无 inverse/normalization 选项 |
| 长度 | 正二次幂；正确性网格到 `1048576` | 表内性能选择只覆盖部分 shape |
| Batch | `(B,N)`，B/N 为正二次幂 | 仅 M4/M0 batch lowering；发布 B=1/4/16/64 子集 |
| 用户布局 | 连续 `torch.complex64` | arbitrary stride 不支持；planar 为内部 ABI |
| 输出 | 自然顺序 | DIT/DIF 均为 forward 分解 |
| 设备 | 参考 xpytorch 栈上的单张 P800 OAM | 其他卡/driver 未校准 |
| API | Python one-shot 与 plan reuse | 无稳定 C ABI、用户 stream、workspace API |

M1/M2/M3/M5 当前只支持单 transform；M4 提供 spread/folded batch。表外 shape 可以走 M0
correctness fallback，但“能执行”不等于已经测得性能最优。

当前记录的 vendor baseline 在 `N>2048` 返回 unsupported，这只适用于该 xpytorch/xdnn 版本。
本项目不宣称 P800 或所有厂商库永远受此长度限制。

未支持：R2C/C2R、inverse、fp16/bf16/fp64、非二次幂、2D/多维、任意 stride、跨卡、异步
stream、自动新硬件 profile/模型。

下一页：[术语](glossary.md)。
