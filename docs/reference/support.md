# 支持范围

本页描述当前发布接口，不定义项目最终目标。

| 能力 | 当前支持 | 限制 |
|---|---|---|
| C2C | FP32/complex64，forward，1D | 无 inverse/normalization 选项 |
| 长度 | 正二次幂；正确性网格到 `1048576` | 表内性能选择只覆盖部分 shape |
| Batch | `(B,N)`，B/N 为正二次幂 | M0/M4 基线及已验收的 M2 matrix/local-fused、M5 radix-4 候选；冻结表只覆盖已测 shape |
| 用户布局 | 连续 `torch.complex64` | arbitrary stride 不支持；planar 为内部 ABI |
| 输出 | 自然顺序 | DIT/DIF 均为 forward 分解 |
| 设备 | 参考 xpytorch 栈上的单张 P800 OAM | 其他卡/driver 未校准 |
| API | Python one-shot 与 plan reuse | 无稳定 C ABI、用户 stream、workspace API |

单变换 M1/M2/M3/M5 与 batch M2/M5 均有真实 runner；Issue #3 focus 还验证了参数化
local-fused matrix 候选。冻结 selector 只收录同一环境下通过正确性与 2% guard 的点；表外 shape
走 conservative M0/M4 fallback，“能执行”不等于已经测得性能最优。packet-Td 与 external
sub-transform 目前只作为 experimental canary，不属于默认执行闭包。

当前记录的 vendor baseline 在 `N>2048` 返回 unsupported，这只适用于该 xpytorch/xdnn 版本。
本项目不宣称 P800 或所有厂商库永远受此长度限制。

未支持：R2C/C2R、inverse、fp16/bf16/fp64、非二次幂、2D/多维、任意 stride、跨卡、异步
stream、安装期一键完成新硬件 profile/重测/构表。手动 `hw_probe`、结构成本模型和环境绑定已经
实现。

下一页：[术语](glossary.md)。
