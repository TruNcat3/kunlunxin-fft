# 支持范围

本页描述当前发布接口，不定义项目最终目标。

## 当前发布证据

Issue #9 v6-1 是迁移验收的最新快照：`41/41` release gate 通过，`469` 个 host 测试和 `370`
个 hardware 测试通过；140 个 current/M0 可比较 cell 的几何平均延迟比为 `0.6086x`。这证明
方法、lowering、环境绑定和选择闭环已经在 P800 上可重放。

这不代表已经追平厂商 FFT。106 个 vendor-supported 且语义匹配的 cell 中，结果为 `4/0/102`
（win/tie/loss），几何平均 current/vendor 约 `0.122x`。支持范围表描述的是“能否正确执行”，
不是“是否达到厂商性能”；详细边界见[Issue #9 v6-1](../benchmarks/issue9-v61.md)。

| 能力 | 当前支持 | 限制 |
|---|---|---|
| C2C | FP32/complex64，1D forward/inverse；`norm=none/forward/backward/ortho` | inverse 复用 forward lowering，epilogue 负责共轭与缩放；不等同于独立 inverse kernel |
| R2C/C2R | FP32 real/complex64，1D，首轮 `N=64..2048` | compatibility path：显式打包/Hermitian 重建后复用 C2C；不是专用 real-FFT 核心 |
| 长度 | 正二次幂；C2C 正确性网格到 `1048576`；语义矩阵快照覆盖 `N=64..65536` 的已列 shape | vendor 对比在当前栈仅 `N<=2048` 可运行；`N>=4096` 的 vendor unsupported 不是本项目的性能胜负 |
| Batch | `(B,N)`，B/N 为正二次幂；语义矩阵首轮包含 `B=1/4/16/64`，长尺寸补测 `B=256` | M0/M4 基线及已验收的 M2 matrix/local-fused、M5 radix-4 候选；冻结表只覆盖已测 shape |
| 用户布局 | 连续 `torch.complex64` | arbitrary stride 不支持；planar 为内部 ABI |
| 输出 | 自然顺序；normalization 与 `torch.fft` 对齐 | DIT/DIF 是 forward 的阶段组织，不代表 API 方向 |
| 设备 | 参考 xpytorch 栈上的单张 P800 OAM | 其他卡/driver 未校准 |
| API | Python one-shot 与 plan reuse | 无稳定 C ABI、用户 stream、workspace API |

单变换 M1/M2/M3/M5 与 batch M2/M5 均有真实 runner；Issue #3 focus 还验证了参数化
local-fused matrix 候选。冻结 selector 只收录同一环境下通过正确性与 2% guard 的点；表外 shape
走 conservative M0/M4 fallback，“能执行”不等于已经测得性能最优。Issue #6 的语义矩阵还记录
了 inverse/R2C/C2R 的可执行兼容路径，但这些路径的广泛性能尚未达到 C2C focus 的成熟度。
`config/composition_table.json` 只含一个精确的 `N=2048,B=16,c2c,fp32,dit` 条目；它是
host-API 子变换组合，不是 resident kernel，也不扩大 selector 的通配范围。

当前记录的 vendor baseline 在 `N>2048` 返回 unsupported，这只适用于该 xpytorch/xdnn 版本。
本项目不宣称 P800 或所有厂商库永远受此长度限制。

未支持：专用 real-FFT 核心、fp16/bf16/fp64、非二次幂、2D/多维、任意 stride、跨卡、异步
stream、稳定 device-pointer/C ABI，以及安装期一键完成新硬件 profile/重测/构表。手动
`hw_probe`、结构成本模型和环境绑定已经实现；自动校准和更广泛语义的性能表仍在路线图中。

## 如何阅读当前证据

- [正式 release matrix](../benchmarks/results.md)：Issue #3 之前的冻结
  C2C 发布快照，强调方法相对 M0 的收益。
- [Issue #6 语义矩阵](../benchmarks/results.md#issue-6-semantic-matrix-v2)：四种语义的统一
  correctness/API-wall 快照，包含 vendor unsupported 的保留记录，不把 unsupported 当作胜出。
- [composition table](../benchmarks/external-cores.md)：
  仅一个 exact shape 的 host-API 组合晋升，不能外推到其他 shape 或 resident pipeline。

下一页：[术语](glossary.md)。
