# 实数变换边界

Ascend-FFT 文档将 R2C/C2R 单独列为设计专题，因为实数变换的频谱半存储、Hermitian 约束和输出
布局会改变数据流。Kunlunxin-FFT 已发布 FP32 C2C forward + inverse（四种 normalization）；
本页用于明确实数变换的剩余边界，避免把未来能力误读为当前接口。

## 当前状态

| 能力 | 状态 |
|---|---|
| forward C2C | 已验证，`torch.complex64` |
| inverse C2C | 已发布（共轭技巧复用 forward lowering + epilogue scale），见变换契约 |
| R2C/C2R | 未发布 |
| normalization | `none`/`forward`/`backward`/`ortho`（`backward` 默认，与 torch.fft 一致） |
| 多维/任意 stride | 未发布 |

实数变换不能只把输入 dtype 替换成 float：它需要独立的频谱长度、共轭对称边界、Nyquist 点和
输出布局合同。未来实现必须把这些语义作为独立 plan key，并与 C2C 使用相同的 correctness-first
和 profile-bound 选择流程。

## 迁移原则

可以复用的部分是阶段/数据混合编排、局部驻留和 processing-unit 选择；不能直接复用的是：

- 半谱输入输出的 ownership；
- 最后一阶段的共轭对称处理；
- real/complex layout 转换与边界重排；
- normalization 和 round-trip 误差合同。

在这些合同实现前，配置文件中的研究字段不得进入 selector。新增 R2C/C2R 时应参考
[Ascend-FFT 的实数变换设计](https://github.com/TruNcat3/ascend-fft/blob/master/docs/design/real-transforms.md)，
但重新验证 P800 的 lowering 和数据布局。

下一页：[选择器](selector.md)。
