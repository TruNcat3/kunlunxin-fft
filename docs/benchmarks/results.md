# 实验设计与结果汇总

本页先给当前正式发布证据，再给宽范围研究数据。两者协议不同，不混合计算平均值。

## 正式 P800 release matrix

<figure class="doc-figure">
  <a href="../../assets/figures/release-overview.svg"><img src="../../assets/figures/release-overview.svg" alt="Release speedup versus M0 and six primary matched vendor points"></a>
  <figcaption>左：同 block M0 / selected；右：N=1024/2048、B=1/16/64 六个主矩阵点的 vendor / selected。大于 1 表示 selected 更快。N=16/32 补充点和 profile-mismatch 负对照未画入右图。</figcaption>
</figure>

| N | B | Selected path | M0 ms | Selected ms | 对 M0 | 对 vendor |
|---:|---:|---|---:|---:|---:|---:|
| 16 | 1 | M1 fused | 0.3262 | 0.2181 | **1.494x** | 0.152x |
| 32 | 1 | M0 fallback | 0.3712 | 0.3687 | 0.997x | 0.056x |
| 1024 | 1 | M2 matrix | 0.4497 | 0.3591 | **1.252x** | 0.682x |
| 1024 | 16 | M4 fallback | 1.7736 | 1.7723 | 0.998x | 0.141x |
| 1024 | 64 | M4 fallback | 6.1552 | 6.1717 | 0.999x | 0.042x |
| 2048 | 1 | M2 matrix | 0.4769 | 0.3872 | **1.232x** | **1.660x** |
| 2048 | 16 | M4 fallback | 3.5845 | 3.6048 | 1.001x | 0.183x |
| 2048 | 64 | M4 fallback | 13.1054 | 13.1717 | 0.996x | 0.051x |
| 4096 | 1 | M2 matrix | 0.6139 | 0.4694 | **1.308x** | n/a |
| 4096 | 16 | M4 folded | 7.1682 | 7.1564 | 1.002x | n/a |
| 4096 | 64 | M4 folded | 27.4772 | 27.5797 | 0.998x | n/a |
| 65536 | 1 | M5 radix-4 | 9.2029 | 4.4455 | **2.072x** | n/a |
| 65536 | 4 | M4 | 35.1713 | 35.1632 | 1.001x | n/a |
| 1048576 | 1 | M5 radix-4 | 160.6676 | 86.0158 | **1.868x** | n/a |

`对 M0` 使用三个 session 的 session-level speedup 中位数，因此不一定等于表中两个四位延迟的
直接相除。vendor 列只在相同 `(B,N)`、FP32 forward C2C 调用成功时给出；当前记录软件栈在
`N>2048` 返回 unsupported，这不是所有 P800 软件版本的永久结论。

### 如何解读

- **方法迁移有效**：单变换中 M1/M2/M5 在多个尺寸稳定优于同 block M0，最大 `2.072x`。
- **闭包有效**：没有晋升点或 profile 不匹配时回退 M0；正常 batch 点维持约 `0.996x--1.002x`。
- **没有全面超过厂商库**：六个主 matched vendor shape 中只有 `N=2048, B=1` 胜出；表中的
  `N=16/32, B=1` 是另外两个小尺寸补充点。大 batch
  仍有明显差距，是路线图中的首要性能缺口。
- **正确性通过**：正式点全部正确；selected/M0 最坏 componentwise peak-normalized error 为
  `9.914935e-7`，阈值 `1e-4`。

profile-mismatch 是独立负对照：`0.990x` M0 闭包，不能作为方法收益或 vendor 胜出点。one-shot
plan 开销 N=16/32 分别约 `0.016/0.012 ms`，也不并入 plan-reuse speedup。

## 宽尺度 Phase 3 研究矩阵

Phase 3 覆盖 18 个长度 `2^3..2^20`、M0/M1/M2/M3/M5，以及 M4 的 B=1/4/16/64
spread/folded。450 条 long rows 全部通过阈值，最坏误差 `1.126129e-5`。它用于观察交叉点和
设计偏好，不是当前 release headline：该快照 commit dirty，block/protocol 也与 release matrix
不同。

<figure class="doc-figure doc-figure--wide">
  <a href="../../assets/figures/phase3-speedup-vs-m0.png"><img src="../../assets/figures/phase3-speedup-vs-m0.png" alt="Phase 3 speedup versus M0 across FFT lengths"></a>
  <figcaption>Phase 3 宽尺度消融。M1 在小长度获益、在大长度回退；M5 radix-4 的收益随长度增大。M2 同时改变核心和 residence，不能作为纯流消融。</figcaption>
</figure>

主要趋势：

- M1 radix-2 fused：`N<=1024` 几何均值 `1.62x`，`N>=4096` 为 `0.67x`；barrier 与
  store-load 依赖使收益反转。
- M2 dense-dot matrix：全范围相对 M0 几何均值 `1.64x`，但属于核心 + lowering 联合效果。
- M3 grouped：全范围约 `1.56x`，用于验证 tail grouping；不是当前 release selector 的唯一解。
- M5 radix-4：固定 global-stage flow 时，DIT 在 `N<=1024` 约 `1.03x`，`N>=4096` 约
  `1.69x`，说明计算核心偏好随尺寸变化。

下一页：[测量协议](methodology.md)。
