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

## Round E：批量聚焦（issue #3，clean-commit 重跑）

正式 release matrix 中四个重点 shape（N=1024/2048 × B=16/64，双向）在聚焦前选中
M4 fallback（对 M0 约 `1.0x`）。评审要求以干净提交重跑重点矩阵：两轮独立 focus
+ 一轮全空间 holdout 均在 commit `70bd471`（`dirty=False`，环境指纹
`6b50bd316032c0fe` 绑定）下测得，各 3 session × 30 trial：

| N | B | focus A 最优（tile=128 家族） | 稳态 incumbent | 提速 | vendor 仍领先 |
|---:|---:|---:|---:|---:|---:|
| 1024 | 16 | 0.695–0.714 | 1.69–1.76 | **2.2–2.5x** | 2.3–2.9x |
| 1024 | 64 | 2.25–2.31 | 5.99–6.16 | **2.6–2.7x** | 8.5–8.9x |
| 2048 | 16 | 1.55–1.57 | 3.41–3.51 | **2.2x** | 2.3–2.4x |
| 2048 | 64 | 4.98–5.04 | 11.9–12.4 | **2.4–2.5x** | 7.5x |

- **赢家稳定性（评审要求的诚实口径）**：两轮精确同名赢家一致 **2/8**，但
  tile=128 家族一致 **8/8**、统计上 5/8 落在 2σ 内——rows 参数是 <1% 近平局，
  家族结论可复现；表内只晋升过 2% guard 的赢家，incumbent 保留为 fallback。
- 2% 正式 guard 下晋升决策一致率 **100%**（130 对 challenger/incumbent）；
  全空间 holdout（9 点枚举 3 shape）模型 top-3 命中实测最优 **6/6**，
  top-1 选择 regret **0.00–0.22%**（均值 0.21%）。
- 消融把收益拆开（旧轮数据，flow/unit 轴结论不变）：batch lowering
  `1.98–4.61x`、radix-4 core `1.75–2.20x`、local-matrix `1.99–2.17x`。
- **P2 Td canary 维持负结果**：`m2_packet_td{1,2}` 相对 `m2_batch` 为
  `1.09x` 更慢（分母 = 当前本地核），不晋升。
- **外部核四格消融（§4-§5）**：A0P1 按构造 abstain（融合程序边界即整核）；
  分解流 × triton 本地核（A1P0）比融合路径慢 **2.6–3.4x**（框架 IO 主导，证实
  评审假设）；分解流 × vendor 分块核（A1P1）在 B=16 反超 A0P0（0.30–0.32 vs
  0.75/1.48），但仍不敌 vendor 整变换基线；该 canary 只报告、不晋升。
- vendor 重探确认当前栈 `N≤2048` 可用、`N≥4096` unsupported；四个 shape 上
  vendor 仍快 `2.3–8.9x`。
- 完整证据（provenance、5 问作答、holdout 明细、外部核 inventory、复现命令）
  见[批量聚焦报告](batch-focus.md)与[外部核清单](external-cores.md)。

下一页：[测量协议](methodology.md)。
