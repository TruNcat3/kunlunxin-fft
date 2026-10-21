# 实验设计与结果汇总

本页先给当前正式发布证据，再给宽范围研究数据。两者协议不同，不混合计算平均值。

## 最新：Issue #9 v6-1 迁移验收

当前应优先阅读[Issue #9 v6-1 独立结果页](issue9-v61.md)。不可变 release 目录为
[`20261020-issue9-semantic-matrix-v6-1`](https://github.com/TruNcat3/kunlunxin-fft/tree/master/results/p800/releases/20261020-issue9-semantic-matrix-v6-1)。

| 指标 | v6-1 结果 | 口径 |
|---|---:|---|
| release gate | `41/41 PASS` | artifact、manifest、路径身份和报告门禁 |
| 验证 | `469 host` / `370 hardware` | 独立 correctness 与设备路径测试 |
| 可比较 cell | `140` | current 与 M0 同语义比较 |
| current/M0 几何平均延迟比 | `0.6086x` | 约 39% 几何平均降时 |
| paired closure | `37.0%--75.2%`，中位 `60.7%` | 10 cells，全部通过 2% guard |
| vendor-supported | `106` | 成功调用且语义匹配 |
| current/vendor | `4/0/102` | win/tie/loss；几何平均约 `0.122x` |

**不要把这张表与下方历史快照混合平均。** 下方的正式 release matrix、Issue #6 语义矩阵和
Phase 3 扫描使用不同提交、候选集合或计时协议，保留它们是为了追溯 M0、旧 selector 和研究趋势。
v6-1 也没有新增真正的 batched M2/hybrid/h_dif、resident/persistent pipeline 或第三方局部
FFT core，因此 vendor 差距仍属于下一轮物理 lowering/processing-unit 工作，不应归因于“搜索已经
覆盖完整架构空间”。

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

## Issue #6 semantic matrix v2

这是当前主线最新的统一语义快照，原始文件与不可变 provenance 位于
[`results/p800/releases/20261017-wp2-semantic-matrix-v2`](https://github.com/TruNcat3/kunlunxin-fft/tree/master/results/p800/releases/20261017-wp2-semantic-matrix-v2)。
它与上面的正式 release matrix 使用不同 commit、候选集合和语义范围，**不能与 Issue #3 或
release matrix 混合平均**。计时是同一 P800 栈上的同步 API wall time；该软件栈没有可靠的独立
device-kernel timer，因此这里不把 API wall time 写成 kernel time。

| 项目 | 快照结果 |
|---|---:|
| 语义 cell 总数 | 144 |
| vendor-supported cell | 120 |
| vendor unsupported cell | 24（原始三 session 中保留为 72 条 unsupported 记录） |
| selector 相对 vendor | 7 胜 / 0 平 / 113 负 |
| 几何平均 `selector/vendor` | `0.0986x` |
| 最差 `selector/vendor` | `0.0165x` |
| selector 平均 regret（相对本快照 oracle 集合） | `0.4704` |

该矩阵覆盖 `c2c-forward`、`c2c-inverse`、`r2c`、`c2r`，并使用三进程交错、correctness-first、
2% incumbent guard 和 bounded measured oracle。`N>=4096` 的 vendor unsupported 只能说明当前
xpytorch/xdnn 版本拒绝这些调用，不能转换成 Kunlunxin-FFT 的性能胜出。

### 一个 shape 的语义分解

`N=2048,B=16` 的 selector/vendor API wall time 如下；同一 shape 的不同语义不能互相外推：

| semantic | selector ms | vendor ms | selector/vendor |
|---|---:|---:|---:|
| C2C forward（composition） | 0.3076 | 0.6643 | **2.160x** |
| C2C inverse（复用 forward lowering） | 0.4414 | 0.6659 | **1.509x** |
| R2C compatibility | 1.5664 | 0.6125 | 0.391x |
| C2R compatibility | 2.0947 | 0.6685 | 0.319x |

另有独立确认后的 exact-shape composition evidence：`N=2048,B=16,c2c,dit` 的 A1P1
`0.2477 ms`，A0P0 production incumbent `1.5076 ms`，相对 incumbent 通过 `+83.6%` guard，
并写入 `config/composition_table.json`。这项证据是 materialized host-API sub-transform
composition，不是 resident single-kernel；它不替代上表，也不扩大 selector 的 shape 范围。

**正确结论：** Issue #6 完成了语义/API/正确性/选择闭环的扩展，但广泛语义的性能仍明显落后
vendor；只有 C2C forward/inverse 的一个 shape 形成正证据。后续应优先优化 compatibility
路径和 candidate/selector 覆盖，而不是把 `7/120` 表述为全面领先。

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

## Round E：批量与模型迁移验收（Issue #3 已关闭）

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
- 2% 正式 guard 下晋升决策一致率 **100%**（按 focus 轮分别统计，各 58 对
  challenger/incumbent，focus 与 holdout 不混池）；全空间 holdout（9 点枚举
  6 个 shape-direction）模型 top-3 命中实测最优 **6/6**、full-space recall
  **6/6**，top-1 选择 regret **0.00–0.22%**（holdout-only 均值 **0.0816%**，
  m2 点按结构参数去重）。
- 消融把收益拆开（旧轮数据，flow/unit 轴结论不变）：batch lowering
  `1.98–4.61x`、radix-4 core `1.75–2.20x`、local-matrix `1.99–2.17x`。
- **P2 Td canary 维持负结果**：`m2_packet_td{1,2}` 相对 `m2_batch` 为
  `1.09x` 更慢（分母 = 当前本地核），不晋升。
- **外部核四格消融（§4-§5，三个可运行格的条件对照，卡6 重测，融合管线）**：
  A0P1 按构造 abstain（融合程序边界即整核）；P1 核是 **materialized
  external-subtransform**（host-API 子变换组合：框架持有分解、在物化子块上
  调用 `torch.fft.fft`），证明可组合性而非片上驻留单核。有界优化落地：
  stage1 融合 transpose+twiddle 单 GM pass、stage2 直接读 grid 视图（消除
  flat2 拷贝）、gather 单次进缓存块。融合后 A1P1 显著提速——**历史单进程
  四格**（commit `5fa0613b`）N2048B16 0.2051 vs vendor 整变换 0.6600 ms
  （3.22×）；N1024B16 单轮 0.2212 vs 0.2420（1.09× 快）——**该单进程翻转
  不是最终跨进程结论**，见下方多进程 canary。有界 T×R 扫描（1024:
  16×64/32×32/64×16；2048: 32×64/64×32，不扩 N/B）显示 32×32 对 A1P1
  最优（0.2053@1024B64）。只报告、不晋升、不外推。
- **P2 多进程复测（3 独立进程 session × 30 trial 交错 + 确认 session，卡6，
  commit `94390586`）**：每 session 新起 spawn 进程（独立 Python 分配器/
  JIT 缓存/runtime 状态，artifact 记录 `distinct_pids=true`），父与每个
  worker 均做 freshness 门，合并前对全部 worker 的 provenance 契约做严格
  相等校验（混合即中止）。**主表与 guard 均用 protocol-session 池化中位**
  （确认 session 从不混入）：N2048B16 A1P1 **0.2298 vs vendor 0.6586 ms
  （2.87×，guard +65.1% 通过）**，逐 session 0.2518/0.2226/0.2228 + 确认
  session 0.2512（guard 触发后单列）；all-session 池化 0.2345/0.6592 仅作
  诊断披露（all-session A1P1 离群 0 个；protocol-only 主表 A1P1 有 3 个
  离群样本，已列入 batch-focus）。1024 shape 仍慢于 vendor（B16 −3.6%，
  B64 −33.3%）——**N1024B16 单进程 1.09× 翻转在独立进程下不成立**；
  A1P1 对 A0P0 全 shape 通过 2% guard（+67.7~85.1%）。A0P0 经
  `selector.select` env 门获取；两份 artifact provenance 完整
  （commit/dirty=false/env 6b50bd316032c0fe，卡族 scope）。
  该结论是 experimental canary——cost model 无 A1 family，本轮不晋升
  selector；是后续工作的数据点，不是冻结声明。
- vendor 重探确认当前栈 `N≤2048` 可用、`N≥4096` unsupported；融合前 A0P0
  路径上 vendor 仍快 `2.3–8.9x`（融合后 A1P1@N2048B16 多进程下反超，见上）。
- 完整证据（provenance、5 问作答、holdout 明细、外部核 inventory、复现命令）
  见[批量聚焦报告](batch-focus.md)与[外部核清单](external-cores.md)。

该轮已经完成代码、数据与 provenance 验收并关闭。这里的 focus winner 已按 guard 进入冻结
selector；A1P1 多进程结果仍是 experimental canary，二者不能合并为同一种“发布性能”。

下一页：[测量协议](methodology.md)。
