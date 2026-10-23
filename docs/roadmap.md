# 未来计划

Kunlunxin-FFT 当前定位是 cuButterfly 方法的 P800 迁移实例和可复现研究库。近期目标是保持已声明
范围内的正确性、选择闭包和可解释收益，而不是无限扩张搜索或预先承诺全面超过专业库。

## v6-1 状态冻结

Issue #9 已完成并关闭。全语义冻结快照 `20261020-issue9-semantic-matrix-v6-1` 的 release gate 为
`41/41 PASS`，host/hardware 验证分别为 `469/370`；140 个可比较 cell 的 current/M0 几何平均
延迟比为 `0.6086x`，10-cell paired closure 的中位改善约 `60.7%`。相对厂商的 106 个
vendor-supported cell 仍为 `4/0/102`、几何平均约 `0.122x`，所以迁移证明已完成，厂商级物理
性能尚未完成。Issue #9 的矩阵不再扩大。

后续[能力门禁 Issue #10](https://github.com/TruNcat3/kunlunxin-fft/issues/10)和 PR #12 已完成：
restricted A-axis + vendor-P 路线在 5 个 fused exact cell 上获得双基线加速证据，但默认 selector
尚未接线；same-buffer in-kernel permutation read 为 `not-lowered`，完整 resident/persistent
pipeline 维持 NO-GO。下一阶段只做有界的 production integration、稳定性门禁和 AscendFFT 风格
覆盖展示，不把本轮扩成无限性能搜索。

## 优先级

| 阶段 | 交付内容 | 退出条件 |
|---|---|---|
| P0 工程闭环 | 安装期硬件 probe、profile 版本化、bench manifest 协议字段自动生成。**已落地**：`hw_probe` + env 指纹绑定 + 运行时 fail-closed；**剩余**：manifest 协议字段自动生成 | 新环境可生成独立 profile；旧表不会静默复用 |
| P1 Batch 性能 | **有界阶段已完成**：M2/M5/local-fused 纳入 focus 比较；Issue #10 获得 5 个 fused、3 个 vendor-P promotion candidates；**下一步**：仅对稳定 exact key 接线 | 第三独立 session 或 spread guard 通过；相对 vendor 与实际 incumbent 的 CI 下界均大于 `1.02`；负点完整报告 |
| P1 选择模型 | 参数化成本模型 + 小规模校准 + 留出验证。**已落地**：声明常数模型 + bounded search + focus/holdout 隔离验证（`model_validation.json`） | 排序误差、选择损失和搜索成本均可复核 |
| P2 算子语义 | **首轮已实现**：C2C inverse、四种 normalization、R2C/C2R compatibility path；Issue #6 semantic matrix v2 已归档 | 继续补专用 real-FFT 核心、扩大语义性能矩阵，并让每种语义都有独立 selector/同语义基线 |
| P2 尺寸/布局 | 非二次幂、stride、2D/多维 | 明确算法与 padding 语义，端到端计时包含重排 |
| P3 平台扩展 | 其他 Kunlunxin 设备与软件栈、FlagOS/vendor Triton 能力核验 | 每台硬件独立 probe、lowering、校准和发布矩阵；能力不足时 fail closed |

## Persistent 混合数据流

当前 M1/M2/M3 已验证局部融合、驻留和 grouping，但 Issue #10 的当前 SDK/探针尚未验证出足以支持
完整跨 worker 多 role persistent pipeline 的合法 lowering，因此本轮判定为 NO-GO。该方向保留为未来 SDK/工具链能力项，
不再作为近期迁移实例收敛的阻塞条件。若以后重新开启，应先在单一代表 shape 上证明
load/compute/store 的可观测重叠，再扩展 role 数与 packet 模板，并报告：

- 物化 GM 边界和实际 bytes；
- 每个 role 的服务时间与最慢角色；
- resident worker/局部容量；
- 同步、填充/排空和尾部；
- 相同 processing unit 下相对非 persistent flow 的净收益。

## 计算核心扩展

radix-8、向量化复数乘或其他局部 FFT 核心可以加入 processing-unit 轴，但必须提供 capability 与
lowering contract。核心优化和流优化不互斥；联合最优可以用于部署，研究归因仍需固定一轴做 A/B。

外部核组合（host-API subtransform composition：框架持有分解、在 materialized 子块上调用
vendor 整变换 API，四格消融 A1P1）已完成**精确 shape 晋升**（issue #6 PR-E）：仅
`N=2048,B=16` 在独立确认 session 下对 A0P0 incumbent 过 2% guard（protocol-session
pooled p50：A1P1 `0.2477 ms` vs A0P0 `1.5076 ms`，+83.6%），证据
`results/p800/composition_confirm.json` 经 `scripts/promote_composition.py` 原子写入独立的
`config/composition_table.json`（绑定 capability contract `a1p1_materialized_subtransform`、
env/profile hash、evidence sha256、entries `result_sha256`）。运行时选择 fail-closed：仅
c2c、仅该 exact shape、profile/env 精确匹配，否则走原 selector 路径；`N=1024,B=16/64`
负点保留为证据、拒绝晋升。**P2 已实现**：
HybridPlan 缓存 twiddle/staging buffer（无 per-call arange/outer/cos）、调用/物化账本
（`plan_ledger`，sub-transform 形状/核调用数/分析 GM 字节）、T×R 有界因子扫描
（1024: 16×64/32×32/64×16；2048: 32×64/64×32，无 N/B 扩张）、`config/external_cores.json`
capability registry、3 session × 30 trial 交错复测脚本 `scripts/run_hybrid_canary.py`。
它不是片上驻留单核，也不嵌入单 kernel（entry 归因 `executed=run_hybrid`、
`selection_source=composition`，绝不冒充 resident kernel）。Issue #3 的独立进程复测得到：`N=2048,B=16`
为 `0.2298 ms`，相对 vendor `0.6586 ms` 为 `2.87x`；`N=1024,B=16/64` 仍分别落后约
`3.6%/33.3%`。因此该路径证明“架构编排可以组合外部核心”，只在满足 2% guard 的 exact shape 进入
composition 表（当前 2048×16；不进 `config/selector_table.json`），其余 shape 仍不进入 selector。

## Issue #6 的当前边界

Issue #6 的 WP0/WP1/WP2 已落地：环境/bootstrap 门、C2C inverse 与 normalization、R2C/C2R
compatibility path、统一语义矩阵和 immutable snapshot 均有代码与 provenance。最新 snapshot
`results/p800/releases/20261017-wp2-semantic-matrix-v2` 共 144 个 cell，其中 120 个 vendor
可运行；selector 相对 vendor 为 `7/0/113`（win/tie/loss），几何平均 speedup `0.0986x`。
因此这一步证明的是**语义迁移和 fail-closed 选择闭环**，不是四种语义已经达到厂商性能。

WP3 仍是定向优化而非发布阻塞：

- 先处理 R2C/C2R 的打包、Hermitian 重建和 epilogue 开销；
- 再为 inverse/real path 建立独立的 candidate family 与 guard，不复用 C2C 的性能结论；
- 对 `N>=4096` 先解决 vendor unsupported 的可比性，再讨论 speedup；
- 保持 `composition_table.json` 的 exact-shape、env/profile 绑定和 fail-closed，不把单个正点
  泛化成全局 selector。

这与 cuButterfly 的迁移原则一致：架构编排、processing unit 和 lowering contract 分层，
但每个语义仍需自己的测量协议和硬件证据。

## 已完成的迁移验收

关闭的 Issue #3 已完成以下闭环：

- batch M2/M5/local-fused 候选、2% guard 和冻结表闭包；
- 手动硬件 probe、环境指纹 fail-closed、结构成本模型与 bounded search；
- 两轮 focus、独立 full-space holdout、多进程 external-core canary；
- 负结果保留：packet-Td 慢 `1.09x`，两个 `N=1024` 外部核心点未翻转。

后续目标不是继续扩大同一轮搜索，而是补齐算子语义、公共 API 与真正 persistent residence。

## 发布原则

1. 新候选先正确性，后性能；
2. 同 shape/block/计时范围比较 incumbent；
3. 表外与新硬件默认 abstain；
4. 首页只使用正式快照，研究扫描单独标协议；
5. 少量负点不通过删点隐藏，而是进入下一轮定向任务。

下一页：[项目与作者](about.md)。
