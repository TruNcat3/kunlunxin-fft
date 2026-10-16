# 未来计划

Kunlunxin-FFT 当前定位是 cuButterfly 方法的 P800 迁移实例和可复现研究库。近期目标是保持已声明
范围内的正确性、选择闭包和可解释收益，而不是无限扩张搜索或预先承诺全面超过专业库。

## 优先级

| 阶段 | 交付内容 | 退出条件 |
|---|---|---|
| P0 工程闭环 | 安装期硬件 probe、profile 版本化、bench manifest 协议字段自动生成。**已落地**：`hw_probe` + env 指纹绑定 + 运行时 fail-closed；**剩余**：manifest 协议字段自动生成 | 新环境可生成独立 profile；旧表不会静默复用 |
| P1 Batch 性能 | **有界阶段已完成**：M2/M5/local-fused 纳入 focus 比较，M2 winner 接入 selector；**剩余**：扩展 N/B 网格与更强 residence | 已声明 B/N 网格相对 incumbent 有净收益，负点完整报告 |
| P1 选择模型 | 参数化成本模型 + 小规模校准 + 留出验证。**已落地**：声明常数模型 + bounded search + focus/holdout 隔离验证（`model_validation.json`） | 排序误差、选择损失和搜索成本均可复核 |
| P2 算子语义 | inverse、R2C/C2R、normalization | 每种语义有独立正确性与同语义基线 |
| P2 尺寸/布局 | 非二次幂、stride、2D/多维 | 明确算法与 padding 语义，端到端计时包含重排 |
| P3 平台扩展 | 其他 Kunlunxin 设备与软件栈 | 每台硬件独立 probe、lowering、校准和发布矩阵 |

## Persistent 混合数据流

当前 M1/M2/M3 已验证局部融合、驻留和 grouping，但还没有完整的跨 worker 多 role persistent
pipeline。后续应先在单一代表 shape 上实现 load/compute/store 的可观测重叠，再扩展 role 数与
packet 模板，并报告：

- 物化 GM 边界和实际 bytes；
- 每个 role 的服务时间与最慢角色；
- resident worker/局部容量；
- 同步、填充/排空和尾部；
- 相同 processing unit 下相对非 persistent flow 的净收益。

## 计算核心扩展

radix-8、向量化复数乘或其他局部 FFT 核心可以加入 processing-unit 轴，但必须提供 capability 与
lowering contract。核心优化和流优化不互斥；联合最优可以用于部署，研究归因仍需固定一轴做 A/B。

外部核组合（host-API subtransform composition：框架持有分解、在 materialized 子块上调用
vendor 整变换 API，四格消融 A1P1）当前是 **experimental canary**。**P2 已实现**：
HybridPlan 缓存 twiddle/staging buffer（无 per-call arange/outer/cos）、调用/物化账本
（`plan_ledger`，sub-transform 形状/核调用数/分析 GM 字节）、T×R 有界因子扫描
（1024: 16×64/32×32/64×16；2048: 32×64/64×32，无 N/B 扩张）、`config/external_cores.json`
capability registry、3 session × 30 trial 交错复测脚本 `scripts/run_hybrid_canary.py`。
它不是片上驻留单核，也不嵌入单 kernel。Issue #3 的独立进程复测得到：`N=2048,B=16`
为 `0.2298 ms`，相对 vendor `0.6586 ms` 为 `2.87x`；`N=1024,B=16/64` 仍分别落后约
`3.6%/33.3%`。因此该路径证明“架构编排可以组合外部核心”，但尚不满足跨 shape 的 production
晋升条件，仍不进入 selector。

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
