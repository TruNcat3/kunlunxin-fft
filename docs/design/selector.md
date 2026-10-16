# 选择器

当前 selector 是**profile-bound frozen measured table**，不是在线 autotuner，也不是拟合模型。

## 数据流

```text
候选扫描 -> correctness gate -> 同 shape/block 与 M0 比较
         -> 至少 2% search-stage median improvement 才晋升
         -> config/selector_table.json
         -> runtime exact-key lookup
         -> legal runner or M0 fallback
```

表键包括 profile hash、precision、direction、N 与 batch；值包括 block、flow、processing unit 和
lowering 参数。2% 是工程 promotion guard，不是统计显著性声明。

## 为什么冻结

运行时在线搜索会把编译、warmup、缓存状态和用户延迟混在一起。冻结表把研究阶段的测量与部署阶段的
执行分开，并允许版本化 provenance。缺点是表外 shape/硬件不能自动得到最优配置，因此当前选择器
宁可 abstain 到 M0，也不外推。

## 当前闭包

M0 始终存在于设计空间。发布汇总以同 block M0 为 incumbent；profile mismatch、缺失 key、
**环境指纹缺失/失配**或无 lowering 时走 conservative fallback。生产 schema 的表必须绑定
`env_hash`（建表时取自测量 manifest）；未绑定的旧表不参与运行时选择，只允许显式离线检查。
每个表项在加载/建表时都过共享 lowering 契约（`candidates.check_lowering`）与设计空间成员检查，
非法 flow/unit/residence/pipeline 一律拒载。

`config/selector_table.json` 来自 `confirm_b1024.json`、`confirm_b256.json`、`confirm_batch.json`
与 Issue #3 的 focus 轮（`batch_focus_a/b`）。当前 `bench.py` 新生成的 manifest 尚未自动补齐
table builder 要求的 `timing_scope/protocol_id`，所以文档不把“任意新 bench JSON 一键构表”
宣传为已完成能力。

下一页：[实验结果](../benchmarks/results.md)。
