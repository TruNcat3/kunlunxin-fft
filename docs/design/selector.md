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

M0 始终存在于设计空间。发布汇总以同 block M0 为 incumbent；profile mismatch、缺失 key 或无
lowering 时走 conservative fallback。表中 batch 只装载当前 M4 合法项，M1/M2/M3/M5 仍为单
transform。

`config/selector_table.json` 来自已归档的 `confirm_b1024.json`、`confirm_b256.json` 和
`confirm_batch.json`。当前 `bench.py` 新生成的 manifest 尚未自动补齐 table builder 要求的
`timing_scope/protocol_id`，所以文档不把“任意新 bench JSON 一键构表”宣传为已完成能力。

下一页：[实验结果](../benchmarks/results.md)。
