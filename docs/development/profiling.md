# 性能与故障诊断

当前仓库没有等价于 NCU 的通用 XPU counter 采集器，也没有安装期自动硬件 profile。文档中的
`kernel_ms` 来自 runtime 计时，GM bytes/launches 部分来自解析账本，不能称为硬件 counter。

## 性能问题的检查顺序

1. **语义与正确性**：先跑 smoke 和目标 shape 的 NumPy 对比；
2. **plan 归属**：记录 `executed/flow_realization/processing_unit/block/fallback/reason`；
3. **计时范围**：区分 plan reuse、one-shot、API wall time 和 kernel time；
4. **结构账本**：launch 数、物化 GM 边界、tile/group、batch spread/folded；
5. **资源假设**：block、局部 live set、同步和尾部；无 counter 时标为假设；
6. **配对实验**：固定核心比较 flow，或固定 flow 比较核心。

## 挂起快照

```bash
bash scripts/diag_snapshot.sh <PID> <bench.log> diagnostics/local
```

该脚本只读取状态，不终止进程。它保存 `ps`、`xpu-smi`、可用时的 `py-spy dump` 和日志尾部。
若进程在 native kernel 中自旋，`py-spy` 失败本身也应保留。

## 新硬件

不要把 `config/device.json` 当作动态探针。新设备需要单独记录硬件/driver/toolchain，验证 context、
lowering 与资源边界，再采集有限代表点并做留出验证。自动 probe、成本建模和 table 生成尚未闭环。

下一页：[验证与发布](validation-and-release.md)。
