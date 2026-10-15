# Plan 与选型

Kunlunxin-FFT 把“决定运行什么”与“重复执行”分开，避免在稳态循环中反复查表。

```text
静态硬件 profile + frozen selector table
                -> plan_fft(shape, semantics)
                -> ExecutionPlan
                -> execute_fft_plan(plan, x) repeated
```

## 解析 plan

```python
from src.xpu import DeviceProfile, load_table, plan_fft

table = load_table()
plan = plan_fft(
    "dit", 2048,
    batch=1,
    precision="fp32",
    profile_hash=DeviceProfile.load().hash(),
    table=table,
)
print(plan)
```

`plan_fft` 只进行合法性验证和 frozen-table lookup，不是在线搜索器，也不是解析性能模型。plan 记录：

- `executed`：实际 runner 入口；
- `flow_realization` 与 `processing_unit`：流组织和计算核心；
- `block` 与 `params`：lowering 参数；
- `fallback`、`reason`、`selection_source`：为什么选择或回退。

## 闭包与回退

表键为 `(device/profile, precision, direction, N, batch)`。以下情况不会猜测候选：

- `profile_hash` 缺失或不匹配；
- shape 不在已测冻结表；
- 选择项缺少当前 direction/batch 的合法 lowering；
- 语义本身不支持。

前三类回退到 M0（batch 使用 M4 folded 的 conservative path），最后一类直接报错。这样 v0.8
候选空间在运行时保持 M0 闭包，但“约持平”仍有测量噪声，不能理解为逐次调用严格不慢。

## One-shot 与 plan reuse

`run_fft` 返回 `(output, ExecutionReport)`，每次包含查表与对象构造。`plan_fft + execute_fft_plan`
把一次性选择移出循环。正式 release matrix 在 N=16/32 测得 one-shot 额外约 `0.016/0.012 ms`
（`7.9%/3.4%`）；这个成本与 kernel/flow 收益分列。

## 换硬件

不要修改 `config/device.json` 的设备名后继续使用旧表。profile hash 只是配置签名，并不自动探测
实际硬件。新硬件应重新验证 lowering、采集候选、构建表并保留独立 provenance；当前自动化缺口见
[未来计划](../roadmap.md)。

下一页：[Python API](../reference/api.md)。
