# Python API

公共入口由 `src.xpu` 导出。Host-only 符号可在无厂商 torch 环境中导入；首次访问 device runner
时才加载 torch 依赖。

## 计划接口

```python
plan_fft(direction, n, batch=1, precision="fp32",
         profile_hash=None, table=None) -> ExecutionPlan

execute_fft_plan(plan, x) -> torch.Tensor

run_fft(x, direction, *, batch=None, precision="fp32",
        profile_hash=None, table=None) -> (torch.Tensor, ExecutionReport)
```

`direction` 当前只能为 `"dit"` 或 `"dif"`，两者均为 forward C2C。`precision` 只能为 `"fp32"`。
`execute_fft_plan` 要求输入 shape 与 plan 完全一致。

## Selector 与配置

```python
load_table(path=...) -> dict
select(direction, n, batch=1, precision="fp32",
       profile_hash=None, table=None) -> Selection
DeviceProfile.load() -> DeviceProfile
DeviceProfile.hash() -> str
```

`select` 返回表 verdict，不执行 kernel；`Selection.selected` 仅表示命中冻结表。推荐调用
`plan_fft`，因为它还验证实际 lowering 并产生可执行 fallback。

## 低层 runner

`fft_dit`、`fft_dif`、`fft_dit_fused`、`fft_*_matrix`、`fft_*_grouped`、`fft_*_batch` 和
`fft_*_radix4` 用于研究与 benchmark。它们绕过 selector，不构成同等级稳定 API；调用者须自行
保证参数合法并检查正确性。

当前无稳定 C ABI、预编译共享库、异步 stream 注入或 workspace 查询接口。

下一页：[为什么混合展开](../design/motivation.md)。
