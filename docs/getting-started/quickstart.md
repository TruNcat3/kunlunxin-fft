# 首次运行

本页假定已完成[环境检查](installation.md)，命令从仓库根目录运行。

## 1. 先排除静默 no-op

```bash
CUDA_VISIBLE_DEVICES=6 python3 scripts/smoke.py
```

smoke 使用非零 sentinel，要求输出发生变化、有限且有能量，并检查错误 dtype、rank、设备、非连续
tensor 和非二次幂输入。它专门防止“kernel 未执行，但全零输出让简单误差检查假通过”。

## 2. 正确性矩阵

```bash
CUDA_VISIBLE_DEVICES=6 python3 scripts/test_correctness.py --profile smoke
CUDA_VISIBLE_DEVICES=6 python3 scripts/test_correctness.py --profile full
```

当前脚本对 seeded-random 输入分别运行 DIT/DIF，与 CPU NumPy complex128 参考比较。`full`
覆盖配置中的 18 个二次幂长度直到 `1048576`；不要把配置文件列出的其他 pattern 名称误解为脚本已
自动执行的测试。

## 3. 单次调用

```python
import torch
import torch_xmlir
from src.xpu import DeviceProfile, run_fft

x = torch.randn(2048, device="cuda", dtype=torch.complex64)
y, report = run_fft(
    x,
    "dit",
    profile_hash=DeviceProfile.load().hash(),
)
print(report.executed)
print(report.fallback, report.reason)
```

`run_fft` 每次都执行 plan lookup，再运行 kernel，适合交互使用。生产循环应复用 plan。

## 4. Plan reuse

```python
from src.xpu import DeviceProfile, execute_fft_plan, plan_fft

plan = plan_fft(
    "dit",
    n=x.shape[-1],
    batch=1,
    precision="fp32",
    profile_hash=DeviceProfile.load().hash(),
)

for _ in range(10):
    y = execute_fft_plan(plan, x)
```

`plan_fft` 是 host-only 冻结表查询，不做在线 benchmark；`execute_fft_plan` 会验证 shape 与 dtype。
发布矩阵的 2% promotion guard 针对 plan-reuse 路径，不能与 one-shot 开销混在一起。

## 5. Batch

```python
x = torch.randn((16, 2048), device="cuda", dtype=torch.complex64)
plan = plan_fft(
    "dit", 2048, batch=16,
    profile_hash=DeviceProfile.load().hash(),
)
y = execute_fft_plan(plan, x)
```

当前只有 M4/M0 batch lowering。不存在已测表项或不满足 lowering contract 时，plan 会保守回退
到 folded M0，而不是猜测一个候选。

下一页：[故障排查](../reference/troubleshooting.md)。
