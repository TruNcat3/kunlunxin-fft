# P800 设备路径

## 当前可执行路径

```text
Python host
  -> torch + torch_xmlir 初始化 XPU runtime/context
  -> Triton JIT
  -> XMLIR
  -> XPU instructions
  -> P800
```

参考容器中，导入 `torch_xmlir` 后 `torch.device("cuda")` 指向 P800。`torch.xpu` 未启用；
`CUDA_VISIBLE_DEVICES` 仍用于筛选物理设备。

## 已排除的路径

| 路径 | 本环境结论 | 范围 |
|---|---|---|
| nvcc -> PTX | P800 不执行 PTX | 架构不匹配 |
| clang 生成 XPU ELF + bare C++ host | 返回成功但无可靠执行；依赖 torch 进程内 runtime 状态 | 记录容器/库版本 |
| XHPC/xpucc | 参考容器未提供 | 不是硬件永远不可用 |
| Triton tensor-subscript `Xt[idx]` | 记录版本 lowering 失败 | pointer offset `tl.load(ptr+idx)` 可用 |

“tensor-subscript 失败”不能简写成“gather/scatter 不可能”。M1 使用 pointer form 与静态索引，M2
使用 dense linear map，M3 使用合法 execution group；当前 shipped kernel 不依赖失败形式。

完整原始调查与 loader 实验保存在[设备路径调查归档](../archive/device-path-investigation.md)。

## Context 与静默 no-op

Triton launch 必须处于已建立的 XPU context。早期路径在 context 缺失时可能静默不写输出，因此
`xpu_context` 统一包围设备段，`scripts/smoke.py` 用非零 sentinel 检查输出确实变化。新增 kernel
不得只以“调用返回 0”作为成功标准。

## 危险操作

非法参数可能让设备长时间满利用率。使用 `scripts/diag_snapshot.sh` 先保存进程、`xpu-smi`、日志和
可选 `py-spy`；不要用会匹配当前 shell 的 `pkill -f <script-name>`。已复盘案例见
[2026-10-14 挂起事件](incidents/2026-10-14-p800-bench-hang.md)。

下一页：[性能与故障诊断](profiling.md)。
