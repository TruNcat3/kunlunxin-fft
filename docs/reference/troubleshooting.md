# 故障排查

## `torch.cuda.is_available()` 为 false

确认已导入 `torch_xmlir`、xpytorch 与 driver 匹配，并检查 `CUDA_VISIBLE_DEVICES`。本栈使用
`torch.device("cuda")` 作为 P800 兼容键，不使用 `torch.xpu`。

## Kernel 返回但输出未变化

先运行 `python3 scripts/smoke.py`。缺失 XPU context 曾导致静默 no-op；不要用全零输入或只检查
返回码。所有设备调用应经过 `xpu_context`。

## Selector 总是 fallback

打印 `report.reason`，检查：

- 是否传入 `DeviceProfile.load().hash()`；
- profile hash 是否与 table 一致；
- `(N,batch,direction,precision)` 是否有表项；
- 表项是否有当前 runner lowering。

Fallback 是保守行为，不是异常。换硬件后不应通过伪造 hash 强行命中旧表。

## 输入被拒绝

公开输入必须是 P800 上连续的 `torch.complex64`。单变换是 `(N,)`，batch 是 `(B,N)`；N 和 B
必须为正二次幂。对 view 调用 `.contiguous()`，不要自行传内部 planar buffer。

## Benchmark 挂起

在另一个 shell 先保存快照：

```bash
bash scripts/diag_snapshot.sh <PID> <bench.log> diagnostics/local
```

避免 `pkill -f` 匹配自己的 shell。确认保存日志后再按服务器策略终止具体 PID/恢复设备。案例见
[挂起事件](../development/incidents/2026-10-14-p800-bench-hang.md)。

## Pages 图片 404

文档图片必须位于 `docs/assets/` 并相对引用。GitHub `blob/...png` 是 HTML 页面，不应作为 `<img>`
源。运行 `mkdocs build --strict` 后还需在浏览器检查生成站点。

下一页：[变换契约](../user-guide/transforms.md)。
