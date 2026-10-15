# 端到端与应用口径

Kunlunxin-FFT 当前没有独立的应用模型 benchmark；本页先定义端到端口径，避免把 device/kernel
时间、API wall time、plan 创建和厂商 baseline 混在一起。发布矩阵采用同步 API wall time，包含
dispatch、分配、内部布局和同步，但 plan 已提前解析并复用。

## 三类时间

| 时间 | 包含内容 | 用途 |
|---|---|---|
| plan creation | 查表、合法性检查、对象构造 | 评估 one-shot/初始化成本 |
| API wall | runner 可见的 dispatch、分配、布局、同步 | 当前 release headline |
| kernel/device | 能可靠获取时的设备执行区间 | 局部归因，不能代替 API 结果 |

正式 release 的 N=16/32 one-shot 额外成本约 `0.016/0.012 ms`，单独列出，不并入 plan-reuse
收益。任何把传输或计划创建移出计时区间的实验，都必须明确为另一种协议。

## 当前与 vendor 的边界

记录的外部 baseline 是同一 P800 xpytorch 栈中的 `torch.fft.fft`。只有 `(B,N)`、dtype、forward
C2C 和计时范围完全匹配且调用成功时才进入 vendor 比较；unsupported 保留原始错误，不填补数值。
当前 `N>2048` 的 unsupported 是该记录软件栈的事实，不是所有 P800 软件版本的永久限制。

因此当前证据支持“在 P800 上形成可执行迁移并在部分 C2C 点相对匹配 baseline 获益”，不支持“已
全面超过专业 FFT 库”。应用 shape、异步 stream、H2D/D2H 和多算子融合属于后续扩展。

下一页：[规模切片](scale.md)。
