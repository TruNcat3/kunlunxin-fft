# 测试策略与扩展矩阵

测试分成 host、设备正确性、候选 lowering、性能证据和文档五层。层次分开，才能判断失败来自
语义、编译器、选择器还是性能回归。

## 五层门禁

| 层 | 命令/入口 | 目标 |
|---|---|---|
| Host | `pytest -q` | plan、selector、配置和参考逻辑 |
| Smoke | `python3 scripts/smoke.py` | 排除 no-op、设备/rank/dtype 错误 |
| Correctness | `scripts/test_correctness.py --profile smoke/full` | DIT/DIF 与 NumPy complex128 |
| Hardware | `pytest -q -m hardware` | 真实 P800 lowering 与边界 |
| Release/docs | release matrix、`mkdocs build --strict` | provenance、性能协议和链接 |

## 扩展矩阵

当前发布语义是 `FP32/complex64 × forward C2C × 1D × 二次幂`。Issue #3 已在
`N=1024,2048`、`B=16,64` 上逐点比较 M0、M4、M2/M5/local-fused，并完成两轮 focus 与全空间
holdout。后续扩展应覆盖 `N=16,64,4096,65536,1048576` 和其他 batch，同时保留该网格作为回归
锚点。

任何超出设备内存、局部容量或 compiler envelope 的点都应在分配前记录为 `unsupported/resource`
并继续后续点，不能用 OOM 或挂起掩盖矩阵缺口。

## 正确性和性能不能互相替代

候选必须先通过误差门禁，再参加计时。性能测量至少保留三次独立 session、随机交错顺序、全部
trial、中位数和波动。API wall、kernel/device、plan creation 分列。没有硬件 counter 时不要把
解析 GM bytes 写成实测带宽或 occupancy。

基于提交 `5e1b0d3` 的本轮工作树，host 门禁为 `170 passed, 9 skipped, 11 deselected`；设备性能
结论来自归档的 P800 独立 session，而不是在无 P800 的文档主机上重算。

下一页：[P800 设备路径](development/device-stack.md)。
