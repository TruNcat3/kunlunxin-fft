# 工程结构

```text
config/                     静态设备描述、设计空间、尺寸与 selector 表
include/butterfly/          Host C++ 数学参考
src/host/                   Host 检查程序
src/xpu/                    Triton/XPU kernel、runner、plan 与 selector
scripts/                    环境、构建、正确性、基准、发布和绘图
tests/                      Host 默认测试与显式 hardware 测试
results/p800/               原始实验、session、报告与历史图
docs/                       分层文档与 Pages 资源
```

## 执行路径

`src/xpu/entry.py` 是统一入口：验证语义、查询 selector、验证 lowering、执行 runner，失败则回退
M0。`runner.py` 管理实虚拆分、twiddle、临时 tensor、context 和 kernel launch；`kernels.py` 保存
Triton kernel；`matrices.py` 生成 M2/M3 的 host 常量。

## Host-only 边界

`src/xpu/config.py`、`selector.py`、`entry.py` 的 plan 解析部分不要求 torch，便于普通 CI 验证
配置、闭包和报告。访问 device runner 时通过 `src/xpu/__init__.py` lazy import 厂商栈。

## 结果生成

- `scripts/bench.py`：宽候选研究扫描；
- `scripts/build_selector_table.py`：从经过审查的确认数据生成冻结表；
- `scripts/run_release_matrix.py`：正确性优先、随机交错、多 session 的正式验证；
- `scripts/report.py`：无 torch 的 CSV/Markdown 渲染；
- `scripts/plot_release_docs.py`：从正式 summary 生成 README 图；
- `scripts/plot_results.py`：从 Phase 3 JSON 生成消融图。

下一页：[P800 设备路径](device-stack.md)。
