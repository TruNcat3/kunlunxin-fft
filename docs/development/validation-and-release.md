# 验证与发布

## 默认 host 门禁

```bash
SKIP_TRITON_WARM=1 bash scripts/build.sh
pytest -q
mkdocs build --strict
git diff --check
```

`pytest -q` 默认排除 `hardware` marker，避免不可用设备在 CI 初始化时挂起。

## Hardware 门禁

```bash
CUDA_VISIBLE_DEVICES=6 python3 scripts/smoke.py
CUDA_VISIBLE_DEVICES=6 python3 scripts/test_correctness.py --profile smoke
CUDA_VISIBLE_DEVICES=6 pytest -q -m hardware
```

正式性能变更还需运行 release matrix。候选必须先 correctness，再 timing；随机交错并跨独立进程
session 聚合。失败、unsupported 和 fallback 行必须保留，不能删除后只报告平均值。

## 文档门禁

1. README、结果页和图来自同一个 `summary.json`；
2. 图片存于 `docs/assets/` 并用相对路径，不引用 GitHub `blob` 作为 `<img>`；
3. 所有倍率说明分子/分母和“越大越快”的方向；
4. release 与 Phase 3 协议分开；解析流量不冒充 counter；
5. 浏览器检查桌面/移动导航、图片 200 响应与上一页/下一页。

## Selector 晋升

当前已归档 selector 可从确认数据重建。新 `bench.py` manifest 的协议字段生成还需完善，因此新的
候选表在进入发布前必须人工检查 `timing_scope`、`protocol_id`、profile、source hash 和 guard，
不能通过手工补字段绕过 provenance。

下一页：[配置文件](../reference/configuration.md)。
