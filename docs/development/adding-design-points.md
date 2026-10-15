# 增加设计点

新增候选必须沿着 Ascend-FFT 和 cuButterfly 采用的顺序推进：先定义语义，再确认硬件 lowering，
最后测量并进入 selector。只扩展 JSON 列表不算完成。

## 四步流程

1. **声明候选**：在 `src/xpu/candidates.py` 中记录 flow、processing unit、layout、residence、
   batch 和参数含义；未实现轴返回 `not-lowered`。
2. **实现 runner/kernel**：在 `src/xpu/runner.py` 和 `src/xpu/kernels.py` 消费每一个会影响执行的
   参数，并记录 launch、边界和资源假设。
3. **加入正确性测试**：覆盖目标长度、DIT/DIF、batch、重复调用、错误输入和边界 shape；编译器
   新形式先做最小 canary。
4. **生成证据与选择**：使用同 shape/block 的 M0 incumbent、随机交错、多 session 测量；只有协议、
   profile、误差和工程晋升门限都满足时才更新 `config/selector_table.json`。

## 归因规则

- 改 flow 时固定 processing unit；
- 改 processing unit 时固定 flow；
- 核心和 residence 同时改变的 M2 标为联合效果；
- `Td` 必须说明是 worker 内 token 循环还是单纯 grid 索引；
- GM bytes 没有硬件 counter 时只能写“解析估算”。

## 推荐检查

```bash
SKIP_TRITON_WARM=1 bash scripts/build.sh
pytest -q
mkdocs build --strict
git diff --check
```

设备上再运行 smoke、目标 correctness 和 release/diagnostic 脚本。发生编译失败时先看
[P800 设备路径](device-stack.md)；发生选择退化时看[选择器](../design/selector.md)和
[测量协议](../benchmarks/methodology.md)。

下一页：[测试策略与扩展矩阵](../testing.md)。
