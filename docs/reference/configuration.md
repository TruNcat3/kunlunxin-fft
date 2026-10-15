# 配置文件

| 文件 | 作用 | 不能代表什么 |
|---|---|---|
| `config/device.json` | 静态 P800 设备/工具链描述 | 实时硬件探测或自动校准 |
| `config/butterfly_space.json` | A/P/L/G/F 研究空间与当前 lowering 注释 | 每个声明值都有生产 kernel |
| `config/sizes.json` | correctness/benchmark 长度与 block 列表 | 所有点都进入 selector |
| `config/test_matrix.json` | smoke/full 正确性 suite 声明 | 脚本已经执行所有列出的 pattern |
| `config/selector_table.json` | 已确认 shape 的冻结运行表 | 表外插值或跨硬件模型 |

`DeviceProfile.hash()` 对 `device.json` 的规范化内容做哈希，用于防止错误表静默命中。它不查询
`xpu-smi`，也不检测当前 driver、设备数或时钟。

设计空间沿用 cuButterfly 的分层：A 是架构/流映射，P 是 processing unit，L 是 layout，G 是
twiddle/系数策略，F 是变换结构，H 是硬件 envelope。JSON 中历史描述若与当前 runner 冲突，以
代码、测试和[支持范围](support.md)为准，并应作为后续修正文档/配置的 issue。

下一页：[支持范围](support.md)。
