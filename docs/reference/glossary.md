# 术语

| 术语 | 含义 |
|---|---|
| M0 | radix-2 global-stage incumbent，也是运行时闭包 fallback |
| M1 | radix-2 fused-inplace local tile |
| M2 | dense-dot FFT64 + matrix-resident lowering 的联合候选 |
| M3 | composite linear-map grouped boundary |
| M4 | radix-2 batch spread/folded 数据映射 |
| M5 | 与 M0 同类 global-stage flow 的 radix-4 核心 |
| `Us/Ts` | 阶段维度的空间展开 / 时间复用 |
| `Ud/Td` | 数据维度的空间展开 / 时间复用 |
| Processing unit | 局部蝶形/FFT 算术核心，不等同于整个架构映射 |
| Lowering contract | 核心与流映射落到 P800 指令、shape、布局、memory 和同步的合法性条件 |
| Residence | 中间状态驻留在寄存器/局部存储或物化到 GM 的方式 |
| Plan reuse | 一次 `plan_fft` 后重复 `execute_fft_plan` |
| Frozen selector | 由已测确认数据生成的 exact-key 表，不在线 benchmark/插值 |
| M0 closure | 未知或非法候选回到 M0，保证实现空间包含基线 |
| API wall time | runner 可见的 dispatch、分配、布局和同步总时间 |
| Matched vendor point | 相同 shape/dtype/方向且厂商调用实际成功的比较点 |

下一页：[未来计划](../roadmap.md)。
