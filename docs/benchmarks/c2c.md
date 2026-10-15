# C2C 性能

本页只讨论当前已发布语义：FP32/complex64、1D、forward C2C、自然顺序。完整结果和限制见
[实验设计与结果汇总](results.md)，计时合同见[测量协议](methodology.md)。

## 正式代表点

| N | B | 选择路径 | 相对同 block M0 | 相对记录 vendor |
|---:|---:|---|---:|---:|
| 16 | 1 | M1 fused | 1.494x | 0.152x |
| 1024 | 1 | M2 matrix | 1.252x | 0.682x |
| 2048 | 1 | M2 matrix | 1.232x | **1.660x** |
| 4096 | 1 | M2 matrix | 1.308x | n/a |
| 65536 | 1 | M5 radix-4 | **2.072x** | n/a |
| 1048576 | 1 | M5 radix-4 | **1.868x** | n/a |

`相对 M0` 是独立 session 的 speedup 中位数；vendor 只在同 shape、同精度和同计时语义实际
成功时记录。表外或 profile 不匹配会回退 M0，不从邻近尺寸外推。

## 读法

- 小长度的 M1 主要节省 launch 和早期边界；长度增大后 barrier/store-load 链可能反转收益。
- 中等长度的 M2 是 dense-dot 核心与局部驻留的联合效果。
- 大长度的 M5 固定 global-stage flow，只替换 radix-4 核心，适合做 core A/B。
- 目前 batch 点主要走 M4/M0-shaped global-stage 路径，不能拿单变换结果外推 batch 吞吐。

宽范围交叉点见[消融](ablations.md)，原始文件和重跑命令见[复现](reproducibility.md)。

下一页：[端到端与应用](end-to-end.md)。
