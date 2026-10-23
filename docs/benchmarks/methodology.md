# 测量协议

## v6-1 全语义冻结协议

Issue #9 v6-1 是全语义迁移验收的权威冻结快照。它使用 clean worktree、路径身份和 selector SHA
一致性检查、artifact ledger、独立 correctness、多个 session 以及 release gate；最终门禁为
`41/41 PASS`（`469` host、`370` hardware）。其 headline 是 current/M0 的 140-cell 几何平均
延迟比 `0.6086x`，而不是旧 release matrix 的单点表。

相对 vendor 的 `4/0/102` 和 `0.122x` 只统计 106 个 vendor-supported、语义匹配 cell；
unsupported、未实现的 resident pipeline 和未实现的第三方 core 不得补成性能数字。详细证据和
release 路径见 [v6-1 验收页](issue9-v61.md)。

Issue #10 / PR #12 采用更窄的 16-cell、5-path、双 session holdout，额外要求候选相对 vendor
和实际 profile-bound incumbent 的 bootstrap speedup 置信区间下界均大于 `1.02`。这套协议用于
能力门禁和 promotion evidence，不替代 v6-1，也不能与其拼接平均；见
[Issue #10 能力与加速证据](issue10-capability.md)。

## Release matrix 合同

| 项目 | 值 |
|---|---|
| 硬件 | 单张 Kunlunxin P800 OAM；12 clusters x 64 cores；256 KiB shared；96 GiB HBM |
| Driver / device | `515.58`；记录设备号 6 |
| Profile | `a78b86793c7142a0` |
| 工具链 | torch `2.9.0+cu129` + torch_xmlir；XPYTORCH `XMLIR--bc1b1dc6f-dev+2026032411`；XRE `5.13.0.0` |
| 语义 | FP32/complex64、forward C2C、DIT、自然顺序、无归一化 |
| 计时 | API wall time，包含 dispatch、分配、内部布局和同步；plan 已复用 |
| 调度 | 候选随机交错；3 个独立进程 session |
| 采样 | 实际 warmup 3；主点每候选 30 trials；N=1M 与 profile mismatch 10 trials |
| 正确性 | NumPy complex128；componentwise peak-normalized error `<1e-4` |

发布目录共有 3492 条 raw rows。session 内的归档摘要使用上中位数；三个 session 的发布汇总使用
标准中位数。`speedup_median` 是 session-level ratio 的中位数，不从显示精度截断后的 latency 重算。

## 三类时间不能混用

1. **API wall time**：正式 headline，包含 runner 可见的 dispatch/alloc/layout/sync；
2. **kernel/device time**：Phase 3 部分行同时保存，仅用于归因；
3. **one-shot `run_fft`**：还包含每次 plan lookup，必须单列。

把 H2D/D2H、plan 或布局移出计时区间可以回答另一个问题，但不能称为原协议下 kernel 加速。

## 基线匹配

内部 incumbent 是同 shape、direction 和 block 的 M0。候选必须先通过正确性，再满足 search-stage
median 至少 2% 的工程 guard 才能晋升。2% 不是置信区间或统计显著性。

外部 baseline 是记录 xpytorch 栈中的 `torch.fft.fft`。只有相同 `(B,N)`、dtype、forward C2C
成功执行的行进入比较；unsupported 保留错误而不填补数字。当前数据只覆盖 N=1024/2048、
B=1/16/64 的正式 matched matrix，不能外推到其他版本或语义。

## Phase 3 合同

Phase 3 使用 block=1024、warmup=10、trials=50，commit `12965bca` 且 `dirty=true`。它覆盖
18 个长度和更宽候选，适合趋势/消融。其 GM bytes 与 launch counts 是实现结构推导的解析账本，
不是硬件 counter；不能标成实测带宽或 sector。

下一页：[C2C 性能](c2c.md)。
