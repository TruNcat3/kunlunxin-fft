# 原始数据与复现

## 复核 Issue #9 v6-1

v6-1 的结果不是从 README 手工抄表，而是由 release manifest、三组 session、路径身份和 artifact
ledger 共同冻结。已有快照可直接审阅：

```text
results/p800/releases/20261020-issue9-semantic-matrix-v6-1/
results/p800/issue9_v61/
```

在同一 P800/XPU 环境重新采集时，使用 release 中的
[`repro_command.txt`](https://github.com/TruNcat3/kunlunxin-fft/blob/master/results/p800/releases/20261020-issue9-semantic-matrix-v6-1/repro_command.txt)，
不要自行删减 cell 后把结果称为 v6-1。命令会生成三 session、warmup/trial、one-shot/E2E、
oracle 模式和 release id；新 release id 必须唯一，且通过严格 manifest/artifact gate 后才能
进入文档。无 P800 或 vendor runtime 时，只能审阅已有快照，不能用 host-only 结果替代硬件结论。

## 权威数据层级

| 用途 | 文件 |
|---|---|
| Issue #9 v6-1 权威快照 | `results/p800/releases/20261020-issue9-semantic-matrix-v6-1/`；路径身份与 ledger 在 `results/p800/issue9_v61/` |
| 正式 headline | `results/p800/release_matrix/summary.json`、`summary.csv`、`server_report.md` |
| Session 与 raw trace | `summary_session{0,1,2}.json` 及 release 目录中的原始记录 |
| 宽范围消融 | `results/p800/bench_phase3.json`、`bench_phase3_long.csv` |
| 冻结表来源 | `confirm_b1024.json`、`confirm_b256.json`、`confirm_batch.json` |
| Batch focus 与 holdout | `batch_focus_{a,b}/summary.json`、`batch_holdout/summary.json` |
| 模型验证 | `model_validation.json`（focus guard 与 holdout recall/regret 分池） |
| 外部核心组合 | `four_cell.json`（历史单进程）、`hybrid_canary.json`（多进程主证据） |
| 历史数据 | Phase 2 与 `selected_sessions/`；不用于当前 headline |

## 运行 release matrix

```bash
mkdir -p results/runs/local/release_matrix
CUDA_VISIBLE_DEVICES=6 python3 scripts/run_release_matrix.py \
  --sessions 3 --trials 30 --warmup 10 \
  --out results/runs/local/release_matrix
```

脚本中的发布 manifest 会记录实际执行 warmup。归档快照因早期实现曾硬编码 3，最终 manifest 已
纠正为实际值 3。新运行使用当前脚本参数，不应为追求数值一致而手改输出。

已有 session 的离线聚合：

```bash
python3 scripts/run_release_matrix.py --aggregate-only \
  --out results/runs/local/release_matrix
```

聚合读取 `summary_session*.json`，并不从所有 raw row 重新构建每项 session 内统计。

## 运行宽范围 benchmark

```bash
mkdir -p results/runs/local
CUDA_VISIBLE_DEVICES=6 python3 scripts/bench.py \
  --sizes 1024,2048,4096 --batchs 1,16,64 \
  --trials 30 --warmup 10 \
  --json results/runs/local/bench.json \
  --csv-long results/runs/local/bench.csv
```

参数名当前是 `--batchs`。新 `bench.py` 运行会自动补齐 WP0 契约要求的
`env_hash` / `toolchain` / `sessions` / `candidates` / `shapes` 等字段，但归档的
`confirm_*.json`（契约前产物）仍缺这些字段——它们可以继续作为历史证据引用，
但不能直接晋升为发布 selector。

## 安装期一键 bootstrap（WP0）

```bash
CUDA_VISIBLE_DEVICES=6 python3 scripts/bootstrap_profile.py \
  --device 6 --output results/local-profile \
  --calibrate --verify --promote
```

阶段化流水（每阶段日志在 `<output>/logs/`，状态在 `<output>/state.json`）：

1. `probe` — `hw_probe.py` 生成独立 calibration（profile/env hash）；
2. `check` — `smoke.py` + `test_correctness.py` 门；
3. `calibrate` — confirm / focus / holdout 代表性测量轮（需 `--calibrate`），
   完成后给每个 staged manifest 盖 `source_hash` / `result_sha256`；
4. `build` — `validate_cost_model.py` + `build_selector_table.py`（含
   `--verify-live-env`）；
5. `verify` — `manifest_contract` 严格校验（缺字段/重复 trial/缺 case
   汇总报告）+ selector smoke + 环境失配 fail-closed（需 `--verify`）。

全程先写 `<output>/.staging/`；失败保留 staging 供 `--resume` 跳过已完成阶段。
**只有显式 `--promote` 才会**用原子 `os.replace` 发布到 `config/selector_table.json`，
否则生产表字节不变。

## WP2 统一语义发布矩阵

```bash
CUDA_VISIBLE_DEVICES=6 python3 scripts/run_semantic_matrix.py \
  --sessions 3 --trials 30 --warmup 5 --scope all --snapshot \
  --release-id <id> --out results/p800/semantic_matrix
```

同一份 manifest 驱动 raw（`raw_session*.jsonl`）、summary（每 case 含
median + MAD 离散度 + `sessions_with_data` 独立 session 数 + 失败状态）、
`summary.csv` 表格、`speedup_vs_vendor.png` 图与不可变快照。每个
case × semantic（c2c forward/inverse、r2c、c2r）记录四类 contender：M0/M4 incumbent、
production selector、bounded measured oracle、vendor（不支持/报错原样保留，不进
speedup）。计时分列 plan_creation / plan-reuse steady / one_shot / e2e 四层，绝不
把 API wall 写成 kernel time。指标：win/tie/loss（2% 阈值）、几何均值、最差 speedup、
selector regret（selector/oracle−1）、相对 incumbent 的 2% guard 结果。

session 模型：每个 session 由父进程 `subprocess` **顺序** exec 一个全新解释器
（`--session N`，结束后 `os._exit`），case/semantic 顺序按 session 种子随机交错；
同一设备绝不并发两个 session（并发会同时抬高所有 contender 的表观时间，污染整个矩阵）。
`multiprocessing.Pool` 会在池拆除时与设备 runtime 的监听线程死锁，不使用。

`--snapshot` 生成 `results/p800/releases/<release-id>/`（manifest、
environment、gzip 后的 `raw_session*.jsonl.gz`（解压后与工作目录 raw 逐字节
一致，manifest 以未压缩 rows 计 `result_sha256`）、`summary.json`、
`summary.csv`、repro 命令、失败分类）；已存在的 release-id 拒绝覆盖，快照
manifest 过 `validate_manifest(strict=True)`。工作目录的
`raw_session*.jsonl` 为可再生产物，不入库。

## 组合表晋升与失败 cell 诊断（issue #6 PR-E）

```bash
# 1) 独立进程 canary：3 protocol session + 自动 confirmation session
CUDA_VISIBLE_DEVICES=6 python3 scripts/run_hybrid_canary.py \
  --shapes 1024x16,1024x64,2048x16 --sessions 3 --trials 30 --warmup 5 \
  --out results/p800/composition_confirm.json

# 2) 证据 -> 原子晋升（复验 guard/confirmation/pid/commit/env 绑定；
#    负点拒绝；写 config/composition_table.json）
CUDA_VISIBLE_DEVICES=6 python3 scripts/promote_composition.py \
  --evidence results/p800/composition_confirm.json --shape 2048x16

# 3) 失败 cell 有界诊断（自动挑最差 K 个可操作 cell，或 --cells 显式指定）
CUDA_VISIBLE_DEVICES=6 python3 scripts/diagnose_failing_cells.py --top 4
```

晋升规则与 fail-closed 语义见
[外部核与组合表](external-cores.md#5-canary-exact-shape-composition-table)。
诊断输出 `results/p800/diagnostics/<date>/{verdicts.json,report.md}`：每级 verdict
记录触发它的数字与证据类别（`measured` / `measured-artifact` / `summary-derived` /
`analytic-model`），ladder 顺序固定（correctness → external-capability →
entry-dispatch → sync/handoff → worker-waves → flow-not-selected →
processing-unit-bound → mixed）；dispatch probe 只对 c2c-forward 做（real-compat 与
epilogue prep 是真实工作，不计入 dispatch），ledger 是解析模型（bench 公式）且明标
`estimate`，两者都**不是**硬件性能计数器。

## 重建已归档 frozen table

```bash
python3 scripts/build_selector_table.py \
  results/p800/confirm_b1024.json \
  results/p800/confirm_b256.json \
  --batch results/p800/confirm_batch.json \
  --focus results/p800/batch_focus_a/summary.json \
          results/p800/batch_focus_b/summary.json \
  --out /tmp/selector_table.json
```

focus artifact 是当前 `env_hash` 的绑定来源；不带 `--focus` 的旧 confirm-only 命令会因无法构建
环境绑定表而 fail closed。比较 `/tmp/selector_table.json` 与 `config/selector_table.json` 前，先
核对 source manifest、profile hash、env hash 和 guard。不要手工伪造缺失协议字段。

## 重建 batch/模型报告

```bash
python3 scripts/validate_cost_model.py \
  results/p800/batch_focus_a/summary.json \
  results/p800/batch_focus_b/summary.json \
  results/p800/batch_holdout/summary.json \
  --out results/p800/model_validation.json

python3 scripts/gen_focus_report.py \
  --focus results/p800/batch_focus_a results/p800/batch_focus_b \
  --holdout results/p800/batch_holdout \
  --validation results/p800/model_validation.json \
  --four-cell results/p800/four_cell.json \
  --hybrid-canary results/p800/hybrid_canary.json \
  --out docs/benchmarks/batch-focus.md
```

## 重建文档图

```bash
python3 scripts/plot_release_docs.py
python3 scripts/plot_results.py \
  --json results/p800/bench_phase3.json \
  --block-json results/p800/bench_block256.json \
  --block-json results/p800/bench_block512.json \
  --block-json results/p800/bench_block2048.json \
  --out results/p800/plots
```

第一条生成当前 README/release 图；第二条生成 Phase 3 研究图。发布时应把对应图片复制到
`docs/assets/figures/`，并用相对路径引用，避免 GitHub `blob` URL 在 Pages 中返回 HTML/404。

下一页：[相关工作与基线](related-work.md)。
