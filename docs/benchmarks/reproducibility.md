# 原始数据与复现

## 权威数据层级

| 用途 | 文件 |
|---|---|
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
