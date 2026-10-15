# 原始数据与复现

## 权威数据层级

| 用途 | 文件 |
|---|---|
| 正式 headline | `results/p800/release_matrix/summary.json`、`summary.csv`、`server_report.md` |
| Session 与 raw trace | `summary_session{0,1,2}.json` 及 release 目录中的原始记录 |
| 宽范围消融 | `results/p800/bench_phase3.json`、`bench_phase3_long.csv` |
| 冻结表来源 | `confirm_b1024.json`、`confirm_b256.json`、`confirm_batch.json` |
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

参数名当前是 `--batchs`。新 `bench.py` manifest 尚未自动包含 table builder 强制要求的
`timing_scope` 和 `protocol_id`，所以新 JSON 不能在未经协议修复/审查时直接晋升为发布 selector。

## 重建已归档 frozen table

```bash
python3 scripts/build_selector_table.py \
  results/p800/confirm_b1024.json \
  results/p800/confirm_b256.json \
  --batch results/p800/confirm_batch.json \
  --out /tmp/selector_table.json
```

比较 `/tmp/selector_table.json` 与 `config/selector_table.json` 前，先核对 source manifest、profile
hash 和 guard。不要手工伪造缺失协议字段。

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
