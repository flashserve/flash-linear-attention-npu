# accuracy 目录说明（chunk_kda_fwd 精度证据）

比赛任务书要求精度报告放在 `test/atk/chunk_kda_fwd/accuracy` 路径下，本目录即
对应交付物。

## 结论：48/48 Pass

- 48 个精度用例（id 250~297，6 个 T 档 × 8 变体）在 `mixed_tolerance_bm`
  （FP64 CPU golden vs NPU DUT）下**全部通过**，逐例明细见
  [精度汇总_48cases.md](精度汇总_48cases.md)。
- **case 297（T=16384 export）** 的通过 run 是
  `atk_chunk_kda_fwd_2026-09-29-13-30-13-931570/`：其 fp64 金标经
  `gen_kda_golden_stream.py` 流式预计算后由
  `executor_chunk_kda_fwd_precomputed.py` 加载（见 ../README.md "case 297
  预计算金标" 一节），ATK 正式判定 acc_pass_result: Pass（summary 页 总用例
  1 / 成功 1 / 通过 1 / 通过率 100%）。该 run 的 xlsx 时间戳为
  `2026-09-29-05-30-14`（UTC），与 `精度汇总_48cases.md` 表中 idx 47 一行对应。

## 目录内容

- 31 个完整 accuracy run 目录（本地时间 UTC+8 命名），每个仅含
  `report/*.xlsx` 与 `log/atk.log`。同一用例多份报告时以最新结果为准
  （见汇总表"报告"列）。
- `determinism/`：DC 确定性回归 run（`atk_chunk_kda_fwd_mss_2026-09-29-02-15-24-182521`，
  report + log），复跑 MSS 用例逐位比较，判定 `is_acc_dc_pass: Pass`。
- `精度汇总_48cases.md`：48 例逐例结果表（idx / id / T / 变体 / 精度 / 结果 /
  报告时间戳）。
- **不含任何 golden npy**（金标数据 GB 级，不入仓；如需重算金标见
  ../README.md）。

## run 目录名 ↔ xlsx 时间戳时差规则

与 perf 目录相同：ATK **run 输出目录名用本地时间（交付机 UTC+8）**，**report
xlsx 文件名与 `log/atk.log` 内日志时间戳用 UTC**，相差 8 小时。例：

```
atk_chunk_kda_fwd_2026-09-29-13-30-13-931570/                      <- run 目录，UTC+8
├── report/atk_chunk_kda_fwd_reports_2026-09-29-05-30-14.xlsx       <- xlsx，UTC
└── log/atk.log                                                     <- 日志时间，UTC
```

因此 `精度汇总_48cases.md` 表中的报告时间戳（UTC，取自 xlsx 文件名）与 run
目录名（UTC+8）对账时，按"目录名 − 8h ≈ 报告时间戳"匹配。

## 完整原始数据位置

`atk_output/accuracy/atk_output/` 下另有若干**无 xlsx 的中断 run**（仅有 log，
无 report）与 profiling 数据，均未入仓；原始数据仍在交付机的
`tests/atk/chunk_kda_fwd/atk_output/` 下（被 `.gitignore` 忽略）。复现方法见
[../README.md](../README.md)。
