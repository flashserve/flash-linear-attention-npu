# perf 目录说明（chunk_kda_fwd 性能证据）

比赛任务书要求性能报告放在 `test/atk/chunk_kda_fwd/pref` 路径下——**"pref" 为
"perf" 的笔误**，本目录（`tests/atk/chunk_kda_fwd/perf/`）即对应交付物。

## 内容

- 5 个终版性能 run 目录（本地时间 UTC+8 命名），每个仅含 `report/*.xlsx`（ATK
  正式报告）与 `log/atk.log`（ATK 全量日志，含 `NPU AVG Profiler Time` 原始行）。
- [perf_汇总.md](perf_汇总.md)：五次实测 809.19 / 807.92 / 803.89 / 804.66 /
  810.72 μs 汇总、-38.9% vs 标杆 1323.7 μs、吞吐 ≈1.27M tokens/s、口径说明
  （ATK Device Perf，即 `device_perf(us)` 列 / `NPU AVG Profiler Time`）。
- 终版记录值取 `atk_chunk_kda_fwd_perf_2026-09-28-23-07-28-932050`：809.19 μs
  （std 3.08）。

## 目录名 / xlsx 文件名时差规则

ATK 的 **run 输出目录名用本地时间（本机 UTC+8）**，而 **report xlsx 文件名用
UTC 时间**，两者相差 8 小时。例：

```
atk_chunk_kda_fwd_perf_2026-09-29-02-36-12-491923/            <- run 目录，UTC+8
└── report/atk_chunk_kda_fwd_perf_reports_2026-09-28-18-36-12.xlsx  <- xlsx，UTC
```

`log/atk.log` 内的日志时间戳与 xlsx 同为 UTC。对账时按"目录名 − 8h ≈ xlsx 名"
匹配即可。

## 完整原始数据位置

入仓仅保留 report + log；原始 run 还包含 `profile/npu_npu_dut/...`（profiling
二进制）与空 `mss/` 目录，体积大未入仓，仍在交付机的
`tests/atk/chunk_kda_fwd/atk_output/perf/atk_output/<run 目录名>/` 下（该路径被
`.gitignore` 的 `/tests/atk/**/atk_output/` 规则忽略）。复现方法见
[../README.md](../README.md)。
