# chunk_kda_fwd 性能汇总（Case 250，ATK Device Perf 口径）

- 用例：`atk_chunk_kda_fwd_perf.json` 仅含 case 250（B=1, T=1024, H=96, K=V=128,
  chunk=64, BSND, bf16 q/k/v + fp32 g/beta），与任务书模型 case 一致。
- 指标：`device_perf(us)`（NPU AVG Profiler Time），即 ATK 对 device 侧 profiler
  采样的平均值；每次 run 独立 warmup 后重复采样，std_deviation 为样本标准差。
- 标杆：任务书给定 **1323.7 μs**。

## 五次终版实测（2026-09-28/29，Ascend 950PR，CANN 9.1.0-beta.3）

| # | run 目录（本地时间 UTC+8） | device_perf (μs) | std | vs 标杆 |
| - | -------------------------- | ---------------- | --- | ------- |
| 1 | `atk_chunk_kda_fwd_perf_2026-09-28-23-07-28-932050` | **809.19** | 3.08 | -38.9% |
| 2 | `atk_chunk_kda_fwd_perf_2026-09-29-02-36-12-491923` | 807.92 | 2.95 | -39.0% |
| 3 | `atk_chunk_kda_fwd_perf_2026-09-29-02-36-34-295082` | 803.89 | 3.00 | -39.2% |
| 4 | `atk_chunk_kda_fwd_perf_2026-09-29-02-37-00-542077` | 804.66 | 3.05 | -39.2% |
| 5 | `atk_chunk_kda_fwd_perf_2026-09-29-03-23-29-117715` | 810.72 | 2.61 | -38.8% |

- 五次均值 ≈ 807.3 μs，极差 6.8 μs（<0.9%），波动稳定。
- **终版结论取 run #1（交付记录值）：809.19 μs（std 3.08），相对标杆 1323.7 μs
  提升 -38.9%。**

## 吞吐换算

- Case 250 单次处理 token 数 = B×T = 1024。
- 809.19 μs → 1024 / 809.19 μs ≈ **1.27M tokens/s**。
- 标杆口径：1323.7 μs → 1024 / 1323.7 μs ≈ **773.6K tokens/s**（约 0.77M）。

## 说明

- 每个子目录为一个完整 ATK run，仅保留 `report/*.xlsx` 与 `log/atk.log`
  （原始 run 还含 `profile/` profiling 数据与 `mss/`，体积大不入仓；
  完整原始数据见 `tests/atk/chunk_kda_fwd/atk_output/perf/atk_output/<run>/`）。
- 目录名/xlsx 文件名时差规则见本目录 [README.md](README.md)。
- 复现命令见上级 [../README.md](../README.md) "性能测试" 一节。
