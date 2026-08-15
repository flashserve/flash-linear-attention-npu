# PR #283 代码修改与性能验证记录

验证时间: 2026-08-13
本地仓库: `/home/m00913889/codex04/bingli/flash-linear-attention-npu`
测试脚本: `/home/m00913889/codex04/bingli/test-perf.py`
编译命令: `FLA_NPU_SOC=ascend910b python -m pip wheel --no-build-isolation --no-deps . -w dist`

## 本轮代码修改

| 文件 | 修改点 | 修改前 | 修改后 | 目的 |
| --- | --- | --- | --- | --- |
| `chunk_scaled_dot_kkt_common.h` | row block 常量拆分 | `SCORE_ROW_BLOCK=16` | `SCORE_ROW_BLOCK_A2=16`, `SCORE_ROW_BLOCK_A5=64` | A5/950 BT=64 使用整块 score，A2/910B 保持稳定路径 |
| `chunk_scaled_dot_kkt.h` | kernel row-block 选择 | 所有架构统一用 16 | `__CCE_AICORE__==310` 用 64，否则用 16 | 让 950 满足 `ProcessKktEpilogue64VF` 命中条件 |
| `chunk_scaled_dot_kkt_tiling.cpp` | host row-block 计数 | host 固定按 16 计算 score block 数 | host 先读取 SOC，再按 950=64、其他=16 计算 | 保持 host tiling 与 kernel 任务解码一致 |

未保留的尝试:

| 尝试 | 结果 | 结论 |
| --- | --- | --- |
| 910B 启用 AtlasA2 Catlass + row-block 64 | 默认 KKT 运行触发 AICore 异常，L0B read/write conflict | 不可保留 |
| 910B 启用 AtlasA2 Catlass + row-block 32 | 小 case 即触发 AICore 异常，L0B read/write conflict | 不可保留 |
| 910B 启用 AtlasA2 Catlass + row-block 16 | 可运行，但默认 KKT 约 23.091 ms，和 fallback 基本一致 | 无收益，不保留 |

## 编译结果

| 项 | 结果 |
| --- | --- |
| wheel | `dist/flash_linear_attention_npu-26.7.0.dev0-910b.aarch64-py3-none-any.whl` |
| sha256 | `e7b33b795c0a9c65589d6db4f07cc519caadf0df3ea8da0f5c629068ab5e6822` |
| 安装验证 | `python -m pip install --force-reinstall --no-deps dist/flash_linear_attention_npu-26.7.0.dev0-910b.aarch64-py3-none-any.whl` 成功 |

## `test-perf.py` 端到端结果

环境说明: 当前 910B3 机器所有 NPU 均有占用，NPU 7 仍有 `npu-smi` 可见的约 31GB 残留占用；Python 端均值受队列、内存分配、aclnn 两阶段调用和设备状态影响较大。

默认 case: `TND, B=1, H=32, T=65536, avg_seq=1024, BT=64, K=128, iters=10`

| backend | cumsum avg | kkt avg | 结论 |
| --- | ---: | ---: | --- |
| AscendC | 5.061 ms | 23.512 ms | Python 端 kkt 明显慢于 Triton |
| Triton | 1.727 ms | 4.639 ms | Python 端基准 |

说明: 早前同机低占用时 AscendC cumsum 跑到 0.742 ms；当前 5.061 ms 说明 Python 端结果存在较明显环境波动。

## msprof kernel 结果

采集命令统一使用:

```bash
msprof --output=<dir> --ascendcl=on --task-time=on --ai-core=on \
  --aic-mode=task-based --aic-metrics=PipeUtilization \
  --application="/usr/bin/env ASCEND_RT_VISIBLE_DEVICES=7 ... python /home/m00913889/codex04/bingli/test-perf.py"
```

| 采集项 | op_summary | 稳态 Task Duration | 关键指标 |
| --- | --- | ---: | --- |
| AscendC cumsum | `logs/pr283_perf/20260813T081000Z_cumsum_pipe/PROF_000001_20260813080612982_00487759JFCEMICC/mindstudio_profiler_output/op_summary_20260813080626.csv` | 43.31 us | AIV vec/scalar 为主 |
| AscendC kkt | `logs/pr283_perf/20260813T081100Z_kkt_pipe/PROF_000001_20260813080709076_00488552INOCGLHQ/mindstudio_profiler_output/op_summary_20260813080723.csv` | 2390.02 us | cube_utilization 稳态约 98%，AIC/AIV 时间约 2.35 ms |
| Triton cumsum | `logs/pr283_perf/20260813T081600Z_triton_both_pipe/PROF_000001_20260813081541837_00493440KRIDKDMJ/mindstudio_profiler_output/op_summary_20260813081602.csv` | 876.32 us | `chunk_local_cumsum_scalar_kernel` |
| Triton kkt | 同上 | 2815.625 us | `chunk_scaled_dot_kkt_fwd_kernel` |

kernel 口径结论:

| 算子 | AscendC kernel | Triton kernel | AscendC/Triton |
| --- | ---: | ---: | ---: |
| cumsum | 43.31 us | 876.32 us | 0.049x |
| kkt | 2390.02 us | 2815.625 us | 0.849x |

因此从 msprof kernel task duration 看，cumsum 和 kkt 的 AscendC kernel 都没有比 Triton 慢，kkt 约为 Triton 的 84.9%。`test-perf.py` Python 端 kkt 约 23.5 ms 的问题主要不在单个 kernel task duration，而在 AscendC Python ctypes/aclnn wrapper 的两阶段接口、descriptor/workspace 创建、输出分配和调度等待等端到端开销。

## 后续建议

1. 950/A5 上继续围绕 reviewer 的 CV stage、score workspace direct-score、AIV resident g/beta cache 做结构性优化。
2. 910B 不建议在本 PR 中启用当前 Catlass path；row-block 32/64 会触发 L0B conflict，row-block 16 无性能收益。
3. 若验收关注 Python 端耗时，需要单独优化 `fla_npu.ops.ascendc` 的 `_call_aclnn` 调用链，复用 executor/workspace 或提供 benchmark 专用直调路径。
