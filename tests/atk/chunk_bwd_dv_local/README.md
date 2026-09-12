# ChunkBwdDvLocal ATK 工程

本目录提供 `chunk_bwd_dv_local` 的 ATK 单算子工程，包含 `executor_chunk_bwd_dv_local.py`、`gen_chunk_bwd_dv_local.py`、`chunk_bwd_dv_local.yaml`、`atk_chunk_bwd_dv_local.json`。

## 输入约束

- `q/k` 必须为 `[B,H_qk,T,K]`，且二者形状完全一致。
- `dO/out` 必须为 `[B,H_do,T,V]`，`g` 必须为 `[B,H_do,T]`。
- `q/k` 与 `dO/g/out` 的 `B`、`T` 必须一致；`H_do` 必须能被 `H_qk` 整除。
- `K` 固定为 `128`，`V` 支持 `128/256`，`chunk_size` 仅支持 `64/128`。
- `q/k/dO/out` 支持 `BFLOAT16/FLOAT16`；`g` 支持 `FLOAT/FLOAT16/BFLOAT16`。
- `g` 为 chunk 内累计 log-decay，每个序列的每个有效 chunk 内沿 T 维单调不增；非单调 `g` 不在支持范围内。
- `gGammaOptional` 和 `aOptional` 当前未启用，必须传 `None`；变长模式下 `cu_seqlens` 与 `chunk_indices` 必须同时提供且 `B=1`。
- 当前 ATK 用例遵循上述约束，并通过 `case_spec` 固定具体取值；扩展用例时应继续满足这些限制。

## 标杆来源

torch_custom/fla_npu/test/golden.py; fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dv_local/README.md

CPU 标杆、输入构造、run_cpu、run_npu 和 FunctionApi 均在本目录的 `executor_chunk_bwd_dv_local.py` 中实现；公共文件只提供基础工具函数。

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93` 和 `ascend950`，可配合统一脚本的 `-soc=ascend910b|ascend910_93|ascend950` 使用。

## 用例矩阵

- 精度 JSON 包含 603 条 `(case, seed)` 组合：200 个原始场景各使用 3 个固定 seed，另含 3 条 Issue #462 高 `hRatio` 直接回归。
- 每个原始场景的 seed 集合为 `{base_seed, base_seed + 100000, base_seed + 200000}`；shape、dtype、属性、有效区域和值域在 3 个 seed 间保持不变。
- 603 条组合中 BF16 300 条、FP16 303 条，定长 579 条、变长 24 条。
- `g` 覆盖与主输入相同的 BF16/FP16，以及独立 FP32；`V` 覆盖 128/256，`chunk_size` 覆盖 64/128。
- shape 覆盖 `B=1/2`、`H_qk=1/2/4/15`、`H_do=1/2/4/8/120`、`T=1/64/128`，head ratio 覆盖 1/2/8。
- case 600-602 固定为 `B=1, H_qk=15, H_do=120, T=1, K=128, V=128, chunk_size=64, dtype=FP16, g_dtype=FP32, hRatio=8`，覆盖 Issue #462 的同步计数边界。
- 变长用例由 executor 根据 `cu_seqlens_len` 构造 `cu_seqlens` 和 `chunk_indices`，并分别传入 CPU 标杆和 NPU DUT。
- `_mss.json` 包含 16 条精简用例，用于确定性和内存检测；`_perf.json` 包含 8 条性能用例。
- 三份 JSON 均使用仓库统一的 `mixed_tolerance_bm` 精度标准。

## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_bwd_dv_local -npu_device_id=6
bash tests/atk/run_test_cpu.sh -op=chunk_bwd_dv_local -npu_device_id=6 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=chunk_bwd_dv_local -npu_device_id=6 -scope=performance
bash tests/atk/run_test_cpu.sh -op=chunk_bwd_dv_local -npu_device_id=6 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=chunk_bwd_dv_local -npu_device_id=6 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=chunk_bwd_dv_local -scope=gen_cases
```

`gen_cases` 默认传入 `-dt 100 -en 0`。所有新增工程的 marker dtype 都保留两路生成入口，生成器会把不支持 FP16 的算子改回合法 BF16 用例。

## 精度验收记录

- 验收日期：2026-09-06。
- 测试标准版本：`b167e679`，使用 `mixed_tolerance_bm`、CPU FP64 单标杆和 GM 初始化；ATK 版本为 `26.8.8`。
- 目标 SoC：A2（`ascend910b`）。
- 执行范围：不设置 case 范围，执行 `atk_chunk_bwd_dv_local.json` 中全部 603 个 `(case, seed)` 组合。
- 结果：总用例 603，执行成功 603，执行失败 0，精度通过 603，通过率 100%，精度结论为 Pass。
- 覆盖结论：定长/变长、BF16/FP16 主输入、BF16/FP16/FP32 gate、`V=128/256`、`chunk_size=64/128`、`hRatio=1/2/8` 均通过；Issue #462 的 case 600-602 三个固定 seed 均执行完成并通过精度比较。
