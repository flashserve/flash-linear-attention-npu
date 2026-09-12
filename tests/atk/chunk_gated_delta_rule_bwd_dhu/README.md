# ChunkGatedDeltaRuleBwdDhu ATK 工程

本目录提供 `chunk_gated_delta_rule_bwd_dhu` 的 ATK 单算子工程，包含 `executor_chunk_gated_delta_rule_bwd_dhu.py`、`gen_chunk_gated_delta_rule_bwd_dhu.py`、`chunk_gated_delta_rule_bwd_dhu.yaml`、`atk_chunk_gated_delta_rule_bwd_dhu.json`。

## 输入约束

- `q/k` 必须为 `[B,HK,T,K]`，且二者形状完全一致。
- `w` 必须为 `[B,HV,T,K]`；`dO/dv/dv2` 必须为 `[B,HV,T,V]`，且 `dO` 与 `dv` 形状完全一致。
- `g` 与 `gk` 必须二选一：`g=[B,HV,T]`，`gk=[B,HV,T,K]`；门控 dtype 需要为 `FLOAT` 或与 `q/k` 一致。
- `h0/dht` 如提供，形状为 `[B,HV,K,V]`；`dh` 输出为 `[B,HV,NT,K,V]`。
- `q/k` 与 `w/dO/dv/g` 的 `B`、`T` 必须一致；`HV % HK == 0`。
- tiling 要求 `K=128`，`V` 支持 `128/256`，`chunk_size` 仅支持 `64/128`。
- 变长模式下 `cu_seqlens` 与 `chunk_indices` 必须同时提供，`chunk_indices` 长度为正偶数，且 `B=1`。
- 当前 ATK 用例遵循上述约束，并通过 `case_spec` 固定具体取值；扩展用例时应继续满足这些限制。

## 标杆来源

fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/README.md; torch_custom/fla_npu/test/test_npu_chunk_gated_delta_rule_bwd_dhu.py

CPU 标杆、输入构造、run_cpu、run_npu 和 FunctionApi 均在本目录的 `executor_chunk_gated_delta_rule_bwd_dhu.py` 中实现；公共文件只提供基础工具函数。

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93` 和 `ascend950`，可配合统一脚本的 `-soc=ascend910b|ascend910_93|ascend950` 使用。

## 默认用例

本目录 `atk_chunk_gated_delta_rule_bwd_dhu.json` 内置 200 条泛化用例（100 个 shape × bf16/fp16），
标准为 `mixed_tolerance_bm`，SOC 为 `ascend950`。覆盖 noGVA（`HK==HV`）与 GVA（`HV>HK`）。

示例（BF16 首条）：
`{"dtype": "bf16", "B": 64, "HK": 8, "HV": 8, "T": 1024, "K": 128, "V": 128, "chunk_size": 64, "op": "chunk_gated_delta_rule_bwd_dhu", "case_id": 0, "seed": 20260817, "route": "ascendc", "soc": "ascend950"}`

## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_bwd_dhu -npu_device_id=6
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_bwd_dhu -npu_device_id=6 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_bwd_dhu -npu_device_id=6 -scope=performance
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_bwd_dhu -npu_device_id=6 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_bwd_dhu -npu_device_id=6 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_bwd_dhu -scope=gen_cases
```

`gen_cases` 默认传入 `-dt 100 -en 0`。所有新增工程的 marker dtype 都保留两路生成入口，生成器会把不支持 FP16 的算子改回合法 BF16 用例。
