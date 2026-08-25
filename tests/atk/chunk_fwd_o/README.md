# ChunkFwdO ATK 工程

本目录提供 `chunk_fwd_o` 的 ATK 单算子工程，包含 `executor_chunk_fwd_o.py`、`gen_chunk_fwd_o.py`、`chunk_fwd_o.yaml`、`atk_chunk_fwd_o.json`。

## 输入约束

- `q/k` 必须为 `[B,HK,T,K]`，且二者形状完全一致。
- `v/o` 必须为 `[B,HV,T,V]`，`g` 必须为 `[B,HV,T]`。
- `h` 必须为 `[B,HV,num_chunks,K,V]`，其中 `num_chunks` 由 `T/chunk_size` 或变长 `chunk_indices` 推导。
- `q/k` 与 `v/g/o` 的 `B`、`T` 必须一致；`HV % HK == 0`。
- `K` 固定为 `128`，`V` 支持 `128/256`，`chunk_size` 仅支持 `64/128`；`scale` 建议按 `1 / sqrt(K)` 设置。
- `q/k/v/h/o` 支持 `BFLOAT16/FLOAT16`；`g` 支持 `FLOAT/FLOAT16/BFLOAT16`。
- 变长模式下 `cu_seqlens` 与 `chunk_indices` 必须同时提供，`chunk_indices` 为 `[seq_id, chunk_id]` 扁平化数组，且 `B=1`。
- 当前 ATK 用例遵循上述约束，并通过 `case_spec` 固定具体取值；扩展用例时应继续满足这些限制。

## 标杆来源

torch_custom/fla_npu/test/test_fwd_o.py; fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/README.md

CPU 标杆、输入构造、run_cpu、run_npu 和 FunctionApi 均在本目录的 `executor_chunk_fwd_o.py` 中实现；公共文件只提供基础工具函数。

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93` 和 `ascend950`，可配合统一脚本的 `-soc=ascend910b|ascend910_93|ascend950` 使用。

## 默认用例

- BF16 用例：`{"dtype": "bf16", "B": 1, "HK": 1, "HV": 1, "T": 16, "K": 128, "V": 128, "chunk_size": 64, "op": "chunk_fwd_o", "case_id": 0, "seed": 20260817, "route": "ascendc", "soc": "ascend910b"}`
- FP16 用例：`{"dtype": "fp16", "B": 1, "HK": 1, "HV": 1, "T": 16, "K": 128, "V": 128, "chunk_size": 64, "op": "chunk_fwd_o", "case_id": 1, "seed": 20260818, "route": "ascendc", "soc": "ascend910b"}`

## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_o -npu_device_id=6
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_o -npu_device_id=6 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_o -npu_device_id=6 -scope=performance
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_o -npu_device_id=6 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_o -npu_device_id=6 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=chunk_fwd_o -scope=gen_cases
```

`gen_cases` 默认传入 `-dt 100 -en 0`。所有新增工程的 marker dtype 都保留两路生成入口，生成器会把不支持 FP16 的算子改回合法 BF16 用例。
