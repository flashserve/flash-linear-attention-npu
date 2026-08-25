# ChunkScaledDotKkt ATK 工程

本目录提供 `chunk_scaled_dot_kkt` 的 ATK 单算子工程，包含 `executor_chunk_scaled_dot_kkt.py`、`gen_chunk_scaled_dot_kkt.py`、`chunk_scaled_dot_kkt.yaml`、`atk_chunk_scaled_dot_kkt.json`。

## 输入约束

- `k` 必须为 4D，shape 为 `[B,Hk,T,K]`，head-first 排布。
- `g/beta` 必须为 3D，shape 为 `[B,Hv,T]`，并且 `B/T` 与 `k` 一致。
- GVA 要求 `Hv % Hk == 0`；输出 `A` 的 head 维与 `k` 对齐，shape 为 `[B,Hk,T,chunk_size]`。
- `k` 支持 `BFLOAT16/FLOAT16`；`g/beta/A` 按源码 README 使用 `FLOAT`。
- `chunk_size` 支持 `16/32/64/128`；当前版本不支持 `gk` 分支。
- `cu_seqlens` 和 `chunk_indices` 必须同时传入或同时省略；`chunk_indices` 以 `[seq_id, chunk_id]` 成对存放。
- 当前 ATK 用例遵循上述约束，并通过 `case_spec` 固定具体取值；扩展用例时应继续满足这些限制。

## 标杆来源

torch_custom/fla_npu/test/test_npu_chunk_scaled_dot_kkt.py; fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_scaled_dot_kkt/README.md

CPU 标杆、输入构造、run_cpu、run_npu 和 FunctionApi 均在本目录的 `executor_chunk_scaled_dot_kkt.py` 中实现；公共文件只提供基础工具函数。

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93` 和 `ascend950`，可配合统一脚本的 `-soc=ascend910b|ascend910_93|ascend950` 使用。

## 默认用例

- BF16 用例：`{"dtype": "bf16", "B": 1, "HK": 1, "HV": 1, "T": 16, "K": 16, "chunk_size": 16, "op": "chunk_scaled_dot_kkt", "case_id": 0, "seed": 20260817, "route": "ascendc", "soc": "ascend910b"}`
- FP16 用例：`{"dtype": "fp16", "B": 1, "HK": 1, "HV": 1, "T": 16, "K": 16, "chunk_size": 16, "op": "chunk_scaled_dot_kkt", "case_id": 1, "seed": 20260818, "route": "ascendc", "soc": "ascend910b"}`

## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_scaled_dot_kkt -npu_device_id=6
bash tests/atk/run_test_cpu.sh -op=chunk_scaled_dot_kkt -npu_device_id=6 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=chunk_scaled_dot_kkt -npu_device_id=6 -scope=performance
bash tests/atk/run_test_cpu.sh -op=chunk_scaled_dot_kkt -npu_device_id=6 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=chunk_scaled_dot_kkt -npu_device_id=6 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=chunk_scaled_dot_kkt -scope=gen_cases
```

`gen_cases` 默认传入 `-dt 100 -en 0`。所有新增工程的 marker dtype 都保留两路生成入口，生成器会把不支持 FP16 的算子改回合法 BF16 用例。
