# ChunkLocalCumsum ATK 工程

本目录提供 `chunk_local_cumsum` 的 ATK 单算子工程，包含 `executor_chunk_local_cumsum.py`、`gen_chunk_local_cumsum.py`、`chunk_local_cumsum.yaml`、`atk_chunk_local_cumsum.json`。

## 输入约束

- `g` 必须为 rank 3，shape 为 `[B,H,T]`，当前 AscendC kernel 仅支持 `head_first=true`。
- `g/out` 支持 `FLOAT/FLOAT16/BFLOAT16`，kernel 内部按 FP32 累加后转换为输出 dtype。
- `chunk_size` 必须为 2 的幂，并满足 host tiling 推导出的 `block_t >= chunk_size`。
- `reverse` 控制 chunk 内累加方向；`scale` 为输出缩放系数。
- 变长模式下 `cu_seqlens` 非空时，`chunk_indices_out` 必须非空且元素数为偶数，同时要求 `B=1`。
- `output_dtype` 支持 `float32/float16/bfloat16` 及跟随输入 dtype 的别名。
- 当前 ATK 用例遵循上述约束，并通过 `case_spec` 固定具体取值；扩展用例时应继续满足这些限制。

## 标杆来源

torch_custom/fla_npu/test/test_npu_chunk_local_cumsum.py; fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/README.md

CPU 标杆、输入构造、run_cpu、run_npu 和 FunctionApi 均在本目录的 `executor_chunk_local_cumsum.py` 中实现；公共文件只提供基础工具函数。

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93` 和 `ascend950`，可配合统一脚本的 `-soc=ascend910b|ascend910_93|ascend950` 使用。

## 默认用例

- BF16 用例：`{"dtype": "bf16", "B": 1, "H": 1, "T": 16, "chunk_size": 8, "reverse": false, "scale": 1.0, "op": "chunk_local_cumsum", "case_id": 0, "seed": 20260817, "route": "ascendc", "soc": "ascend910b"}`
- FP16 用例：`{"dtype": "fp16", "B": 1, "H": 1, "T": 16, "chunk_size": 8, "reverse": false, "scale": 1.0, "op": "chunk_local_cumsum", "case_id": 1, "seed": 20260818, "route": "ascendc", "soc": "ascend910b"}`

## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_local_cumsum -npu_device_id=6
bash tests/atk/run_test_cpu.sh -op=chunk_local_cumsum -npu_device_id=6 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=chunk_local_cumsum -npu_device_id=6 -scope=performance
bash tests/atk/run_test_cpu.sh -op=chunk_local_cumsum -npu_device_id=6 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=chunk_local_cumsum -npu_device_id=6 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=chunk_local_cumsum -scope=gen_cases
```

`gen_cases` 默认传入 `-dt 100 -en 0`。所有新增工程的 marker dtype 都保留两路生成入口，生成器会把不支持 FP16 的算子改回合法 BF16 用例。
