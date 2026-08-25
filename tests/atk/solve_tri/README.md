# SolveTri ATK 工程

本目录提供 `solve_tri` 的 ATK 单算子工程，包含 `executor_solve_tri.py`、`gen_solve_tri.py`、`solve_tri.yaml`、`atk_solve_tri.json`。

## 输入约束

- `layout` 支持 `bsnd/bnsd/tnd/ntd`。
- `BSND/BNSD` 输入必须为 4D；`TND/NTD` 输入必须为 3D。
- `BSND` 逻辑为 `[B,T,H,chunk_size]`，`BNSD` 逻辑为 `[B,H,T,chunk_size]`；`TND/NTD` 使用打包 token 维。
- `x/out` 仅支持 `FLOAT16/BFLOAT16`，输出 shape 与输入一致。
- 最后一维 `chunk_size` 仅支持 `64/128`。
- `TND/NTD` 变长模式必须同时提供 `cu_seqlens` 和 `chunk_indices`；`chunk_indices` 描述每个 chunk 的序列与局部 chunk 序号。
- 当前 ATK 用例遵循上述约束，并通过 `case_spec` 固定具体取值；扩展用例时应继续满足这些限制。

## 标杆来源

torch_custom/fla_npu/test/test_npu_solve_tri_ascend910b.py; fla/ops/ascendc/gdn/chunk_gdn_fwd/solve_tri/README.md

CPU 标杆、输入构造、run_cpu、run_npu 和 FunctionApi 均在本目录的 `executor_solve_tri.py` 中实现；公共文件只提供基础工具函数。

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93` 和 `ascend950`，可配合统一脚本的 `-soc=ascend910b|ascend910_93|ascend950` 使用。

## 默认用例

- BF16 用例：`{"dtype": "bf16", "B": 1, "H": 1, "T": 16, "chunk_size": 64, "layout": "bnsd", "op": "solve_tri", "case_id": 0, "seed": 20260817, "route": "ascendc", "soc": "ascend910b"}`
- FP16 用例：`{"dtype": "fp16", "B": 1, "H": 1, "T": 16, "chunk_size": 64, "layout": "bnsd", "op": "solve_tri", "case_id": 1, "seed": 20260818, "route": "ascendc", "soc": "ascend910b"}`

## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=solve_tri -npu_device_id=6
bash tests/atk/run_test_cpu.sh -op=solve_tri -npu_device_id=6 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=solve_tri -npu_device_id=6 -scope=performance
bash tests/atk/run_test_cpu.sh -op=solve_tri -npu_device_id=6 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=solve_tri -npu_device_id=6 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=solve_tri -scope=gen_cases
```

`gen_cases` 默认传入 `-dt 100 -en 0`。所有新增工程的 marker dtype 都保留两路生成入口，生成器会把不支持 FP16 的算子改回合法 BF16 用例。
