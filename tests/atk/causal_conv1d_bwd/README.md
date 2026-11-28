# CausalConv1dBwd ATK 工程

本目录提供 `causal_conv1d_bwd` 的 ATK 单算子工程，包含 `executor_causal_conv1d_bwd.py`、`gen_causal_conv1d_bwd.py`、`causal_conv1d_bwd.yaml`、`atk_causal_conv1d_bwd.json`。

## 输入约束

- `x/dx` 在 `BSND/BSH/BNSD` 下为 `[B,T,D]`，在 `TND/NTD` 下为 `[totalTokens,D]`。
- `y/dy` 的物理 shape 由 `inputLayout` 决定；`BSND/BSH/TND` 与逻辑 shape 一致，`BNSD/NTD` 使用 `[B,N,T,Dh]` 或 `[N,totalTokens,Dh]`，且 `D=N*Dh`。
- `weight` 必须为 `[W,D]`；`initial_state/dht/dh0` 如提供，逻辑 shape 为 `[B,W,D]`。
- `BSND/TND` 的逻辑特征维 `D` 必须为 `16` 的倍数；`BNSD/NTD` 的 `Dh` 必须为 `16` 的倍数。
- `x/y/weight/dy/initial_state/dht` 数据类型需要一致，支持 `FLOAT/FLOAT16/BFLOAT16`；`activation=1/2` 时必须提供 `y`。
- `TND/NTD` 必须提供 `queryStartLoc`，其首项为 `0`、末项为 `totalTokens`，且单调不减；固定 batch 模式要求 `T > 0`。
- 当前 ATK 用例使用 `inputLayout=BSND`，并通过 `case_spec` 固定具体取值；扩展用例时应继续满足这些限制。

## 标杆来源

torch_custom/fla_npu/test/test_npu_causal_conv1d_bwd.py; fla/ops/ascendc/gdn/gdn_preprocess/causal_conv1d_bwd/README.md

CPU 标杆、输入构造、run_cpu、run_npu 和 FunctionApi 均在本目录的 `executor_causal_conv1d_bwd.py` 中实现；公共文件只提供基础工具函数。

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93` 和 `ascend950`，可配合统一脚本的 `-soc=ascend910b|ascend910_93|ascend950` 使用。

## 默认用例

- BF16 用例：`{"dtype": "bf16", "B": 1, "T": 8, "D": 16, "W": 4, "op": "causal_conv1d_bwd", "case_id": 0, "seed": 20260817, "route": "ascendc", "soc": "ascend910b"}`
- FP16 用例：`{"dtype": "fp16", "B": 1, "T": 8, "D": 16, "W": 4, "op": "causal_conv1d_bwd", "case_id": 1, "seed": 20260818, "route": "ascendc", "soc": "ascend910b"}`

## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=causal_conv1d_bwd -npu_device_id=6
bash tests/atk/run_test_cpu.sh -op=causal_conv1d_bwd -npu_device_id=6 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=causal_conv1d_bwd -npu_device_id=6 -scope=performance
bash tests/atk/run_test_cpu.sh -op=causal_conv1d_bwd -npu_device_id=6 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=causal_conv1d_bwd -npu_device_id=6 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=causal_conv1d_bwd -scope=gen_cases
```

`gen_cases` 默认传入 `-dt 100 -en 0`。所有新增工程的 marker dtype 都保留两路生成入口，生成器会把不支持 FP16 的算子改回合法 BF16 用例。
