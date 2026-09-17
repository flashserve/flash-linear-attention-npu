# CausalConv1d ATK 工程

本目录提供 `causal_conv1d` 的 ATK 单算子工程，包含 `executor_causal_conv1d.py`、`gen_causal_conv1d.py`、`causal_conv1d.yaml`、`atk_causal_conv1d.json`。

## 输入约束

- `x` 使用 dim-last `[totalTokens,D]`，`queryStartLoc` 划分 batch。
- `weight` 必须为 `[W,D]`，`W` 支持 `2/3/4`，`D` 需要是 `16` 的倍数。
- `bias` 为可选 `[D]`；`convStates` 为 `[state_rows,state_len,D]`，`state_len >= W - 1`。
- `queryStartLoc`、`cacheIndices`、`numAcceptedTokens` 使用 INT32 device Tensor，`hasInitialState` 使用 BOOL/INT32 device Tensor。
- `x/weight/bias/convStates/y` 数据类型支持 `BFLOAT16/FLOAT16`，且输入数据类型需要保持一致。
- 本工程通过 `causal_conv1d_fn` 调用 FN，数据保持 dim-last，不做布局转换。
- 当前 ATK 用例遵循上述约束，并通过 `case_spec` 固定具体取值；扩展用例时应继续满足这些限制。

## 标杆来源

torch_custom/fla_npu/test/test_npu_causal_conv1d.py; fla/ops/ascendc/gdn/gdn_preprocess/causal_conv1d/README.md

CPU 标杆、输入构造、run_cpu、run_npu 和 FunctionApi 均在本目录的 `executor_causal_conv1d.py` 中实现；公共文件只提供基础工具函数。

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93` 和 `ascend950`，可配合统一脚本的 `-soc=ascend910b|ascend910_93|ascend950` 使用。

## 默认用例

- BF16 用例：`{"dtype": "bf16", "B": 1, "T": 8, "D": 16, "W": 4, "op": "causal_conv1d", "case_id": 0, "seed": 20260817, "route": "ascendc", "soc": "ascend910b"}`
- FP16 用例：`{"dtype": "fp16", "B": 1, "T": 8, "D": 16, "W": 4, "op": "causal_conv1d", "case_id": 1, "seed": 20260818, "route": "ascendc", "soc": "ascend910b"}`

## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=causal_conv1d -npu_device_id=6
bash tests/atk/run_test_cpu.sh -op=causal_conv1d -npu_device_id=6 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=causal_conv1d -npu_device_id=6 -scope=performance
bash tests/atk/run_test_cpu.sh -op=causal_conv1d -npu_device_id=6 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=causal_conv1d -npu_device_id=6 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=causal_conv1d -scope=gen_cases
```

`gen_cases` 默认传入 `-dt 100 -en 0`。所有新增工程的 marker dtype 都保留两路生成入口，生成器会把不支持 FP16 的算子改回合法 BF16 用例。
