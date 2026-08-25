# RecomputeWUFwd ATK 工程

本目录提供 `recompute_w_u_fwd` 的 ATK 单算子工程，包含 `executor_recompute_w_u_fwd.py`、`gen_recompute_w_u_fwd.py`、`recompute_w_u_fwd.yaml`、`atk_recompute_w_u_fwd.json`。

## 输入约束

- `k` 必须为 `[B,HK,T,K]`，`v/u` 必须为 `[B,HV,T,V]`。
- `beta/g` 必须为 `[B,HV,T]`；`A` 必须为 `[B,HV,T,chunk_size]`。
- `w` 输出为 `[B,HV,T,K]`，不是 `empty_like(k)` 的 `[B,HK,T,K]`。
- `k` 与 `v` 的 `B/T` 必须一致；`beta/g/A` 的 head 维与 `v` 的 `HV` 对齐；`HV % HK == 0`。
- `k/v/A/w/u` 支持 `BFLOAT16/FLOAT16`；`beta/g` 支持 `FLOAT/BFLOAT16/FLOAT16`。
- `g` 当前 ACLNN 封装要求必传；`gk` 当前未启用，必须传 `None`。
- `K` 固定为 `128`，`V` 支持 `128/256`，`chunk_size` 仅支持 `64/128`；变长模式要求 `B=1` 且 `cu_seqlens/chunk_indices` 成对传入。
- 当前 ATK 用例遵循上述约束，并通过 `case_spec` 固定具体取值；扩展用例时应继续满足这些限制。

## 标杆来源

torch_custom/fla_npu/test/test_npu_recompute_w_u_fwd.py; fla/ops/ascendc/gdn/chunk_gdn_fwd/recompute_w_u_fwd/README.md

CPU 标杆、输入构造、run_cpu、run_npu 和 FunctionApi 均在本目录的 `executor_recompute_w_u_fwd.py` 中实现；公共文件只提供基础工具函数。

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93` 和 `ascend950`，可配合统一脚本的 `-soc=ascend910b|ascend910_93|ascend950` 使用。

## 默认用例

本目录 `atk_recompute_w_u_fwd.json` 内置 200 条泛化用例（100 个 shape × bf16/fp16），
标准为 `cv_fused_double_benchmark`，SOC 为 `ascend950`。覆盖 noGVA（`HK==HV`）与 GVA（`HV>HK`），
`T` 均为完整 chunk（`T >= chunk_size`，不再用小于单个 chunk 的小 `T`）。

示例（BF16 首条）：
`{"dtype": "bf16", "B": 64, "HK": 8, "HV": 8, "T": 1024, "K": 128, "V": 128, "chunk_size": 64, "op": "recompute_w_u_fwd", "case_id": 0, "seed": 20260817, "route": "ascendc", "soc": "ascend950"}`

## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=recompute_w_u_fwd -npu_device_id=6
bash tests/atk/run_test_cpu.sh -op=recompute_w_u_fwd -npu_device_id=6 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=recompute_w_u_fwd -npu_device_id=6 -scope=performance
bash tests/atk/run_test_cpu.sh -op=recompute_w_u_fwd -npu_device_id=6 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=recompute_w_u_fwd -npu_device_id=6 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=recompute_w_u_fwd -scope=gen_cases
```

`gen_cases` 默认传入 `-dt 100 -en 0`。所有新增工程的 marker dtype 都保留两路生成入口，生成器会把不支持 FP16 的算子改回合法 BF16 用例。
