# RecurrentKda ATK 工程

本目录提供 `recurrent_kda` 的 ATK 单算子工程，包含 `executor_recurrent_kda.py`、`gen_recurrent_kda.py`、`recurrent_kda.yaml`、`atk_recurrent_kda.json`。

## 输入约束

- `layout` 支持 `BSND/TND`；`BSND` 下 `q/k=[B,T,H,K]`，`v/out=[B,T,HV,V]`，`g=[B,T,HV,K]`，`beta=[B,T,HV]`。
- `TND` 下 `q/k=[T,H,K]`，`v/out=[T,HV,V]`，`g=[T,HV,K]`，`beta=[T,HV]`。
- `initial_state` 是 state pool；`state_v_first=true` 时为 `[state_capacity,HV,V,K]`，否则为 `[state_capacity,HV,K,V]`。
- `cu_seqlens` 为必传同设备 `INT32/INT64` tensor，shape 为 `[seq_num+1]`，首项为 `0`，末项不超过 token capacity，offset 单调不减。
- `q/k/v/out` 仅支持 `BFLOAT16`；`g/beta` 支持 `FLOAT/BFLOAT16/FLOAT16`，`A_log/dt_bias` 支持 `FLOAT`。
- `HV % H == 0`，`H/HV <= 256`；当前支持 `K=128,V=128` 或 `K=128,V=256`。
- 每段序列长度不超过 `8`；显式 `ssm_state_indices` 可使 `state_capacity > seq_num`。
- 当前 ATK 用例遵循上述约束，并通过 `case_spec` 固定具体取值；扩展用例时应继续满足这些限制。

## 标杆来源

tests/reference/recurrent_kda_reference.py; fla/ops/ascendc/kda/recurrent_kda/README.md

CPU 标杆、输入构造、run_cpu、run_npu 和 FunctionApi 均在本目录的 `executor_recurrent_kda.py` 中实现；公共文件只提供基础工具函数。

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93` 和 `ascend950`，可配合统一脚本的 `-soc=ascend910b|ascend910_93|ascend950` 使用。

## 默认用例

- BF16 用例：`{"dtype": "bf16", "B": 2, "T": 2, "H": 2, "HV": 4, "K": 128, "V": 128, "layout": "BSND", "state_v_first": true, "op": "recurrent_kda", "case_id": 0, "seed": 20260817, "route": "ascendc", "soc": "ascend910b"}`

## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6 -scope=performance
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -npu_device_id=6 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=recurrent_kda -scope=gen_cases
```

`gen_cases` 默认传入 `-dt 100 -en 0`。所有新增工程的 marker dtype 都保留两路生成入口，生成器会把不支持 FP16 的算子改回合法 BF16 用例。
