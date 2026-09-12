# ChunkGatedDeltaRuleFwdH ATK 工程

本目录提供 `chunk_gated_delta_rule_fwd_h` 的 ATK 单算子工程，包含 `executor_chunk_gated_delta_rule_fwd_h.py`、`gen_chunk_gated_delta_rule_fwd_h.py`、`chunk_gated_delta_rule_fwd_h.yaml`、`atk_chunk_gated_delta_rule_fwd_h.json`。

## 输入约束

- `k` 必须为 `[B,HK,T,K]`；**`w` 与 `u` 都必须为 `[B,HV,T,K]/[B,HV,T,V]`，head 同为 HV**（GVA 语义，对齐 ACLNN `w.H == u.H`）。
- `g` 与 `gk` 至少提供一个：`g=[B,HV,T]`，`gk=[B,HV,T,K]`。
- `h` 输出为 `[B,HV,Nc,K,V]`，`state_v_first=true` 时末两维为 `[V,K]`；`initial_state/final_state` 同样受 `state_v_first` 解释。
- `k/w/u` 的 `B`、`T` 必须一致；`u` 的 `HV` 必须大于等于 `HK` 且 `HV % HK == 0`（`w` 亦为 HV，故 GVA 时 `w` 的 head 数与 `u` 相同）。
- `k/w/u/h/v_new` 支持 `BFLOAT16/FLOAT16`；gate 支持 `FLOAT/FLOAT16/BFLOAT16`，state 支持 `FLOAT/BFLOAT16/FLOAT16`。
- `V` 支持 `128/256`，`chunk_size` 仅支持 `64/128`；变长模式支持 `cu_seqlens/chunk_indices` 成对传入。
- 当前 ATK 用例遵循上述约束，并通过 `case_spec` 固定具体取值；扩展用例时应继续满足这些限制。

> **CPU 标杆（`_forward_h_ref`）**使用 `w[b,hv]`（HV head）+ 共享 `k[b,hk]`（`hk = hv // (HV/HK)`），与内核/ACLNN 的 w=HV 语义一致；请不要把 `w` 建成 `[B,HK,…]` 或沿用 HK 索引标杆，那会与 `u` 的 HV 不一致触发 `161002`（历史误判根因）。

## 标杆来源

torch_custom/fla_npu/test/test_fwd_h.py; fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/README.md

CPU 标杆、输入构造、run_cpu、run_npu 和 FunctionApi 均在本目录的 `executor_chunk_gated_delta_rule_fwd_h.py` 中实现；公共文件只提供基础工具函数。

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93` 和 `ascend950`，可配合统一脚本的 `-soc=ascend910b|ascend910_93|ascend950` 使用。

## 默认用例

本目录 `atk_chunk_gated_delta_rule_fwd_h.json` 内置 200 条泛化用例（100 个 shape × bf16/fp16），
标准为 `mixed_tolerance_bm`，SOC 为 `ascend950`。覆盖 noGVA（`HK==HV`）与 GVA（`HV>HK`）。

示例（BF16 首条）：
`{"dtype": "bf16", "B": 64, "HK": 8, "HV": 8, "T": 1024, "K": 128, "V": 128, "chunk_size": 64, "op": "chunk_gated_delta_rule_fwd_h", "case_id": 0, "seed": 20260817, "route": "ascendc", "soc": "ascend950"}`

## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_h -npu_device_id=6
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_h -npu_device_id=6 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_h -npu_device_id=6 -scope=performance
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_h -npu_device_id=6 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_h -npu_device_id=6 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_h -scope=gen_cases
```

`gen_cases` 默认传入 `-dt 100 -en 0`。所有新增工程的 marker dtype 都保留两路生成入口，生成器会把不支持 FP16 的算子改回合法 BF16 用例。
