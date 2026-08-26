# ChunkGatedDeltaRuleFwdH ATK 工程

本目录提供 `chunk_gated_delta_rule_fwd_h` 的 ATK 单算子工程，包含 `executor_chunk_gated_delta_rule_fwd_h.py`、`gen_chunk_gated_delta_rule_fwd_h.py`、`chunk_gated_delta_rule_fwd_h.yaml`、`atk_chunk_gated_delta_rule_fwd_h.json`。

本算子的单算子测试资产统一维护在本目录；算子实现目录下不再设置独立的 `tests/` 或 `test/` 目录。

## 输入约束

- `k` 必须为 `[B,HK,T,K]`；**`w` 与 `u` 都必须为 `[B,HV,T,K]/[B,HV,T,V]`，head 同为 HV**（GVA 语义，对齐 ACLNN `w.H == u.H`）。
- `g` 与 `gk` 必须且只能提供一个：GDN v1 使用 `g=[B,HV,T]`，KDA/GDN2 使用
  `gk=[B,HV,T,K]`；两者同时为空或同时非空均返回 `ACLNN_ERR_PARAM_INVALID`。
- `use_exp2=false` 时 `g/gk` 均使用 `exp`，`use_exp2=true` 时均使用 `exp2`。KDA/GDN2
  Prepare 输出 log2 域 `gk`，因此该调用链固定使用 `use_exp2=true`。
- `h` 输出为 `[B,HV,Nc,K,V]`，`state_v_first=true` 时末两维为 `[V,K]`；`initial_state/final_state` 同样受 `state_v_first` 解释。
- `k/w/u` 的 `B`、`T` 必须一致；`u` 的 `HV` 必须大于等于 `HK` 且 `HV % HK == 0`（`w` 亦为 HV，故 GVA 时 `w` 的 head 数与 `u` 相同）。
- `k/w/u/h/v_new` 支持 `BFLOAT16/FLOAT16`；gate 使用 `FLOAT` 或与 `k/w/u` 相同的 dtype；
  state 支持 `FLOAT/BFLOAT16`，initial/final state 必须同 dtype。
- `V` 固定为 `128`，`chunk_size` 固定为 `64`；变长模式支持 `cu_seqlens/chunk_indices` 成对传入。
- 当前 ATK 用例遵循上述约束，并通过 `case_spec` 固定具体取值；扩展用例时应继续满足这些限制。

> **CPU 标杆（`_forward_h_ref`）**使用 `w[b,hv]`（HV head）+ 共享 `k[b,hk]`（`hk = hv // (HV/HK)`），与内核/ACLNN 的 w=HV 语义一致；请不要把 `w` 建成 `[B,HK,…]` 或沿用 HK 索引标杆，那会与 `u` 的 HV 不一致触发 `161002`（历史误判根因）。

## 标杆实现

CPU 标杆、输入构造、`run_cpu`、`run_npu` 和 `FunctionApi` 均在本目录的
`executor_chunk_gated_delta_rule_fwd_h.py` 中实现；数学语义与算子 README 保持一致，
不依赖其他测试目录中的实现。

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93` 和 `ascend950`，可配合统一脚本的 `-soc=ascend910b|ascend910_93|ascend950` 使用。

## 默认用例

本目录 `atk_chunk_gated_delta_rule_fwd_h.json` 只保留满足 `V=128、chunk_size=64` 的泛化用例，
标准为 `cv_fused_double_benchmark`。前 72 条 `soc=ascend950`，覆盖 noGVA（`HK==HV`）、
GVA（`HV>HK`）以及 BF16/FP16 成对的 `gk-only` `exp/exp2` 标杆。ID 72/73 为
`soc=ascend910b` 的 `B=1,HK=4,HV=96,T=128,K=V=128` 回归；A2 的 20 个 AIC 下每核负载
为 5 个连续 head，必须按 `4+1` 两轮完成，并跨两个 chunk 验证 bank/slot 复用。

ID 74/75 同为 `soc=ascend910b`，使用 BF16 `initial_state`，分别覆盖 BF16 data 下
`output_final_state=false`，以及 FP16 data 下的 BF16→FP16 初态转换和 BF16 final-state
写回。关闭时稳定 Python 入口第三项必须为 `None`；开启时 `final_state` 必须为 BF16
并参与双标杆比较。

ID 76-83 使用 3 个 chunk 的同 seed 成对用例，验证 rolling state 始终按 state dtype 回写，且
`output_final_state=false/true` 不改变 `h/v_new`。覆盖 FP16 data + FP32 state、FP16 data +
BF16 state、BF16 data + BF16 state，以及 gk-only + FP32 state。

示例（BF16 首条）：
`{"dtype": "bf16", "B": 1, "HK": 4, "HV": 4, "T": 512, "K": 128, "V": 128, "chunk_size": 64, "op": "chunk_gated_delta_rule_fwd_h", "case_id": 0, "seed": 20260817, "route": "ascendc", "soc": "ascend950"}`

## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_h -npu_device_id=6
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_h -npu_device_id=6 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_h -npu_device_id=6 -scope=performance
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_h -npu_device_id=6 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_h -npu_device_id=6 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_h -scope=gen_cases
```

本算子重生成 84 条 profile 时使用
`GEN_CASES_DTYPE_NUMBERS=42 bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_h -scope=gen_cases`。
全局脚本默认仍传入 `-dt 100 -en 0`；marker dtype 保留 BF16/FP16 两路生成入口。
