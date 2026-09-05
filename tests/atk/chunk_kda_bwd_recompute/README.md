# ChunkKdaBwdRecompute ATK 工程

本目录提供 `chunk_kda_bwd_recompute` 的 ATK 单算子工程，包含 `executor_chunk_kda_bwd_recompute.py`、`gen_chunk_kda_bwd_recompute.py`、`chunk_kda_bwd_recompute.yaml`、`atk_chunk_kda_bwd_recompute.json`。

## 输入约束

- 布局固定为 dense BNSD：`q/k=[B,HK,T,K]`，`v=[B,HV,T,V]`，`g=[B,HV,T,K]`，`beta=[B,HV,T]`，`A=[B,HV,T,chunk_size]`。
- `A_log` 为 `[HV]` 的 `FLOAT`；`dt_bias` 可选，为 `[HV,K]` 的 `FLOAT`。
- `q/k/v/A` 仅支持 `BFLOAT16`；`g/beta` 支持 `BFLOAT16/FLOAT`。
- `K=V=128`，`chunk_size=64`；`HV % HK == 0`。
- `use_gate_in_kernel=true` 时必须提供 `A_log`；`safe-gate` 下 `lower_bound` 默认 `-5.0`。
- kernel 仅注册 `ascend950`；当前 ATK 看护用例均为定长 dense，不含 `cu_seqlens/chunk_indices`。
- 输出顺序为 `(gk, w, u, qg, kg)`。`gk` 为 `FLOAT`，其余为 `BFLOAT16`。`w/u` 为 `[B,HV,T,K/V]`，`qg/kg` 为 `[B,HV,T,K]`。

## 标杆来源

`kda_gate_wu_fusion_golden/kda_gate_wu_golden.py` 的 `fused_cpu`（safe-gate + chunk cumsum + GQA 展开 + `A @ kbg/vb`）；`torch_custom/fla_npu/test/test_npu_chunk_kda_bwd_recompute.py`。

CPU 标杆、输入构造、run_cpu、run_npu 和 FunctionApi 均在本目录的 `executor_chunk_kda_bwd_recompute.py` 中实现；公共文件只提供基础工具函数。

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93` 和 `ascend950`，可配合统一脚本的 `-soc=ascend910b|ascend910_93|ascend950` 使用。实际 kernel 仅 `ascend950` 可跑 NPU DUT。

## 默认用例

- BF16 单 chunk：`{"dtype": "bf16", "B": 1, "HK": 1, "HV": 1, "T": 64, "K": 128, "V": 128, "chunk_size": 64, "g_dtype": "bf16", "beta_dtype": "bf16", "use_gate": true, "use_exp2": true, "has_dt_bias": true, "lower_bound": -5.0, "op": "chunk_kda_bwd_recompute", "case_id": 0, "seed": 20260817, "route": "ascendc", "soc": "ascend950"}`
- leftover：`T=16`，其余同上，`case_id=1`
- GQA：`HK=2, HV=4, T=128`，`case_id=2`

## 执行方式

```bash
bash tests/atk/run_test_cpu.sh -op=chunk_kda_bwd_recompute -npu_device_id=0 -scope=accuracy -soc=ascend950
bash tests/atk/run_test_cpu.sh -op=chunk_kda_bwd_recompute -npu_device_id=0 -scope=performance -soc=ascend950
bash tests/atk/run_test_cpu.sh -op=chunk_kda_bwd_recompute -npu_device_id=0 -scope=determinism -soc=ascend950
bash tests/atk/run_test_cpu.sh -op=chunk_kda_bwd_recompute -npu_device_id=0 -scope=mssanitizer -soc=ascend950
bash tests/atk/run_test_cpu.sh -op=chunk_kda_bwd_recompute -scope=gen_cases -soc=ascend950
```

`gen_cases` 默认传入 `-dt 100 -en 0`。所有新增工程的 marker dtype 都保留两路生成入口，生成器会把不支持 FP16 的算子改回合法 BF16 用例。
