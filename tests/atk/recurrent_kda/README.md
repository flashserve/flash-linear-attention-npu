# RecurrentKda ATK 工程

本目录提供 `recurrent_kda` 的 ATK 单算子工程，包含 `executor_recurrent_kda.py`、`gen_recurrent_kda.py`、`recurrent_kda.yaml`、`atk_recurrent_kda.json`。

## 输入约束

- `layout` 支持 `BSND/TND`；`BSND` 下 `q/k=[B,T,H,K]`，`v/out=[B,T,HV,V]`，`g=[B,T,HV,K]`，`beta=[B,T,HV]`。
- `TND` 下 `q/k=[T,H,K]`，`v/out=[T,HV,V]`，`g=[T,HV,K]`，`beta=[T,HV]`。
- `initial_state` 是 state pool；`state_v_first=true` 时为 `[state_capacity,HV,V,K]`，否则为 `[state_capacity,HV,K,V]`。
- `cu_seqlens` 可选；传入时为同设备 `INT32/INT64` tensor，shape 为 `[seq_num+1]`，首项为 `0`，末项不超过 token capacity，offset 单调不减。
- `q/k/v/out` 仅支持 `BFLOAT16`；`g/beta` 支持 `FLOAT/BFLOAT16/FLOAT16`，`A_log/dt_bias` 支持 `FLOAT`。
- `HV % H == 0`，`H/HV <= 256`；当前支持 `K=128,V=128` 或 `K=128,V=256`。
- A2/A3 支持多时间窗口长序列；A5 的 V=128/V=256 路径均支持长序列，内部按最多四个 token 的窗口递推；显式 `ssm_state_indices` 可使 `state_capacity > seq_num`。
- 当前 ATK 用例遵循上述约束，并通过 `case_spec` 固定具体取值；扩展用例时应继续满足这些限制。

A5 验收须按 [API 文档的模板选择说明](../../../fla/ops/ascendc/kda/recurrent_kda/docs/api.md)覆盖 V-first/K-first 的通用与常用融合四种模板模式，并分别验证 FP32/BF16 State、V=128/V=256、长序列和 indexed State。默认小用例只用于基础检查，不能作为全部 A5 TilingKey 的覆盖证据；compat 模式仅用于 A2/A3。

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

## A5 Q/K/V stride 寻址诊断

`scripts/qkv_stride_tiling.cpp` 调用当前算子的实际 host tiling processor；使用单算子构建生成的
`ophost_transformer_tiling_obj.dir/flags.make` 中的 include 参数编译，并链接当前 CANN 的
`error_manager/ascendalog/unified_dlog/platform/register/graph`，得到 `qkv_stride_tiling`。

```bash
python scripts/test_qkv_strides.py \
  --tiling-probe <qkv_stride_tiling> \
  --kernel-dir <installed_vendor>/op_impl/ai_core/tbe/kernel/ascend950/recurrent_kda \
  --output <diagnostic_output>
```

该脚本通过 ACL runtime 直接加载当前安装的 kernel，保留 Q/K/V view 的原始 stride，避免公开
aclnn 连续化掩盖 L0 寻址问题。80 个场景覆盖四种 A5 模式、FP32/BF16 State、V=128/256，
以及 batch gap、head-major、Q/K/V 不同 stride、固定长度、跨 batch 的 varlen 和空请求。
out 和有效 State 槽须与连续输入逐位相同，并使用既有 CPU 标杆及 `rtol=0.02, atol=0.01` 检查精度；
另外检查重叠 batch 与 A2 head-major 的 tiling 拒绝行为。这是 L0 专项诊断，不替代上述 ATK 验收。

2026-11-11 在 A5_1/Ascend950PR、CANN 9.2.0-beta.2 上，当前修复通过全部 80 个场景和两个负向检查；
同一诊断加载修复前的 kernel 时，首个 batch-gap 用例即出现 stride/连续输入结果不一致。
同次验证中，既有 ATK 小用例的精度、确定性（50 次）与内存检测各通过 1/1，公开 State mutation/None
契约回归通过；576 组 A2 host tiling 的成功/失败和分核结果与修复前相同，未在 A2 设备执行。
