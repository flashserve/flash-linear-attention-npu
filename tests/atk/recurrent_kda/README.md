# RecurrentKda ATK 工程

本目录提供 `recurrent_kda` 的 ATK 单算子工程，包含 `executor_recurrent_kda.py`、`gen_recurrent_kda.py`、`recurrent_kda.yaml`、`atk_recurrent_kda.json`。

## 输入约束

- `layout` 支持 `BSND/TND`；`BSND` 下 `q/k=[B,T,H,K]`，`v/out=[B,T,HV,V]`，`g=[B,T,HV,K]`，`beta=[B,T,HV]`。
- `TND` 下 `q/k=[T,H,K]`，`v/out=[T,HV,V]`，`g=[T,HV,K]`，`beta=[T,HV]`。
- `initial_state` 是 state pool；`state_v_first=true` 时为 `[state_capacity,HV,V,K]`，否则为 `[state_capacity,HV,K,V]`。
- `cu_seqlens` 可选；传入时为同设备 `INT32/INT64` tensor，shape 为 `[seq_num+1]`，首项为 `0`，末项不超过 token capacity，offset 单调不减。
- `q/k/v/out` 仅支持 `BFLOAT16`；`g/beta` 支持 `FLOAT/BFLOAT16/FLOAT16`，`A_log/dt_bias` 支持 `FLOAT`。
- `HV % H == 0`，`H/HV <= 256`；当前支持 `K=128,V=128` 或 `K=128,V=256`。
- A2/A3 支持多时间窗口长序列；A5 每段仍不超过 8；显式 `ssm_state_indices` 可使 `state_capacity > seq_num`。
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

## 模型 case 与性能用例映射（评审稿 v2，2026-11-10）

按 `tests/atk/README.md` 的正式验收用例包要求建立三类映射。`atk_recurrent_kda_perf.json`
已按工作区根 `target.md` §3 的 A5 模型 case 重建（原文件只有 1 个 `ascend910b` 用例）；
全部 case 标记 `review_status=draft_await_device_validation`，
**设备侧逐 case 跑通校验前不得作为正式验收依据**。目标 SoC 统一为 `ascend950`。

### 模型 case → perf case id

| 模型 case | shape（B,T,H_k,H_v,K,V） | state | perf case id | 目标 kernel | 基线 | 目标 | 统计 |
| --- | --- | --- | ---: | --- | --- | --- | --- |
| A | 1,8,8,96,128,128 | FP32 | 0 | `recurrent_kda` | 待实测 | 1.20x | `Task Duration(us)` 5/10 均值 |
| B | 2,2,2,4,128,128 | FP32 | 1 | `recurrent_kda` | 待实测 | 1.20x | 同上 |
| C | 1,8,2,6,128,128 | BF16（意图） | 2 | `recurrent_kda` | 待实测 | 1.20x | 同上 |
| D | 1,8,1,2,128,128 | FP32 | 3 | `recurrent_kda` | 待实测 | 1.20x | 同上 |

> case C：`executor_recurrent_kda.py` 的 `build_inputs` 把 `initial_state` 固定为 FP32，
> `state_dtype_intent=BF16` 目前只是意图标记。扩展前该 case 实际按 FP32 运行，
> **不得作为 BF16 state 的达标证据**。

### 另外两类映射（现状与缺口）

| 映射 | 现状 | 待办 |
| --- | --- | --- |
| 逻辑分支 → 精度 case id | `atk_recurrent_kda.json` 仅 1 个 case（id 0） | 按 `docs/design.md` §14.13 补 `H_v=2`、`ssm_state_indices` 一维/二维、varlen 空序列与尾块、`state_v_first` 两布局，并建立逐分支映射 |
| 可达 TilingKey → mss case id | `atk_recurrent_kda_mss.json`、`atk_recurrent_kda_longseq_mss.json` 各 1 case | R2 引入 `vChunkNum` 后补新 key 的最小用例；未补齐不得进入正式验收 |

### 设备侧校验门禁

1. 每个 perf case 在 A5 上以 `-scope=performance` 单独运行，确认进入 NPU kernel、无参数校验拒绝、单次 <60s。
2. 校验 `soc=ascend950`、`T<=8`、`H_v % H_k == 0`、`K=V=128` 与本文「输入约束」一致。
3. 不合法项按本文输入约束调整，并把调整结果写回 `docs/validation.md` 的追溯表。