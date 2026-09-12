# ChunkGatedDeltaRuleFwdPrepare ATK 工程

本目录提供 `chunk_gated_delta_rule_fwd_prepare` 的 ATK 单算子工程：`executor_chunk_gated_delta_rule_fwd_prepare.py`、`gen_chunk_gated_delta_rule_fwd_prepare.py`、`chunk_gated_delta_rule_fwd_prepare.yaml`、`atk_chunk_gated_delta_rule_fwd_prepare.json`。

精度标准为 `mixed_tolerance_bm`（NPU DUT vs CPU 高精度 golden）。CPU 标杆为本目录 `scripts/cpu_golden.py` 的 `cpu_gdn_fwd_l2norm_to_recompute`。

## 输入约束

- 布局 BNSD：`q/k=[B,HK,T,K]`，`v=[B,HV,T,V]`，`g/beta=[B,HV,T]`。
- `K=128`，`V∈{128,256}`，`chunk_size=64`。
- `HV % HK == 0` 且 `HV/HK ∈ {1,2,3,4}`。任务按 HV 计数，K 头按 `hk = hv / (HV/HK)` 复用。
- `q/k/v` 当前仅 `BFLOAT16`；`g/beta` 为 `FLOAT`（golden / DUT 与单元测试一致）。
- 当前仅支持 `use_qk_l2norm_in_kernel=True`、`use_gate_in_kernel=False`；`use_exp2` 支持 True/False。
- `use_beta_sigmoid_in_kernel` 与 `allow_neg_eigval` 支持 True/False；`allow_neg_eigval=True` 要求 sigmoid。精度 JSON 覆盖全部 3 组合法组合：`sig1_neg1` / `sig1_neg0` / `sig0_neg0`。
- 变长要求 `B=1` 且 `cu_seqlens` 与 `chunk_indices` 成对（Python 未传 `chunk_indices` 时自动生成）。JSON 用 `seqlens` 列表表示，executor 转成 `cu_seqlens`。
- 尾块 `T % 64 != 0`：只在该 chunk 填 0，按有效行写出。
- 精度 JSON 含 packed varlen；性能 JSON 仍只取定长 `T>=256` 的前 6 条，不把变长算进基线。

## TilingKey

| TilingKey | 选择条件 | 普通用例 | 边界用例 | 适用 SoC | 实际选择证据 |
| --- | --- | --- | --- | --- | --- |
| 0 | ascend950 MIX 1:2，K=128，V=128/256，BT=64 | case 0（GVA T=256 `sig1_neg1`）、`r2_T64_V256` | `tail_T160`、`partial_pack_HV6`、`sig0_neg0` | A5 | host tiling 固定 `SetTilingKey(0)` |

## SOC 支持

YAML 元信息覆盖 `ascend910b`、`ascend910_93`、`ascend950`。内核当前只注册 `ascend950`。

## 默认用例

`atk_chunk_gated_delta_rule_fwd_prepare.json` 内置 300 条（100 shape × 3 组合法 flag，bf16）。当前回归集固定 `use_qk_l2norm=True`、`use_exp2=True`、`use_gate=False`；Torch 接口和 kernel 同时支持 `use_exp2=False`。每组 shape 按序覆盖：

| tag | sigmoid | neg | 含义 |
| --- | --- | --- | --- |
| `sig1_neg1` | True | True | `beta_eff = 2 * sigmoid(beta)` |
| `sig1_neg0` | True | False | `beta_eff = sigmoid(beta)` |
| `sig0_neg0` | False | False | 不做 sigmoid；host dummy `beta_eff`，Python 回填 `beta.float()` |

shape 覆盖 GVA、尾块 `T%64!=0`、不满 pack、`V=128/256`、packed varlen（`B=1` + `seqlens`）。

`atk_chunk_gated_delta_rule_fwd_prepare_perf.json` 仍为定长 `T>=256` 且 `sig1_neg1` 的前 6 条（不含 `seqlens`），不把另外两组 flag 和变长算进基线。

## 执行方式

本算子有 9 路输出。ATK 默认 GM 初始化会把 HBM 顶满，后续 case 会卡在 `rtStreamSynchronize`。精度请关 GM init；一次跑满 300 若占卡，按 50 条分批（`CASE_END` 不含右端）。

```bash
ATK_GM_INIT_MODE=off \
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_prepare -npu_device_id=5 -scope=accuracy -soc=ascend950

ATK_GM_INIT_MODE=off CASE_START=0 CASE_END=50 \
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_prepare -npu_device_id=5 -scope=accuracy -soc=ascend950

bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_prepare -npu_device_id=5 -scope=performance
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_prepare -npu_device_id=5 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_prepare -npu_device_id=5 -scope=mssanitizer
bash tests/atk/run_test_cpu.sh -op=chunk_gated_delta_rule_fwd_prepare -scope=gen_cases
```
