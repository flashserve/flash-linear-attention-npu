# ChunkGatedDeltaRuleFwdH

## 1. 功能概述

Gated Delta Rule 的跨 chunk 状态推进算子。它根据 K/W/U、标量 gate 或逐 K 维 gate，从 initial_state 递推各 chunk 起始状态 H、修正值 V_new，并按需返回 final_state。

## 2. 数学定义

令 `S_c` 为 chunk c 的起始状态，`G_c` 为该 chunk 末端门控：

```text
V_new_c = U_c - W_c @ S_c
S_(c+1) = decay(G_c) * S_c + K_gated_c^T @ V_new_c
H[c]    = S_c
```

`g` 路径使用每 head 标量衰减，`gk` 路径对 K 维逐元素衰减；二者至少提供一个。

## 3. 输入、输出和属性

### 3.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Layout | 说明 |
| --- | --- | --- | --- | --- | --- |
| `k` | 必选 | `[B,H_k,T,K]` | FP16/BF16 | BNSD | Key 或已门控 KG |
| `w` | 必选 | `[B,H_v,T,K]` | FP16/BF16 | BNSD | WY 的 W |
| `u` | 必选 | `[B,H_v,T,V]` | FP16/BF16 | BNSD | WY 的 U |
| `g` | 条件可选 | `[B,H_v,T]` | FP16/BF16/FP32 | BNS | 标量 gate |
| `gk` | 条件可选 | `[B,H_v,T,K]` | FP16/BF16/FP32 | BNSD | 逐 K 维 gate |
| `initial_state` | 可选 | `[N,H_v,K,V]` | FP16/BF16/FP32 | ND | 每序列初始状态 |
| `cu_seqlens` | 可选 | `[N+1]` | INT64 | ND | 变长序列累计长度 |
| `chunk_indices` | 可选 | `[2*N_c]` | INT64 | ND | 展平 chunk 索引 |

### 3.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `h` | `[B,H_v,N_c,K,V]` | 与 k 一致 | 每个 chunk 起始状态 |
| `v_new` | `[B,H_v,T,V]` | 与 u 一致 | 状态修正后的 Value |
| `final_state` | `[N,H_v,K,V]` | 跟随 initial_state 或 FP32 | 按需返回末状态 |

### 3.3 属性

| 名称 | 类型 | 默认值 | 取值范围 | 说明 |
| --- | --- | --- | --- | --- |
| `output_final_state` | bool | `false` | `{false, true}` | 是否返回末状态 |
| `chunk_size` | int | `64` | `{64, 128}` | chunk 长度 |
| `save_new_value` | bool | `true` | `{true}` | 当前必须 true |
| `use_exp2` | bool | `false` | `{false}` | 当前必须 false |
| `transpose_state_layout` | bool | `false` | `{false}` | 当前必须 false |

## 4. 支持范围

| 项目 | 支持范围 |
| --- | --- |
| SOC | A2 (`ascend910b`)、A3 (`ascend910_93`)、A5 (`ascend950`) |
| Dtype | K/W/U 为 FP16/BF16；gate/state 可为 FP32 |
| Format/Layout | BNSD；initial/final state 为 `[N,H_v,K,V]` |
| 模式 | 定长/变长序列、g/gk、可选 initial/final state |

变长序列模式中，`cu_seqlens[0]` 必须为 0、末项等于 `T` 且序列非递减。`chunk_indices` 必须按 sequence-major 列出全部 `(seq_id, local_chunk_id)`；其条目数和当前调用的 `N_c` 一致。定长与变长序列、尾块与整块遵循同一数学定义。

## 5. 调用入口

实现类型：`ascendc`

| 入口 | API |
| --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd_h` |
| aclnn | `aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize` / `aclnnChunkGatedDeltaRuleFwdH` |
| Ascend C `<<<>>>` | `chunk_gated_delta_rule_fwd_h<<<blockDim, l2ctrl, stream>>>(...)` |
| legacy（可选） | `torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h` |

完整签名和示例见 [API 文档](docs/api.md)，kernel、tiling、同步与内存设计见[设计文档](docs/design.md)。

## 6. 精度与性能

- 主精度入口：`tests/operators/chunk_gated_delta_rule_fwd_h/accuracy/test_chunk_gated_delta_rule_fwd_h.py`，主调用使用 `fla_npu.ops.ascendc`。
- 用例规格：`tests/op_cases/chunk_gated_delta_rule_fwd_h.json`；覆盖定长/变长序列、g/gk、可选 initial/final state。
- 参考实现：`torch_chunk_gated_delta_rule_fwd_h_reference`；容差按 JSON 中各 dtype 的 `rtol/atol` 执行，不允许为规避失败而收窄输入范围。
- 性能：使用 msopprof 覆盖 JSON performance case，并与当前主线基线比较设备侧 kernel duration，禁止性能回退。

## 7. 已知限制

- `K` 仅支持 128，`V` 仅支持 128/256，`chunk_size` 仅支持 64/128。
- `g` 与 `gk` 至少提供一个；`H_v % H_k == 0`。
- 变长序列当前仅支持物理 `B=1`，索引必须完整且 sequence-major。
- `save_new_value=true`、`use_exp2=false`、`transpose_state_layout=false`。

## 8. 构建与验证

```bash
FLA_NPU_SOC=ascend910b FLA_NPU_OPS=chunk_gated_delta_rule_fwd_h python -m pip wheel --no-build-isolation --no-deps . -w dist
pytest -q tests/operators/chunk_gated_delta_rule_fwd_h/accuracy/test_chunk_gated_delta_rule_fwd_h.py
python scripts/check_operator_compliance.py
```

A3/A5 分别将 `FLA_NPU_SOC` 替换为 `ascend910_93`/`ascend950`。aclnn 与直调通路源文件位于
`tests/operators/chunk_gated_delta_rule_fwd_h/routes/`，均使用同一份 JSON 规格。

<a id="shape-symbols"></a>

## 9. 附录：Shape 变量说明

- 模型/算法族：Gated Delta Network (GDN)
- 模型级符号表：[GDN 模型符号表](../../README.md#model-shape-symbols)
- 符号表版本：`gdn-shape-v1`

| 变量 | 语义 |
| --- | --- |
| `B` | Batch size；变长序列打包场景通常为 1 |
| `N` | 变长序列的逻辑序列数 |
| `T` | 定长序列长度或变长序列打包后的 token 总数 |
| `H_k` | Query/Key head 数 |
| `H_v` | Value/Output/State head 数 |
| `K` | Query/Key 单 head 特征维 |
| `V` | Value/Output 单 head 特征维 |
| `C` | chunk_size |
| `N_c` | 当前调用中的 chunk 总数 |
