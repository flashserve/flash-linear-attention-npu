# RecurrentGatedDeltaRule

## 1. 功能概述

面向 decode/投机推理的小步长 recurrent Gated Delta Rule。它按逻辑序列选择初始状态槽，逐 token 生成输出，并把每步状态写入 `ssm_state_indices` 指定的槽；支持标量 gate、逐 K gate 与接受 token 数。

## 2. 数学定义

对逻辑序列 b，`actual_seq_lengths[0]` 是前置跳过 token 数，后续元素是各序列长度。令序列起点为 `p_b`；无 `num_accepted_tokens` 时从 `ssm_state_indices[p_b]` 读取初始状态，有该输入时从 `ssm_state_indices[p_b+a_b-1]` 读取。对序列内 token t：

```text
S_t = exp(g_t) * S_(t-1)                         # g 存在时
S_t = S_t * exp(gk_t)[None, :]                   # gk 存在时
delta_t = beta_t * (v_t - S_t @ k_t)
S_t = S_t + outer(delta_t, k_t)
o_t = S_t @ (q_t * scale)
state_ref[ssm_state_indices[t]] = S_t
```

`g` 与 `gk` 可独立使用、同时使用或同时为空；同时为空时不施加衰减。`state_ref` 是可变输入输出。

## 3. 输入、输出和属性

### 3.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Layout | 说明 |
| --- | --- | --- | --- | --- | --- |
| `query` | 必选 | `[T,H_k,K]` | BF16 | TND | Query |
| `key` | 必选 | `[T,H_k,K]` | BF16 | TND | Key |
| `value` | 必选 | `[T,H_v,V]` | BF16 | TND | Value |
| `beta` | 必选 | `[T,H_v]` | BF16 | TN | 更新权重 |
| `state_ref` | 必选/可变 | `[D_s,H_v,V,K]` | BF16/FP32 | ND | 状态槽，原地更新 |
| `actual_seq_lengths` | 必选 | `[B+1]` | INT32 | ND | 首项为前置跳过长度，后续 B 项为各逻辑序列长度，总和为 T |
| `ssm_state_indices` | 必选 | `[T]` | INT32 | ND | token 到状态槽映射 |
| `g` | 可选 | `[T,H_v]` | FP32 | TN | 标量 gate |
| `gk` | 可选 | `[T,H_v,K]` | FP32 | TND | 逐 K 维 gate |
| `num_accepted_tokens` | 可选 | `[B]` | INT32 | ND | 每序列用于选择初始状态槽的位置，范围 [1,seq_len] |

### 3.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `out` | `[T,H_v,V]` | BF16 | recurrent 输出 |
| `state_ref` | `[D_s,H_v,V,K]` | 与输入一致 | 原地更新后的状态 |

### 3.3 属性

| 名称 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `scale_value` | float | `1.0` | 推荐 1/sqrt(K) |

## 4. 支持范围

| 项目 | 支持范围 |
| --- | --- |
| SOC | A2 (`ascend910b`)、A3 (`ascend910_93`)、A5 (`ascend950`) |
| Dtype | Q/K/V/beta/out 为 BF16；state 为 BF16/FP32；gate FP32；索引 INT32 |
| Format/Layout | TND + 独立状态槽布局 `[D_s,H_v,V,K]` |
| 模式 | 变长 recurrent、g/gk 独立或组合、可选 accepted-token 状态选择、原地 state |



## 5. 调用入口

实现类型：`ascendc`

| 入口 | API |
| --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.recurrent_gated_delta_rule` |
| aclnn | `aclnnRecurrentGatedDeltaRuleGetWorkspaceSize` / `aclnnRecurrentGatedDeltaRule` |
| Ascend C `<<<>>>` | `recurrent_gated_delta_rule<<<blockDim, l2ctrl, stream>>>(...)` |
| legacy（可选） | 未实现 |

完整签名和示例见 [API 文档](docs/api.md)，kernel、tiling、同步与内存设计见[设计文档](docs/design.md)。

## 6. 精度与性能

- 主精度入口：`tests/operators/recurrent_gated_delta_rule/accuracy/test_recurrent_gated_delta_rule.py`，主调用使用 `fla_npu.ops.ascendc`。
- 用例规格：`tests/op_cases/recurrent_gated_delta_rule.json`；覆盖 变长 recurrent、g/gk 独立或组合、可选 accepted-token 状态选择、原地 state。
- 参考实现：`torch_recurrent_gated_delta_rule_reference`；容差按 JSON 中各 dtype 的 `rtol/atol` 执行，不允许为规避失败而收窄输入范围。
- 性能：使用 msopprof 覆盖 JSON performance case，并与当前主线基线比较设备侧 kernel duration，禁止性能回退。

## 7. 已知限制

- 每条序列本次有效 token 数不超过 8。
- g/gk 均为空表示单位衰减；二者存在时 dtype 必须为 FP32，shape 分别为 `[T,H_v]`、`[T,H_v,K]`。
- state 槽索引必须位于 `[0,D_s)`；actual_seq_lengths 有效项之和等于 T。
- `H_k/H_v <= 256`、`K/V <= 512` 且 `H_v % H_k == 0`。
- state_ref 为原地更新参数，不能 require_grad；非连续 state 依赖 CANN >= 9.1。

## 8. 构建与验证

```bash
FLA_NPU_SOC=ascend910b FLA_NPU_OPS=recurrent_gated_delta_rule python -m pip wheel --no-build-isolation --no-deps . -w dist
pytest -q tests/operators/recurrent_gated_delta_rule/accuracy/test_recurrent_gated_delta_rule.py
python scripts/check_operator_compliance.py
```

A3/A5 分别将 `FLA_NPU_SOC` 替换为 `ascend910_93`/`ascend950`。aclnn 与直调通路源文件位于
`tests/operators/recurrent_gated_delta_rule/routes/`，均使用同一份 JSON 规格。

<a id="shape-symbols"></a>

## 9. 附录：Shape 变量说明

- 模型/算法族：Gated Delta Network (GDN)
- 模型级符号表：[GDN 模型符号表](../../README.md#model-shape-symbols)
- 符号表版本：`gdn-shape-v1`

| 变量 | 语义 |
| --- | --- |
| `B` | Batch size；varlen 打包场景通常为 1 |
| `T` | dense 序列长度或 varlen 打包后的 token 总数 |
| `H_k` | Query/Key head 数 |
| `H_v` | Value/Output/State head 数 |
| `K` | Query/Key 单 head 特征维 |
| `V` | Value/Output 单 head 特征维 |
| `D_s` | 状态槽位数 |
| `Q_a` | 单次调用实际接受的 token 数 |
