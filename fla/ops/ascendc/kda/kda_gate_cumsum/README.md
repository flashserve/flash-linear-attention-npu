# KdaGateCumsum

## 1. 功能概述

把 KDA 的逐 token gate 转换为每个 chunk 内的 log2 累积 gate `gk`。既支持已计算好的 step gate，也支持带 A_log/dt_bias 的 safe-gate raw 输入。

## 2. 数学定义

对每条序列、value head 和 K 维，累加在 chunk 边界重新置零：

```text
step = g                                             use_gate_in_kernel=false
x    = (g + dt_bias[h_v,k]) * exp(A_log[h_v])        raw safe-gate
step = lower_bound * sigmoid(x)                      raw safe-gate
gk[t] = sum(step[chunk_start:t]) / ln(2)
```

因下游使用 `exp2(gk_i-gk_j)`，输出统一为 FP32。safe-gate 的逐步值位于
`[lower_bound,0]`，合法的长 chunk 累积可达到较大负值，不能通过收窄输入范围掩盖写回或同步问题。

## 3. 输入、输出和属性

### 3.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Layout | 说明 |
| --- | --- | --- | --- | --- | --- |
| `g` | 必选 | `[B,T,H_v,K]/[B,H_v,T,K]/[T,H_v,K]/[H_v,T,K]` | FP16/BF16/FP32 | BSND/BNSD/TND/NTD | step gate 或 raw gate |
| `A_log` | 条件必选 | `[H_v]` | FP32 | ND | use_gate_in_kernel=true 时必选 |
| `dt_bias` | 可选 | `[H_v*K] 或 [H_v,K]` | FP32 | ND | safe-gate 偏置 |
| `cu_seqlens` | 可选 | `[N+1]` | INT64 | ND | 变长序列累计长度 |

### 3.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `gk` | `与 g 相同` | FP32 | chunk 内 log2 累积 gate |

### 3.3 属性

| 名称 | 类型 | 默认值 | 取值范围 | 说明 |
| --- | --- | --- | --- | --- |
| `chunk_size` | int | `无` | `{32, 64, 128}` | chunk 长度 |
| `use_gate_in_kernel` | bool | `false` | `{false, true}` | 是否把 g 当作 raw gate |
| `safe_gate` | bool | `false` | `{false, true}` | raw gate 模式下必须为 true，非 raw gate 模式下必须为 false |
| `lower_bound` | double | `-5.0` | `[-5, 0)` | safe gate 下限 |
| `layout` | str | `BSND` | `{"BSND", "BNSD", "TND", "NTD"}` | 仅接受大写取值 |

## 4. 支持范围

| 项目 | 支持范围 |
| --- | --- |
| SOC | A2 (`ascend910b`)、A3 (`ascend910_93`)、A5 (`ascend950`) |
| Dtype | g 为 FP16/BF16/FP32；A_log/dt_bias/gk 为 FP32；cu_seqlens 为 INT64 |
| Format/Layout | rank4 使用 BSND/BNSD，rank3 使用 TND/NTD；layout 必须显式且不根据 shape 推导 |
| 模式 | 定长/变长序列、step-gate/safe raw-gate、四种 layout、整块/尾块 |

变长序列模式中，`cu_seqlens[0]` 必须为 0、末项等于 `T` 且序列非递减。定长与变长序列、尾块与整块遵循同一数学定义。

## 5. 调用入口

实现类型：`ascendc`

| 入口 | API |
| --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.kda_gate_cumsum` |
| aclnn | `aclnnKdaGateCumsumGetWorkspaceSize` / `aclnnKdaGateCumsum` |
| Ascend C `<<<>>>` | `kda_gate_cumsum<<<blockDim, l2ctrl, stream>>>(...)` |
| legacy（可选） | `torch.ops.npu.npu_kda_gate_cumsum` |

完整签名和示例见 [API 文档](docs/api.md)，kernel、tiling、同步与内存设计见[设计文档](docs/design.md)。

## 6. 精度与性能

- 主精度入口：`tests/operators/kda_gate_cumsum/accuracy/test_kda_gate_cumsum.py`，主调用使用 `fla_npu.ops.ascendc`。
- 用例规格：`tests/op_cases/kda_gate_cumsum.json`；覆盖定长/变长序列、step-gate/safe raw-gate、四种 layout、整块/尾块。
- 参考实现：`_kda_gate_cumsum_reference`；容差按 JSON 中各 dtype 的 `rtol/atol` 执行，不允许为规避失败而收窄输入范围。
- 性能：使用 msopprof 覆盖 JSON performance case，并与当前主线基线比较设备侧 kernel duration，禁止性能回退。

## 7. 已知限制

- K<=256，chunk_size 仅支持 32/64/128。
- use_gate_in_kernel=true 时 A_log 必须为 [H_v]、safe_gate 必须 true，dt_bias 若存在须为 [H_v*K] 或 [H_v,K]。
- lower_bound 仅支持 [-5,0)；use_gate_in_kernel=false 时 safe_gate 必须 false。
- use_gate_in_kernel=false 时 A_log 与 dt_bias 必须为空，避免未消费输入在不同通路产生歧义。
- rank4 变长序列物理 B 必须为 1；cu_seqlens 首项为 0、非递减且末项等于 T。

## 8. 构建与验证

```bash
FLA_NPU_SOC=ascend910b FLA_NPU_OPS=kda_gate_cumsum python -m pip wheel --no-build-isolation --no-deps . -w dist
pytest -q tests/operators/kda_gate_cumsum/accuracy/test_kda_gate_cumsum.py
python scripts/check_operator_compliance.py
```

A3/A5 分别将 `FLA_NPU_SOC` 替换为 `ascend910_93`/`ascend950`。aclnn 与直调通路源文件位于
`tests/operators/kda_gate_cumsum/routes/`，均使用同一份 JSON 规格。

<a id="shape-symbols"></a>

## 9. 附录：Shape 变量说明

- 模型/算法族：Kimi Delta Attention (KDA)
- 模型级符号表：[KDA 模型符号表](../README.md#model-shape-symbols)
- 符号表版本：`kda-shape-v1`

| 变量 | 语义 |
| --- | --- |
| `B` | Batch size |
| `N` | 变长序列的逻辑序列数 |
| `T` | 定长序列长度或变长序列打包 token 总数 |
| `H_v` | Value/Output head 数 |
| `K` | Query/Key 单 head 特征维 |
| `C` | chunk_size |
