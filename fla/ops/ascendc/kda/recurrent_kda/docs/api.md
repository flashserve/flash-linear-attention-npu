# RecurrentKda API 与调用示例

## 1. API 总览

| 通路 | API/入口 | 支持情况 |
| --- | --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.recurrent_kda` | 支持 |
| aclnn | `aclnnRecurrentKdaGetWorkspaceSize` / `aclnnRecurrentKda` | 支持 |
| Ascend C `<<<>>>` | `recurrent_kda<<<blockDim, nullptr, stream>>>` | 支持 |
| legacy | `torch.ops.npu.npu_recurrent_kda` | 支持（显式加载） |

各入口表达同一个 fused recurrent KDA 前向语义。算子计算本身在一个 AICore kernel 内完成，不把
`KdaGateCumsum + recurrent` 暴露为两段式公共实现。

## 2. 公共参数与约束

Shape 符号统一引用 [KDA 模型符号表](../../README.md#model-shape-symbols)。

### 2.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Layout | 说明 |
| --- | --- | --- | --- | --- | --- |
| `q` | 必选 | `BSND=[B,T,H_k,K]` 或 `TND=[T,H_k,K]` | BF16 | BSND/TND | query |
| `k` | 必选 | 与 `q` 相同 | BF16 | BSND/TND | key |
| `v` | 必选 | `BSND=[B,T,H_v,V]` 或 `TND=[T,H_v,V]` | BF16 | BSND/TND | value |
| `g` | 必选 | `BSND=[B,T,H_v,K]` 或 `TND=[T,H_v,K]` | FP32/BF16/FP16 | BSND/TND | 预计算 step log gate 或 raw gate |
| `beta` | 必选 | `BSND=[B,T,H_v]` 或 `TND=[T,H_v]` | FP32/BF16/FP16 | BSND/TND | delta 更新系数 |
| `initial_state` | 可选 | `[seq_num,H_v,V,K]` | FP32/BF16 | ND | Python 入口为空时创建全零 FP32 状态 |
| `cu_seqlens` | 可选 | `[N+1]` | INT64 | ND | Python/legacy 为 `int[]`，aclnn 为 `aclIntArray` |
| `ssm_state_indices` | 可选 | `[>=T]` | INT32/INT64 | ND | MTP decode 状态槽索引 |
| `A_log` | 条件必选 | `[H_v]` | FP32 | ND | `use_gate_in_kernel=True` 时必选 |
| `dt_bias` | 可选 | `[H_v*K]` 或 `[H_v,K]` | FP32 | ND | raw gate 偏置 |
| `num_accepted_tokens` | 可选 | `[seq_num]` | INT32/INT64 | ND | 必须与 `ssm_state_indices` 一起传 |

### 2.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `out` | 与 `v` 相同 | BF16 | recurrent 输出 |
| `final_state` | 与 `initial_state` 相同；`output_final_state=False` 时 Python/legacy 返回空 tensor | FP32/BF16 | 最终状态 |

### 2.3 属性

| 名称 | 类型 | 默认值 | 取值范围 | 说明 |
| --- | --- | --- | --- | --- |
| `layout` | str | `BSND` | `{"BSND", "TND"}` | 输入布局 |
| `scale` | float? | Python/legacy 为 `K ** -0.5` | 任意有限浮点 | 乘到 query 上 |
| `output_final_state` | bool | `false` | `{false, true}` | 是否返回最终状态 |
| `use_qk_l2norm_in_kernel` | bool | `false` | `{false, true}` | 是否在 kernel 内对 q/k 做 L2 normalize |
| `use_gate_in_kernel` | bool | `false` | `{false, true}` | 是否把 `g` 解释为 raw gate |
| `use_beta_sigmoid_in_kernel` | bool | `false` | `{false, true}` | 是否在 kernel 内计算 `sigmoid(beta)` |
| `allow_neg_eigval` | bool | `false` | `{false, true}` | beta sigmoid 后是否乘 2 |
| `safe_gate` | bool | `false` | `{false, true}` | raw gate 的 safe 分支 |
| `lower_bound` | float? | `-5.0` | `[-5,0)` when `safe_gate=True` | safe gate 下界 |
| `state_v_first` | bool | `true` | 当前必须为 `true` | 状态布局为 `[seq_num,H_v,V,K]` |

## 3. aclnn API

### 3.1 接口签名

```cpp
aclnnStatus aclnnRecurrentKdaGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *gate,
    const aclTensor *beta,
    const aclTensor *initialState,
    const aclIntArray *cuSeqlensOptional,
    const aclTensor *ssmStateIndicesOptional,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclTensor *numAcceptedTokensOptional,
    const char *layout,
    double scale,
    bool outputFinalState,
    bool useQkL2normInKernel,
    bool useGateInKernel,
    bool useBetaSigmoidInKernel,
    bool allowNegEigval,
    bool safeGate,
    double lowerBound,
    bool stateVFirst,
    const aclTensor *out,
    const aclTensor *finalState,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnRecurrentKda(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

`GetWorkspaceSize` 完成参数校验、连续化/cast 预处理和 executor 创建；第二段在传入 stream 上异步执行。
输入、输出、workspace 和 executor 必须保持有效，直到 stream 完成。

### 3.2 调用示例

```cpp
// 按 2.1/2.2 的 shape、dtype 和 layout 创建输入/输出 aclTensor。
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
ACLNN_CHECK(aclnnRecurrentKdaGetWorkspaceSize(
    q, k, v, g, beta, initialState, cuSeqlens, ssmStateIndices,
    aLog, dtBias, numAcceptedTokens, "BSND", scale, true, true, true,
    true, false, false, -5.0, true, out, finalState,
    &workspaceSize, &executor));
void *workspace = nullptr;
if (workspaceSize != 0) {
    ACL_CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
}
ACLNN_CHECK(aclnnRecurrentKda(workspace, workspaceSize, executor, stream));
ACL_CHECK(aclrtSynchronizeStream(stream));
```

## 4. `fla_npu.ops.ascendc` API

### 4.1 接口签名

```python
recurrent_kda(q, k, v, g, beta, initial_state=None, *,
              cu_seqlens=None, ssm_state_indices=None, A_log=None,
              dt_bias=None, num_accepted_tokens=None, layout="BSND",
              scale=None, output_final_state=False,
              use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
              use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
              safe_gate=False, lower_bound=None, state_v_first=True)
```

稳定入口通过 ctypes 直调 aclnn，不依赖 `torch.ops.npu` 注册。`initial_state=None` 时由 wrapper 创建
`[seq_num,H_v,V,K]` 的全零 FP32 状态。

### 4.2 调用示例

```python
import torch
from fla_npu.ops.ascendc import recurrent_kda

B, T, H, H_v, K, V = 2, 2, 2, 4, 128, 128
q = torch.randn(B, T, H, K, device="npu", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn(B, T, H_v, V, device="npu", dtype=torch.bfloat16)
g = torch.randn(B, T, H_v, K, device="npu", dtype=torch.float32)
beta = torch.randn(B, T, H_v, device="npu", dtype=torch.float32)
A_log = torch.randn(H_v, device="npu", dtype=torch.float32)

out, final_state = recurrent_kda(
    q, k, v, g, beta, A_log=A_log, layout="BSND",
    output_final_state=True, use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True)
torch.npu.synchronize()
```

## 5. Ascend C `<<<>>>` 直调

`blockDim`、workspace 和序列化 tiling data 必须来自同一组 host tiling 结果。参数顺序与 kernel 定义保持一致：

```cpp
recurrent_kda<<<blockDim, nullptr, stream>>>(
    q, k, v, g, beta, initialState, cuSeqlens, ssmStateIndices,
    aLog, dtBias, numAcceptedTokens, out, finalState, workspace, tiling);
```

直调通路只作为 route/诊断入口；公开 Python 和 aclnn API 负责完整参数校验。

## 6. `torch.ops.npu` API（可选）

```python
import torch
import fla_npu

fla_npu.load_legacy_torch_ops()
out, final_state = torch.ops.npu.npu_recurrent_kda(
    q, k, v, g, beta, None, A_log=A_log, layout="BSND",
    output_final_state=True, use_gate_in_kernel=True)
```

## 7. 已知限制

- `q/k/v/out` 当前仅支持 BF16。
- `K/V` 当前仅支持 `K=128,V=128` 或 `K=128,V=256` 两档枚举。
- 每条 recurrent 序列长度必须 `<=8`；dense 模式未传 `cu_seqlens` 时 `T<=8`。
- 仅支持 `layout="BSND"` 和 `layout="TND"`；BSND 变长序列物理 `B` 必须为 1。
- 仅支持 `state_v_first=True`。
- `use_gate_in_kernel=false` 时 `A_log/dt_bias/safe_gate` 必须为空或 false。

## 8. 异常与返回码

| 条件 | 返回码/异常 |
| --- | --- |
| 必选 tensor、workspaceSize 或 executor 为空 | `ACLNN_ERR_PARAM_NULLPTR` |
| rank/shape/dtype/layout、序列长度或属性组合非法 | `ACLNN_ERR_PARAM_INVALID` |
| 内部 tensor 创建或 L0 调用失败 | `ACLNN_ERR_INNER_NULLPTR` |
| Python 输入不是 NPU tensor 或 runtime/op_api 未加载 | `RuntimeError` |

负向 case 的预期返回码与消息片段集中定义在 `tests/op_cases/recurrent_kda.json`。
