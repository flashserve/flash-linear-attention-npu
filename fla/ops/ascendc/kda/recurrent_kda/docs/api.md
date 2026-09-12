# RecurrentKda API 与调用示例

## 1. API 总览

| 通路 | API/入口 | 支持情况 |
| --- | --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.recurrent_kda` | 支持 |
| aclnn | `aclnnRecurrentKdaGetWorkspaceSize` / `aclnnRecurrentKda` | 支持 |
| Ascend C `<<<>>>` | `recurrent_kda<<<blockDim, nullptr, stream>>>` | 支持 |
| legacy | `torch.ops.npu.npu_recurrent_kda` | 不支持 |

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
| `initial_state` | Python 可选，aclnn 必选 | V-first `[state_capacity,H_v,V,K]` 或 K-first `[state_capacity,H_v,K,V]` | FP32/BF16 | ND | 可变 state pool；Python 仅在非原位模式下允许为空，并创建 `[seq_num,...]` 全零 FP32 状态 |
| `cu_seqlens` | 可选 | `[seq_num+1]` | INT32/INT64 | ND | 提供时使用 fla-org 累积 offset；为空时 BSND 按 batch 行划分，TND 视为一条序列 |
| `ssm_state_indices` | 可选 | packed `[>=T]` 或 speculative `[seq_num,max_step]` | INT32/INT64 | ND | 每个 token 对应的 state pool 槽索引 |
| `A_log` | 条件必选 | `[H_v]` | FP32 | ND | `use_gate_in_kernel=True` 时必选 |
| `dt_bias` | 可选 | `[H_v*K]` 或 `[H_v,K]` | FP32 | ND | raw gate 偏置 |
| `num_accepted_tokens` | 可选 | `[seq_num]` | INT32/INT64 | ND | 必须与 `ssm_state_indices` 一起传 |

### 2.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `out` | 与 `v` 相同 | BF16 | recurrent 输出 |
| `final_state` | 与 `initial_state` 同 shape/dtype | FP32/BF16 | 原位模式与输入 alias；非原位模式为独立输出；`output_final_state=False` 时 Python 主入口不返回 state |

### 2.3 属性

| 名称 | 类型 | 默认值 | 取值范围 | 说明 |
| --- | --- | --- | --- | --- |
| `layout` | str | `BSND` | `{"BSND", "TND"}` | 输入布局 |
| `scale` | float? | Python 为 `K ** -0.5`；aclnn 由调用方传入 | 浮点数 | 乘到 query 上 |
| `output_final_state` | bool | `false` | `{false, true}` | Python 主入口是否返回最终状态 |
| `inplace_final_state` | bool | `true` | `{false, true}` | 是否把最终状态写回 `initial_state` |
| `use_qk_l2norm_in_kernel` | bool | `false` | `{false, true}` | 是否在 kernel 内对 q/k 做 L2 normalize |
| `use_gate_in_kernel` | bool | `false` | `{false, true}` | 是否把 `g` 解释为 raw gate |
| `use_beta_sigmoid_in_kernel` | bool | `false` | `{false, true}` | 是否在 kernel 内计算 `sigmoid(beta)` |
| `allow_neg_eigval` | bool | `false` | `{false, true}` | beta sigmoid 后是否乘 2 |
| `safe_gate` | bool | `false` | `{false, true}` | raw gate 的 safe 分支 |
| `lower_bound` | float? | `-5.0` | `[-5,0)` when `safe_gate=True` | safe gate 下界 |
| `state_v_first` | bool | `false` | `{false, true}` | true 为 `[state_capacity,H_v,V,K]`，false 为 `[state_capacity,H_v,K,V]` |

## 3. aclnn API

### 3.1 接口签名

```cpp
aclnnStatus aclnnRecurrentKdaGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *gate,
    const aclTensor *beta,
    aclTensor *initialStateRef,
    const aclTensor *cuSeqlensOptional,
    const aclTensor *ssmStateIndicesOptional,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclTensor *numAcceptedTokensOptional,
    const char *layout,
    double scale,
    bool outputFinalState,
    bool inplaceFinalState,
    bool useQkL2normInKernel,
    bool useGateInKernel,
    bool useBetaSigmoidInKernel,
    bool allowNegEigval,
    bool safeGate,
    double lowerBound,
    bool stateVFirst,
    const aclTensor *attnOut,
    const aclTensor *finalState,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnRecurrentKda(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

`GetWorkspaceSize` 完成参数校验、Q/K/V stride 判定、其余非 state tensor 连续化预处理和 executor
创建；第二段在传入 stream 上异步执行。满足约束的非连续 Q/K/V 与 state 均通过 `CreateView` 保留
shape、storage、stride 和 offset，kernel 按 tiling
中的真实 stride 直接访问。原位模式直接写回输入 view；非原位模式直接写入 `finalState` view。原位模式若调用者
另外传入独立的 `finalState` 输出，仅为该输出保留一次必要的 `ViewCopy`。`cuSeqlensOptional` 为空时，BSND
按 batch 行划分序列，TND 视为一条序列；提供时仅在 host 检查 rank/dtype，具体 offset 值由 device kernel 读取，
因而可在 ACLGraph capture/replay 中变化。首项必须为 0，offset 必须单调不减，末项不得超过输入 token capacity，
且每个相邻 offset 的差值不超过 8。
输入、输出、workspace 和 executor 必须保持有效，直到 stream 完成。

### 3.2 调用示例

```cpp
// 按 2.1/2.2 的 shape、dtype 和 layout 创建输入/输出 aclTensor。
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
ACLNN_CHECK(aclnnRecurrentKdaGetWorkspaceSize(
    q, k, v, g, beta, state, cuSeqlens, ssmStateIndices,
    aLog, dtBias, numAcceptedTokens, "BSND", scale, true, true, true,
    true, true, false, false, -5.0, true, out, finalState,
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
              inplace_final_state=True,
              use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
              use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
              safe_gate=False, lower_bound=None, state_v_first=False)
```

稳定入口通过 ctypes 直调 aclnn，不依赖 `torch.ops.npu` 注册。`initial_state=None` 仅在
`inplace_final_state=False` 时有效，此时 wrapper 创建与 `state_v_first` 对应布局的全零 FP32 状态；原位模式必须
显式传入 state。显式传入的非连续 state view 由 kernel 按真实 stride 直接访问；满足第 7 节约束的 Q/K/V view
同样保留 storage offset 并直接读取，不满足时自动连续化。原位模式返回与输入相同 storage/stride
的 state，非原位模式保持输入不变并返回独立 state。

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
state = torch.zeros(B, H_v, V, K, device="npu", dtype=torch.float32)
cu_seqlens = torch.arange(B + 1, device="npu", dtype=torch.int32) * T

out, final_state = recurrent_kda(
    q, k, v, g, beta, state, cu_seqlens=cu_seqlens,
    A_log=A_log, layout="BSND",
    output_final_state=True, use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True, state_v_first=True)
torch.npu.synchronize()
```

## 5. Ascend C `<<<>>>` 直调

`blockDim`、workspace 和序列化 tiling data 必须来自同一组 host tiling 结果。参数顺序与 kernel 定义保持一致：

```cpp
recurrent_kda<<<blockDim, nullptr, stream>>>(
    q, k, v, g, beta, initialState, cuSeqlens, ssmStateIndices,
    aLog, dtBias, numAcceptedTokens, out, initialStateOut, finalState, workspace, tiling);
```

直调通路只作为 route/诊断入口；公开 Python 和 aclnn API 负责完整参数校验。直调通路按连续物理地址或 tiling data 中的 Q/K/V/state stride 解释 GM 地址；非连续直调必须使用
与实际 view shape、stride 和首地址匹配的 host tiling 结果。

## 6. legacy `torch.ops.npu` 通路

`torch.ops.npu.npu_recurrent_kda` 不属于当前支持范围，也不提供兼容性保证。请使用稳定的 Python ctypes
入口或 aclnn API。

## 7. 已知限制

- `q/k/v/out` 当前仅支持 BF16。
- `K/V` 当前仅支持 `K=128,V=128` 或 `K=128,V=256` 两档枚举。
- Python/aclnn 入口支持符合 stride 约束的非连续 Q/K/V 与 `initial_state`。
- Q/K/V 直接访问要求 feature stride=1、head/token 区间不重叠、stride 为正；BSND `B>1` 时
  `batchStride == T * tokenStride`。已验证的 feature-gap、zero-stride、batch-gap view 由 ACLNN
  回退连续化；token/head 置换或正 stride 重叠 descriptor 返回参数错误。
- 未传 `ssm_state_indices` 时 `state_capacity=seq_num`；传入后容量可大于序列数，所有有效 slot 必须位于 `[0,state_capacity)`。
- `ssm_state_indices` 支持 packed `[T]` 和 speculative `[seq_num,max_step]`；活跃序列不得共享正在写入的 state slot。
- 空序列不读取 `ssm_state_indices/num_accepted_tokens`，也不读写 state pool。
- `cu_seqlens` 可选；为空时 BSND 按 batch 行划分，TND 视为一条序列。提供时首项必须为 0，offset
  必须单调不减，末项为有效 token 数且不得超过输入
  token capacity；相邻差值为序列长度。这些值约束由 device kernel 检查，host 只检查 shape/dtype。
- 末项小于 capacity 时，kernel 仅处理有效前缀并逐行跳过零长度序列；padding tail 输出不作保证。
- state 支持 V-first 与 K-first，支持 FP32/BF16，以及原位/非原位 final state。
- 非连续 state 仅允许 slot/head 外层维存在间隔，最后两维必须为行主序稠密矩阵，且外层地址区间不得重叠。
- 接口面向 decode / MTP 短序列；dense 单序列长度与 varlen 每段有效序列长度均必须 `<=8`。
- 仅支持 `layout="BSND"` 和 `layout="TND"`。
- `use_gate_in_kernel=false` 时 `A_log/dt_bias/safe_gate` 必须为空或 false。

## 8. 异常与返回码

| 条件 | 返回码/异常 |
| --- | --- |
| 必选 tensor、workspaceSize 或 executor 为空 | `ACLNN_ERR_PARAM_NULLPTR` |
| rank/shape/dtype/layout 或属性组合非法 | `ACLNN_ERR_PARAM_INVALID` |
| 内部 tensor 创建或 L0 调用失败 | `ACLNN_ERR_INNER_NULLPTR` |
| Python 输入不是 NPU tensor 或 runtime/op_api 未加载 | `RuntimeError` |

参数错误由 Python wrapper 与 aclnn `GetWorkspaceSize` 按各自接口约束返回。
