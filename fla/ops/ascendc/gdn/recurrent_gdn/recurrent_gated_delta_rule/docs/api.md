# RecurrentGatedDeltaRule API 与调用示例

## 1. API 总览

| 通路 | API/入口 | 支持情况 |
| --- | --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.recurrent_gated_delta_rule` | 支持 |
| aclnn | `aclnnRecurrentGatedDeltaRuleGetWorkspaceSize` / `aclnnRecurrentGatedDeltaRule` | 支持 |
| Ascend C `<<<>>>` | `recurrent_gated_delta_rule<<<blockDim, nullptr, stream>>>` | 支持 |
| legacy | `-` | 未实现 |

各已实现入口使用 README 中定义的同一公式、shape、dtype、边界和可选参数语义。

## 2. 公共参数与约束

Shape 符号见[算子 README 附录](../README.md#shape-symbols)。

### 2.1 输入

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

### 2.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `out` | `[T,H_v,V]` | BF16 | recurrent 输出 |
| `state_ref` | `[D_s,H_v,V,K]` | 与输入一致 | 原地更新后的状态 |

### 2.3 属性

| 名称 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `scale_value` | float | `1.0` | 推荐 1/sqrt(K) |

## 3. aclnn API

### 3.1 接口签名

```cpp
ACLNN_API aclnnStatus aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(
const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *beta, aclTensor *stateRef,
const aclTensor *actualSeqLengths, const aclTensor *ssmStateIndices, const aclTensor *g, const aclTensor *gk,
const aclTensor *numAcceptedTokens, float scaleValue, aclTensor *out, uint64_t *workspaceSize,
aclOpExecutor **executor);

aclnnStatus aclnnRecurrentGatedDeltaRule(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

`GetWorkspaceSize` 完成校验并创建 executor；第二段在传入 stream 上异步执行。输入、输出、workspace 和 executor 必须保持有效，直到 stream 完成。

### 3.2 调用示例

```cpp
int32_t deviceId = 0;
ACL_CHECK(aclInit(nullptr));
ACL_CHECK(aclrtSetDevice(deviceId));
aclrtStream stream = nullptr;
ACL_CHECK(aclrtCreateStream(&stream));

// 按 2.1/2.2 的 shape、dtype 和 layout 创建输入/输出 aclTensor。
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
aclnnStatus status = aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(
    query, key, value, beta, stateRef, actualSeqLengths, ssmStateIndices, g, nullptr, numAcceptedTokens, scaleValue, out, &workspaceSize, &executor);
ACLNN_CHECK(status);
void *workspace = nullptr;
if (workspaceSize != 0) {
    ACL_CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
}
ACLNN_CHECK(aclnnRecurrentGatedDeltaRule(workspace, workspaceSize, executor, stream));
ACL_CHECK(aclrtSynchronizeStream(stream));
if (workspace != nullptr) {
    ACL_CHECK(aclrtFree(workspace));
}
// 依次销毁本例创建的 aclTensor、aclIntArray 等 descriptor。
ACL_CHECK(aclrtDestroyStream(stream));
ACL_CHECK(aclrtResetDevice(deviceId));
ACL_CHECK(aclFinalize());
```

`tests/operators/recurrent_gated_delta_rule/routes/test_aclnn_recurrent_gated_delta_rule.cpp` 固化两段式函数签名；tensor descriptor 必须严格按本页公共
参数表构造，不能用物理补齐 shape 替代逻辑 shape。

## 4. `fla_npu.ops.ascendc` API

### 4.1 接口签名

```python
recurrent_gated_delta_rule(query, key, value, beta, state_ref, actual_seq_lengths, ssm_state_indices, *, g=None, gk=None, num_accepted_tokens=None, scale_value=1.0)
```

稳定入口加载 OPP op_api 动态库、创建 aclTensor/aclIntArray、取得当前 NPU stream 并保活异步资源；普通 import 不注册 `torch.ops.npu`。

### 4.2 调用示例

```python
import torch
from fla_npu.ops.ascendc import recurrent_gated_delta_rule

B, L, H_k, H_v, K, V, D_s = 2, 2, 2, 4, 128, 128, 4
T = B * L
q = torch.randn(T, H_k, K, device="npu", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn(T, H_v, V, device="npu", dtype=torch.bfloat16)
beta = torch.rand(T, H_v, device="npu", dtype=torch.bfloat16)
state = torch.zeros(D_s, H_v, V, K, device="npu", dtype=torch.float32)
lengths = torch.tensor([0, L, L], device="npu", dtype=torch.int32)
indices = torch.arange(T, device="npu", dtype=torch.int32)
g = torch.zeros(T, H_v, device="npu", dtype=torch.float32)
out, state = recurrent_gated_delta_rule(q, k, v, beta, state, lengths, indices, g=g, scale_value=K ** -0.5)
torch.npu.synchronize()
assert out.shape == v.shape
```

## 5. Ascend C `<<<>>>` 直调

`blockDim`、workspace 和序列化 tiling data 必须来自同一组 host tiling 结果，不能手写猜测。参数顺序与 kernel 定义保持一致：

```cpp
recurrent_gated_delta_rule<<<blockDim, nullptr, stream>>>(query, key, value, beta, state, cuSeqlens, ssmStateIndices, g, gk, numAcceptedTokens, out, stateOut, workspaceGM, tilingGM);
ACL_CHECK(aclrtSynchronizeStream(stream));
```

可编译的参数声明和 launch 包装位于 `tests/operators/recurrent_gated_delta_rule/routes/test_direct_recurrent_gated_delta_rule.cpp`。

## 6. `torch.ops.npu` API（可选）

当前未注册 `torch.ops.npu` 入口；调用方使用 `fla_npu.ops.ascendc`。

## 7. 平台支持

| 平台 | SOC | 状态 |
| --- | --- | --- |
| A2 | `ascend910b` | 支持 |
| A3 | `ascend910_93` | 支持 |
| A5 | `ascend950` | 支持；存在 arch35 时使用特化 |

## 8. 已知限制

- 每条序列本次有效 token 数不超过 8。
- g/gk 均为空表示单位衰减；二者存在时 dtype 必须为 FP32，shape 分别为 `[T,H_v]`、`[T,H_v,K]`。
- state 槽索引必须位于 `[0,D_s)`；actual_seq_lengths 有效项之和等于 T。
- `H_k/H_v <= 256`、`K/V <= 512` 且 `H_v % H_k == 0`。
- state_ref 为原地更新参数，不能 require_grad；非连续 state 依赖 CANN >= 9.1。

## 9. 异常与返回码

| 条件 | 返回码/异常 |
| --- | --- |
| 任一必选 tensor、workspaceSize 或 executor 为空 | ACLNN_ERR_PARAM_INVALID；workspaceSize/executor 为空为 ACLNN_ERR_PARAM_NULLPTR |
| Q/K/V/beta/out 非 BF16，state 非 BF16/FP32，gate 非 FP32，索引非 INT32 | ACLNN_ERR_PARAM_INVALID |
| rank、T/H/K/V、GVA、state/out、g/gk 或 num_accepted_tokens shape 不匹配 | ACLNN_ERR_PARAM_INVALID |
| 每序列长度超过 8，accepted token 不在 [1,seq_len]，或索引值越界 | 不支持输入；kernel 防护分支提前结束，调用方必须在提交前校验 |

负向 case 的 `expect.return_code` 与消息片段集中定义在 `tests/op_cases/recurrent_gated_delta_rule.json`，修改拦截时必须同步更新。

## 10. 文档自检

- [x] aclnn、Python 与 `<<<>>>` 均提供签名和调用示例。
- [x] Shape 使用模型符号，固定值仅列在已知限制。
- [x] A2/A3/A5、变长 recurrent、g/gk 独立或组合、可选 accepted-token 状态选择、原地 state 与错误码均有说明。
- [x] 主入口为 `fla_npu.ops.ascendc`，未把 Triton 声明为并列正式入口。
