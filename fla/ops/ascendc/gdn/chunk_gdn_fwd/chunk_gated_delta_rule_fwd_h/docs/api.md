# ChunkGatedDeltaRuleFwdH API 与调用示例

## 1. API 总览

| 通路 | API/入口 | 支持情况 |
| --- | --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd_h` | 支持 |
| aclnn | `aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize` / `aclnnChunkGatedDeltaRuleFwdH` | 支持 |
| Ascend C `<<<>>>` | `chunk_gated_delta_rule_fwd_h<<<blockDim, nullptr, stream>>>` | 支持 |
| legacy | `npu_chunk_gated_delta_rule_fwd_h` | 支持（显式加载） |

各已实现入口使用 README 中定义的同一公式、shape、dtype、边界和可选参数语义。

## 2. 公共参数与约束

Shape 符号见[算子 README 附录](../README.md#shape-symbols)。

### 2.1 输入

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

### 2.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `h` | `[B,H_v,N_c,K,V]` | 与 k 一致 | 每个 chunk 起始状态 |
| `v_new` | `[B,H_v,T,V]` | 与 u 一致 | 状态修正后的 Value |
| `final_state` | `[N,H_v,K,V]` | 跟随 initial_state 或 FP32 | 按需返回末状态 |

### 2.3 属性

| 名称 | 类型 | 默认值 | 取值范围 | 说明 |
| --- | --- | --- | --- | --- |
| `output_final_state` | bool | `false` | `{false, true}` | 是否返回末状态 |
| `chunk_size` | int | `64` | `{64, 128}` | chunk 长度 |
| `save_new_value` | bool | `true` | `{true}` | 当前必须 true |
| `use_exp2` | bool | `false` | `{false}` | 当前必须 false |
| `transpose_state_layout` | bool | `false` | `{false}` | 当前必须 false |

## 3. aclnn API

### 3.1 接口签名

```cpp
aclnnStatus aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize(
const aclTensor *k,
const aclTensor *w,
const aclTensor *u,
const aclTensor *gOptional,
const aclTensor *gkOptional,
const aclTensor *initalStateOptional,
bool outputFinalState,
int64_t chunkSize,
bool saveNewValue,
const aclIntArray *cuSeqlensOptional,
const aclIntArray *chunkIndicesOptional,
bool useExp2,
bool transposeStateLayout,
const aclTensor *hOut,
const aclTensor *vNewOut,
const aclTensor *finalStateOut,
uint64_t *workspaceSize,
aclOpExecutor **executor);

aclnnStatus aclnnChunkGatedDeltaRuleFwdH(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
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
aclnnStatus status = aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize(
    k, w, u, g, nullptr, nullptr, true, chunkSize, true, nullptr, nullptr, false, false, h, vNew, finalState, &workspaceSize, &executor);
ACLNN_CHECK(status);
void *workspace = nullptr;
if (workspaceSize != 0) {
    ACL_CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
}
ACLNN_CHECK(aclnnChunkGatedDeltaRuleFwdH(workspace, workspaceSize, executor, stream));
ACL_CHECK(aclrtSynchronizeStream(stream));
if (workspace != nullptr) {
    ACL_CHECK(aclrtFree(workspace));
}
// 依次销毁本例创建的 aclTensor、aclIntArray 等 descriptor。
ACL_CHECK(aclrtDestroyStream(stream));
ACL_CHECK(aclrtResetDevice(deviceId));
ACL_CHECK(aclFinalize());
```

`tests/operators/chunk_gated_delta_rule_fwd_h/routes/test_aclnn_chunk_gated_delta_rule_fwd_h.cpp` 固化两段式函数签名；tensor descriptor 必须严格按本页公共
参数表构造，不能用物理补齐 shape 替代逻辑 shape。

## 4. `fla_npu.ops.ascendc` API

### 4.1 接口签名

```python
chunk_gated_delta_rule_fwd_h(k, w, u, g=None, *, gk=None, initial_state=None, output_final_state=False, chunk_size=None, save_new_value=True, cu_seqlens=None, chunk_indices=None, use_exp2=False, transpose_state_layout=False)
```

稳定入口加载 OPP op_api 动态库、创建 aclTensor/aclIntArray、取得当前 NPU stream 并保活异步资源；普通 import 不注册 `torch.ops.npu`。

### 4.2 调用示例

```python
import torch
from fla_npu.ops.ascendc import chunk_gated_delta_rule_fwd_h

B, H_k, H_v, T, K, V, C = 1, 2, 4, 128, 128, 128, 64
k = torch.randn(B, H_k, T, K, device="npu", dtype=torch.bfloat16)
w = torch.randn(B, H_v, T, K, device="npu", dtype=torch.bfloat16)
u = torch.randn(B, H_v, T, V, device="npu", dtype=torch.bfloat16)
g = -torch.rand(B, H_v, T, device="npu", dtype=torch.float32).cumsum(-1)
h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
    k, w, u, g, chunk_size=C, output_final_state=True)
torch.npu.synchronize()
assert v_new.shape == u.shape and final_state.shape == (B, H_v, K, V)
```

## 5. Ascend C `<<<>>>` 直调

`blockDim`、workspace 和序列化 tiling data 必须来自同一组 host tiling 结果，不能手写猜测。参数顺序与 kernel 定义保持一致：

```cpp
chunk_gated_delta_rule_fwd_h<<<blockDim, nullptr, stream>>>(k, w, u, g, gk, inital_state, cu_seqlens, chunk_indices, h, v_new, final_state, workspace, tiling);
ACL_CHECK(aclrtSynchronizeStream(stream));
```

可编译的参数声明和 launch 包装位于 `tests/operators/chunk_gated_delta_rule_fwd_h/routes/test_direct_chunk_gated_delta_rule_fwd_h.cpp`。

## 6. `torch.ops.npu` API（可选）

```python
import torch
import fla_npu

fla_npu.load_legacy_torch_ops()

B, H_k, H_v, T, K, V, C = 1, 2, 4, 128, 128, 128, 64
k = torch.randn(B, H_k, T, K, device="npu", dtype=torch.bfloat16)
w = torch.randn(B, H_v, T, K, device="npu", dtype=torch.bfloat16)
u = torch.randn(B, H_v, T, V, device="npu", dtype=torch.bfloat16)
g = -torch.rand(B, H_v, T, device="npu", dtype=torch.float32).cumsum(-1)
h, v_new, final_state = torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h(
    k, w, u, g, chunk_size=C, output_final_state=True)
torch.npu.synchronize()
assert v_new.shape == u.shape and final_state.shape == (B, H_v, K, V)
```

## 7. 平台支持

| 平台 | SOC | 状态 |
| --- | --- | --- |
| A2 | `ascend910b` | 支持 |
| A3 | `ascend910_93` | 支持 |
| A5 | `ascend950` | 支持；存在 arch35 时使用特化 |

## 8. 已知限制

- `K` 仅支持 128，`V` 仅支持 128/256，`chunk_size` 仅支持 64/128。
- `g` 与 `gk` 至少提供一个；`H_v % H_k == 0`。
- 变长序列当前仅支持物理 `B=1`，索引必须完整且 sequence-major。
- `save_new_value=true`、`use_exp2=false`、`transpose_state_layout=false`。

## 9. 异常与返回码

| 条件 | 返回码/异常 |
| --- | --- |
| k/w/u/hOut/vNewOut、workspaceSize 或 executor 为空 | ACLNN_ERR_PARAM_INVALID；workspaceSize/executor 为空为 ACLNN_ERR_PARAM_NULLPTR |
| g 与 gk 同时为空，或 g/gk shape/dtype 不匹配 | ACLNN_ERR_PARAM_INVALID / Python RuntimeError |
| save_new_value=false、use_exp2=true、transpose_state_layout=true | ACLNN_ERR_PARAM_INVALID / Python RuntimeError |
| cu_seqlens/chunk_indices 未成对提供，或 chunk_size 非 64/128 | ACLNN_ERR_PARAM_INVALID / Python RuntimeError |
| K/W/U、initial/final state、H/V_new 或 GVA shape 不匹配 | ACLNN_ERR_PARAM_INVALID |

负向 case 的 `expect.return_code` 与消息片段集中定义在 `tests/op_cases/chunk_gated_delta_rule_fwd_h.json`，修改拦截时必须同步更新。

## 10. 文档自检

- [x] aclnn、Python 与 `<<<>>>` 均提供签名和调用示例。
- [x] Shape 使用模型符号，固定值仅列在已知限制。
- [x] A2/A3/A5、定长/变长序列、g/gk、可选 initial/final state 与错误码均有说明。
- [x] 主入口为 `fla_npu.ops.ascendc`，未把 Triton 声明为并列正式入口。
