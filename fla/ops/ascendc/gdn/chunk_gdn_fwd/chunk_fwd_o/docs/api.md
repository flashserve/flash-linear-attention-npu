# ChunkFwdO API 与调用示例

## 1. API 总览

| 通路 | API/入口 | 支持情况 |
| --- | --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.chunk_fwd_o` | 支持 |
| aclnn | `aclnnChunkFwdOGetWorkspaceSize` / `aclnnChunkFwdO` | 支持 |
| Ascend C `<<<>>>` | `chunk_fwd_o<<<blockDim, nullptr, stream>>>` | 支持 |
| legacy | `npu_chunk_fwd_o` | 支持（显式加载） |

各已实现入口使用 README 中定义的同一公式、shape、dtype、边界和可选参数语义。

## 2. 公共参数与约束

Shape 符号见[算子 README 附录](../README.md#shape-symbols)。

### 2.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Layout | 说明 |
| --- | --- | --- | --- | --- | --- |
| `q` | 必选 | `[B,H_k,T,K]` | FP16/BF16 | BNSD | Query |
| `k` | 必选 | `[B,H_k,T,K]` | FP16/BF16 | BNSD | Key |
| `v` | 必选 | `[B,H_v,T,V]` | FP16/BF16 | BNSD | Value |
| `h` | 必选 | `[B,H_v,N_c,K,V]` | FP16/BF16 | ND | 每个 chunk 的起始状态 |
| `g` | 必选 | `[B,H_v,T]` | FP16/BF16/FP32 | BNS | 累积标量 gate |
| `g_gamma` | 预留 | `-` | - | - | 上层兼容参数，当前必须为 None |
| `cu_seqlens` | 可选 | `[N+1]` | INT64 | ND | 变长序列累计长度 |
| `chunk_indices` | 可选 | `[2*N_c]` | INT64 | ND | 展平 chunk 索引 |

### 2.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `o` | `[B,H_v,T,V]` | 与 v 一致 | chunk 注意力输出 |

### 2.3 属性

| 名称 | 类型 | 默认值 | 取值范围 | 说明 |
| --- | --- | --- | --- | --- |
| `scale` | double | `无` | - | Query 缩放 |
| `chunk_size` | int | `64` | `{64, 128}` | chunk 长度 |
| `transpose_state_layout` | bool | `false` | `{false}` | 预留参数，当前必须 false |

## 3. aclnn API

### 3.1 接口签名

```cpp
aclnnStatus aclnnChunkFwdOGetWorkspaceSize(
const aclTensor *q,
const aclTensor *k,
const aclTensor *v,
const aclTensor *h,
const aclTensor *g,
const aclIntArray *cuSeqlensOptional,
const aclIntArray *chunkOffsetsOptional,
double scale,
int64_t chunkSize,
const aclTensor *oOut,
uint64_t *workspaceSize,
aclOpExecutor **executor);

aclnnStatus aclnnChunkFwdO(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
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
aclnnStatus status = aclnnChunkFwdOGetWorkspaceSize(
    q, k, v, h, g, nullptr, nullptr, scale, chunkSize, o, &workspaceSize, &executor);
ACLNN_CHECK(status);
void *workspace = nullptr;
if (workspaceSize != 0) {
    ACL_CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
}
ACLNN_CHECK(aclnnChunkFwdO(workspace, workspaceSize, executor, stream));
ACL_CHECK(aclrtSynchronizeStream(stream));
if (workspace != nullptr) {
    ACL_CHECK(aclrtFree(workspace));
}
// 依次销毁本例创建的 aclTensor、aclIntArray 等 descriptor。
ACL_CHECK(aclrtDestroyStream(stream));
ACL_CHECK(aclrtResetDevice(deviceId));
ACL_CHECK(aclFinalize());
```

`tests/operators/chunk_fwd_o/routes/test_aclnn_chunk_fwd_o.cpp` 固化两段式函数签名；tensor descriptor 必须严格按本页公共
参数表构造，不能用物理补齐 shape 替代逻辑 shape。

## 4. `fla_npu.ops.ascendc` API

### 4.1 接口签名

```python
chunk_fwd_o(q, k, v, h, scale, *, g=None, g_gamma=None, cu_seqlens=None, chunk_indices=None, chunk_size=None, transpose_state_layout=False)
```

稳定入口加载 OPP op_api 动态库、创建 aclTensor/aclIntArray、取得当前 NPU stream 并保活异步资源；普通 import 不注册 `torch.ops.npu`。

### 4.2 调用示例

```python
import torch
from fla_npu.ops.ascendc import chunk_fwd_o

B, H_k, H_v, T, K, V, chunk_size = 1, 2, 4, 129, 128, 128, 64
N_c = (T + chunk_size - 1) // chunk_size
q = torch.randn(B, H_k, T, K, device="npu", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn(B, H_v, T, V, device="npu", dtype=torch.float16)
h = torch.randn(B, H_v, N_c, K, V, device="npu", dtype=torch.float16)
g = -torch.rand(B, H_v, T, device="npu", dtype=torch.float32).cumsum(-1)
o = chunk_fwd_o(q, k, v, h, K ** -0.5, g=g, chunk_size=chunk_size)
torch.npu.synchronize()
assert o.shape == v.shape
```

## 5. Ascend C `<<<>>>` 直调

`blockDim`、workspace 和序列化 tiling data 必须来自同一组 host tiling 结果，不能手写猜测。参数顺序与 kernel 定义保持一致：

```cpp
chunk_fwd_o<<<blockDim, nullptr, stream>>>(q, k, v, h, g, cu_seqlens, chunk_offsets, o, workspace, tiling);
ACL_CHECK(aclrtSynchronizeStream(stream));
```

可编译的参数声明和 launch 包装位于 `tests/operators/chunk_fwd_o/routes/test_direct_chunk_fwd_o.cpp`。

## 6. `torch.ops.npu` API（可选）

```python
import torch
import fla_npu

fla_npu.load_legacy_torch_ops()

B, H_k, H_v, T, K, V, chunk_size = 1, 2, 4, 129, 128, 128, 64
N_c = (T + chunk_size - 1) // chunk_size
q = torch.randn(B, H_k, T, K, device="npu", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn(B, H_v, T, V, device="npu", dtype=torch.float16)
h = torch.randn(B, H_v, N_c, K, V, device="npu", dtype=torch.float16)
g = -torch.rand(B, H_v, T, device="npu", dtype=torch.float32).cumsum(-1)
o = torch.ops.npu.npu_chunk_fwd_o(q, k, v, h, K ** -0.5, g=g, chunk_size=chunk_size)
torch.npu.synchronize()
assert o.shape == v.shape
```

## 7. 平台支持

| 平台 | SOC | 状态 |
| --- | --- | --- |
| A2 | `ascend910b` | 支持 |
| A3 | `ascend910_93` | 支持 |
| A5 | `ascend950` | 支持；存在 arch35 时使用特化 |

## 8. 已知限制

- `K` 仅支持 128，`V` 仅支持 128/256，`chunk_size` 仅支持 64/128。
- 必须满足 `H_v % H_k == 0`，h 的 chunk 数必须与索引推导一致。
- 变长序列当前仅支持物理 `B=1`，两个索引必须同时提供。
- `g` 是 kernel 必选输入；`g_gamma` 必须为 None，`transpose_state_layout` 必须为 false。

## 9. 异常与返回码

| 条件 | 返回码/异常 |
| --- | --- |
| q/k/v/h/g/oOut、workspaceSize 或 executor 为空 | ACLNN_ERR_PARAM_INVALID；workspaceSize/executor 为空为 ACLNN_ERR_PARAM_NULLPTR |
| g_gamma 非 None、transpose_state_layout=true | Python RuntimeError |
| cu_seqlens/chunk_indices 未成对提供，或 chunk_size 非 64/128 | ACLNN_ERR_PARAM_INVALID / Python RuntimeError |
| rank、B/T、GVA head、H/K/V/output shape 或 dtype 不匹配 | ACLNN_ERR_PARAM_INVALID |

负向 case 的 `expect.return_code` 与消息片段集中定义在 `tests/op_cases/chunk_fwd_o.json`，修改拦截时必须同步更新。

## 10. 文档自检

- [x] aclnn、Python 与 `<<<>>>` 均提供签名和调用示例。
- [x] Shape 使用模型符号，固定值仅列在已知限制。
- [x] A2/A3/A5、定长/变长序列、GVA、整块/尾块；g 为必选标量 gate 与错误码均有说明。
- [x] 主入口为 `fla_npu.ops.ascendc`，未把 Triton 声明为并列正式入口。
