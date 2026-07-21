# ChunkBwdDvLocal API 与调用示例

## 1. API 总览

| 通路 | API/入口 | 支持情况 |
| --- | --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.chunk_bwd_dv_local` | 支持 |
| aclnn | `aclnnChunkBwdDvLocalGetWorkspaceSize` / `aclnnChunkBwdDvLocal` | 支持 |
| Ascend C `<<<>>>` | `chunk_bwd_dv_local<<<blockDim, nullptr, stream>>>` | 支持 |
| legacy | `npu_chunk_bwd_dv_local` | 支持（显式加载） |

各已实现入口使用 README 中定义的同一公式、shape、dtype、边界和可选参数语义。

## 2. 公共参数与约束

Shape 符号统一引用[GDN 模型符号表](../../../README.md#model-shape-symbols)。

### 2.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Layout | 说明 |
| --- | --- | --- | --- | --- | --- |
| `q` | 必选 | `[B,H_k,T,K]` | FP16/BF16 | BNSD | Query |
| `k` | 必选 | `[B,H_k,T,K]` | FP16/BF16 | BNSD | Key，与 q 同形 |
| `d_o` | 必选 | `[B,H_v,T,V]` | FP16/BF16 | BNSD | 输出梯度 |
| `g` | 必选 | `[B,H_v,T]` | FP16/BF16/FP32 | BNS | chunk-local 累积 gate |
| `g_gamma` | 预留 | `-` | FP32 | ND | 当前必须为 None |
| `A` | 预留 | `-` | FP16/BF16 | ND | 当前必须为 None |
| `cu_seqlens` | 可选 | `[N+1]` | INT64 | ND | 变长序列累计长度 |
| `chunk_indices` | 可选 | `[2*N_c]` | INT64 | ND | 展平的 (seq_id,chunk_id) |

### 2.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `d_v` | `[B,H_v,T,V]` | 与 d_o 一致 | Value 的 chunk-local 梯度 |

### 2.3 属性

| 名称 | 类型 | 默认值 | 取值范围 | 说明 |
| --- | --- | --- | --- | --- |
| `scale` | double | `无` | - | 通常为 1/sqrt(K) |
| `chunk_size` | int | `无` | `{64, 128}` | chunk 长度 |

## 3. aclnn API

### 3.1 接口签名

```cpp
aclnnStatus aclnnChunkBwdDvLocalGetWorkspaceSize(
const aclTensor *q,
const aclTensor *k,
const aclTensor *dO,
const aclTensor *g,
const aclTensor *gGammaOptional,
const aclTensor *aOptional,
const aclIntArray *cuSeqlensOptional,
const aclIntArray *chunkIndicesOptional,
double scale,
int64_t chunkSize,
const aclTensor *out,
uint64_t *workspaceSize,
aclOpExecutor **executor);

aclnnStatus aclnnChunkBwdDvLocal(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
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
aclnnStatus status = aclnnChunkBwdDvLocalGetWorkspaceSize(
    q, k, dO, g, nullptr, nullptr, nullptr, nullptr, scale, chunkSize, out, &workspaceSize, &executor);
ACLNN_CHECK(status);
void *workspace = nullptr;
if (workspaceSize != 0) {
    ACL_CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
}
ACLNN_CHECK(aclnnChunkBwdDvLocal(workspace, workspaceSize, executor, stream));
ACL_CHECK(aclrtSynchronizeStream(stream));
if (workspace != nullptr) {
    ACL_CHECK(aclrtFree(workspace));
}
// 依次销毁本例创建的 aclTensor、aclIntArray 等 descriptor。
ACL_CHECK(aclrtDestroyStream(stream));
ACL_CHECK(aclrtResetDevice(deviceId));
ACL_CHECK(aclFinalize());
```

`tests/operators/chunk_bwd_dv_local/routes/test_aclnn_chunk_bwd_dv_local.cpp` 固化两段式函数签名；tensor descriptor 必须严格按本页公共
参数表构造，不能用物理补齐 shape 替代逻辑 shape。

## 4. `fla_npu.ops.ascendc` API

### 4.1 接口签名

```python
chunk_bwd_dv_local(q, k, d_o, g, scale, chunk_size, *, g_gamma=None, A=None, cu_seqlens=None, chunk_indices=None)
```

稳定入口加载 OPP op_api 动态库、创建 aclTensor/aclIntArray、取得当前 NPU stream 并保活异步资源；普通 import 不注册 `torch.ops.npu`。

### 4.2 调用示例

```python
import torch
from fla_npu.ops.ascendc import chunk_bwd_dv_local

B, H_k, H_v, T, K, V, chunk_size = 1, 2, 4, 129, 128, 128, 64
q = torch.randn(B, H_k, T, K, device="npu", dtype=torch.bfloat16)
k = torch.randn_like(q)
d_o = torch.randn(B, H_v, T, V, device="npu", dtype=torch.bfloat16)
g = -torch.rand(B, H_v, T, device="npu", dtype=torch.float32).cumsum(-1)
d_v = chunk_bwd_dv_local(q, k, d_o, g, K ** -0.5, chunk_size)
torch.npu.synchronize()
assert d_v.shape == d_o.shape
```

## 5. Ascend C `<<<>>>` 直调

`blockDim`、workspace 和序列化 tiling data 必须来自同一组 host tiling 结果，不能手写猜测。参数顺序与 kernel 定义保持一致：

```cpp
chunk_bwd_dv_local<<<blockDim, nullptr, stream>>>(q, k, d_o, g, g_gamma, A, cu_seqlens, chunk_indices, d_v, workspace, tiling);
ACL_CHECK(aclrtSynchronizeStream(stream));
```

可编译的参数声明和 launch 包装位于 `tests/operators/chunk_bwd_dv_local/routes/test_direct_chunk_bwd_dv_local.cpp`。

## 6. `torch.ops.npu` API（可选）

```python
import torch
import fla_npu

fla_npu.load_legacy_torch_ops()

B, H_k, H_v, T, K, V, chunk_size = 1, 2, 4, 129, 128, 128, 64
q = torch.randn(B, H_k, T, K, device="npu", dtype=torch.bfloat16)
k = torch.randn_like(q)
d_o = torch.randn(B, H_v, T, V, device="npu", dtype=torch.bfloat16)
g = -torch.rand(B, H_v, T, device="npu", dtype=torch.float32).cumsum(-1)
d_v = torch.ops.npu.npu_chunk_bwd_dv_local(q, k, d_o, g, K ** -0.5, chunk_size)
torch.npu.synchronize()
assert d_v.shape == d_o.shape
```

## 7. 平台支持

| 平台 | SOC | 状态 |
| --- | --- | --- |
| A2 | `ascend910b` | 支持 |
| A3 | `ascend910_93` | 支持 |
| A5 | `ascend950` | 支持；存在 arch35 时使用特化 |

## 8. 已知限制

- `K` 仅支持 128，`V` 仅支持 128/256。
- `chunk_size` 仅支持 64/128。
- 必须满足 `H_v % H_k == 0`；变长序列当前仅支持物理 `B=1`。
- `g_gamma` 和 `A` 尚未实现，必须传 `None`。

## 9. 异常与返回码

| 条件 | 返回码/异常 |
| --- | --- |
| workspaceSize 或 executor 为空 | ACLNN_ERR_PARAM_NULLPTR |
| 必选 tensor 为空 | ACLNN_ERR_PARAM_INVALID |
| g_gamma/A 非空，或变长序列元数据只提供一个 | ACLNN_ERR_PARAM_INVALID |
| shape/dtype/GVA/chunk_size 不受模板支持 | tiling 失败；aclnn 执行返回 ACLNN_ERR_INNER |
| Python g_gamma/A 非空 | RuntimeError |

负向 case 的 `expect.return_code` 与消息片段集中定义在 `tests/op_cases/chunk_bwd_dv_local.json`，修改拦截时必须同步更新。

## 10. 文档自检

- [x] aclnn、Python 与 `<<<>>>` 均提供签名和调用示例。
- [x] Shape 使用模型符号，固定值仅列在已知限制。
- [x] A2/A3/A5、定长与变长序列；变长序列的两个索引必须同时提供 与错误码均有说明。
- [x] 主入口为 `fla_npu.ops.ascendc`，未把 Triton 声明为并列正式入口。
