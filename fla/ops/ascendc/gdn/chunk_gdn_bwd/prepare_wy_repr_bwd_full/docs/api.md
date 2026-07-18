# PrepareWyReprBwdFull API 与调用示例

## 1. API 总览

| 通路 | API/入口 | 支持情况 |
| --- | --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.prepare_wy_repr_bwd_full` | 支持 |
| aclnn | `aclnnPrepareWyReprBwdFullGetWorkspaceSize` / `aclnnPrepareWyReprBwdFull` | 支持 |
| Ascend C `<<<>>>` | `prepare_wy_repr_bwd_full<<<blockDim, nullptr, stream>>>` | 支持 |
| legacy | `npu_prepare_wy_repr_bwd_full` | 支持（显式加载） |

各已实现入口使用 README 中定义的同一公式、shape、dtype、边界和可选参数语义。

## 2. 公共参数与约束

Shape 符号见[算子 README 附录](../README.md#shape-symbols)。

### 2.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Layout | 说明 |
| --- | --- | --- | --- | --- | --- |
| `k` | 必选 | `[B,H_k,T,K]` | FP16/BF16 | BNSD | Key |
| `v` | 必选 | `[B,H_v,T,V]` | FP16/BF16 | BNSD | Value |
| `beta` | 必选 | `[B,H_v,T]` | FP16/BF16/FP32 | BNS | WY 权重 |
| `A` | 必选 | `[B,H_v,T,C]` | FP16/BF16 | BNSD | 前向局部矩阵 |
| `dA` | 必选 | `[B,H_v,T,C]` | FP16/BF16 | BNSD | A 梯度 |
| `dw` | 必选 | `[B,H_v,T,K]` | FP16/BF16 | BNSD | W 梯度 |
| `du` | 必选 | `[B,H_v,T,V]` | FP16/BF16 | BNSD | U 梯度 |
| `g` | 必选 | `[B,H_v,T]` | FP16/BF16/FP32 | BNS | chunk-local gate |
| `cu_seqlens` | 可选 | `[N+1]` | INT64 | ND | varlen 累计长度 |
| `chunk_indices` | 可选 | `[2*N_c]` | INT64 | ND | 展平 chunk 索引 |

### 2.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `dk` | `[B,H_k,T,K]` | 与 k 一致 | Key 梯度；GVA value head 已归约 |
| `dv` | `[B,H_v,T,V]` | 与 v 一致 | Value 梯度 |
| `dbeta` | `[B,H_v,T]` | 与 beta 一致 | beta 梯度 |
| `dg` | `[B,H_v,T]` | 与 g 一致 | gate 梯度 |

### 2.3 属性

| 名称 | 类型 | 默认值 | 取值范围 | 说明 |
| --- | --- | --- | --- | --- |
| `chunk_size` | int | `无` | `{64, 128}` | 必须等于 A/dA 最后一维 |

## 3. aclnn API

### 3.1 接口签名

```cpp
aclnnStatus aclnnPrepareWyReprBwdFullGetWorkspaceSize(
const aclTensor *k,
const aclTensor *v,
const aclTensor *beta,
const aclTensor *a,
const aclTensor *dA,
const aclTensor *dw,
const aclTensor *du,
const aclTensor *g,
const aclIntArray *cuSeqlensOptional,
const aclIntArray *chunkIndicesOptional,
int64_t chunkSize,
const aclTensor *dkOut,
const aclTensor *dvOut,
const aclTensor *dbetaOut,
const aclTensor *dgOut,
uint64_t *workspaceSize,
aclOpExecutor **executor);

aclnnStatus aclnnPrepareWyReprBwdFull(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
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
aclnnStatus status = aclnnPrepareWyReprBwdFullGetWorkspaceSize(
    k, v, beta, A, dA, dw, du, g, nullptr, nullptr, chunkSize, dk, dv, dbeta, dg, &workspaceSize, &executor);
ACLNN_CHECK(status);
void *workspace = nullptr;
if (workspaceSize != 0) {
    ACL_CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
}
ACLNN_CHECK(aclnnPrepareWyReprBwdFull(workspace, workspaceSize, executor, stream));
ACL_CHECK(aclrtSynchronizeStream(stream));
if (workspace != nullptr) {
    ACL_CHECK(aclrtFree(workspace));
}
// 依次销毁本例创建的 aclTensor、aclIntArray 等 descriptor。
ACL_CHECK(aclrtDestroyStream(stream));
ACL_CHECK(aclrtResetDevice(deviceId));
ACL_CHECK(aclFinalize());
```

`tests/operators/prepare_wy_repr_bwd_full/routes/test_aclnn_prepare_wy_repr_bwd_full.cpp` 固化两段式函数签名；tensor descriptor 必须严格按本页公共
参数表构造，不能用物理补齐 shape 替代逻辑 shape。

## 4. `fla_npu.ops.ascendc` API

### 4.1 接口签名

```python
prepare_wy_repr_bwd_full(k, v, beta, A, dA, dw, du, g, chunk_size, *, cu_seqlens=None, chunk_indices=None)
```

稳定入口加载 OPP op_api 动态库、创建 aclTensor/aclIntArray、取得当前 NPU stream 并保活异步资源；普通 import 不注册 `torch.ops.npu`。

### 4.2 调用示例

```python
import torch
from fla_npu.ops.ascendc import prepare_wy_repr_bwd_full

B, H_k, H_v, T, K, V, C = 1, 2, 4, 128, 128, 256, 64
k = torch.randn(B, H_k, T, K, device="npu", dtype=torch.bfloat16)
v = torch.randn(B, H_v, T, V, device="npu", dtype=torch.bfloat16)
beta = torch.rand(B, H_v, T, device="npu", dtype=torch.float32)
A = torch.randn(B, H_v, T, C, device="npu", dtype=torch.bfloat16)
dA = torch.randn_like(A)
dw = torch.randn(B, H_v, T, K, device="npu", dtype=torch.bfloat16)
du = torch.randn_like(v)
g = -torch.rand(B, H_v, T, device="npu", dtype=torch.float32).cumsum(-1)
dk, dv, dbeta, dg = prepare_wy_repr_bwd_full(k, v, beta, A, dA, dw, du, g, C)
torch.npu.synchronize()
assert dk.shape == k.shape and dv.shape == v.shape
```

## 5. Ascend C `<<<>>>` 直调

`blockDim`、workspace 和序列化 tiling data 必须来自同一组 host tiling 结果，不能手写猜测。参数顺序与 kernel 定义保持一致：

```cpp
prepare_wy_repr_bwd_full<<<blockDim, nullptr, stream>>>(k, v, beta, A, dA, dw, du, g, cu_seqlens, chunk_indices, dk, dv, dbeta, dg, workspace, tiling);
ACL_CHECK(aclrtSynchronizeStream(stream));
```

可编译的参数声明和 launch 包装位于 `tests/operators/prepare_wy_repr_bwd_full/routes/test_direct_prepare_wy_repr_bwd_full.cpp`。

## 6. `torch.ops.npu` API（可选）

```python
import torch
import fla_npu

fla_npu.load_legacy_torch_ops()

B, H_k, H_v, T, K, V, C = 1, 2, 4, 128, 128, 256, 64
k = torch.randn(B, H_k, T, K, device="npu", dtype=torch.bfloat16)
v = torch.randn(B, H_v, T, V, device="npu", dtype=torch.bfloat16)
beta = torch.rand(B, H_v, T, device="npu", dtype=torch.float32)
A = torch.randn(B, H_v, T, C, device="npu", dtype=torch.bfloat16)
dA = torch.randn_like(A)
dw = torch.randn(B, H_v, T, K, device="npu", dtype=torch.bfloat16)
du = torch.randn_like(v)
g = -torch.rand(B, H_v, T, device="npu", dtype=torch.float32).cumsum(-1)
dk, dv, dbeta, dg = torch.ops.npu.npu_prepare_wy_repr_bwd_full(k, v, beta, A, dA, dw, du, g, C)
torch.npu.synchronize()
assert dk.shape == k.shape and dv.shape == v.shape
```

## 7. 平台支持

| 平台 | SOC | 状态 |
| --- | --- | --- |
| A2 | `ascend910b` | 支持 |
| A3 | `ascend910_93` | 支持 |
| A5 | `ascend950` | 支持；存在 arch35 时使用特化 |

## 8. 已知限制

- `K` 仅支持 128，`V` 仅支持 128/256。
- `chunk_size` 仅支持 64/128，并须等于 A/dA 最后一维。
- 必须满足 `H_v % H_k == 0`；varlen 当前仅支持物理 `B=1`。
- `cu_seqlens` 与 `chunk_indices` 必须同时提供或同时省略。

## 9. 异常与返回码

| 条件 | 返回码/异常 |
| --- | --- |
| workspaceSize 或 executor 为空 | ACLNN_ERR_PARAM_NULLPTR |
| 必选 tensor 为空，chunk_size 非 64/128，或 varlen 元数据只提供一个 | ACLNN_ERR_PARAM_INVALID |
| shape/dtype/GVA 不受模板支持 | tiling 失败；aclnn 执行返回 ACLNN_ERR_INNER |
| 执行器或 kernel launch 失败 | ACLNN_ERR_INNER |

负向 case 的 `expect.return_code` 与消息片段集中定义在 `tests/op_cases/prepare_wy_repr_bwd_full.json`，修改拦截时必须同步更新。

## 10. 文档自检

- [x] aclnn、Python 与 `<<<>>>` 均提供签名和调用示例。
- [x] Shape 使用模型符号，固定值仅列在已知限制。
- [x] A2/A3/A5、fixed 与 varlen，支持 GVA 与错误码均有说明。
- [x] 主入口为 `fla_npu.ops.ascendc`，未把 Triton 声明为并列正式入口。
