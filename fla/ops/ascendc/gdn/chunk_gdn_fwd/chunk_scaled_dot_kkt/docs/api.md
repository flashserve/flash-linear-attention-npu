# ChunkScaledDotKkt API 与调用示例

## 1. API 总览

| 通路 | API/入口 | 支持情况 |
| --- | --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.chunk_scaled_dot_kkt` | 支持 |
| aclnn | `aclnnChunkScaledDotKktGetWorkspaceSize` / `aclnnChunkScaledDotKkt` | 支持 |
| Ascend C `<<<>>>` | `chunk_scaled_dot_kkt<<<blockDim, nullptr, stream>>>` | 支持 |
| legacy | `npu_chunk_scaled_dot_kkt` | 支持（显式加载） |

各已实现入口使用 README 中定义的同一公式、shape、dtype、边界和可选参数语义。

## 2. 公共参数与约束

Shape 符号见[算子 README 附录](../README.md#shape-symbols)。

### 2.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Layout | 说明 |
| --- | --- | --- | --- | --- | --- |
| `k` | 必选 | `[B,H_k,T,K]` | FP16/BF16 | BNSD | Key |
| `g` | 必选 | `[B,H_v,T]` | FP32 | BNS | 累积 gate |
| `beta` | 必选 | `[B,H_v,T]` | FP32 | BNS | 行缩放 |
| `cu_seqlens` | 可选 | `[N+1]` | INT64 | ND | varlen 累计长度 |
| `chunk_indices` | 可选 | `[2*N_c]` | INT64 | ND | 展平 chunk 索引 |

### 2.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `A` | `[B,H_k,T,C]` | FP32 | 严格下三角 scaled KKT |

### 2.3 属性

| 名称 | 类型 | 默认值 | 取值范围 | 说明 |
| --- | --- | --- | --- | --- |
| `chunk_size` | int | `64` | `{16, 32, 64, 128}` | chunk 长度 |

## 3. aclnn API

### 3.1 接口签名

```cpp
aclnnStatus aclnnChunkScaledDotKktGetWorkspaceSize(/* 参数见本页公共参数表 */, uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnChunkScaledDotKkt(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
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
aclnnStatus status = aclnnChunkScaledDotKktGetWorkspaceSize(
    k, g, beta, nullptr, nullptr, chunkSize, A, &workspaceSize, &executor);
ACLNN_CHECK(status);
void *workspace = nullptr;
if (workspaceSize != 0) {
    ACL_CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
}
ACLNN_CHECK(aclnnChunkScaledDotKkt(workspace, workspaceSize, executor, stream));
ACL_CHECK(aclrtSynchronizeStream(stream));
if (workspace != nullptr) {
    ACL_CHECK(aclrtFree(workspace));
}
// 依次销毁本例创建的 aclTensor、aclIntArray 等 descriptor。
ACL_CHECK(aclrtDestroyStream(stream));
ACL_CHECK(aclrtResetDevice(deviceId));
ACL_CHECK(aclFinalize());
```

`tests/operators/chunk_scaled_dot_kkt/routes/test_aclnn_chunk_scaled_dot_kkt.cpp` 固化两段式函数签名；tensor descriptor 必须严格按本页公共
参数表构造，不能用物理补齐 shape 替代逻辑 shape。

## 4. `fla_npu.ops.ascendc` API

### 4.1 接口签名

```python
chunk_scaled_dot_kkt(k, g, beta, *, cu_seqlens=None, chunk_indices=None, chunk_size=64)
```

稳定入口加载 OPP op_api 动态库、创建 aclTensor/aclIntArray、取得当前 NPU stream 并保活异步资源；普通 import 不注册 `torch.ops.npu`。

### 4.2 调用示例

```python
import torch
from fla_npu.ops.ascendc import chunk_scaled_dot_kkt

B, H_k, H_v, T, K, C = 1, 2, 4, 129, 128, 64
k = torch.randn(B, H_k, T, K, device="npu", dtype=torch.float16)
g = -torch.rand(B, H_v, T, device="npu", dtype=torch.float32).cumsum(-1)
beta = torch.rand(B, H_v, T, device="npu", dtype=torch.float32)
A = chunk_scaled_dot_kkt(k, g, beta, chunk_size=C)
torch.npu.synchronize()
assert A.shape == (B, H_k, T, C) and A.dtype == torch.float32
```

## 5. Ascend C `<<<>>>` 直调

`blockDim`、workspace 和序列化 tiling data 必须来自同一组 host tiling 结果，不能手写猜测。参数顺序与 kernel 定义保持一致：

```cpp
chunk_scaled_dot_kkt<<<blockDim, nullptr, stream>>>(k, g, beta, cuSeqlens, chunkIndices, A, workspace, tiling);
ACL_CHECK(aclrtSynchronizeStream(stream));
```

可编译的参数声明和 launch 包装位于 `tests/operators/chunk_scaled_dot_kkt/routes/test_direct_chunk_scaled_dot_kkt.cpp`。

## 6. `torch.ops.npu` API（可选）

```python
import torch
import fla_npu

fla_npu.load_legacy_torch_ops()

B, H_k, H_v, T, K, C = 1, 2, 4, 129, 128, 64
k = torch.randn(B, H_k, T, K, device="npu", dtype=torch.float16)
g = -torch.rand(B, H_v, T, device="npu", dtype=torch.float32).cumsum(-1)
beta = torch.rand(B, H_v, T, device="npu", dtype=torch.float32)
A = torch.ops.npu.npu_chunk_scaled_dot_kkt(k, g, beta, chunk_size=C)
torch.npu.synchronize()
assert A.shape == (B, H_k, T, C) and A.dtype == torch.float32
```

## 7. 平台支持

| 平台 | SOC | 状态 |
| --- | --- | --- |
| A2 | `ascend910b` | 支持 |
| A3 | `ascend910_93` | 支持 |
| A5 | `ascend950` | 支持；存在 arch35 时使用特化 |

## 8. 已知限制

- chunk_size 仅支持 16/32/64/128。
- 必须满足 H_v % H_k == 0；A 的 head 维为 H_k。
- cu_seqlens/chunk_indices 必须同时提供或同时省略；varlen 物理 B 必须为 1。
- varlen 累计长度必须覆盖 [0,T]，chunk_indices 必须按 sequence-major 完整列出每个 C 大小的 chunk。
- 指数差固定 clip 到 [-50,50]；H_v>H_k 时当前实现读取 g/beta 的前 H_k 个 head。

## 9. 异常与返回码

| 条件 | 返回码/异常 |
| --- | --- |
| k/g/beta/A、workspaceSize 或 executor 为空 | ACLNN_ERR_PARAM_NULLPTR / ACLNN_ERR_PARAM_INVALID |
| k 非 FP16/BF16，g/beta 非 FP32，或 B/T/H_v shape 不匹配 | ACLNN_ERR_PARAM_INVALID / Python RuntimeError |
| H_v 不能被 H_k 整除，或 chunk_size 不在 16/32/64/128 | ACLNN_ERR_PARAM_INVALID / Python RuntimeError |
| varlen 索引未成对、累计长度/索引顺序非法或物理 B!=1 | ACLNN_ERR_PARAM_INVALID / Python RuntimeError |

负向 case 的 `expect.return_code` 与消息片段集中定义在 `tests/op_cases/chunk_scaled_dot_kkt.json`，修改拦截时必须同步更新。

## 10. 文档自检

- [x] aclnn、Python 与 `<<<>>>` 均提供签名和调用示例。
- [x] Shape 使用模型符号，固定值仅列在已知限制。
- [x] A2/A3/A5、fixed/varlen、GVA 输入、整块/尾块 与错误码均有说明。
- [x] 主入口为 `fla_npu.ops.ascendc`，未把 Triton 声明为并列正式入口。
