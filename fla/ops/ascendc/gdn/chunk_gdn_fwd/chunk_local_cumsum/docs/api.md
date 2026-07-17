# ChunkLocalCumsum API 与调用示例

## 1. API 总览

| 通路 | API/入口 | 支持情况 |
| --- | --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.chunk_local_cumsum` | 支持 |
| aclnn | `aclnnChunkLocalCumsumGetWorkspaceSize` / `aclnnChunkLocalCumsum` | 支持 |
| Ascend C `<<<>>>` | `chunk_local_cumsum<<<blockDim, nullptr, stream>>>` | 支持 |
| legacy | `npu_chunk_local_cumsum` | 支持（显式加载） |

各已实现入口使用 README 中定义的同一公式、shape、dtype、边界和可选参数语义。

## 2. 公共参数与约束

Shape 符号见[算子 README 附录](../README.md#shape-symbols)。

### 2.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Layout | 说明 |
| --- | --- | --- | --- | --- | --- |
| `g` | 必选 | `[B,H_v,T] 或 [B,H_v,T,...]` | FP32 | head-first | 待累加 gate/特征 |
| `cu_seqlens` | 可选 | `[N+1]` | INT64 | ND | varlen 累计长度；aclnn 使用 host `aclIntArray`，Python 使用整数序列 |
| `chunk_indices_out` | 可选 | `[N_b,2] 或 [2*N_b]` | INT64 | ND | varlen 块映射；aclnn 使用展平的 host `aclIntArray` |

### 2.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `out` | `与 g 相同` | FP32 | chunk-local 累加结果 |

### 2.3 属性

| 名称 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `chunk_size` | int | `无` | 2 的幂 |
| `reverse` | bool | `false` | 前缀/后缀 |
| `scale` | double | `1.0` | 输出缩放 |
| `head_first` | bool | `true` | 当前必须 true |
| `output_dtype` | str | `float32` | 仅 float32/torch.float/torch.float32 |

## 3. aclnn API

### 3.1 接口签名

```cpp
aclnnStatus aclnnChunkLocalCumsumGetWorkspaceSize(
const aclTensor *g,
const aclIntArray *cuSeqlensOptional,
const aclIntArray *chunkIndicesOutOptional,
int64_t chunkSize,
bool reverse,
double scale,
bool headFirst,
char *outputDtypeOptional,
const aclTensor *out,
uint64_t *workspaceSize,
aclOpExecutor **executor);

aclnnStatus aclnnChunkLocalCumsum(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
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
// varlen 时在 host 创建 cuSeqlens/chunkIndicesOut 两个 aclIntArray；后者按 [seq_id, block_id, ...] 展平。
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
aclnnStatus status = aclnnChunkLocalCumsumGetWorkspaceSize(
    g, nullptr, nullptr, chunkSize, false, scale, true, const_cast<char *>("float32"), out, &workspaceSize, &executor);
ACLNN_CHECK(status);
void *workspace = nullptr;
if (workspaceSize != 0) {
    ACL_CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
}
ACLNN_CHECK(aclnnChunkLocalCumsum(workspace, workspaceSize, executor, stream));
ACL_CHECK(aclrtSynchronizeStream(stream));
if (workspace != nullptr) {
    ACL_CHECK(aclrtFree(workspace));
}
// 依次销毁本例创建的 aclTensor、aclIntArray 等 descriptor。
ACL_CHECK(aclrtDestroyStream(stream));
ACL_CHECK(aclrtResetDevice(deviceId));
ACL_CHECK(aclFinalize());
```

`tests/operators/chunk_local_cumsum/routes/test_aclnn_chunk_local_cumsum.cpp` 固化两段式函数签名；tensor descriptor 必须严格按本页公共
参数表构造，不能用物理补齐 shape 替代逻辑 shape。

## 4. `fla_npu.ops.ascendc` API

### 4.1 接口签名

```python
chunk_local_cumsum(g, chunk_size, *, cu_seqlens=None, chunk_indices_out=None, reverse=False, scale=1.0, head_first=True, output_dtype='float32')
```

稳定入口加载 OPP op_api 动态库、创建 aclTensor/aclIntArray、取得当前 NPU stream 并保活异步资源；普通 import 不注册 `torch.ops.npu`。

### 4.2 调用示例

```python
import torch
from fla_npu.ops.ascendc import chunk_local_cumsum

B, H_v, T, C = 1, 4, 129, 64
g = torch.randn(B, H_v, T, device="npu", dtype=torch.float32)
cu_seqlens = [0, 65, 129]
# 本 shape 下 B_T=2048，每条逻辑序列各有一个处理块。
chunk_indices_out = [[0, 0], [1, 0]]
out = chunk_local_cumsum(
    g,
    C,
    cu_seqlens=cu_seqlens,
    chunk_indices_out=chunk_indices_out,
    reverse=False,
    scale=1.0,
    head_first=True,
)
torch.npu.synchronize()
assert out.shape == g.shape and out.dtype == torch.float32
```

## 5. Ascend C `<<<>>>` 直调

`blockDim`、workspace 和序列化 tiling data 必须来自同一组 host tiling 结果，不能手写猜测。参数顺序与 kernel 定义保持一致：

```cpp
chunk_local_cumsum<<<blockDim, nullptr, stream>>>(g, cuSeqlens, chunkIndices, out, workspace, tiling);
ACL_CHECK(aclrtSynchronizeStream(stream));
```

可编译的参数声明和 launch 包装位于 `tests/operators/chunk_local_cumsum/routes/test_direct_chunk_local_cumsum.cpp`。

## 6. `torch.ops.npu` API（可选）

```python
import torch
import fla_npu

fla_npu.load_legacy_torch_ops()

B, H_v, T, C = 1, 4, 129, 64
g = torch.randn(B, H_v, T, device="npu", dtype=torch.float32)
out = torch.ops.npu.npu_chunk_local_cumsum(g, C, reverse=False, scale=1.0, head_first=True)
torch.npu.synchronize()
assert out.shape == g.shape and out.dtype == torch.float32
```

## 7. 平台支持

| 平台 | SOC | 状态 |
| --- | --- | --- |
| A2 | `ascend910b` | 支持 |
| A3 | `ascend910_93` | 支持 |
| A5 | `ascend950` | 支持；存在 arch35 时使用特化 |

## 8. 已知限制

- g rank 至少 3，所有维度为正；head_first 当前必须 true。
- chunk_size 必须为 2 的幂，且满足 `B_T >= C`；交付矩阵至少覆盖 `P=1,C=64` 的 varlen 尾块和
  `P=16,C=64` 的 dense 尾块。
- output_dtype 仅支持 FP32 别名。
- varlen 物理 B=1，两个 host 整数数组必须同时提供；cu_seqlens 首项为 0、末项为 T 且非递减。
- chunk_indices_out 必须按 sequence-major 完整列出内部处理块；处理块长度 B_T 由 UB、C 和 P 共同决定，不能直接按 C 构造。

## 9. 异常与返回码

| 条件 | 返回码/异常 |
| --- | --- |
| g/out、workspaceSize 或 executor 为空 | ACLNN_ERR_PARAM_NULLPTR / ACLNN_ERR_PARAM_INVALID |
| g 非 FP32、rank<3、存在非正维或 head_first=false | ACLNN_ERR_PARAM_INVALID / Python RuntimeError |
| chunk_size 非正 2 的幂，或 output_dtype 非 FP32 别名 | ACLNN_ERR_PARAM_INVALID / Python RuntimeError |
| 两个 varlen 索引未成对、非 INT64、shape/值/顺序非法或物理 B!=1 | ACLNN_ERR_PARAM_INVALID / Python RuntimeError |

负向 case 的 `expect.return_code` 与消息片段集中定义在 `tests/op_cases/chunk_local_cumsum.json`，修改拦截时必须同步更新。

## 10. 文档自检

- [x] aclnn、Python 与 `<<<>>>` 均提供签名和调用示例。
- [x] Shape 使用模型符号，固定值仅列在已知限制。
- [x] A2/A3/A5、fixed/varlen、forward/reverse、任意连续尾部 P 与错误码均有说明。
- [x] 主入口为 `fla_npu.ops.ascendc`，未把 Triton 声明为并列正式入口。
