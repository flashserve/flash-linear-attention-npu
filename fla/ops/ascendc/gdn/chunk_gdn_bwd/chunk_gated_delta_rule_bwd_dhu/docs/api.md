# ChunkGatedDeltaRuleBwdDhu API 与调用示例

## 1. API 总览

| 通路 | API/入口 | 支持情况 |
| --- | --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.chunk_gated_delta_rule_bwd_dhu` | 支持 |
| aclnn | `aclnnChunkGatedDeltaRuleBwdDhuGetWorkspaceSize` / `aclnnChunkGatedDeltaRuleBwdDhu` | 支持 |
| Ascend C `<<<>>>` | `chunk_gated_delta_rule_bwd_dhu<<<blockDim, nullptr, stream>>>` | 支持 |
| legacy | `npu_chunk_gated_delta_rule_bwd_dhu` | 支持（显式加载） |

各已实现入口使用 README 中定义的同一公式、shape、dtype、边界和可选参数语义。

## 2. 公共参数与约束

Shape 符号见[算子 README 附录](../README.md#shape-symbols)。

### 2.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Layout | 说明 |
| --- | --- | --- | --- | --- | --- |
| `q` | 必选 | `[B,H_k,T,K]` | FP16/BF16 | BNSD | Query |
| `k` | 必选 | `[B,H_k,T,K]` | FP16/BF16 | BNSD | Key |
| `w` | 必选 | `[B,H_v,T,K]` | FP16/BF16 | BNSD | WY 的 W |
| `d_o` | 必选 | `[B,H_v,T,V]` | FP16/BF16 | BNSD | 输出梯度 |
| `dv` | 必选 | `[B,H_v,T,V]` | FP16/BF16 | BNSD | 已有 Value 梯度 |
| `g` | 必选 | `[B,H_v,T]` | FP16/BF16/FP32 | BNS | 标量 gate |
| `gK` | 预留 | `[B,H_v,T,K]` | FP16/BF16 | BNSD | 当前必须为 None/nullptr |
| `h0` | 预留 | `[B,H_v,K,V]` | FP16/BF16 | ND | 当前必须为 None/nullptr |
| `dht` | 预留 | `[B,H_v,K,V]` | FP16/BF16 | ND | 当前必须为 None/nullptr |
| `cu_seqlens` | 可选 | `[N+1]` | INT64 | ND | varlen 累计长度 |
| `chunk_indices` | 可选 | `[2*N_c]` | INT64 | ND | 展平 chunk 索引 |

### 2.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `dh` | `[B,H_v,N_c,K,V]` | 与 q 一致 | 每个 chunk 起点的状态梯度 |
| `dh0` | `-` | - | 预留输出；Python 固定返回 None，aclnn/直调必须传 nullptr |
| `dv2` | `[B,H_v,T,V]` | 与 dv 一致 | 合并状态贡献后的 Value 梯度 |

### 2.3 属性

| 名称 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `scale` | double | `1.0` | Query 分支缩放 |
| `chunk_size` | int | `64` | chunk 长度 |
| `use_exp2` | bool | `false` | 当前仅 false |
| `transpose_state_layout` | bool | `false` | 当前仅 false |

## 3. aclnn API

### 3.1 接口签名

```cpp
aclnnStatus aclnnChunkGatedDeltaRuleBwdDhuGetWorkspaceSize(
const aclTensor *q,
const aclTensor *k,
const aclTensor *w,
const aclTensor *dO,
const aclTensor *dv,
const aclTensor *gOptional,
const aclTensor *gkOptional,
const aclTensor *h0Optional,
const aclTensor *dhtOptional,
const aclIntArray *cuSeqlensOptional,
const aclIntArray *chunkIndicesOptional,
double scale,
int64_t chunkSize,
const aclTensor *dhOut,
const aclTensor *dh0Out,
const aclTensor *dv2Out,
uint64_t *workspaceSize,
aclOpExecutor **executor);

aclnnStatus aclnnChunkGatedDeltaRuleBwdDhu(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
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
aclnnStatus status = aclnnChunkGatedDeltaRuleBwdDhuGetWorkspaceSize(
    q, k, w, dO, dv, g, nullptr, nullptr, nullptr, cuSeqlens, chunkIndices, scale, chunkSize, dh, nullptr, dv2, &workspaceSize, &executor);
ACLNN_CHECK(status);
void *workspace = nullptr;
if (workspaceSize != 0) {
    ACL_CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
}
ACLNN_CHECK(aclnnChunkGatedDeltaRuleBwdDhu(workspace, workspaceSize, executor, stream));
ACL_CHECK(aclrtSynchronizeStream(stream));
if (workspace != nullptr) {
    ACL_CHECK(aclrtFree(workspace));
}
// 依次销毁本例创建的 aclTensor、aclIntArray 等 descriptor。
ACL_CHECK(aclrtDestroyStream(stream));
ACL_CHECK(aclrtResetDevice(deviceId));
ACL_CHECK(aclFinalize());
```

`tests/operators/chunk_gated_delta_rule_bwd_dhu/routes/test_aclnn_chunk_gated_delta_rule_bwd_dhu.cpp` 固化两段式函数签名；tensor descriptor 必须严格按本页公共
参数表构造，不能用物理补齐 shape 替代逻辑 shape。

## 4. `fla_npu.ops.ascendc` API

### 4.1 接口签名

```python
chunk_gated_delta_rule_bwd_dhu(q, k, w, d_o, dv, scale, chunk_size, *, g=None, gK=None, h0=None, dht=None, cu_seqlens=None, chunk_indices=None, use_exp2=False, transpose_state_layout=False)
```

稳定入口加载 OPP op_api 动态库、创建 aclTensor/aclIntArray、取得当前 NPU stream 并保活异步资源；普通 import 不注册 `torch.ops.npu`。

### 4.2 调用示例

```python
import torch
from fla_npu.ops.ascendc import chunk_gated_delta_rule_bwd_dhu

B, H_k, H_v, T, K, V, C = 1, 2, 4, 128, 128, 128, 64
q = torch.randn(B, H_k, T, K, device="npu", dtype=torch.float16)
k = torch.randn_like(q)
w = torch.randn(B, H_v, T, K, device="npu", dtype=torch.float16)
d_o = torch.randn(B, H_v, T, V, device="npu", dtype=torch.float16)
dv = torch.randn_like(d_o)
g = -torch.rand(B, H_v, T, device="npu", dtype=torch.float32).cumsum(-1)
dh, dh0, dv2 = chunk_gated_delta_rule_bwd_dhu(q, k, w, d_o, dv, K ** -0.5, C, g=g)
torch.npu.synchronize()
assert dv2.shape == dv.shape
```

## 5. Ascend C `<<<>>>` 直调

`blockDim`、workspace 和序列化 tiling data 必须来自同一组 host tiling 结果，不能手写猜测。参数顺序与 kernel 定义保持一致：

```cpp
chunk_gated_delta_rule_bwd_dhu<<<blockDim, nullptr, stream>>>(q, k, w, d_o, dv, g, gk, h0, dht, cu_seqlens, chunk_indices, dh, dh0, dv2, workspace, tiling);
ACL_CHECK(aclrtSynchronizeStream(stream));
```

可编译的参数声明和 launch 包装位于 `tests/operators/chunk_gated_delta_rule_bwd_dhu/routes/test_direct_chunk_gated_delta_rule_bwd_dhu.cpp`。

## 6. `torch.ops.npu` API（可选）

```python
import torch
import fla_npu

fla_npu.load_legacy_torch_ops()

B, H_k, H_v, T, K, V, C = 1, 2, 4, 128, 128, 128, 64
q = torch.randn(B, H_k, T, K, device="npu", dtype=torch.float16)
k = torch.randn_like(q)
w = torch.randn(B, H_v, T, K, device="npu", dtype=torch.float16)
d_o = torch.randn(B, H_v, T, V, device="npu", dtype=torch.float16)
dv = torch.randn_like(d_o)
g = -torch.rand(B, H_v, T, device="npu", dtype=torch.float32).cumsum(-1)
dh, dh0, dv2 = torch.ops.npu.npu_chunk_gated_delta_rule_bwd_dhu(q, k, w, d_o, dv, K ** -0.5, C, g=g)
torch.npu.synchronize()
assert dv2.shape == dv.shape
```

## 7. 平台支持

| 平台 | SOC | 状态 |
| --- | --- | --- |
| A2 | `ascend910b` | 支持 |
| A3 | `ascend910_93` | 支持 |
| A5 | `ascend950` | 支持；存在 arch35 时使用特化 |

## 8. 已知限制

- `K <= 128`、`V <= 256`；交付矩阵覆盖 K=128、V=128/256。
- `chunk_size` 仅支持 64/128；`H_v % H_k == 0`。
- `g` 必须提供；`gK`、`h0`、`dht` 和 `dh0` 当前为预留，必须为空。
- varlen 当前仅支持物理 `B=1`，两个索引必须同时提供。
- `use_exp2` 与 `transpose_state_layout` 当前必须为 false。

## 9. 异常与返回码

| 条件 | 返回码/异常 |
| --- | --- |
| workspaceSize 或 executor 为空 | ACLNN_ERR_PARAM_NULLPTR |
| 必选 tensor 为空，或 g 未提供 | ACLNN_ERR_PARAM_INVALID |
| gK/h0/dht/dh0 非空 | ACLNN_ERR_PARAM_INVALID |
| chunk_size 不是 64/128，或 varlen 索引只提供一个 | ACLNN_ERR_PARAM_INVALID |
| Python use_exp2/transpose_state_layout 为 true | RuntimeError |
| 执行器或 kernel launch 失败 | ACLNN_ERR_INNER |

负向 case 的 `expect.return_code` 与消息片段集中定义在 `tests/op_cases/chunk_gated_delta_rule_bwd_dhu.json`，修改拦截时必须同步更新。

## 10. 文档自检

- [x] aclnn、Python 与 `<<<>>>` 均提供签名和调用示例。
- [x] Shape 使用模型符号，固定值仅列在已知限制。
- [x] A2/A3/A5、fixed/varlen、标量 g、FP16/BF16 主张量、FP16/BF16/FP32 gate 与错误码均有说明。
- [x] 主入口为 `fla_npu.ops.ascendc`，未把 Triton 声明为并列正式入口。
