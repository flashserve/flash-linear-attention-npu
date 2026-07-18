# ChunkKdaFwd API 与调用示例

## 1. API 总览

| 通路 | API/入口 | 支持情况 |
| --- | --- | --- |
| Python 主入口 | `fla_npu.ops.ascendc.chunk_kda_fwd` | 支持 |
| aclnn | `aclnnChunkKdaFwdGetWorkspaceSize` / `aclnnChunkKdaFwd` | 支持 |
| Ascend C `<<<>>>` | `chunk_kda_fwd<<<blockDim, nullptr, stream>>>` | 历史诊断通路，待整改 |
| legacy | `npu_chunk_kda_fwd` | 支持（显式加载） |

表中标记为“支持”的正式入口使用 README 中定义的同一公式、shape、dtype、边界和可选参数语义；待整改诊断通路不据此宣称已经满足完整公开语义。

公共 API 应只表达完整算子语义，不要求调用者传入或理解内部 stage 编号；类型转换应由 kernel 完成，不由 L2 Cast 组成调用前置步骤。现有实现不满足时，必须像本页一样明确标为架构债务和待整改通路。

## 2. 公共参数与约束

Shape 符号见[算子 README 附录](../README.md#shape-symbols)。

### 2.1 输入

| 名称 | 必选/可选 | Shape | Dtype | Layout | 说明 |
| --- | --- | --- | --- | --- | --- |
| `q` | 必选 | `按 layout 为 [B,T,H_k,K]/[B,H_k,T,K]/[T,H_k,K]/[H_k,T,K]` | FP16/BF16 | BSND/BNSD/TND/NTD | Query |
| `k` | 必选 | `与 q 相同` | 与 q 相同 | 与 q 相同 | Key |
| `v` | 必选 | `对应 [B,T,H_v,V]/[B,H_v,T,V]/[T,H_v,V]/[H_v,T,V]` | 与 q 相同 | 同 layout | Value |
| `gk` | 必选 | `与 k 的 token/head/K 维对应` | FP32/BF16 | 同 layout | chunk 内 log2 累积 key gate |
| `beta` | 必选 | `去掉 K 维的 gk shape` | FP32/BF16 | 同 layout | Delta 更新系数 |
| `initial_state` | 可选 | `[N,H_v,K,V]` | FP32 | ND | 每条逻辑序列的初始状态 |
| `cu_seqlens` | 可选 | `[N+1]` | INT64 | ND | 变长序列累计长度 |
| `chunk_indices` | 可选 | `[2*N_c]` | INT64 | ND | sequence-major chunk 二元组 |

### 2.2 输出

| 名称 | Shape | Dtype | 说明 |
| --- | --- | --- | --- |
| `o` | `与 v 相同` | 与 v 相同 | KDA 输出 |
| `final_state` | `[N,H_v,K,V] 或空` | FP32 | output_final_state=false 时 Python 返回空 tensor |
| `g` | `与 gk 相同` | FP32 | Python 返回槽：gk 转 FP32 |
| `Aqk/Akk` | `按 layout 为 [...,T,chunk_size]` | 与 q 相同 | chunk 内因果矩阵；内部计算可使用 FP32 |
| `w/qg/kg` | `按 layout 为 [...,T,K]` | 与 q 相同 | K 维中间量 |
| `u/v_new` | `与 v 相同` | 与 v 相同 | V 维中间量 |
| `h` | `按 layout 为 [B,H_v,N_c,K,V] 或 [B,N_c,H_v,K,V]` | 与 q 相同 | 每个 chunk 的起始状态 |
| `initial_state_out` | `与 initial_state 相同或空` | FP32 | Python 预留透传槽 |

### 2.3 属性

| 名称 | 类型 | 默认值 | 取值范围 | 说明 |
| --- | --- | --- | --- | --- |
| `layout` | str | `BSND` | `{"BSND", "BNSD", "TND", "NTD"}` | 只接受大写 BSND/BNSD/TND/NTD |
| `scale` | double | `无` | - | 通常为 1/sqrt(K) |
| `chunk_size` | int | `无` | `{64, 128}` | 64 或 128 |
| `output_final_state` | bool | `false` | `{false, true}` | 是否返回有效 final_state |
| `return_intermediate` | bool | `false` | `{false, true}` | 是否物化八个中间张量 |
| `safe_gate` | bool | `false` | `{false}` | 预留，当前必须 false |
| `transpose_state_layout` | bool | `false` | `{false}` | 预留，当前必须 false |

## 3. aclnn API

### 3.1 接口签名

```cpp
aclnnStatus aclnnChunkKdaFwdGetWorkspaceSize(
const aclTensor *q,
const aclTensor *k,
const aclTensor *v,
const aclTensor *gk,
const aclTensor *beta,
const aclTensor *initialStateOptional,
const aclIntArray *cuSeqlensOptional,
const aclIntArray *chunkIndicesOptional,
const char *layout,
double scale,
int64_t chunkSize,
bool outputFinalState,
int64_t totalChunks,
const aclTensor *oOut,
const aclTensor *finalStateOut,
const aclTensor *aqkOut,
const aclTensor *akkOut,
const aclTensor *wOut,
const aclTensor *uOut,
const aclTensor *qgOut,
const aclTensor *kgOut,
const aclTensor *vNewOut,
const aclTensor *hOut,
uint64_t *workspaceSize,
aclOpExecutor **executor);

aclnnStatus aclnnChunkKdaFwd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
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
aclnnStatus status = aclnnChunkKdaFwdGetWorkspaceSize(
    q, k, v, gk, beta, initialState, cuSeqlens, chunkIndices, layout, scale, chunkSize, outputFinalState, totalChunks, o, finalState, aqk, akk, w, u, qg, kg, vNew, h, &workspaceSize, &executor);
ACLNN_CHECK(status);
void *workspace = nullptr;
if (workspaceSize != 0) {
    ACL_CHECK(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
}
ACLNN_CHECK(aclnnChunkKdaFwd(workspace, workspaceSize, executor, stream));
ACL_CHECK(aclrtSynchronizeStream(stream));
if (workspace != nullptr) {
    ACL_CHECK(aclrtFree(workspace));
}
// 依次销毁本例创建的 aclTensor、aclIntArray 等 descriptor。
ACL_CHECK(aclrtDestroyStream(stream));
ACL_CHECK(aclrtResetDevice(deviceId));
ACL_CHECK(aclFinalize());
```

`tests/operators/chunk_kda_fwd/routes/test_aclnn_chunk_kda_fwd.cpp` 固化两段式函数签名；tensor descriptor 必须严格按本页公共
参数表构造，不能用物理补齐 shape 替代逻辑 shape。

## 4. `fla_npu.ops.ascendc` API

### 4.1 接口签名

```python
chunk_kda_fwd(q, k, v, gk, beta, scale, chunk_size, *, layout='BSND', initial_state=None, output_final_state=False, cu_seqlens=None, chunk_indices=None, return_intermediate=False, safe_gate=False, transpose_state_layout=False)
```

稳定入口加载 OPP op_api 动态库、创建 aclTensor/aclIntArray、取得当前 NPU stream 并保活异步资源；普通 import 不注册 `torch.ops.npu`。

### 4.2 调用示例

```python
import torch
from fla_npu.ops.ascendc import chunk_kda_fwd

B, H_k, H_v, T, K, V, chunk_size = 1, 2, 4, 128, 128, 128, 64
q = torch.randn(B, H_k, T, K, device="npu", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn(B, H_v, T, V, device="npu", dtype=torch.bfloat16)
gk = -torch.rand(B, H_v, T, K, device="npu", dtype=torch.float32).cumsum(2)
beta = torch.sigmoid(torch.randn(B, H_v, T, device="npu", dtype=torch.float32))
outputs = chunk_kda_fwd(q, k, v, gk, beta, K ** -0.5, chunk_size,
                layout="BNSD", output_final_state=True)
o, final_state = outputs[:2]
torch.npu.synchronize()
assert o.shape == v.shape and final_state.dtype == torch.float32
```

## 5. Ascend C `<<<>>>` 直调


> **架构债务：** 下述 stage launch 仅记录当前历史实现和诊断通路，不满足“`<<<>>>` 对外提供完整算子语义、
> 不暴露内部 stage、L2 不拼接 Cast”的开发规范。新增算子不得照搬，KDA 后续必须按设计文档整改。

`chunk_kda_fwd` 是复合算子，完整直调通路必须使用 host 为同一 case 生成的四组 launch 配置，
并按同一 stream 串行执行以下流水；单独发射一次 `chunk_kda_fwd` 不构成公开算子语义：

```cpp
// 输入已按公共 layout 契约规范化为内部 BNSD/NTD，并完成 contiguous/gate cast。
chunk_kda_fwd<<<stage1.blockDim, nullptr, stream>>>(/* stage=1 参数、workspace、tiling */);
ScaleAndCastAqkAndQg(aqkFp32, qg, scale, aqkScaled, qgScaled, stream);
chunk_kda_fwd<<<stage3.blockDim, nullptr, stream>>>(/* stage=3 参数、workspace、tiling */);
chunk_gated_delta_rule_fwd_h<<<state.blockDim, nullptr, stream>>>(
    kg, w, u, neutralG, gk, initialState, cuSeqlens, chunkIndices,
    h, vNew, finalState, state.workspace, state.tiling);
chunk_kda_fwd<<<stage2.blockDim, nullptr, stream>>>(/* stage=2 参数、workspace、tiling */);
ACL_CHECK(aclrtSynchronizeStream(stream));
```

stage1 生成矩阵与预处理量，缩放/转换步骤与 aclnn L2 完全相同，stage3 生成 `w/u/kg`，
GDN 状态 kernel 递推 `h/v_new/final_state`，stage2 合成 `o`。四组 `blockDim/workspace/tiling`
不能混用，`stage` 分别固定为 1、3、GDN 状态阶段和 2。可编译的两个 kernel 原型及单 stage
launch 包装见 `tests/operators/chunk_kda_fwd/routes/test_direct_chunk_kda_fwd.cpp`；直调执行器还必须
负责公开 layout 的前后转换和中间张量生命周期。


## 6. `torch.ops.npu` API（可选）

```python
import torch
import fla_npu

fla_npu.load_legacy_torch_ops()

B, H_k, H_v, T, K, V, chunk_size = 1, 2, 4, 128, 128, 128, 64
q = torch.randn(B, H_k, T, K, device="npu", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn(B, H_v, T, V, device="npu", dtype=torch.bfloat16)
gk = -torch.rand(B, H_v, T, K, device="npu", dtype=torch.float32).cumsum(2)
beta = torch.sigmoid(torch.randn(B, H_v, T, device="npu", dtype=torch.float32))
outputs = torch.ops.npu.npu_chunk_kda_fwd(q, k, v, gk, beta, K ** -0.5, chunk_size,
                layout="BNSD", output_final_state=True)
o, final_state = outputs[:2]
torch.npu.synchronize()
assert o.shape == v.shape and final_state.dtype == torch.float32
```

## 7. 平台支持

| 平台 | SOC | 状态 |
| --- | --- | --- |
| A2 | `ascend910b` | 支持 |
| A3 | `ascend910_93` | 支持 |
| A5 | `ascend950` | 支持；存在 arch35 时使用特化 |

## 8. 已知限制

- chunk_size 仅支持 64/128；K/V 均须在 [16,256] 且为 16 的倍数；交付矩阵覆盖 K=128、V=128/256。
- H_k/H_v 必须在 [1,128] 且 H_v % H_k == 0；TND 仅支持 H_k=1，多 head rank3 使用 NTD。
- 变长序列的 cu_seqlens 至少含首尾、非递减且末项等于 T；单次最多 1024 条逻辑序列。
- 显式 chunk_indices 必须完整、合法并严格采用 sequence-major 规范顺序。
- safe_gate 与 transpose_state_layout 当前必须为 false；raw gate 应先调用 kda_gate_cumsum。

## 9. 异常与返回码

| 条件 | 返回码/异常 |
| --- | --- |
| workspaceSize 或 executor 为空 | ACLNN_ERR_PARAM_NULLPTR |
| q/k/v/gk/beta 或任一必选输出为空 | ACLNN_ERR_PARAM_INVALID（CheckParams 外层映射） |
| layout、rank、shape、dtype、GVA、K/V、chunk_size 或 scale 不合法 | ACLNN_ERR_PARAM_INVALID |
| 变长序列累计长度、chunk 顺序、物理 B 或状态 shape 不合法 | ACLNN_ERR_PARAM_INVALID |
| return_intermediate 与中间输出的全有/全无契约不一致 | ACLNN_ERR_PARAM_INVALID |
| 执行器创建、内部布局转换或 kernel 执行失败 | ACLNN_ERR_INNER/内部错误码 |

负向 case 的 `expect.return_code` 与消息片段集中定义在 `tests/op_cases/chunk_kda_fwd.json`，修改拦截时必须同步更新。

## 10. 文档自检

- [x] aclnn、Python 与 `<<<>>>` 均提供签名和调用示例。
- [x] Shape 使用模型符号，固定值仅列在已知限制。
- [x] A2/A3/A5、定长/变长序列、四种显式 layout、可选初始/最终状态、可选中间量 与错误码均有说明。
- [x] 主入口为 `fla_npu.ops.ascendc`，未把 Triton 声明为并列正式入口。
