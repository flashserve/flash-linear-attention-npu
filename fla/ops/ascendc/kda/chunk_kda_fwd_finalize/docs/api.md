# ChunkKdaFwdFinalize API

## Python

```python
from fla_npu.ops.ascendc import chunk_kda_fwd_finalize

attn_out = chunk_kda_fwd_finalize(
    qg_scaled,
    aqk,
    v_new,
    h,
    output_layout="BSND",
    state_v_first=False,
    cu_seqlens=None,
    chunk_indices=None,
)
```

这是 ctypes 直调 `aclnnChunkKdaFwdFinalize` 的稳定入口，不注册
legacy `torch.ops.npu` 接口。输入、输出和变长元数据的形状与约束
见下文及[算子 README](../README.md#输入输出)。

## 参数

| 参数 | 类型 | 必选 | 说明 |
| --- | --- | --- | --- |
| `qg_scaled` | BF16 tensor | 是 | Prepare 输出的已缩放 QG；rank-4 `[B,HV,T,128]` 或 packed rank-3 `[HV,T,128]` |
| `aqk` | BF16 tensor | 是 | Prepare 的 Aqk（已乘 scale）；最后一维固定 64 |
| `v_new` | BF16 tensor | 是 | FwdH 的 `v_new`；dense `[B,HV,T,128]`，packed 接受 `[1,HV,T,128]` 或 `[HV,T,128]` |
| `h` | BF16 tensor | 是 | FwdH 的 NT-first 状态 `[B,C,HV,128,128]`，有变长元数据时为 `[1,C,HV,128,128]` |
| `output_layout` | str | 否 | `BSND`（默认）/`BNSD`/`TND`/`NTD`，区分大小写；rank 与合法取值见下文 |
| `state_v_first` | bool | 否 | `h` 末两维顺序；`false`（默认）为 `[K,V]`，`true` 为 `[V,K]` |
| `cu_seqlens` | list[int] | 否 | 首 0、末 T、严格递增；传入时 rank-4 输入要求 `B=1` |
| `chunk_indices` | list[int] | 否 | 仅在传入 `cu_seqlens` 时可用，按 sequence-major 列出 `(sequence_id, local_chunk_id)` |

返回：单个 BF16 张量 `attn_out`，布局由 `output_layout` 决定，形状见
[算子 README](../README.md#输入输出)。dtype、layout 与变长约束见 `## aclnn`
和[算子 README](../README.md#支持范围)；平台支持范围为 A2/A3/A5/A6。

## aclnn

```cpp
aclnnStatus aclnnChunkKdaFwdFinalizeGetWorkspaceSize(
    const aclTensor *qgScaled,
    const aclTensor *aqk,
    const aclTensor *vNew,
    const aclTensor *h,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const char *outputLayout,
    bool stateVFirst,
    const aclTensor *attnOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnChunkKdaFwdFinalize(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

四个 tensor 输入和 `attnOut` 必传，且 descriptor 必须为非私有
ND 格式；输入允许 L2 连续化，kernel 直接写的输出必须连续。
均为 BF16，`K=V=128`，`Aqk` 的列数为 64；支持 A2/A3/A5/A6
（A6 为 Ascend960DT、`__NPU_ARCH__=9201`，单 AIV 串行搬移）。
`outputLayout` 对 rank-4 `qgScaled/Aqk` 只接受 `BSND/BNSD`，
对 rank-3 `qgScaled/Aqk` 只接受 `TND/NTD`；packed 场景的
`vNew` 可为 FwdH 的 `[1,HV,T,128]` 或独立调用的
`[HV,T,128]`，区分大小写。输入形状和输出形状见
[算子 README](../README.md#输入输出)。

`stateVFirst=false` 时 `h` 表示 `[K,V]`；为 true 时表示 `[V,K]`。
`cuSeqlensOptional` 是严格递增、首 0 末 T 的 host int array。
有变长元数据时 rank-4 输入要求 `B=1`；`chunkIndicesOptional`
若提供必须与 `cuSeqlensOptional` 同时存在，且为规范的
sequence-major `(sequence_id,local_chunk_id)` 列表。

唯一输出 `attnOut` 是 BF16。`BSND/TND` 按 token 优先写出，
`BNSD/NTD` 按 value head 优先写出；三个 token 输入仍然按 value head
优先排列；h 为 NT-first，未传 cu_seqlens 时为 `[B,C,HV,128,128]`，
传入 cu_seqlens 时为 `[1,C,HV,128,128]`，与 token 输入的 rank 分开判断。
旧 head-first h 调用者需要交换 chunk/head 轴并连续化，不能只 reshape。
数值精度边界见[设计文档](design.md#数学与精度)。

## 返回码

| 返回码 | 触发条件 |
| --- | --- |
| `ACLNN_SUCCESS` | workspace 查询和执行成功 |
| `ACLNN_ERR_PARAM_NULLPTR` | 必传输入/输出或 workspaceSize/executor 为空 |
| `ACLNN_ERR_PARAM_INVALID` | dtype、shape、format、layout、连续性或变长元数据不合法 |
| `ACLNN_ERR_INNER_CREATE_EXECUTOR` | executor 创建失败 |
| `ACLNN_ERR_INNER_NULLPTR` | 中间 descriptor 创建失败 |
| `ACLNN_ERR_INNER` | kernel 调用失败 |
