# ChunkFwdH API

## Python

```python
from fla_npu.ops.ascendc import chunk_fwd_h

h, v_new, final_state = chunk_fwd_h(
    k,
    w,
    u,
    *,
    g=None,
    gk=None,
    initial_state=None,
    output_final_state=False,
    chunk_size=64,
    save_new_value=True,
    cu_seqlens=None,
    chunk_indices=None,
    use_exp2=False,
    state_v_first=False,
)
```

这是 ctypes 直调 `aclnnChunkFwdH` 的稳定入口，不注册 legacy `torch.ops.npu` 接口。

## aclnn

```cpp
aclnnStatus aclnnChunkFwdHGetWorkspaceSize(
    const aclTensor *k,
    const aclTensor *w,
    const aclTensor *u,
    const aclTensor *gOptional,
    const aclTensor *gkOptional,
    const aclTensor *initialStateOptional,
    bool outputFinalState,
    int64_t chunkSize,
    bool saveNewValue,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    bool useExp2,
    bool stateVFirst,
    const aclTensor *hOut,
    const aclTensor *vNewOut,
    const aclTensor *finalStateOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnChunkFwdH(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## 参数

| 参数 | dtype | 约束 |
| --- | --- | --- |
| `k` | BF16 | g-only `[B,HK,T,128]` raw K；gk-only `[B,HV,T,128]` prepared kg |
| `w` | BF16 | `[B,HV,T,128]` |
| `u` | BF16 | `[B,HV,T,128]` |
| `g` | BF16/FP32 | g-only `[B,HV,T]` |
| `gk` | BF16/FP32 | gk-only `[B,HV,T,128]` |
| `initial_state` | BF16/FP32 | 可空；`[N,HV,128,128]`，末两维物理顺序由 `state_v_first` 决定 |
| `cu_seqlens` | host int array | 可空；变长时从 0 开始、以 T 结束且严格递增 |
| `chunk_indices` | host int array | 可空；必须为 canonical sequence-major pair 序列 |

`g` 与 `gk` 必须且只能提供一个。g-only 要求 `HV%HK==0`；gk-only 要求 `k` 的 head 数等于
HV，本算子不会再次把 gk 应用到 prepared kg。

`B/HK/HV/T` 必须为正；固定 `K=V=128`、`chunk_size=64`、`save_new_value=true`。
变长模式要求输入容器 `B=1`。

## 输出

| 输出 | dtype | Shape |
| --- | --- | --- |
| `h` | BF16 | dense `[B,HV,C,128,128]`；varlen `[1,HV,total_chunks,128,128]` |
| `v_new` | BF16 | `[B,HV,T,128]` |
| `final_state` | StateT | `[N,HV,128,128]`，仅 `output_final_state=true` 时存在 |

`state_v_first=false` 时 state/H 的末两维语义为 `[K,V]`；为 true 时物理布局为 `[V,K]`。
存在 initial_state 时 StateT 等于其 dtype；Python 在没有 initial_state 但请求 final_state 时使用
FP32。aclnn 调用者可通过 final_state 输出 dtype 选择 BF16 或 FP32。

`output_final_state` 必须与 `finalStateOut` 的物理存在性一致。参数空指针返回
`ACLNN_ERR_PARAM_NULLPTR`；dtype、shape、layout、属性、gate 组合和变长元数据不合法返回
`ACLNN_ERR_PARAM_INVALID`。
