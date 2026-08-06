# ChunkGatedDeltaRuleFwdH

## 功能

共享的 chunk 间状态递推算子，供 GDN 和 KDA 使用。输入固定为 head-major BNSD：

```text
k/w: [B,H,T,K]
u:   [B,H_v,T,V]
g:   [B,H_v,T]       optional
gk:  [B,H_v,T,K]     optional
```

`g` 和 `gk` 至少提供一个；两者同时提供时分别作用于 scalar gate 和 key-wise gate。

## 计算

对每个 chunk：

```text
v_new = u - w @ h_prev

scalar_decay = exp(g_last)       if g is provided
key_decay    = exp2(gk_last)     if gk is provided

h_next = h_prev * scalar_decay * key_decay + kg^T @ v_new
```

逐 K gate `gk` 固定处于 log2 空间，因此固定使用 `exp2`。标量 gate `g` 固定处于自然对数空间，
因此固定使用 `exp`；公共接口不再暴露 `use_exp2`。`v_new` 始终保存，不需要独立 `save_new_value` 属性。

## Python API

```python
from fla_npu.ops.ascendc import chunk_gated_delta_rule_fwd_h

h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
    k,
    w,
    u,
    g=None,
    gk=None,
    initial_state=None,
    output_final_state=False,
    chunk_size=64,
    cu_seqlens=None,
    chunk_indices=None,
    state_v_first=False,
)
```

## 输出

| 输出 | 必选性 | Shape |
| --- | --- | --- |
| `h` | 必选 | `[B,H_v,N_c,K,V]`，`state_v_first=true` 时末两维为 `[V,K]` |
| `v_new` | 必选 | `[B,H_v,T,V]` |
| `final_state` | 可选 | `[N,H_v,K,V]`，`state_v_first=true` 时末两维为 `[V,K]` |

`h` 是每个 chunk 的起始状态，反向会继续使用，因此保持 head-major。`final_state` 仅供调用者输出，
在 `output_final_state=false` 时不创建公开输出。

## aclnn

```cpp
aclnnStatus aclnnChunkGatedDeltaRuleFwdHGetWorkspaceSize(
    const aclTensor *k,
    const aclTensor *w,
    const aclTensor *u,
    const aclTensor *gOptional,
    const aclTensor *gkOptional,
    const aclTensor *initialStateOptional,
    bool outputFinalState,
    int64_t chunkSize,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    bool stateVFirst,
    const aclTensor *hOut,
    const aclTensor *vNewOut,
    const aclTensor *finalStateOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);
```

## 支持范围

- A2/A3/A5。
- FP16/BF16 数据路径，gate 支持 FP16/BF16/FP32，状态支持 FP16/BF16/FP32。
- V=128/256，chunk_size=64/128。
- dense 与 `cu_seqlens/chunk_indices` 变长模式。
- `state_v_first` true/false。
