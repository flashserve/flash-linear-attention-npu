# ChunkGatedDeltaRuleFwdH

## 功能

共享的 chunk 间状态递推算子，供 GDN 和 KDA 使用。公开短名为 `chunk_fwd_h`，
物理算子继续复用 `ChunkGatedDeltaRuleFwdH`。输入固定为 head-major BNSD：

```text
k:   [B,H_k,T,K]
w:   [B,H_v,T,K]
u:   [B,H_v,T,V]
g:   [B,H_v,T]       optional
gk:  [B,H_v,T,K]     optional
```

`g` 和 `gk` 至少提供一个；两者同时提供时分别作用于 scalar gate 和 key-wise gate。

## 计算

对每个 chunk：

```text
v_new = u - w @ h_prev

scalar_decay = exp(g_last)       if g is provided and use_exp2 is false
scalar_decay = exp2(g_last)      if g is provided and use_exp2 is true
key_decay    = exp2(gk_last)     if gk is provided

h_next = h_prev * scalar_decay * key_decay + k^T @ v_new
```

逐 K gate `gk` 固定处于 log2 空间。`use_exp2=true` 表示 `exp2(x)` 语义，false 时逐 K gate 使用
数学等价的 `exp(x * ln(2))`；当前 NPU 路径允许用后者实现等价 fallback。GDN natural-log scalar gate
路径固定传 `use_exp2=false`。KDA 路径传 `k=kg`、零值 `g` 和 Prepare 产生的 `gk`，因此 scalar
gate 的单位不影响结果，并由组合接口显式透传 `use_exp2`。`save_new_value` 是兼容属性，首版仅支持
`True`。

## Python API

```python
from fla_npu.ops.ascendc import chunk_fwd_h

h, v_new, final_state = chunk_fwd_h(
    k,
    w,
    u,
    g=None,
    *,
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

`state_v_first=True` 只改变公开 h/state 的最后 K/V 两轴。Python adapter 在进入物理 aclnn 前
把 `initial_state` 规范化为 `[N,H_v,K,V]`，并始终向物理接口传
`stateVFirst=false`；返回时再把 h/final_state 转回 `[V,K]`。

## 输出

| 输出 | 必选性 | Shape |
| --- | --- | --- |
| `h` | 必选 | `[B,H_v,N_c,K,V]`，`state_v_first=true` 时末两维为 `[V,K]` |
| `v_new` | 必选 | `[B,H_v,T,V]` |
| `final_state` | 必选返回槽 | `[N,H_v,K,V]`，`state_v_first=true` 时末两维为 `[V,K]` |

`h` 是每个 chunk 的起始状态，反向会继续使用，因此保持 head-major。`final_state` 仅供调用者输出，
在 `output_final_state=false` 时返回同 device 的 FP32 空 Tensor，shape 为 `[0]`。

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
```

## 支持范围

- A2/A3/A5。
- FP16/BF16 数据路径，gate 支持 FP16/BF16/FP32，初始/最终状态使用 FP32。
- V=128/256；首版 `chunk_size=64`。
- dense 时不传 `cu_seqlens/chunk_indices`；varlen 要求 B=1、`cu_seqlens` 从 0 到 T 严格递增，
  `chunk_indices` 使用 `[NT,2]` 的 canonical sequence-major 顺序。
- `state_v_first` true/false。
