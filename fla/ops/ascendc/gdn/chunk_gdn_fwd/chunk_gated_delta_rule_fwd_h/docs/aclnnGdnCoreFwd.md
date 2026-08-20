# aclnnGdnCoreFwd

`aclnnGdnCoreFwd` is the default composite ACLNN entry for the GDN core forward path.

Versioned entries preserve ablation checkpoints:

| Entry | Path |
| --- | --- |
| `aclnnGdnCoreFwdPhase1` | six independent GDN stage kernels in one executor |
| `aclnnGdnCoreFwdPhase2` | fused `ChunkKktSolveTri` plus the other four stage kernels |
| `aclnnGdnCoreFwdPhase3` | cumulative `ChunkCumsumKktSolveTri` plus the remaining three stage kernels |
| `aclnnGdnCoreFwdPhase4` | Phase 3 preprocessing plus fused `ChunkGatedDeltaRuleFwdHO` |
| `aclnnGdnCoreFwdPhase5` | Phase 3 preprocessing plus fused `ChunkRecomputeWUFwdHO` |
| `aclnnGdnCoreFwdPhase6` | single `ChunkGdnCoreFwd` for `ABC+DEF`, including public BHT-to-BTH cumsum writeback |
| `aclnnGdnCoreFwd` | compatibility alias, currently equivalent to Phase 2 |

Phase 1 preserves:

```text
chunk_local_cumsum -> chunk_scaled_dot_kkt -> solve_tri ->
recompute_w_u -> chunk_gated_delta_rule_fwd_h -> chunk_fwd_o
```

Phase 2 replaces KKT and solve_tri with `ChunkKktSolveTri`.
Phase 3 additionally absorbs the raw-g local cumsum into
`ChunkCumsumKktSolveTri`, while preserving the public FP32 `gCumsumOut` and the
same solved-A contract. Phase 4 preserves that preprocessing route and executes
`fwd_h -> fwd_o` in one `ChunkGatedDeltaRuleFwdHO` MIX kernel; its internal
`h` and `v_new` buffers live in the ACLNN workspace rather than as executor
tensors. Phase 5 additionally fuses `recompute_w_u -> fwd_h -> fwd_o` into one
`ChunkRecomputeWUFwdHO` MIX kernel. Its accepted Round2 scheduler flattens the
recompute work over `chunk x value_head`; `w/u/h/v_new` remain internal
workspace hand-offs rather than public executor tensors. Phase 6 additionally
combines the cumulative ABC prefix and the DEF suffix in one MIX kernel. It uses
owner-complete rows for the public BTH `gCumsumOut` and keeps ABC/recompute tasks
on the same AIC without an outer full-core barrier. All fixed entries share the same public tensor contract and are
exported by the same package so they can be compared without reinstalling a
different wheel.

Phase-to-Phase production benchmarks measure this complete six-stage GDN
forward core path. They are not full Demo/model benchmarks and do not include
causal convolution, RMSNorm, or the output gate.

## Interface

```cpp
aclnnStatus aclnnGdnCoreFwdGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *initialStateOptional,
    bool outputFinalState,
    int64_t chunkSize,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    double scale,
    const aclTensor *oOut,
    const aclTensor *finalStateOutOptional,
    const aclTensor *gCumsumOut,
    const aclTensor *aOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnGdnCoreFwd(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## Tensor Contract

| Tensor | Shape | Dtype |
| --- | --- | --- |
| `q`, `k` | `[B, Hk, T, 128]` | `FLOAT16`, `BFLOAT16` |
| `v`, `oOut` | `[B, Hv, T, 128]` | same as `q` |
| `beta` | `[B, T, Hv]` | same as `q`, or `FLOAT` |
| `g`, `gCumsumOut` | `[B, T, Hv]` | `FLOAT` |
| `aOut` | `[B, Hv, T, chunkSize]` | same as `q` |
| `initialStateOptional` | `[N, Hv, 128, 128]` | `FLOAT`, or same as `q` |
| `finalStateOutOptional` | `[N, Hv, 128, 128]` | same as initial state; `FLOAT` when it is absent |

All tensors use `ND` storage format. The first six-stage version expects the
model's existing GVA expansion to have made `Hk == Hv` before entry. Native
`Hk != Hv` is a later optimization. `chunkSize` must be `64` or `128`.

For dense input, `N = B`. For variable-length input, physical `B` must be `1`,
`N = len(cuSeqlensOptional) - 1`, and both metadata arrays must be supplied.
`cuSeqlensOptional` starts at zero and ends at `T`. `chunkIndicesOptional` is a
flattened sequence-major list of `(sequence_id, local_chunk_id)` pairs.

The first composite implementation is deliberately limited to `K = V = 128`.
`K = 128, V = 256` will be added without changing the high-level GDN call site.

## Python

```python
from fla_npu.ops.ascendc import gdn_core_fwd

o, final_state, g_cumsum, a = gdn_core_fwd(
    q,
    k,
    v,
    g,
    beta,
    initial_state=initial_state,
    output_final_state=True,
    chunk_size=64,
    cu_seqlens=cu_seqlens,
    chunk_indices=chunk_indices,
    scale=128 ** -0.5,
)
```

Use `gdn_core_fwd_phase1`, `gdn_core_fwd_phase2`, `gdn_core_fwd_phase3`,
`gdn_core_fwd_phase4`, `gdn_core_fwd_phase5`, and `gdn_core_fwd_phase6` for
versioned Phase A/B.
Phase 5 is accepted for the
frozen A2 `K==V==128`, external-GVA scope; later specification gates must add
new checkpoints instead of changing this path in place. The unversioned
`gdn_core_fwd` is only the current default and remains equivalent to Phase 2
for compatibility.

Phase 6 has completed its A2 validation and Git archive but is not the unversioned
default. The accepted scope is `K=V=128`, external GVA, physical `H=8`, dense
and canonical varlen metadata, FP16/BF16, and `chunk_size=64/128` (C64/C128),
including the frozen initial/final-state contracts. It remains intentionally
out of scope for `V=256`, native GVA, backward, and outer-model fusion. The
unversioned alias stays on Phase 2 until a separate default-entry decision.

The public wrapper is eager-only. The forward's existing Python autograd wrapper
continues to use the established GDN backward chain.
