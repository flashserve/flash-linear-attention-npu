# ChunkKdaFwdPrepare API

## Python

```python
from fla_npu.ops.ascendc import chunk_kda_fwd_prepare

(
    gk,
    Aqk,
    Akk,
    w,
    u,
    qg,
    kg,
    qg_scaled,
    q_hat,
    k_hat,
    q_rstd,
    k_rstd,
    beta_eff,
) = chunk_kda_fwd_prepare(
    q,
    k,
    v,
    g,
    beta,
    scale=1.0,
    *,
    layout="BNSD",
    chunk_size=64,
    epsilon=1e-6,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    lower_bound=-5.0,
    use_exp2=False,
    a_log=None,
    dt_bias=None,
    cu_seqlens=None,
    chunk_indices=None,
)
```

这是 ctypes 直调 `aclnnChunkKdaFwdPrepare` 的稳定入口，不进入 legacy
`torch.ops.npu` 注册路径。

## aclnn

```cpp
aclnnStatus aclnnChunkKdaFwdPrepareGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const char *layout,
    double scale,
    int64_t chunkSize,
    double epsilon,
    bool useQkL2normInKernel,
    bool useGateInKernel,
    bool useBetaSigmoidInKernel,
    bool allowNegEigval,
    bool safeGate,
    double lowerBound,
    bool useExp2,
    const aclTensor *gkOut,
    const aclTensor *aqkOut,
    const aclTensor *akkOut,
    const aclTensor *wOut,
    const aclTensor *uOut,
    const aclTensor *qgOut,
    const aclTensor *kgOut,
    const aclTensor *qgScaledOut,
    const aclTensor *qHatOut,
    const aclTensor *kHatOut,
    const aclTensor *qRstdOut,
    const aclTensor *kRstdOut,
    const aclTensor *betaEffOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnChunkKdaFwdPrepare(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## 参数约束

| 参数 | dtype | 约束 |
| --- | --- | --- |
| `q/k` | BF16 | K=128；head 数为 HK |
| `v` | BF16 | V=128；head 数为 HV |
| `g` | BF16/FP32 | 与 `v` 的 B/HV/T 一致，末维 128 |
| `beta` | BF16/FP32 | 与 `v` 的 B/HV/T 一致 |
| `a_log` | FP32 | kernel 内 gate 模式必传，shape `[HV]` |
| `dt_bias` | FP32 | 可空；shape `[HV*128]` |
| `layout` | string | `BNSD`、`BSND`、`NTD` 或 `TND`，区分大小写 |
| `scale` | float | 有限值 |
| `chunk_size` | int | 固定为 64 |
| `epsilon` | float | 有限且大于 0 |
| `lower_bound` | float | SafeSigmoid 模式要求 `[-5,0)` |

`B/T/HK/HV` 必须为正数且可由 `uint32_t` 表示；GVA 要求 `0 < HK <= HV` 且
`HV % HK == 0`。tensor descriptor 必须使用标准、
非私有且可连续化的 ND 物理布局；输出必须连续，因为 kernel 直接写入输出地址。
`BSND/TND` 输入还要求 q/k、v、g 和 beta 的跨 head DMA stride 均可由 `uint32_t`
表示，完整公式见 [README 输入输出约束](../README.md#输入输出)。

## Shape

输入 shape 和固定的 head-major 输出 shape 见 [README Shape 表](../README.md#输入输出)。
所有输出都是必选输出，dtype 与顺序如下：

```text
FP32: gk, q_rstd, k_rstd, beta_eff
BF16: Aqk, Akk, w, u, qg, kg, qg_scaled, q_hat, k_hat
```

## 返回码

| 返回码 | 触发条件 |
| --- | --- |
| `ACLNN_SUCCESS` | workspace 查询或执行成功 |
| `ACLNN_ERR_PARAM_NULLPTR` | 必传输入/输出、`workspaceSize` 或 `executor` 为空 |
| `ACLNN_ERR_PARAM_INVALID` | dtype、shape、format、属性、模式组合或变长元数据不合法 |
| `ACLNN_ERR_INNER_CREATE_EXECUTOR` | executor 创建失败 |
| `ACLNN_ERR_INNER_NULLPTR` | 连续化、元数据转换或 L0 调用未返回有效 tensor |
| `ACLNN_ERR_INNER` | kernel executor 执行失败 |
