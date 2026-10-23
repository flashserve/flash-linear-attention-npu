# ChunkGatedDeltaRuleBwdFinalize API

## Python

```python
from fla_npu.ops.ascendc import chunk_gated_delta_rule_bwd_finalize

dq, dk, dv, dbeta, dg = chunk_gated_delta_rule_bwd_finalize(
    q,
    k,
    v,
    v_new,
    do,
    du,
    g,
    beta,
    h,
    dh,
    a,
    q_rstd=None,
    k_rstd=None,
    beta_raw=None,
    cu_seqlens=None,
    chunk_indices=None,
    scale=None,
    chunk_size=64,
    use_qk_l2_norm_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    use_gate_in_kernel=False,
    state_v_first=False,
    use_exp2=True,
)
```

这是经 Stable-ABI 适配层调用 `aclnnChunkGatedDeltaRuleBwdFinalize` 的稳定入口。算子覆盖
`chunk_bwd_dqkwg` 到 `fused_beta_sigmoid_bwd` 的反向 finalize 计算链路。

## 支持平台

实现注册 A2（Ascend 910B）、A3（Ascend 910C，`ascend910_93`）和 A5（Ascend 950），使用相同公开接口。
Python 公共入口会在调用前检查输入 tensor 所在的 NPU；其他设备返回不支持错误。
A2/A3 共用 `DAV_2201` / `arch22` 路径，完整 Finalize 链在一次混合 AIC/AIV kernel 中执行，
tiling 和 workspace 在内部按平台选择。精度、性能和内存检测结论须以对应设备、版本的 ATK 记录为准；
910B 的实机验证不能替代 910C 的实机验证。

新增的 Host 校验、view 规范化和 arch22 tiling 位于 A2/A3 私有路径，A5 沿用原有流程。
纯 `ascend950` 构建不编入这些新增 Host helper；含 A2/A3 的混合 SoC 构建保留运行期分派。

## aclnn

```cpp
aclnnStatus aclnnChunkGatedDeltaRuleBwdFinalizeGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *vNew, const aclTensor *dO, const aclTensor *du,
    const aclTensor *g, const aclTensor *beta, const aclTensor *h,
    const aclTensor *dh, const aclTensor *a,
    const aclTensor *qRstdOptional, const aclTensor *kRstdOptional,
    const aclTensor *betaRawOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    double scale, int64_t chunkSize,
    bool useQkL2NormInKernel,
    bool useBetaSigmoidInKernel,
    bool useGateInKernel,
    bool stateVFirst,
    bool useExp2,
    const aclTensor *dqOut, const aclTensor *dkOut,
    const aclTensor *dvOut, const aclTensor *dbetaOut,
    const aclTensor *dgOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnChunkGatedDeltaRuleBwdFinalize(
    void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream);
```

## 输入

| 参数 | dtype | Shape 与约束 |
| --- | --- | --- |
| `q`, `k` | BF16 | `[B,HK,T,128]` |
| `v`, `v_new`, `do`, `du` | BF16 | `[B,HV,T,V]` |
| `g`, `beta` | 同为 BF16 或同为 FP32 | `[B,HV,T]`；两者 dtype 必须相同 |
| `h`, `dh` | BF16 | `[B,NT,HV,K,V]`；`state_v_first=true` 时为 `[B,NT,HV,V,K]` |
| `a` | BF16 | `[B,HV,T,BT]` |
| `q_rstd`, `k_rstd` | FP32 | 可空；`[B,HK,T]` |
| `beta_raw` | 与 `beta` 相同 | 可空；`[B,HV,T]` |
| `cu_seqlens` | host int array | 可空；变长序列累计长度 |
| `chunk_indices` | host int array | 可空；展平的 `(seqIdx, localChunkIdx)` pair |

所有支持平台均要求 `K=128`、`B/HK/HV/T>0` 和 `HV % HK == 0`。

| 平台 | V | BT（chunk_size） | HV/HK | g/beta dtype |
| --- | --- | --- | --- | --- |
| A2 / A3 | 128、256 | 64、128 | 正整数 | 同为 BF16 或同为 FP32 |
| A5 | 128 | 64 | 1 到 4 | 同为 BF16 或同为 FP32 |

定长模式下 `NT=ceil(T/BT)`；变长模式下 `B=1`，`NT` 为所有序列的 chunk 总数。
两种模式均支持不满一个 chunk 的尾块。
A2/A3 的 B/HK/HV/T、逻辑序列数及总 chunk 数不能超出 uint32 调度范围；
host 同时检查内部地址乘积和 workspace 大小，溢出时返回参数错误。

`cu_seqlens` 和 `chunk_indices` 必须同时为空或同时提供。普通稠密格式的非连续输入
view 会由 aclnn 层连续化。A2/A3 输出必须连续，并分别保持 q/k/v/beta/g 的 shape 和 dtype；
其输出 storage/view format 接受 ND、NCHW、NCDHW、NCL，拒绝分块私有格式。
这些标签沿用 Stable/ctypes 的描述符；逻辑 BNSD/BNS 由 shape 和 stride 定义。
A5 保留原有的连续化、输出分配和描述符处理，本次不增加其 DFX 校验。

## 属性

| 属性 | 默认值 | 约束 |
| --- | --- | --- |
| `scale` | Python: `1/sqrt(128)` | 有限值；A2/A3 Host 显式检查 FP32 可表示范围 |
| `chunk_size` | `64` | A2/A3 支持 64/128；A5 支持 64；必须等于 A 的末维 |
| `use_qk_l2_norm_in_kernel` | `false` | `true` 时必须提供 `q_rstd/k_rstd` |
| `use_beta_sigmoid_in_kernel` | `false` | `true` 时必须提供 `beta_raw` |
| `use_gate_in_kernel` | `false` | 只支持 `false` |
| `state_v_first` | `false` | 支持 `false/true`，控制 `h/dh` 末两维的存储顺序 |
| `use_exp2` | `true` | 支持 `true` 和 `false` |

两个可选反向开关相互独立，四种组合均由 TilingKey 模板支持。关闭某个开关时，
对应可选输入允许为空，kernel 不读取该地址。

## 输出

| 输出 | dtype | Shape |
| --- | --- | --- |
| `dq` | 与 `q` 相同 | `[B,HK,T,128]` |
| `dk` | 与 `k` 相同 | `[B,HK,T,128]` |
| `dv` | 与 `v` 相同 | `[B,HV,T,V]` |
| `dbeta` | 与 `beta` 相同 | `[B,HV,T]` |
| `dg` | 与 `g` 相同 | `[B,HV,T]` |

dq/dk/dv 分别跟随 q/k/v；dbeta/dg 分别跟随 beta/g。由于本接口要求 g/beta 同 dtype，
dbeta/dg 也必须同为 BF16 或同为 FP32。

## 返回码

| 返回码 | 触发条件 |
| --- | --- |
| `ACLNN_SUCCESS` | workspace 查询或执行成功 |
| `ACLNN_ERR_PARAM_NULLPTR` | 必传输入/输出、`workspaceSize` 或 `executor` 为空 |
| `ACLNN_ERR_PARAM_INVALID` | shape、dtype、format、属性或变长元数据不合法 |
| `ACLNN_ERR_INNER_CREATE_EXECUTOR` | executor 创建失败 |
| `ACLNN_ERR_INNER_NULLPTR` | Contiguous/ViewCopy 或内部算子返回空 tensor |
| `ACLNN_ERR_INNER` | kernel executor 执行失败 |
