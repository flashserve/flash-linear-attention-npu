# ChunkKdaFwd API

## Python 主入口

```python
from fla_npu.ops.ascendc import chunk_kda_fwd

outputs = chunk_kda_fwd(
    q, k, v, g, beta, scale, chunk_size,
    layout="BSND",
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    chunk_indices=None,
    safe_gate=False,
    lower_bound=None,
    use_gate_in_kernel=False,
    A_log=None,
    dt_bias=None,
    disable_recompute=False,
    return_intermediate_states=False,
    state_v_first=False,
    epsilon=1e-6,
    use_qk_l2norm_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    use_exp2=True,
)
```

返回：

```text
(attn_out, final_state, gk, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state)
```

可选输出在 Python 层返回 `None`。`Aqk/Akk` 始终存在；其余保留策略见算子 README。

反向 L2 norm 的保存值不占上述 12 个返回槽位，而是由调用方按需传入输出张量导出：
`q_hat_out/k_hat_out/q_rstd_out/k_rstd_out/beta_eff_out`（都不传即 `nullptr`，行为与历史
版本逐位一致）。`use_qk_l2norm_in_kernel=false` 时不产出 `q_rstd/k_rstd`；
`use_beta_sigmoid_in_kernel=false` 时不产出 `beta_eff`。导出的 `q_rstd/k_rstd` 可直接交给
`chunk_kda_bwd` 走 optimized（L2Norm 回代）路径。

输入维度契约：`K/V` 只支持 `K=V=64` 与 `K=V=128` 两档，混合档（如 `K=64,V=128`）与其它
取值（含 `V=256`）都在参数校验阶段返回 `ACLNN_ERR_PARAM_INVALID`，报错文本会打印实际的
`Kdim/Vdim`；Python 入口在发起调用前给出同一条约束说明。

## aclnn

### 融合入口 `aclnnChunkKdaFwd`（签名与 ABI 未变）

```cpp
aclnnStatus aclnnChunkKdaFwdGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclTensor *initialStateOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const char *layout,
    double scale,
    int64_t chunkSize,
    bool safeGate,
    double lowerBound,
    bool useGateInKernel,
    bool stateVFirst,
    const aclTensor *attnOut,
    const aclTensor *finalStateOut,
    const aclTensor *gkOut,
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

aclnnStatus aclnnChunkKdaFwd(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

aclnn L2 只描述张量与算法契约，不接收或解释 autograd 重计算策略：

- `attnOut/aqkOut` 是必选输出；`akkOut` 与 op def 一致为可选，传 `nullptr` 时算子内部
  自建不导出的占位张量，三算子组合入口随之落到 Prepare 的 `none` 档（少一次 `Akk` 搬出）。
- `finalStateOut/gkOut/wOut/uOut/qgOut/kgOut/vNewOut/hOut` 均为相互独立的可选输出。
- `w/u/qg/kg/vNew/h` 的 L0 阶段固定写内部 compute 张量；对应可选输出非空时，L2 通过
  `ViewCopy` 导出，为空时只保留前向内部生命周期。`gkOut` 非空时直接复用为 `gkCompute`，
  避免目标场景额外复制整张 FP32 gate。
- `finalStateOut != nullptr` 同时表示本次需要计算并写出最终状态。
- `hCompute` 是 FwdH 到 Finalize 的内部必需 head-major 张量；`hOut` 是独立的公开可选输出。
  `hOut == nullptr` 不会跳过内部 `hCompute`，只是不向调用方公开该中间状态；非空时由
  L2 转为固定 sequence-major 后导出。

`output_final_state/disable_recompute/return_intermediate_states` 只存在于 Python 和 legacy torch
包装层，由上层按 FLA 的保留策略决定向 L2 传入哪些输出指针。

### 组合入口 `aclnnChunkKdaFwdV2`

`aclnnChunkKdaFwdV2GetWorkspaceSize/aclnnChunkKdaFwdV2` 在同一个 executor 内按
`ChunkKdaFwdPrepare -> ChunkFwdH -> ChunkKdaFwdFinalize` 组合三个已交付算子，并接受
归一化 / gate 开关。它与融合入口的关系：

| 入口 | 实现 | 归一化 / gate 开关 |
| --- | --- | --- |
| `aclnnChunkKdaFwdGetWorkspaceSize` | 私有 L0 融合实现 | 固定默认组合（调用方预先归一化 q/k、预先 sigmoid beta、`exp2` 门控） |
| `aclnnChunkKdaFwdV2GetWorkspaceSize` | 三个独立算子组合 | 由 5 个可选开关控制 |

V2 支持范围：`q/k/v` 为 BF16、`K=V=128`、`chunk_size=64`、公开输出连续、`cu_seqlens`
严格递增；不满足时返回 `ACLNN_ERR_PARAM_INVALID`（提示改用融合入口）。
场景选择由 Python 入口完成：`fla_npu.ops.ascendc.chunk_kda_fwd` 命中上述场景时优先调用 V2，
其余场景（FP16、`K=V=64`、`chunk_size=128`、含空序列、输出非连续）回落到
`aclnnChunkKdaFwd`。两个入口共用同一套参数校验、输出指针语义与返回码契约，公开输出布局一致。

V2 入口的形参尾部另有 5 个可选输出指针 `qHatOut/kHatOut/qRstdOut/kRstdOut/betaEffOut`，
用于导出反向 L2 norm 需要的保存值；传 `nullptr` 表示本次不导出（Prepare 档位由非空指针
组合推导）。同一个输入下，传与不传这些指针的**计算结果逐位一致**，只有是否落盘的区别。

### 归一化 / gate 开关

V2 入口新增五个可选开关，Python 侧以关键字参数暴露，默认值即历史 fla_npu 语义：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `epsilon` | `1e-6` | 仅 `useQkL2normInKernel=true` 时参与 rsqrt 计算 |
| `useQkL2normInKernel` | `false` | `false` 时 q/k 由调用方预先归一化 |
| `useBetaSigmoidInKernel` | `false` | `false` 时 beta 由调用方预先 sigmoid |
| `allowNegEigval` | `false` | `true` 时必须同时 `useBetaSigmoidInKernel=true` |
| `useExp2` | `true` | 门控走 `exp2`；`false` 时走自然指数 |

私有 L0 融合实现只实现默认组合，因此显式打开
`useQkL2normInKernel`/`useBetaSigmoidInKernel`/`allowNegEigval` 或关闭 `useExp2` 时，
调用必须落在 V2 的组合场景范围内，否则 Python 入口直接报错（提示场景要求），不会静默忽略开关。

`ChunkKdaFwdPrepare` 的编译期 `outputMask` 由公开输出指针组合推导：

| 公开输出指针组合 | Prepare 档位 |
| --- | --- |
| 只要 `attn/final_state/gk/h`（即 `akkOut == nullptr`） | `none` |
| 额外要 `Akk`（`akkOut != nullptr`，不要 `w/u/qg/kg/v_new`） | `forward` |
| 还要 `w/u/qg/kg/v_new` 中任意一项 | `save` |

`recompute` 档（额外搬 `qHat/kHat/qRstd/kRstd/betaEff`）只由独立的 `ChunkKdaFwdPrepare`
入口使用，`chunk_kda_fwd` 的两个入口都不会触发它。

另需注意：两个入口的公开输出都固定 `attnOut` 为 sequence-major（rank-4 为 BSND、rank-3 为 TND），
`gk/Aqk/Akk/w/u/qg/kg/v_new` 固定 head-major，`h` 固定 sequence-major。

## 输入与输出布局

`layout` 只解释 q/k/v/g/beta 输入。输出固定为：

- `attnOut`: BSND 或 TND。
- `finalStateOut`: `[N,H_v,K,V]` 或 `stateVFirst=true` 时 `[N,H_v,V,K]`。
- `gkOut/AqkOut/AkkOut/wOut/uOut/qgOut/kgOut/vNewOut`: BNSD/NTD。
- `hOut`: dense 为 `[B,N_c,H_v,K,V]`，varlen 为 `[N_c,H_v,K,V]`；
  `stateVFirst=true` 时交换末两维。

完整 Shape 表见 [KDA 模型符号表](../../README.md#model-shape-symbols)。

## Gate 语义

```text
useGateInKernel=false:
    gate = g
useGateInKernel=true, safeGate=false:
    gate = -exp(A_log) * softplus(g + dt_bias)
useGateInKernel=true, safeGate=true:
    gate = lowerBound * sigmoid(exp(A_log) * (g + dt_bias))
gk = chunk_local_cumsum(gate) / ln(2)
```

`safeGate` 的 true/false 都支持；`useGateInKernel=false` 时仍支持 `safeGate=true` 的后续稳定计算路径。

## 示例

```python
import torch
from fla_npu.ops.ascendc import chunk_kda_fwd

B, T, H, K, V = 1, 128, 4, 128, 128
q = torch.randn(B, T, H, K, device="npu", dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn(B, T, H, V, device="npu", dtype=torch.float16)
g = -torch.rand(B, T, H, K, device="npu", dtype=torch.float32) * 0.01
beta = torch.rand(B, T, H, device="npu", dtype=torch.float32)

attn_out, final_state, *_ = chunk_kda_fwd(
    q, k, v, g, beta, K ** -0.5, 64,
    layout="BSND",
    output_final_state=True,
    safe_gate=True,
)
assert attn_out.shape == (B, T, H, V)
assert final_state.shape == (B, H, K, V)
```

## 调用途径

| 路径 | 入口 |
| --- | --- |
| 稳定 Python | `fla_npu.ops.ascendc.chunk_kda_fwd` |
| aclnn | `aclnnChunkKdaFwdGetWorkspaceSize/aclnnChunkKdaFwd` |
| legacy | 显式加载后的 `torch.ops.npu.npu_chunk_kda_fwd` |
| 受限直调样例 | `torch.ops.ascend_ops.chunk_kda_fwd_direct` |

直调样例仅覆盖 dense BNSD、K=128、V=128，并保留“调用方传入已累计 gk”的低层测试接口；
直调路径是低层诊断入口，不套用公开的 `K/V` 档位拦截；公开顶层语义与全部参数约束以
稳定 Python/aclnn 接口为准。
