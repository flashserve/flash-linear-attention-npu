# ChunkKdaFwd API

## Python 主入口

`from fla_npu.ops.ascendc import chunk_kda_fwd`（同一实现也以 `npu_chunk_kda_fwd` 暴露，
Stable-ABI launcher 与 ctypes 两条后端共用这一签名）。

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
    # 反向 L2 norm 保存值出口：五个都不传就仍是空槽，行为与历史版本逐位一致
    q_hat_out=None,
    k_hat_out=None,
    q_rstd_out=None,
    k_rstd_out=None,
    beta_eff_out=None,
)
```

### 参数

| 名称 | 默认 | 类型 / Shape | 说明 |
| --- | --- | --- | --- |
| `q` / `k` | 必选 | 输入 layout 对应 Shape；FP16/BF16 | query / key；三算子组合场景要求 BF16 |
| `v` | 必选 | 与 `q` 同 dtype | value |
| `g` | 必选 | 去掉 V 维的 Shape（含 K 维）；FP32/BF16 | raw gate，或已激活的自然对数 gate |
| `beta` | 必选 | 去掉 K 维的 Shape；FP32/BF16 | delta 系数 |
| `scale` | 必选 | float | attention scale |
| `chunk_size` | `64` | int | 分块长度；支持 `64`（三段式）与 `32`（单 kernel 大融合，见下），其它取值参数校验阶段返回 `ACLNN_ERR_PARAM_INVALID` |
| `layout` | `"BSND"` | str | `BSND`/`BNSD`/`TND`/`NTD`，只解释输入 |
| `initial_state` | `None` | `[N,H_v,K,V]` 或 `state_v_first=true` 时 `[N,H_v,V,K]`；FP32 | 算子就地更新，第 12 个返回值就是它本身 |
| `output_final_state` | `False` | bool | 控制第 2 槽 `final_state` 是否返回 |
| `cu_seqlens` | `None` | `[N+1]`；INT64 | 变长序列边界 |
| `chunk_indices` | `None` | `[2*N_c]`；INT64 | canonical chunk 顺序；只给 `cu_seqlens` 时按 `chunk_size` 自动生成 |
| `safe_gate` | `False` | bool | 走 safe gate 形式 |
| `lower_bound` | `None` | float | safe gate 下界，缺省等价 `-5.0` |
| `use_gate_in_kernel` | `False` | bool | `true` 时 kernel 内用 `A_log`/`dt_bias` 算 gate |
| `A_log` | `None` | `[H_v]`；FP32 | `use_gate_in_kernel=true` 时必选 |
| `dt_bias` | `None` | `[H_v*K]`；FP32 | gate bias |
| `disable_recompute` | `False` | bool | `true` 时返回反向所需的 `w/u/qg/kg/v_new`（见返回策略） |
| `return_intermediate_states` | `False` | bool | `true` 时额外返回 `h` |
| `state_v_first` | `False` | bool | state 末两维顺序 |
| `epsilon` | `1e-6` | float | 仅 `use_qk_l2norm_in_kernel=true` 时参与 rsqrt |
| `use_qk_l2norm_in_kernel` | `False` | bool | `true` 时 kernel 内做 q/k 归一化，并产出 `q_hat/k_hat/q_rstd/k_rstd` |
| `use_beta_sigmoid_in_kernel` | `False` | bool | `true` 时 kernel 内做 `sigmoid(beta)`，并产出 `beta_eff` |
| `allow_neg_eigval` | `False` | bool | `true` 时必须同时 `use_beta_sigmoid_in_kernel=true` |
| `use_exp2` | `True` | bool | 门控走 `exp2`；`false` 时走自然指数 |
| `q_hat_out` / `k_hat_out` | `None` | `[B,HK,T,D]`（packed `[HK,T,D]`）；与 q/k 同 dtype | 传入即导出归一化后的 q/k |
| `q_rstd_out` / `k_rstd_out` | `None` | `[B,HK,T]`（packed `[HK,T]`）；FP32 | 传入即导出 rstd；`use_qk_l2norm_in_kernel=false` 时不产出 |
| `beta_eff_out` | `None` | `[B,HV,T]`（packed `[HV,T]`）；FP32 | 传入即导出生效 beta；`use_beta_sigmoid_in_kernel=false` 时不产出 |

`epsilon` / `use_qk_l2norm_in_kernel` / `use_beta_sigmoid_in_kernel` / `allow_neg_eigval` /
`use_exp2` 取非默认值时，chunk64 调用必须落在三算子组合场景（BF16 q/k/v、`K=V=128`、
`chunk_size=64`）；chunk32 则必须取下文规定的 fused 固定组合。其它场景由 Python 入口在
发起调用前按 fla-org 参考实现拒绝，不会静默忽略；具体字段语义见
[归一化 / gate 开关](#归一化--gate-开关)。

`chunk_size=32` 使用本目录 `arch35/chunk32/` 中的单 kernel 大融合实现（issue #829）。当前契约：

- q/k/v 为 BF16，`K=V=128`；g/beta 各自可为 BF16 或 FP32，任一为 FP32 时 host 将两者
  统一上转 FP32；`A_log`、`dt_bias` 和 state 为 FP32。
- `T>0`，kernel 处理非 128 对齐尾块；支持 dense batch、varlen，以及 `HV` 为 `H` 正整数倍的 GVA。
  varlen 支持 BSND（`B=1`）、TND、NTD，不支持 BNSD；最多 1024 个序列并允许空序列。
- 支持可选 `initial_state` 和 `output_final_state`；`state_v_first=true` 由 host 转置适配。
- 必须 `safe_gate=true`、`use_gate_in_kernel=true` 并提供 `A_log`；`dt_bias=None` 等价于零偏置。
  gate 语义为 `lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`。
- 必须 `use_qk_l2norm_in_kernel=true`、`use_beta_sigmoid_in_kernel=true`、
  `allow_neg_eigval=false`、`use_exp2=true`。
- 不导出 recompute 中间态或反向保存值：`disable_recompute` 和
  `return_intermediate_states` 必须为 `false`，`Aqk`/`Akk` 槽位返回 `None`，
  `gk/w/u/qg/kg/v_new/h` 也为 `None`。
- `chunk_indices` 仅接受与 `cu_seqlens` 对应的 canonical sequence-major 顺序。
- Python 的 `dt_bias=None` 会补成 FP32 零张量；aclnn chunk32 路径本身要求
  `aLogOptional/dtBiasOptional` 非空。Python 的 `state_v_first=true` 在调用前后转置，
  传到 fused L2 的 state 固定为 K-first。

### 返回

12 个槽位顺序固定，槽位数与参数取值无关；不产出的槽位为 `None`：

```text
(attn_out, final_state, gk, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state)
```

| # | 返回 | 何时非 `None` | 布局 |
| --- | --- | --- | --- |
| 0 | `attn_out` | 始终 | 固定 sequence-major（rank-4 为 BSND，rank-3 为 TND） |
| 1 | `final_state` | `output_final_state=true` | `[N,H_v,K,V]`，`state_v_first=true` 时末两维交换 |
| 2 | `gk` | chunk64 且 `use_gate_in_kernel=false` 或 `disable_recompute=true`；chunk32 返回 `None` | head-major |
| 3 | `Aqk` | `chunk_size=64`；chunk32 不导出，返回 `None` | head-major |
| 4 | `Akk` | `chunk_size=64`；chunk32 不导出，返回 `None` | head-major |
| 5–9 | `w` / `u` / `qg` / `kg` / `v_new` | chunk64 且 `disable_recompute=true`；chunk32 返回 `None` | head-major |
| 10 | `h` | chunk64 且 `disable_recompute=true` 或 `return_intermediate_states=true`；chunk32 返回 `None` | sequence-major |
| 11 | `initial_state` | 始终 | Python 层对入参 `initial_state` 的原对象透传（算子就地更新），不是 aclnn 输出 |

保留策略与 fla-org
[`chunk_kda_fwd`](https://github.com/fla-org/flash-linear-attention/blob/0f0f0c97af39343855b43bbbaddcedfda5cb9d77/fla/ops/kda/chunk_fwd.py)
提交 `0f0f0c97af39343855b43bbbaddcedfda5cb9d77` 对齐。各槽位的完整 Shape 见
[输入与输出布局](#输入与输出布局)。

### 反向 L2 norm 保存值（可选导出）

`use_qk_l2norm_in_kernel=true` 时算子内部完成 q/k 归一化并算出反向回代所需的保存值。
这些值不占上面 12 个返回槽位，而是由调用方**按需传入输出张量**导出；不传即 `nullptr`，
行为与历史版本逐位一致：

| 输出 | Shape | dtype | 何时产出 |
| --- | --- | --- | --- |
| `q_hat` / `k_hat` | `[B,HK,T,D]`（packed `[HK,T,D]`） | 与 q/k 同 dtype | 传入对应输出张量 |
| `q_rstd` / `k_rstd` | `[B,HK,T]`（packed `[HK,T]`） | FP32 | 同上；`use_qk_l2norm_in_kernel=false` 时不产出 |
| `beta_eff` | `[B,HV,T]`（packed `[HV,T]`） | FP32 | 同上；`use_beta_sigmoid_in_kernel=true` 时为 `sigmoid(beta)`（`allow_neg_eigval=true` 时为 `2*sigmoid(beta)`） |

```python
# 密集 BSND：q/k 为 [B,T,HK,D]，v 为 [B,T,HV,V]
q_hat = torch.empty((B, HK, T, D), dtype=q.dtype, device=q.device)
k_hat = torch.empty_like(q_hat)
q_rstd = torch.empty((B, HK, T), dtype=torch.float32, device=q.device)
k_rstd = torch.empty_like(q_rstd)
beta_eff = torch.empty((B, HV, T), dtype=torch.float32, device=q.device)

attn_out, final_state, gk, aqk, akk, w, u, qg, kg, v_new, h, state = chunk_kda_fwd(
    q, k, v, g, beta, D ** -0.5, 64,
    layout="BSND",
    safe_gate=True,
    lower_bound=-5.0,
    use_gate_in_kernel=True,
    A_log=A_log,
    dt_bias=dt_bias,
    disable_recompute=True,
    use_qk_l2norm_in_kernel=True,
    use_exp2=True,
    q_hat_out=q_hat,
    k_hat_out=k_hat,
    q_rstd_out=q_rstd,
    k_rstd_out=k_rstd,
    beta_eff_out=beta_eff,
)
# 不传这 5 个输出张量时返回槽位与取值都不变，只是不会写出保存值。
```

packed（`TND`/`NTD`）下把 `[B,HK,T,D]`/`[B,HK,T]`/`[B,HV,T]` 换成
`[HK,T,D]`/`[HK,T]`/`[HV,T]` 即可。

导出的 `q_rstd/k_rstd` 可直接交给 `chunk_kda_bwd`，走 optimized（L2Norm 回代）路径，语义与
fla-org 的 `l2norm_fwd` → `save_for_backward` → `l2norm_bwd` 一致。配套入口
`fla_npu.ops.ascendc.chunk_kda_fwd_prepare` 暴露三算子组合里的 Prepare 段（13 个输出槽同样
可选传），调用方可以自行编排 `Prepare -> ChunkFwdH -> ChunkKdaFwdFinalize` 并直接取用这些
保存值。

输入维度契约：`K/V` 只支持 `K=V=64` 与 `K=V=128` 两档，混合档（如 `K=64,V=128`）与其它
取值（含 `V=256`）都在参数校验阶段返回 `ACLNN_ERR_PARAM_INVALID`，报错文本会打印实际的
`Kdim/Vdim`；Python 入口在发起调用前给出同一条约束说明。

空 tensor 契约：`B` 或序列长度（TND/NTD 为总 token 数）为 0 时同样在参数校验阶段返回
`ACLNN_ERR_PARAM_INVALID`，报错文本会指明是 batch 维还是序列维为 0；host 侧不会把空
tensor 交给 tiling，因此调用方不会只拿到缺少参数上下文的错误码。

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

- chunk64 的 `attnOut/aqkOut` 是必选输出；`akkOut` 与 op def 一致为可选，传 `nullptr`
  时算子内部自建不导出的占位张量，三算子组合入口随之落到 Prepare 的 `none` 档
  （少一次 `Akk` 搬出）。chunk32 只要求 `attnOut`，详见下面的 chunk32 输出契约。
- `finalStateOut/gkOut/wOut/uOut/qgOut/kgOut/vNewOut/hOut` 均为相互独立的可选输出。
- `w/u/qg/kg/vNew/h` 的 L0 阶段固定写内部 compute 张量；对应可选输出非空时，L2 通过
  `ViewCopy` 导出，为空时只保留前向内部生命周期。`gkOut` 非空时直接复用为 `gkCompute`，
  避免目标场景额外复制整张 FP32 gate。
- `finalStateOut != nullptr` 同时表示本次需要计算并写出最终状态。
- `hCompute` 是 FwdH 到 Finalize 的内部必需 NT-first 张量；`hOut` 是独立的公开可选输出，两者不再需要交换 chunk/head 轴。
  `hOut == nullptr` 不会跳过内部 `hCompute`，只是不向调用方公开该中间状态；非空时由
  L2 转为固定 sequence-major 后导出。

`output_final_state/disable_recompute/return_intermediate_states` 只存在于 Python 和 legacy torch
包装层，由上层按 FLA 的保留策略决定向 L2 传入哪些输出指针。

### V2 入口 `aclnnChunkKdaFwdV2`：chunk64 组合与 chunk32 融合分发

`aclnnChunkKdaFwdV2GetWorkspaceSize/aclnnChunkKdaFwdV2` 同时承接两类实现：chunk64 在
同一个 executor 内组合 `ChunkKdaFwdPrepare -> ChunkFwdH -> ChunkKdaFwdFinalize`；chunk32
则转入 `RunChunkKdaFwd32Fused`，单次启动私有 `ChunkKdaFwdFused`。入口关系如下：

| 调用场景 | 公开入口 | 实际实现 |
| --- | --- | --- |
| chunk64 私有 L0 路径 | `aclnnChunkKdaFwdGetWorkspaceSize` | 外层 `ChunkKdaFwd`，固定默认组合（调用方预先归一化 q/k、预先 sigmoid beta、`exp2` 门控） |
| chunk64 三算子组合 | `aclnnChunkKdaFwdV2GetWorkspaceSize` | `ChunkKdaFwdPrepare -> ChunkFwdH -> ChunkKdaFwdFinalize`，由 5 个可选开关控制 |
| chunk32 大融合 | `aclnnChunkKdaFwdV2GetWorkspaceSize`（Python 主路径）；V1 也有同实现分支 | `RunChunkKdaFwd32Fused -> ChunkKdaFwdFused`，核内 L2 norm/beta sigmoid/safe gate |
| 低层直调 | `aclnnChunkKdaFwdFusedGetWorkspaceSize` | 同一个私有 `ChunkKdaFwdFused`，但直接使用 kernel 头连续/token-major 视图 |

chunk64 V2 支持范围：`q/k/v` 为 BF16、`K=V=128`、`chunk_size=64`、公开输出连续、
`cu_seqlens` 严格递增；不满足时返回 `ACLNN_ERR_PARAM_INVALID`（提示改用融合入口）。
Python 在该场景按既有工作量/开关规则优先调用 V2，其余 chunk64 场景回落到
`aclnnChunkKdaFwd`。chunk32 因强制核内 L2 norm 和 beta sigmoid，Python 总是选择 V2；
V2 的 chunk32 分支不再套用 chunk64 的“输出连续、cu 严格递增”组合判据，而使用 fused
专用校验，允许空序列。

V2 chunk64 的形参尾部另有 5 个可选输出指针
`qHatOut/kHatOut/qRstdOut/kRstdOut/betaEffOut`，传 `nullptr` 表示不导出；同一个输入下，
传与不传这些指针的计算结果逐位一致。chunk32 不实现这些出口，任一指针对应张量非空都
直接返回 `ACLNN_ERR_PARAM_INVALID`。

#### chunk32 aclnn 输出契约

- `q/k/v/g/beta/attnOut` 必须非空；`aLogOptional/dtBiasOptional` 在 fused L2 同样必须非空。
  Python 层的 `dt_bias=None` 零填充不属于 aclnn 的可空语义。
- `initialStateOptional/finalStateOut` 可独立为空；请求 state IO 时 L2 要求
  `stateVFirst=false`，shape 为 `[N,HV,128,128]`、dtype FP32。未请求 `finalStateOut` 时，
  host 传 `[1,1,1]` 占位，tiling 将其识别为“不写出”。
- `gkOut/wOut/uOut/qgOut/kgOut/vNewOut/hOut` 必须全部为 `nullptr`，以及 V2 的 5 个反向
  保存值指针也必须全部为 `nullptr`；任一非空返回 `ACLNN_ERR_PARAM_INVALID`。
- fused 不导出 `Aqk/Akk`，Python 对 `aqkOut/akkOut` 均传 `nullptr`，返回槽也为 `None`。
  当前 L2 对非空 `Aqk/Akk` 仅检查 shape/dtype，不会由 fused kernel 写入；调用方不能把
  非空指针理解为请求了有效中间输出。
- `attnOut` 始终为 BF16 sequence-major（rank-4 BSND、rank-3 TND），包括 NTD 输入。

#### chunk32 参数错误与返回码

Python/stable 包装层会在发起 aclnn 前以自身异常拒绝不符合 chunk32 契约的调用。直接调用
`aclnnChunkKdaFwdGetWorkspaceSize` 或 V2 时，fused guard 覆盖：必需指针缺失；非法 layout、
rank/shape/dtype；`T<=0`、`K!=128` 或 `V!=128`、非 GVA；非递减 varlen 元数据或 canonical
`chunkIndices` 不满足、序列数超过 1024、rank-4 varlen `B!=1`、BNSD varlen；非正/非有限
`epsilon`；`safeGate/useGateInKernel` 未打开或缺少 `A_log/dt_bias`；L2 state V-first；以及
上述不支持的中间输出/反向保存值。V2 还要求 `useQkL2normInKernel=true`、
`useBetaSigmoidInKernel=true`、`allowNegEigval=false`、`useExp2=true`。

guard 内部对 `q/k/v/g/beta/attnOut` 空指针使用 `ACLNN_ERR_PARAM_NULLPTR`，缺少
`A_log/dt_bias` 及其它参数错误使用 `ACLNN_ERR_PARAM_INVALID`；但 V1/V2 外层 chunk32 分支
通过 `CHECK_RET` 将 `RunChunkKdaFwd32Fused` 的非成功状态统一映射为
`ACLNN_ERR_PARAM_INVALID`。因此公开 GetWorkspaceSize 的 chunk32 参数失败对外表现为
`ACLNN_ERR_PARAM_INVALID`（161002），不能仅根据最内层空指针分支推断最终返回码。
executor 创建等非 chunk32 guard 错误仍走原有 aclnn 状态契约。

低层 `aclnnChunkKdaFwdFusedGetWorkspaceSize` 也将其 checker 失败统一映射为
`ACLNN_ERR_PARAM_INVALID`。该入口要求 `aLog/dtBias/o` 非空，`finalState` 可空；dense 使用
头连续 `[B*H,T,K]`/`[B*HV,T,*]`，varlen 使用 token-major TND，epsilon 固定为 `1e-6`。
公开 layout、NTD varlen、epsilon 和 `state_v_first` 适配应使用 V2，而不是假设低层直调入口
具有相同 host 转换能力。

### 归一化 / gate 开关

V2 入口新增五个可选开关，Python 侧以关键字参数暴露，默认值即历史 fla_npu 语义：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `epsilon` | `1e-6` | 仅 `useQkL2normInKernel=true` 时参与 rsqrt 计算 |
| `useQkL2normInKernel` | `false` | `false` 时 q/k 由调用方预先归一化 |
| `useBetaSigmoidInKernel` | `false` | `false` 时 beta 由调用方预先 sigmoid |
| `allowNegEigval` | `false` | `true` 时必须同时 `useBetaSigmoidInKernel=true` |
| `useExp2` | `true` | 门控走 `exp2`；`false` 时走自然指数 |

外层 `ChunkKdaFwd` 的 chunk64 私有 L0 只实现默认组合；chunk64 显式打开
`useQkL2normInKernel`/`useBetaSigmoidInKernel`/`allowNegEigval` 或关闭 `useExp2` 时，
调用必须落在 V2 三算子组合范围内。chunk32 是独立例外：这些开关必须取 fused 规定的
固定组合，并由 V2 路由到大融合实现；其它取值直接报错，不会静默忽略。

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
| 稳定 Python | `fla_npu.ops.ascendc.chunk_kda_fwd`；chunk32 选择 V2，chunk64 按场景选择 V1/V2 |
| aclnn 通用 | `aclnnChunkKdaFwdGetWorkspaceSize/aclnnChunkKdaFwd` |
| aclnn V2 | `aclnnChunkKdaFwdV2GetWorkspaceSize/aclnnChunkKdaFwdV2`；chunk64 三算子组合或 chunk32 fused |
| aclnn fused 低层直调 | `aclnnChunkKdaFwdFusedGetWorkspaceSize/aclnnChunkKdaFwdFused`；仅 chunk32 低层视图契约 |
| legacy | 显式加载后的 `torch.ops.npu.npu_chunk_kda_fwd`；当前 op_plugin 仅接受 chunk64/128，不路由 chunk32 |
| 受限直调样例 | `torch.ops.ascend_ops.chunk_kda_fwd_direct` |

直调样例仅覆盖 dense BNSD、K=128、V=128，并保留“调用方传入已累计 gk”的低层测试接口；
直调路径是低层诊断入口，不套用公开的 `K/V` 档位拦截；公开顶层语义与全部参数约束以
稳定 Python/aclnn 接口为准。
