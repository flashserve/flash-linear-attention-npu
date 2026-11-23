# ChunkGatedDeltaRuleBwd API

对 Gated Delta Rule 的分块前向结果求梯度。接口使用前向保存的门控值、Delta 系数和块内系数矩阵，以及输出和状态的上游梯度。计算流程见[设计文档](design.md)。

## Python 入口

```python
from fla_npu.ops.ascendc import npu_chunk_gated_delta_rule_bwd

dq, dk, dv, d_beta, d_g, dh0, d_a_log, d_dt_bias = npu_chunk_gated_delta_rule_bwd(
    q, k, v, g, beta, A, d_o, scale,
    chunk_size=64, cu_seqlens=None, chunk_indices=None,
    initial_state=None, dht=None, q_rstd=None, k_rstd=None, beta_raw=None,
    use_exp2=False, use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    return_intermediate_states=False,
    state_v_first=False, a_log=None, dt_bias=None, layout="BNSD",
)
```

`return_intermediate_states` 是 Python 兼容参数，默认为 `False`，取值会被忽略。

## ACLNN L2

先调用 `GetWorkspaceSize` 校验参数并取得 workspace 大小及 executor，再在指定 stream 上执行。工作区以本次调用返回的大小为准。声明见[头文件](../op_host/op_api/aclnn_chunk_gated_delta_rule_bwd.h)。

```cpp
aclnnStatus aclnnChunkGatedDeltaRuleBwdGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *g, const aclTensor *beta, const aclTensor *a,
    const aclTensor *dO, const aclTensor *initialStateOptional,
    const aclTensor *dhtOptional, const aclTensor *qRstdOptional,
    const aclTensor *kRstdOptional, const aclTensor *betaRawOptional,
    const aclTensor *aLogOptional, const aclTensor *dtBiasOptional,
    const aclIntArray *cuSeqlensOptional, const aclIntArray *chunkIndicesOptional,
    const char *layout, double scale, int64_t chunkSize, bool useExp2,
    bool useGateInKernel, bool useQkL2normInKernel, bool useBetaSigmoidInKernel,
    bool stateVFirst, const aclTensor *dqOut, const aclTensor *dkOut,
    const aclTensor *dvOut, const aclTensor *dBetaOut, const aclTensor *dGOut,
    const aclTensor *dh0OutOptional, const aclTensor *dALogOutOptional,
    const aclTensor *dDtBiasOutOptional, uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnChunkGatedDeltaRuleBwd(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
    aclrtStream stream);
```

## 支持平台

| 平台 | NPU 架构 | `Hv / Hk` |
| --- | --- | --- |
| A2 Ascend 910B、A3 Ascend 910C | `DAV_2201` | 1～4 的整数 |
| A5 Ascend 950 | `DAV_3510` | 1～16 的整数 |

各平台的 `K`、`V` 均为 128，`chunk_size` 为 64。非空输入要求 `B`、`Hk`、`Hv` 和 `T` 均为正，变长模式使用物理 `B=1`。前向保存的张量和上游梯度使用同一 NPU。

## 输入

`B` 是物理 batch，`T` 是 token 维度长度，`Hk`、`Hv` 分别是 Q/K 头数和 V 头数，`K`、`V` 是对应头维度。`Nseq` 为逻辑序列数，定长为 `B`，变长为 `len(cu_seqlens) - 1`。

| Python | ACLNN | Shape、dtype 与语义 |
| --- | --- | --- |
| `q`、`k` | `q`、`k` | 必选，BF16，shape 相同并由 `layout` 决定。开启 Q/K L2Norm 反向时传前向归一化后的 Q/K。 |
| `v` | `v` | 必选，BF16，shape 由 `layout` 决定。 |
| `g` | `g` | 必选，BSN `[B,T,Hv]`，BF16 或 FP32，传前向保存的块内累计门控值。 |
| `beta` | `beta` | 必选，BSN `[B,T,Hv]`，BF16 或 FP32，传前向使用的 Delta 系数。 |
| `A` | `a` | 必选，`[B,Hv,T,64]`，BF16，传前向保存的块内系数矩阵。 |
| `d_o` | `dO` | 必选，BSND `[B,T,Hv,V]`，BF16，前向输出的上游梯度。 |
| `initial_state` | `initialStateOptional` | 可选，BF16，前向初始状态。ACLNN 中与 `dh0OutOptional` 成对提供。 |
| `dht` | `dhtOptional` | 可选，BF16，末状态的上游梯度。缺省时按零末状态梯度计算。 |
| `q_rstd`、`k_rstd` | `qRstdOptional`、`kRstdOptional` | BNS `[B,Hk,T]`，FP32。开启 Q/K L2Norm 反向时传前向保存的两项 rstd，关闭时两项传空。 |
| `beta_raw` | `betaRawOptional` | BSN `[B,T,Hv]`，与 `beta` 同 dtype。开启 beta sigmoid 反向时传 sigmoid 前的值，关闭时传空。 |
| `a_log`、`dt_bias` | `aLogOptional`、`dtBiasOptional` | 预留输入，常规调用传空。当前计算忽略这两项。 |
| `cu_seqlens` | `cuSeqlensOptional` | 变长累计长度，`Nseq + 1` 个 INT64 整数。 |
| `chunk_indices` | `chunkIndicesOptional` | 按序列号和序列内 chunk 号排列的 INT64 整数对，展平后长度为总 chunk 数的两倍。 |

`g` 和 `beta` 可以各自选择 BF16 或 FP32。只要其中一项为 FP32，两项和参与计算的 `beta_raw` 均按 FP32 进入后续计算。对应梯度按各自输入 dtype 写回。

## 属性

下表为 Python 默认值。ACLNN 属性按签名显式传入。

| Python | ACLNN | 类型与默认值 | 语义 |
| --- | --- | --- | --- |
| `layout` | `layout` | str，`"BNSD"` | 取 `BNSD`、`BSND`、`NTD` 或 `TND`，控制 Q/K/V 及其梯度的公开布局。 |
| `scale` | `scale` | float，必传 | 有限的 Query 缩放因子，通常用 `K**-0.5`。 |
| `chunk_size` | `chunkSize` | int，`64` | 当前取 64。 |
| `use_exp2` | `useExp2` | bool，`False` | `True` 使用以 2 为底的指数，`False` 使用自然指数。和前向保持一致。 |
| `use_gate_in_kernel` | `useGateInKernel` | bool，`False` | 当前取 `False`，使用前向已处理的门控值。 |
| `use_qk_l2norm_in_kernel` | `useQkL2normInKernel` | bool，`False` | 开启 Q/K L2Norm 反向。 |
| `use_beta_sigmoid_in_kernel` | `useBetaSigmoidInKernel` | bool，`False` | 开启 beta sigmoid 反向。 |
| `state_v_first` | `stateVFirst` | bool，`False` | `False` 按 `[Nseq,Hv,K,V]` 解释状态，`True` 按 `[Nseq,Hv,V,K]` 解释状态。 |

## 布局

| `layout` | `q/k/dq/dk` | `v/dv` |
| --- | --- | --- |
| `BNSD`、`NTD` | `[B,Hk,T,K]` | `[B,Hv,T,V]` |
| `BSND`、`TND` | `[B,T,Hk,K]` | `[B,T,Hv,V]` |

四个布局名均对应四维输入。其余数据张量采用输入和输出表所列布局。状态 `initial_state`、`dht`、`dh0` 的末两维顺序由 `state_v_first` 控制。BSND/TND 还校验公开布局的地址上限，`Hk`、`Hv` 均不超过 `2^24`，且 `B*T*Hv*128*2 <= INT64_MAX`。

### 定长与变长

定长时两项序列元数据均传空。变长时 `cu_seqlens` 至少有两个元素，以 0 开头、以 `T` 结尾，按非递减顺序排列。每条序列按长度向上取整分块，序列内的 chunk 号从 0 开始。ACLNN 接收成对的 `aclIntArray`；Python 接收扁平整数序列或对应 Tensor，只提供 `cu_seqlens` 时自动推导 `chunk_indices`。

例如 `cu_seqlens=[0,64,64,130]` 对应 `chunk_indices=[0,0,2,0,2,1]`。空逻辑序列占用原来的序列编号，不生成 chunk。若提供初始状态，该空序列的 `dh0` 等于对应的 `dht`，缺省 `dht` 时为零。

### 连续性契约

| 输入 | L2 处理方式 |
| --- | --- |
| `q`、`k`、`v`、`A` | 连续张量直接使用。非连续 view 先按各自公开布局单独连续化。 |
| `g`、`beta`、`d_o`、`beta_raw` | 非连续 view 先单独连续化，再转换到内部计算布局。 |
| `initial_state`、`dht`、`q_rstd`、`k_rstd` | 非连续 view 先单独连续化。使用 V-first 初始状态时，重算所用的初始状态再转为 K-first。 |

Python 的 Stable-ABI 适配层按原张量格式构造描述符。布局转换和连续化由 ACLNN executor 执行。

## 输出

Python 按表中顺序返回固定八元组。ACLNN 前五个输出描述符必传，调用方按表中 shape 和 dtype 分配。

| Python | ACLNN | Shape、dtype 与语义 |
| --- | --- | --- |
| `dq` | `dqOut` | 与 `q` 同 shape 和 dtype。 |
| `dk` | `dkOut` | 与 `k` 同 shape 和 dtype。 |
| `dv` | `dvOut` | 与 `v` 同 shape 和 dtype。 |
| `d_beta` | `dBetaOut` | BSN `[B,T,Hv]`，与 `beta` 同 dtype。开启 sigmoid 反向时为 `beta_raw` 的梯度。 |
| `d_g` | `dGOut` | BSN `[B,T,Hv]`，与 `g` 同 dtype。 |
| `dh0` | `dh0OutOptional` | 提供 `initial_state` 时，与初始状态同 shape、布局和 dtype。缺省初始状态时，Python 返回 `None`，ACLNN 传 `nullptr`，包括仅提供 `dht` 的情况。 |
| `d_a_log` | `dALogOutOptional` | 预留。Python 固定返回 `None`，ACLNN 传 `nullptr`。 |
| `d_dt_bias` | `dDtBiasOutOptional` | 预留。Python 固定返回 `None`，ACLNN 传 `nullptr`。 |

### 输出存储关系

Python 为梯度分配新张量。ACLNN 将梯度写入调用方指定的存储。A2/A3 的 `dq/dk/dv/dh0` 非连续输出用连续暂存并回填；`dBeta/dG` 转换为 BSN 并恢复 dtype 后写入输出 view。

A2/A3 对 `dq/dk/dv/dBeta/dG` 判断与输入和其他梯度的存储重叠，可能重叠时使用独立暂存。公开输出的覆盖顺序为 `dq → dk → dv → dBeta → dG`，重叠字节保留顺序较后的值。需要同时保留全部梯度时，给每项输出分配独立存储。A5 调用使用连续且互不重叠的输出。实现细节见[设计文档](design.md)。

## 返回码

以下配对、属性、shape 和 dtype 约束在 L2 Host 检查。必传指针有效且 executor 创建成功后，任一必选输入为空张量时，`GetWorkspaceSize` 以 `ACLNN_SUCCESS` 和 `workspaceSize=0` 短路，不下发计算。

| 返回码 | 触发条件 |
| --- | --- |
| `ACLNN_SUCCESS` | 参数校验及对应阶段执行成功，或命中上述零 shape 短路。 |
| `ACLNN_ERR_PARAM_NULLPTR`（161001） | 必选输入或输出、`workspaceSize`、`executor` 或 `layout` 指针为空。状态输出、变长元数据、Q/K rstd 配对缺项，或开启 beta sigmoid 反向后缺少 `beta_raw`。 |
| `ACLNN_ERR_PARAM_INVALID`（161002） | SoC、layout、属性、shape、dtype 或变长数组内容超出支持范围。向预留输出传入非空描述符，或关闭相应反向选项却提供 rstd / `beta_raw`。 |
| `ACLNN_ERR_INNER_CREATE_EXECUTOR` | 创建 executor 失败。 |
| `ACLNN_ERR_INNER_NULLPTR` | 连续化、布局转换、中间 Tensor 分配、子算子组合或回填失败。 |

参数错误日志包含关联参数名和约束，例如 `chunkSize` 的实际值及预期值 64。第二段 `aclnnChunkGatedDeltaRuleBwd` 原样返回 executor 的执行错误码。Python 入口会对 `use_gate_in_kernel=True` 或未识别的布局字符串直接抛出 `RuntimeError`。
