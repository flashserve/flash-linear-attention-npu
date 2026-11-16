# ChunkKdaBwdRecompute API

公开语义与支持范围以本文和 [算子 README](../README.md) 为准；流水、存储与同步见
[设计文档](design.md)。

## Python

```python
from fla_npu.ops.ascendc import chunk_kda_bwd_recompute

gk, w, u, qg, kg = chunk_kda_bwd_recompute(
    q,
    k,
    v,
    g,
    beta,
    A,
    chunk_size=64,
    A_log=A_log,
    dt_bias=dt_bias,
    cu_seqlens=None,
    chunk_indices=None,
    use_gate_in_kernel=True,
    use_exp2=True,
    lower_bound=-5.0,
    input_layout="BNSD",
)
```

这是 Stable-ABI / ctypes 直调 `aclnnChunkKdaBwdRecompute` 的稳定入口。
同一实现也以 `npu_chunk_kda_bwd_recompute` 暴露。Python 返回顺序以 `gk`
开头；aclnn 输出登记顺序为 `w/u/qg/kg/gk`，适配层在返回前重排。

### 参数

| 名称 | 默认 | 类型 / Shape | 说明 |
| --- | --- | --- | --- |
| `q` / `k` | 必选 | `[B, HK, T, 128]`；BF16 | query / key；GQA 要求 `HV % HK == 0` |
| `v` | 必选 | `[B, HV, T, 128]`；BF16 | value |
| `g` | 必选 | `[B, HV, T, 128]`；BF16/FP32 | raw gate；`use_gate_in_kernel=true` 时与 `A_log`/`dt_bias` 一起算 safe-gate |
| `beta` | 必选 | `[B, HV, T]`；BF16/FP32 | delta 系数（已生效，kernel 内不再 sigmoid） |
| `A` | 必选 | `[B, HV, T, 64]`；BF16 | 前向保存的三角系统解（`Akk` 或等价量） |
| `chunk_size` | `64` | int | 仅支持 `64` |
| `A_log` | `None` | `[HV]`；FP32 | `use_gate_in_kernel=true` 时必选 |
| `dt_bias` | `None` | `[HV, 128]` 或等价扁平；FP32 | 可选 gate bias |
| `cu_seqlens` | `None` | host int array `[N+1]` | 变长边界；须与 `chunk_indices` 同时提供 |
| `chunk_indices` | `None` | host int array `[2*N_c]` | sequence-major `(seq_id, local_chunk_id)` |
| `use_gate_in_kernel` | `True` | bool | true 时 kernel 内算 gate 并产出 `gk` |
| `use_exp2` | `True` | bool | true 时累计 gate 落在 log2 域（乘 `1/ln(2)`） |
| `lower_bound` | `-5.0` | float | safe-gate 下界，须位于 `[-5, 0)` |
| `input_layout` | `"BNSD"` | str | 仅接受 `BNSD` / `NTD`；不触发布局转换 |

输入须为非私有 ND 格式；L2 会对非连续输入做连续化。独立入口与 ATK 主路径
以定长 dense BNSD 为主。Shape 符号见
[KDA 模型符号表](../../README.md#model-shape-symbols)。

### 返回

Python 固定五元组：

```text
(gk, w, u, qg, kg)
```

| 输出 | shape | dtype | 说明 |
| --- | --- | --- | --- |
| `gk` | `[B, HV, T, 128]` | FP32 | chunk 内累计 gate；`use_gate_in_kernel=false` 时 Python 可为 `None` |
| `w` / `qg` / `kg` | `[B, HV, T, 128]` | BF16 | 反向检查点 |
| `u` | `[B, HV, T, 128]` | BF16 | 反向检查点 |

## aclnn

```cpp
aclnnStatus aclnnChunkKdaBwdRecomputeGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *a,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize,
    bool useExp2,
    double lowerBound,
    const char *inputLayout,
    const aclTensor *wOut,
    const aclTensor *uOut,
    const aclTensor *qgOut,
    const aclTensor *kgOut,
    const aclTensor *gkOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnChunkKdaBwdRecompute(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

`q/k/v/g/beta/a` 与 `wOut/uOut/qgOut/kgOut` 必传。`inputLayout` 仅精确接受
`BNSD`/`NTD`。独立 L2 用「`gkOutOptional` 是否非空」推导
`useGateInKernel`：非空则开启 gate 并写出 `gk`；为空则分配内部 FP32 缓冲
占位，不作为公开输出。L0 仍接收显式 `useGateInKernel` 布尔属性。

L0 签名：

```cpp
const std::array<const aclTensor *, 5> l0op::ChunkKdaBwdRecompute(
    const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *g, const aclTensor *beta, const aclTensor *a,
    const aclTensor *aLogOptional, const aclTensor *dtBiasOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    int64_t chunkSize, bool useGateInKernel, bool useExp2,
    double lowerBound, const char *inputLayout,
    const aclTensor *wOut, const aclTensor *uOut,
    const aclTensor *qgOut, const aclTensor *kgOut,
    const aclTensor *gkOut, aclOpExecutor *executor);
```

返回数组顺序为 `{wOut, uOut, qgOut, kgOut, gkOut}`。`inputLayout` 在 L0/L2
仅做字符串校验与属性透传；张量仍须满足既有 rank-4 BNSD（或总层提供的
batch=1 视图）契约，本算子不按该串重建视图或转置。

## 数学边界

令 `x = g + dt_bias`。`use_gate_in_kernel=true` 时：

```text
gate = lower_bound * sigmoid(exp(A_log) * x)
gk_i = cumsum(gate)_i * (1 / ln(2))    # use_exp2=true；实现里折入 gateScale
```

随后按 GQA 将 `q/k` 展开到 `HV`，每个 chunk：

```text
e2  = exp2(gk)
qg  = q_hv * e2
kbg = k_hv * beta * e2
vb  = v * beta
kg  = k_hv * exp2(gk_last - gk)
w   = A @ kbg
u   = A @ vb
```

`w/u` 以 BF16 操作数进入 Cube、FP32 累加后 Fixpipe 为 BF16。尾 chunk 只消费
有效行/列。完整精度与 Stage 划分见 [设计文档](design.md#数学与精度)。

## 支持范围

| 维度 | 支持 |
| --- | --- |
| SoC | 注册 A2 / A3 / A5；ATK 与性能验收以 A5 为主 |
| `K` / `V` | 固定 `128` |
| `chunk_size` | 固定 `64` |
| head | `0 < HK <= HV` 且 `HV % HK == 0` |
| dtype | `q/k/v/A` 为 BF16；`g/beta` 为 BF16 或 FP32；`A_log`/`dt_bias`/`gk` 为 FP32 |
| layout | `input_layout ∈ {BNSD, NTD}`；独立入口张量为 dense BNSD rank-4 |
| 变长 | 可选 `cu_seqlens` + `chunk_indices`；当前 ATK 主路径为定长 |

A2/A3 tiling 要求公开 `gk` 输出缓冲存在；A5 允许 `gk` 为空并由 L2 分配
内部占位。在整体 `chunk_kda_bwd` 的 A2/A3 编排中，本算子位于
`Recompute → FwdH → Prepare → Dhu → Finalize` 链的起点，见
[`../chunk_kda_bwd/docs/A2A3_design.md`](../chunk_kda_bwd/docs/A2A3_design.md)。

## 返回码

| 返回码 | 触发条件 |
| --- | --- |
| `ACLNN_SUCCESS` | workspace 查询和执行成功 |
| `ACLNN_ERR_PARAM_NULLPTR` | 必传输入/输出或 `workspaceSize`/`executor` 为空 |
| `ACLNN_ERR_PARAM_INVALID` | dtype、shape、`chunk_size`、`input_layout` 或不连续等不合法；L0 launch 失败 |
| `ACLNN_ERR_INNER_CREATE_EXECUTOR` | executor 创建失败 |
| `ACLNN_ERR_INNER_NULLPTR` | 中间 descriptor / 内部 `gk` 缓冲创建失败 |

## 验证

ATK：[`tests/atk/chunk_kda_bwd_recompute`](../../../../../../tests/atk/chunk_kda_bwd_recompute/README.md)。
离线冒烟：`torch_custom/fla_npu/test/test_npu_chunk_kda_bwd_recompute.py`。

精度门限（相对融合 CPU 标杆的 max abs diff）：`gk/qg/kg < 0.05`，
`w/u < 0.15`。
