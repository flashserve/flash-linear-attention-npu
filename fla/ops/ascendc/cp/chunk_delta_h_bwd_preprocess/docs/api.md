# ChunkDeltaHBwdPreprocess API

## 1. 算子原型

```text
ChunkDeltaHBwdPreprocess(
    q, k, w, d_o, dv, g?, gk?, cu_seqlens?,
    scale=1.0, chunk_size=64
) -> dhm
```

## 2. 参数

### 2.1 输入

| 参数 | dtype | Shape | 必选 | 说明 |
| --- | --- | --- | --- | --- |
| `q` | BF16 / FP16 | `[B, Hk, T, K]` | 是 | GDN 传原始 `q`；KDA/GDN2 传已应用 key gate 的 `qg` |
| `k` | BF16 / FP16 | `[B, Hk, T, K]` | 是 | GDN 传原始 `k`；KDA/GDN2 传 `kg`；必须与 `q` 同 shape |
| `w` | BF16 / FP16 | `[B, Hv, T, K]` | 是 | WY 辅助量 `w_wy`；不是公开输入 `w_gate` |
| `d_o` | BF16 / FP16 | `[B, Hv, T, V]` | 是 | 输出梯度 `dO` |
| `dv` | BF16 / FP16 | `[B, Hv, T, V]` | 是 | `dv_local`：不含 `K̄ @ dH` 项的本地 `dV` |
| `g` | BF16 / FP16 / FP32 | `[B, Hv, T]` | 否 | GDN 的 chunk-local 累计标量 log2 gate |
| `gk` | BF16 / FP16 | `[B, Hv, T, K]` | 否 | KDA/GDN2 的 chunk-local 累计逐 K log2 gate |
| `cu_seqlens` | INT64 | `[N+1]` | 否 | 本 segment 的 `[bos, eos)`；传入时要求 `B == 1` |

`g` 与 `gk` 必须二选一或同时缺省；同时非空时 host 直接拦截。

### 2.2 属性

| 参数 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `scale` | float | 1.0 | 只乘 `Q̄ᵀ dO` 项；通常取 `K^-0.5`。缺省时按 `1.0` 处理并把 `isScale` 置 0 |
| `chunk_size` | int | 64 | 必须与 gate cumsum、WY 生成、正式 backward 一致；本版只支持 64 |

### 2.3 输出

| 参数 | dtype | Shape | 说明 |
| --- | --- | --- | --- |
| `dhm` | FP32 | `[Hv, K, V+K]` | `[..., 0:V] = E_r`；`[..., V:V+K] = P_r` |

`dhm` 的布局与 `state_v_first` 无关；本算子不做原地写入，输入张量均只读。

## 3. aclnn 接口

```c
aclnnStatus aclnnChunkDeltaHBwdPreprocessGetWorkspaceSize(
    const aclTensor *q, const aclTensor *k, const aclTensor *w, const aclTensor *dO, const aclTensor *dv,
    const aclTensor *gOptional, const aclTensor *gkOptional, const aclIntArray *cuSeqlensOptional,
    double scale, int64_t chunkSize, const aclTensor *dhmOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

aclnnStatus aclnnChunkDeltaHBwdPreprocess(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

调用示例：

```cpp
uint64_t workspaceSize = 0;
aclOpExecutor *executor = nullptr;
auto ret = aclnnChunkDeltaHBwdPreprocessGetWorkspaceSize(
    q, k, w, dO, dv, gOptional, gkOptional, cuSeqlensOptional, scale, chunkSize, dhm,
    &workspaceSize, &executor);
void *workspace = nullptr;
aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
ret = aclnnChunkDeltaHBwdPreprocess(workspace, workspaceSize, executor, stream);
```

Python 主入口（目标形态）：

```python
from fla_npu.ops.ascendc import chunk_delta_h_bwd_preprocess

dhm = chunk_delta_h_bwd_preprocess(
    q=qg, k=kg, w=w_wy, d_o=do, dv=dv_local, gk=gk,
    scale=K ** -0.5, chunk_size=64, cu_seqlens=cu_seqlens,
)
```

## 4. `<<<>>>` 直调

设备函数签名（`kernel_operator.h` 侧）：

```cpp
extern "C" __global__ __aicore__ void chunk_delta_h_bwd_preprocess(
    GM_ADDR q, GM_ADDR k, GM_ADDR w, GM_ADDR d_o, GM_ADDR dv, GM_ADDR g, GM_ADDR gk, GM_ADDR cu_seqlens,
    GM_ADDR dhm, GM_ADDR workspace, GM_ADDR tiling);
```

直调需要调用方自行准备：`tiling`（`ChunkDeltaHBwdPreprocessTilingData`）、workspace（≥ tiling 中
`totalWsBytes` + 系统 workspace）、以及 blockDim（= tiling 中 `blockDim`）。blockDim 必须与 tiling 一致，
否则分核范围计算会越界。

## 5. tilingKey

| tilingKey | gate 模式 | `GT` 模板参数 |
| --- | --- | --- |
| 1 | 无门控（`g`、`gk` 均未传） | `DTYPE_Q` |
| 2 | `USE_G`，`g` 与 `q` 同 dtype | `DTYPE_G` |
| 3 | `USE_G`，`g` 为 FP32 | `float` |
| 4 | `USE_GK` | `DTYPE_GK` |

## 6. 返回码与拦截

| 返回码 | 触发条件 |
| --- | --- |
| `ACLNN_ERR_PARAM_NULLPTR` | `q`/`k`/`w`/`d_o`/`dv`/`dhm` 为空 |
| `ACLNN_ERR_PARAM_INVALID` | tiling 校验失败：`k` 与 `q` shape 不一致；`w` 不是 `[B,Hv,T,K]`；`dv` 与 `d_o` shape 不一致；`Hv % Hk != 0`；`K > 256`；`chunk_size != 64`；`g`/`gk` 同时非空；`g`/`gk` shape 与 dtype 不匹配；`cu_seqlens` 少于 2 项；`B > 1`（无论 dense 还是 varlen） |

报错文本会给出实际 shape、`Hk`/`Hv` 与触发的约束，便于定位；`g`/`gk` 同时非空的报错会明确指出二者互斥。

## 7. 典型 shape

shape 与取值范围的唯一维护处是 [README](../README.md) 的「支持的场景」「不支持（本版显式拦截）」「已知限制」三节；
下表只是典型调用档位示例。

| 场景 | `Hk`/`Hv` | `K` | `V` | `chunk_size` | `dhm` |
| --- | --- | --- | --- | --- | --- |
| GDN / KDA 主线 | 96 / 96 | 128 | 128 | 64 | `[96, 128, 256]` FP32（12 MiB） |
| KDA Kimi K3 档位 | 96 / 96 | 128 | 128 | 64 | 同上 |
| GVA | 32 / 96 | 128 | 128 | 64 | `[96, 128, 256]` |
