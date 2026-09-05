# ChunkGdnBwdIntra API

[设计文档](design.md) |
[ATK 测试](../../../../../../../tests/atk/chunk_gdn_bwd_intra/README.md)

## 功能

`ChunkGdnBwdIntra` 融合以下两个设备调用，并按固定顺序返回 `w`、`u` 和
`dv_local`：

```text
recompute_w_u_fwd(k, v, beta, A, g, cu_seqlens) -> w, u
chunk_bwd_dv_local(q, k, d_o, g, scale, cu_seqlens, chunk_size) -> dv_local
```

两条计算支路没有结果依赖；算子只定义融合后的输入、属性、输出和数值语义，Stage、
workspace 与同步方案见[设计文档](design.md)。

## Python

```python
from fla_npu.ops.ascendc import chunk_gdn_bwd_intra

w, u, dv_local = chunk_gdn_bwd_intra(
    q,
    k,
    v,
    g,
    beta,
    A,
    d_o,
    scale,
    chunk_size,
    *,
    cu_seqlens=None,
    chunk_indices=None,
    use_exp2=True,
    stage=2,
)
```

私有接口支持连续 BNSD 布局，返回顺序固定为 `(w, u, dv_local)`。

## OpDef

```text
OpDef: ChunkGdnBwdIntra
Inputs:
  q, k, v, g, beta, A, d_o                  REQUIRED
  cu_seqlens, chunk_indices                  OPTIONAL, value-dependent INT64 Tensor
Attrs:
  scale                                      REQUIRED float
  chunk_size                                 REQUIRED int
  use_exp2                                   OPTIONAL bool = true
  stage                                      OPTIONAL int = 2
Outputs:
  w, u, dv_local                             REQUIRED
```

## aclnn

```cpp
ACLNN_API aclnnStatus aclnnChunkGdnBwdIntraGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *a,
    const aclTensor *dO,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    double scale,
    int64_t chunkSize,
    bool useExp2,
    int64_t stage,
    const aclTensor *wOut,
    const aclTensor *uOut,
    const aclTensor *dvLocalOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

ACLNN_API aclnnStatus aclnnChunkGdnBwdIntra(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

## L0

```cpp
namespace l0op {
const std::array<const aclTensor *, 3> ChunkGdnBwdIntra(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *a,
    const aclTensor *dO,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    double scale,
    int64_t chunkSize,
    bool useExp2,
    int64_t stage,
    const aclTensor *wOut,
    const aclTensor *uOut,
    const aclTensor *dvLocalOut,
    aclOpExecutor *executor);
}
```

## 输入与输出

| 名称 | 输入/输出 | dtype | shape |
| --- | --- | --- | --- |
| `q` | 输入 | FP16/BF16 | `[B,HK,T,K]` |
| `k` | 输入 | 与 `q` 相同 | `[B,HK,T,K]` |
| `v` | 输入 | 与 `q` 相同 | `[B,HV,T,V]` |
| `g` | 输入 | BF16/FP32 | `[B,HV,T]` |
| `beta` | 输入 | BF16/FP32 | `[B,HV,T]` |
| `A` | 输入 | 与 `q` 相同 | `[B,HV,T,chunk_size]` |
| `d_o` | 输入 | 与 `q` 相同 | `[B,HV,T,V]` |
| `cu_seqlens` | 可选输入 | INT64 | `[Seq+1]` |
| `chunk_indices` | 可选输入 | INT64 | `[Nchunk,2]` |
| `w` | 输出 | 与 `q` 相同 | `[B,HV,T,K]` |
| `u` | 输出 | 与 `q` 相同 | `[B,HV,T,V]` |
| `dv_local` | 输出 | 与 `q` 相同 | `[B,HV,T,V]` |

`q/k/v/A/d_o` 使用同一主 dtype；`g` 与 `beta` 可以分别选择 BF16 或 FP32，均不支持
FP16。当前版本固定 `K=V=128`、`chunk_size=64`，要求 `HV % HK == 0` 且
`G=HV/HK` 属于 `{1,2,3,4}`。

## 支持范围

```text
SoC                 A5 / Ascend 950
layout              BNSD [B,H,T,D]
B/HK/HV/T           正整数
K/V                 固定为 128/128
chunk_size          固定为 64
G=HV/HK             1、2、3 或 4
```

定长场景支持任意正 `B`；变长场景使用 `B=1` 的输入容器，由 `cu_seqlens` 描述多个
sequence 的边界。所有输出保持输入的 BNSD 逻辑排布。

## 属性

| 属性 | 类型 | 默认值 | 约束 |
| --- | --- | --- | --- |
| `scale` | float | 无，必传 | 直接参与 `dv_local` 支路计算 |
| `chunk_size` | int | 无，必传 | 当前版本必须为 64 |
| `use_exp2` | bool | `true` | `true` 使用 `exp2`，`false` 使用 `exp` |
| `stage` | int | `2` | 开发期精度门禁：只允许 `0/1/2` |

`stage=2` 是正常算子语义。`stage=0/1` 仅用于开发期逐 Stage 精度检查，输出仍保持正式
BNSD shape，但只读取前 `chunk_size` 列：

```text
stage=0: w[..., :chunk_size]        = 对每个 hv 复用其 hk Score；忽略 u、dv_local
stage=1: w[..., :chunk_size]        = A_bg
         u[..., :chunk_size]        = A_beta
         dv_local[..., :chunk_size] = D
stage=2: w、u、dv_local             = 正式最终输出
```

调试模式不改变输入、输出数量和顺序；输出中未在上表定义的区域不参与比较。

## 定长与变长

- 定长场景：`cu_seqlens=None` 且 `chunk_indices=None`，每个 batch 独立按
  `ceil(T/chunk_size)` 个 chunk 处理。
- 变长场景：`cu_seqlens` 和 `chunk_indices` 必须同时提供；输入容器 `B=1`。
  `cu_seqlens` 从 0 开始、以物理 token 总数结束并单调不减；`chunk_indices` 每行是
  `(sequence_index, local_chunk_index)`，且不能超出对应 sequence 的 chunk 范围。
- 尾 chunk 只计算有效 token；输出仍按原输入的 `T` 维写回。

对变长 pair `(seq, lc)`：

```text
base = cu_seqlens[seq] + lc * chunk_size
M    = min(chunk_size, cu_seqlens[seq+1] - base)
```

物理 token 范围为 `[0,cu_seqlens[-1])`。`M=0` 不生成有效计算；尾 chunk 只使用前
`M` 行/列，输出按原 tensor stride 写回。

输入的 dtype、shape、布局、属性或变长元数据不满足上述契约时，接口在执行 kernel 前
返回参数错误。
