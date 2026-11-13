# ChunkKdaBwdRecompute 接口

公开 Python 入口为 `fla_npu.ops.ascendc.npu_chunk_kda_bwd_recompute`。

```python
npu_chunk_kda_bwd_recompute(
    q, k, v, g, beta, a, chunk_size, *,
    A_log=None, dt_bias=None, cu_seqlens=None, chunk_indices=None,
    use_gate_in_kernel=True, use_exp2=True, lower_bound=-5.0,
    input_layout="BNSD",
)
```

`chunk_size` 是必传位置参数，唯一支持值为 64。
返回 `(gk, w, u, qg, kg)`；`use_gate_in_kernel=False` 时 `gk=None`。

| 输入 | shape | dtype |
| --- | --- | --- |
| q、k | `[B,HK,T,128]` | BF16 |
| v | `[B,HV,T,128]` | BF16 |
| g | `[B,HV,T,128]` | BF16 / FP32 |
| beta | `[B,HV,T]` | BF16 / FP32 |
| a | `[B,HV,T,64]` | BF16 |
| A_log | `[HV]` | FP32 |
| dt_bias | `[HV,128]`，可选 | FP32 |

`HK>0`、`HV>0`、`HV%HK=0`，q/k 的 source head 为 `hk=hv/(HV/HK)`。
所有输入按 dense BNSD 寻址；`input_layout="NTD"` 是保留取值，仍使用同一 rank-4 存储契约，
不表示 rank-3 tensor。输出均为 `[B,HV,T,128]`，gk 为 FP32，其余 BF16。
`use_gate_in_kernel=True` 的公开调用应提供 A_log；关闭 gate 时忽略 bias/A_log。

定长输入逐 batch 按 64 token 分 chunk；A6 变长输入使用 `B=1` 的 packed token 轴，
`cu_seqlens` 为从 0 到 T 的非递减 int64 序列。`chunk_indices` 为展平的
`[seq_idx, chunk_idx, ...]` int64 序列，列出所有非空 chunk 各一次，允许调整顺序；
`chunk_idx` 从各 sequence 的起点计数。两项必须同时提供。空 sequence 不生成 work。
Python 接口接受可转换为 host int 序列的元数据；L2 把它们作为 aclIntArray 传入。

A6 Host Tiling 沿用公共 `CheckSpec`，不额外校验各输入 shape、变长元数据内容或平台 UB/L1 容量；
调用方按上述约定提供合法输入。

`use_gate_in_kernel=True` 时，g 先经 safe-gate，再按 chunk 做 cumsum；关闭时对原始 g 做 cumsum。
`use_exp2=True` 将 cumsum 换成以 2 为底的单位，并用 exp2 生成 qg/kg 等结果；否则使用自然指数。
完整数学及浮点计算顺序见 [design.md](design.md)。

已有实现注册 A2/A3/A5；A6 `ascend960dt` 使用独立 arch92 源码路径。
A6 的 K/V 固定为 128，用户 workspace 为 0，另保留平台要求的系统 workspace。

底层 C 接口见 [aclnn_chunk_kda_bwd_recompute.h](../op_host/op_api/aclnn_chunk_kda_bwd_recompute.h)：
先调用 `aclnnChunkKdaBwdRecomputeGetWorkspaceSize`，再调用 `aclnnChunkKdaBwdRecompute`。
L2 的输出参数顺序为 `w/u/qg/kg/gk`，适配层负责返回 Python 顺序；L2 通过 gk 输出是否为空推导 gate 开关，
gk 不对外输出时仍分配内部 FP32 结果。参数或 shape 校验失败返回现有 ACLNN 参数错误或 tiling 失败。

```python
from fla_npu.ops.ascendc import npu_chunk_kda_bwd_recompute

gk, w, u, qg, kg = npu_chunk_kda_bwd_recompute(
    q, k, v, g, beta, a, 64, A_log=a_log, dt_bias=dt_bias,
    use_gate_in_kernel=True, use_exp2=True,
)
```
