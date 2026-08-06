# KDA 模型符号表

本文是 `fla/ops/ascendc/kda/` 下 KDA 算子的 Shape 与布局语义来源。

- 符号表版本：`kda-shape-v1`
- 内部计算布局：dense 使用 BNSD，rank-3 使用 NTD
- `layout` 只描述输入；输出布局由接口固定约定

<a id="model-shape-symbols"></a>

## 核心符号

| 符号 | 语义 |
| --- | --- |
| `B` | dense batch size |
| `N` | 变长序列数，`cu_seqlens` 长度为 `N+1` |
| `T` | 序列长度或打包 token 总数 |
| `H_k` | Query/Key head 数 |
| `H_v` | Value/Output head 数 |
| `K` | Query/Key head dim |
| `V` | Value head dim |
| `chunk_size` | chunk 长度 |
| `N_c` | 当前调用的 chunk 总数 |

Head 映射必须满足 `0 < H_k <= H_v <= 128` 且 `H_v % H_k == 0`。

## 输入布局

| Layout | Q/K | V/G | Beta |
| --- | --- | --- | --- |
| `BSND` | `[B,T,H_k,K]` | `[B,T,H_v,V/K]` | `[B,T,H_v]` |
| `BNSD` | `[B,H_k,T,K]` | `[B,H_v,T,V/K]` | `[B,H_v,T]` |
| `TND` | `[T,H_k,K]` | `[T,H_v,V/K]` | `[T,H_v]` |
| `NTD` | `[H_k,T,K]` | `[H_v,T,V/K]` | `[H_v,T]` |

BSND/TND 输入在 L2 接口中通过 `l0op::Transpose` 转为内部 head-major 布局。仓内不再维护独立 layout-swap 算子。

## 固定输出布局

| 输出 | rank-4 | rank-3 | 说明 |
| --- | --- | --- | --- |
| `attn_out` | `[B,T,H_v,V]` | `[T,H_v,V]` | 固定 sequence-major |
| `final_state` | `[N,H_v,K,V]` 或 `[N,H_v,V,K]` | 同左 | 固定 sequence-major；末两维由 `state_v_first` 控制 |
| `gk/w/qg/kg` | `[B,H_v,T,K]` | `[H_v,T,K]` | 供反向使用，固定 head-major |
| `u/v_new` | `[B,H_v,T,V]` | `[H_v,T,V]` | 供反向使用，固定 head-major |
| `Aqk/Akk` | `[B,H_v,T,chunk_size]` | `[H_v,T,chunk_size]` | 供反向使用，固定 head-major |
| `h` | `[B,H_v,N_c,K,V]` 或 `[B,H_v,N_c,V,K]` | 去掉 B 维 | 供反向使用，固定 head-major |

## 变长元数据

`cu_seqlens` 是从 0 开始、以 `T` 结束的非递减 INT64 数组。`chunk_indices` 是 canonical sequence-major 的
`(seq_id, local_chunk_id)` 列表；省略时由 L2 接口一次性生成。`N_c` 在接口内部根据序列长度推导，不是公共入参。
