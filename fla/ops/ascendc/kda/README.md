# KDA 模型符号表

本文是 `fla/ops/ascendc/kda/` 下 KDA 算子的模型级符号权威来源。`chunk_kda_fwd`、`kda_gate_cumsum`、`kda_layout_swap12` 的 README、设计文档、API 文档和 JSON 用例必须引用本表。

- 符号表版本：`kda-shape-v1`
- 适用范围：KDA chunk 前向、gate 预处理和 layout 转换辅助算子
- 内部主布局：dense 使用 `BNSD`，varlen 使用 `NTD`

<a id="model-shape-symbols"></a>

## 1. 核心符号

| 符号 | 语义 | 典型张量维度 |
| --- | --- | --- |
| `B` | Batch size | dense 输入第 0 维 |
| `N` | varlen 逻辑序列数 | `cu_seqlens` 长度为 `N+1` |
| `T` | dense 序列长度或 varlen 打包 token 总数 | token 维 |
| `H_k` | Query/Key head 数 | `q`、`k` 的 head 维 |
| `H_v` | Value/Output head 数 | `v`、`o`、`beta` 的 head 维 |
| `R_h` | H_v/H_k 的 head 分组比 | GVA head 映射 |
| `K` | Query/Key 单 head 特征维 | `q`、`k`、`gk` 的最后一维 |
| `V` | Value 单 head 特征维 | `v`、`o` 的最后一维 |
| `C` | chunk_size | chunk 计算粒度和三角块宽度 |
| `N_c` | 当前调用中的 chunk 总数 | 中间状态 `h` 的 chunk 维 |
| `S_n` | 第 n 条 varlen 序列的有效长度 | `cu_seqlens[n+1]-cu_seqlens[n]` |

KDA 的 head 关系统一写为 `H_v % H_k == 0`，`h_k=floor(h_v/R_h)`。不得混用 `H`、`HV`、`qHeadNum`、`vHeadNum` 等实现字段作为公开 Shape 符号。

## 2. 状态与中间量符号

| 符号 | 语义 | 典型 Shape |
| --- | --- | --- |
| `S_0` | 可选初始状态 | `[N,H_v,K,V]` 或 dense `[B,H_v,K,V]` |
| `S_f` | 最终状态 | 与 `S_0` 相同 |
| `A_{qk}` | chunk 内 QK 严格因果权重 | token 维后附加 `C` |
| `A_{kk}` | chunk 内 KK/WY 权重 | token 维后附加 `C` |
| `W_k` | WY key-side 中间量 | 与 `q` 的特征维一致并按 `H_v` 展开 |
| `U_v` | WY value-side 中间量 | 与 `v` 同 Shape |
| `G_k` | 逐 token、逐 key 维的累计 gate | 与 `gk` 同 Shape |
| `D_0` | 通用 ND 输入的第 0 维 | `kda_layout_swap12` |
| `D_1` | 通用 ND 输入的第 1 维 | `kda_layout_swap12` |
| `D_2` | 通用 ND 输入的第 2 维 | `kda_layout_swap12` |
| `D_3` | 通用 ND 输入的第 3 维（rank>=4） | `kda_layout_swap12` |
| `D_4` | 通用 ND 输入的第 4 维（rank>=5） | `kda_layout_swap12` |

中间量名称用于公式，不作为 JSON Shape 字段。除 `kda_layout_swap12` 使用 `D_i` 表达通用 ND 维度外，JSON 使用 `B/H_k/H_v/T/K/V/C/N/N_c`。

## 3. 布局映射

| Layout | Q/K Shape | V/O Shape | Gate/Beta Shape | 说明 |
| --- | --- | --- | --- | --- |
| `BSND` | `[B,T,H_k,K]` | `[B,T,H_v,V]` | `gk=[B,T,H_v,K]`、`beta=[B,T,H_v]` | 对外兼容布局，op_api 转换到内部布局 |
| `BNSD` | `[B,H_k,T,K]` | `[B,H_v,T,V]` | `gk=[B,H_v,T,K]`、`beta=[B,H_v,T]` | dense 性能布局 |
| `TND` | `[T,H_k,K]` | `[T,H_v,V]` | `gk=[T,H_v,K]`、`beta=[T,H_v]` | 单 head 或兼容 varlen 布局 |
| `NTD` | `[H_k,T,K]` | `[H_v,T,V]` | `gk=[H_v,T,K]`、`beta=[H_v,T]` | multi-head varlen 性能布局 |

`kda_layout_swap12` 只交换 layout 中第 1、2 个逻辑维度，不改变元素值、batch 维和尾部连续特征维。

## 4. varlen 元数据

| 名称 | Shape | Dtype | 语义 |
| --- | --- | --- | --- |
| `cu_seqlens` | `[N+1]` | INT64 | 从 0 开始、以 `T` 结束的累计长度 |
| `chunk_indices` | `[N_c,2]` 或 `[2*N_c]` | INT64 | sequence-major 的 `(seq_id, local_chunk_id)` 列表 |

KDA 当前要求 `chunk_indices` 为 canonical sequence-major 顺序。调用方省略该输入时，Python wrapper 可根据 `cu_seqlens` 和 `C` 生成等价列表。

## 5. 文档使用约定

1. 每个 KDA 算子 README 的附录引用 `kda-shape-v1`，只列实际使用符号。
2. 设计和 API 文档通过算子 README 锚点引用符号，不重复本表。
3. `stage`、`usedCoreNum`、`seqStart` 等 tiling 实现字段不得进入公开 Shape 定义。
4. `K`、`V`、`C`、head 数和序列数的固定上限统一写入算子“已知限制”。
5. 修改布局语义或符号后，必须同步三个 KDA 算子及 `tests/reference/chunk_kda_reference.py`。
