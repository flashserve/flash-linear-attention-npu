# Gated Delta Network（GDN）模型符号表

本文是 `fla/ops/ascendc/gdn/` 下所有算子的模型级符号权威来源。算子 README、设计文档、API 文档和 `tests/op_cases/*.json` 必须使用本文定义的名称与语义，不得为同一维度另起别名。

- 符号表版本：`gdn-shape-v1`
- 适用范围：chunk Gated Delta Rule 前向、反向、预处理、递归推理和辅助算子
- Shape 约定：表格中的 Shape 只写符号；固定支持值由各算子 README 的“已知限制”说明

<a id="model-shape-symbols"></a>

## 1. 核心张量符号

| 符号 | 语义 | 典型张量维度 |
| --- | --- | --- |
| `B` | Batch size；变长序列打包场景通常为 1 | 定长张量 batch 维 |
| `N` | 变长序列的逻辑序列数 | `cu_seqlens` 的长度为 `N+1` |
| `T` | 定长序列长度或变长序列打包后的 token 总数 | Q/K/V 的 token 维 |
| `H_k` | Query/Key head 数 | `q`、`k` 的 head 维 |
| `H_v` | Value/Output/State head 数 | `v`、`o`、`g` 的 head 维 |
| `R_h` | Value head 与 Q/K head 的分组比 H_v/H_k | GVA head 映射 |
| `K` | Query/Key 单 head 特征维 | `q`、`k` 的最后一维 |
| `V` | Value/Output 单 head 特征维 | `v`、`o` 的最后一维 |
| `chunk_size` | 每个 chunk 的 token 数，也是三角块宽度 | chunk 维或局部矩阵最后一维 |
| `N_c` | 当前调用中的 chunk 总数 | `chunk_indices` 的 pair 数、状态 chunk 维 |
| `N_{c,b}` | dense 场景每个 batch 的 chunk 数，`ceil(T/chunk_size)` | dense 状态 chunk 维 |
| `S_n` | 第 n 条变长序列的有效长度 | `cu_seqlens[n+1]-cu_seqlens[n]` |

`H_v` 必须能按组映射到 `H_k` 时，统一写作 `H_v % H_k == 0`，映射关系为 `h_k=floor(h_v/R_h)`。不得使用 `H`、`Hq`、`Hk`、`Hv`、`query_head` 或 `value_head` 替代模型级符号。

## 2. 辅助符号

| 符号 | 语义 | 典型用途 |
| --- | --- | --- |
| `D` | 不区分 Q/K 与 V 时使用的通道维 | causal convolution 输入通道 |
| `W` | 一维卷积核宽度 | causal convolution `weight` 的卷积维 |
| `L_s` | convolution state 保存的历史长度 | decode/update 状态维 |
| `M` | 三角矩阵有效阶数 | `solve_tri` |
| `P` | token 后连续尾部元素乘积 | `chunk_local_cumsum` |
| `B_T` | 单个变长序列处理块覆盖的 token 数，由 tiling 根据 UB、chunk_size、P 计算 | `chunk_local_cumsum` 变长序列分块 |
| `N_b` | 变长序列内部处理块总数；由各序列 ceil(seq_len/B_T) 求和 | `chunk_local_cumsum` 的 `chunk_indices_out` pair 数 |
| `D_s` | 状态槽位数 | recurrent decode state 第一维 |
| `Q_a` | 单次调用实际接受的 token 数 | speculative decode |

辅助符号只能在对应算子使用，不能覆盖核心符号的语义。

## 3. 统一布局

| Layout | Shape | 说明 |
| --- | --- | --- |
| `BNSD` | `[B,H,T,D]` | head-first；GDN chunk 算子的主要内部布局 |
| `BSND` | `[B,T,H,D]` | sequence-first 四维布局 |
| `TND` | `[T,H,D]` | 变长序列 token-first 打包布局 |
| `NTD` | `[H,T,D]` | 变长序列 head-first 打包布局 |
| `BSH` | `[B,T,D]` | 合并 head 的 dense 布局 |

当 `D` 在具体算子中表示 Q/K 特征时应改写为 `K`，表示 Value/Output 特征时应改写为 `V`。API 文档必须同时写明逻辑 Shape 与实际 layout。

## 4. 变长序列元数据

| 名称 | Shape | Dtype | 语义 |
| --- | --- | --- | --- |
| `cu_seqlens` | `[N+1]` | INT64 | 严格非递减的累计长度，首元素为 0，末元素为 `T` |
| `chunk_indices` | `[N_c,2]` 或展平后的 `[2*N_c]` | INT64 | 按 sequence-major 顺序保存 `(seq_id, local_chunk_id)` |
| `chunk_offsets` | `[N_c]` 或算子定义的等价形式 | INT64 | chunk 到 token 起点的映射；具体格式由算子 README 定义 |

若算子要求两个元数据同时存在，README、op_host 拦截和 JSON 负向用例必须使用相同组合规则。

## 5. 文档使用约定

1. 每个算子 README 末尾保留“Shape 变量说明”附录，引用 `gdn-shape-v1` 并只列本算子实际使用的符号。
2. `docs/design.md` 和 `docs/api.md` 不复制本表，只链接算子 README 的附录。
3. 固定维度，例如 `K`、`V` 或 `chunk_size` 的离散支持值，只能写在算子“已知限制”中。
4. JSON 的 `shape` 字段必须使用本表名称；卷积等辅助算子可使用本文定义的 `D/W/L_s`。
5. 修改符号名称或语义时，必须提高符号表版本并同步所有受影响文档和 JSON。
