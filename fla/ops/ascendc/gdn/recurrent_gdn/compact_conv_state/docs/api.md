# CompactConvState 接口

CANN 根据 OpDef 生成内部两阶段接口 `aclnnInnerCompactConvStateGetWorkspaceSize` 和 `aclnnInnerCompactConvState`。Python 适配器调用这两个入口。

公开 Python 入口为 `compact_conv_state`，AscendC 算子名为
`CompactConvState`。

| 参数 | Shape | dtype | 说明 |
| --- | --- | --- | --- |
| `conv_state_ref` | `[P,C,S]` | BF16 / FP32 | 用于推导通道数、槽数和元素类型 |
| `conv_state_base_addrs` | `[Layer]` | INT64 | 每层物理 cache 基地址 |
| `conv_state_block_strides` | `[Layer]` | INT64 | 每层 page stride |
| `conv_state_dim_strides` | `[Layer]` | INT64 | 每层 channel stride |
| `conv_state_token_strides` | `[Layer]` | INT64 | 每层 history-slot stride |
| `state_indices` | `[B]` | INT32/INT64 | source page |
| `commit_lens` | `[B]` | INT32/INT64 | final accepted length |
| `final_state_indices` | `[B]` | INT32/INT64 | final destination page |
| `boundary_state_indices` | `[B]` | INT32/INT64 | boundary destination page |
| `boundary_recovery_lens` | `[B]` | INT32/INT64 | boundary accepted length |

属性：

- `conv_history_len`: 需要保留的卷积历史槽数，`1..32`。
- `align_mode`: 是否写 boundary page。
- `dim_stride`: 各层共享 channel stride；`None` 时从 table 读取。
- `null_block_id`: page 小于等于该值表示无效。

final 拷贝使用 source `[c-1:c-1+W)` 到 final `[0:W)`；boundary 使用
`boundary_recovery_len` 计算源起点。boundary 先写，final 后写，因此同页时
final 结果获胜。
