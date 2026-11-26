# CompactConvState 设计

任务域为 `B * Layer * ceil(C / 256)`。每个 AIV task 处理一个 request、
一个 layer 和一个 channel tile。

Kernel 使用两个任务槽，每槽包含 final 和 boundary 两块 UB，共四块。
final 和 boundary 都在任何写回之前完成搬入，因此 source page 同时作为
boundary destination 时不会污染 final 输入。

支持两种物理布局：

- `dim_stride == 1`: channel 连续，按 history 行搬移。
- `token_stride == 1`: history-slot 连续，按 channel 行搬移。

host 校验 shape、dtype、metadata 连续性和 UB 预算。kernel 只处理合法输入。
MTE2 和 MTE3 通过每个任务槽的 mutex 建立搬入/搬出生命周期。
