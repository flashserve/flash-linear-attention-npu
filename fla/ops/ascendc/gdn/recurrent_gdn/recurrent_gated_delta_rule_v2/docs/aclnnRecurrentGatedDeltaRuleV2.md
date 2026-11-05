# aclnnRecurrentGatedDeltaRuleV2A5

`aclnnRecurrentGatedDeltaRuleV2A5GetWorkspaceSize` and
`aclnnRecurrentGatedDeltaRuleV2A5` are the A5-only V2 ABI for the reviewed
Recurrent GDN contract. The legacy `aclnnRecurrentGatedDeltaRule*` symbols are
unchanged.

V2 adds `A_log`, `dt_bias`, the Q/K normalization, raw-gate, beta-sigmoid and
negative-eigenvalue switches, and accepts `ssm_state_indices` as either
`[T]` or `[B, W]`. `gk` is rejected. Q/K/V and State views are passed to the
kernel with their element strides; the V2 host path does not insert a
contiguous conversion.

Q/K/V stride 以 element 为单位，三个维度依次为 token、head、feature。
stride 必须为正，head/token span 不得重叠；feature stride 可以大于 1，
但对应的 A5 MTE2 source gap 必须可由 32-bit byte gap 表示。State 的最后两维
必须为 `(K, 1)` 稠密布局，slot/head 外层可以采用 block-major 或 head-major，
包括 transpose view 和带 padding/storage offset 的不重叠视图。beta、scalar g
和输出仍要求稠密布局。

`actual_seq_lengths` uses the reviewed packed form `[P, L_0, ..., L_{B-1}]`:
the first value is the invalid prefix length and the remaining values are
per-request lengths. Empty requests are valid. The V2 tiling record marks this
length encoding explicitly.

The supported A5 shape is packed Q/K/V with `K=V=128`, `Nv % Nk == 0`, BF16
Q/K/V/out, BF16 or FP32 State, and INT32 sequence/index metadata. `beta` and
scalar `g` may be BF16 or FP32. When `use_gate_in_kernel` is true, `g` and
`A_log` are required; `dt_bias` is optional. The kernel TilingKey is the
six-dimensional `(State dtype, g dtype, beta dtype, Q/K L2Norm, raw gate,
beta sigmoid)` tuple and registers all 64 dtype/function combinations. If `g`
is absent, its key component uses BF16; no Gk flag participates in TilingKey
selection.
