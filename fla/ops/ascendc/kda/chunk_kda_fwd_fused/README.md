# chunk_kda_fwd_fused — KDA forward chunk32 大融合(issue #829)

面向 Ascend950 的单 kernel KDA 正向实现：AIC 裸指令 + AIV MicroAPI，一次 launch 完成
prepare、state forward 和 finalize。公开调用统一走 `aclnnChunkKdaFwdV2`，由顶层
`chunk_kda_fwd` 在 `chunk_size=32` 时路由到本目录的融合实现。

## 当前契约

- q/k/v/o 为 BF16，`K=V=128`；g/beta 必须同为 BF16 或同为 FP32。
- `A_log`、`dt_bias`、initial/final state 为 FP32。
- `T>0`，kernel 处理非 128 对齐尾块；支持 dense batch 和 varlen。
- 支持 GVA：`HV` 为 `H` 的正整数倍，`H==HV` 是特例。
- 支持可选 `initial_state` 和 `output_final_state`；`state_v_first=true` 由 host 转置适配。
- 公开 chunk32 路径要求 `safe_gate=true`、`use_gate_in_kernel=true` 并提供 `A_log`；
  `dt_bias=None` 等价于零偏置。gate 语义为
  `lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`。
- q/k 核内 L2 normalization，beta 核内 sigmoid；要求
  `use_qk_l2norm_in_kernel=true`、`use_beta_sigmoid_in_kernel=true`、
  `allow_neg_eigval=false`、`use_exp2=true`。
- `chunk_indices` 仅接受与 `cu_seqlens` 对应的 canonical sequence-major 顺序。

直接 aclnn 融合入口使用头连续视图：q/k 为 `[H,T,K]`，v/g 为 `[HV,T,K/V]`，beta 为
`[HV,T]`；dense state 为 `[HV,K,V]`，varlen state 为 `[nSeq*HV,K,V]`。公开 V2 入口负责
BSND/BNSD/NTD/TND 布局适配及 dense/varlen 参数转换。

## 不导出的内容

chunk32 融合路径不导出 recompute 中间态或反向保存值：
`disable_recompute=false`、`return_intermediate_states=false`，`Aqk`/`Akk` 返回槽为 `None`，
不导出 `q_hat/k_hat/q_rstd/k_rstd/beta_eff`。这不影响可选 `final_state` 输出。

## 验证

- A5 官方 ATK `accuracy` 扩展套件：34/34 通过，覆盖 dense/varlen、GVA、尾块、
  initial/final state、不同 layout 及 BF16/FP32 g/beta。
- Stable/ctypes ABI 测试：22 passed、122 subtests passed、1 skipped。
