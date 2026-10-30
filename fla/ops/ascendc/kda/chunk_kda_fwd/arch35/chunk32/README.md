# chunk_kda_fwd_fused — KDA forward chunk32 大融合(issue #829)

面向 Ascend950 的单 kernel KDA 正向实现：AIC 裸指令 + AIV MicroAPI，一次 launch 完成
prepare、state forward 和 finalize。公开调用统一走 `aclnnChunkKdaFwdV2`，由顶层
`chunk_kda_fwd` 在 `chunk_size=32` 时经 `RunChunkKdaFwd32Fused` 路由到本目录的
`ChunkKdaFwdFused`，不经过 chunk64 三算子组合或外层 `ChunkKdaFwd` L0。

fused tiling 在 `op_host/chunk_kda_fwd_fused_tiling.cpp` 中实际执行 `SetTilingKey(0)`；
外层 `ChunkKdaFwd` 仍按模板使用 key 1/2。`chunk_size=32` 中的 32 不是 TilingKey。

## 支持范围

- 仅支持 Ascend950；q/k/v/o 为 BF16，`K=V=128`。
- 公开 V2 的 g/beta 各自可为 BF16 或 FP32；任一为 FP32 时 host 将两者统一上转 FP32，
  进入 L0 后使用相同的 BF16 或 FP32 组合。
- `A_log`、`dt_bias`、initial/final state 为 FP32；`T>0`，kernel 处理非 128 对齐尾块。
- 支持 GVA：`HV` 为 `H` 的正整数倍，`H==HV` 是特例。
- 支持可选 `initial_state` 和 `output_final_state`；公开 Python 的
  `state_v_first=true` 由包装层在调用前后转置，fused L2/L0 只接收 K-first state。
- 公开 chunk32 路径要求 `safe_gate=true`、`use_gate_in_kernel=true` 并提供 `A_log`；
  Python 的 `dt_bias=None` 会补成 FP32 零张量。gate 语义为
  `lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`，其中 `exp(A_log)` 由 L0 host 的
  `l0op::Exp` 计算。
- q/k 核内 L2 normalization，beta 核内 sigmoid；要求
  `use_qk_l2norm_in_kernel=true`、`use_beta_sigmoid_in_kernel=true`、
  `allow_neg_eigval=false`、`use_exp2=true`，`epsilon` 必须为正有限值。
- `chunk_indices` 仅接受与 `cu_seqlens` 对应的 canonical sequence-major 顺序。

### layout 与 varlen

- dense 支持 BSND/BNSD。BSND batch 由 host 合成整齐 `cu_seqlens` 后走 packed 直读；
  BNSD 拍平 batch/head 后保持头连续。
- varlen 支持 BSND（要求 `B=1`）、TND、NTD，不支持 BNSD；`cu_seqlens` 非递减、首项为 0、
  末项为总 token 数，最多 1024 个序列，允许空序列。
- NTD varlen 原生按 `[H,T_total,K]` / `[HV,T_total,*]` 寻址，不先物化为完整 TND；
  beta 在 kernel 中统一按 `[HV,T_total]` 读取。
- 公开 `attn_out` 始终为 sequence-major：rank-4 是 BSND，rank-3 是 TND；NTD 输入也不例外。
- 空序列不进入计算流水。请求 final state 时，有 initial state 由 host 预填
  `final=initial`，无 initial state 由 kernel 写零。

直接 aclnn 融合入口 `aclnnChunkKdaFwdFusedGetWorkspaceSize` 使用低层视图，不承担上述全部
公开适配：dense q/k 为 `[B*H,T,K]`，v/g 为 `[B*HV,T,K/V]`，beta 为 `[B*HV,T]`；
varlen 使用 token-major TND，state 拍平为 `[N*HV,K,V]`。该入口要求
`aLog/dtBias/o` 非空、`finalState` 可空，epsilon 固定为 `1e-6`。公开 layout、NTD varlen、
自定义 epsilon 和 `state_v_first` 适配应使用 V2。

## 输出与已知限制

chunk32 融合路径只产出 `attn_out` 和可选 `final_state`：

- `disable_recompute=false`、`return_intermediate_states=false`，`Aqk`/`Akk` 返回槽为
  `None`，`gk/w/u/qg/kg/v_new/h` 也为 `None`。
- 不导出 `q_hat/k_hat/q_rstd/k_rstd/beta_eff`；向 V2 传任一非空反向保存值输出会返回
  `ACLNN_ERR_PARAM_INVALID`。
- L2 的 `gkOut/wOut/uOut/qgOut/kgOut/vNewOut/hOut` 必须为空；fused 不写 `Aqk/Akk`，
  Python 对这两个 L2 输出同样传空。
- 不支持 FP16 q/k/v、混合 K/V 档、BNSD varlen 或超出 1024 的 varlen 序列；chunk32 没有
  不满足 fused 契约时自动回落到 chunk64 实现的路径。
- T=1 不记为“全部通过”的专项验收结论；非 128 对齐尾块的通用支持不替代该项通过记录。

## 验证

- 原始同精度 benchmark 的 A5 官方 ATK `accuracy` 扩展套件：40/40 通过。
- 全 FP32 q/k 准备中间计算版本在原始同精度 benchmark 下：40/40 通过。40 条 suite 覆盖 dense/varlen、GVA、尾块、
  initial/final state、不同 layout、空序列及 BF16/FP32 g/beta；id 34–35 覆盖
  FP32 g/beta + GVA + final_state，id 36–39 覆盖 TND/NTD varlen。
- 正常数据 T=128/1024 额外回归：4/4 通过；不将此记录扩展为 T=1 全部通过。
- 元数据还冻结 performance 2 条、mss 5 条；这些数量不计入 accuracy 40/40。
- Stable/ctypes ABI 测试：22 passed、122 subtests passed、1 skipped。
