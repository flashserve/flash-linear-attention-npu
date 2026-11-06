# chunk_kda_fwd_fused

Ascend950 专用的 KDA 前向 chunk32 单 kernel 大融合算子（L0）。由
`aclnnChunkKdaFwdV2` 在 `chunk_size=32` 场景下路由调用（`l0op::KdaChunkForwardFused`），
不对最终用户直接暴露 aclnn 入口。

## 能力契约

- 平台：仅 Ascend950（DAV_3510）；其它 SoC 无实现，路由侧显式拒绝。
- dtype：q/k/v、attn_out 为 BF16；g/beta 为 FP32 或 BF16（两者同型，由路由侧统一）；
  a_log/dt_bias/initial_state/final_state 为 FP32。
- 形状：q/k 头连续 [H,T,K]，v [HV,T,V]，g [HV,T,K]，beta [HV,T]；K=V=128；
  HV % H == 0；a_log [HV_real] 且 HV % HV_real == 0；T > 0 任意（尾块核内钳制）。
- 布局：dense 由路由侧统一转置到头连续后传入；varlen 支持 token-major（TND）
  与 head-major（NTD），经 cu_seqlens（int64，host ValueDepend 进 tiling）驱动，
  序列数 1..1024，允许空序列。
- 语义固定：chunk_size=32、核内 q/k L2 归一（epsilon 固定 1e-6，由 tiling 写入）、
  核内 beta sigmoid、safe_gate（lower_bound 经 sigmoid 下界）、exp2 gate。
- 状态：可选 initial_state / final_state（FP32，[任务头数,128,128] 拍平视图，
  varlen 时任务头 = 序列数 × HV）。

## 输出

| 输出 | dtype | 说明 |
| --- | --- | --- |
| attn_out | BF16 | 头连续 [HV,T,V]（路由侧负责转置回公开布局） |
| final_state | FP32，可选 | 缺省时按"未请求"处理，kernel 不写 |

不提供 gk/Aqk/Akk/w/u/qg/kg/v_new/h 等中间量导出；V2 的反向保存值出口
（q_hat/k_hat/q_rstd/k_rstd/beta_eff）在本算子无意义，路由侧显式拒绝。

## 源码位置

- `op_host/chunk_kda_fwd_fused_def.cpp`：窄接口原型（默认 TilingKey）。
- `op_host/chunk_kda_fwd_fused_tiling.cpp`：契约校验与 TilingData 填充。
- `op_host/op_api/chunk_kda_fwd_fused.cpp`：`l0op::KdaChunkForwardFused`
  （Exp(a_log) 预处理 + cu_seqlens 转 tensor + 发射）。
- `op_kernel/chunk_kda_fwd_fused.cpp`：薄入口；实现位于
  `op_kernel/arch35/chunk_kda_fwd_fused/`（AIC/AIV 双角色单 kernel）。

调用链：`aclnnChunkKdaFwdV2`（chunk_size=32 分支，见
`fla/ops/ascendc/kda/chunk_kda_fwd/op_host/op_api/aclnn_chunk_kda_fwd.cpp`）
→ `l0op::KdaChunkForwardFused` → `ChunkKdaFwdFused`。
