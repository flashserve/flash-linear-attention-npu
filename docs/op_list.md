# 算子概览

Ascend C 使用 `fla_npu.ops.ascendc`，同时提供带 `npu_` 前缀和不带前缀的入口。各算子的参数和支持范围见名称链接，安装见根 [README](../README.md)，旧调用方式迁移见 [兼容与迁移指南](兼容与迁移指南.md)。

<a id="atk-op-options"></a>

## Ascend C 算子

“算子工程名”对应 `fla/ops/ascendc` 下的源码工程目录名，可通过 `build.sh --ops=<工程名>` 或 `FLA_NPU_OPS=<工程名>` 筛选构建；存在同名 ATK 测试工程时，也可通过 `tests/atk/run_test_cpu.sh -op=<工程名>` 指定测试，详见 [ATK 说明](../tests/atk/README.md)。

| 类别 | 算子 | 算子工程名 | 基本用途 | 备注 |
| --- | --- | --- | --- | --- |
| GDN | [`npu_chunk_gated_delta_rule_fwd`](../fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd/README.md) | `chunk_gated_delta_rule_fwd` | 融合前向 | - |
| GDN | [`npu_chunk_gated_delta_rule_bwd`](../fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd/README.md) | `chunk_gated_delta_rule_bwd` | 融合反向 | - |
| GDN | [`npu_chunk_gated_delta_rule_fwd_prepare`](../fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_prepare/README.md) | `chunk_gated_delta_rule_fwd_prepare` | 前向预处理 | - |
| GDN | [`npu_chunk_gated_delta_rule_fwd_h`](../fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/README.md) | `chunk_gated_delta_rule_fwd_h` | 前向状态计算 | - |
| GDN | [`npu_chunk_gated_delta_rule_bwd_dhu`](../fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/README.md) | `chunk_gated_delta_rule_bwd_dhu` | 反向状态与中间梯度计算 | - |
| GDN | `npu_chunk_gated_delta_rule_bwd_finalize` | `chunk_gated_delta_rule_bwd_finalize` | 反向梯度收尾 | 暂缺README |
| GDN | [`npu_chunk_gdn_bwd_intra`](../fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gdn_bwd_intra/README.md) | `chunk_gdn_bwd_intra` | chunk 内反向计算 | - |
| GDN | [`npu_chunk_fwd_h`](../fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_h/README.md) | `chunk_fwd_h` | 独立 chunk 状态前向 | - |
| GDN | [`npu_chunk_fwd_o`](../fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/README.md) | `chunk_fwd_o` | chunk 输出计算 | - |
| GDN | [`npu_solve_tri`](../fla/ops/ascendc/gdn/chunk_gdn_fwd/solve_tri/README.md) | `solve_tri` | 三角矩阵求解 | - |
| GDN | [`npu_recurrent_gated_delta_rule`](../fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule/README.md) | `recurrent_gated_delta_rule` | 递推推理 | - |
| GDN | [`npu_chunk_bwd_dqkwg`](../fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dqkwg/README.md) | `chunk_bwd_dqkwg` | chunk 反向梯度计算 | 已弃用（日落）；2027年1月15日（v26.12.0）移除；替代：`npu_chunk_gated_delta_rule_bwd` |
| GDN | [`npu_chunk_bwd_dv_local`](../fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dv_local/README.md) | `chunk_bwd_dv_local` | 局部 Value 梯度计算 | 已弃用（日落）；2027年1月15日（v26.12.0）移除；替代：`npu_chunk_gated_delta_rule_bwd` |
| GDN | `npu_prepare_wy_repr_bwd` | `prepare_wy_repr_bwd` | WY 表示反向，内部生成 dA | 暂缺README；已弃用（日落）；2027年1月15日（v26.12.0）移除；替代：`npu_chunk_gated_delta_rule_bwd` |
| GDN | [`npu_prepare_wy_repr_bwd_da`](../fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd_da/README.md) | `prepare_wy_repr_bwd_da` | WY 表示的 dA 梯度计算 | 已弃用（日落）；2027年1月15日（v26.12.0）移除；替代：`npu_chunk_gated_delta_rule_bwd` |
| GDN | [`npu_prepare_wy_repr_bwd_full`](../fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd_full/README.md) | `prepare_wy_repr_bwd_full` | 输入 dA 的 WY 表示反向 | 已弃用（日落）；2027年1月15日（v26.12.0）移除；替代：`npu_chunk_gated_delta_rule_bwd` |
| GDN | [`npu_chunk_local_cumsum`](../fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/README.md) | `chunk_local_cumsum` | 门控 chunk 内累加 | 已弃用（日落）；2027年1月15日（v26.12.0）移除；替代：`npu_chunk_gated_delta_rule_fwd` |
| GDN | [`npu_chunk_scaled_dot_kkt`](../fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_scaled_dot_kkt/README.md) | `chunk_scaled_dot_kkt` | chunk 内缩放点积 | 已弃用（日落）；2027年1月15日（v26.12.0）移除；替代：`npu_chunk_gated_delta_rule_fwd` |
| GDN | [`npu_recompute_w_u_fwd`](../fla/ops/ascendc/gdn/chunk_gdn_fwd/recompute_w_u_fwd/README.md) | `recompute_w_u_fwd` | 重计算 w/u 中间量 | 已弃用（日落）；2027年1月15日（v26.12.0）移除；替代：`npu_chunk_gated_delta_rule_fwd` / `npu_chunk_gated_delta_rule_bwd` |
| KDA | [`npu_chunk_kda_fwd`](../fla/ops/ascendc/kda/chunk_kda_fwd/README.md) | `chunk_kda_fwd` | chunk 前向 | - |
| KDA | [`npu_chunk_kda_fwd_prepare`](../fla/ops/ascendc/kda/chunk_kda_fwd_prepare/README.md) | `chunk_kda_fwd_prepare` | 前向预处理 | - |
| KDA | [`npu_chunk_kda_fwd_finalize`](../fla/ops/ascendc/kda/chunk_kda_fwd_finalize/README.md) | `chunk_kda_fwd_finalize` | 前向收尾 | - |
| KDA | `npu_chunk_kda_bwd` | `chunk_kda_bwd` | chunk 反向 | 暂缺README |
| KDA | [`npu_chunk_kda_bwd_prepare`](../fla/ops/ascendc/kda/chunk_kda_bwd_prepare/README.md) | `chunk_kda_bwd_prepare` | 反向预处理 | - |
| KDA | `npu_chunk_kda_bwd_intra` | `chunk_kda_bwd_intra` | chunk 内反向计算 | 暂缺README |
| KDA | `npu_chunk_kda_bwd_recompute` | `chunk_kda_bwd_recompute` | 反向中间量重计算 | 暂缺README |
| KDA | [`npu_chunk_kda_bwd_finalize`](../fla/ops/ascendc/kda/chunk_kda_bwd_finalize/README.md) | `chunk_kda_bwd_finalize` | 反向收尾 | - |
| KDA | [`npu_recurrent_kda`](../fla/ops/ascendc/kda/recurrent_kda/README.md) | `recurrent_kda` | 递推推理 | - |
| KDA | `npu_kda_gate_cumsum` | `kda_gate_cumsum` | 门控累加 | 暂缺README；已弃用（日落）；2027年1月15日（v26.12.0）移除；替代：`npu_chunk_kda_fwd` |
| 卷积 | [`npu_causal_conv1d_fn`](../fla/ops/ascendc/gdn/gdn_preprocess/causal_conv1d/README.md) | `causal_conv1d` | FN/prefill | - |
| 卷积 | [`npu_causal_conv1d_update`](../fla/ops/ascendc/gdn/gdn_preprocess/causal_conv1d/README.md) | `causal_conv1d` | UPDATE/decode | - |
| 卷积 | [`npu_causal_conv1d_bwd`](../fla/ops/ascendc/gdn/gdn_preprocess/causal_conv1d_bwd/README.md) | `causal_conv1d_bwd` | 卷积反向 | - |
| 卷积 | [`npu_causal_conv1d`](../fla/ops/ascendc/gdn/gdn_preprocess/causal_conv1d/README.md) | `causal_conv1d` | 旧 Host metadata 接口 | 已弃用，兼容保留；2027年1月15日（v26.12.0）移除；替代：`causal_conv1d_fn` / `causal_conv1d_update` |
| CP | [`npu_pre_process_fwd_kernel_merged`](../fla/ops/ascendc/cp/pre_process_fwd_kernel_merged/README.md) | `pre_process_fwd_kernel_merged` | 正向状态预处理与仿射摘要 | - |
| CP | [`npu_pre_process_bwd_kernel_merged`](../fla/ops/ascendc/cp/pre_process_bwd_kernel_merged/README.md) | `pre_process_bwd_kernel_merged` | 反向状态预处理与仿射摘要 | - |
| CP | `npu_merge_fwd_bwd_kernel` | `merge_fwd_bwd_kernel` | 合并各 rank 状态摘要 | 暂缺README |

备注中的弃用说明同时适用于不带 `npu_` 前缀的 Ascend C 入口。迁移至融合算子时，按其接口调整输入、输出和中间状态。

## 修订记录

| 日期 | 修改者 | 内容 |
| --- | --- | --- |
| 2026-10-27 |SilenceSYC | 整理算子工程映射、基本用途及弃用备注 |
