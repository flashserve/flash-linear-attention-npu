# RecurrentGatedDeltaRule 接口与支持范围

## 统一接口

公开 Python 入口为 `fla_npu.ops.ascendc.recurrent_gated_delta_rule`。
输入 Q/K 为 BF16 `[T,HK,128]`，V 为 BF16 `[T,HV,128]`，State 为
BF16/FP32 `[Cstate,HV,128,128]`，beta 为 BF16/FP32 `[T,HV]`。
T、HK、HV、Cstate 均为正数；HV/HK 必须是 1–16 范围内的整数。
Q/K/V 允许正 stride 的 view，但最后一维 stride 必须为 1，token stride 不小于 128；
State 最后一维 stride 为 1，行 stride 不小于 128，不允许内部重叠或与输入重叠。
storage offset 与实际 backing storage 一并保留，State 原地更新。

`actual_seq_lengths` 为 INT32 `[B+1]`：首项为无效前缀长度，后 B 项为每条序列的长度，
所有项之和为 T；长度为零的序列不参与计算。`ssm_state_indices` 为 INT32 `[T]` 或
`[B,Ckpt]`，有效索引位于 `[0,Cstate)`；跨请求写槽以及初始读槽不得冲突。
`num_accepted_tokens` 为可选 INT32 `[B]`，用于选择各序列的初始状态槽。

`g` 为可选 BF16/FP32 `[T,HV]`；不提供时不衰减。`gk` 是保留参数，非空时拒绝。
`scale` 为有限实数，默认 1.0。四个融合开关默认均为 False：

- `use_qk_l2norm_in_kernel`：Q/K 使用 FP32 `x / sqrt(sum(x*x)+1e-6)`。
- `use_gate_in_kernel`：需要 g 和 FP32 `[HV]` 的 A_log，dt_bias 可选；
  `g_eff = -exp(A_log) * softplus(g + dt_bias)`。
- `use_beta_sigmoid_in_kernel`：beta 使用 FP32 sigmoid。
- `allow_neg_eigval`：仅在 beta sigmoid 打开时有效，beta_eff 再乘 2。

输出为 BF16 `[T,HV,128]`；无效前缀没有数值语义。每个有效 token 写回一个 State checkpoint，
递推过程在 FP32 中进行，checkpoint 的 BF16 舍入不反馈到下一 token。

保留 `aclnnRecurrentGatedDeltaRule` 和 `aclnnRecurrentGatedDeltaRuleV2` 两套 host ABI。
前者维持历史参数表（BF16 beta、FP32 g、无融合开关），后者暴露完整融合契约。
两者在 A2/A3 上均执行 `op_kernel/arch22/recurrent_gated_delta_rule.h`，在 A5 上均执行
`op_kernel/arch35/recurrent_gated_delta_rule.h`，不存在旧 kernel 回退。
Python `_stable.py` 无论融合开关是否关闭，始终调用同一个 Stable-ABI adapter，
adapter 使用 V2 host ABI；V2 只保留为接口名称。

## 架构分派

A2/A3 和 A5 共享上述 host、dtype、shape、stride 与融合属性契约。差异仅位于 kernel：
A2/A3 使用 arch22 的 8-token 窗口实现；A5 使用 arch35 的 4-token ping-pong、三 State
bank 流水实现。A5 编译不启用 `--cce-auto-sync=on`，跨 MTE2/Vector/MTE3 依赖由 kernel
显式同步。历史接口参数说明参见 [ACLNN 文档](aclnnRecurrentGatedDeltaRule.md)。

## 校验与异常

Python 层在修改 State 前校验 dtype、shape、stride、元数据值和所有权。
host 侧校验 dtype、shape、view 边界、参数组合，tiling 再校验平台及资源。
非法输入返回参数错误并报告参数名与原因；缺少必要指针返回空指针错误。
