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

仅保留一套两段式 host ABI，使用原接口名承载完整融合参数表：

```cpp
ACLNN_API aclnnStatus aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *beta,
    aclTensor *stateRef, const aclTensor *actualSeqLengths, const aclTensor *ssmStateIndices,
    const aclTensor *g, const aclTensor *gk, const aclTensor *numAcceptedTokens,
    const aclTensor *aLog, const aclTensor *dtBias, float scaleValue,
    bool useQkL2NormInKernel, bool useGateInKernel, bool useBetaSigmoidInKernel,
    bool allowNegEigval, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);

ACLNN_API aclnnStatus aclnnRecurrentGatedDeltaRule(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
```

参数顺序、类型及功能沿用原 V2 契约；不再导出带 V2 后缀的接口。
历史精简参数表已被替换，C/C++ 调用方必须按新头文件补充 aLog、dtBias 和四个布尔参数并重新编译；
动态调用方也必须更新函数签名。launcher 与 OPP 须配套更新。
Python 签名、默认值、返回值和 State 原地更新语义保持一致；Stable adapter 始终调用上述完整接口。
A2/A3 执行 `op_kernel/arch22/recurrent_gated_delta_rule.h`，A5 执行
`op_kernel/arch35/recurrent_gated_delta_rule.h`；本次接口收敛不改变 kernel、tiling 或融合路径。

## 架构分派

A2/A3 和 A5 共享上述 host、dtype、shape、stride 与融合属性契约。差异仅位于 kernel：
A2/A3 使用 arch22 的 8-token 窗口实现；A5 使用 arch35 的 4-token ping-pong、三 State
bank 流水实现。A5 编译不启用 `--cce-auto-sync=on`，跨 MTE2/Vector/MTE3 依赖由 kernel
显式同步。两段式调用示例参见 [ACLNN 文档](aclnnRecurrentGatedDeltaRule.md)。

## 校验与异常

Python 层仅转发参数、获取当前 stream，并在调用成功后更新 State 版本；不执行输入校验，
也不回读设备上的 metadata。
C++ Stable-ABI 适配层在创建 ACL 描述符和分配输出前，检查所有已提供的 tensor 均位于
NPU 且与 query 同设备；设备不匹配时抛出包含参数名及设备信息的异常。
host 侧校验 dtype、shape、stride、view 边界、内存别名和参数组合，tiling 再校验平台及资源。
metadata 的实际值由调用方保证符合上述约束，包括长度及其总和、索引范围、accepted 范围和
跨请求 State 槽所有权；这些内容不在 Python 调用路径中同步校验。
host 检出的非法输入返回参数错误并报告参数名与原因；缺少必要指针返回空指针错误。
