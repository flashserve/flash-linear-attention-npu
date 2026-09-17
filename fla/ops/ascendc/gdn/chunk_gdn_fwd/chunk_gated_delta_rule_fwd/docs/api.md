# GDN 前向返回合同

本次变更基于 main `e22e0bf4`，输入、属性、支持范围及原六项输出的定义见 [README](../README.md)。

Python 入口 `npu_chunk_gated_delta_rule_fwd`（及其别名 `chunk_gated_delta_rule_fwd`）固定返回：

```python
(o, final_state, g_cumsum, A, beta_eff, h, q_hat, k_hat, q_rstd, k_rstd)
```

| 输出 | 开启 use_qk_l2norm_in_kernel | 关闭 use_qk_l2norm_in_kernel |
| --- | --- | --- |
| q_hat / k_hat | prepare 的归一化结果；shape、layout、dtype 与各自输入一致 | 原始 q / k 对象本身，无额外分配或复制 |
| q_rstd / k_rstd | FP32，固定 `[B,Hk,T]`，所有 layout 相同 | None |

归一化公式沿用 prepare：`rstd = rsqrt(sum(x**2, dim=-1) + 1e-6)`，
`hat = cast(x * rstd, input_dtype)`。不重新计算或改变 epsilon、舍入和有效区语义。
四项输出不受 disable_recompute 或 return_intermediate_states 控制。
别名输出与原始输入共享存储，调用方修改其中任意一个都会影响另一个。

反向始终使用返回的 q_hat/k_hat；归一化开关与前向一致，并传入 q_rstd/k_rstd。
梯度 dq/dk 对应归一化前的原始输入。rstd 的头数是 Hk，不按 Hv 扩展。
变长时保持物理 batch/token 维度，不按逻辑序列拆分。

ACLNN C 参数顺序、数量不变。关闭归一化时四个可选 ACLNN 输出仍传空，别名只在 Python 层返回。
开启时 qHat/kHat 输出 descriptor 与输入一致，rstd descriptor 必须为 `[B,Hk,T]`。
BSND/TND 下旧 `[B,T,Hk]` rstd descriptor 不再接受。
Python 返回值数量由 6 变为 10，原有六变量解包必须迁移。
