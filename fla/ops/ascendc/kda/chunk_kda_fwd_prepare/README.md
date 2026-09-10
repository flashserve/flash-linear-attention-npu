# Chunk KDA Forward Prepare

本目录当前只包含 `op_kernel/pseudocode/` 下的 A2/A3 Arch22 与 A5 Arch35 设计伪代码，
不是已注册算子。
仓库构建系统、Host Tiling、算子定义、aclnn/Python API 和设备入口参数均未接入，
因此不能从本目录导入、构建或运行 `chunk_kda_fwd_prepare`。
`chunk_kda_fwd_finalize` 的设计与实现也不在本目录范围内。

伪代码用于冻结共享的八 Stage 数据流、S=4 causal-prefix 72 KiB score packing、chunk-first 分核，
`HK` Q/K head 到 `HV` value/gate head 的 QK 头组映射、静态 UB/L1/workspace 生命周期和
ready/free 合同。代码直接采用 `GlobalTensor/LocalTensor`、`DataCopy`、`LoadData`、
`Mmad`、`Fixpipe`、HardEvent 和 Arch35 Mutex 的真实 API 形态，各 Stage 的搬运、
计算与同步都在对应函数内直接展开；核内 EventID/Mutex ID 和核间 ready/free flag ID
也在申请或主循环现场逐项列出。
`USE_EXP2=false` 与 `USE_EXP2=true` 都是必须保留的编译期路径，分别在自然对数域和
log2 域完成同一组门控计算。
本设计的 `q/k/v`、score 操作数及
`Aqk/Akk/w/u/qg/kg/qg_scaled/q_hat/k_hat` 固定为 BF16，
`gk/q_rstd/k_rstd/beta_eff` 固定为 FP32，gate/beta 只允许 FP32 或 BF16。
设计只保留一套公开输出接口，以下 13 个结果都写回 GM，不再按输出是否公开拆分
编译模式：

```text
gk, Aqk, Akk, w, u, qg, kg, qg_scaled,
q_hat, k_hat, q_rstd, k_rstd, beta_eff
```

输出固定为 head-major。dense 与 varlen 的 shape 分别为：

| 输出 | dense shape | varlen shape | dtype |
| --- | --- | --- | --- |
| `gk/w/qg/kg/qg_scaled` | `[B,H_v,T,128]` | `[H_v,T,128]` | `gk` 为 FP32，其余为 BF16 |
| `u` | `[B,H_v,T,128]` | `[H_v,T,128]` | BF16 |
| `Aqk/Akk` | `[B,H_v,T,64]` | `[H_v,T,64]` | BF16 |
| `q_hat/k_hat` | `[B,H_k,T,128]` | `[H_k,T,128]` | BF16 |
| `q_rstd/k_rstd` | `[B,H_k,T]` | `[H_k,T]` | FP32 |
| `beta_eff` | `[B,H_v,T]` | `[H_v,T]` | FP32 |

这些输出按消费者分类时允许重叠：`gk/w/u/kg` 是后续 FwdH 的输入，
`Aqk/qg_scaled` 是 Finalize 的输入；`q_hat/k_hat/q_rstd/k_rstd/beta_eff` 是反向
保存量，其中 `q_rstd/k_rstd` 在启用 L2Norm 时由其反向消费；
`Aqk/Akk/gk/w/qg/kg` 也会被反向直接使用或按重计算策略保存。
`u` 仅为对齐既有返回策略而随禁用重计算路径保留，当前反向不读取它。
`h/final_state` 由后续 FwdH 产生，不是 Prepare 输出；其中内部 `hCompute` 始终供
Finalize 使用，只有公开 `hOut` 和 `final_state` 属于用户可选状态结果。

`QkNormMode::Identity` 仍完整写出 `q_hat=q`、`k_hat=k`，并把有效 token 的
`q_rstd/k_rstd` 写为 FP32 `1`；`BetaMode::Raw` 把 beta 转为 FP32 后写入
`beta_eff`。因此固定接口在任一合法编译模式下都不存在未初始化输出。
GVA 下前四个 Q/K 保存量按 `H_k` 编址，一个 Q/K head 对应的连续 value head
组成一个 QK 头组，只有首个 value head 是公开输出 owner；其余 value head 可以保留本 head 的内部计算
副本，但不得重叠写同一段公开 GM。`beta_eff` 按 `H_v` 编址，由每个 value head
分别写回。
尚未由目标 CANN 头文件确认的布局和参数在调用现场标为 **TODO**；因此不能据此声称 A2/A3/A5
已经具备可构建、可调用的生产支持。详细说明见
[`op_kernel/pseudocode/README.md`](op_kernel/pseudocode/README.md)。

所有标为 **TODO** 的设备 API 参数、同步 mode 映射与计数深度、内存布局、TilingKey
与公开接口必须在正式实现前依据目标 CANN 版本重新确认；资源账本 host 测试不等价于
NPU 编译或测试。
