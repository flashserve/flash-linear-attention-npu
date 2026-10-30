# ChunkKdaBwdFinalize

## 功能

KDA 反向优化链路的最后阶段，合并 Prepare 和 Dhu 的结果，输出 Q/K/V、
beta、Gate 与参数梯度。A5 可融合 Q/K L2 归一化反向；A2/A3 复用原整体 Bwd 的
WY/Intra 计算，并在同一次 launch 中完成可选 GatePost。

A2/A3 的 Python `npu_chunk_kda_bwd` 和 `aclnnChunkKdaBwd` 接口保持不变，
内部由一个 executor 依次调用 Prepare、共用 Dhu、Finalize 三个 L0 算子，
不再执行整体 Bwd device kernel。A5 保持原路径。

## 输入

以下为 Finalize 算子自身的接口；Python V2 入口的类型转换与可选参数处理见
[优化接口](../chunk_kda_bwd/docs/api.md)。

| 输入 | 类型 | 含义 |
|---|---|---|
| q、k、v、v_new | BF16 | 前向 Token 张量 |
| akk | BF16 | 块内中间量 |
| h、dh | BF16 | 前向状态与状态梯度 |
| dv_scan | BF16 | Dhu 输出 |
| gk | FP32 | 累积 Gate |
| raw_g | BF16 / FP32 | 原始 Gate |
| d_aqk、dq_raw | FP32 | Prepare 输出 |
| beta | BF16 / FP32 | 衰减参数 |
| a_log | FP32 | Gate 参数 |
| dt_bias | FP32 | Gate 偏置 |
| q_rstd、k_rstd | 可选 FP32 | 成对提供的归一化中间量 |
| cu_seqlens、chunk_indices | 可选 INT64 | 变长序列元数据 |

输入须连续，K=V=128、chunk_size=64、Hq=Hv。支持定长及变长序列，
Token 为 [B,H,T,D] 或 [H,T,D]；h 为 [B,Nc,H,K,V] 或 [Nc,H,K,V]，
dh 与 h 一致，使用 dense [B,Nc,H,K,V] 或 packed [Nc,H,K,V]。

## 输出与属性

输出布局以 [接口约定](docs/api.md) 为准：output_layout 统一控制
dq/dk/dv/d_beta/d_g，dense 默认 BNSD，packed 默认 NTD。
写回实现见 [输出布局设计](docs/design.md)。

dq/dk/dv 为 BF16，d_beta 与 beta 同类型，d_a_log 为 FP32。
d_g 与 raw_g 同类型，d_dt_bias 为 FP32；raw_g/beta 有四种 dtype 组合，a_log/dt_bias 固定 FP32。
Stage11 按需将 raw_g 转 FP32；Stage12 以 FP32 归约并写回参数梯度。

上表及本节类型描述的是独立 `aclnnChunkKdaBwdFinalize` 接口。A2/A3 整体 Bwd
通过 L0 调用时，dq/dk/dv 直接输出 BF16，db/dg 分别跟随总接口的
beta/raw_g 输入 dtype，dA/dbias 固定 FP32；五个 Token 梯度
均按总接口的 output_layout 写回，无需总接口再做布局转置。
A2/A3 支持 BF16/FP32 beta，融合 sigmoid 反向时需要前向保存的 FP32 beta_eff；
raw_g 支持 BF16/FP32，a_log/dt_bias 固定 FP32。成对传入 q_rstd/k_rstd 时，在 Finalize 内完成 Q/K L2Norm 反向。

| 属性 | 要求 |
|---|---|
| scale | 必填，与前向一致 |
| lower_bound | 默认 -5，范围 [-5,0) |
| chunk_size | 64 |
| safe_gate、use_gate_in_kernel、use_exp2 | True |
| state_v_first | False |
| db/dg 输出 shape 与 output_layout | 独立调用默认 token-major；A5/V2 用现有 `BNSD/NTD_GATE_HEAD_MAJOR` 显式选择 head-major，不新增属性 |

独立 Finalize 默认按 token-major 输出 db/dg；A5/V2 通过现有 `output_layout` 显式选择并直接写公开
head-major db/dg，取消 token-major 临时缓冲与布局恢复。输出 shape 用于校验，方形场景不再猜测布局。
两种路径的 Gate 反扫、参数归约和算术顺序一致，公开 Python/aclnn、op def 与内部 L0 签名不变。

## 实现

采用 MIX_AIC_1_2，矩阵计算与向量处理协同执行；末阶段同步后归约参数梯度。
Tiling key 1/2 对应定长/变长，3/4 为对应的归一化反向版本。
A2/A3 使用 tiling key 101，支持 dense batch、尾块及 packed 元数据；
公开 Python 入口保留原有尾块和奇数 head 适配。独立阶段直接调用的边界场景
以及正式验收状态以 PR 验证记录为准。

| 文件/目录 | 职责 |
|---|---|
| op_host/ | 算子注册、形状推导、Tiling 与 ACLNN 接口 |
| op_kernel/ | Ascend950 kernel |
| [优化设计](../chunk_kda_bwd/docs/design.md) | 分带、精度与存储方案 |

构建时设置 `FLA_NPU_SOC=ascend950 FLA_NPU_OPS=chunk_kda_bwd`，
会包含 V1/V2 及优化链路依赖。
