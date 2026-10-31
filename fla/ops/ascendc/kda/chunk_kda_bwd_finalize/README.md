# ChunkKdaBwdFinalize

## 功能

KDA 反向优化链路的最后阶段，合并 Prepare 和 Dhu 的结果，输出 Q/K/V、
beta、Gate 与参数梯度。A5 可融合 Q/K L2 归一化反向；A2/A3 复用原整体 Bwd 的
WY/Intra 计算，并在同一次 launch 中完成可选 GatePost。

A2/A3 的 Python `npu_chunk_kda_bwd` 和 `aclnnChunkKdaBwd` 接口保持不变，
内部由一个 executor 依次调用 Prepare、共用 Dhu、Finalize 三个 L0 算子，
不再执行整体 Bwd device kernel。A5 V2 的保存与重计算路径也复用本文的 Finalize。

## 支持范围

- 支持 Ascend910B、Ascend910_93 和 Ascend950 上已注册的 Finalize 路径；输入须连续，
  Hq=Hv、K=V=128、chunk_size=64，支持定长及变长序列。
- q/k/v 及 dq/dk/dv 为 BF16；raw_g 与 beta 独立支持 BF16/FP32，dg/db 分别跟随
  raw_g/beta 的 dtype；A_log/dt_bias 及 dA/dbias 固定 FP32。
- safe_gate、use_gate_in_kernel、use_exp2 必须为 True，state_v_first=False；
  lower_bound 范围为 `[-5,0)`。
- q_rstd/k_rstd 必须成对提供。A2/A3 支持融合 beta sigmoid 反向，计算输入为前向保存的
  FP32 beta_eff；A5 当前要求 `use_beta_sigmoid_in_kernel=False` 且
  `allow_neg_eigval=False`。
- 总接口重计算路径保留 H≤256 且 H 为 8 的倍数的限制；保存路径不受该限制。
  独立 Finalize 的边界场景以其接口校验为准。

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
| beta | BF16 / FP32 | 衰减参数；独立调用和内部计算输入均为 head-major |
| a_log | FP32 | Gate 参数 |
| dt_bias | FP32 | Gate 偏置 |
| q_rstd、k_rstd | 可选 FP32 | 成对提供的 head-major 归一化中间量 |
| cu_seqlens、chunk_indices | 可选 INT64 | 变长序列元数据 |

独立调用的 q/k/v/raw_g 为 head-major，Token shape 为 `[B,H,T,D]` 或 `[H,T,D]`；
beta 为 `[B,H,T]` 或 `[H,T]`。h 为 `[B,Nc,H,K,V]` 或 `[Nc,H,K,V]`，dh 与 h 一致。
总接口混合布局的输入规则见下节。

## 输出、布局与属性

七个输出顺序为 `(dq,dk,dv,d_beta,d_g,d_a_log,d_dt_bias)`，具体 shape 以
[接口约定](docs/api.md) 为准。dq/dk/dv 为 BF16，d_beta 与 beta 同类型，
d_g 与 raw_g 同类型，d_a_log/d_dt_bias 固定 FP32。raw_g/beta 有四种 dtype 组合；
Stage11 按需将 BF16 输入转换为 FP32，Stage12 以 FP32 归约并写回参数梯度。

上表及本节类型描述的是独立 `aclnnChunkKdaBwdFinalize` 接口。A2/A3 整体 Bwd
通过 L0 调用时，dq/dk/dv 直接输出 BF16，db/dg 分别跟随总接口的
beta/raw_g 输入 dtype，dA/dbias 固定 FP32；五个 Token 梯度
均按总接口的 output_layout 写回，无需总接口再做布局转置。
A2/A3 支持 BF16/FP32 beta，融合 sigmoid 反向时需要前向保存的 FP32 beta_eff；
raw_g 支持 BF16/FP32，a_log/dt_bias 固定 FP32。成对传入 q_rstd/k_rstd 时，在 Finalize 内完成 Q/K L2Norm 反向。

### 独立 Finalize

独立 `aclnnChunkKdaBwdFinalize` 的公开签名保持不变，不暴露 `model_layout` 参数。
此时内部属性 `model_layout=""`，`output_layout` 统一控制 **全部五个逐 token 梯度**
dq/dk/dv/db/dg：dense 支持 BNSD/BSND，默认 BNSD；packed 支持 NTD/TND，默认 NTD。
dA/dbias 不受 `output_layout` 控制。

### 总接口内部混合布局

PR993 总接口只有一个公开 `layout`，合法值为 `BSND/BNSD/TND/NTD`。内部 L0 调用将同一
布局同时传给 `output_layout` 和 `model_layout`，形成以下契约：

- q/k/v 与 dq/dk/dv 随布局；有 rstd 时 q/k 固定为 head-major q_hat/k_hat。
- 公开 do/raw_g/beta 与 dg/db 固定 token-major。A5 ACLNN 将 do 直接传给 Prepare/Dhu
  进行原生 stride 读取，A2/A3 仍转置一次供两个小算子共用；普通 beta 转置一次供
  Recompute/Finalize 共用，Finalize 收到的 beta 仍为 head-major，raw_g 按 token-major 直接读取。
- `model_layout` 是内部可选 string 属性，默认空字符串表示 standalone；非空仅接受上述
  四个布局。op 定义的 `attr_num=11`，内部 L0 函数签名在末尾追加 `modelLayout`，
  独立 ACLNN 通过默认值继续走 standalone。

输出 shape 只用于校验，不能根据 `T == H` 或交换维度优先级推断布局。独立调用和总接口
混合布局由显式字符串消歧；非法 `model_layout` 的 host 报告包含参数名、实际值和合法集合。

## Raw tiling

A5 `ChunkKdaBwdFinalizeTilingData` 的逻辑结构大小为 **184 字节**。host 将 raw tiling
size 设为 `sizeof(TilingData)+8`，即 **192 字节**，与设备侧 `opParaSize` 对齐；
Arch22 使用自身 tiling 路径，不套用该 A5 大小。host 测试读取 raw size 后再使用 typed
getter 检查字段，避免 typed getter 重置 raw size 后造成误判。

## 实现

采用 MIX_AIC_1_2，矩阵计算与向量处理协同执行；末阶段同步后归约参数梯度。
Tiling key 1/2 对应定长/变长，3/4 为对应的归一化反向版本。
A2/A3 使用 tiling key 101，归一化反向版本使用 102，支持 dense batch、尾块及 packed 元数据；
公开 Python 入口保留原有尾块和奇数 head 适配。

| 文件/目录 | 职责 |
|---|---|
| op_host/ | 算子注册、形状推导、Tiling 与 ACLNN 接口 |
| op_kernel/ | Ascend C kernel |
| [输出布局与 dtype 设计](docs/design.md) | 布局直写、片上转换和 raw tiling |
| [优化设计](../chunk_kda_bwd/docs/design.md) | 分带、精度、存储与 O1/O2 方案 |

构建时设置 `FLA_NPU_SOC=ascend950 FLA_NPU_OPS=chunk_kda_bwd`，
会包含 V1/V2 及优化链路依赖。
