# GDN backward finalize 全融合 Ascend C 算子设计

> 方案设计规则版本：`V1`
>
> 本案例只规定方案设计文档的章节结构、推导过程和细节深度。新算子设计以
> [`03-solution-design.md`](../../03-solution-design.md) 的 `R01`–`R21` 为准；具体公式、
> Stage、shape、地址和资源配置由当前算子根据已确认接口、CPU 标杆和目标硬件约束独立推导。

## 1. 目标

本文设计一个覆盖 stage golden 全部计算的 GDN backward `finalize` 全融合算子。
算子完整包含：

```text
chunk_bwd_dqkwg
  -> prepare_wy_repr_bwd
  -> dk/dg 合成
  -> chunk 内 reverse cumsum
  -> Q/K L2Norm backward（可选）
  -> beta sigmoid backward（可选）
```

目标固定对标 `use_gate_in_kernel=False`：`g` 已是 chunk local cumsum 后的 log-space
gate，本算子不融合 GDN gate 激活 backward。

数学边界严格对齐：

- `gdn_backward_golden.py`：Stage 0--9 逐 stage 公式；
- `run_split_stage_golden.py`：完整调用和最终输出；
- `run_triton_dqkwg_prepare.py`：Triton 真实调用链。

实现结构参考 Ascend C
[`chunk_gated_delta_rule_bwd_dhu`](https://github.com/flashserve/flash-linear-attention-npu/tree/main/fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu)
的混合 AIC/AIV、定长/变长 offset helper、跨核同步和 workspace 管理方式，但采用不同的
逻辑分核：**只按 chunk 分核，不按 head 分核**。每个 chunk task 在核内遍历所有
`HK/HV` head。本版本仅支持 GVA 比例 `1:1`--`1:4`，即 `G=HV/HK in {1,2,3,4}`。
核内使用动态 HV 任务组：`G=3` 时组大小取 3，`G=1/2/4` 时取 4。对每个
任务组。默认 Stage 中两个 AIV 按任务序号交错承包完整 HV：AIV0 处理 0、2、4...，
AIV1 处理 1、3、5...；Stage 0/2/4/6/12/15 例外，按 4.1 节将同一 HEAD 的 token 行
切成两个 half。两个 AIV 不设置无真实依赖的 stage 级汇合；每个
AIV 完成自己承包的当前 stage 任务后即可独立进入下一 stage。只有存在真实的跨
AIV 数据依赖或归约时才增加对应同步。执行方式参考 `bwd_dhu` 的 head round，但
不固定为 4 个 HV。

本文 Stage 0--15 是按 L1/UB 驻留约束重新规划后的目标编号，不沿用当前代码中的旧
Stage 编号。后续实现应整体按本文的公式、数据流、workspace/UB 布局和 AIV 交错
分工重新对应。

## 2. 范围与 shape

第一版目标：

```text
SoC               Ascend 950 only
q/k layout        [B, HK, T, K]
value/gate layout [B, HV, T]
state layout      [B, HV, NT, K, V]
K                 128
V                 128
chunk_size        64
dtype             bf16，gate/beta 允许 fp32
GVA               HV % HK == 0, G=HV/HK in {1,2,3,4}
```

目标 case：

```text
B=1, HK=16, HV=32, T=12288, K=V=128, BT=64, BF16
```

`BT/K/V` 是固定规格：`BT=64`、`K=128`、`V=128`。host 侧必须明确拦截其它值，
不能选择其它 tiling 分支或静默执行。尾 chunk 和 varlen 仍允许当前有效长度 `M<64`。
算子定义和编译配置只注册 `ascend950`，不提供 910B、910_93 或其它 SoC 的 fallback
kernel。

本算子不包含前序 `bwd_dhu` 隐藏状态递推，而是接收已经生成的 `h/dh`。golden
中生成但未参与 Stage 0--9 公式的 `_w` 不进入接口。

符号：

```text
B   batch size
T   定长每 batch token 数；varlen 时为 packed token 总数
HK  q/k head 数
HV  value/gate head 数
G   HV / HK
K   q/k dim
V   value dim
BT  chunk_size
NT  总 chunk 数
M   当前 chunk 有效 token 数，M <= BT
TG  HV task group 大小，`TG = (G == 3) ? 3 : 4`
hk  当前 hv 映射的 q/k head，hk = hv / G
```

主要张量：

| 张量 | Shape | 说明 |
|---|---:|---|
| `q, k, dq, dk` | `[B,HK,T,K]` | q/k 已是 L2Norm forward 输出 |
| `v, v_new, do, du, dv` | `[B,HV,T,V]` | value 侧输入和输出 |
| `g, beta, dbeta, dg` | `[B,HV,T]` | `g/beta` 为变换后值 |
| `beta_raw` | `[B,HV,T]` | beta sigmoid 变换前值，仅 beta 融合分支使用 |
| `h, dh` | `[B,HV,NT,K,V]` | 每 chunk state，head-first |
| `A` | `[B,HV,T,BT]` | 每 token 行存一个 chunk 内矩阵行 |
| `q_rstd, k_rstd` | `[B,HK,T]` | L2Norm forward 保存值，fp32 |

golden 单 chunk 中将 `h/dh` 读取为逻辑 `[V,K]`，将 `A` 读取为逻辑
`[BT,BT]`。kernel 的 GM layout 和 Cube 搬运必须保持该转置语义。

## 3. 算子接口

接口名称固定为：

```text
GE/Ascend C 算子名：ChunkGatedDeltaRuleBwdFinalize
ACLNN workspace：  aclnnChunkGatedDeltaRuleBwdFinalizeGetWorkspaceSize
ACLNN execute：    aclnnChunkGatedDeltaRuleBwdFinalize
Python：           fla_npu.ops.ascendc.chunk_gated_delta_rule_bwd_finalize
```

### 3.1 输入

| 名称 | Shape | dtype | 来源/用途 |
|---|---:|---|---|
| `q` | `[B,HK,T,K]` | bf16 | 归一化 Q |
| `k` | `[B,HK,T,K]` | bf16 | 归一化 K |
| `v` | `[B,HV,T,V]` | 同 `q` | prepare backward |
| `v_new` | `[B,HV,T,V]` | 同 `q` | dqkwg value 输入 |
| `do` | `[B,HV,T,V]` | 同 `q` | 上游输出梯度 |
| `du` | `[B,HV,T,V]` | 同 `q` | 原始 value 梯度 |
| `g` | `[B,HV,T]` | bf16/fp32 | 变换后的 chunk gate |
| `beta` | `[B,HV,T]` | 同 `g` | sigmoid 后 beta |
| `h` | `[B,HV,NT,K,V]` | 同 `q` | forward chunk state |
| `dh` | `[B,HV,NT,K,V]` | 同 `q` | `bwd_dhu` 输出 |
| `A` | `[B,HV,T,BT]` | 同 `q` | WY inverse/中间矩阵 |
| `q_rstd` | `[B,HK,T]` | fp32 | Q L2Norm 保存值 |
| `k_rstd` | `[B,HK,T]` | fp32 | K L2Norm 保存值 |
| `beta_raw` | `[B,HV,T]` | 同 `g` | beta sigmoid 原始输入 |
| `cu_seqlens` | `[seqNum+1]` | int64 | 可选 varlen 序列边界 |
| `chunk_indices` | `[2*NT]` | int64 | 可选 `(seqIdx,localChunkIdx)` pair |

属性：

| 名称 | 类型 | 默认值 |
|---|---|---:|
| `scale` | float | `1/sqrt(K)` |
| `chunk_size` | int64 | 64 |
| `use_qk_l2norm_in_kernel` | bool | true |
| `use_beta_sigmoid_in_kernel` | bool | true |
| `allow_neg_eigval` | bool | false |

`du` 必须保持调用本算子前的原始值。dqkwg 只用它生成 `dw0`，不能先覆盖成其它
`dv` 再传入 prepare 路径。

参数约束：

- `scale=None` 时 wrapper 传 `1/sqrt(128)`；
- `chunk_size` 为兼容上层调用保留，但只接受 `64`；
- `use_qk_l2norm_in_kernel=True` 时 `q_rstd/k_rstd` 必须非空；
- `use_beta_sigmoid_in_kernel=True` 时 `beta_raw` 必须非空；
- `allow_neg_eigval=True` 要求 `use_beta_sigmoid_in_kernel=True`，并将
  beta sigmoid backward 的导数系数乘 2；
- 本算子固定对标 `use_gate_in_kernel=False`，输入 `g` 已是 chunk local cumsum 后的
  log-space gate；接口不接收 `g_input/A_log/dt_bias`；
- `cu_seqlens/chunk_indices` 必须同时为 `None` 或同时非空；
- 返回顺序固定为 `(dq, dk, dv, dbeta, dg)`。


## 4. Stage 0--15 完整数学语义

以下按单个 `(chunk,hv)` 描述。令 `hk=hv/G`，当前 chunk 有效长度为 `M`。
本轮只确定 Stage、完整张量搬运和容量，不确定最终计算精度；资源表暂按大型张量
BF16、小向量与归约标量 FP32 估算。下文统一标注
完整 chunk 的逻辑 shape；尾 chunk 仍按相同 `[BT,...]` 逻辑 shape 描述，但只有前
`M` 行有效，矩阵只有左上角 `M x M` 有效。

单个 `(chunk,hv)` 的基础 shape 为：

```text
q, k                       # [BT,K]
v, v_new, do, du           # [BT,V]
g, beta                    # [BT]
h, dh                      # [V,K]
A                          # [BT,BT]
q_rstd, k_rstd             # [BT]
beta_raw                   # [BT]，仅 use_beta_sigmoid_in_kernel=True
```

本节保持 golden Stage 0--9 的数学结果，只把
`dkBase -> dkIntra -> dkHv` 调整到 DQ 公式之后计算。

记当前 K head 对应的 GVA value-head 集合为：

```text
H(hk) = { hv_i | floor(hv_i / G) == hk }
```

除非公式中另有说明，token 下标 `i/j/t` 的有效范围都是 `[0,M)`；`rowSum` 删除
最后一维，`colSum` 删除倒数第二维，`tril` 不改变输入 shape。

本节所有地址均以整块物理存储起点为 `0 KiB`，使用半开区间 `[start,end)`：

```text
L1[0,512)      当前 Cube Stage 可使用的完整 L1 空间
UB[0,248)      当前 Vector Stage 可使用的完整 UB 空间
```

后续“空间布局”中的偏移均指上述绝对 KiB 偏移。L1 和 UB 内部没有固定
输入区、tensor 区或小向量区边界；任一 Stage 可在完整物理空间内
重新安排已结束生命周期的地址。UB 采用两端向中间挤压：大型 BF16 tensor 从
`UB[0)` 向高地址增长，FP32 `[2,BT]` 小向量从 `UB[248)` 按 0.5 KiB 子槽
向低地址增长。该策略只用于 tensor 首次分配；一旦分配，生命周期内物理地址固定。
因此短生命周期 tensor 释放后，后续 Stage 可能在长生命周期 tensor 之间留下内部
空洞，这类空洞只能分配给尺寸匹配的新 tensor，不能通过搬移存活 tensor 消除。
若保存两个 scalar，则对应 0.5 KiB 子槽中
只有开头 8 Byte 有效。这只是当前生命周期布局，不是不可借用的物理分区。
L1 中标记为 `[4]` 的区间
按四等份连续存放 HV0--HV3。例如
`L1[128,256)=h[4]` 等价于 `L1[128,160)=h[HV0]`、
`L1[160,192)=h[HV1]`、`L1[192,224)=h[HV2]`、
`L1[224,256)=h[HV3]`；其余 `[4]` 区间采用相同等分规则。

所有 Stage 都必须实现可验证的双缓冲流水，不允许只在文档中写
“可并发”而不给出物理 slot 和事件闭环。Vector Stage 使用两份
UB ping/pong：当前 HV 的 VF 执行时，MTE2 将下一个交错 HV
搬入另一 slot；当前结果由 MTE3 读取时，V 可继续使用另一
slot。每个 UB slot 使用 `MTE2_V -> V_MTE3 -> MTE3_MTE2`
闭环；覆盖 slot 前必须等上一轮 MTE3 读完，`MTE2_V`
负责保证后续 V 只消费新搬入的数据。Cube Stage 的 L0A、L0B、L0C 分别维护独立 ping/pong，每次
使用后立即取反；同时用不同 resident 的独立 flag 发射
下一 HEAD 的 GM->L1 搬运，使 MTE2 与当前 MTE1/Cube/Fixpipe 重叠。每层流水
都要按 slot 建立生产者/消费者事件，复用前闭环上一轮依赖。

任务组之间不允许仅依据“当前已实现的最后一个 Stage 完成”就复用仍属于完整
S0--S15 DAG 的 resident。下一任务组开始写 L1/UB 前，必须确认当前任务组中所有
与目标地址重叠的 tensor 都已完成真实末次消费。正式完整实现默认同一核上的任务组
按 S0--S15 完整闭环后再进入下一组；如果阶段性调试只实现到某个中间 Stage，则该
Stage 的调试输出不得占用下一任务组会提前重写的 resident 地址，应放入当前未冲突
的独立地址，或只写 GM 调试输出。不能把“后续 Stage 尚未实现”等同于 tensor
生命周期已经结束。

### 4.1 同一 HEAD 的 token-half Vector 协议

Stage 0/2/4/6/12/15 固定将同一 HEAD 沿 token 维分给两个 AIV，不再按 HEAD 分核：

```text
ROW_CAP = BT / 2 = 32
split = min(ROW_CAP, M)
AIV0 rows = [0, split)
AIV1 rows = [split, M)
```

每个 AIV 的大型 tensor 只分配 `[ROW_CAP,...]`。尾块 `M<=32` 时 AIV1 没有有效行，
不发射空 MTE/V 指令，也不参与该 HEAD 的数据 ready；`M>32` 时 AIV0 固定处理前 32 行，
AIV1 处理剩余 `M-32` 行。两核写 GM/L1 时使用原 tensor 的全局 row offset，输出半区
互不重叠。以下 6 个 Stage 的 per-AIV 布局覆盖原完整 HEAD `[2]` 执行解释；其它 Stage
仍使用各自原有布局。

Stage 0 每个 AIV 的 UB 布局：

```text
UB[0,8)        k_half[32,K]；BF16；保留至 S14
UB[8,16)       v_half[32,V]；BF16；保留至 S6
UB[16,20)      gate_dA_half[32,BT]；BF16；保留至 S12
UB[20,28)      vb_half[32,V]；BF16；写 GM 后释放
UB[28,36)      kbg_half[32,K]；BF16；写 GM 后释放
UB[36,244.75)  空闲；S0 可复用及容纳无依赖 Cube 输出
UB[244.75,245) g_full[BT]；FP32；两个 AIV 各从 GM 读取一份，S0 后释放
UB[245,245.125) beta_half[32]；FP32；保留至 S12
UB[245.125,245.25) g_exp_half[32]；FP32；保留至 S12
UB[245.25,245.375) decay_half[32]；FP32；保留至 S12
UB[245.375,245.5) bg_half[32]；FP32；保留至 S12
UB[245.5,245.625) g_last_half[32]；FP32 广播；保留至 S10
UB[245.625,248) 空闲；S0 可复用
```

`gate_dA_half[i,j]` 只拆输出行，列 `j` 仍覆盖完整 `[0,M)`，因此两个 AIV 都需要完整
`g_full[BT]`。这是 S0 唯一的跨 half 重复 GM 读取；不需要跨核交换 g。

Stage 2 每个 AIV 的 UB 布局：

```text
UB[0,8)        k_half[32,K]；BF16；保留
UB[8,16)       v_half[32,V]；BF16；保留
UB[16,20)      gate_dA_half[32,BT]；BF16；保留
UB[20,28)      dvb_half 后续输出槽[32,V]；BF16
UB[28,32)      dA_u_half -> dA_u_lower_half[32,BT]；BF16；保留至 S4
UB[32,36)      dA_w0_half 后续输出槽[32,BT]；BF16
UB[36,44)      kb_half[32,K]；BF16；写 GM 后释放
UB[44,247.375) 空闲；S2 可复用
UB[247.375,248) beta/g_exp/g_last/decay/bg half；FP32；沿用 S0 固定地址
```

Stage 4 每个 AIV 的 UB 布局：

```text
UB[0,28)       k/v/gate_dA/dvb half；BF16；保持原地址
UB[28,32)      dA_u_lower_half[32,BT]；BF16；S4 消费后释放
UB[32,36)      dA_w0_half -> dA0_half[32,BT]；BF16；写 GM 后释放
UB[36,247.375) 空闲；S4 可复用
UB[247.375,248) beta/g_exp/g_last/decay/bg half；FP32；保持原地址
```

Stage 6 每个 AIV 的 UB 布局：

```text
UB[0,8)        k_half[32,K]；BF16；保留
UB[8,16)       v_half[32,V]；BF16；S6 消费后释放
UB[16,20)      gate_dA_half[32,BT]；BF16；保留
UB[20,28)      dvb_half -> dv_half[32,V]；BF16；写 GM 后释放
UB[28,32)      a2_half[32,K]；BF16；保留至 S8
UB[32,247.25)  空闲；S6 可复用
UB[247.25,247.375) db_v_partial_half[32]；FP32；保留至 S8
UB[247.375,248) beta/g_exp/g_last/decay/bg half；FP32；保持原地址
```

Stage 12 每个 AIV 的 UB 布局：

```text
UB[0,8)        k_half[32,K]；BF16；S12 使用后保留至 S14
UB[8,16)       dkb_half[32,K]；BF16；S12 从 GM 搬入，消费后释放
UB[16,20)      gate_dA_half[32,BT]；BF16；消费后释放
UB[20,28)      dkb_t_half[32,K]；BF16；S12 从 GM 搬入，消费后释放
UB[28,36)      do_half -> do_g_half[32,V]；BF16；写 GM/L1 后释放
UB[36,44)      v_new_half -> v_decay_half[32,V]；BF16；写 GM/L1 后释放
UB[44,48)      ds0_half -> ds_half[32,BT]；BF16；写 GM/L1 后释放
UB[48,56)      dkbg0_half -> dk_prepare_half[32,K]；BF16；保留至 S14
UB[56,247)     空闲；S12 可复用
UB[247,247.125) beta_raw_half -> dbeta_half[32]；FP32；写 GM 后释放
UB[247.125,247.25) db_v_half -> db_prepare_half[32]；FP32；消费后释放
UB[247.25,247.375) dg_prepare_half[32]；FP32；保留至 S14
UB[247.375,247.5) beta_half[32]；FP32；消费后释放
UB[247.5,247.625) g_exp_half[32]；FP32；消费后释放
UB[247.625,247.75) state_term_half[32]；FP32 广播；保留至 S14
UB[247.75,247.875) decay_half[32]；FP32；消费后释放
UB[247.875,248) bg_half[32]；FP32；消费后释放
```

S12 的 `ds/do_g/v_decay` 由两个 AIV 写入同一 HEAD 的不同行半区。S13 Cube 读取完整
矩阵，因此这是一个真实的 per-HEAD 汇合：AIC 必须确认两个有效 half 都完成后才能消费
该 HEAD；尾块 `M<=32` 只等待 AIV0。该汇合只针对当前 HEAD，不建立整组 Stage 12
barrier，也不允许用它解释其它 UB 地址覆盖。

### Stage 0：Vector，gate/beta 系数与 prepare Cube 输入

```text
g_exp[i]  = exp2(g[i])                    # g_exp: [BT]
g_last    = g_exp[M-1]                    # 数学上为 scalar；实现中广播保存为 [BT]
gate_dA[i,j] = exp2(g[i] - g[j])           # gate_dA: [BT,BT]
decay[i]  = exp2(-g[i] + g[M-1])           # decay: [BT]
bg[i]     = beta[i] * g_exp[i]             # bg: [BT]
kbg       = k * bg[:,None]                 # kbg: [BT,K]
vb        = v * beta[:,None]                # vb: [BT,V]
```

空间布局（UB，单位 KiB）：

```text
UB[0,32)       k[2]；BF16；S0 搬入，保留至 S15
UB[32,64)      v[2]；BF16；S0 搬入，保留至 S6
UB[64,80)      gate_dA[2]；BF16；S0 生成，保留至 S12
UB[80,112)     vb[2]；BF16；S0 生成，写入 GM 后释放
UB[112,128)    dA_u[2] 后续输出槽；BF16；S1 生成，S2 原位处理后保留至 S4
UB[128,160)    kbg[2]；BF16；S0 生成，写入 GM 后释放
UB[160,245)    空闲；无数据；S0 可复用
UB[245,245.5)  g[2]；FP32；S0 搬入，S0 结束后释放
UB[245.5,246)  beta[2]；FP32；S0 搬入，保留至 S12
UB[246,246.5)  g_exp[2]；FP32；S0 生成，保留至 S12
UB[246.5,247)  g_last[2]；FP32；S0 生成，保留至 S10
UB[247,247.5)  decay[2]；FP32；S0 生成，保留至 S12
UB[247.5,248)  bg[2]；FP32；S0 生成，保留至 S12
```



当前 AIV 最多承包 2 个交错 HV，因此 `g/beta/g_exp/g_last/decay/bg/gate_dA`
各预留 2 份 UB 空间。
当前只设计 Stage 和容量，不讨论精度；`gate_dA` 以及
后续大型跨 Stage 中间量统一按 BF16 驻留。`k/v` 在本 Stage 首次进入 Vector 路径后，分别以
BF16 `[2,BT,K]` 和 `[2,BT,V]` 保留在 UB；`k` 保留到 Stage 15，
`v` 保留到 Stage 6。`beta/g_exp/bg/gate_dA` 保留到 Stage 12，其中
`gate_dA` 和 `bg` 均由 Stage 0 计算一次，后续 Vector Stage 直接复用。
原始 `g` 完成 Stage 0 的全部公式后释放。

操作流程：

1. 首先只搬入当前一个 HV 的完整 `g/beta/k/v`。处理当前
   ping/pong slot 时，MTE2 将本 AIV 的下一个交错 HV 搬入另一
   slot。覆盖下一 slot 前等待该 slot 的 `MTE3_MTE2`；每次搬运只对应一个 HV，不合并两个 head；尾块仍分配完整
   `[BT,...]` 逻辑 buffer，无效行在 VF 中通过有效长度 `M` 屏蔽。
2. 每次 VF 严格只处理一个 HV，完整计算该 HV 的
   `g_exp/g_last/gate_dA/decay/bg/kbg/vb`。当前 VF 结束后
   `streamSlot` 取反，下一次 VF 等待并消费另一份
   `MTE2_V` flag 对应的 slot。
   VF 首先将 `g/beta` 转换为 FP32 `gReg/betaReg` 并生成 `bgReg`；
   后续逐行因子使用 RegTensor Gather 直接从 `gReg/betaReg/bgReg`
   广播，不重复计算公共因子。
   需要跨 Stage 的
   `beta/g_exp/g_last/decay/bg/gate_dA/k/v` 直接写入各自跨 Stage 保存地址；仅供后续
   Cube 使用的 `vb/kbg` 写入本 Stage 临时地址。
3. `kbg[2,BT,K] + vb[2,BT,V]` 按 BF16 共占 64 KiB。S0 自身的大型 tensor 为
   `gate_dA 16 KiB + k 32 KiB + v 32 KiB + kbg/vb 64 KiB = 144 KiB`。
   计入无执行顺序约束的 Stage 1 写入 `dA_u[2]` 后，大型数据占用连续的
   `UB[0,160)`；3 KiB FP32 小向量从 UB 尾端反向排布，中间仅保留一个连续的
   `UB[160,245)` 可复用区。
   每个 HV 的 VF 完成后，将当前 `vb/kbg` 写入 GM。当前 MTE3 读取一份结果时，下一份 VF 可使用
   另一 slot，两份 slot 的 `MTE3_MTE2` 事件独立闭环。
4. `vb/kbg` 完成 GM 搬运后均不在 UB 保留。公共因子
   `bg=beta*g_exp` 保留在 UB 小向量地址，后续 Vector 公式直接复用。

### Stage 1：Cube，DW 与 dA_u

```text
dw0  = du @ h                               # dw0: [BT,K]
dA_u = du @ vb.T                            # dA_u: [BT,BT]
```

两条矩阵乘只共享只读 `du`，彼此不读取本 Stage 输出。

空间布局（L1，单位 KiB）：

```text
L1[0,128)    空闲；无数据；S1 可复用
L1[128,256)  h[4]；BF16；S1 搬入，保留至 S13
L1[256,320)  du[4]；BF16；S1 搬入，保留至 S3
L1[320,384)  kbg[4]；BF16；S3 从 GM 搬入，S3 消费后释放
L1[384,448)  vb[4]；BF16；S1 从 GM 搬入，S1 消费后释放
L1[448,512)  dw0[4]；BF16；S1 生成，保留至 S7
```

操作流程：

1. `vb` 从 Stage 0 的 GM 结果搬入 L1。先搬入当前 HEAD 的
   `du/h`；当前 HEAD 进入 MTE1/Cube 时，MTE2 将下一 HEAD 的
   `du/h` 提前搬入另一 resident，使输入搬运与当前矩阵乘重叠。
2. Cube 完成两条独立矩阵乘。L0A/L0B/L0C 各自使用
   独立 ping/pong，每发射一次 MMAD 就对应取反一次，不用同一
   slot 连续承载两条 GEMM。`dw0[4]` 写入 `L1[448,512)`；`dA_u[2]` 写入
   `UB[112,128)`。
3. `dA_u` 的核间握手只使用 mode=`0x2`，每个 HEAD 一组
   release/ready flag。两个 AIV 各 set release，AIC 聚合 wait 一次后，
   Fixpipe 用 `subBlockId` 只写 owner AIV；AIC set ready 一次，两个
   AIV 各 wait 一次后共同归还 release。
4. `vb` 完成末次 Cube 消费后释放。`h/du/dw0` 保留到 Stage 3；其中 `du`
   继续供 `dvb` 使用，不从 GM 重读。

### Stage 2：Vector，dA_u 预处理并提前计算 kb

```text
dA_u_lower = tril(dA_u, diagonal=-1)       # dA_u_lower: [BT,BT]
kb = k * beta[:,None]                      # kb: [BT,K]
```

空间布局（UB，单位 KiB）：

```text
UB[0,32)       k[2]；BF16；保留至 S15
UB[32,64)      v[2]；BF16；保留至 S6
UB[64,80)      gate_dA[2]；BF16；保留至 S12
UB[80,112)     dvb[2] 并发输出槽；BF16；S3 生成，保留至 S6
UB[112,128)    dA_u[2] -> dA_u_lower[2]；BF16；S1 写入，S2 原位处理后保留至 S4
UB[128,136)    dA_w0[HV0] 后续输出槽；BF16；S3 生成，保留至 S4
UB[136,144)    空闲；无数据；原 kbg bank 0 的后半段，S2 可复用
UB[144,152)    dA_w0[HV1] 后续输出槽；BF16；S3 生成，保留至 S4
UB[152,160)    空闲；无数据；原 kbg bank 1 的后半段，S2 可复用
UB[160,192)    kb[2]；BF16；S2 生成，写入 GM 后释放
UB[192,245.5)  空闲；无数据；S2 可复用
UB[245.5,246)  beta[2]；FP32；保留至 S12
UB[246,246.5)  g_exp[2]；FP32；保留至 S12
UB[246.5,247)  g_last[2]；FP32；保留至 S10
UB[247,247.5)  decay[2]；FP32；保留至 S12
UB[247.5,248)  bg[2]；FP32；保留至 S12
```

操作流程：

1. 本 Stage 读取 Stage 1 的 `dA_u`，因此必须在 Stage 1 完成后执行。单次 VF 完成
   `dA_u -> dA_u_lower` 和 `kb=k*beta`，不执行任何转置。
2. `kb[2]` 从 `UB[160,192)` 写入 GM；搬运完成后释放该 UB 临时地址，Stage 9 再搬入 L1。
3. Stage 3 从 GM 搬入 Stage 0 生成的 `kbg[4]`，不依赖 Stage 2 输出，
   因此 S2/S3 无执行顺序约束。S2 在 `UB[80,112)` 预留 S3 的 `dvb[2]`；
   `dA_w0[2]` 沿用原 `kbg[2]` 的两个 16 KiB bank 起点，分别使用
   `UB[128,136)` 和 `UB[144,152)`，不把两份 8 KiB 矩阵紧密压到同一 bank。

### Stage 3：Cube，dA_w0 与 dvb

```text
dA_w0 = dw0 @ kbg.T                        # dA_w0: [BT,BT]
dvb    = A @ du                             # dvb: [BT,V]
```

空间布局（L1，单位 KiB）：

```text
L1[0,32)     A[4]；BF16；S3 从 GM 搬入，原址保留至 S7
L1[32,128)   空闲；无数据；S3 可复用
L1[128,256)  h[4]；BF16；保留至 S13
L1[256,320)  du[4]；BF16；保留至 S3，消费后释放
L1[320,384)  kbg[4]；BF16；S3 从 GM 搬入，S3 消费后释放
L1[384,448)  空闲；kb 仍在 GM workspace，S9 再搬入
L1[448,512)  dw0[4]；BF16；保留至 S7
```

操作流程：

1. `dw0/du` 读取前序 L1，`kbg` 从 Stage 0 的 GM 结果搬入；Cube 将
   `kbg[BT,K]` 作为转置的 B 操作数读取，
   不生成实体转置 tensor。`A[4]` 在 Cube 路径中从 GM 一次搬入
   `L1[0,32)`，并原址保留到 Stage 7。
2. 两条矩阵乘彼此独立。Fixpipe 将两份 `dA_w0` 分别写入原 kbg bank 的
   `UB[128,136)`、`UB[144,152)`，将 `dvb[2]` 写入 `UB[80,112)`。
3. `du/kbg` 完成末次 Cube 消费后释放。

### Stage 4：Vector，dA0

```text
dA0 = dA_u_lower - tril(dA_w0, diagonal=-1) # dA0: [BT,BT]
```

空间布局（UB，单位 KiB）：

```text
UB[0,32)       k[2]；BF16；保留至 S15
UB[32,64)      v[2]；BF16；保留至 S6
UB[64,80)      gate_dA[2]；BF16；保留至 S12
UB[80,112)     dvb[2]；BF16；S3 写入，保留至 S6
UB[112,128)    dA_u_lower[2]；BF16；保留至 S4，消费后释放
UB[128,136)    dA_w0[HV0] -> dA0[HV0]；BF16；写入 GM 后释放
UB[136,144)    空闲；无数据；原 kbg bank 0 的后半段，S4 可复用
UB[144,152)    dA_w0[HV1] -> dA0[HV1]；BF16；写入 GM 后释放
UB[152,245.5)  空闲；无数据；S4 可复用
UB[245.5,246)  beta[2]；FP32；保留至 S12
UB[246,246.5)  g_exp[2]；FP32；保留至 S12
UB[246.5,247)  g_last[2]；FP32；保留至 S10
UB[247,247.5)  decay[2]；FP32；保留至 S12
UB[247.5,248)  bg[2]；FP32；保留至 S12
```

操作流程：

1. 单次 VF 读取完整 `dA_w0/dA_u_lower`，生成 `dA0`。
2. `dA0[2]` 在两个原 kbg bank 起点原位覆盖 `dA_w0[2]`，再写入 GM；
   释放 `UB[112,152)` 中本 Stage 使用的有效子区。

### Stage 5：Cube，dA1 与 a2

```text
dA1 = dA0 @ A                              # dA1: [BT,BT]
a2  = k @ k.T                              # a2: [BT,BT]
```

空间布局（L1，单位 KiB）：

```text
L1[0,32)     A[4]；BF16；保留至 S7
L1[32,96)    空闲；无数据；S5 可复用
L1[96,128)   dA0[4] -> dA1[4]；BF16；S5 从 GM 搬入 dA0，原址 Fixpipe dA1 并保留至 S7
L1[128,256)  h[4]；BF16；保留至 S11
L1[256,320)  k[4]；BF16；S5 搬入，保留至 S13
L1[320,384)  空闲；无数据；S5 可复用
L1[384,448)  空闲；kb 仍在 GM workspace，S9 再搬入
L1[448,512)  dw0[4]；BF16；保留至 S7
```

操作流程：

1. `dA0` 从 GM 一次搬入 `L1[96,128)`；`A` 读取 `L1[0,32)` 的前序保存数据。
   `k` 从 GM 一次搬入 `L1[256,320)` 并保留到 Stage 13；同一份 `A/k` L1 resident
   分别按 L1A/L1B layout 解释后送入两侧 L0，不创建副本。
2. 两条矩阵乘彼此独立。`dA0` 完成 MTE1 读取后，Fixpipe 将 `dA1[4]`
   原址写入 `L1[96,128)`；将 `a2[2]` 写入 `UB[128,144)`。
3. `A/dA1` 均保留到 Stage 7；Stage 6 不读取 `dA1`。

### Stage 6：Vector，提前完成 dv 与 db_v_partial

```text
dv = dvb * beta[:,None]                    # dv: [BT,V]
db_v_partial = rowSum(dvb * v)             # db_v_partial: [BT]
```

空间布局（UB，单位 KiB）：

```text
UB[0,32)       k[2]；BF16；保留至 S15
UB[32,64)      v[2]；BF16；保留至 S6，消费后释放
UB[64,80)      gate_dA[2]；BF16；保留至 S12
UB[80,112)     dvb[2] -> dv[2]；BF16；dvb 保留至 S6，dv 写 GM 后释放
UB[112,128)    空闲；无数据；S6 可复用
UB[128,144)    a2[2]；BF16；S5 写入，保留至 S8
UB[144,245)    空闲；无数据；S6 可复用
UB[245,245.5)  db_v_partial[2]；FP32；S6 生成，保留至 S8
UB[245.5,246)  beta[2]；FP32；保留至 S12
UB[246,246.5)  g_exp[2]；FP32；保留至 S12
UB[246.5,247)  g_last[2]；FP32；保留至 S10
UB[247,247.5)  decay[2]；FP32；保留至 S12
UB[247.5,248)  bg[2]；FP32；保留至 S12
```

操作流程：

1. 单次 VF 完成 `db_v_partial` 和 `dvb -> dv`，不读取或处理 `dA1`。
2. `dv` 写入 GM，`db_v_partial` 保留在 UB 到 Stage 8，释放 `v/dvb` 地址。
   Stage 7 使用独立的 `UB[192,240)` 输出区，与本 Stage 无执行顺序约束。

### Stage 7：Cube，dA2 与 dkbg0

```text
dA2 = A @ dA1                              # dA2: [BT,BT]
dkbg0 = A @ dw0                            # dkbg0: [BT,K]
```

空间布局（L1，单位 KiB）：

```text
L1[0,32)     A[4]；BF16；保留至 S7，消费后释放
L1[32,96)    空闲；无数据；S7 可复用
L1[96,128)   dA1[4]；BF16；S5 Fixpipe 写入，S7 消费后释放
L1[128,256)  h[4]；BF16；保留至 S11
L1[256,320)  k[4]；BF16；保留至 S13
L1[320,384)  空闲；无数据；S7 可复用
L1[384,448)  空闲；kb 仍在 GM workspace，S9 再搬入
L1[448,512)  dw0[4]；BF16；保留至 S7，消费后释放
```

操作流程：

1. 两条矩阵乘只共享只读 `A`，彼此不依赖。
2. Fixpipe 将 `dA2[2]` 写入 `UB[192,208)`，将 `dkbg0[2]` 写入
   `UB[208,240)`；两块均不与 Stage 6 的 UB 数据重叠。
3. `dA1/A/dw0` 完成末次 Cube 消费后释放对应 L1 地址。

### Stage 8：Vector，prepare G 完整合成

```text
dA = tril(-(dA2 * gate_dA), diagonal=-1)   # dA: [BT,BT]

AdA = dA * (a2 * beta[:,None])             # AdA: [BT,BT]
dg_A = rowSum(AdA) - colSum(AdA)           # dg_A: [BT]

dg0 = -rowSum(dkbg0 * k * bg[:,None])       # dg0: [BT]
db0 = -rowSum(dkbg0*k*g_exp[:,None])       # db0: [BT]
db_v = db0 + db_v_partial                  # db_v: [BT]
dg_prepare = dg0 + dg_A                    # dg_prepare: [BT]
```

空间布局（UB，单位 KiB）：

```text
UB[0,32)       k[2]；BF16；保留至 S15
UB[32,64)      空闲；无数据；S8 可复用
UB[64,80)      gate_dA[2]；BF16；S0 生成，保留至 S12
UB[80,112)     空闲；无数据；S8 可复用
UB[112,128)    空闲；无数据；S8 可复用
UB[128,144)    a2[2]；BF16；保留至 S8，消费后释放
UB[144,192)    空闲；无数据；S8 可复用
UB[192,208)    dA2[2] -> dA[2]；BF16；dA2 保留至 S8，dA 写入 GM 后释放
UB[208,240)    dkbg0[2]；BF16；S7 写入，保留至 S12
UB[240,244.5)  空闲；无数据；S8 可复用
UB[244.5,245)  db_v_partial[2] -> db_v[2]；FP32；db_v 保留至 S12
UB[245,245.5)  dg_A[2] -> dg_prepare[2]；FP32；dg_prepare 保留至 S14
UB[245.5,246)  beta[2]；FP32；保留至 S12
UB[246,246.5)  g_exp[2]；FP32；保留至 S12
UB[246.5,247)  g_last[2]；FP32；保留至 S10
UB[247,247.5)  decay[2]；FP32；保留至 S12
UB[247.5,248)  bg[2]；FP32；保留至 S12
```

操作流程：

1. 一次 VF 允许读取本 Stage 前序向量公式的结果，因此依次完成上述全部公式。
2. `dA[2]` 写入 GM，供 Stage 9 搬入 L1；`kb[2]` 已由 Stage 2 写入
   GM workspace，本 Stage 不重复计算或搬运，Stage 9 再搬入 `L1[384,448)`。
3. `gate_dA` 和 `dkbg0` 保持首次分配的物理地址不变。保留
   `k/gate_dA/dkbg0/bg/db_v/dg_prepare`，释放 `dA/a2` 的 UB 地址。

### Stage 9：Cube，prepare K 矩阵乘

```text
dkb   = dA @ k                             # dkb: [BT,K]
dkb_t = dA.T @ kb                          # dkb_t: [BT,K]
```

空间布局（L1，单位 KiB）：

```text
L1[0,96)     空闲；无数据；S9 可复用
L1[96,128)   dA[4]；BF16；S9 从 GM 搬入，S9 消费后释放
L1[128,256)  h[4]；BF16；保留至 S11
L1[256,320)  k[4]；BF16；保留至 S13
L1[320,384)  空闲；无数据；S9 可复用
L1[384,448)  kb[4]；BF16；S9 从 GM 搬入，S9 消费后释放
L1[448,512)  空闲；无数据；S9 可复用
```

操作流程：

1. 两条矩阵乘只共享只读 `dA`，彼此不依赖。第二条使用恒等式
   `(kb.T @ dA).T = dA.T @ kb`，只保留左操作数转置语义，不对 Cube 结果转置。
2. 令当前 `(chunk,task-group)` 的临时区基址为 `W9`：`dkb[2]` 写入
   `GM[W9+0,W9+32 KiB)`，`dkb_t[2]` 写入 `GM[W9+32 KiB,W9+64 KiB)`。
3. `dA/kb` 完成末次 Cube 消费后释放 L1 地址。

### Stage 10：Vector，state 归约

```text
state_term = sum(h * dh) * g_last          # state_term: scalar per HV
```

空间布局（UB，单位 KiB）：

```text
UB[0,32)       k[2]；BF16；保留至 S15
UB[32,64)      h[0]；BF16；S10 搬入，S10 归约后释放
UB[64,80)      gate_dA[2]；BF16；保留至 S12
UB[80,112)     h[1]；BF16；S10 搬入，S10 归约后释放
UB[112,144)    dh[0]；BF16；S10 搬入，S10 归约后释放
UB[144,176)    dh[1]；BF16；S10 搬入，S10 归约后释放
UB[176,192)    ds0[2] 并发输出槽；BF16；S11 生成，保留至 S12
UB[192,208)    空闲；无数据；S10 可复用
UB[208,240)    dkbg0[2]；BF16；保留至 S12
UB[240,244.5)  空闲；无数据；S10 可复用
UB[244.5,245)  db_v[2]；FP32；保留至 S12
UB[245,245.5)  dg_prepare[2]；FP32；保留至 S14
UB[245.5,246)  beta[2]；FP32；保留至 S12
UB[246,246.5)  g_exp[2]；FP32；保留至 S12
UB[246.5,247)  g_last[2] -> state_term[2]；FP32；g_last 保留至 S10，state_term 保留至 S14
UB[247,247.5)  decay[2]；FP32；保留至 S12
UB[247.5,248)  bg[2]；FP32；保留至 S12
```

操作流程：

1. `h/dh` 在 Vector 路径中各从 GM 完整搬入一次。S10 的四份 32 KiB
   slot 不覆盖 S8 的 MTE3 源区，因此不等待 S8 的 `MTE3_MTE2`；复用区只按
   `V_MTE2` 生命周期在 S0、S8、S10 之间闭环。
2. 单次 VF 完成完整归约，`state_term[2]` 覆盖 `g_last` 子槽；释放 `h/dh` UB。

### Stage 11：Cube，DS 基础矩阵乘

```text
ds0 = do @ v_new.T                         # ds0: [BT,BT]
```

空间布局（L1，单位 KiB）：

```text
L1[0,32)     do 当前 Stage 输入[2]；BF16；S11 从 GM 搬入，S11 结束后释放
L1[32,128)   空闲；无数据；S11 可复用
L1[128,256)  h[4]；BF16；保留至 S13
L1[256,320)  k[4]；BF16；保留至 S13
L1[320,384)  v_new[4]；BF16；S11 搬入，S11 消费后释放
L1[384,512)  空闲；无数据；S11 可复用
```

操作流程：

1. `v_new` 从 GM 一次搬入 `L1[320,384)`；`do` 一次搬入 `L1[0,32)`，Cube
   完成 `ds0=do@v_new.T`。
2. `ds0[2]` 由 Fixpipe 直接写入 `UB[176,192)`。该地址不与无依赖的 S10 活跃
   数据重叠，因此不经过 GM workspace。
3. `v_new` 完成 Cube 路径末次消费后释放；`h/k` 原地址保留到 Stage 13。

### Stage 12：Vector，prepare K、DS 后处理与 Cube 输入预处理

```text
dk_prepare_hv = -dkbg0 * bg[:,None]
                + dkb * beta[:,None] + dkb_t           # dk_prepare_hv: [BT,K]
db_prepare = db_v + rowSum(dkb * k)                    # db_prepare: [BT]

ds = tril(ds0 * gate_dA) * scale                       # ds: [BT,BT]
do_g = do * g_exp[:,None] * scale                      # do_g: [BT,V]
v_decay = v_new * decay[:,None]                        # v_decay: [BT,V]
```

空间布局（UB，单位 KiB）：

```text
UB[0,32)       k[2]；BF16；保留至 S15
UB[32,64)      dkb[2]；BF16；S12 从 GM 搬入，S12 消费后释放
UB[64,80)      gate_dA[2]；BF16；S0 生成，S12 消费后释放
UB[80,112)     dkb_t[2]；BF16；S12 从 GM 搬入，S12 消费后释放
UB[112,144)    do[2] -> do_g[2]；BF16；S12 从 GM 搬入并原位生成，写入 GM 后释放
UB[144,176)    v_new[2] -> v_decay[2]；BF16；S12 从 GM 搬入并原位生成，写入 GM 后释放
UB[176,192)    ds0[2] -> ds[2]；BF16；S11 写入并由 S12 原位生成，写入 GM 后释放
UB[192,208)    空闲；无数据；S12 可复用
UB[208,240)    dkbg0[2] -> dk_prepare_hv[2]；BF16；S12 原位生成，dk_prepare_hv 保留至 S14
UB[240,244)    空闲；无数据；S12 可复用
UB[244,244.5)  beta_raw[2] -> dbeta[2]；FP32；S12 搬入，写 GM 后释放
UB[244.5,245)  db_v[2] -> db_prepare[2]；FP32；S12 生成 dbeta 后释放
UB[245,245.5)  dg_prepare[2]；FP32；保留至 S14
UB[245.5,246)  beta[2]；FP32；保留至 S12，消费后释放
UB[246,246.5)  g_exp[2]；FP32；保留至 S12，消费后释放
UB[246.5,247)  state_term[2]；FP32；保留至 S14
UB[247,247.5)  decay[2]；FP32；S12 消费后释放
UB[247.5,248)  bg[2]；FP32；S12 消费后释放
```

操作流程：

1. `dkb/dkb_t/beta_raw/do/v_new` 各完整搬入一次，`ds0` 读取 S11 的固定 UB 地址，
   所有输入在一次 VF 前到齐。
   `do/v_new` 在 Vector 路径中首次且仅此次从 GM 读取；Cube 路径的独立读取不计为
   Vector 路径重复搬运。
2. 一次 VF 完成全部公式；`dk_prepare_hv` 原位覆盖 `dkbg0`，`do_g/v_decay` 分别
   原位覆盖 `do/v_new`。`db_prepare` 完成 beta backward 后将 `dbeta` 写入 GM。
3. `ds[2]` 原位覆盖 `ds0[2]`；`ds/do_g/v_decay` 分别写入 GM workspace，
   Stage 13 再搬入 `L1[96,128)`、`L1[320,384)` 和 `L1[384,448)`，
   展开为四份固定地址。
   Stage 12 是
   `gate_dA/bg/decay` 的最后一个消费者，完成后释放三者；保留
   `k/dk_prepare_hv/dg_prepare/state_term`；这些数据均保持原物理地址。

### Stage 13：Cube，DQ/DK 最终矩阵乘

```text
L0C_dq  = do_g @ h                         # L0C_dq: [BT,K]
L0C_dq += ds @ k                           # L0C_dq: [BT,K]
dq_hv   = Fixpipe(L0C_dq)                  # dq_hv: [BT,K]

dk_intra = ds.T @ q                        # dk_intra: [BT,K]
dk_base  = v_decay @ dh                    # dk_base: [BT,K]
```

空间布局（L1，单位 KiB）：

```text
L1[0,32)     q 当前 Stage 输入[2]；BF16；S13 从 GM 搬入，S13 结束后释放
L1[32,96)    dh 当前 Stage 输入[2]；BF16；S13 从 GM 搬入，S13 结束后释放
L1[96,128)   ds[4]；BF16；S13 从 GM 搬入，S13 消费后释放
L1[128,256)  h[4]；BF16；保留至 S13，消费后释放
L1[256,320)  k[4]；BF16；保留至 S13，消费后释放
L1[320,384)  do_g[4]；BF16；S13 从 GM 搬入，S13 消费后释放
L1[384,448)  v_decay[4]；BF16；S13 从 GM 搬入，S13 消费后释放
L1[448,512)  空闲；无数据；S13 可复用
```

操作流程：

1. Stage 13 整体依赖 Stage 12 的 `ds/do_g/v_decay`。`q/dh/ds/do_g/v_decay`
   在 Cube 路径中各从 GM 完整搬入一次；`h/k` 读取固定 L1 地址。
2. DQ 使用同一块 L0C：第一次 MMAD 执行 `do_g@h`，第二次 MMAD 以 accumulate
   方式执行 `ds@k`。两次 MMAD 完成后只执行一次 Fixpipe，将当前 HV 的 `dq_hv`
   写入 `W13_DQ`；中间 L0C 结果不落 UB，也不执行第二次 Fixpipe。
3. `dk_base=v_decay@dh` 与 `dk_intra=ds.T@q` 使用独立 L0C，各自只执行一次
   Fixpipe，分别写入 `W13_DK_BASE` 和 `W13_DK_INTRA`。Stage 13 不写 UB，因此不会
   覆盖仍由另一 AIV 使用的 Stage 12 `dkb/dkb_t/ds` 临时槽。
4. 释放全部 L1 数据。

Stage 13 workspace 使用三段逐 HV 固定区域：

```text
HV_VECTOR_BYTES = BT * K * sizeof(BF16) = 16 KiB
W13_DQ       = W2
W13_DK_BASE  = W3
W13_DK_INTRA = ALIGN_UP(workspace_used_before_W13, 512)

dq_hv[task,hv]   -> W13_DQ + (task * HV + hv) * HV_VECTOR_BYTES
dk_base[task,hv]  -> W13_DK_BASE + (task * HV + hv) * HV_VECTOR_BYTES
dk_intra[task,hv] -> W13_DK_INTRA + (task * HV + hv) * HV_VECTOR_BYTES

W13_DK_INTRA_BYTES = taskNum * HV * HV_VECTOR_BYTES
```

`W2/W3` 的前序语义均已在 Stage 13 前完成末次消费。`W13_DK_INTRA` 是新增的独立
workspace 尾部区域，不与任何前序 workspace 复用，总 workspace 明确增加
`ALIGN_UP(taskNum * HV * 16 KiB, 512 Byte)`。每个 HEAD 的 Stage 13 只依赖
该 HEAD 的 Stage 12 `ds/do_g/v_decay` 已写回并搬入对应 L1 地址；owner AIV 为当前
HEAD 发布普通 ready 后，AIC 即可消费该 HEAD。这里不再要求两个 AIV 完成整组
Stage 12，不使用 `mode=0x2` 聚合建立整组边界，也不按 taskCount 补发空 ready。

### Stage 14：Vector，逐 HV DQ/DK partial 与 DG finalize

```text
dg_chunk_partial = rowSum(dq_hv * q)       # dg_chunk_partial: [BT]
dg_chunk_partial[M-1] += state_term        # dg_chunk_partial: [BT]

dk_base_dot = rowSum(dk_base * k)          # dk_base_dot: [BT]
dk_hv = dk_base + dk_intra                 # dk_hv: [BT,K]
dg_chunk = dg_chunk_partial - dk_base_dot
           - rowSum(dk_intra * k)          # dg_chunk: [BT]
dg_chunk[M-1] += sum(dk_base_dot)          # dg_chunk: [BT]
dk_raw_hv = dk_hv + dk_prepare_hv          # dk_raw_hv: [BT,K]

x[t] = dg_chunk[t] + dg_prepare[t]          # x: [BT]
dg[t] = sum(x[j], j=t..M-1)                 # dg: [BT]
```

空间布局（UB，单位 KiB）：

```text
UB[0,32)       k[2]；BF16；保留至 S14，消费后释放
UB[32,64)      dk_base[2] -> dk_hv[2] -> dk_raw_hv[2]；BF16；S14 按当前 HV 从 W13_DK_BASE 搬入，先生成 dk_base_dot 再原位覆盖，写 W14 后释放
UB[64,96)      dq_hv[2]；BF16；S14 按当前 HV 从 W13_DQ 搬入，写 W14 后释放
UB[96,112)     空闲；无数据；S14 可复用
UB[112,144)    q[2]；BF16；S14 从 GM 搬入，S14 消费后释放
UB[144,176)    dk_intra[2]；BF16；S14 按当前 HV 从 W13_DK_INTRA 搬入，S14 消费后释放
UB[176,208)    空闲；无数据；S14 可复用
UB[208,240)    dk_prepare_hv[2]；BF16；保留至 S14，消费后释放
UB[240,244)    空闲；无数据；S14 可复用
UB[244,244.5)  dk_base_dot[2]；FP32；S14 生成，供 dg_chunk 和尾元素补偿读取，S14 后释放
UB[244.5,245)  dg_prepare[2]；FP32；保留至 S14，消费后释放
UB[245,245.5)  dg_chunk_partial[2] -> dg_chunk[2] -> x[2] -> dg[2]；FP32；S14 依次原位生成，写正式输出后释放
UB[245.5,246)  state_term[2]；FP32；保留至 S14，消费后释放
UB[246,248)    空闲；无数据；S14 可复用
```

Stage 14 大型 BF16 数据占用 192 KiB，FP32 小向量占用 2 KiB，完整峰值为
194 KiB，剩余 54 KiB。所有跨 Stage tensor 保持原物理地址；本 Stage 仅在源数据
完成末次读取后进行上述原位语义覆盖。

操作流程：

1. `q` 在 Vector 路径中按当前 HK 完整搬入；`dq_hv/dk_base/dk_intra` 按当前 HV
   分别从 W13 三段 workspace 搬入对应固定 UB 槽。同一 AIV 连续处理同一 HK 的多个
   HV 时，允许复用已经驻留的 `q`，不得重复从 GM 搬运。
2. 每个 HV 只调用一次 VF。VF 先从尚未覆盖的 `dk_base` 生成 `dk_base_dot`，再执行
   `dk_base -> dk_hv -> dk_raw_hv` 原位覆盖；随后依次完成
   `dg_chunk_partial -> dg_chunk -> x -> dg`，并在同一次 VF 中完成 reverse suffix sum。
   本 Stage 不执行 DQ/DK GVA 规约。`dk_base_dot` 只计算一次，供 `dg_chunk` 和尾元素
   补偿共同读取。
3. 令 GM 临时区基址为 `W14`：每个 HV 的 `dq_hv/dk_raw_hv` 写入各自固定 workspace
   槽，供 Stage 15 活跃核规约；最终 `dg` 由 Stage 14 直接写正式输出，不进入 workspace。
4. `q/k` 在 Stage 14 完成末次消费后释放。Q/K norm 开启时，Stage 15 核0按当前 HK
   从 GM 重新读取一次 `q/k`；这是改变 Stage 15 核归属后的容量 fallback。

Stage 14 workspace 复用前序 Stage 已完成末次消费的区域，不增加总 workspace：

```text
HV_VECTOR_BYTES = BT * K * sizeof(BF16) = 16 KiB
W14_DQ = W2
W14_DK = W3

dq_hv[task,hv]     -> W14_DQ + (task * HV + hv) * HV_VECTOR_BYTES
dk_raw_hv[task,hv] -> W14_DK + (task * HV + hv) * HV_VECTOR_BYTES
```

Stage 14 从 `W13_DQ` 读取 `dq_hv` 后原址写回相同值，因此 `W2` 地址语义不变；从
`W13_DK_BASE` 读取 `dk_base` 并完成末次消费后，才将同一 `W3` 槽覆盖为
`dk_raw_hv`。同一 W3 槽必须按 `MTE2 read complete -> VF complete -> MTE3 overwrite`
闭环，禁止 MTE3 在 MTE2 读完 `dk_base` 前覆盖。`W13_DK_INTRA` 在 Stage 14 读取后
结束生命周期。

Stage 14 双缓冲与事件协议：

1. 两个 AIV 继续按任务序号交错承包完整 HV；每个 AIV 的 `streamSlot` 只在自己
   承包的 HV 之间轮转。Stage 13 为当前 owner HV 发出 ready 后，owner 才能消费
   当前 slot 中的 `dk_base/dq_hv/dk_intra`。
2. 当前 slot 执行 VF 时，MTE2 可使用另一 slot 的独立 `MTE2_V` EventID 搬入该
   AIV 的下一个交错 HV 所需 `q`。每次搬运仍只对应一个 HV；
   没有下一个 owner HV 时不发射空搬运。
3. 当前 HV 的所有输入到齐后只调用一次 VF。VF 先生成 `dk_base_dot`，再原位生成
   `dk_hv/dk_raw_hv/dg_chunk/x/dg`；
   不拆 pass，不分 tile，不在 VF 外重复向量公式。
4. VF 完成后通过 `V_MTE3` 保证 MTE3 只读取已生成的数据。MTE3 将更新后的
   `dq_hv/dk_raw_hv` 写 W14，并将 `dg` 写正式输出；随后为
   当前 slot 发布 `MTE3_MTE2`。下一任务组 Stage 0 覆盖该物理 slot 前必须等待
   此反向事件，不能只依赖 Stage 13/14 的 stage 编号或核间 ready。
5. Stage 14 不执行任何 DQ/DK 规约。Stage 15 核0消费某个 HV partial 前只等待该 HV
   写回完成；写最终 accumulator 前必须确认对应 HK 的 G 个 HV 均已累加完成。

### Stage 15：Vector，分核 GVA 规约与 Q/K norm backward

```text
# 两个 AIV 处理相同 HEAD/HV 集合的不同 token half
ACC_CAP_PER_CORE = 4                              # 物理固定预留
row_begin = core_id * 32
row_end = min(row_begin + 32, M)
group_hv = hv_i - task_group_hv_start
acc_id = group_hv / G                             # G=1/2/3/4 -> 4/2/1/1 个有效 acc

dq_acc[acc_id] += dq_hv[row_begin:row_end]        # dq_acc: [4,32,K]
dk_acc[acc_id] += dk_raw_hv[row_begin:row_end]    # dk_acc: [4,32,K]

dq_chunk = dq_acc[acc_id]                         # dq_chunk: [32,K]
dk_chunk = dk_acc[acc_id]                         # dk_chunk: [32,K]

if use_qk_l2norm_in_kernel:
    dq = dq_chunk * q_rstd[:,None]
         - rowSum(dq_chunk*q)[:,None] * q * q_rstd[:,None] # dq: [BT,K]
    dk = dk_chunk * k_rstd[:,None]
         - rowSum(dk_chunk*k)[:,None] * k * k_rstd[:,None] # dk: [BT,K]
else:
    dq = dq_chunk                                      # dq: [BT,K]
    dk = dk_chunk                                      # dk: [BT,K]

```

核0空间布局（UB，单位 KiB；token `[0,32)`）：

```text
UB[0,32)       dq_acc_half[4,32,K] -> dq_half[4,32,K]；BF16；仅前 TG/G 份有效
UB[32,64)      dk_acc_half[4,32,K] -> dk_half[4,32,K]；BF16；仅前 TG/G 份有效
UB[64,72)      dq_hv_half 当前输入[32,K]；BF16；从 W14_DQ 搬入，累加后复用
UB[72,80)      dk_raw_hv_half 当前输入[32,K]；BF16；从 W14_DK 搬入，累加后复用
UB[80,88)      q_half 当前 accumulator[32,K]；BF16；norm 开启时从 GM 搬入
UB[88,96)      k_half 当前 accumulator[32,K]；BF16；norm 开启时从 GM 搬入
UB[96,244)     空闲；无数据；S15 核0可复用
UB[244,244.125) q_rstd_half[32]；FP32；norm 开启时从 GM 搬入
UB[244.125,244.25) k_rstd_half[32]；FP32；norm 开启时从 GM 搬入
UB[244.25,248) 空闲；无数据；S15 核0可复用
```

核1空间布局（UB，单位 KiB；token `[32,M)`，所有 G）：

```text
UB[0,32)       dq_acc_half[4,32,K] -> dq_half[4,32,K]；BF16；仅前 TG/G 份有效
UB[32,64)      dk_acc_half[4,32,K] -> dk_half[4,32,K]；BF16；仅前 TG/G 份有效
UB[64,72)      dq_hv_half 当前输入[32,K]；BF16；从 W14_DQ 搬入，累加后复用
UB[72,80)      dk_raw_hv_half 当前输入[32,K]；BF16；从 W14_DK 搬入，累加后复用
UB[80,88)      q_half 当前 accumulator[32,K]；BF16；norm 开启时从 GM 搬入
UB[88,96)      k_half 当前 accumulator[32,K]；BF16；norm 开启时从 GM 搬入
UB[96,244)     空闲；无数据；S15 核1可复用
UB[244,244.125) q_rstd_half[32]；FP32；norm 开启时从 GM 搬入
UB[244.125,244.25) k_rstd_half[32]；FP32；norm 开启时从 GM 搬入
UB[244.25,248) 空闲；无数据；S15 核1可复用
```

操作流程：

1. 两个 AIV 对完全相同的 task group HEAD/HV 顺序执行；核0只读取每个 partial 的前
   32 行，核1只读取后 32 行。每核固定保留 4 份 half-row DQ accumulator 和 4 份
   half-row DK accumulator，`G=1/2/3/4` 分别使用前 `4/2/1/1` 份。
2. 当前 HK 的首个 HV 直接初始化对应 half accumulator，后续 HV 只搬入当前 token
   half 并原位累加。两核不读写彼此 UB，也不需要跨核合并 accumulator。
3. 当前 task group 的全部 HV 遍历完成后，每个 AIV分别按 accumulator 顺序处理自己的
   token half 输出。
   `use_qk_l2norm_in_kernel=True` 时，每次为当前 accumulator 对应 HK 从 GM 搬入一次
   `q/k/q_rstd/k_rstd`，完成 norm 后写 `dq/dk`；关闭时 accumulator 直接写最终输出。
4. 每核固定峰值：8 个 half accumulator 共 64 KiB，当前 DQ/DK half 输入共 16 KiB，
   当前 `q/k` half 共 16 KiB，两个 half rstd 共 0.25 KiB，总计 96.25 KiB。关闭 norm
   时实际活跃数据为 80 KiB，但地址布局不改变。
5. 两个 AIV 分别维护独立的 MTE2_V/V_MTE3/MTE3_MTE2 事件闭环。`M<=32` 时核1不
   发射空指令，也不参与完成计数；否则两个核分别写最终 `dq/dk` 的不重叠 token 半区。

关闭 `use_qk_l2norm_in_kernel` 时：

```text
dq = dq_chunk                                   # dq: [BT,K]
dk = dk_chunk                                   # dk: [BT,K]
```

beta sigmoid backward：

```text
s     = sigmoid(beta_raw)                     # s: [BT]
coef  = 2 if allow_neg_eigval else 1
dbeta = db_prepare * s * (1-s) * coef         # dbeta: [BT]
```

关闭 `use_beta_sigmoid_in_kernel` 时：

```text
dbeta = db_prepare                            # dbeta: [BT]
```

固定 DG finalize 语义：

```text
dg = suffixSum(dg_chunk + dg_prepare)          # dg: [BT]
```

本算子不实现 GDN gate 激活 backward，不接收 `g_input/A_log/dt_bias`，也不产生
`dA_log/ddt_bias`。

## 5. 新 Stage 资源分配方案

本案例按照 [`03-solution-design.md`](../../03-solution-design.md) 的 `R01`–`R21` 完成
Stage 划分、逐 Stage 详设和资源核算。规则正文统一由 `03-solution-design.md` 维护；
下文只保留本算子的具体推导和证据。

完整 DAG 核算中，S10 的全部 UB 活跃数据已占 211.5 KiB。无依赖的 S9 若将
`dkb/dkb_t` 直接写 UB，需要额外 64 KiB，与 S10 的并发集合为 275.5 KiB，超过
完整 248 KiB UB，因此 S9 使用 `R14` 的 GM 中转写 `W9`。S11 现在只输出 16 KiB
`ds0`，与 S10 的并发集合为 227.5 KiB，可直接写入 `UB[176,192)`；S12 原位消费，
不再使用 `W11`。S10 与 S9/S11 不增加执行顺序约束。

### 5.1 Stage 序列

```text
S0  Vector : gate/beta 系数、kbg、vb
S1  Cube   : dw0、dAu
S2  Vector : dAuLower、kb
S3  Cube   : dAw0、dvb
S4  Vector : dA0
S5  Cube   : dA1、a2
S6  Vector : dv、dbVPartial
S7  Cube   : dA2、dkbg0
S8  Vector : dA、dbV、dgPrepare
S9  Cube   : dkb、dkbT
S10 Vector : stateTerm
S11 Cube   : ds0
S12 Vector : dkPrepare、dbPrepare、ds、doG、vDecay
S13 Cube   : dqHv（L0C 累加）、dkBase、dkIntra
S14 Vector : 逐 HV dq/dk partial、DG reverse cumsum 与最终 dg
S15 Vector : 所有 G 均双核分 token half 做 GVA 规约与可选 Q/K norm backward
```

Cube Stage 输入依赖检查：

```text
S1  dw0  <- du,h          dAu <- du,S0.vb
S3  dAw0 <- S1.dw0,S0.kbg  dvb <- A,S1.du
S5  dA1  <- S4.dA0,A      a2  <- k,k
S7  dA2 <- A,S5.dA1       dkbg0 <- A,S1.dw0
S9  dkb  <- S8.dA,k       dkbT <- S2.kb,S8.dA
S11 ds0 <- do,v_new
S13 dqHv <- S12.doG,h,S12.ds,k   dkBase <- S12.vDecay,dh   dkIntra <- S12.ds,q
```

同一行中的多个矩阵乘只共享只读输入，互不读取本 Stage 产生的结果；存在结果依赖的
`dw0 -> dAw0`、`dA0 -> dA1 -> dA2` 已拆到不同 Stage。当前 16 个 Stage 从 S0
到 S15；S0--S14 按 Vector/Cube 交替，S14/S15 为一组连续 Vector。两者不能合并：
S15 核0按 HV ready 顺序逐份消费 Stage 14 partial，并累加到对应 HK accumulator；
只有写最终 `dq/dk` 前才需要确认该 accumulator 的 G 个 HV 全部完成。`dg` 的 reverse
cumsum 已融合在 Stage 14 当前 HV 的同一次 VF 中并直接写正式输出。S14 承担逐 HV
公式、DG finalize 和写回，S15 只承担规约/norm，连续 Vector Stage 的任务已经
最小化。S2 前移 `kb`，并用
`dA_u_lower` 建立对 S1 的依赖；S3 从 GM 搬入 S0 生成的 `kbg`，因此 S2/S3
无执行顺序约束，对应 Cube 输出槽已计入 S2 的 UB 并发集合。S5 将 `dA1`
直接保留在 L1 供 S7 使用；S6 完成 `dv/db_v_partial`，其 UB 地址与 S7 输出不重叠，
因此 S6 和 S7 没有执行顺序约束。负号延后到 S8 与 `gate_dA` 的乘法融合。
两个插入的 Vector 都承担真实公式，不是空 Stage。Stage 数仅在以上条件满足后再缩减。

本方案中，`dA_u/dA_w0/dvb/a2/dA2/dkbg0/ds0` 由 Fixpipe 写入两份 UB；
`dkb/dkb_t` 因并发峰值超限按 `R14` 经 `W9` 中转。`dw0/dA1` 作为 Cube 后继输入
保留在四份 L1。`kbg/vb/kb/dA0/dA/ds/do_g/v_decay` 由 Vector Stage 写入 GM，
再由各自的 Cube 消费 Stage 搬入 L1。`gate_dA/bg/k/v/state_term/dk_prepare_hv/dg_prepare`
保留在两份 UB，直到最后一个 Vector 消费者完成。

### 5.2 L1 全空间分配

```text
L1_total = 512 KiB
L1_fixed_reservation = 0 KiB
L1_available_per_cube_stage = 512 KiB
```

完整 `L1[0,512)` 均可存放跨 Stage 保存数据或当前 Cube Stage 输入。地址可随 Stage
改变语义；只有仍有后续 Cube 消费者的数据不可覆盖。凡是跨 Stage 保存的 tensor，
必须按最多 4 个 HV 预留 4 份容量。按 BF16 Cube 输入计算，各关键 Stage 的完整 L1
同时存活量为：

```text
S0 end  : 无 L1 驻留数据                                  = 0 KiB
S1 peak : h[4] + du[4] + vb[4] + dw0[4]                 = 320 KiB
S1 end  : h[4] + du[4] + dw0[4]                         = 256 KiB
S2 end  : h[4] + du[4] + dw0[4]                         = 256 KiB
S3 peak : h[4] + du[4] + kbg[4] + dw0[4] + A[4]        = 352 KiB
S3 end  : h[4] + A[4] + dw0[4]                         = 224 KiB
S4 end  : h[4] + A[4] + dw0[4]                         = 224 KiB
S5/S6   : h[4] + k[4] + A[4] + (dA0 -> dA1)[4] + dw0[4]
                                                        = 320 KiB
S7 end  : h[4] + k[4]                               = 192 KiB
S8/S9   : h[4] + k[4] + dA临时输入[4] + kb[4]      = 288 KiB
S9 end  : h[4] + k[4]                               = 192 KiB
S11 peak : h[4] + k[4] + v_new[4] + do 当前输入[2] = 288 KiB
S11 end : h[4] + k[4]                               = 192 KiB
S13 peak : h[4] + k[4] + doG[4] + vDecay[4] + ds[4] + q[2] + dh[2]
                                                        = 448 KiB
```

L1 完整峰值为 S13 的 448 KiB，剩余 64 KiB。不存在固定不可借用区域。

S3 中 `A[4]` 一次搬入 `L1[0,32)` 并原址保留到 S7。`k[4]` 在 S5 首次搬入
`L1[256,320)` 并保留到 S13。`kb[2]` 在 S2 写入 GM workspace，S9 再搬入
`L1[384,448)` 并在本 Stage 消费；`h` 从 S1 保留到 S13，
`A/dw0` 保留到 S7；`do_g/v_decay` 从 S12 保留到 S13。
`L1[96,128)` 作为短生命周期 Cube 输入区，严格按
`dA0(S5) -> dA1(S5--S7) -> dA(S9) -> ds(S13)`
的顺序复用。`dA1` 由 S5 原址 Fixpipe 生成，其余 tensor 在对应 Cube Stage 内搬入。该地址
不与下一任务组 S3 的 `kbg=L1[320,384)` 或 S1 的 `vb=L1[384,448)` 重叠。

任何原始输入在对应计算路径中
均只从 GM 读取一次。

#### 5.2.1 L1 生命周期与任务组边界检查

逐 Stage 检查后的关键 L1 生命周期如下：

| 地址 | tensor 生命周期 | 后续复用条件 |
|---|---|---|
| `L1[0,32)` | `A`：S3 写，S5/S7 读 | S7 完成后才可复用 |
| `L1[96,128)` | `dA0` 由 S5 从 GM 搬入并原址覆盖为 `dA1`，之后与 S9 搬入的 `dA`、S13 搬入的 `ds` 顺序复用 | 每个 tensor 的消费 Stage 完成后才可覆盖 |
| `L1[128,256)` | `h`：S1 写，S13 读 | S13 完成后才可复用 |
| `L1[256,320)` | `k`：S5 写，S13 读 | S13 完成后才可复用 |
| `L1[320,384)` | `kbg`：S3；`v_new`：S11；`do_g`：S13 | `kbg/do_g` 均由对应 Stage 从 GM workspace 搬入 |
| `L1[384,448)` | `vb`：S1；`kb`：S9；`v_decay`：S13 | 三者均由对应 Stage 从 GM workspace 搬入 |
| `L1[448,512)` | `dw0`：S1--S7 | S7 完成后才可复用 |

检查结论：同一任务组内部不存在仍未消费就被后续 Stage 覆盖的 L1 地址。短生命周期
Cube 输入写入 `L1[96,128)`，不会与下一任务组的 Cube 输入地址重叠。但长生命周期 resident 仍有意复用下一任务组的 S1/S3 地址，
例如 `v_new`、`kb` 和 `dw0`。因此当前单份 resident 设计只允许同一任务组完成
S0--S15 完整闭环后再启动下一任务组；若未来需要任务组间重叠执行，必须为所有跨组
冲突的 resident 增加独立 bank 或改用 GM workspace，不能只调整核间 flag。

### 5.3 UB 全空间分配

完整 UB 容量：

```text
UB_total = 248 KiB
UB_fixed_reservation = 0 KiB
UB_available_per_vector_stage = 248 KiB
```

当前布局中，大型 BF16 tensor 从 UB 低地址向上紧凑排列，FP32 小向量从
`UB[248)` 向下紧凑排列。`bg` 从 S0 保留到 S12，使用 `UB[247.5,248)`；
`gLast` 在 S10 完成末次消费后由 `stateTerm[2]` 原位覆盖。两端排布只决定首次
分配地址；任何跨 Stage 数据在生命周期内均保持该地址不变。地址只有在原数据完成
真实末次消费后才可分配给新数据。

每个跨 Stage Vector 中间量以及 Cube-to-Vector 结果，都按两个 HV 分配两份对应
shape 的 UB 空间。大型 tensor 按 BF16 保存，小向量和归约标量按 FP32 保存。
按依赖 DAG 计算的完整 UB 活跃集合为：

```text
S0 VF peak:
    S0 Vector 大型 tensor 144 KiB + FP32 小向量 3 KiB    = 147 KiB
S2 与 S3 并发峰值:
    gateDA[2] + k[2] + v[2] + dAuLower[2] + kb[2]
    + S3.dAw0[2] + S3.dvb[2]                             = 176 KiB
    FP32 小向量                                           = 2.5 KiB
    S2/S3 total                                           = 178.5 KiB
S3 -> S4:
    gateDA[2] + k[2] + v[2] + dAw0[2] + dAuLower[2]
    + dvb[2]                                              = 144 KiB
    FP32 小向量                                           = 2.5 KiB
    S4 total                                              = 146.5 KiB
S5 -> S6:
    gateDA[2] + k[2] + v[2] + dvb[2] + a2[2]             = 128 KiB
    FP32 小向量（含 dbVPartial）                           = 3 KiB
    S6 total                                              = 147 KiB
S7 -> S8:
    gateDA[2] + k[2] + dA2[2] + a2[2] + dkbg0[2]         = 112 KiB
    FP32 小向量                                           = 3.5 KiB
    S8 total                                              = 115.5 KiB
S10 与 S11 并发:
    S10 活跃数据 211.5 KiB + S11.ds0[2] 16 KiB          = 227.5 KiB
S12:
    k[2] + gateDA[2] + (dkbg0[2] -> dkPrepareHv[2])
    + dkb[2] + dkbT[2] + (ds0[2] -> ds[2])
    + (do[2] -> doG[2]) + (vNew[2] -> vDecay[2])         = 224 KiB
    FP32 小向量和 betaRaw[2]                              = 4 KiB
    S12 total                                             = 228 KiB
S14:
    k[2] + dkPrepareHv[2] + dqHv[2] + dkBase[2]
    + dkIntra[2] + q[2]                                  = 192 KiB
    FP32 小向量                                           = 2 KiB
    S14 total                                             = 194 KiB
S15 每个 AIV（固定 half-row 最悲观布局）:
    dqAccHalf[4] + dkAccHalf[4]                           = 64 KiB
    dqHvHalfCurrent[1] + dkRawHvHalfCurrent[1]            = 16 KiB
    qHalf[1] + kHalf[1]                                   = 16 KiB
    qRstdHalf[1] + kRstdHalf[1]                           = 0.25 KiB
    S15 per-AIV total（Q/K norm 开启）                     = 96.25 KiB
    S15 per-AIV active（Q/K norm 关闭）                    = 80 KiB
```

UB 完整峰值仍为 S12/S14 的 228 KiB，剩余 20 KiB。不存在固定
不可借用区域：

```text
UB_peak = 228 KiB
UB_free_at_peak = 248 KiB - 228 KiB = 20 KiB
```

UB 生命周期检查同样以完整任务组为边界。`gate_dA/k/v/bg` 等从前序 Stage 保留到
后续 Stage 的数据，在当前任务组完成真实末次消费前，不允许被下一任务组 S0 的
输入搬运覆盖。各 Stage 表内的原位覆盖只发生在源 tensor 已完成末次消费之后；
同一任务组内部未发现提前覆盖。当前 UB 也只有一份跨 Stage resident，因此任务组间
重叠执行的限制与 L1 相同：必须先扩充 resident bank 或将跨组数据落到 GM，不能
依赖双缓冲 EventID 或核间 flag 自动解决地址冲突。

每个 Vector Stage 在 VF 调用前一次性完成完整逻辑输入搬运，并通过一次 VF 完成全部
公式。当前 Stage 输入、临时量、跨 Stage 数据和输出共同计入 248 KiB 总量。

各 Stage 的 UB 绝对偏移已经写在对应“空间布局”中。`dkbg0` 由 S7 写入
`UB[208,240)` 并原址保留到 S12；S9 的 `dkb/dkb_t` 在 GM workspace 驻留，
只在 S12 调用 VF 前搬入 UB。S11 的 `ds0` 直接写 `UB[176,192)`。

Vector Stage 的完整 shape 结果按生命周期复用已消费的 UB 地址，例如
`dkbg0 -> dk_prepare_hv`、`dk_base -> dk_hv -> dk_raw_hv`。
S14 将 `dq_hv/dk_raw_hv` 作为 S15 活跃核规约输入写入 W14；`dg` 在 S14 直接写正式输出，
不进入 W14，也不在 UB 内长期保留；
重复使用的 `dkBaseDot[2]` 保存在 Stage 12 已释放的小向量地址中，行列归约结果使用
当前 Stage 布局中列出的 FP32 地址。

### 5.4 跨 Stage 数据保留

```text
dA_u/dA_w0/dvb/a2/dA2/dkbg0/ds0：Fixpipe -> UB[2] -> 后续 Vector Stage
dkb/dkb_t：Fixpipe -> W9 -> UB[2] -> S12
dw0/dA1：Fixpipe -> L1[4] -> 后续 Cube Stage
kbg/vb/kb/dA0/dA/ds/do_g/v_decay：UB -> GM workspace -> 后续 Cube Stage 的 L1
gate_dA/bg/k/v/state_term/dk_prepare_hv/dg_prepare：UB[2] 保留到末次 Vector 消费
最终输出：Vector 从 UB 写 GM
```

矩阵乘输入第一次从 GM 搬入 L1 后，只要后续 Cube 仍会使用，就必须保留在四份 L1
保存空间中；不允许因 stage 切换重复从 GM 搬运。Cube 结果若将由 Vector 使用，
优先落入两份 UB；仅当完整依赖 DAG 的并发活跃集合超过 UB 总容量时，
允许写一次 GM workspace，再由 Vector 完整读取一次。当前方案中的
`dkb/dkb_t` 使用该回退路径。

Vector 输入第一次从 GM 搬入 UB 后，只要后续 Vector Stage 仍会使用，就必须保留在
两份 UB；不允许因 stage 切换再次从 GM 搬运。当前方案中 `k` 在 S0--S15、
`v` 在 S0--S6、`beta` 在 S0--S12 的 Vector 生命周期内连续驻留；原始 `g` 在 S0
完成全部派生量后释放。
同一原始输入若同时参与 Cube 和 Vector，两条计算路径相互独立：允许各有一次
`GM -> L1` 和一次 `GM -> UB`，但任一路径内部均不得发生第二次 GM 读取。

Stage 切换时，完整 `UB[0,248)` 和 `L1[0,512)` 内已结束生命周期的地址均可改变
tensor 语义。任何复用都必须以真实末次消费者完成为前提。

### 5.5 `R01`–`R21` 检查表

| 规则 | 结论 | 本案例证据位置 |
|---|---|---|
| `R01` | 满足 | 5.1 的 Stage 序列和第 4 章各 Stage 类型 |
| `R02` | 满足 | 5.1 的 Cube 输入依赖检查和第 4 章逐 Stage 公式 |
| `R03` | 满足 | 第 2 章的 Ascend 950 范围，以及 5.2 的 L1 容量与峰值、5.3 的 UB 容量与峰值 |
| `R04` | 满足 | S1、S3、S5、S7、S11 的 Fixpipe 输出布局 |
| `R05` | 满足 | 第 4 章各 Vector resident 的地址和末次消费点 |
| `R06` | 满足 | S0、S2、S4、S8、S12 的 GM 写回及后续 Cube Stage 搬入 |
| `R07` | 满足 | 5.2 中 `dw0`、`dA1` 等 Cube resident 的四份 L1 规划 |
| `R08` | 满足 | 5.3 的逐 Stage UB 活跃集合、并发集合与峰值计算 |
| `R09` | 满足 | 5.2 的完整 L1 地址图、生命周期和四份 resident 规划 |
| `R10` | 满足 | 5.2、5.3 和 5.4 的 GM 搬运次数说明 |
| `R11` | 满足 | 5.3 的 UB 地址复用顺序与生命周期检查 |
| `R12` | 满足 | 第 4 章各 Vector Stage 的一次完整搬入和一次 VF 流程 |
| `R13` | 满足 | 5.2、5.3 中无依赖 Cube/Vector Stage 的并发峰值核算 |
| `R14` | 满足 | 第 5 章开头 S9 与 S10 并发达到 275.5 KiB 后使用 `W9` 的计算 |
| `R15` | 满足 | 5.1 对连续 Stage、真实插入任务和不可继续合并原因的说明 |
| `R16` | 满足 | S0、S8、S12、S14 的 Vector 中间量保留与复用 |
| `R17` | 满足 | 5.2、5.3 的绝对地址、末次消费、复用条件和连续空闲区 |
| `R18` | 满足 | 4.1 的 head/task 映射和第 4 章各 Stage 的统一存放规则 |
| `R19` | 满足 | 5.1 的 16 个 Stage 最小性说明 |
| `R20` | 满足 | 4.1 的 head 分工说明，以及 5.1 的任务映射 |
| `R21` | 满足 | 第 4 章各 Stage 的计算 dtype、输入与中间值范围、cast 和 mask 设计 |
