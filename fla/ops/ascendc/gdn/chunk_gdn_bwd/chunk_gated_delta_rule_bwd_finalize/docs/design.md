# GDN backward finalize 全融合 Ascend C 算子设计

方案设计规则版本：`V1`

## 1. 目标

本文设计一个覆盖 stage golden 全部计算的 GDN backward `finalize` 全融合算子。
算子完整包含：

```text
chunk_bwd_dqkwg
  -> prepare_wy_repr_bwd
  -> dk/dg 合成
  -> chunk 内 reverse cumsum
  -> Q/K L2Norm backward
  -> beta sigmoid backward
```

数学边界以
`tests/atk/chunk_gated_delta_rule_bwd_finalize/scripts/chunk_gated_delta_rule_bwd_finalize_cpu.py`
中经 GPU/Triton 对齐的 `chunk_gated_delta_rule_bwd_finalize_golden` 为唯一目标语义。
该 golden 覆盖从 `chunk_bwd_dqkwg` 到 `fused_beta_sigmoid_bwd` 的完整调用链，
并显式保留 GPU 路径中的 BF16 落盘边界。

实现结构参考 Ascend C
[`chunk_gated_delta_rule_bwd_dhu`](https://github.com/flashserve/flash-linear-attention-npu/tree/main/fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu)
的混合 AIC/AIV、定长/变长 offset helper、跨核同步和 workspace 管理方式，但采用不同的
逻辑分核：**只按 chunk 分核，不按 head 分核**。每个 chunk task 在核内遍历所有
`HK/HV` head。本版本仅支持 GVA 比例 `1:1`--`1:4`，即 `G=HV/HK in {1,2,3,4}`。
核内使用动态 HV 任务组：`G=3` 时组大小取 3，`G=1/2/4` 时取 4。对每个
任务组，两个 AIV 按任务序号交错承包完整 HV：AIV0 处理 0、2、4...，AIV1
处理 1、3、5...，不切分单个 HV 内部数据。两个 AIV 不设置 stage 级汇合；每个
AIV 完成自己承包的当前 stage 任务后即可独立进入下一 stage。只有存在真实的跨
AIV 数据依赖或归约时才增加对应同步。执行方式参考 `bwd_dhu` 的 head round，但
不固定为 4 个 HV。

本文 Stage 编号采用代码中的 S0--S15。数学公式和 dtype 落盘边界描述
GPU 对齐 golden 的目标语义；UB/L1 地址表、workspace 映射、Stage 顺序和
同步协议描述当前代码。第 4.1 节记录三个转置敏感 GEMM 的实现方式，
用于约束后续修改不得退回 row-major `A` 语义。

## 2. 范围与 shape

第一版目标：

```text
SoC               Ascend 950 only
q/k layout        [B, HK, T, K]
value/gate layout [B, HV, T]
state layout      [B, HV, NT, K, V]，state_v_first=true 时为 [B, HV, NT, V, K]
K                 128
V                 128
chunk_size        64
dtype             bf16，gate/beta 允许共同使用 fp32
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
| `q, k, dq, dk` | `[B,HK,T,K]` | q/k 是 forward 实际使用值；可选融合 L2Norm backward |
| `v, v_new, do, du, dv` | `[B,HV,T,V]` | value 侧输入和输出 |
| `g, beta, dbeta, dg` | `[B,HV,T]` | `g/beta` 为变换后值且 dtype 必须相同 |
| `beta_raw` | `[B,HV,T]` | beta sigmoid 变换前值，仅融合 sigmoid backward 时使用 |
| `h, dh` | `[B,HV,NT,K,V]` 或 `[B,HV,NT,V,K]` | 每 chunk state，末两维由 `state_v_first` 控制 |
| `A` | `[B,HV,T,BT]` | 每 token 行存一个 chunk 内矩阵行 |
| `q_rstd, k_rstd` | `[B,HK,T]` | L2Norm forward 保存值，fp32，仅融合 norm backward 时使用 |

golden 单 chunk 中将 `h/dh` 统一读取为逻辑 `[V,K]`，将 `A` 读取为逻辑
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
| `q` | `[B,HK,T,K]` | bf16 | forward 实际使用的 Q |
| `k` | `[B,HK,T,K]` | bf16 | forward 实际使用的 K |
| `v` | `[B,HV,T,V]` | 同 `q` | prepare backward |
| `v_new` | `[B,HV,T,V]` | 同 `q` | dqkwg value 输入 |
| `do` | `[B,HV,T,V]` | 同 `q` | 上游输出梯度 |
| `du` | `[B,HV,T,V]` | 同 `q` | 原始 value 梯度 |
| `g` | `[B,HV,T]` | bf16/fp32 | 变换后的 chunk gate |
| `beta` | `[B,HV,T]` | 同 `g` | sigmoid 后 beta |
| `h` | `[B,HV,NT,K,V]` | 同 `q` | forward chunk state |
| `dh` | `[B,HV,NT,K,V]` | 同 `q` | `bwd_dhu` 输出 |
| `A` | `[B,HV,T,BT]` | 同 `q` | WY inverse/中间矩阵 |
| `q_rstd` | `[B,HK,T]` | fp32 | 可选；Q L2Norm 保存值 |
| `k_rstd` | `[B,HK,T]` | fp32 | 可选；K L2Norm 保存值 |
| `beta_raw` | `[B,HV,T]` | 同 `g` | 可选；beta sigmoid 原始输入 |
| `cu_seqlens` | `[seqNum+1]` | int64 | 可选 varlen 序列边界 |
| `chunk_indices` | `[2*NT]` | int64 | 可选 `(seqIdx,localChunkIdx)` pair |

输出：

| 名称 | Shape | dtype | 说明 |
|---|---:|---|---|
| `dq, dk` | `[B,HK,T,K]` | 同 `q` | Q/K 梯度 |
| `dv` | `[B,HV,T,V]` | 同 `v` | V 梯度 |
| `dbeta` | `[B,HV,T]` | 同 `beta` | beta 梯度；Vector 内以 FP32 计算和驻留 |
| `dg` | `[B,HV,T]` | 同 `g` | gate 梯度；Vector 内以 FP32 计算和驻留 |

属性：

| 名称 | 类型 | 默认值 |
|---|---|---:|
| `scale` | float | `1/sqrt(K)` |
| `chunk_size` | int64 | 64 |
| `use_qk_l2norm_in_kernel` | bool | false |
| `use_beta_sigmoid_in_kernel` | bool | false |
| `use_gate_in_kernel` | bool | false |
| `state_v_first` | bool | false |
| `use_exp2` | bool | true |

`du` 必须保持调用本算子前的原始值。dqkwg 只用它生成 `dw0`，不能先覆盖成其它
`dv` 再传入 prepare 路径。

参数约束：

- `scale=None` 时 wrapper 传 `1/sqrt(128)`；
- `chunk_size` 为兼容上层调用保留，但只接受 `64`；
- `use_qk_l2norm_in_kernel=True` 时 `q_rstd/k_rstd` 必须非空；
- `use_beta_sigmoid_in_kernel=True` 时 `beta_raw` 必须非空；
- 对应开关为 `False` 时，可选输入允许为空且 kernel 不绑定、不搬运该 GM 地址；
- `use_gate_in_kernel` 当前只接受 `False`，输入 `g` 必须是核外已经准备好的 chunk gate；
- `use_exp2` 当前只接受 `True`，所有 gate 指数项均按 `exp2` 计算；
- `state_v_first=False` 时 `h/dh` 末两维按 `[K,V]` 存储，`True` 时按 `[V,K]` 存储；
- `cu_seqlens/chunk_indices` 必须同时为 `None` 或同时非空；
- 当前实现返回顺序固定为 `(dq, dk, dv, dbeta, dg)`。

### 3.2 TilingKey 模板

`D_T_Q` 固定为 BF16，`D_T_G` 同时表示 `g/beta` 的共同 BF16 或 FP32 dtype；
`USE_QK_L2NORM` 和 `USE_BETA_SIGMOID` 是两个独立 bool 模板参数：

| `D_T_G` | `USE_QK_L2NORM` | `USE_BETA_SIGMOID` | kernel 行为 |
|---|---:|---:|---|
| BF16 或 FP32 | 0 | 0 | 直接输出聚合后的 `dq/dk` 与 `db_prepare` |
| BF16 或 FP32 | 0 | 1 | 只融合 beta sigmoid backward |
| BF16 或 FP32 | 1 | 0 | 只融合 Q/K L2Norm backward |
| BF16 或 FP32 | 1 | 1 | 同时融合两条 backward |

因此总计生成 `2 * 2 * 2 = 8` 个模板实例。Host 使用属性值调用
`GET_TPL_TILING_KEY`，kernel 通过 `if constexpr` 裁掉关闭路径的输入搬运和 VF
公式；这两个开关不再存入 `ChunkGatedDeltaRuleBwdFinalizeTilingData`。定长/变长、shape、
任务组、workspace 调度和 `state_v_first` 仍是运行时 tiling 数据。

`use_gate_in_kernel` 和 `use_exp2` 是兼容上层调用的固定属性，不扩展 TilingKey：
前者只支持 `False`，后者只支持 `True`。ACLNN 和 Tiling 两层均校验该支持域，
因此无法从 GE 图路径绕过 ACLNN 校验进入不受支持的计算分支。


## 4. Stage 0--15 GPU 对齐语义与当前空间划分

以下按单个 `(chunk,hv)` 描述。令 `hk=hv/G`，当前 chunk 有效长度为 `M`。
下文按当前 kernel 的 Stage、完整张量搬运和物理容量描述；大型张量为
BF16，小向量与归约标量为 FP32。下文统一标注
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
beta_raw                   # [BT]
```

外部 `h/dh` 在 `state_v_first=False` 时按 `[K,V]` 物理存储，在 `True` 时按
`[V,K]` 物理存储。下文公式中的 `Hc/DHc` 统一是 `[V,K]` 逻辑视图：
False 路径使用列主序 view 零拷贝转置，True 路径按行主序直接搬入。
`A` 的物理存储为 `[BT,BT]` row-major；公式中的 `A.T` 必须通过
column-major 读取视图表达。

本节保持 GPU 对齐 golden 的数学结果，并按当前代码把后半段明确拆为
S12 向量预处理、S13 三条 Cube 结果、S14 每 HV partial、S15 跨 HV 聚合。

### 4.1 GPU 对齐 golden 的转置实现

| 位置 | GPU 对齐目标 | 当前 kernel | 实现约束 |
|---|---|---|---|
| S5 `dA1` | `dA1 = dA0 @ A.T` | `TileCopyDau` 将同一份 row-major `A` resident 按 column-major L1B 视图读取 | 不增加实体转置或新 L1 区 |
| S7 `dA2` | `dA2 = A.T @ dA1` | `TileCopyDvb` 将同一份 row-major `A` resident 按 column-major L1A 视图读取 | 复用 S3 已有的 `A.T` 路径 |
| S7 `dkbg0` | `dkbg0 = A.T @ dw0` | `TileCopyDvb` 将同一份 row-major `A` resident 按 column-major L1A 视图读取 | 与 `dA2` 使用同一 `A.T` 视图 |

后续实现和验证必须按最早偏离点递进：

1. 单独落盘 S5 `dA1`，确认 `dA0 @ A.T` 的 layout 与 BF16 Cube 结果。
2. S5 通过后，再分别落盘 S7 `dA2` 和 `dkbg0`，不用最终输出反推转置是否正确。
3. S5/S7 中间结果全部对齐后，再执行全量 ATK；任一失败 case 先保存输出并用
   CT 检查结构性误差，不改值域或精度阈值。

### 4.2 跨 Stage 数据依赖总表

下表中的“来源”是 GPU 对齐目标 DAG 的生产 Stage，Stage 划分和传递位置
沿用当前 kernel。原始输入记为 `input`；
“UB resident”表示数据在同一 AIV 的固定 UB 地址保留，“L1 resident”表示数据在
同一 AIC 的固定 L1 地址保留，“workspace”表示生产者先落 GM、消费者再搬入片上。

| 消费 Stage | 依赖数据 | 来源与计算 | 传递位置 |
|---|---|---|---|
| S1 | `vb` | S0：`vb = v * beta` | `vbDkbWorkspace` -> L1 |
| S2 | `dA_u` | S1：`dA_u = du @ vb.T` | Fixpipe -> owner AIV UB |
| S2 | `k, beta` | S0 从 input 搬入并保留 | UB resident |
| S3 | `dw0` | S1：`dw0 = du @ Hc`，`Hc=h.T` | L1 resident |
| S3 | `kbg` | S0：`kbg = k * (beta * exp2(g))` | `kbgDoGWorkspace` -> L1 |
| S4 | `dA_u_lower` | S2：`tril(dA_u,-1)` | UB resident |
| S4 | `dA_w0` | S3：`dA_w0 = dw0 @ kbg.T` | Fixpipe -> owner AIV UB |
| S5 | `dA0` | S4：`dA_u_lower - tril(dA_w0,-1)` | `dA0Workspace` -> L1 |
| S5 | `A` | S3 从 input 搬入并保留 | L1 resident |
| S6 | `dvb` | S3：`dvb = A.T @ du` | Fixpipe -> owner AIV UB |
| S7 | `dA1` | S5：`dA1 = dA0 @ A.T` | L1 resident |
| S7 | `dw0, A` | 分别由 S1、S3 产生/搬入 | L1 resident |
| S8 | `dA2` | S7：`dA2 = A.T @ dA1` | Fixpipe -> owner AIV UB |
| S8 | `dkbg0` | S7：`dkbg0 = A.T @ dw0` | Fixpipe -> owner AIV UB |
| S8 | `a2` | S5：`a2 = k @ k.T` | UB resident |
| S8 | `db_v_partial` | S6：`rowSum(dvb * v)` | UB resident |
| S9 | `dA` | S8：`tril(-(dA2*gate_dA),-1)` | `dA0Workspace` -> L1 |
| S9 | `kb` | S2：`kb = k * beta` | `dk0Workspace` -> L1 |
| S10 | `g_last` | S0：`exp2(g[M-1])` | UB resident |
| S11 | 无前序计算结果 | 只读取原始 `do/v_new` | input -> L1 |
| S12 | `dkb` | S9：`dkb = dA @ k` | `vbDkbWorkspace` -> UB |
| S12 | `dkb_t` | S9：`dkb_t = dA.T @ kb` | `dvbDkbTWorkspace` -> UB |
| S12 | `ds0` | S11：`ds0 = do @ v_new.T` | Fixpipe -> owner AIV UB |
| S12 | `dkbg0, db_v, dg_prepare` | 分别由 S7、S8 产生 | UB resident |
| S13 | `ds` | S12：`tril(ds0*gate_dA)*scale` | `dsWorkspace` -> L1 |
| S13 | `do_g` | S12：`do * exp2(g) * scale` | `kbgDoGWorkspace` -> L1 |
| S13 | `v_decay` | S12：`v_new * decay` | `dk0Workspace` -> L1 |
| S14 | `dq_hv` | S13：`do_g @ Hc + ds @ k` | Fixpipe -> owner AIV UB |
| S14 | `dk_intra` | S13：`ds.T @ q` | Fixpipe -> owner AIV UB |
| S14 | `dk_base` | S13：`v_decay @ DHc`，`DHc=dh.T` | Fixpipe -> owner AIV UB |
| S14 | `dk_prepare` | S12 的三项合成结果 | UB resident |
| S15 | `dq_hv, dk_raw_hv` | S14 生成的每 HV partial | `dqStage12Workspace`、`dvbDkbTWorkspace` -> UB |

因此，例如 `dkb` 不是输入或调试量，而是 **Stage 9 通过 `dA @ k` 计算得到**，
Stage 12 再从 `vbDkbWorkspace` 读回参与 `dk_prepare` 和 `db_prepare` 的计算。

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

前序依赖：无。S0 只读取原始 `g/beta/k/v`；它生产的 `kbg/vb` 落 workspace，
其余派生量保留在 owner AIV 的 UB，供后续 Vector Stage 使用。

空间布局（UB，单位 KiB）：

```text
UB[0,32)       k[2]；BF16；S0 搬入，保留至 S14
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
UB[247,247.5)  decay[2]；FP32；S0 生成，保留至 S14
UB[247.5,248)  bg[2]；FP32；S0 生成，保留至 S12
```



当前 AIV 最多承包 2 个交错 HV，因此 `g/beta/g_exp/g_last/decay/bg/gate_dA`
各预留 2 份 UB 空间。
`gate_dA` 以及后续大型跨 Stage 中间量按当前代码统一以 BF16 驻留。`k/v` 在本
Stage 首次进入 Vector 路径后，分别以
BF16 `[2,BT,K]` 和 `[2,BT,V]` 保留在 UB；`k` 保留到 Stage 14，
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
dw0  = du @ Hc                              # Hc=h.T: [V,K]；dw0: [BT,K]
dA_u = du @ vb.T                            # dA_u: [BT,BT]
```

前序依赖：`vb` 是 S0 通过 `v * beta` 产生并写入 `vbDkbWorkspace` 的结果；
`du/h` 为原始输入。S1 生产的 `dw0` 留在 L1，`dA_u` 通过 Fixpipe 写入 owner AIV UB。

两条矩阵乘只共享只读 `du`，彼此不读取本 Stage 输出。

空间布局（L1，单位 KiB）：

```text
L1[0,128)    空闲；无数据；S1 可复用
L1[128,256)  h[4]；BF16；S1 搬入，保留至 S13
L1[256,320)  du[4]；BF16；S1 搬入，保留至 S3
L1[320,384)  空闲；S3 后续在此从 workspace 搬入 kbg[4]
L1[384,448)  vb[4]；BF16；S1 从 GM 搬入，S1 消费后释放
L1[448,512)  dw0[4]；BF16；S1 生成，保留至 S7
```

操作流程：

1. `vb` 从 Stage 0 的 GM 结果搬入 L1。每次只搬入当前 HEAD 的
   `du/h/vb`，当前 HEAD 完成对应的 GM->L1 后再进入 MTE1/Cube；
   不把下一 HEAD 的输入预先搬入当前 HEAD 的 resident。Stage0 与 Stage1
   可以按 HEAD 交错推进，但不能把 Stage0 整个任务组完成作为 Stage1
   的启动条件。
2. Cube 完成两条独立矩阵乘。L0A/L0B/L0C 各自使用
   独立 ping/pong，每发射一次 MMAD 就对应取反一次，不用同一
   slot 连续承载两条 GEMM。`dw0[4]` 写入 `L1[448,512)`；`dA_u[2]` 写入
   `UB[112,128)`。
3. `dA_u` 的核间握手只使用 mode=`0x2`。Stage1 消费完当前 HEAD 的
   `vb` 后，AIC 将该 HEAD 的 `dA_u` 通过 Fixpipe 定向写入 owner AIV
   的 resident UB slot；`subBlockId` 选择 owner AIV，当前 AIV 的本地
   slot 选择该 AIV 承包的第 0/1 个 HV。随后 AIC 发送一次 C->V ready，
   两个 AIV 按同一 HEAD 顺序参与聚合 wait，owner 执行后续 Vector 计算。
   Stage0/Stage1 不设置整组汇合。
4. `vb` 完成末次 Cube 消费后释放。`h/du/dw0` 保留到 Stage 3；其中 `du`
   继续供 `dvb` 使用，不从 GM 重读。

### Stage 2：Vector，dA_u 预处理并提前计算 kb

```text
dA_u_lower = tril(dA_u, diagonal=-1)       # dA_u_lower: [BT,BT]
kb = k * beta[:,None]                      # kb: [BT,K]
```

前序依赖：`dA_u` 是 S1 通过 `du @ vb.T` 产生的 Cube 结果；`k/beta` 由 S0
从原始输入搬入后保留在 UB。S2 将 `kb` 写入 `dk0Workspace` 供 S9 使用。

空间布局（UB，单位 KiB）：

```text
UB[0,32)       k[2]；BF16；保留至 S14
UB[32,64)      v[2]；BF16；保留至 S6
UB[64,80)      gate_dA[2]；BF16；保留至 S12
UB[80,112)     dvb[2] 并发输出槽；BF16；S3 生成，保留至 S6
UB[112,128)    dA_u[2] -> dA_u_lower[2]；BF16；S1 写入，S2 原位处理后保留至 S4
UB[128,136)    dA_w0[HV0] 后续输出槽；BF16；S3 生成，保留至 S4
UB[136,144)    空闲；原 kbg bank 0 的后半段
UB[144,152)    dA_w0[HV1] 后续输出槽；BF16；S3 生成，保留至 S4
UB[152,160)    空闲；原 kbg bank 1 的后半段
UB[160,192)    kb[2]；BF16；S2 生成，写入 GM 后释放
UB[192,245.5)  空闲；无数据；S2 可复用
UB[245.5,246)  beta[2]；FP32；保留至 S12
UB[246,246.5)  g_exp[2]；FP32；保留至 S12
UB[246.5,247)  g_last[2]；FP32；保留至 S10
UB[247,247.5)  decay[2]；FP32；保留至 S14
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
dvb    = A.T @ du                           # dvb: [BT,V]
```

前序依赖：`dw0` 是 S1 通过 `du @ Hc` 产生并保留在 L1 的结果；`kbg` 是 S0
产生并落入 `kbgDoGWorkspace` 的结果；`A/du` 为原始输入。`dvb` 必须读取 `A.T`，
这是 `ecdaf290` 修复的布局语义。S3 的两份输出均通过 Fixpipe 写入 owner AIV UB。

空间布局（L1，单位 KiB）：

```text
L1[0,32)     A[4]；BF16 row-major 物理存储；dvb 路径以 column-major 视图读取 A.T
L1[32,128)   空闲；无数据；S3 可复用
L1[128,256)  h[4]；BF16；保留至 S13
L1[256,320)  du[4]；BF16；保留至 S3，消费后释放
L1[320,384)  kbg[4]；BF16；S0 写 workspace，S3 从 workspace 搬入并消费
L1[384,448)  空闲；kb 到 S9 才从 workspace 搬入
L1[448,512)  dw0[4]；BF16；保留至 S7
```

操作流程：

1. `dw0/kbg/du` 读取前序 L1；Cube 将 `kbg[BT,K]` 作为转置的 B 操作数读取，
   不生成实体转置 tensor。`A[4]` 以 row-major 从 GM 搬入 `L1[0,32)` 并保留到
   S7；计算 `dvb` 时，同一物理地址通过 `TileCopyDvb(LayoutColumnMajor)` 解释为
   `A.T`，并由匹配的 `TileMmadDvb` 执行 `A.T @ du`，不生成实体 A 转置副本。
2. 两条矩阵乘彼此独立。Fixpipe 将两份 `dA_w0` 分别写入原 kbg bank 的
   `UB[128,136)`、`UB[144,152)`，将 `dvb[2]` 写入 `UB[80,112)`。
3. `du/kbg` 完成末次 Cube 消费后释放。

### Stage 4：Vector，dA0

```text
dA0 = dA_u_lower - tril(dA_w0, diagonal=-1) # dA0: [BT,BT]
```

前序依赖：`dA_u_lower` 由 S2 对 S1 的 `dA_u` 取严格下三角得到；`dA_w0`
由 S3 通过 `dw0 @ kbg.T` 得到。S4 生成的 `dA0` 写入 `dA0Workspace`。

空间布局（UB，单位 KiB）：

```text
UB[0,32)       k[2]；BF16；保留至 S14
UB[32,64)      v[2]；BF16；保留至 S6
UB[64,80)      gate_dA[2]；BF16；保留至 S12
UB[80,112)     dvb[2]；BF16；S3 写入，保留至 S6
UB[112,128)    dA_u_lower[2]；BF16；保留至 S4，消费后释放
UB[128,136)    dA_w0[HV0] -> dA0[HV0]；BF16；写入 GM 后释放
UB[136,144)    空闲；原 kbg bank 0 的后半段
UB[144,152)    dA_w0[HV1] -> dA0[HV1]；BF16；写入 GM 后释放
UB[152,245.5)  空闲；无数据；S4 可复用
UB[245.5,246)  beta[2]；FP32；保留至 S12
UB[246,246.5)  g_exp[2]；FP32；保留至 S12
UB[246.5,247)  g_last[2]；FP32；保留至 S10
UB[247,247.5)  decay[2]；FP32；保留至 S14
UB[247.5,248)  bg[2]；FP32；保留至 S12
```

操作流程：

1. 单次 VF 读取完整 `dA_w0/dA_u_lower`，生成 `dA0`。
2. `dA0[2]` 在两个原 kbg bank 起点原位覆盖 `dA_w0[2]`，再写入 GM；
   释放 `UB[112,152)` 中本 Stage 使用的有效子区。

### Stage 5：Cube，dA1 与 a2

```text
dA1 = dA0 @ A.T                            # dA1: [BT,BT]
a2  = k @ k.T                              # a2: [BT,BT]
```

前序依赖：`dA0` 由 S4 生成并从 `dA0Workspace` 搬入；`A` 是 S3 已搬入并保留
在 L1 的原始输入；`k` 为原始输入。S5 的 `dA1` 留在 L1，`a2` Fixpipe 到 UB。

空间布局（L1，单位 KiB）：

```text
L1[0,32)     A[4]；BF16；保留至 S7
L1[32,96)    空闲；无数据；S5 可复用
L1[96,128)   dA0[4] -> dA1[4]；BF16；S5 从 GM 搬入 dA0，原址 Fixpipe dA1 并保留至 S7
L1[128,256)  h[4]；BF16；保留至 S13
L1[256,320)  k[4]；BF16；S5 搬入，保留至 S13
L1[320,384)  空闲；无数据；S5 可复用
L1[384,448)  空闲；kb 到 S9 才从 workspace 搬入
L1[448,512)  dw0[4]；BF16；保留至 S7
```

操作流程：

1. `dA0` 从 GM 一次搬入 `L1[96,128)`；`A` 读取 `L1[0,32)` 的前序保存数据。
   `dA1` 路径必须把这份 row-major `A` resident 解释为 column-major `A.T`；
   不创建实体转置副本，也不改变 L1 地址。`k` 从 GM 一次搬入
   `L1[256,320)` 并保留到 Stage 13。kernel 使用
   `TileCopyDau::LayoutTagL1B` 形成转置视图。
2. 两条矩阵乘彼此独立。`dA0` 完成 MTE1 读取后，Fixpipe 将 `dA1[4]`
   原址写入 `L1[96,128)`；将 `a2[2]` 写入 `UB[128,144)`。
3. `A/dA1` 均保留到 Stage 7；Stage 6 不读取 `dA1`。

### Stage 6：Vector，提前完成 dv 与 db_v_partial

```text
dv = dvb * beta[:,None]                    # dv: [BT,V]
db_v_partial = rowSum(dvb * v)             # db_v_partial: [BT]
```

前序依赖：`dvb` 由 S3 通过 `A.T @ du` 得到；`v/beta` 由 S0 保留在 UB。
本 Stage 与 S7 无数据依赖，`a2` 虽已由 S5 产生，但只由 S8 消费。

空间布局（UB，单位 KiB）：

```text
UB[0,32)       k[2]；BF16；保留至 S14
UB[32,64)      v[2]；BF16；保留至 S6，消费后释放
UB[64,80)      gate_dA[2]；BF16；保留至 S12
UB[80,112)     dvb[2] -> dv[2]；BF16；dvb 保留至 S6，dv 写 GM 后释放
UB[112,128)    空闲；无数据；S6 可复用
UB[128,144)    a2[2]；BF16；S5 写入，保留至 S8
UB[144,244.5)  空闲；无数据；S6 可复用
UB[244.5,245)  db_v_partial[2]；FP32；S6 生成，保留至 S8
UB[245.5,246)  beta[2]；FP32；保留至 S12
UB[246,246.5)  g_exp[2]；FP32；保留至 S12
UB[246.5,247)  g_last[2]；FP32；保留至 S10
UB[247,247.5)  decay[2]；FP32；保留至 S14
UB[247.5,248)  bg[2]；FP32；保留至 S12
```

操作流程：

1. 单次 VF 完成 `db_v_partial` 和 `dvb -> dv`，不读取或处理 `dA1`。
2. `dv` 写入 GM，`db_v_partial` 保留在 UB 到 Stage 8，释放 `v/dvb` 地址。
   Stage 7 使用独立的 `UB[192,240)` 输出区，与本 Stage 无执行顺序约束。

### Stage 7：Cube，dA2 与 dkbg0

```text
dA2 = A.T @ dA1                            # dA2: [BT,BT]
dkbg0 = A.T @ dw0                          # dkbg0: [BT,K]
```

前序依赖：`dA1` 由 S5 通过 `dA0 @ A.T` 得到；`dw0` 由 S1 通过 `du @ Hc`
得到；`A` 从 S3 起保留在 L1。S7 的 `dA2/dkbg0` 均 Fixpipe 到 owner AIV UB。

空间布局（L1，单位 KiB）：

```text
L1[0,32)     A[4]；BF16；保留至 S7，消费后释放
L1[32,96)    空闲；无数据；S7 可复用
L1[96,128)   dA1[4]；BF16；S5 Fixpipe 写入，S7 消费后释放
L1[128,256)  h[4]；BF16；保留至 S13
L1[256,320)  k[4]；BF16；保留至 S13
L1[320,384)  空闲；无数据；S7 可复用
L1[384,448)  空闲；kb 到 S9 才从 workspace 搬入
L1[448,512)  dw0[4]；BF16；保留至 S7，消费后释放
```

操作流程：

1. 两条矩阵乘只共享只读 `A.T` 视图，彼此不依赖。两条路径都必须
   使用 S3 已验证的 column-major A 搬运/MMAD 语义，复用同一
   `L1[0,32)` resident，不增加新 L1 区。
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

前序依赖：`dA2/dkbg0` 均由 S7 产生，`a2` 由 S5 产生，`db_v_partial` 由 S6
产生，`gate_dA/k/bg/g_exp/beta` 由 S0 产生或保留。S8 将 `dA` 写入
`dA0Workspace`，其余 `db_v/dg_prepare` 留在 UB。

空间布局（UB，单位 KiB）：

```text
UB[0,32)       k[2]；BF16；保留至 S14
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
UB[247,247.5)  decay[2]；FP32；保留至 S14
UB[247.5,248)  bg[2]；FP32；保留至 S12
```

操作流程：

1. 一次 VF 允许读取本 Stage 前序向量公式的结果，因此依次完成上述全部公式。
2. `dA[2]` 写入 GM，供 Stage 9 搬入 L1；`kb` 已由 Stage 2 写入
   `dk0Workspace`，本 Stage 不读取它。
3. `gate_dA` 和 `dkbg0` 保持首次分配的物理地址不变。保留
   `k/gate_dA/dkbg0/bg/db_v/dg_prepare`，释放 `dA/a2` 的 UB 地址。

### Stage 9：Cube，prepare K 矩阵乘

```text
dkb   = dA @ k                             # dkb: [BT,K]
dkb_t = dA.T @ kb                          # dkb_t: [BT,K]
```

前序依赖：`dA` 是 S8 通过 `tril(-(dA2*gate_dA),-1)` 得到的结果；`kb` 是
S2 通过 `k*beta` 得到的结果；`k` 为 S5 搬入后保留的原始输入。S9 将 `dkb`
写入 `vbDkbWorkspace`，将 `dkb_t` 写入 `dvbDkbTWorkspace`。

空间布局（L1，单位 KiB）：

```text
L1[0,96)     空闲；无数据；S9 可复用
L1[96,128)   dA[4]；BF16；S9 从 GM 搬入，S9 消费后释放
L1[128,256)  h[4]；BF16；保留至 S13
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

前序依赖：`g_last` 由 S0 计算并保留在 UB；`h/dh` 为本 Stage 从原始输入搬入。
S10 与 S9、S11 均无数据依赖，结果原位覆盖 `g_last` 的标量槽。

空间布局（UB，单位 KiB）：

```text
UB[0,32)       k[2]；BF16；保留至 S14
UB[32,64)      h[0]；BF16；S10 搬入，S10 归约后释放
UB[64,80)      gate_dA[2]；BF16；保留至 S12
UB[80,112)     h[1]；BF16；S10 搬入，S10 归约后释放
UB[112,144)    dh[0]；BF16；S10 搬入，S10 归约后释放
UB[144,176)    dh[1]；BF16；S10 搬入，S10 归约后释放
UB[176,208)    空闲；无数据；S10 可复用
UB[208,240)    dkbg0[2]；BF16；保留至 S12
UB[240,244.5)  空闲；无数据；S10 可复用
UB[244.5,245)  db_v[2]；FP32；保留至 S12
UB[245,245.5)  dg_prepare[2]；FP32；保留至 S14
UB[245.5,246)  beta[2]；FP32；保留至 S12
UB[246,246.5)  g_exp[2]；FP32；保留至 S12
UB[246.5,247)  g_last[2] -> state_term[2]；FP32；g_last 保留至 S10，state_term 保留至 S14
UB[247,247.5)  decay[2]；FP32；保留至 S14
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

前序依赖：无前序计算结果。S11 只读取原始 `do/v_new`，与 S10 无数据依赖。
当前代码 **不在 S11 计算 `do @ Hc`**；该乘法在 S12 先生成 `do_g` 后，作为
S13 的 `do_g @ Hc` 执行。`ds0` 通过 Fixpipe 直接写入 owner AIV UB。

空间布局（L1，单位 KiB）：

```text
L1[0,32)     do 当前 Stage 输入[2]；BF16；S11 从 GM 搬入，S11 结束后释放
L1[32,128)   空闲；无数据；S11 可复用
L1[128,256)  h[4]；BF16；S1 搬入，保留至 S13
L1[256,320)  k[4]；BF16；保留至 S13
L1[320,384)  v_new[4]；BF16；S11 搬入，保留至 S13
L1[384,512)  空闲；无数据；S11 可复用
```

操作流程：

1. `v_new` 从 GM 搬入 `L1[320,384)`，`do` 搬入 `L1[0,32)`；每个 HEAD
   只发射一条 `do @ v_new.T`。
2. `ds0` Fixpipe 到 `UB[176,192)` 的两份 8 KiB slot，并以逐 HEAD C->V ready
   通知 S12；不为 `ds0` 额外写 GM。
3. `v_new` 保留到 S13；`h/k` 虽不被 S11 消费，但继续占用各自 L1 resident。

### Stage 12：Vector，prepare K、DS/DQ 后处理

```text
dk_prepare_hv = -dkbg0 * bg[:,None]
                + dkb * beta[:,None] + dkb_t           # dk_prepare_hv: [BT,K]
db_prepare = db_v + rowSum(dkb * k)                    # db_prepare: [BT]

do_g = do * g_exp[:,None] * scale                      # do_g: [BT,V]
v_decay = v_new * decay[:,None]                        # v_decay: [BT,V]
ds = tril(ds0 * gate_dA) * scale                       # ds: [BT,BT]
```

前序依赖：`dkb/dkb_t` 是 S9 的两条 GEMM 输出，`ds0` 是 S11 的 Cube 输出，
`dkbg0` 来自 S7，`db_v` 来自 S8；`gate_dA/beta/g_exp/decay/bg` 来自 S0。
S12 生产 `do_g/v_decay/ds` 供 S13 使用，同时输出最终 `dbeta`。

空间布局（UB，单位 KiB）：

```text
UB[0,32)       k[2]；BF16；保留至 S14
UB[32,64)      dkb[2]；BF16；S12 从 GM 搬入，S12 消费后释放
UB[64,80)      gate_dA[2]；BF16；S0 生成，S12 消费后释放
UB[80,112)     dkb_t[2]；BF16；S12 从 GM 搬入，S12 消费后释放
UB[112,144)    do[2] -> do_g[2]；BF16；S12 从 GM 搬入并原位生成，写 workspace
UB[144,176)    v_new[2] -> v_decay[2]；BF16；S12 从 GM 搬入并原位生成，写 workspace
UB[176,192)    ds0[2] -> ds[2]；BF16；S11 Fixpipe 写入，S12 原位生成后写 workspace
UB[208,240)    dkbg0[2] -> dk_prepare[2]；BF16；S12 原位生成，dk_prepare 保留至 S14
UB[240,244)    空闲；无数据；S12 可复用
UB[244,244.5)  beta_raw[2]；仅 sigmoid backward 搬入；BF16 ABI 写回前复用为临时打包 dbeta
UB[244.5,245)  db_v[2] -> db_prepare[2] -> dbeta[2]；始终 FP32；写 GM 后释放
UB[245,245.5)  dg_prepare[2]；FP32；保留至 S14
UB[245.5,246)  beta[2]；FP32；保留至 S12，消费后释放
UB[246,246.5)  g_exp[2]；FP32；保留至 S12，消费后释放
UB[246.5,247)  state_term[2]；FP32；保留至 S14
UB[247,247.5)  decay[2]；FP32；保留至 S14
UB[247.5,248)  bg[2]；FP32；S12 消费后释放
```

操作流程：

1. `dkb/dkb_t/do/v_new` 从 GM 搬入；仅 `USE_BETA_SIGMOID=1` 时搬入
   `beta_raw`。`ds0` 由 S11 Fixpipe 已写入固定 UB slot；当前模板所需输入在一次
   VF 前到齐。
2. 一次 VF 完成上述公式；`dk_prepare` 原位覆盖 `dkbg0`，`do_g` 原位覆盖
   `do`，`v_decay` 原位覆盖 `v_new`。`USE_BETA_SIGMOID=1` 时执行 sigmoid
   backward；否则 `dbeta=db_prepare`。`dbeta` 始终以 FP32 驻留；FP32 ABI 直接
   写回，BF16 ABI 仅在 MTE3 前生成临时打包结果。
3. `ds[2]` 原位覆盖 `ds0[2]`。`ds/do_g/v_decay` 分别写入
   `dsWorkspace/kbgDoGWorkspace/dk0Workspace`，供 S13 搬入 L1。Stage 12 是
   `gate_dA/bg` 的最后一个消费者，完成后释放二者；保留
   `k/dk_prepare/dg_prepare/state_term/decay`；这些数据均保持原物理地址。

### Stage 13：Cube，DQ/DK 三条最终矩阵乘

```text
dq_hv    = do_g @ Hc + ds @ k              # Hc=h.T: [V,K]；dq_hv: [BT,K]
dk_intra = ds.T @ q                        # dk_intra: [BT,K]
dk_base  = v_decay @ DHc                   # DHc=dh.T: [V,K]；dk_base: [BT,K]
```

前序依赖：`ds/do_g/v_decay` 均由 S12 生成并分别落入三个 workspace；`h` 从 S1、
`k` 从 S5 起保留在 L1，`q/dh` 为本 Stage 原始输入。S11 搬入的 `v_new` resident
在 S13 被 `do_g/v_decay` 按顺序复用，不再保留原始语义。
其中 `dq_hv` 在同一 L0C 中先执行 `do_g @ Hc`，再累加 `ds @ k`，最后只做一次 Fixpipe。

空间布局（L1，单位 KiB）：

```text
L1[0,32)     q 当前 Stage 输入[2]；BF16；S13 从 GM 搬入，S13 结束后释放
L1[32,96)    dh 当前 Stage 输入[2]；BF16；S13 从 GM 搬入，S13 结束后释放
L1[96,128)   ds[4]；BF16；S13 从 GM 搬入，S13 消费后释放
L1[128,256)  h[4]；BF16；S1 搬入，S13 消费后释放
L1[256,320)  k[4]；BF16；保留至 S13，消费后释放
L1[320,384)  do_g[4] / v_decay[4]；BF16；两者在不同时间复用同一组 resident
L1[384,512)  空闲；无数据；S13 可复用
```

操作流程：

1. Stage 13 的每个 HEAD 仅依赖 S12 同 HEAD 的三份 workspace 输出。
   owner AIV 在本 HEAD 的 `ds/do_g/v_decay` MTE3 全部完成后发布 ready，
   non-owner 只参与该 HEAD 的聚合到达；Cube 因此无需等待整组 S12。
   `ds/do_g/v_decay/q/dh` 在本 Stage 按真实末次消费顺序搬入 L1；
   `do_g` 消费后其 resident 原址复用为 `v_decay`。
2. 三条结果通过 Fixpipe 写入 owner AIV：`dk_base -> UB[32,64)`、
   `dq_hv -> UB[80,112)`、`dk_intra -> UB[144,176)`。它们分别逐 slot
   复用 S12 已结束生命周期的 `dkb`、`dkb_t`、`v_decay` 物理槽；`ds` 位于
   `UB[176,192)`，不被 S13 输出覆盖。同 owner 的后续 HEAD 使用另一个
   local-HV slot，所以 S12/S13 可以保持单 HEAD 粒度的流水。
3. 释放全部 L1 数据。

### Stage 14：Vector，DQ/DK 与 gate 最终 backward

```text
dk_raw_hv = dk_base + dk_intra + dk_prepare # dk_raw_hv: [BT,K]
dq_dot = rowSum(dq_hv * q)                 # dq_dot: [BT]
dk_base_dot = rowSum(dk_base * k)          # dk_base_dot: [BT]
dk_intra_dot = rowSum(dk_intra * k)        # dk_intra_dot: [BT]
x = dq_dot - dk_base_dot - dk_intra_dot
x[M-1] += sum(dk_base_dot) + state_term
dg[t] = sum(x[j] + dg_prepare[j], j=t..M-1) # reverse suffix sum
```

前序依赖：`dq_hv/dk_intra/dk_base` 是 S13 的三份 Fixpipe 输出；`dk_prepare`
由 S12 生成并保留在 UB；`dg_prepare` 来自 S8，`state_term` 来自 S10，`q/k`
为原始输入或 resident。S14 **不做跨 HV 聚合和 Q/K norm backward**；它只生成
每个 HV 的 `dq_hv/dk_raw_hv` partial 和最终 `dg`，前两者落 workspace 交给 S15。

空间布局（UB，单位 KiB）：

```text
UB[0,32)       k[2]；BF16；保留至 S14，计算结束后释放
UB[32,64)      dk_base[2] -> dk_raw_hv[2]；BF16；S13 写入，S14 原位合成并写 workspace
UB[80,112)     dq_hv[2]；BF16；S13 逐 slot 覆盖已消费的 dkb_t，S14 写 workspace
UB[112,144)    q[2]；BF16；S14 从 GM 搬入；G=3/4 时同一 AIV 内复用
UB[144,176)    dk_intra[2]；BF16；S13 写入，S14 消费后释放
UB[208,240)    dk_prepare[2]；BF16；保留至 S14，消费后释放
UB[240,244)    空闲；无数据；S14 可复用
UB[244,244.5)  空闲；S14 不读取 q_rstd/k_rstd
UB[244.5,245)  空闲；无数据
UB[245,245.5)  dg_prepare[2]；FP32；保留至 S14，消费后释放
UB[245.5,246)  空闲；无数据；S14 可复用
UB[246,246.5)  空闲；无数据
UB[246.5,247)  state_term[2]；FP32；保留至 S14，消费后释放
UB[247,247.5)  dg_scratch[2]；始终 FP32；BF16 ABI 仅在写回前原位生成临时打包结果
UB[247.5,248)  空闲；S12 已完成 bg 的末次消费
```

操作流程：

1. `q` 从 GM 搬入；本 Stage 不读取 `q_rstd/k_rstd`。
2. 单次 VF 完成 `dk_raw_hv`、三个点积和 `dg` reverse suffix sum；`dg` 以
   FP32 驻留。FP32 ABI 直接写回，BF16 ABI 仅在最终 MTE3 前打包。
3. `dq_hv` 写入 `dqStage12Workspace`，`dk_raw_hv` 写入
   `dvbDkbTWorkspace`，`dg` 直接写最终输出。两份 partial 使用当前
   `core/window/HEAD` 固定 slot，S15 在本组内按相同 slot 聚合。
4. 两个 AIV 完成整组 S14 MTE3 后在 `stage14To15Flag` 聚合。该点之后
   S13 的 `dk_base/dq_hv/dk_intra` Fixpipe 目标已全部完成末次消费，
   两个 AIV 立即通过 `vecToCube` 释放 AIC；S15 不再属于该跨核等待区间。

### Stage 15：Vector，GVA 聚合与可选 Q/K norm backward

```text
dq_acc[hk] = sum(dq_hv[hv_i], hv_i in H(hk))
dk_acc[hk] = sum(dk_raw_hv[hv_i], hv_i in H(hk))

if use_qk_l2norm_in_kernel:
    dq = (dq_acc - q * rowSum(dq_acc * q)[:,None]) * q_rstd[:,None]
    dk = (dk_acc - k * rowSum(dk_acc * k)[:,None]) * k_rstd[:,None]
else:
    dq = dq_acc
    dk = dk_acc
```

前序依赖：`dq_hv/dk_raw_hv` 均由 S14 逐 HV 写入 workspace。S15 先等待同一
AI Core 的两个 AIV 完成整组 S14 MTE3，再按 `H(hk)` 聚合；`q/k/q_rstd/k_rstd`
只在启用 norm backward 时从原始输入搬入。S15 生产最终 `dq/dk`。

空间布局（UB，单位 KiB）：

```text
UB[0,32)       dq_acc[2]；BF16；每个 AIV 最多保留两个 HK accumulator
UB[32,64)      S13 dk_base[2] Fixpipe 保留区；S15 不覆盖
UB[64,80)      dq_partial_input；BF16；逐 HV 复用
UB[80,112)     S13 dq_hv[2] Fixpipe 保留区；S15 不覆盖
UB[112,144)    dk_acc[2]；BF16
UB[144,176)    S13 dk_intra[2] Fixpipe 保留区；S15 不覆盖
UB[176,192)    dk_partial_input；BF16；逐 HV 复用
UB[192,208)    q；BF16；仅 norm backward 使用
UB[208,224)    k；BF16；仅 norm backward 使用
UB[224,244)    空闲；无数据；S15 可复用
UB[244,244.25) q_rstd；FP32；仅有效 M 个元素
UB[244.25,244.5) k_rstd；FP32；仅有效 M 个元素
```

S15 的 8 个 16 KiB 大槽位于 S13 三份 Fixpipe 目标的补集，因此 S14
整组消费完成后，AIC 可以立即复用 S13 目标，无需等待 S15 的聚合、
norm backward 或最终 MTE3。S15 对 S14 地址的复用仍由 `stage14To15Flag` 保护。

操作流程：

1. 每个 HK 的首个 HV partial 直接搬入 accumulator；后续 HV 经共享 input slot
   搬入，并由 `Stage15AccumulateVF` 原位累加。
2. `G=1/2` 时两个 AIV 分担连续 HK；`G=3/4` 时只由 AIV0 聚合，避免跨 AIV
   accumulator 通信。
3. `USE_QK_L2NORM=1` 的模板搬入 `q/k/q_rstd/k_rstd` 并调用
   `Stage15NormVF`；为 0 时不绑定也不搬运这些可选输入，累加结果直接写出。
   两条路径都保留 workspace accumulator 的 MTE2 到最终 MTE3 所需事件链。

beta sigmoid backward：

```text
s     = sigmoid(beta_raw)                     # s: [BT]
dbeta = db_prepare * s * (1-s)                # dbeta: [BT]
```

关闭 `use_beta_sigmoid_in_kernel` 时：

```text
dbeta = db_prepare                            # dbeta: [BT]
```

`use_gate_in_kernel=False` 时，`dg` 按前述 dqkwg/prepare 计算链生成；本算子不读取
原始 `g_input`、`A_log` 或 `dt_bias`，也不产生对应的全局参数梯度。

## 5. 当前 Stage 资源分配与复用

本节汇总当前代码已经落地的物理分配，不是后续目标方案。具体偏移以
`chunk_gated_delta_rule_bwd_finalize_vector.h`、`chunk_gated_delta_rule_bwd_finalize_cube.h` 和 kernel 入口为准。
每个 stage 只能执行 Cube 矩阵乘或 Vector 向量操作中的一种。L1 只作为
Cube 的输入/中间结果空间，UB 只作为 Vector 运算空间以及 Cube-to-Vector 结果落点。
Vector 不在 L1 上计算，Cube 不从 UB 取矩阵乘输入。
GM/L1/UB 之间的数据搬运和必要同步不改变 Stage 的计算类型：Cube Stage 只能发出
矩阵乘，Vector Stage 只能发出向量指令。

本方案必须同时满足以下固定约束：

1. 每个 Stage 只能包含 Cube 矩阵乘或 Vector 向量操作之一。
2. 同一 Cube Stage 内的矩阵乘不能读取该 Stage 内任一 Cube/Vector 操作刚生成的
   输出，即同一 Cube Stage 内的矩阵乘必须彼此独立。Vector Stage 允许后续向量公式
   读取该 Stage 内前序向量公式的输出。一个 Stage 完成后，其输出才视为可供后续
   Stage 依赖的数据。
3. L1 固定为 512 KiB 且只供 Cube 使用；UB 固定为 248 KiB 且只供 Vector 使用。
4. Cube-to-Vector 非算子输出必须写入两份 UB 保存空间。
5. Vector-to-Vector 非算子输出必须保留两份 UB 保存空间。
6. Vector-to-Cube 非算子输出必须先由 MTE3 写 GM workspace，再由消费 Stage 的
   AIC 搬入最多四份 L1 resident；当前实现不存在 UB 直达 L1 的通路。
7. Cube-to-Cube 非算子输出必须保留四份 L1 保存空间。
8. UB 不预留固定子区域，完整 `UB[0,248)` 均可由当前 Vector Stage 使用；跨 Stage
   保存数据、当前 Stage 输入、临时量和输出的同时存活总量不得超过 248 KiB。
9. L1 不预留固定子区域，完整 `L1[0,512)` 均可由当前 Cube Stage 使用，并可在不同
   Stage 改变地址语义；需要跨 Stage 保存的 tensor 必须按四份容量预留。
10. 同一原始输入在同一计算路径内不得无理由重复从 GM 搬运。若同一原始输入同时参与
    Cube 和 Vector，允许分别执行 `GM -> L1` 和 `GM -> UB`；Vector-to-Cube
    中间量按第 6 条执行一次 workspace 写入和一次消费搬入。
11. UB 内任一地址在前一数据完成真实末次消费后都可改变语义，不存在固定
    tensor、输入、临时量或小向量固定边界。
12. Vector Stage 不允许拆分 VF pass 或 tile。进入 VF 前必须一次搬完本 Stage 的完整
    逻辑输入，随后通过一次 VF 调用完成该 Stage 的全部向量公式；尾块仍分配完整
    `BT=64` buffer，通过 `M` 屏蔽无效元素。
13. 没有依赖关系的 Cube Stage 和 Vector Stage 不建立执行顺序约束，允许并发执行；
    因此两者同时存活的数据地址不能重叠。地址只有在前一 tensor 的真实末次
    消费者完成后才能复用，不能依据 Stage 编号先后复用无依赖数据的地址。
14. 若第 1--13 条无法同时满足，允许通过 GM workspace 落地并再次搬入解决容量冲突。
    回退只用于经过完整依赖 DAG 和容量计算证明无法片上驻留的 tensor，并且每次写入、
    读取的对象和 GM 相对偏移都必须在对应 Stage 中明确列出。
15. 优先避免连续 Cube Stage 或连续 Vector Stage。只要能前移或后移真实计算任务
    作为中间 Stage，就允许增加 Stage 数；不得插入无公式、无输出的空 Stage。若连续
    同类型 Stage 无法消除，必须均衡可调度任务，使连续段内每个 Stage 的任务尽可能少。
16. Vector 路径尽量避免重复计算相同中间量；只要容量允许，重复使用的向量结果应按
    两份容量保留在 UB，直到最后一个 Vector 消费者完成。当前实现将
    `gate_dA` 和 `bg=beta*g_exp` 保留到 Stage 12，不重新计算；S14 的
    `dk_base_dot` 在寄存器内同时供逐行 `dg` 项和尾元素补偿使用。
17. 减少 Stage 总数的优先级最低。只有在第 1--16 条均已满足、不会重新形成连续
    同类型 Stage、也不会增加连续同类型 Stage 的任务数时，才允许合并 Stage。
18. 任一 tensor 一旦在 UB 或 L1 分配物理区间，从首次写入到真实末次消费完成，
    其物理区间必须保持不变，禁止中途搬移、压紧或重排。只允许在原地址原位更新
    tensor 语义，或在原 tensor 生命周期彻底结束后将该地址分配给新 tensor。

当前实现中 S9 的 `dkb/dkb_t` 必须落 workspace；S11 的 `ds0` 则直接 Fixpipe 到
owner AIV 的 `UB[176,192)`，不分配 S11 GM 中间区。S12 将 `ds0` 原位变为
`ds`，并把 `ds/do_g/v_decay` 分别落到三个 workspace 供 S13 使用。

### 5.1 GM workspace 实际映射

调度将 `(chunkTaskIdx, headGroupIdx)` 展平为逻辑任务：

```text
taskIdx = chunkTaskIdx * headGroupNum + headGroupIdx
taskNum = totalChunkNum * headGroupNum
```

head group 以一个完整 GVA bundle（同一 HK 对应的 `GVA=HV/HK`
个 HV）为不可拆分原子。参考 dhu 的动态 head window，host 先计算：

```text
maxGvaPerTask = floor(4 / GVA)
targetGvaPerTask = min(maxGvaPerTask,
    ceil(totalChunkNum * HK / aicCoreNum))
gvaPerTask = max(d | d <= targetGvaPerTask and HK % d == 0)
taskGroupSize = gvaPerTask * GVA
headGroupNum = HK / gvaPerTask
```

`taskGroupSize <= 4` 与当前 L1/UB/workspace 的 4 HEAD 物理容量一致。
`taskGroupSize` 始终是 GVA 的整数倍，所以 GVA bundle 不跨组；
`gvaPerTask` 同时整除 HK，所以每个 head group 等大，不产生轻量尾组。

每个 AIC 固定分配两套 HEAD window，每套最多容纳 4 个 HEAD。
`localTaskRound` 对当前物理核承包的逻辑任务计数，workspace 槽位为：

```text
slot = coreIdx * 8 + (localTaskRound % 2) * 4 + headOffset
S = 64 * 128 * sizeof(DT)
R = blockDim * 8 * S
workspaceUserBytes = 7 * R
```

kernel 入口切分 7 个 region。向量在当前 slot 中使用 `[64,128]`；矩阵
使用同一 slot 的 `[64,64]` 前半区。AIC 最多比 AIV 超前一个 task group，
因此奇偶双 window 能保留上一个逻辑任务仍在 S15 消费的 partial；第三个任务的 S1
必须等待 AIV 完成上一组 S15 后发出的 S0 ready，不会提前覆盖同奇偶 slot。
同一 HK 对应的 `headRatio` 个 HV 不会跨 head group，所以 S15 的
`dq/dk` 组内归约不需要额外的物理核间同步。

| Region | 入口变量 | 生命周期与别名 |
|---:|---|---|
| 0 | `kbgDoGWorkspace` | S0 `kbg` -> S3；释放后 S12 `do_g` -> S13 |
| 1 | `vbDkbWorkspace` | S0 `vb` -> S1；释放后 S9 `dkb` -> S12 |
| 2 | `dqStage12Workspace` | S14 `dq_hv` -> S15 |
| 3 | `dvbDkbTWorkspace` | S9 `dkb_t` -> S12；释放后 S14 `dk_raw_hv` -> S15 |
| 4 | `dA0Workspace` | S4 `dA0` -> S5；释放后 S8 `dA` -> S9 |
| 5 | `dk0Workspace` | S2 `kb` -> S9；释放后 S12 `v_decay` -> S13 |
| 6 | `dsWorkspace` | S12 `ds` -> S13 |

`ds0/dA_u/dA_w0/dvb/a2/dA2/dkbg0/dq_hv(S13)/dk_intra/dk_base` 都不占用
额外 workspace：它们由 Cube Fixpipe 直接写 owner AIV UB。最终 `dv/dbeta/dg` 在各自
Vector Stage 直接写输出，最终 `dq/dk` 由 S15 写输出。

### 5.2 Stage 序列

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
S12 Vector : dkPrepare、dbPrepare/dbeta、ds、doG、vDecay
S13 Cube   : dqHv、dkIntra、dkBase
S14 Vector : dkRawHv、dg；dqHv/dkRawHv 写 W14
S15 Vector : 按 HK 聚合 GVA partial、可选 Q/K norm backward、输出 dq/dk
```

Cube Stage 输入依赖检查：

```text
S1  dw0  <- du,h.T        dAu <- du,S0.vb
S3  dAw0 <- S1.dw0,S0.kbg  dvb <- A.T,S1.du
S5  dA1  <- S4.dA0,A.T    a2  <- k,k
S7  dA2 <- A.T,S5.dA1     dkbg0 <- A.T,S1.dw0
S9  dkb  <- S8.dA,k       dkbT <- S2.kb,S8.dA
S11 ds0 <- do,v_new
S13 dqHv <- S12.doG,h.T + S12.ds,k
    dkIntra <- S12.ds,q
    dkBase <- S12.vDecay,dh.T
```

同一 Stage 中的多个矩阵乘不读取该 Stage 新产生的结果；S13 的 `dqHv` 是同一
输出 L0C 上的两次 MMAD 累加，而不是后一条 GEMM读取前一条 GEMM 的输出。存在结果依赖的
`dw0 -> dAw0`、`dA0 -> dA1 -> dA2` 已拆到不同 Stage。当前 16 个 Stage 从 S0
到 S14 按 Vector/Cube 交替；S15 是必要的连续 Vector Stage，用于读取 S14 的跨 AIV
partial 并完成 GVA 聚合。S2 前移 `kb`，并用
`dA_u_lower` 建立对 S1 的依赖；S3 从 workspace 搬入 S0 生成的 `kbg`，因此 S2/S3
无执行顺序约束，对应 Cube 输出槽已计入 S2 的 UB 并发集合。S5 将 `dA1`
直接保留在 L1 供 S7 使用；S6 完成 `dv/db_v_partial`，其 UB 地址与 S7 输出不重叠，
因此 S6 和 S7 没有执行顺序约束。负号延后到 S8 与 `gate_dA` 的乘法融合。
当前 S3 到 S9 的跨核同步按单 HEAD 颗粒度闭环，但 Vector Stage
仍按任务组分段执行，不将 `S4/S6/S8` 压入同一 HEAD 串行链。Cube
先对整组逐 HEAD 执行
`S3 -> 通知S4 -> 等待S4 -> S5`，再对整组逐 HEAD 执行
`S7 -> 通知S8 -> 等待S8 -> S9`。Vector 对应分成三个任务组循环：
`逐HEAD等待S3并执行S4/通知S5 -> 整组S6 -> 逐HEAD在owner执行S8前消费S7/通知S9`。
S4 写回 `dA0` 的 `PIPE_MTE3` 信号与 S5 的 GM 读取建立 RAW 依赖；
S7 的 `PIPE_FIX` 信号与 S8 的 UB 读取建立 RAW 依赖；S8 写回
`dA` 的 `PIPE_MTE3` 信号与 S9 的 GM 读取建立 RAW 依赖。两个
AIV 对每个 HEAD 都参与 `mode=0x2` 聚合。S7->S8 链上 non-owner
不原地等待，而是先发送 S8->S9 参与信号；同一 AIV 到下一 owner
HEAD 的 `Stage8VF` 前，再按序消费自上一 owner 以来的全部 S7 token。
任务组以 non-owner HEAD 结尾时，在进入后续 Stage 前排空尾 token。
由于 owner/non-owner 按 HEAD 交替，待消费深度最多为 2，不超过跨核
计数 flag 允许的 15 层深度。
两个插入的 Vector 都承担真实公式，不是空 Stage。Stage 数仅在以上条件满足后再缩减。

Cube 输出只要后续由 Vector 消费且不是算子输出，容量可行时由 Fixpipe 直接写入两份
UB 常驻槽位；容量不可行时允许按第 14 条规则经 GM workspace 中转一次。Vector 输出
只要后续由 Cube 消费且不是算子输出，就按 `UB -> GM workspace -> L1` 传递；当前
硬件路径不存在 UB 直接写 L1。Cube 输出只要后续仍由 Cube 消费且
不是算子输出，就直接写入 4 份 L1 保存空间。Vector 输出只要后续仍由 Vector 消费且
不是算子输出，就保留在两份 UB 保存空间。当前方案中的 `dw0/dA1/kb` 走 L1
保存空间的说法仅适用于 `dw0/dA1`；`kb` 由 S2 写 workspace、S9 再搬入 L1。
`gate_dA/bg/k/v/state_term/dk_prepare/dg_prepare` 走 UB resident。

### 5.3 L1 全空间分配

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
S1 peak : h[4] + du[4] + vb[4] + dw0[4]                = 320 KiB
S1 end  : h[4] + du[4] + dw0[4]                        = 256 KiB
S3 peak : h[4] + du[4] + kbg[4] + dw0[4] + A[4]       = 352 KiB
S3 end  : h[4] + A[4] + dw0[4]                         = 224 KiB
S4 end  : h[4] + A[4] + dw0[4]                         = 224 KiB
S5/S6   : h[4] + k[4] + A[4] + (dA0 -> dA1)[4] + dw0[4]
                                                        = 320 KiB
S7 end  : h[4] + k[4]                                  = 192 KiB
S8/S9   : h[4] + k[4] + dA临时输入[4] + kb[4]      = 288 KiB
S9 end  : h[4] + k[4]                               = 192 KiB
S11 peak : h[4] + k[4] + v_new[4] + do 当前输入[2] = 288 KiB
S11 end : h[4] + k[4] + v_new[4]                    = 256 KiB
S13 peak : h[4] + k[4] + do_g/v_decay[4] + ds[4] + q[2] + dh[2] = 384 KiB
```

L1 完整峰值为 S13 的 384 KiB，剩余 128 KiB。不存在固定不可借用区域。

S3 中 `A[4]` 一次搬入 `L1[0,32)` 并原址保留到 S7。`k[4]` 在 S5 首次搬入
`L1[256,320)` 并保留到 S13。`kb[4]` 仅在 S9 从 workspace 搬入并消费，
`h` 从 S1 保留到 S13，
`A/dw0` 保留到 S7。S11 的 `v_new` resident 在 S13 按真实消费顺序复用为
`do_g/v_decay` 输入槽。
`L1[96,128)` 作为 Cube Stage 从 GM 搬入短生命周期输入的临时区，严格按
`dA0(S5) -> dA1(S5--S7) -> dA(S9) -> ds(S13)`
的顺序复用。`dA1` 由 S5 原址 Fixpipe 生成，其余 tensor 在对应 Cube Stage 内搬入。该地址
不与下一任务组 S0 的 `kbg=L1[320,384)` 或 S1 的 `vb=L1[384,448)` 重叠。

任何原始输入在对应计算路径中
均只从 GM 读取一次。

#### 5.3.1 L1 生命周期与任务组边界检查

逐 Stage 检查后的关键 L1 生命周期如下：

| 地址 | tensor 生命周期 | 后续复用条件 |
|---|---|---|
| `L1[0,32)` | `A`：S3--S7；`do`：S11；`q`：S13 | 每个语义的末次 MTE1 消费后才可覆盖 |
| `L1[32,96)` | `dh`：S13 双槽 | S13 `v_decay @ DHc` 进入 L0 后可复用 |
| `L1[96,128)` | `dA0 -> dA1`：S5--S7；`dA`：S9；`ds`：S13 | 每个 tensor 的消费 Stage 完成后才可覆盖 |
| `L1[128,256)` | `h`：S1 写，S13 读 | S13 完成后才可复用 |
| `L1[256,320)` | `du`：S1--S3；`k`：S5--S13 | `du` 末次消费后切换为 `k` |
| `L1[320,384)` | `kbg`：S0--S3；`v_new`：S11；`do_g/v_decay`：S13 内顺序复用 | 前一语义末次消费后才可切换 |
| `L1[384,448)` | `vb`：S1；`kb`：S9 | 两者分别从 workspace 搬入并在本 Stage 消费 |
| `L1[448,512)` | `dw0`：S1--S7 | S7 完成后才可复用 |

检查结论：同一任务组内部不存在仍未消费就被后续 Stage 覆盖的 L1 地址。短生命周期
Cube 输入从 GM 搬入 `L1[96,128)`，不会与下一任务组 S0 `kbg` 地址重叠。但长生命周期 resident 仍有意复用下一任务组的 S0/S1 地址，
例如 `v_new`、`kb` 和 `dw0`。因此当前单份 resident 设计只允许同一任务组完成
S0--S15 完整闭环后再启动下一任务组；若未来需要任务组间重叠执行，必须为所有跨组
冲突的 resident 增加独立 bank 或改用 GM workspace，不能只调整核间 flag。

### 5.4 UB 全空间分配

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
S10 与 S9/S11 交错执行（S9 输出写 workspace，S11 ds0 直写 UB）:
    大型 tensor 208 KiB + FP32 小向量 3.5 KiB           = 211.5 KiB
S12:
    k[2] + gateDA[2] + (dkbg0[2] -> dkPrepare[2])
    + dkb[2] + dkbT[2] + (ds0[2] -> ds[2])
    + (do[2] -> doG[2]) + (vNew[2] -> vDecay[2])        = 224 KiB
    FP32 小向量和可选 betaRaw[2]                          = 4 KiB 物理预留
    S12 total                                             = 228 KiB
S14:
    k[2] + dkPrepare[2] + dqHv[2] + dkIntra[2]
    + dkBase[2] + q[2]                                   = 192 KiB
    FP32 小向量                                           <= 3 KiB
S15:
    dqAcc[2] + dkAcc[2] + dq/dk partial input + q + k   = 128 KiB
    可选 qRstd + kRstd                                   = 0.5 KiB 物理预留
```

UB 完整峰值为 S12 的 228 KiB，剩余 20 KiB。不存在固定不可借用区域：

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
S11 的 `ds0` 则直接 Fixpipe 到 `UB[176,192)`。

Vector Stage 的完整 shape 结果按生命周期复用已消费的 UB 地址，例如
`do -> do_g`、`v_new -> v_decay`、`dkbg0 -> dk_prepare`、
`dk_base -> dk_raw_hv`。
`AdA/dq_hv` 等只在单次 VF 表达式内部使用，不分配跨 Stage 完整 tensor；
S14 的点积归约只在寄存器和 `dg_scratch` 中存在；`q_rstd/k_rstd` 仅由 S15
在启用 norm backward 时搬入。

### 5.5 跨 stage 数据保留原则

```text
Cube -> Vector 且容量可行：Fixpipe -> UB[2] 保存空间 -> Vector 消费
Cube -> Vector 且容量不可行：Fixpipe -> GM workspace -> UB[2] -> Vector 消费
Cube -> Cube 且非算子输出：Fixpipe -> L1[4] 保存空间，保留到末次 Cube 消费
Vector -> Cube：UB -> GM -> L1[4]，由消费它的 Cube Stage 完整搬入
Vector -> Vector 且非算子输出：保留 UB[2]，直到末次 Vector 消费
算子最终输出：Vector 从 UB 写 GM
```

矩阵乘输入第一次从 GM 搬入 L1 后，只要后续 Cube 仍会使用，就必须保留在四份 L1
保存空间中；不允许因 stage 切换重复从 GM 搬运。Cube 结果若将由 Vector 使用，
优先落入两份 UB；仅当完整依赖 DAG 的并发活跃集合超过 UB 总容量时，
允许写一次 GM workspace，再由 Vector 完整读取一次。当前方案中的
`dkb/dkb_t` 使用该回退路径；`ds0` 容量可行，直接 Fixpipe 到 UB。

Vector 输入第一次从 GM 搬入 UB 后，只要后续 Vector Stage 仍会使用，就必须保留在
两份 UB；不允许因 stage 切换再次从 GM 搬运。当前方案中 `k` 在 S0--S14、
`v` 在 S0--S6、`beta` 在 S0--S12 的 Vector 生命周期内连续驻留；原始 `g` 在 S0
完成全部派生量后释放。
同一原始输入若同时参与 Cube 和 Vector，两条计算路径相互独立：允许各有一次
`GM -> L1` 和一次 `GM -> UB`，但任一路径内部均不得发生第二次 GM 读取。

Stage 切换时，完整 `UB[0,248)` 和 `L1[0,512)` 内已结束生命周期的地址均可改变
tensor 语义。任何复用都必须以真实末次消费者完成为前提。
