# chunk_gdn_fwd_prepare 全融合 Ascend C 算子设计

## 1. 目标

本文设计 GDN forward `prepare` 全融合算子 `ChunkGdnFwdPrepare`。
它把当前拆开的前向准备链收进一个 kernel：

```text
ChunkGdnFwdPrepare
  -> L2Norm(q/k)
  -> fused gate + chunk-local cumsum
  -> beta sigmoid
  -> DotKkt / 严格下三角 L
  -> SolveTri (VCS + MBH) 得到 A = (I+L)^{-1}
  -> RecomputeWU 得到 w/u
```

算法最长路径与 `design/整体流程.png` 一致。数学边界对齐：

- Triton / FLA：`l2norm_fwd`、`chunk_local_cumsum`、`chunk_scaled_dot_kkt_fwd`、
`solve_tril`、`recompute_w_u`；
- 仓内已有单算子：`chunk_scaled_dot_kkt`、`solve_tri`（VCS+MBH）、
`recompute_w_u_fwd`、`chunk_local_cumsum`；
- CPU标杆在：/data/w00933206/ops/Gate_of_Babylon/tests/gdn/cpu_golden.py
- 接口：同目录 `接口设计.md`。



## 2. 范围与 shape

第一版目标：

```text
SoC               Ascend 950 only
q/k layout        [B, HK, T, K]     BNSD
value/gate layout [B, HV, T, V] / [B, HV, T]
A layout          [B, HV, T, BT]
K                 128
V                 {128, 256}
chunk_size        64
dtype             q/k/v/w/u/A 为 bf16；g/beta 允许 fp32 或 bf16
                  g_cumsum / beta_out / rstd 为 fp32
GVA               HV % HK == 0, G=HV/HK in {1,2,3,4}
use_qk_l2norm     第一版只支持 True
use_exp2          第一版只支持 True
use_gate_in_kernel 第一版只支持 False
```

目标 case：

```text
B=1, HK=8, HV=8, T=1792, K=V=128, BT=64, BF16
```

`BT/K` 是固定规格：`BT=64`、`K=128`。host 侧必须明确拦截其它值。
`V` 只允许 128 或 256。尾 chunk 和 varlen 仍允许当前有效长度 `M<64`。
算子定义和编译配置只注册 `ascend950`。

符号：

```text
B   batch size
T   定长每 batch token 数；varlen 时为 packed token 总数
HK  q/k head 数
HV  value/gate head 数
G   HV / HK
K   q/k dim，固定 128
V   value dim，128 或 256
BT  chunk_size，固定 64
NT  总 chunk 数
M   当前 chunk 有效 token 数，M <= BT
TG  HV task group 大小，TG = (G == 3) ? 3 : 4
hk  当前 hv 映射的 q/k head，hk = hv / G
RCP_LN2 = 1.4426950408889634    # 1 / ln(2)，use_exp2 路径把 exp 换成 exp2
eps     = 1e-6                  # L2Norm
```

主要张量：


| 张量               | Shape         | 说明                                     |
| ---------------- | ------------- | -------------------------------------- |
| `q, k`           | `[B,HK,T,K]`  | 原始 query/key，bf16                      |
| `v`              | `[B,HV,T,V]`  | value，不做 L2Norm                        |
| `g, beta`        | `[B,HV,T]`    | **原始** gate / beta                     |
| `a_log`          | `[HV]`        | fused gate 参数，可选                       |
| `dt_bias`        | `[HV]`        | fused gate bias，可选                     |
| `cu_seqlens`     | `[seqNum+1]`  | varlen 序列边界                            |
| `chunk_indices`  | `[2*NT]`      | `(seqIdx, localChunkIdx)` pair         |
| `q_hat, k_hat`   | `[B,HK,T,K]`  | L2Norm 后的 q/k                          |
| `q_rstd, k_rstd` | `[B,HK,T]`    | L2Norm 保存值，fp32                        |
| `beta_out`       | `[B,HV,T]`    | sigmoid 后（或 fp32 拷贝）的 β                |
| `g_cumsum`       | `[B,HV,T]`    | chunk-local cumsum × `RCP_LN2` 后的 `g'` |
| `A`              | `[B,HV,T,BT]` | `(I+L)^{-1}`，与 k 同 dtype               |
| `w`              | `[B,HV,T,K]`  | `A @ kbg`                              |
| `u`              | `[B,HV,T,V]`  | `A @ vb`                               |


golden 单 chunk 中将 `A` 视为逻辑 `[BT,BT]`，将 `q/k` 视为逻辑 `[BT,K]`。 kernel 的 GM layout 保持 BNSD。

## 3. 算子接口

接口名称固定为：

```text
GE/Ascend C 算子名：ChunkGdnFwdPrepare
ACLNN workspace：  aclnnChunkGdnFwdPrepareGetWorkspaceSize
ACLNN execute：    aclnnChunkGdnFwdPrepare
Python：           fla_npu.ops.ascendc.chunk_gdn_fwd_prepare
```

字段、可选 tensor 推断规则、ACLNN 两段式签名以 `接口设计.md` 为准。
本文只强调实现必须遵守的约束：

- `use_qk_l2norm` 由 `qHat/kHat/qRstd/kRstd` 是否非空推断，第一版 kernel 按 True 实现；
- `use_gate_in_kernel` 由 `aLogOptional` 非空推断；`dt_bias` 只能在 gate 打开时出现；
- `use_beta_sigmoid` 由 `betaEffOptional` 非空推断；
- `use_exp2=True`：所有 `exp(Δg)` 与 `exp(g')` 都走 `exp2`，cumsum 输出乘 `RCP_LN2`；
- `allow_neg_eigval=True` 且已做 sigmoid 时：`beta_eff = 2 * sigmoid(beta)`，否则 `beta_eff = sigmoid(beta)`；
- 变长 `B=1`，`cu_seqlens` 与 `chunk_indices` 须同时非空（或 host 由 `cu_seqlens` 生成后者）。



## 4. Stage 0--7 完整数学语义

为了提高 AIC/AIV 并行度，计算被切成 **Stage 0--7**。每个 Stage 基本只走 Cube 或只走 Vector。

### Stage 总表


| Stage | 类型     | 计算                                           |
| ----- | ------ | -------------------------------------------- |
| S0    | Vector | Mask / I_NZ / I_VCS / Zero，常驻 UB 或 L1        |
| S1    | Vector | L2Norm(q/k)、fused gate、cumsum、`g'`、β sigmoid |
| S2    | Cube   | `kkt = k' @ k'^T`                            |
| S3    | Vector | `exp2(g'),`构造严格下三角 `L`，VCS 求两个 32×32 叶子逆     |
| S4    | Cube   | `I@I`，`tmp =LeafInvRight @ (-L)`             |
| S5    | Cube   | `I @ LeafInvRight`，`A = tmp @ LeafInvLeft`   |
| S6    | Vector | `vb = v*β`，`kbg = k'*β*exp2(g')`             |
| S7    | Cube   | `u = A @ vb`，`w = A @ kbg`                   |




### Stage 0：Vector，辅助矩阵与 mask

需要完成的计算：

```text
生成下三角掩码mask与后续 VCS/MBH 反复用到的常数阵，常驻 UB 或 L1，避免每个 chunk 重算。

I_nz     = NZ(I_64)                         # [BT, BT] fp32 NZ，分型 16×8
I_vcs    = concat_K(I_32, I_32)             # [32, 64] fp32，两个 32×32 单位阵沿 K 拼接
Zero     = 0_{BT×BT}                        # [BT, BT] fp32 NZ
Mask     = 1{i > j}                         # 严格下三角 bit-mask，[BT, BT]/8
```

`I_nz` 给 MBH 的 Cube 路径用。FP32 Cube 分型是 `16(M)×8(N)`，单位阵按该分型写成 NZ，才能直接 `LoadData` 进 L0。  
`I_vcs` 给 S3 VCS 用：一次按 `32×64` 看待两个 32×32 对角叶子，避免两次 32×32搬运。 

`Zero` 给 MBH 清非对角槽、以及尾块 padding。  
`Mask` 给 S3 把 `kkt` 打成严格下三角。

内存分配（单位 KiB）：

```text
L1:
L1[480, 496)   I_nz          64*64*4 = 16 KiB
L1[496, 512)   Zero          64*64*4 = 16 KiB
L1 小计        32 KiB

UB驻留：（之后的stage都存在，驻留区的数据不发生改变）
UB[0, 0.5)     Mask          64*64/8 = 0.5 KiB
UB[1, 9)       I_vcs         32*64*4 = 8 KiB  

UB临时：
UB[9, 25) I_nz 的 UB 副本  16 KiB   # 生成后拷到 L1，UB 可在 S1 前释放；
UB[25, 31) Zero 的 UB 副本  16 KiB   # 生成后拷到 L1，UB 可在 S1 前释放；
```

操作流程：

1. 每个 AICore 在进入 chunk 循环前由 AIV 生成上述四份常数。两个 AIV 可以各生成一份 UB 副本；L1 上的 `I_nz/Zero` 由 AIV0 写一份，AIC 只读。
2. `I_nz`、`Zero` 以 NZ fp32 写入 `L1[480， 512)`。
3. `Mask`、`I_vcs` 留在 UB。
4. 本 Stage 不读 GM 输入，UB到L1的数据通过MTE3直接拷贝，都可以通过一次Datacopy指令实现。

---



### Stage 1：Vector，L2Norm / gate / cumsum / β

需要完成的计算：

```text
# L2Norm，按 hk，eps=1e-6
q_rstd[t] = rsqrt(sum_d q[t,d]^2 + eps)          # [BT]
k_rstd[t] = rsqrt(sum_d k[t,d]^2 + eps)          # [BT]
q'[t]     = q[t] * q_rstd[t]                     # [BT, K]
k'[t]     = k[t] * k_rstd[t]                     # [BT, K]

# fused gate，按 hv
if use_gate:
    z[t]     = g[t] + (dt_bias[hv] if has_dt_bias else 0)
    g_raw[t] = -exp(a_log[hv]) * softplus(z[t])
else:
    g_raw[t] = to_fp32(g[t])

# chunk-local cumsum，use_exp2=True
g'[t] = RCP_LN2 * sum_{s in [chunk_start, t]} g_raw[s]    # [BT]，写出 g_cumsum
g_exp[t] = exp2(g'[t])                                     # [BT]，S6 的 exp(g')

# beta
s[t] = sigmoid(to_fp32(beta[t]))
if use_beta_sigmoid:
    beta_eff[t] = (2 * s[t]) if allow_neg_eigval else s[t]
else:
    beta_eff[t] = to_fp32(beta[t])

g' 写 GM g_cumsum；
打开 L2Norm 时 `q'/k'/q_rstd/k_rstd` 写 GM；
打开 beta sigmoid 时 `beta_eff` 写 GM；
未打开的可选输出不写。

```

内存分配（单位 KiB）：

```text

UB驻留：（之后的stage都存在，驻留区的数据不发生改变）
UB[0, 9)        S0 辅助驻留 9 KiB
UB[9, 10)       g'[2] + beta[2]     64*4*2*2 = 1 KiB 计算完驻留，beta根据是否开sigmoid来驻留原始beta或者betaEff

UB临时：
UB[10, 75)       输入 db：q+k+g
                 (64*128*2*2 + 64*4)*2 ≈ 65 KiB
UB[75, 140)      输出 db：q'+k'+q_rstd+k_rstd
                 (64*128*2 + 64*128*2 + 64*4 + 64*4)*2 ≈ 65 KiB

写入：
L1[0, 16)      k' tile 来自Stage1的任务0(AIV0)
L1[16, 32)     k' tile 来自Stage1的任务1(AIV1)
L1[32, 48)     k' tile 来自Stage1的任务2(AIV0)
L1[48, 64)     k' tile 来自Stage1的任务3(AIV1)
```

操作流程：

1. 当前 AIV 一次搬入本轮 HV 对应的 `q/k/g/beta`；`a_log/dt_bias` 按 head 标量
  搬入 small 槽。尾块仍分配完整 `[BT,...]`，无效行用 `M` 屏蔽。
2. 单次 VF 完成 L2Norm、gate、cumsum、`g_exp`、β。`g'/beta_eff` 写入
  `UB[9, 10)` 驻留；`q'/k'` 走输出 db 后，`k'` 再搬 L1。
3. `q_hat/k_hat/rstd/g_cumsum/beta_out` 按 flag 写 GM。
4. `q` 完成 L2Norm 后不在 UB 保留。`k'` 的 UB 输出 db 在搬进 L1 后释放。

GVA：先按 `hk` 做完 `q'/k'`，再对 `H(hk)` 里本 AIV 承包的 HV 做 gate/β。
`k'` 只从 GM 读一次原始 `k`。同一 `hk` 的 `k'` 在任务组内 4 个 HV 之间共享，4 份槽位仍按任务组预留，避免
G=1 与 G=4 走两套地址。

---



### Stage 2：Cube，Kkt

需要完成的计算：

```text
kkt = k' @ k'.T                         # kkt: [BT, BT] fp32
本 Stage 只做这一条矩阵乘。`kkt` 尚未乘 β、尚未乘 gate、尚未掩成下三角；
那三步都在 S3 Vector。同一 `hk` 只算一次 `kkt`，任务组内 HV 复用。
```

内存分配（单位 KiB）：

```text
临时：
L1[0, 16)      k' tile 来自Stage1的任务0(AIV0)
L1[16, 32)     k' tile 来自Stage1的任务1(AIV1)
L1[32, 48)     k' tile 来自Stage1的任务2(AIV0)
L1[48, 64)     k' tile 来自Stage1的任务3(AIV1)

驻留：
L1[480, 496)   I_nz          64*64*4 = 16 KiB
L1[496, 512)   Zero          64*64*4 = 16 KiB

写入：
UB[140, 156)     Stage2 KKT结果的Ping
UB[156, 172)     Stage2 KKT结果的Pong

```

操作流程：

`k'` 已在 S1 进入 PingPongBuffer，直接从 PingPongBuffer `LoadData`，不必
再从 GM 读。Fixpipe 把 `kkt[2]`（当前 AIV 的 2 个 HV 暂不需要 4 份，因为 S3
是 Vector；同一 `hk` 一份即可）写成 FP32 `[BT,BT]` 到 UB，这里需要把NZ转成ND格式；

1. AIC 对当前 `hk` 做一次 `k' @ k'^T`，k'[64,128];
2. Fixpipe NZ→UB ND，fp32，写入 `UB[140, 172)`。

S2 与 S1 有 AIV→AIC 依赖：必须等 `k'` 在 L1 就绪。`CrossCoreSetFlag` 在 S1
末尾、`CrossCoreWaitFlag` 在 S2 开头。

---



### Stage 3：Vector，构造 L + VCS

需要完成的计算：

```text
# 严格下三角 L，按 hv
g_ij[i,j] = exp2(clip(g'[i] - g'[j], -50, 50))     # [BT, BT]
L[i,j]    = beta_eff[i] * g_ij[i,j] * kkt[i,j]     # i > j
L[i,j]    = 0                                      # i <= j，用 Mask

# 叶子分块，leaf = 32
L00 = L[0:32,  0:32]
L11 = L[32:64, 32:64]
L10 = L[32:64, 0:32]                               # 供 MBH，不必单独驻留整块外

# VCS：单位下三角 (I+Lii)^{-1}，i=0,1
X_R = (I_32 + L00)^{-1}                            # LeafInv_Right，other
X_L = (I_32 + L11)^{-1}                            # LeafInv_left，driving
LeafInv = concat_K(X_R, X_L)                       # [32, 64] fp32
```

内存分配（单位 KiB）：

```text

UB驻留：
UB[0, 10)        Mask + I_vcs + g'/β 

UB临时：
UB[140, 156)     Stage2 KKT结果的Ping
UB[156, 172)     Stage2 KKT结果的Pong

UB[10, 18)       L_vcs 临时              32*64*4 = 8 KiB   # 当前叶子 (I+Lii)
UB[18, 34)       res_vcs db              32*64*4*2 = 16 KiB
UB[34, 50)       L_full 或 -L            64*64*4 = 16 KiB  # 给 S4/S5 Cube


写入：
L1[64, 80)      -L tile 来自Stage3的任务0(AIV0)
L1[80, 96)      -L tile 来自Stage3的任务1(AIV1)
L1[96, 112)     -L tile 来自Stage3的任务2(AIV0)
L1[112, 128)    -L tile 来自Stage3的任务3(AIV1)

L1[128, 144)     LeafRight tile 来自Stage3的任务0(AIV0)
L1[144, 160)     LeafRight tile 来自Stage3的任务1(AIV1)
L1[160, 176)     LeafRight tile 来自Stage3的任务2(AIV0)
L1[176, 192)     LeafRight tile 来自Stage3的任务3(AIV1)

L1[192, 208)      LeafLeft tile 来自Stage3的任务0(AIV0)
L1[208, 224)      LeafLeft tile 来自Stage3的任务1(AIV1)
L1[224, 240)      LeafLeft tile 来自Stage3的任务2(AIV0)
L1[240, 256)      LeafLeft tile 来自Stage3的任务3(AIV1)


```

操作流程：

1. VF 读 `kkt`、`g'`、`beta_eff`、`Mask`，原地把 `kkt` 变成 `L`，再取负得到`-L`。
2. 从 `L` 抽出 `L00/L11` 到 `L_vcs`，用 `I_vcs` 做初值跑VCS迭代出两个叶子结点，写出`res_vcs`。
3. `-L` 写入 L1；`LeafInv` 中的两个叶子结点分别写入L1的LeafRight和LeafLeft。

---



### Stage 4：Cube，MBH 独立矩阵乘

需要完成的计算：

数学：

```text
Y = I + LeafLeft @ (-L)                    # driving = LeafLeft / X_L
```

与 `solve_tri` 的 B/C两步对齐（`cur=64`，`blockSize=32`，`is_lower`）：

```text
B:  L0C_X = I @ I                          # 累加器置 I
C:  L0C_X = LeafLeft @ (-L) + I  -> Y     
```

内存分配（单位 KiB）：

```text
L1驻留：
L1[480, 496)   I_nz          64*64*4 = 16 KiB
L1[496, 512)   Zero          64*64*4 = 16 KiB

L1临时：
L1[64, 80)      -L tile 来自Stage3的任务0(AIV0)
L1[80, 96)      -L tile 来自Stage3的任务1(AIV1)
L1[96, 112)     -L tile 来自Stage3的任务2(AIV0)
L1[112, 128)    -L tile 来自Stage3的任务3(AIV1)

L1[128, 144)     LeafRight tile 来自Stage3的任务0(AIV0)
L1[144, 160)     LeafRight tile 来自Stage3的任务1(AIV1)
L1[160, 176)     LeafRight tile 来自Stage3的任务2(AIV0)
L1[176, 192)     LeafRight tile 来自Stage3的任务3(AIV1)

L1[192, 208)      LeafLeft tile 来自Stage3的任务0(AIV0)
L1[208, 224)      LeafLeft tile 来自Stage3的任务1(AIV1)
L1[224, 240)      LeafLeft tile 来自Stage3的任务2(AIV0)
L1[240, 256)      LeafLeft tile 来自Stage3的任务3(AIV1)

写出：
Workspace[0,16)   L0C_X = LeafLeft @ (-L) + I  -> Y tile 来自Stage4的任务0
Workspace[16,32)  L0C_X = LeafLeft @ (-L) + I  -> Y tile 来自Stage4的任务1
Workspace[32,48)  L0C_X = LeafLeft @ (-L) + I  -> Y tile 来自Stage4的任务2
Workspace[48,64)  L0C_X = LeafLeft @ (-L) + I  -> Y tile 来自Stage4的任务3

L1[0, 16)      Y tile 来自Stage4的任务0
L1[16, 32)     Y tile 来自Stage4的任务1
L1[32, 48)     Y tile 来自Stage4的任务2
L1[48, 64)     Y tile 来自Stage4的任务3

```

操作流程：

1. 从L1驻留的I_nz读取数据到L0完成mmad，且init_flag=True
2. 从L1的LeafLeft和-L读取数据到L0完成mmad，且init_flag=False，累加到1的MMAD的结果上
3. 把L0C_X = LeafLeft @ (-L) + I  -> Y 的结果从L0C通过FixPipe写入Workspace，这里需要使能splitChanner将16*16的fp32分型转换为16*8的fp32分型，然后再通过MTE2将workspace中的数据搬运回L1上的Y tile储存区

---



### Stage 5：Cube，MBH 合并得到 A

需要完成的计算：

数学：

```text
A = LeafLeft + Y @ LeafRight           # other = LeafRight / X_R
```

```text
tmp = I @ LeafLeft           
A = Y @ LeafRight + tmp   
```

内存分配（单位 KiB）：

```text
L1驻留：
L1[480, 496)   I_nz          64*64*4 = 16 KiB
L1[496, 512)   Zero          64*64*4 = 16 KiB

L1临时：
L1[0, 16)      Y tile 来自Stage4的任务0
L1[16, 32)     Y tile 来自Stage4的任务1
L1[32, 48)     Y tile 来自Stage4的任务2
L1[48, 64)     Y tile 来自Stage4的任务3

L1[64, 80)      -L tile 来自Stage3的任务0(AIV0)
L1[80, 96)      -L tile 来自Stage3的任务1(AIV1)
L1[96, 112)     -L tile 来自Stage3的任务2(AIV0)
L1[112, 128)    -L tile 来自Stage3的任务3(AIV1)

L1[128, 144)     LeafRight tile 来自Stage3的任务0(AIV0)
L1[144, 160)     LeafRight tile 来自Stage3的任务1(AIV1)
L1[160, 176)     LeafRight tile 来自Stage3的任务2(AIV0)
L1[176, 192)     LeafRight tile 来自Stage3的任务3(AIV1)

L1[192, 208)      LeafLeft tile 来自Stage3的任务0(AIV0)
L1[208, 224)      LeafLeft tile 来自Stage3的任务1(AIV1)
L1[224, 240)      LeafLeft tile 来自Stage3的任务2(AIV0)
L1[240, 256)      LeafLeft tile 来自Stage3的任务3(AIV1)

写出：
L1[256, 264)     A tile 来自Stage5的任务0
L1[264, 272)     A tile 来自Stage5的任务1
L1[272, 280)     A tile 来自Stage5的任务2
L1[280, 288)     A tile 来自Stage5的任务3
这里需要注意的是，从L0C拷贝到L1时可以Cast到bf16
```

操作流程：

1. 从L1驻留的I_nz读取数据到L0以及LeafLeft读取数据到L0完成mmad，且init_flag=True
2. 从L1的Y tile读取数据到L0以及LeafRight读取数据到L0完成mmad，且init_flag=False，累加到1的MMAD的结果上
3. 把A = Y @ LeafRight + LeafLeft -> A 的结果从L0C通过FixPipe写入L1，这里需要cast回bf16并且保持NZ格式

尾块：`-L` 只对前 `M` 行有效，其余行保持 0，padding 参与 MBH 后对应行应得到
I 的对应行；GM 只写 `M` 行，避免跨 chunk 覆写。

---



### Stage 6：Vector，vb 与 kbg

需要完成的计算：

```text
vb[t]  = v[t] * beta_eff[t]                      # [BT, V]
kbg[t] = k'[t] * beta_eff[t] * exp2(g'[t])       # [BT, K]
```

对应 `recompute_w_u_fwd` 的向量部分。`exp2(g')` 用驻留 `g'` 现算。

内存分配（单位 KiB）：

```text
UB驻留：
UB[0, 10)        Mask + I_vcs + g'/β 

UB临时：         
UB[32,48)        从GM读取K' Ping
UB[48,64)        从GM读取K' Pong
UB[64,96)        从GM读取V  Ping
UB[96,128)       从GM读取V  Pong

写入：
L1[288, 304)     kbg tile 来自Stage6的任务0
L1[304, 320)     kbg tile 来自Stage6的任务1
L1[320, 336)     kbg tile 来自Stage6的任务2
L1[336, 354)     kbg tile 来自Stage6的任务3

L1[354, 386)     vb tile 来自Stage6的任务0
L1[386, 418)     vb tile 来自Stage6的任务1
L1[418, 440)     vb tile 来自Stage6的任务2
L1[440, 472)     vb tile 来自Stage6的任务3

```

`vb/kbg` 是 Vector→Cube，搬 4 份 BF16 进 L1：

操作流程：

1. 搬入 `v`、`k'`，一次 VF 算出 `vb/kbg`，vf需要先升v和k'的精度，然后计算完后cast回bf16。
2. `vb/kbg` 写 L1 4 份 resident；。

---



### Stage 7：Cube，RecomputeWU

需要完成的计算：

```text
u = A @ vb                                  # [BT, V]
w = A @ kbg                                 # [BT, K]
```

内存分配（单位 KiB）：

```text
L1[256, 264)     A tile 来自Stage5的任务0
L1[264, 272)     A tile 来自Stage5的任务1
L1[272, 280)     A tile 来自Stage5的任务2
L1[280, 288)     A tile 来自Stage5的任务3

L1[288, 304)     kbg tile 来自Stage6的任务0
L1[304, 320)     kbg tile 来自Stage6的任务1
L1[320, 336)     kbg tile 来自Stage6的任务2
L1[336, 354)     kbg tile 来自Stage6的任务3

L1[354, 386)     vb tile 来自Stage6的任务0
L1[386, 418)     vb tile 来自Stage6的任务1
L1[418, 440)     vb tile 来自Stage6的任务2
L1[440, 472)     vb tile 来自Stage6的任务3
```

Fixpipe 直接写 GM `u_out/w_out`，BF16 ND，只写 `M` 行。
V=256 时 `u` 的 N 维是 256，一次 `n=256`,但仍然是1次mmad。

操作流程：

1. `A/vb/kbg` 全部来自 L1 resident，不再读 GM。
2. 完成一次mmad后就把结果写 GM，需要cast回bf16且转化为ND格式。

---



## 5. 先验知识：



### 0. 数据格式：

1. NZ格式，由一个个小分型拼接而成，fp32下分型大小为16*8，bf16下分型大小为16*16；



### 1. 片上内存：

L1的大小为512KB，L0的大小为64KB；L0C的大小为256KB
一个AIC搭配两个AIV，一个AIV中存在一块UB，大小248KB

### 2. 数据搬运：

1. Fixpipe

当数据从L0C搬到GM或者UB时，当前的流水线是Fixpipe，参考文档[https://www.hiascend.com/document/detail/zh/canncommercial/900/API/ascendcopapi/atlasascendc_api_07_0251.html](https://www.hiascend.com/document/detail/zh/canncommercial/900/API/ascendcopapi/atlasascendc_api_07_0251.html)
中的FixpipeParamsArch3510结构体参数说明；
2. 普通的数据搬运如从UB搬运到L1或者从Gm搬运到L1/UB,参考文档
[https://www.hiascend.com/document/detail/zh/canncommercial/latest/API/ascendcopapi/atlasascendc_api_07_0103.html](https://www.hiascend.com/document/detail/zh/canncommercial/latest/API/ascendcopapi/atlasascendc_api_07_0103.html)
从UB搬运到L1的时候，没有随路的Nd2Nz，需要你使用Datacopy来每次搬运8列(fp32)来手动构造L1上的NZ数据

接下来我会告诉你如何把一个ND的bf16的k‘使用Datacopy搬运到L1中去
首先你需要知道，当前的K’的shape为[64,128]，如果把它使用Datacopy从UB的ND转化为NZ，NZ的分型大小为16*16，那么当前我认为需要调用128/16=8次datacopy，搬运参数配置应为(64,1,7,0)；搬运参数的含义在
[https://www.hiascend.com/document/detail/zh/canncommercial/latest/API/ascendcopapi/atlasascendc_api_07_0103.html](https://www.hiascend.com/document/detail/zh/canncommercial/latest/API/ascendcopapi/atlasascendc_api_07_0103.html) DataCopyParams结构体参数定义；

1. L1->L0的数据搬运可以参考：

[https://gitcode.com/cann/asc-devkit/blob/master/examples/01_simd_cpp_api/03_basic_api/03_matrix_compute/load_data_2dv2_l12l0/README.md#5-load2dv2](https://gitcode.com/cann/asc-devkit/blob/master/examples/01_simd_cpp_api/03_basic_api/03_matrix_compute/load_data_2dv2_l12l0/README.md#5-load2dv2)

### 3.矩阵计算MMAD

一次MMAD可以计算L0A @ L0B中的两个Tensor，输出在L0C中，必然是fp32的NZ格式数据(分型大小为16*8)，设置init_flag可以控制是否在输出的时候做累加

### 4. 如何同步

1. 核内同步使用SetFlag、WaitFlag实现，具体可以参考

[https://www.hiascend.com/document/detail/zh/canncommercial/latest/API/ascendcopapi/atlasascendc_api_07_0270.html](https://www.hiascend.com/document/detail/zh/canncommercial/latest/API/ascendcopapi/atlasascendc_api_07_0270.html)
2. 核间同步使用CrossCoreSetFlag、CrossCoreWaitFlag实现，具体可以参考
[https://www.hiascend.com/document/detail/zh/canncommercial/latest/API/ascendcopapi/atlasascendc_api_07_0273.html](https://www.hiascend.com/document/detail/zh/canncommercial/latest/API/ascendcopapi/atlasascendc_api_07_0273.html)
[https://www.hiascend.com/document/detail/zh/canncommercial/latest/API/ascendcopapi/atlasascendc_api_07_0274.html](https://www.hiascend.com/document/detail/zh/canncommercial/latest/API/ascendcopapi/atlasascendc_api_07_0274.html)

本 kernel 是 `KERNEL_TYPE_MIX_AIC_1_2`：一个 AIC 配两个 AIV。Vector 打给 Cube 的 flag 用

```text
CrossCoreSetFlag<0x4, PIPE_MTE3>(flagId);
```

Cube 侧用 `PIPE_MTE1` 等（和 AIV 的 `PIPE_MTE3` 配对）：

```text
CrossCoreWaitFlag<0x4, PIPE_MTE1>(flagId);
```

950 上 mode `0x4` 走的是 `set_intra_block` / `wait_intra_block`，不是跨核 FFTS。同一 `flagId` 被两个 AIV 各自 `Set` 时，硬件把 AIV1 映射到 `flagId + 16`。CANN 的 `SyncAll` 也是 AIC 连等两次：`id` 和 `id + 16`。

因此 AIV 侧仍然按任务号打 0/1/2/3，AIC 侧必须按 AIV 身份重映射：

```text
每一轮 4 个 tile：AIV0 做 task 0、2，AIV1 做 task 1、3。

AIV0: CrossCoreSetFlag<0x4, PIPE_MTE3>(0);   // task 0 做完
      CrossCoreSetFlag<0x4, PIPE_MTE3>(2);   // task 2 做完
AIV1: CrossCoreSetFlag<0x4, PIPE_MTE3>(1);   // task 1 做完
      CrossCoreSetFlag<0x4, PIPE_MTE3>(3);   // task 3 做完

AIC : Wait(0);    // AIV0 task 0
      Wait(17);   // AIV1 task 1  →  1 + 16
      Wait(2);    // AIV0 task 2
      Wait(19);   // AIV1 task 3  →  3 + 16
```

AIC 如果直接 `Wait(1)`，等的是 **AIV0 的 flag 1**。AIV0 从不打 1，AIV1 打的 1 落在硬件 17 上，Cube 会永远卡住。Stage0 只由 AIV0 打 `flag=7`，AIC 等 7 即可，不要再等 `7+16`。

1. MIX 下 pack 循环必须用「Cube 个数」做步长。

本机 `PRINTF` 前缀是 `[AIV Block 0/56]`，但 `GetBlockNum()` **在 AIV 上返回的是 Cube 数 28**（与 AIC 相同），不是 56。`GetBlockIdx()` 仍是 `2 * cubeId + subBlock`（AIV0=0, AIV1=1, …）。

若 AIV 再写 `numCore = GetBlockNum() / 2` 会变成 14，和 AIC 的 28 错开，第二轮 flag 对不上会挂。

正确写法：两边都按 Cube 核号 round-robin，**步长都是 28**。

```text
AIV: coreIdx = GetBlockIdx() / 2;
     numCore = GetBlockNum();       // 28，不要再 /2
AIC: coreIdx = GetBlockIdx();
     numCore = GetBlockNum();       // 28

nPacks = Ceil(totalTiles / 4)       // 本 case 224/4 = 56
for pack = coreIdx; pack < nPacks; pack += numCore
    AIV: t = subBlock; t < 4; t += 2    // AIV0: 0,2  AIV1: 1,3
         workId = pack * 4 + t
         Stage1 后 SetFlag(t)
    AIC: Wait 0, 17, 2, 19
```

未消耗的 `SetFlag` 再 `Set` 同一 id 也会挂；每一轮 AIC 必须把 0/17/2/19 四个 wait 都做完，即使这一轮 Cube 还没有真正算。

### 5. 如何写Vector上的矢量计算（VF）

参考[https://www.hiascend.com/document/detail/zh/canncommercial/latest/API/ascendcopapi/atlasascendc_api_07_0314.html](https://www.hiascend.com/document/detail/zh/canncommercial/latest/API/ascendcopapi/atlasascendc_api_07_0314.html)

参考VF优化手册
[https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/programug/Ascendcopdevg/docs/zh/guide/%E7%AE%97%E5%AD%90%E5%AE%9E%E8%B7%B5%E5%8F%82%E8%80%83/SIMD%E7%AE%97%E5%AD%90%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96/%E7%9F%A2%E9%87%8F%E8%AE%A1%E7%AE%97/VF%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96/VF%E5%BE%AA%E7%8E%AF%E4%BC%98%E5%8C%96.md](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/programug/Ascendcopdevg/docs/zh/guide/%E7%AE%97%E5%AD%90%E5%AE%9E%E8%B7%B5%E5%8F%82%E8%80%83/SIMD%E7%AE%97%E5%AD%90%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96/%E7%9F%A2%E9%87%8F%E8%AE%A1%E7%AE%97/VF%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96/VF%E5%BE%AA%E7%8E%AF%E4%BC%98%E5%8C%96.md)
[https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/programug/Ascendcopdevg/docs/zh/guide/%E7%AE%97%E5%AD%90%E5%AE%9E%E8%B7%B5%E5%8F%82%E8%80%83/SIMD%E7%AE%97%E5%AD%90%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96/%E7%9F%A2%E9%87%8F%E8%AE%A1%E7%AE%97/VF%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96/%E6%8C%87%E4%BB%A4%E5%8F%8C%E5%8F%91%E4%BC%98%E5%8C%96.md](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/programug/Ascendcopdevg/docs/zh/guide/%E7%AE%97%E5%AD%90%E5%AE%9E%E8%B7%B5%E5%8F%82%E8%80%83/SIMD%E7%AE%97%E5%AD%90%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96/%E7%9F%A2%E9%87%8F%E8%AE%A1%E7%AE%97/VF%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96/%E6%8C%87%E4%BB%A4%E5%8F%8C%E5%8F%91%E4%BC%98%E5%8C%96.md)
[https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/programug/Ascendcopdevg/docs/zh/guide/%E7%AE%97%E5%AD%90%E5%AE%9E%E8%B7%B5%E5%8F%82%E8%80%83/SIMD%E7%AE%97%E5%AD%90%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96/%E7%9F%A2%E9%87%8F%E8%AE%A1%E7%AE%97/VF%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96/%E8%BF%9E%E7%BB%AD%E9%9D%9E%E5%AF%B9%E9%BD%90%E5%9C%BA%E6%99%AF%E4%BC%98%E5%8C%96.md](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/programug/Ascendcopdevg/docs/zh/guide/%E7%AE%97%E5%AD%90%E5%AE%9E%E8%B7%B5%E5%8F%82%E8%80%83/SIMD%E7%AE%97%E5%AD%90%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96/%E7%9F%A2%E9%87%8F%E8%AE%A1%E7%AE%97/VF%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96/%E8%BF%9E%E7%BB%AD%E9%9D%9E%E5%AF%B9%E9%BD%90%E5%9C%BA%E6%99%AF%E4%BC%98%E5%8C%96.md)

### 6. 代码架构

代码架构与内存管理参考：
 `/data/w00933206/ops/flash-linear-attention-npu/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gdn_fwd_prepare/op_kernel/arch35/solve_tri_ascend950_64.h` 

### 7. GVA

仅支持 GVA 比例 `1:1`--`1:4`，即 `G=HV/HK ∈ {1,2,3,4}`。核内使用动态 HV 任务组： `G=3` 时组大小取 3，`G=1/2/4` 时取 4。

以下按单个 `(chunk, hv)` 描述。令 `hk = hv / G`，当前 chunk 有效长度为 `M`。

单个 `(chunk, hv)` 的基础 shape：

```text
q, k, q_hat, k_hat          # [BT, K]
v, u                        # [BT, V]
w                           # [BT, K]
g, beta, g', beta_eff       # [BT]
A, L, kkt                   # [BT, BT]
q_rstd, k_rstd              # [BT]
a_log[hv], dt_bias[hv]      # scalar
```

记当前 K head 对应的 GVA value-head 集合为：

```text
H(hk) = { hv_i | floor(hv_i / G) == hk }
```

`k_hat` 与 `kkt` 以 `hk` 为粒度计算一次，组内所有 `hv ∈ H(hk)` 复用同一份
`k'` / `kkt`；`g' / β / L / A / w / u` 按 `hv` 计算。

### 8. 精度调试

使用 AscendC::DumpTensor(srcLocal, 5, dataLen); 可以打印出数据来观察是否符合预期

### 9.原则

1. 搬运一块数据，搬运指令调用次数越少越好
2. 当前阶段务必先保证精度没问题
3. 使用的变量名必须符合语义的驼峰命名，不能随便敷衍了事
4. 每个stage尽量分装成一个函数，方便阅读



## 6. 当前实现步骤

1. 调通下面这条 case（精度对比也用这条）

```text
B=1, HK=8, HV=8, T=1792, K=V=128, BT=64, BF16
G=1, 28 chunks, TG=4
use_qk_l2norm_in_kernel=True, use_exp2=True
gate / beta sigmoid 开启
```

1. 环境：Docker 容器 `w00933206`，conda `py311`，NPU Ascend950PR。容器停了先 `docker start w00933206`。不要在宿主机上编。
2. 编译（只编 fused prepare）

```bash
docker exec w00933206 bash -lc '
source /root/miniconda3/etc/profile.d/conda.sh && conda activate py311
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
export TBE_PARALLEL_COMPILE_ENABLE=0 PARALLEL_COMPILE=0
cd /data/w00933206/ops/flash-linear-attention-npu
bash build.sh --pkg --soc=ascend950 --vendor_name=fla_npu --ops=chunk_gdn_fwd_prepare -j8
'
```

日志里必须出现 `Built target chunk_gdn_fwd_prepare`。编译失败时 `--install` 仍会装上一版 kernel。

1. 安装

```bash
docker exec w00933206 bash -lc '
source /root/miniconda3/etc/profile.d/conda.sh && conda activate py311
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
cd /data/w00933206/ops/flash-linear-attention-npu
bash build_out/fla-npu-fla_npu_linux-x86_64.run --install --quiet
'
```

scoped `--ops=chunk_gdn_fwd_prepare` 会换掉共享 opapi，独立 S1–S7 会暂时不可用，直到做一次完整 vendor 重编。

1. 精度对比

```bash
docker exec w00933206 bash -lc '
source /root/miniconda3/etc/profile.d/conda.sh && conda activate py311
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
cd /data/w00933206/ops/flash-linear-attention-npu
TEST_DEVICE_ID=3 ASCEND_LAUNCH_BLOCKING=1 timeout 90 \
  python fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gdn_fwd_prepare/test/test_chunk_gdn_fwd_prepare.py
'
```

- kernel：`fla_npu.ops.ascendc.chunk_gdn_fwd_prepare`
- golden：`/data/w00933206/ops/Gate_of_Babylon/tests/gdn/cpu_golden.py` 的 `cpu_gdn_fwd_l2norm_to_recompute(..., layout="bnsd")`
- 对比必须在 **CPU** 上做（`A.float().cpu()` vs `ref.a.float().cpu()`）

