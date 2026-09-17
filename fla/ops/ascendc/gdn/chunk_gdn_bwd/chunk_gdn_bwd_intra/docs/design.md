# ChunkGdnBwdIntra Ascend C 融合算子设计

> 方案设计规则版本：`V2`
>
> 本文按仓库 `docs/agents/03-方案设计.md` 的 R01--R21 及“公式书写规范”独立推导。
> Stage 采用“Cube score -> Vector 合并 w/u 与 D -> Cube 合并 w/u 与 dv_local”的三阶段
> 方案。

稳定入口、参数、属性、输出、默认值和支持范围以 [API 文档](api.md) 为唯一接口契约；
本文只描述计算方案、资源分配和同步。

## 1. 目标

本算子融合以下两个设备调用，并按固定顺序返回三个结果：

```text
recompute_w_u_fwd(k, v, beta, A, g, cu_seqlens) -> w, u
chunk_bwd_dv_local(q, k, d_o, g, scale, cu_seqlens, chunk_size) -> dv_local
```

目标 SoC 为 A5（Ascend 950）。模型 case 为 `B=1`、`HK=16`、`HV=32`、`T=11K`、
`K=V=128`、`chunk_size=64`。该 case 在 H20 上的设备耗时为 300 us；融合算子的性能
目标是达到 H20 性能的 0.8 倍，即 A5 设备耗时不超过 `300 us / 0.8 = 375 us`。

数学边界如下。令 `BT=chunk_size`，当前 chunk 有效长度为 `M<=BT`，按 API 定义的
`G=HV/HK` 将 value head `hv` 映射到 q/k head：`hk=floor(hv/G)`；模型 case 为 `G=2`。

```text
Gate(x) = exp2(x), use_exp2=True
          exp(x),  use_exp2=False

bg_hv[BT] = beta_hv[BT] * Gate(g_hv[BT])
A_bg_hv[BT,BT]   = A_hv[BT,BT] * bg_hv[None,:]
A_beta_hv[BT,BT] = A_hv[BT,BT] * beta_hv[None,:]
w_hv[BT,K] = A_bg_hv[BT,BT] @ k_hk[BT,K]
u_hv[BT,V] = A_beta_hv[BT,BT] @ v_hv[BT,V]

S_hk[BT,BT] = k_hk[BT,K] @ q_hk[BT,K]^T
Delta_hv[t,s] = FP32(g_hv[s]) - FP32(g_hv[t])
GateDelta_hv[t,s] = Gate(Delta_hv[t,s]), CausalValidMask(M)[t,s]=1
                    not evaluated,             otherwise
D_hv[t,s] = cast_main(scale * S_hk[t,s] * GateDelta_hv[t,s]), CausalValidMask(M)[t,s]=1
            0,                                                    otherwise
dv_local_hv[BT,V] = D_hv[BT,BT] @ d_o_hv[BT,V]
```

上面三个 `@` 均为 Cube 语义；每个操作数的 shape 已写在表达式中，不用逐元素求和式
描述 matmul。`GateDelta`、scale 和 `CausalValidMask` 是 Vector 逐元素操作。
`S` 在 Cube 累加后以 FP32 驻留在 UB，`Delta/GateDelta` 也以 FP32 计算；`D` 在写入
共享 L1 前转换为主 dtype（FP16/BF16）。`A_bg/A_beta`、`w/u/dv_local` 使用主
dtype。`g/beta` 进入 Vector 计算后统一转 FP32。上述 W/U 变换来自矩阵结合律：
`A @ (diag(bg) @ k) = (A * bg[None,:]) @ k`，U 同理，不改变外部数学语义。

## 2. Stage 0--2 完整详设

下文所有计算均针对输入中的一个 chunk 和一个 value head `hv`。`q/k` 通过
`hk=floor(hv/G)` 读取；A、v、g、beta、d_o 使用当前 `hv`。无效行按 `M` 屏蔽，
矩阵 shape 固定为 `BT=64`。

输入 A/v/g/beta/d_o 的 value-head 轴长度为 `HV`。一个 MixBlock 由 1 个 AIC 和
2 个 AIV 组成；`CG` 表示它一次沿该输入轴连续处理的 head 数。每个 work 的连续切片
从 `hv_begin` 开始，包含 `[hv_begin,min(hv_begin+CG,HV))`。`r` 是切片内偏移。
`G=3` 时取 `CG=3`，其它支持场景取 `CG=4`。全文调度变量只按下表定义：

```text
hv = hv_begin + r                              当前切片中第 r 个 value head
hk = floor(hv/G)                               当前 value head 读取的 q/k head
valid_r = min(CG,HV-hv_begin)                  当前切片的有效 value head 数
leader = r - (r mod G)                         当前 hk 共用的 S_hk 和 k 槽
leaders = [0,valid_r) 内所有不同的 leader，按升序排列

Nchunk = B * ceil(T/BT),         fixed length
         chunk_indices.shape[0], variable length
Nwork = Nchunk * ceil(HV/CG)
AIC_CORE_NUM = 目标 SoC 可供算子使用的 AIC 数
blockDim = min(AIC_CORE_NUM,Nwork)
```

每个实际启动的 MixBlock 由硬件提供 `block_idx`，取值范围为 `[0,blockDim)`。
按 value head 展开的 Cube/Vector 操作都对每个 `r` 执行一份任务；Vector 任务沿结果
的第一个 `BT` 轴切成两个不重叠的半片，AIV0 处理前 `BT/2` 行，AIV1 处理后
`BT/2` 行，因此两个 AIV 都遍历全部 `r=0..valid_r-1`。Stage 0 按唯一 `hk` 计算
Score，同一 `hk` 对应的连续 `hv` 顺序复用 leader 槽中的同一份结果。

TilingKey 直接编码输入确定的 `G=1/2/3/4`，Kernel 由模板参数 `G` 计算 `hk`、leader
和最后消费者。所有 `G` 使用同一调度：work 按 chunk 优先、value-head 切片次序编号，
并按 `work_id=block_idx+n*blockDim` 网格步进。每个 work 始终包含完整的 `valid_r`，
末轮也保持相同任务粒度；`G` 只影响 `hv -> hk` 映射和 Score/k 的复用次数。

模型 case 有 `Nwork=1408`、`blockDim=28`：`block_idx=0..7` 各处理 51 个完整 work，
其余 20 个核各处理 50 个完整 work；每个完整 work 处理 4 个 value head。

Stage 结果如下：

| Stage | 执行单元 | 功能 | 主要空间 |
| --- | --- | --- | --- |
| Stage 0 | Cube | 计算 `S_hk = k_hk @ q_hk^T`，保留 k 给 Stage 2 | 固定预留 4 份 q/k L1 slot；FP32 `S_hk` 写入消费它的 AIV UB |
| Stage 1 | Vector/MTE3 | 每个 AIV 生成 A_bg、A_beta、D 的 `BT/2` 行，同时整理 v/d_o 半片 | 结果在 UB 中 ping/pong，并由 MTE3 写入共享 L1 |
| Stage 2 | Cube | 从共享 L1 读取 D/A_bg/A_beta/v/d_o，计算 `D @ d_o`、`A_bg @ k`、`A_beta @ v` | 每个 `r` 有独立 record，v/d_o 使用两份 ping/pong；Fixpipe 直接写输出 GM |

### 2.1 Stage 0：Cube，计算局部 score

公式：

```text
S_hk[BT,BT] = k_hk[BT,K] @ q_hk[BT,K]^T
```

AIC 在输入 `HV` 轴的当前连续切片中找出映射到不同 q/k head 的首个位置。仅当
`r=leader` 时计算当前 `hk` 的 Score，后续映射到
相同 `hk` 的 value head 不搬 q/k，也不重复 MMAD。所有 leader
的 q/k 先连续发射 MTE2 预取，再按 leader 顺序进入 MTE1/Cube。由于模板要求
`G=3` 时 `CG=3`、其它场景 `CG=4`，完整切片在 `G=1/2/3/4` 时分别预取
`4/2/1/1` 份不同 q/k；最后一个不足 `CG` 的切片预取 `ceil(valid_r/G)` 份。

Stage 0 将同类型数据放在连续地址中，并按最多 4 份任务统一预留。空间布局
（每个 AIC，单位 KiB）：

```text
L1[0,64)      k_hk[4]；主 dtype；每份 16 KiB；S0 按 leader 搬入，保留至对应 S2 W 末次读取
L1[64,128)    q_hk[4]；主 dtype；每份 16 KiB；S0 按转置语义送入 L0B，S0 末次读取后不再使用
L1[128,288)   S1/S2 共用区；S0 不访问
L1[288,512)   空闲；无数据
```

这里只是预留 4 份物理 slot，并非每次都搬入 4 份数据。只有 `r<valid_r` 且为
leader 的 slot 才有效：`G=1` 激活
`r=0/1/2/3` 四个 slot，`G=2` 激活 `r=0/2` 两个 slot，`G=3/4` 只激活
`r=0` 一个 slot；非 leader slot 不搬运、不执行 MMAD，只复用 leader 产生的 Score
和保留的 k。
Stage 0 的 L0A/L0B 各使用两份 16 KiB，L0C 使用两份 16 KiB FP32 `[BT,BT]`；
Stage 0 和 Stage 2 顺序执行并复用这些 L0 资源。每个 leader 的 q/k L1 块使用一个
`qk_l1_mutex[leader]`：MTE2 在该 Mutex 内完成 q/k 搬入，MTE1 随后在同一 Mutex 内读取。
MTE1 保持该 Mutex 到对应 hk 的最后一个 S2 W 完成 L1 读取，确保保留的 k 不被下一
work 覆盖。每个 leader 使用不同 Mutex，后续 q/k 的 MTE2 可以和前一份 Score 的
MTE1/Cube 重叠。
q/k 从 GM 到 L1 只搬入一次；k 的后续复用发生在 L1 内。因此 q/k 的 GlobalTensor
关闭 L2 Cache，避免一次性输入占用 L2。
q 以右操作数转置语义送入 L0B，不建立实体 q.T。MMAD 只对前 `M x M` 有效，其余元素
清零。一个唯一 Score 只计算一次；Fixpipe 将 `S_hk[0:BT/2,:]` 写入 AIV0、将
`S_hk[BT/2:BT,:]` 写入 AIV1 的 leader 槽。同一 `hk` 的连续 value head 依次读取该槽，
最后一个 value head 完成 Stage 1 写回后才释放。例如 `CG=4,G=2` 时，`S_hk0`
使用两个 AIV 的 slot0，`S_hk1` 使用两个 AIV 的 slot2；不复制到 slot1/3。

执行顺序：

1. 生成升序 `leaders`，并等待每个 leader 在两个 AIV 上的 `score_free[part,leader]`。
2. 按 `leaders` 连续发射 q/k 的 GM->L1 搬运；每一对搬运使用
   `Mutex::Lock/Unlock<PIPE_MTE2>(qk_l1_mutex[leader])` 保护自己的固定 L1 块。
   `G=1` 时四份搬运均可在第一份 Score 完成前进入 MTE2 流水，`G=2` 时同理预取两份。
3. 按 `leaders` 顺序使用 `Mutex::Lock<PIPE_MTE1>(qk_l1_mutex[leader])` 读取当前 q/k，
   MTE1 将 q/k 交错装入 L0A/L0B，完成一次 `k @ q^T` MMAD。q 在 S0 末次读取后不再
   使用，k 保留在 leader 对应的 L1 前 16 KiB，供 S2 中映射到同一 hk 的 W 复用；
   MTE1 在该 hk 的最后一个 W 读完 k 后才 `Unlock`，下一 work 随后才能覆盖整块 q/k L1。
4. Fixpipe 向 AIV0/AIV1 的 leader 槽分别交付 Score 的前 `BT/2` 行和后 `BT/2`
   行；该槽保留到同一 `hk` 的最后一个 value head 完成 VF 与 MTE3 写出。

定长、变长、tail 和 `M=0` 按 [API 文档](api.md)处理；无效展开实例不发射 MTE 或 MMAD。

### 2.2 Stage 1：Vector，合并三个矩阵预处理

对当前 value head `hv`，令 `hk=floor(hv/G)`；两个 AIV 各用一次 VF 完成
以下公式中自己负责的第一个 `BT` 轴半片：

```text
gate_hv[s]            = Gate(FP32(g_hv[s]))
bg_hv[s]              = beta_hv[s] * gate_hv[s]
A_bg_hv[BT,BT]        = A_hv[BT,BT] * bg_hv[None,:]
A_beta_hv[BT,BT]      = A_hv[BT,BT] * beta_hv[None,:]
Delta_hv[t,s]         = FP32(g_hv[s]) - FP32(g_hv[t])
GateDelta_hv[t,s]     = Gate(Delta_hv[t,s]), CausalValidMask(M)[t,s]=1 # FP32 [BT,BT]
                        not evaluated,                   otherwise
D_fp32_hv[t,s]        = scale * S_hk[t,s] * GateDelta_hv[t,s], CausalValidMask(M)[t,s]=1
                        0,                                                otherwise
D_hv[BT,BT]           = cast_main(D_fp32_hv)              # main dtype [BT,BT]
```

Stage 1 的每个 Vector 操作有 `valid_r` 份任务，和按 `r` 展开的 Cube 操作任务数一致。
每份任务都沿第一个 `BT` 轴切成两个半片：AIV0/AIV1 均循环执行全部 `r`，但只计算
和写回各自负责的 `BT/2` 行。`CG` 只改变 `valid_r` 的循环上限，不改变处理逻辑和物理布局。

```text
part = 0 for AIV0, 1 for AIV1

A_bg_part[BT/2,BT]   = A_hv[part*BT/2:(part+1)*BT/2,:] * bg_hv[None,:]
A_beta_part[BT/2,BT] = A_hv[part*BT/2:(part+1)*BT/2,:] * beta_hv[None,:]
Delta_part[t,s]      = FP32(g_hv[s]) - FP32(g_hv[t]), t in [part*BT/2,(part+1)*BT/2)
GateDelta_part[t,s]  = Gate(Delta_part[t,s]), CausalValidMask(M)[t,s]=1
                       not evaluated,             otherwise
D_part[t,s]          = cast_main(scale * S_hk[t,s] * GateDelta_part[t,s]), CausalValidMask(M)[t,s]=1
                       0,                                                        otherwise
```

`D` 分支必须先在 FP32 中逐元素计算 `Delta[t,s]=g[s]-g[t]`，再对该差值直接执行一次
`Gate(Delta[t,s])`。禁止先分别计算 `Gate(g[s])`、`Gate(-g[t])` 后相乘，也禁止计算
`Gate(g[s])/Gate(g[t])`；这些等价变换会放大中间值的溢出、下溢和舍入风险。该分支不对
`Delta` 额外 clamp。因果及尾块无效位置不执行 Gate，直接写 0，避免先产生非有限值再乘 0。

同一 AIV 上每个 `hv` 读取 `S_hk[leader]` 中的 `[BT/2,BT]` 半片，并搬入自己负责的
`A_hv[BT/2,BT]`、完整 `g_hv[BT]` 和完整 `beta_hv[BT]`。一次 VF 同时生成
`D_part/A_bg_part/A_beta_part`。三份 ND 行半片在 UB 中连续排列。
当前 `r` 使用共享 L1 中固定的 `record[r]`；下一 work 覆盖前必须等待 Stage 2 释放。

物理空间按最多 4 份任务统一预留。空间布局（每个 AIV，单位 KiB）：

```text
UB[0,32)       S_hk[4]；FP32；每份 8 KiB；S0 仅写 leader 槽，保留至对应 S1 末次读取
UB[32,48)      A_part[4]；主 dtype；每份 4 KiB；S1 MTE2 按有效 r 搬入，对应 input Mutex 被 V 释放后复用
UB[48,72)      record_part[2]；主 dtype ND；每份 12 KiB，依次为 D_part/A_bg_part/A_beta_part；MTE3 读完后复用
UB[72,74)      g/beta raw[4]；输入 dtype；每份 0.5 KiB；当前 r 的 VF 结束后释放
UB[74,106)     v_part[4]；主 dtype；每份 8 KiB；V 完成 ND->NZ 整理后释放
UB[106,138)    d_o_part[4]；主 dtype；每份 8 KiB；生命周期同 v_part
UB[138,170)    v/d_o NZ ping/pong[2]；主 dtype；每份 16 KiB；MTE3 写入共享 L1 后释放
UB[170,248)    空闲；无数据；S1 可复用
```

大型矩阵区按 512 B 对齐，FP32 小向量按 256 B 对齐。每个 `S_hk` slot 为 8 KiB；
`A_part` 每份 4 KiB；每份 `record_part` 为连续的 12 KiB，由三个 4 KiB 半片组成；
每份 `g/beta raw` 槽为 0.5 KiB，其中 g 和 beta 原始输入各占 0.25 KiB。
S1 的 UB 最高已用地址为 170 KiB，连续空闲为 78 KiB。`Delta/GateDelta`
逐行保存在 Vector 寄存器中，不在 UB 中物化完整矩阵。所有循环只使用
`r<valid_r` 的 Score 和 raw slot，其余预留地址不发射读写指令。S2 不新增 AIV UB，
因此整个算子的每个 AIV UB 峰值就是 S1 的 170 KiB。

内部执行顺序：

1. 两个 AIV 都按 `r=0..valid_r-1` 顺序执行。完整 chunk 将当前切片最多 4 个 `r` 的
   A 行半片、完整 g 和完整 beta 分别合并为一条二维 MTE2 搬运；尾块保持逐 `r`
   搬运。每个 hk 的首个 value head等待
   leader Score ready，后续连续 value head 直接复用驻留槽；每个 `hv` 各搬一次
   A 行半片、完整 g 和完整 beta。A_bg 分支复用 `Gate(g)`；D 分支对每个输出元素的
   FP32 `g[s]-g[t]` 直接执行 `Gate`，不复用指数商或正负指数乘积。
2. MTE2 取得当前切片全部有效 `input_mutex[r]`，成组搬入后交给 Vector；Vector 按 `r`
   取得对应输入 Mutex，并取得 `output_mutex[r mod 2]`，一次生成 D/A_bg/A_beta。
   四份输入地址分离，使 MTE2 可在 Vector 消费前两个 `r` 时成组预取后两个 `r`。
3. Vector 同时把 v/d_o 半片整理为 Cube 可读布局。MTE3 取得
   `output_mutex[r mod 2]`，等待 Stage 2 释放即将覆盖的 L1 ping/pong 槽，将
   D/A_bg/A_beta 与 v/d_o 的当前半片写入共享 L1，随后发布 `stage1_ready[part,r]`。

每个活跃 `r` 独立维护一个 input Mutex，两个输出槽各维护一个 output Mutex；无效 `r` 不占 UB slot，
也不发射 MTE2、VF、MTE3 或同步操作。

Stage 1 和 Stage 2 通过每个 MixBlock 的共享 L1 交接。每个 `r` 有一份 24 KiB
record，保存 D/A_bg/A_beta 三个完整矩阵；AIV0/AIV1 分别写前、后 `BT/2` 行。
v 和 d_o 各使用两个 16 KiB ping/pong 槽。定义：

```text
matrix_bytes = BT * BT * sizeof(main_dtype) = 8 KiB
record_bytes = 3 * matrix_bytes              = 24 KiB
L1_RECORD[r] = L1[128 KiB + r*24 KiB] = D | A_bg | A_beta
L1_V_DO_SLOT[r mod 2] = v | d_o
```

每个 AIV 用 MTE3 写出自己负责的行半片；AIC 收到两个 `stage1_ready` 后直接从 L1
读取。AIC 完成当前 record、v、d_o 的最后一次 MTE1 读取后，向两个 AIV 返回
`stage1_free`，下一份使用同一物理槽的数据才可覆盖。对外 workspace 仅保留框架运行时
所需空间，不保存算子中间矩阵。

尾 chunk 的无效行/列在 VF 中置零；`M=0` 不发射 VF 或 GM 写入。

### 2.3 Stage 2：Cube，合并 dv_local/W/U

对当前 `hv`，同一个 Cube Stage 内依次完成三次矩阵乘：

```text
dv_local_hv[BT,V] = D_hv[BT,BT] @ d_o_hv[BT,V]
w_hv[BT,K]        = A_bg_hv[BT,BT] @ k_hk[BT,K]
u_hv[BT,V]        = A_beta_hv[BT,BT] @ v_hv[BT,V]
```

Stage 2 按 `r=0..valid_r-1` 依次处理。D/A_bg/A_beta、v 和 d_o 由两个 AIV 写入
共享 L1 的前、后半行；AIC 等待两个 `stage1_ready` 后直接读取。W 的右操作数 k 不从 GM 重复搬运，
而是读取 Stage 0 中 `leader` 槽保留的 `k_hk`。每份操作内三次 MMAD 顺序使用
L0A/L0B/L0C ping/pong。三个 L0C FP32 结果均由 Fixpipe 转为主 dtype，并直接写入
正式 `dvLocalOut/wOut/uOut` 的 GM 地址。最终输出不再被 Vector 消费，因此 S2
不经过 AIV UB、不触发 CrossCore 输出事件，也不增加 UB 占用。

每个 work 在进入 `r` 循环前计算一次当前 `hv_begin` 的输出 GM 基址；循环内按
head stride 递增。
完整 chunk 在编译期固定 `M=BT=64`，只让尾块路径执行 L1 清零和动态行数处理。

Stage 2 依次处理最多 4 份任务。4 份任务在 L1 中使用各自独立的 record，v 和 d_o
使用两份 ping/pong。
空间布局（每个 AIC，单位 KiB）：

```text
L1[0,64)      k_hk[4]；主 dtype；每份 16 KiB；S0 按 leader 搬入，保留至对应 S2 W 末次读取
L1[64,128)    q_hk[4]；主 dtype；每份 16 KiB；S0 末次读取后不再使用
L1[128,224)   record[4]；主 dtype NZ；每份 24 KiB，依次为 D_hv/A_bg_hv/A_beta_hv；当前 record 的 MTE1 末次读取后释放
L1[224,256)   v_hv ping/pong[2]；主 dtype NZ；每份 16 KiB；当前槽的 MTE1 末次读取后释放
L1[256,288)   d_o_hv ping/pong[2]；主 dtype NZ；每份 16 KiB；生命周期同 v_hv
L1[288,512)   空闲；无数据；S2 可复用
```

L1 最高已用地址为 288 KiB，不超过 A5 的 512 KiB 上限。q/k 与 record 使用
不重叠的地址区间。S2 的 W 从
`k_hk slot[leader]` 读取 S0 保留的 k；该 slot 的起始地址为
`leader*16 KiB`。当前任务 `r` 的 record 槽为 `r`，v 和 d_o 物理槽为
`r mod 2`；所有循环只访问 `r<valid_r` 的数据及其映射到的物理槽。

输入 `HV` 轴的当前切片只激活 `r<valid_r` 的 slot；无效份不发射搬运、MMAD 或事件。
v/d_o 对每个 value head 也只从 GM 搬入一次，其 GlobalTensor 关闭 L2 Cache。

Stage 2 的 L0A/L0B 各使用两份 16 KiB，并与 Stage 0 分时复用 L0 地址。L0C 使用两份 32 KiB FP32 `[BT,128]`
accumulator（共 64 KiB），关闭 unit flag。每份 L0A/L0B 使用一个 `l0ab_mutex[slot]`，
每份 L0C 使用一个 `l0c_mutex[slot]`，MTE1、Cube、Fixpipe 按访问的物理 slot 加锁和解锁。
`dv_local/W/U` 轮换使用两个 L0C slot：前一结果进入 Fixpipe 后，下一次 MMAD 可立即在
另一槽发射 MTE1/Cube。每个 L0C slot 的闭环是
`Cube lock -> MMAD -> Cube unlock -> Fixpipe lock -> 写 GM -> Fixpipe unlock`。两个输出均只写前 `M`
行，尾块其余行不参与有效语义。

执行顺序：

1. Stage 0 完成后，AIC 按 `r` 等待两个 AIV 的 `stage1_ready[part,r]`，确认 record、
   v 和 d_o 的两份行半片已经写入共享 L1。
2. 对每个 `r`，MTE1 依次发射
   `D @ d_o`、`A_bg @ k`、`A_beta @ v`；同一 hk 的多个 W 操作复用 Stage 0 leader 槽中的 k。
   MTE1 完成三个输入读取后发布两个 `stage1_free[part,r]`；在同一 hk 的最后一个 W
   装入 k 后释放 q/k L1。Vector 可在 Cube 消费当前槽时准备另一槽的后续 `r`。
3. 三个 FP32 L0C 结果分别由 Fixpipe 转为主 dtype，直接写 `dvLocalOut/wOut/uOut`
   的正式 GM 地址。Fixpipe 完成当前 L0C 读取后释放对应 slot；完成当前 `HV` 轴
   切片的 `valid_r` 份操作后结束。

### 2.4 Stage 间同步方案

同步分为两层。`Mutex` 只负责同一 AIC 或同一 AIV 内异步流水对本地 L1/UB/L0 地址的
互斥访问；AIC 与 AIV0/AIV1 属于分离模式核间通信，Score 和共享 L1 数据的
ready/free 继续使用 `CrossCoreSetFlag/CrossCoreWaitFlag`。这一边界遵循 Ascend C
[Mutex（ISASI）](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/ascendcopapi/docs/zh/api/SIMD-API/basic_api/sync_control/intra_core_sync/Mutex_ISASI.md)
和 CrossCore 接口定义。

当前 Catlass Resource 通过 `TPipe/TBuf` 建立本地 Tensor。AIC 和每个 AIV 分别使用
`AllocMutexID` 获取下表 Mutex，并在 kernel 退出前调用 `ReleaseMutexID`；不使用固定
MutexID 与框架资源混用。

| 所属核 | Mutex | 数量 | 保护的地址和流水 |
| --- | --- | ---: | --- |
| AIC | `qk_l1_mutex[CG]` | 4 | 每个 leader 的 q/k L1；MTE2 -> S0 MTE1，保留的 k -> S2 MTE1 |
| AIC | `l0ab_mutex[2]` | 2 | L0A/L0B ping/pong；MTE1 -> Cube |
| AIC | `l0c_mutex[2]` | 2 | L0C ping/pong；Cube -> Fixpipe |
| AIV | `input_mutex[CG]` | 4 | 每份 A/g/beta 输入；MTE2 -> Vector |
| AIV | `output_mutex[2]` | 2 | record 与 v/d_o NZ 输出 ping/pong；Vector -> MTE3 L1 |
| AIV | `raw_init_mutex` | 1 | g/beta raw 区初始化完成后，一次性交给 MTE2 覆盖 |

同一 MutexID 上每次 `Lock/Unlock` 必须成对且使用相同 pipe。Vector 同时访问输入和输出
slot 时固定按 input、output 的顺序加锁，并按 input、output 的顺序释放；Cube 同时访问
L0A/L0B 和 L0C 时固定按 `l0ab_mutex`、`l0c_mutex` 的顺序加锁，避免形成循环等待。
MutexID 的编号范围 `0~27` 和 28 个可用 ID 均以单个核为作用域分别核算：AIC 最多
申请 8 个，每个 AIV 最多申请 7 个，均不超过单核上限。AIC、AIV0、AIV1 的数量
不相加，因为它们不共享同一个核内 MutexID 空间。

下列伪代码中，`Lock<PIPE_X>` 表示 PIPE_X 在访问本地地址前等待上一任使用者释放，
`Unlock<PIPE_X>` 表示 PIPE_X 已完成本次访问并把地址交给下一条流水。`score_*` 和
`stage1_*` 是 CrossCore 事件：`ready` 表示生产者已经写完，`free` 表示消费者已经
读完、生产者可以覆盖原地址。

Stage 间只通过下列数据和 CrossCore 信号协作：

```text
Stage0AIC -- S_hk + score_ready --> Stage1AIV -- record/v/d_o + stage1_ready --> Stage2AIC
Stage0AIC <--       score_free  -- Stage1AIV <--                 stage1_free  -- Stage2AIC
Stage0AIC -- 保留 k，并持续持有 qk_l1_mutex --------------------------------> Stage2AIC
```

各 Stage 的同步伪代码如下：

```text
# `with PIPE_X mutex` 等价于 Lock<PIPE_X>、执行缩进块、Unlock<PIPE_X>；
# 同时列出多个 Mutex 时，按列出顺序 Lock 和 Unlock。
# 公共 MMAD helper：左右 L1 已由调用方锁定，只同步 MTE1 -> Cube -> Fixpipe。
MmadFromOwnedL1(left_l1, right_l1, slot, output_gm):
  with PIPE_MTE1 l0ab_mutex[slot]: MTE1 L1 -> L0A/B[slot]
  with PIPE_M l0ab_mutex[slot], l0c_mutex[slot]: Cube -> L0C[slot]
  with PIPE_FIX l0c_mutex[slot]: Fixpipe L0C[slot] -> output_gm

Stage0AIC(chunk_id,hv_begin,valid_r):
  leaders = [0,valid_r) 内所有不同的 r-(r mod G)，按升序排列
  for leader in leaders:
    wait score_free[0,leader], score_free[1,leader] from PIPE_MTE2 # 两个 AIV 都读完后才能覆盖。
    hk = floor((hv_begin+leader)/G)
    with PIPE_MTE2 qk_l1_mutex[leader]:
      MTE2 q/k[chunk_id,hk] -> L1_K_Q[leader]            # 每个唯一 hk 只搬一份 q/k。

  for leader in leaders:
    hk = floor((hv_begin+leader)/G)
    slot = (leader/G) mod 2
    Lock<PIPE_MTE1>(qk_l1_mutex[leader])                 # 保留 k 到 S2 最后一次 W 读取。
    with PIPE_MTE1 l0ab_mutex[slot]: MTE1 q/k -> L0A/B[slot]
    with PIPE_M l0ab_mutex[slot], l0c_mutex[slot]: Cube S_hk -> L0C[slot]
    with PIPE_FIX l0c_mutex[slot]:
      Fixpipe L0C[slot] -> AIV[part].S_hk[leader] for part in [0,2)
      set score_ready[part,leader] for part in [0,2)      # 逐半片通知对应 AIV。

Stage1AIV(part,chunk_id,hv_begin,valid_r,next_item):
  # Process 启动时，PIPE_MTE2 只等待一次 raw 区初始化；各 r 使用独立地址。
  if chunk 是完整的 64 token:
    MTE2 成对预取 A/g/beta/v/d_o[0:valid_r] -> UB input # 先 0/1，再 2/3；消费时可预取下一对。
  else:
    for r in [0,valid_r):
      with PIPE_MTE2 input_mutex[r]: MTE2 A/g/beta/v/d_o[r] -> UB input
  for r in [0,valid_r):
    hv = hv_begin+r
    leader = r-(r mod G)
    vslot = r mod 2
    if r == leader: wait score_ready[part,leader] from PIPE_V
    if vslot 已使用: wait stage1_free[part,上一次使用该 vslot 的 r] from PIPE_MTE3
    with PIPE_V input_mutex[r], output_mutex[vslot]:
      Stage1VF(...) -> ND(D_part|A_bg_part|A_beta_part) # 一次 VF 生成三个矩阵半片。
      Pack(v_part,d_o_part) -> NZ(v_part,d_o_part)      # 同一个 AIV 只整理自己负责的行。
    with PIPE_MTE3 output_mutex[vslot]:
      MTE3 record_part -> L1_RECORD[r]
      MTE3 v/d_o NZ part -> L1_V_DO[vslot]
    set stage1_ready[part,r] from PIPE_MTE3              # 当前半片已经可供 AIC 读取。
    if r == valid_r-1 or floor(hv/G) != floor((hv+1)/G):
      set score_free[part,leader] from PIPE_V            # 该 hk 的 Score 已完成末次消费。

Stage2AIC(chunk_id,hv_begin,valid_r):
  for r in [0,valid_r):
    wait stage1_ready[part,r] for part in [0,2) from PIPE_MTE1 # record/v/d_o 两个半片均已写完。
    hv = hv_begin+r
    leader = r-(r mod G)
    vslot = r mod 2
    MmadFromOwnedL1(D[r],d_o[r],0,dv_local[chunk_id,hv])
    MmadFromOwnedL1(A_bg[r],k[leader],1,w[chunk_id,hv])
    if r == valid_r-1 or floor(hv/G) != floor((hv+1)/G):
      Unlock<PIPE_MTE1>(qk_l1_mutex[leader])
    MmadFromOwnedL1(A_beta[r],v[r],0,u[chunk_id,hv])
    set stage1_free[part,r] for part in [0,2) from PIPE_MTE1 # 三个 MMAD 输入读完，可覆盖 L1。

Schedule<G>(block_idx):
  map work_id -> (chunk_id=floor(work_id/ceil(HV/CG)),hv_slice=work_id mod ceil(HV/CG))
  yield work_id in [block_idx,Nwork) step blockDim       # 所有 G 依次处理 chunk 内的 head 切片。

TilingKey 的模板参数 `G` 是 head 映射的唯一来源。AIC 和 AIV 的热循环直接按编译期
`G` 计算 `leader`、`hk` 和最后消费者；`G=2` 时编译器可将除法化简为移位。

template Process<G>:
  parallel MixBlock block_idx in [0,blockDim):
    parallel:
      AIC:
        allocate 8 AIC MutexID
        for item in Schedule<G>(block_idx):
          Stage0AIC(item); Stage2AIC(item)               # S0 生产 Score，S2 等待 S1 后消费。
        wait final score_free                            # 确认两个 AIV 不再读取 Score UB。
        release 8 AIC MutexID
      parallel AIV part in [0,2):
        allocate 7 AIV MutexID
        with PIPE_V raw_init_mutex:
          initialize raw slots                            # 首个 MTE2 等待后，raw 区可以覆盖。
        with PIPE_MTE2 raw_init_mutex: pass               # 一次性交接，热循环不再重复等待。
        for item in Schedule<G>(block_idx):
          Stage1AIV(part,item,next_item)                   # 当前后半对期间可预取下一 work 前半对。
        release 7 AIV MutexID
```
