# ChunkGdnBwdIntra Ascend C 融合算子设计

> 方案设计规则版本：`V2`
>
> 本文按仓库 `docs/agents/03-solution-design.md` 的 R01--R20 及“公式书写规范”独立推导。
> Stage 采用“Cube score -> Vector 合并 W/U 与 D -> Cube 合并 W/U 与 DV”的三阶段
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
W_hv[BT,K] = A_bg_hv[BT,BT] @ k_hk[BT,K]
U_hv[BT,V] = A_beta_hv[BT,BT] @ v_hv[BT,V]

S_hk[BT,BT] = k_hk[BT,K] @ q_hk[BT,K]^T
Delta_hv[t,s] = FP32(g_hv[s]) - FP32(g_hv[t])
GateDelta_hv[t,s] = Gate(Delta_hv[t,s]), CausalValidMask(M)[t,s]=1
                    not evaluated,             otherwise
D_hv[t,s] = cast_main(scale * S_hk[t,s] * GateDelta_hv[t,s]), CausalValidMask(M)[t,s]=1
            0,                                                    otherwise
DV_hv[BT,V] = D_hv[BT,BT] @ d_o_hv[BT,V]
```

上面三个 `@` 均为 Cube 语义；每个操作数的 shape 已写在表达式中，不用逐元素求和式
描述 matmul。`GateDelta`、scale 和 `CausalValidMask` 是 Vector 逐元素操作。
`S` 在 Cube 累加后以 FP32 驻留在 UB，`Delta/GateDelta` 也以 FP32 计算；`D` 在写入
GM workspace 前转换为主 dtype（FP16/BF16）。`A_bg/A_beta`、`w/u/dv_local` 使用主
dtype。`g/beta` 进入 Vector 计算后统一转 FP32。上述 W/U 变换来自矩阵结合律：
`A @ (diag(bg) @ k) = (A * bg[None,:]) @ k`，U 同理，不改变外部数学语义。

## 2. Stage 0--2 完整详设

下文所有计算均针对输入中的一个 chunk 和一个 value head `hv`。`q/k` 通过
`hk=floor(hv/G)` 读取；A、v、g、beta、d_o 使用当前 `hv`。无效行按 `M` 屏蔽，
矩阵 shape 固定为 `BT=64`。

输入 A/v/g/beta/d_o 的 value-head 轴长度为 `HV`。一个 MixBlock 由 1 个 AIC 和
2 个 AIV 组成；`CG` 表示它一次沿该输入轴连续处理的 head 数。第 `hv_slice` 个连续
切片从 `hv_begin=hv_slice*CG` 开始，
包含 `[hv_begin,min(hv_begin+CG,HV))`。`r=hv-hv_begin` 是切片内偏移，
`hv_r=hv_begin+r`，`hk_r=floor(hv_r/G)`，
`valid_r=min(CG,HV-hv_begin)`。`G=3` 时取 `CG=3`，其它支持场景取 `CG=4`。
按 value head 展开的 Cube/Vector 操作都对每个 `r` 执行一份任务；Vector 任务沿结果
的第一个 `BT` 轴切成两个不重叠的半片，AIV0 处理前 `BT/2` 行，AIV1 处理后
`BT/2` 行，因此两个 AIV 都遍历全部 `r=0..valid_r-1`。Stage 0 按唯一 `hk` 计算
Score，同一 `hk` 对应的连续 `hv_r` 顺序复用 leader 槽中的同一份结果。

Stage 结果如下：

| Stage | 执行单元 | 功能 | 主要空间 |
| --- | --- | --- | --- |
| Stage 0 | Cube | 计算 `S_hk = k_hk @ q_hk^T`，保留 k 给 Stage 2 | 固定预留 4 份 q/k L1 slot；FP32 Score 写入消费它的 AIV UB |
| Stage 1 | Vector | 每个 AIV 各用一次 VF 生成 A_bg、A_beta、D 的 `BT/2` 行 | 每个 AIV 的 S1 工作区最高地址为 81 KiB；三个结果通过 MTE3 写入 GM ring workspace 的不同行半片 |
| Stage 2 | Cube | 从 GM 读入 A_bg/A_beta/D 后计算 `A_bg @ k`、`A_beta @ v`、`D @ d_o` | L1 最高地址为 288 KiB；L0C 结果由 Fixpipe 转为主 dtype 并直接写正式输出 GM，不分配输出 UB |

### 2.1 Stage 0：Cube，计算局部 score

公式：

```text
S_{hk_r}[BT,BT] = k_{hk_r}[BT,K] @ q_{hk_r}[BT,K]^T
```

AIC 在输入 `HV` 轴的当前连续切片中找出映射到不同 q/k head 的首个位置。令
`hv=hv_begin+r`、`hk=floor(hv/G)`，只有 `score_leader(r)` 为真时才计算该 `hk`
的 Score；后续映射到相同 `hk` 的 value head 不搬 q/k，也不重复 MMAD。所有 leader
的 q/k 先连续发射 MTE2 预取，再按 leader 顺序进入 MTE1/Cube。由于模板要求
`G=3` 时 `CG=3`、其它场景 `CG=4`，完整切片在 `G=1/2/3/4` 时分别预取
`4/2/1/1` 份不同 q/k；最后一个不足 `CG` 的切片预取 `ceil(valid_r/G)` 份。

```text
hv(r)           = hv_begin + r                  当前切片中第 r 个 value head
hk(r)           = floor(hv(r) / G)              当前 value head 读取的 q/k head
score_leader(r) = (r == 0) or (hk(r) != hk(r-1))
leaders         = 所有满足 score_leader(r) 的 r，按升序排列
leader_r(r)     = r - (r mod G)                 当前 hk 对应的首个 r
sslot(r)        = leader_r(r)                   同一 hk 的 value head 共用 leader Score 槽
```

Stage 0 将同类型数据放在连续地址中，并按最多 4 份任务统一预留。空间布局
（每个 AIC，单位 KiB）：

```text
L1[0,64)      k_hk slot[4]；主 dtype；每份 16 KiB；S0 按 leader 搬入，保留至对应 hk 的最后一个 S2 W 完成读取
L1[64,128)    q_hk slot[4]；主 dtype；每份 16 KiB；S0 按 leader 搬入，按转置语义送入 L0B，S0 末次读取后释放
L1[128,512)   空闲；无数据；S0 可复用
```

这里只是预留 4 份物理 slot，并非每次都搬入 4 份数据。只有 `r<valid_r` 且为
leader 的 slot 才有效：`G=1` 激活
`r=0/1/2/3` 四个 slot，`G=2` 激活 `r=0/2` 两个 slot，`G=3/4` 只激活
`r=0` 一个 slot；非 leader slot 不搬运、不执行 MMAD，只复用 leader 产生的 Score
和保留的 k。

Stage 0 的 L0A/L0B 各使用两份 16 KiB，L0C 使用两份 16 KiB FP32 `[BT,BT]`；
Stage 0 和 Stage 2 顺序执行并复用这些 L0 资源。所有 leader 的 q/k 搬运按物理块
连续发射，每一块分别发布 `qk_ready[leader]`。MTE1/Cube 等到第一块 ready 即可启动，
不等待后续物理块全部搬完；后续 q/k 的 MTE2 与前一份 Score 的 MTE1/Cube 重叠。
q 以右操作数转置语义送入 L0B，不建立实体 q.T。MMAD 只对前 `M x M` 有效，其余元素
清零。一个唯一 Score 只计算一次；Fixpipe 将 `S[0:BT/2,:]` 写入 AIV0、将
`S[BT/2:BT,:]` 写入 AIV1 的 leader 槽。同一 `hk` 的连续 value head 依次读取该槽，
最后一个 value head 完成 Stage 1 写回后才释放。例如 `CG=4,G=2` 时，`S_hk0`
使用两个 AIV 的 slot0，`S_hk1` 使用两个 AIV 的 slot2；不复制到 slot1/3。

执行顺序：

1. 生成升序 `leaders`，并等待每个 leader 在两个 AIV 上的 `score_free[part,leader]`。
2. 按 `leaders` 连续发射 q/k 的 GM->L1 搬运；每一对 q/k 到达自己的固定 L1 块后，
   立即发布 `qk_ready[leader]`。`G=1` 时四份搬运均可在第一份 Score 完成前进入
   MTE2 流水，`G=2` 时同理预取两份。
3. 按 `leaders` 顺序等待当前 `qk_ready[leader]`，MTE1 将 q/k 交错装入 L0A/L0B，
   完成一次 `k @ q^T` MMAD。q 在 S0 末次读取后释放；k 保留在 leader 对应的 L1
   前 16 KiB，供 S2 中映射到同一 hk 的 W 复用。当前 work 的 S2 完成后，下一 work
   才能覆盖这些 L1 区间。
4. Fixpipe 向 AIV0/AIV1 的 leader 槽分别交付 Score 的前 `BT/2` 行和后 `BT/2`
   行；该槽保留到同一 `hk` 的最后一个 value head 完成 VF 与 MTE3 写出。

定长、变长、tail 和 `M=0` 按 [API 文档](api.md)处理；无效展开实例不发射 MTE 或 MMAD。

### 2.2 Stage 1：Vector，合并 W/U 预处理与 D 门控

对当前 value head `hv=hv_r`，令 `hk_r=floor(hv_r/G)`；两个 AIV 各用一次 VF 完成
以下公式中自己负责的第一个 `BT` 轴半片：

```text
gate_hv[s]            = Gate(FP32(g_hv[s]))
bg_hv[s]              = beta_hv[s] * gate_hv[s]
A_bg_hv[BT,BT]        = A_hv[BT,BT] * bg_hv[None,:]
A_beta_hv[BT,BT]      = A_hv[BT,BT] * beta_hv[None,:]
mask[t,s]             = CausalValidMask(M)[t,s]
Delta_hv[t,s]         = FP32(g_hv[s]) - FP32(g_hv[t])
GateDelta_hv[t,s]     = Gate(Delta_hv[t,s]), mask[t,s]=1 # FP32 [BT,BT]
                        not evaluated,        otherwise
D_fp32_hv[t,s]        = scale * S_{hk_r}[t,s] * GateDelta_hv[t,s], mask[t,s]=1
                        0,                                         otherwise
D_hv[BT,BT]           = cast_main(D_fp32_hv)              # main dtype [BT,BT]
```

Stage 1 的每个 Vector 操作有 `valid_r` 份任务，和按 `r` 展开的 Cube 操作任务数一致。
每份任务都沿第一个 `BT` 轴切成两个半片：AIV0/AIV1 均循环执行全部 `r`，但只计算
和写回各自负责的 `BT/2` 行。`CG` 只改变 `valid_r` 的循环上限，不改变处理逻辑和物理布局。

```text
part(AIV0) = 0                                  前半片
part(AIV1) = 1                                  后半片
vslot(r)   = r mod 2                            每个 AIV 的 Vector UB ping/pong 槽
rows(part) = [part*BT/2,(part+1)*BT/2)           二维 tensor 的第一维行范围

A_bg_part[BT/2,BT]      = A_hv[rows(part),:] * bg_hv[None,:]
A_beta_part[BT/2,BT]    = A_hv[rows(part),:] * beta_hv[None,:]
Delta_part[t,s]         = FP32(g_hv[s]) - FP32(g_hv[t]), t in rows(part)
GateDelta_part[t,s]     = Gate(Delta_part[t,s]), mask[t,s]=1
                          not evaluated,          otherwise
D_part[t,s]             = cast_main(scale * S_hk[t,s] * GateDelta_part[t,s]), mask[t,s]=1
                          0,                                             otherwise
```

`D` 分支必须先在 FP32 中逐元素计算 `Delta[t,s]=g[s]-g[t]`，再对该差值直接执行一次
`Gate(Delta[t,s])`。禁止先分别计算 `Gate(g[s])`、`Gate(-g[t])` 后相乘，也禁止计算
`Gate(g[s])/Gate(g[t])`；这些等价变换会放大中间值的溢出、下溢和舍入风险。该分支不对
`Delta` 额外 clamp。因果及尾块无效位置不执行 Gate，直接写 0，避免先产生非有限值再乘 0。

每个实际启动的 MixBlock 由硬件提供 `block_idx`，并独占下文固定的 Stage 2 L1
地址区间。该 MixBlock 的多个 work 顺序复用同一组地址，只有 `r<valid_r` 的 slot 有效。

同一 AIV 上每个 `hv_r` 读取映射到 leader 槽的 `S_hk_r[BT/2,BT]`。每个 `hv_r` 搬入自己负责的
`A_hv[BT/2,BT]`，以及完整 `g_hv[BT]` 和 `beta_hv[BT]`。一次 VF 对每个输出行
同时生成 A_bg、A_beta 和 D，不拆为多个 VF pass；三个结果转换为主 dtype 后由
MTE3 写入 GM ring workspace 中由该 AIV 负责的不重叠行区间。

物理空间按最多 4 份任务统一预留。空间布局（每个 AIV，单位 KiB）：

```text
UB[0,32)       Score slot[4]；FP32；S0 仅写 leader 槽，保留至映射到同一 hk 的最后一个 S1 操作完成
UB[32,40)      A_part[2]；主 dtype；S1 MTE2 搬入，当前 ping/pong 槽被 VF 末次读取后释放
UB[40,48)      A_bg_part[2]；主 dtype；S1 VF 生成，当前 ping/pong 槽被 MTE3 写入 GM workspace 后释放
UB[48,56)      A_beta_part[2]；主 dtype；S1 VF 生成，当前 ping/pong 槽被 MTE3 写入 GM workspace 后释放
UB[56,58)      g/beta raw -> gate_inv[4]；输入 dtype -> FP32；S1 按 r 搬入，当前 r 的 VF 结束后释放
UB[58,72)      空闲；无数据；S1 可复用
UB[72,80)      D_part[2]；主 dtype；S1 VF 生成，当前 ping/pong 槽被 MTE3 写入 GM workspace 后释放
UB[80,81)      g_fp32/beta_fp32[2]；FP32；S1 按 ping/pong 槽生成，当前槽的 VF 结束后释放
UB[81,248)     空闲；无数据；S1 可复用
```

大型矩阵区按 512 B 对齐，FP32 小向量按 256 B 对齐。每个 Score slot 为 8 KiB；
`A_part/A_bg_part/A_beta_part/D_part` 每份 4 KiB；每份 `g/beta raw -> gate_inv`
槽为 0.5 KiB，其中 g 和 beta 原始输入各占 0.25 KiB，g 完成转换后原址保存
`gate_inv`。S1 的 UB 最高已用地址为 81 KiB，连续空闲为 167 KiB。`Delta/GateDelta`
逐行保存在 Vector 寄存器中，不在 UB 中物化完整矩阵。所有循环只使用
`r<valid_r` 的 Score 和 raw slot，其余预留地址不发射读写指令。S2 不分配或访问
AIV UB，因此整个算子的每个 AIV UB 峰值就是 S1 的 81 KiB。

内部执行顺序：

1. 两个 AIV 都按 `r=0..valid_r-1` 顺序执行。每个 hk 的首个 value head 等待
   leader Score ready，后续连续 value head 直接复用驻留槽；每个 `hv_r` 各搬一次
   A 行半片、完整 g 和完整 beta。A_bg 分支复用 `Gate(g)`；D 分支对每个输出元素的
   FP32 `g[s]-g[t]` 直接执行 `Gate`，不复用指数商或正负指数乘积。
2. 每个 `vslot` 将矩阵输入区和输出区分开管理。覆盖 A 前等待本 slot 的
   `input_free`；原始 g/beta 使用当前 `r` 的独立槽，转换后的 FP32 g/beta 跟随
   `vslot` 复用。MTE2 搬入后通过 MTE2->V ready 允许 VF 读取；VF 完成最后一次输入
   读取后立即返回 V->MTE2 `input_free`，因此后续本地 value head 的输入搬运不等待当前 MTE3。
3. VF 写 A_bg/A_beta/D 半片前等待本 `vslot` 的 `output_free`；覆盖当前 ring record 前，
   AIV 还必须等待 AIC 发布 `workspace_free[part,r]`。一次 VF 完成当前 `hv_r` 的半片公式，
   通过 V->MTE3 ready 允许 MTE3 读取。MTE3 将三个半片写入当前 `rho` 对应的 GM record；
   AIV0 写前 `BT/2` 行，AIV1 写后 `BT/2` 行，地址不重叠。三个半片写完后分别发布
   `workspace_ready[part,r]`，并返回本地 MTE3->V `output_free`。

每个活跃 `vslot` 独立维护 `input_free/output_free`；无效 `r` 不占 UB slot，也不发射
MTE2、VF 或 MTE3。

Stage 1 和 Stage 2 之间使用三段有限 GM ring workspace。AIV 以 ND 行主序写 GM，
AIC 在两个行半片都 ready 后以 MTE2 ND2NZ 搬入 L1；禁止 AIV 直接写 L1。定义：

```text
matrix_stride = ALIGN512(BT * BT * sizeof(main_dtype)) = 8 KiB
ring_count     = blockDim
slot_count     = ring_count * CG
rho            = block_idx * CG + r

A_BG_BASE      = 0
A_BETA_BASE    = A_BG_BASE   + slot_count * matrix_stride
D_BASE         = A_BETA_BASE + slot_count * matrix_stride

GM_X(rho)      = X_BASE + rho * matrix_stride, X in {A_BG,A_BETA,D}
workspace_size = 3 * slot_count * matrix_stride
```

每个活跃 MixBlock 独占一个 `ring_slot=block_idx`，它顺序处理的多个 work 复用该 slot。
每个矩阵的完整 record 为 8 KiB；每个 AIV 写 4 KiB 半片，三个矩阵合计每个 value head
写 24 KiB，AIC 随后读 24 KiB。AIC 必须等待同一 `rho` 的两个 `workspace_ready` 后再读，
并在三次 GM->L1 搬运全部完成后分别返回两个 `workspace_free`，下一 work 才能覆盖该 record。

尾 chunk 的无效行/列在 VF 中置零；`M=0` 不发射 VF 或 GM 写入。

### 2.3 Stage 2：Cube，合并 W/U 与 dv_local

对当前 `hv_r`，同一个 Cube Stage 内依次完成三个彼此独立的矩阵乘：

```text
w_hv_r[BT,K]        = A_bg_hv_r[BT,BT] @ k_hk_r[BT,K]
u_hv_r[BT,V]        = A_beta_hv_r[BT,BT] @ v_hv_r[BT,V]
dv_local_hv_r[BT,V] = D_hv_r[BT,BT] @ d_o_hv_r[BT,V]
```

Stage 2 按 `r=0..valid_r-1` 依次处理。A_bg/A_beta/D 由两个 AIV 写入当前 `rho` 的
GM 行半片，AIC 等待两半 ready 后将三个完整矩阵搬入当前 `r` 的 L1；v、d_o 按
`hv_r` 读取。W 的右操作数 k 不从 GM 重复搬运，
而是读取 Stage 0 中 `leader_r(r)` 槽保留的 `k_hk_r`。每份操作内三次 MMAD 顺序使用
L0A/L0B/L0C ping/pong。三个 L0C FP32 结果均由 Fixpipe 转为主 dtype，并直接写入
正式 `dvLocalOut/wOut/uOut` 的 GM 地址。最终输出不再被 Vector 消费，因此 S2
不经过 AIV UB、不触发 CrossCore 输出事件，也不增加 UB 占用。

物理空间按最多 4 份任务统一预留。空间布局（每个 AIC，单位 KiB）：

```text
L1[0,64)      k_hk slot[4]；主 dtype；每份 16 KiB；S0 按 leader 搬入，保留至对应 hk 的最后一个 S2 W 完成读取
L1[64,96)     A_bg_hv[4]；主 dtype；每份 8 KiB；S2 复用已释放的 q_hk 区间并从 GM workspace 搬入，当前 r 的 MTE1 末次读取后释放
L1[96,128)    A_beta_hv[4]；主 dtype；每份 8 KiB；S2 复用已释放的 q_hk 区间并从 GM workspace 搬入，当前 r 的 MTE1 末次读取后释放
L1[128,160)   D_hv[4]；主 dtype；每份 8 KiB；S2 从 GM workspace 搬入，当前 r 的 MTE1 末次读取后释放
L1[160,224)   v_hv[4]；主 dtype；每份 16 KiB；S2 MTE2 成组预取，当前 r 的 MTE1 末次读取后释放
L1[224,288)   d_o_hv[4]；主 dtype；每份 16 KiB；S2 MTE2 成组预取，当前 r 的 MTE1 末次读取后释放
L1[288,512)   空闲；无数据；S2 可复用
```

L1 最高已用地址为 288 KiB，不超过 A5 的 512 KiB 上限。S2 的 W 从
`k_hk slot[leader_r(r)]` 读取 S0 保留的 k；该 slot 的起始地址为
`leader_r(r)*16 KiB`。所有循环只访问 `r<valid_r` 的 slot。

输入 `HV` 轴的当前切片只激活 `r<valid_r` 的 slot；无效份不发射搬运、MMAD 或事件。

Stage 2 的 L0A/L0B 各使用两份 16 KiB，并与 Stage 0 分时复用 L0 地址。L0C 使用两份 32 KiB FP32 `[BT,128]`
accumulator（共 64 KiB），关闭 unit flag，并通过显式 `FIX_M` ready/free 事件交替复用。
每次只在即将复用对应 L0C 时等待：`D` 在 L0C0 上进入 Fixpipe 后，`W` 立即使用
L0C1 发射 Cube；`U` 到达复用 L0C0 的位置时才等待 `D` 的 Fixpipe 完成读取。
这样 `D` 的 Fixpipe GM 写回可与 `W` 的 MTE1/Cube 重叠，同时仍保证
`dv_local -> w -> u` 不覆盖尚未释放的 L0C。每个 L0C slot 的闭环是
`wait FIX_M -> MMAD -> M_FIX -> Fixpipe 写 GM -> FIX_M`。三个输出均只写前 `M`
行，尾块其余行不参与有效语义。

执行顺序：

1. Stage 0 完成后，AIC 不等待 Stage 1，先用两条 `ndNum=valid_r` 的 ND2NZ 分别将
   全部有效 `r` 的 v/d_o 预取到各自 L1 区间。随后按 `r` 等待当前 `rho` 的两个
   AIV 半片 ready，再将 A_bg/A_beta/D 三个完整 GM record 以 ND2NZ 搬入对应 L1。
   最后一次 MTE2 完成后立即归还该 `rho` 的两个 `workspace_free`。
2. 先以 D 为左操作数完成 `D @ d_o`，再依次完成 `A_bg @ k`、
   `A_beta @ v`；同一 hk 的多个 W 操作
   复用 Stage 0 leader 槽中的 k。MTE1 完成该份全部 L1 输入的最后一次读取后释放
   该 L1 slot；GM workspace 的生命周期已在 MTE2 完成时结束。
3. 三个 FP32 L0C 结果分别由 Fixpipe 转为主 dtype，直接写 `dvLocalOut/wOut/uOut`
   的正式 GM 地址。Fixpipe 完成当前 L0C 读取后释放对应 slot；完成当前 `HV` 轴
   切片的 `valid_r` 份操作后结束。

### 2.4 Stage 间同步方案

```text
template Process<CG>:
  BT = chunk_size
  G  = HV / HK
  CG = 3 if G == 3 else 4
  rows(part) = [part*BT/2,(part+1)*BT/2)
  if fixed length:
    Nchunk = B * ceil(T / BT)
  else:
    Nchunk = chunk_indices.shape[0]
  AIC_CORE_NUM = 目标 SoC 可供算子使用的 AIC 数
  NhvSlice = ceil(HV / CG)
  Nwork    = Nchunk * NhvSlice
  blockDim = min(AIC_CORE_NUM, Nwork)
  ring_slot(block_idx) = block_idx
  rho(block_idx,r)     = block_idx * CG + r

  parallel MixBlock block_idx in [0, blockDim):
    work[local_id] = block_idx + local_id * blockDim
    meta(local_id) = (chunk_id, hv_begin, valid_r)

    # Score、Stage 2 L1 和本 MixBlock 的 GM ring record 均按 r 分槽。
    init score_free[part,r] = FREE for part in [0,2), r in [0,CG)
    init workspace_free[part,r] = FREE for part in [0,2), r in [0,CG)
    # FIX_M 由 Fixpipe 发布给 Cube；两个 L0C slot 初始可由 MMAD 写入。
    init FIX_M[slot] = READY for slot in [0,2)

    parallel:
      AIC:
        for local_id while work[local_id] exists:
          (chunk_id, hv_begin, valid_r) = meta(local_id)

          # Stage 0：先把全部唯一 hk 的 q/k 发进 MTE2 流水。
          leaders = [r for r in [0,valid_r) if score_leader(r)]
          for leader in leaders:
            wait score_free[0,leader]
            wait score_free[1,leader]
            hk_r = floor((hv_begin+leader)/G)
            MTE2 k[chunk_id,hk_r] -> Stage0.L1_K[leader]
            MTE2 q[chunk_id,hk_r] -> Stage0.L1_Q[leader]
            set qk_ready[leader] from PIPE_MTE2

          # 每块独立 ready；Cube 消费当前块时，MTE2 可继续搬后续块。
          for leader in leaders:
            wait qk_ready[leader] from PIPE_MTE1
            hk_r = floor((hv_begin+leader)/G)
            S_hk = k[chunk_id,hk_r] @ q[chunk_id,hk_r]^T
            Fixpipe S_hk[0:BT/2,:]  -> AIV0.Score[leader]
            Fixpipe S_hk[BT/2:BT,:] -> AIV1.Score[leader]
            set score_ready[0,leader] from PIPE_FIX
            set score_ready[1,leader] from PIPE_FIX

          # v/d_o 对所有有效 r 成组预取；各 r 的 L1 地址互不重叠。
          MTE2 v[0:valid_r], d_o[0:valid_r] -> Stage2.L1[0:valid_r]

          for r in [0,valid_r):
            hv_r = hv_begin+r
            rho = rho(block_idx,r)
            # 两个 AIV 分别写 GM 前后行半片；只等当前 r，不设置组级屏障。
            wait workspace_ready[0,r] from PIPE_MTE2
            wait workspace_ready[1,r] from PIPE_MTE2
            MTE2 GM_A_BG[rho]   -> Stage2.L1_ABG[r]
            MTE2 GM_A_BETA[rho] -> Stage2.L1_ABETA[r]
            MTE2 GM_D[rho]      -> Stage2.L1_D[r]
            # 三个 record 已完全进入 L1，下一 work 可覆盖同一 rho。
            set workspace_free[0,r] from PIPE_MTE2
            set workspace_free[1,r] from PIPE_MTE2
            # L0C0/L0C1 交替使用；Fixpipe 转主 dtype 并直接写正式输出 GM。
            wait FIX_M[0] by PIPE_M
            Cube D @ d_o -> L0C[0]
            set M_FIX[0] by PIPE_M
            wait M_FIX[0] by PIPE_FIX
            Fixpipe cast_main(L0C[0]) -> dv_local[chunk_id,hv_r]
            set FIX_M[0] by PIPE_FIX

            wait FIX_M[1] by PIPE_M
            Cube A_bg @ k -> L0C[1]
            set M_FIX[1] by PIPE_M
            wait M_FIX[1] by PIPE_FIX
            Fixpipe cast_main(L0C[1]) -> w[chunk_id,hv_r]
            set FIX_M[1] by PIPE_FIX

            wait FIX_M[0] by PIPE_M
            Cube A_beta @ v -> L0C[0]
            set M_FIX[0] by PIPE_M
            wait M_FIX[0] by PIPE_FIX
            Fixpipe cast_main(L0C[0]) -> u[chunk_id,hv_r]
            set FIX_M[0] by PIPE_FIX

      parallel AIV part in [0,2):
        for local_id while work[local_id] exists:
          (chunk_id, hv_begin, valid_r) = meta(local_id)
          for r in [0,valid_r):
            hv_r = hv_begin+r
            leader = r-(r mod G)
            if r == leader:
              # leader 对应的新 Score 半片已经由 Stage 0 发布。
              wait score_ready[part,leader]
            A_bg_part, A_beta_part, D_part = Stage1VF(
                chunk_id, hv_r, part, Score[leader], r mod 2)
            # 上一 work 的同一 GM 半片已由 AIC 搬入 L1，当前 r 才能覆盖该 record。
            wait workspace_free[part,r] from PIPE_MTE3
            rho = rho(block_idx,r)
            MTE3 D_part      -> GM_D[rho][rows(part),:]
            MTE3 A_bg_part   -> GM_A_BG[rho][rows(part),:]
            MTE3 A_beta_part -> GM_A_BETA[rho][rows(part),:]
            if r == valid_r-1 or floor(hv_r/G) != floor((hv_r+1)/G):
              set score_free[part,leader] from PIPE_MTE3
            set workspace_ready[part,r] from PIPE_MTE3
```
