# ChunkFwdO A6 差异设计

方案设计规则版本：V2

## 1. 目标

接口见 [api.md](api.md)。以 A5 134524a6 为基线，按用户提供的三 head 详设和 a6_chunk_fwd_o.drawio 实现 arch92，两条路径均保留与 arch35 对应的文件层级。
本次不设性能验收目标。用户提供的 A6 1:1、UB→L1、Stage3 两次 VF 方案是本次架构输入，替代通用参考的 1:2 和 GM 中转映射。

## 2. Stage 详设

以下优化路径资源与公式来自用户详设；兼容路径差异及稳定性分析见末尾。

### 优化路径原始设计

## 1. 核心改动一览

| 项目 | A5 arch35 | A6 arch92 方案 |
| --- | --- | --- |
| 核配比 | 32 AIC : 64 AIV，MIX 1:2 | 64 AIC : 64 AIV，MIX 1:1 |
| 单 AIV UB | 248 KB，TBuf 248 KB | 物理 384 KB，按当前 arch92 Prepare 的 376 KB 可用预算设计 |
| 工作项 | 同一 chunk 最多 4 个 hv，GVA=3 时 3 个；每 AIV 最多 2 个 | 同一 chunk 最多 **3 个 hv**；一个 AIV 处理整组，按 `r=0,1,2` 使用独立槽 |
| L0C→UB | 整块 FP32 ND 发往 owner subblock | 整块 FP32 ND 发往唯一 AIV；消除 owner 路由 |
| UB→L1 | owner AIV 将整块 A_prime 从 ND 按 16 列条带写入 L1 Nz | 单 AIV 生产完整 A_prime，直接写对应 L1 槽；保留 Cube 所需的布局重排 |
| UB / L1 布局覆盖范围 | 248 KB/AIV；352 KB/AIC | **313.5 KB/AIV；264 KB/AIC** |
| 同步 | 两个 AIV 都参加 mode 0x2 握手 | 1:1 mode 0x4；QK/QH 分别发 ready，Stage3 两次 VF 独立等待，Stage 5 结束后归还整组 UB |

## 2. 数学依赖与 Stage 划分

设 `c` 为全局 chunk，`r` 为工作项内 head，`hv=hvBase+r`，`hk=hv/GVA`（整数除法）。有效行 `1≤m≤64`。Q/K 为 BF16 [m,128]，来自 hk。H 为 BF16 逻辑 [128,128]，V 为 BF16 [m,128]，g 为 [m]，来自 hv。所有 Cube 累加为 FP32。

### 2.1 基础公式

对一个 chunk 和一个 value head，`H` 是进入该 chunk 时的状态快照，`V` 是当前块的 value 输入（GDN 前向由 `chunk_fwd_h` 提供 `v_new`）。公式中的 `g[i]` 表示第 i 个 token 的 FP32 gate 值。`E(x)` 在 `use_exp2=true` 时表示 `2^x`，实现为 FP32 `Exp(x × ln2)`，否则为 FP32 `Exp(x)`。

```text
Stage 1  gate_o[i] = E(g[i])
         gate_A[i,j] = E(g[i] − g[j])
Stage 2  A_raw = Q @ Kᵀ
         O_s_raw = Q @ H
Stage 3  A_prime = BF16(causal(A_raw ⊙ gate_A))
         O_s_prime = gate_o ⊙ O_s_raw
Stage 4  O_l = A_prime @ V
Stage 5  O = BF16((O_s_prime + O_l) × scale)
```

`causal(X)` 仅保留 `0 ≤ j ≤ i < m` 的元素，其余置零。`⊙` 表示逐元素乘，`gate_o ⊙ O_s_raw` 按行广播，对应 `O_s_prime[i,:] = gate_o[i] × O_s_raw[i,:]`。`BF16` 表示舍入到最近的 BF16 值，恰好居中时取偶数。

`O_s_prime` 是由历史状态 H 得到的贡献，`O_l` 是用因果权重 A_prime 聚合当前块 V 得到的贡献。S5 在两路贡献相加后统一乘 scale。

### 2.2 各 Stage 的职责

| Stage | 单元 | 做什么 | 公式与写出 | 依赖 |
| --- | --- | --- | --- | --- |
| Stage1 gate | Vector | 准备两类门控权重。`gate_o` 用于历史状态项，`gate_A` 用于块内 token 对。 | `gate_o[i] = E(g[i])`；有效域内 `gate_A[i,j] = E(g[i] − g[j])` | g 到 UB；与 S2 独立 |
| Stage2 原始矩阵 | Cube | 计算块内 Q/K 原始得分，以及当前 Q 对历史状态 H 的投影。 | `A_raw = Q @ Kᵀ`，FP32 [64,64]；`O_s_raw = Q @ H`，FP32 [64,128]；两者 Fixpipe 转 ND 到 UB | Q/K/H 到 L1；同 hk 共享 Q/K 搬入 |
| Stage3 门控 | Vector | 给两路结果施加门控。块内得分裁成因果下三角并转 BF16，历史项按行缩放。 | `A_prime = BF16(causal(A_raw ⊙ gate_A))`；`O_s_prime = gate_o ⊙ O_s_raw`，FP32 原位 | `A_prime` 等 `A_raw` 和 `gate_A`；`O_s_prime` 等 `O_s_raw` 和 `gate_o` |
| Stage4 块内输出 | Cube | 用因果得分对当前块的 V 加权求和，得到块内贡献。 | `O_l = A_prime @ V`，逻辑 FP32 [m,128]，Fixpipe 转 ND 到 UB | 本 head A_prime 和 V |
| Stage5 合成 | Vector | 合并块内贡献和历史贡献，统一乘 scale，转换并写出最终 O。 | `O = BF16((O_s_prime + O_l) × scale)`，加法和缩放为 FP32，BF16 O 从 UB 写 GM | 本 head O_s_prime / O_l |

## 3. 流水伪代码与同步闭环

以下是指令下发语义。PIPE_S 等待在下发端阻塞，其他 Wait 放到表内目标 pipe。每个 ready 绑定工作项和本地槽。跨核使用 mode 0x4，表中的 13 个候选 flag ID 在仓内 arch92 Prepare 使用的 0–15 范围内，落地时按目标 SDK 校验。

| 同步资源 | 数量与方向 | 用法和回收点 |
| --- | --- | --- |
| G_READY | 3 个 MTE2_V | 每 head 每组搬 g 后置位，S1 consume。下组 g scratch 的 MTE2 覆盖另受下述 G_FREE 约束 |
| G_FREE | 3 个 V_MTE2 | 初始化空闲；S1 完成读 g 后发出，下组 MTE2 等待后写入；退出逐槽 drain |
| QK ready/free | 每方向 3 个 MTE2_MTE1 / MTE1_MTE2 | Q/K 作为一对。第一个引用产生 ready，最后一次 QH MTE1 读完归还，期间按 MTE1 FIFO 供同 hk 的其他 GEMM 使用 |
| HV ready/free | 每方向 3 个 MTE2_MTE1 / MTE1_MTE2 | 同 r 的 H 和 V 顺序使用同一事件。H 被 QH MTE1 读完 → V MTE2；V 被 AV MTE1 读完 → 下组 H MTE2 |
| L0A/B 与 L0C | 每方向 2 个 M_MTE1、MTE1_M、M_FIX、FIX_M | AB 同槽轮换，Cube 读完释放 AB；Fixpipe 读完释放 C。初始化 AB/C free，退出收齐 |
| A_RAW_READY | 3 个跨核，FIX → V，候选 ID 0–2 | A_raw 写完，S3 的 A_prime VF 可读取 |
| OS_RAW_READY | 3 个跨核，FIX → V，候选 ID 3–5 | O_s_raw 写完，S3 的 O_s_prime VF 可原位更新 |
| APRIME_UB_READY / O_UB_READY | 各 3 个 V_MTE3 | 前者保护 A_prime，后者保护压缩 O，按独立事件分配 |
| APRIME_READY | 3 个跨核，MTE3 → MTE1，候选 ID 6–8 | A_prime 所有条带写 L1 完成。MTE1 等本 head 该信号以及本地 V ready |
| OL_READY | 3 个跨核，FIX → V，候选 ID 9–11 | 本 head O_l 写 UB 完成 |
| ROUND_DONE | 1 个跨核，MTE3 → S，候选 ID 12 | 每个非空工作项最后一个 O 写回之后发布；AIC 在下一组首个 S2 或退出时消耗 |

## 4. UB 分配与生命周期

A6 Vector 用 `AscendC::TOTAL_UB_SIZE` 和 `AscendC::TOTAL_L1_SIZE` 初始化 UB 和 L1 的 TBuf。单位 KB，1 KB=1024 B，区间右开。每行按 `base+r*stride` 定位，所有槽基址按 256 B 对齐。容量用满块，尾组只使用 `r<n`。未写入的静态槽空闲，空闲时仍保留地址。

| 数据 / 物理布局 | 单 head KB | Head 0 | Head 1 | Head 2 | 写入 → 最后消费 / 复用 |
| --- | ---: | --- | --- | --- | --- |
| gate_o，FP32 [64] | 0.25 | [0,0.25) | [0.25,0.5) | [0.5,0.75) | S1 → S3 的 O_s_prime VF |
| gate_A，FP32 ND [64,64] | 16 | [0.75,16.75) | [16.75,32.75) | [32.75,48.75) | S1 → S3 的 A_prime VF |
| gScratch，FP32 [64]，支持 BF16 输入 | 0.25 | [48.75,49) | [49,49.25) | [49.25,49.5) | S1 MTE2 → S1 VF；G_FREE 后重填 |
| O_s_raw → O_s_prime，FP32 ND [64,128] | 32 | [49.5,81.5) | [81.5,113.5) | [113.5,145.5) | S2 的 QH Fixpipe → S3 的 O_s_prime VF 原位改写 → S5 VF |
| A_raw，FP32 ND [64,64] | 16 | [145.5,161.5) | [161.5,177.5) | [177.5,193.5) | S2 的 QK Fixpipe → S3 的 A_prime VF |
| O_l → O，FP32 ND [64,128] → BF16 | 32 | [193.5,225.5) | [225.5,257.5) | [257.5,289.5) | S4 Fixpipe → S5 VF；O 从各槽起点写低 16 KB，MTE3 搬出后可覆盖 |
| A_prime，BF16 ND [64,64] | 8 | [289.5,297.5) | [297.5,305.5) | [305.5,313.5) | S3 的 A_prime VF → MTE3 搬入 L1 完成 |

O_l→O 的原址压缩沿行内 64 元素 tile 正向执行。每个 tile 先读各 256 B 的 O_s_prime/O_l，再把 128 B BF16 写到基址的紧凑前缀，写区末地址不超过当前 tile 的已读末地址，后继 tile 未读 FP32 不被覆盖。保留单 VF 内 load-before-store，核对编译后访问序。

每 head **104.5 KB**，三份 **313.5 KB**。物理余量 **70.5 KB**。生成 A_prime 的 FP32 乘积只在 VF 寄存器内，g 的 BF16→FP32 扩展同样先获取寄存器输入后才覆盖 UB。对 gScratch 的尾行访问优先使用有效 lane mask。

## 5. L1、L0 与数据搬运

### 5.1 L1 地址

| 对象 / 物理布局 | 单份 KB | slot 0 | slot 1 | slot 2 | 生产 → 最后消费 |
| --- | ---: | --- | --- | --- | --- |
| Q BF16 packed [64,128] | 16 | [0,16) | [16,32) | [32,48) | MTE2 按不同 hk 读一次 → 最后一次 QH MTE1 |
| K BF16 packed [128,64] | 16 | [48,64) | [64,80) | [80,96) | MTE2 按不同 hk 读一次 → 最后一次 QK MTE1；与 Q 一起释放 |
| H BF16 packed [128,128] | 32 | [96,128) | [128,160) | [160,192) | S2 MTE2 → 本 head QH MTE1 |
| A_prime BF16 zN [64,64] | 8 | [192,200) | [200,208) | [208,216) | S3 MTE3 → 本 head AV MTE1 |
| V BF16 packed [64,128] | 16 | [216,232) | [232,248) | [248,264) | S4 MTE2 → 本 head AV MTE1 |

Q/K slot 是本组首次引用该 hk 的 head offset，H/A_prime/V slot 是 r。L1 分配 **264 KB**，以仓内 arch92 Prepare 记载的 512 KB 为预算，连续余量 **248 KB**。全部基址按 8 KB 对齐。AIC MTE1 消费的是按 CATLASS tag 定义的 A/B packed tile。A_prime 的 16 列 zN 条带排布须与 S4 的 L1A→L0A 视图逐地址对齐。

L0A/B/C 各自按 slot 0/1 轮转，字节偏移为 `slot*stride`。单份有效 tile 大小与槽间距分别如下，单位为 KiB（1024 B）；槽间距以 [chunk_fwd_o_common.h](../op_kernel/arch92/chunk_fwd_o_common.h) 的 `CHUNK_FWD_O_L0_{A,B,C}_BYTES` 为准。

| 资源 | 单份最大有效 tile | 槽间距 stride | slot 0 预留区间 | slot 1 预留区间 | 双槽预留量 / Resource 分配量 |
| --- | ---: | ---: | --- | --- | ---: |
| L0A | 16（Q） | 32 | [0,32) | [32,64) | 64 |
| L0B | 32（H） | 32 | [0,32) | [32,64) | 64 |
| L0C | 32（QH/AV） | 128 | [0,128) | [128,256) | 256 |

当前 CATLASS `Resource<Ascend960DT>` 按架构容量分配 L0A/B/C，分别为 64/64/256 KiB。有效 tile 小于槽间距时，剩余空间仍属于该槽的预留区，不能据有效数据量将其视为未分配空间；例如 L0C slot 1 的起点为 128 KiB。L0 的预留区与 UB/L1 地址预算分别计算。

### 5.2 GM 搬运与 workspace

| 对象 | 工作项流量（n 个 hv、d 个 hk、m 行） | 路径 |
| --- | ---: | --- |
| Q/K | 读 `d*m*512 B` | 每 hk 各读一次，MTE2 转 packed L1，同组复用；QK GEMM 仍按 hv 下发 |
| H | 读 `n*32768 B` | MTE2 按 hv 读一次，K/V 视图由 state_v_first 确定 |
| V | 读 `n*m*256 B` | MTE2 按 hv 读一次，先提交下一 head 的 V |
| g | 读 `n*m*sizeof(GT) B` | AIV MTE2 按 hv 读一次 |
| O | 写 `n*m*256 B` | AIV MTE3 按 token 行写，单行 256 B，行间 gap 为 `(HV−1)*256 B` |
| 内部中间结果 | 0 B GM 中转 | A_raw/O_s_raw/O_l：L0C→UB；A_prime：UB→L1；O_s_prime 留 UB |
| varlen 元数据 | 每个 AIC/AIV 工作项解析索引 | chunkOffsets 取索引对，cu_seqlens 取相邻边界 |

图示满块工作项的 GM 张量流量为 224.75 KB，S2/S4 合计 6,291,456 MAC，三 head 的 Fixpipe 传输共 240 KB，A_prime 的 UB→L1 共 24 KB。A5 的 Q/K/H/V/g L2 Cache hint 策略作为第一版基线。按用户确认，host 与 A5 一致预留 `aicCoreNum * CHUNK_FWD_O_APRIME_WORKSPACE_BYTES`，即每 AIC 64 KiB，`aPrimeWorkspaceOffset=0`；kernel 不访问该预留区，A_prime 仍直接从 UB 搬入 L1，GM 中转流量为 0 B。总 workspace 为这笔预留加平台返回的系统 workspace。

## 6. GVA、分核与 N 选择

```text
GVA = HV / HK，HV % HK == 0，GVA ∈ [1,16]
N = 3
headGroupNum = ceil(HV / N)
Nwork = chunkNum * headGroupNum
blockDim = aicCoreNum                            # 与 A5 一致，使用全部 AIC 核数
work = coreIdx, coreIdx + blockDim, ...
c = work / headGroupNum
hvBase = (work % headGroupNum) * N
n = min(N, HV - hvBase)
hv = hvBase + r，hk = floor(hv / GVA)，r ∈ [0,n)
```

AIC/AIV 用同一 1:1 逻辑 coreIdx 和 blockDim。按用户确认，host 与 A5 一致直接使用 `SetBlockDim(aicCoreNum)`，不按 AIV 核数或工作项数缩减。工作项不足时，无任务的核不进入工作项循环，只完成本地事件初始化与收尾。定长按 batch×chunk 网格展开，变长按元数据 pair 顺序展开，H 选择全局 chunk 对应的快照。零长度 sequence 无合法 chunk 工作项。零工作沿 host 原有空输入约定完成；只对具有正 m 和 n 的工作项进入 Stage 和握手。

图中 S1 和 S5 的 Vector 计算各占一格，S3 的两个 VF 分格。第一轮 Head 0 在 t5 生成 A_prime，t6 原位更新 O_s_prime，同时 MTE3 开始搬运 A_prime。其他 head 采用相同依赖。A_prime 的四次连续条带搬运合为一个 MTE3 格。Q、K 的 GM→L1 搬运分别画格，同一次 GEMM 的 L1→L0A 和 L1→L0B 搬运顺序执行，在 AIC MTE1 合为一格。图示配置取定长 B=1、T=4160、HK=1、HV=3、BT=64、K=V=128、g FP32、`use_exp2=true`、`state_v_first=false`、BSND。共 65 个满块工作项，blockDim=64，两波。Core 0 处理 chunk 0 和 64，其他核各处理一项。每项三 head 共用一份 Q/K，Head 0/1/2 各有独立 H/A_prime/V。GVA=2 或 4 时允许三 head 工作项跨 hk 边界，按每组实际 hk 去重。例如 GVA=4、HV=8 时分为 [0,1,2]、[3,4,5]、[6,7]，每个 chunk 的 Q/K 加载 4 对，A5 按 4 对齐时加载 2 对。性能评估需覆盖该重复搬运。

| N | UB KB | 最坏 L1 KB | 结构取舍 |
| ---: | ---: | ---: | --- |
| 1 | 104.5 | 88 | 每工作项 n=1，增加工作项数；欠饱和场景可扩大并行的核对数，组内无后续 head 可填流水 |
| 2 | 209 | 176 | 两 head 驻留，GVA=2/4 对齐时 Q/K 复用更好，可作为对照 |
| **3** | **313.5** | **264** | 三 head 驻留，GVA=3 完整覆盖一个 hk，Stage 轮次内增加可交叠的 head；存在 HV 尾组和 GVA=2/4 跨 hk 的负载、搬运代价 |
| 4 | 418 | 352 | UB 超预算。即使去掉全部 A_prime 出口槽，其他独立驻留数据 386 KB，仍超物理 384 KB |

对 GVA=5–16 同样按 hv 每三个分组。组内每个 hk 搬一次 Q/K，相邻组再次引用同一个 hk 时重新搬入。以 HK=1、HV=16 为例，六个工作项各自加载一对 Q/K，最后一组只有 hv=15。

选择 N=3 作为首版静态方案。L1 按 3 个不同 hk 的最坏情况预算。N=4 若要成立，需改变 A_raw/O_s_prime/O_l/gate 的同时驻留关系与流水。N=2/3 对照时用 `ceil(Nwork/blockDim)` 和末波活跃核数衡量并行度，结合不同 GVA 的 Q/K 重复搬运、单 Vector 串行工作和组间等待分析。当前没有设备数据可判定最快 N。

## 7. 实现位置与验收

| 位置（算子目录及公共测试入口） | 实现内容 |
| --- | --- |
| `op_host/chunk_fwd_o_def.cpp`、host/kernel CMake 与打包 | 注册 `ascend960dt`，接入 arch92 编译和相应 CATLASS 版本 |
| `op_host/chunk_fwd_o_tiling.cpp`、`chunk_fwd_o_tiling_processor.h` | 识别 A6，按 BSND/TND 和形状/dtype 组合校验，设置 taskGroupSize=3、blockDim 与系统 workspace。核对 arch92 平台枚举和 AIC/AIV 可用核数 |
| `op_kernel/chunk_fwd_o_tiling_key.h`、`chunk_fwd_o.cpp` | A6 的可选 key 使用 MIX_AIC_1_1，task type 与 host key 一致；`__NPU_ARCH__==9201` 走显式 arch92 入口 |
| 新 `op_kernel/arch92/chunk_fwd_o_{common,cube,vector}.h` | UB/L1 地址、按 hv 计算的 QK/QH/AV、整 tile Fixpipe、S1/S3/S5 VF（S3 两次 VF）、五组跨核语义 flag 和本地 HardEvent 生命周期 |
| `tests/atk/chunk_fwd_o/` 及 Torch 调用 | 对齐 CPU/DUT 的 E、输出布局与 dtype，增加 A6 组合、尾块和跨工作项复用回归 |


### 兼容路径差异

保持 A5 的 Cube1 QK、Vec1 gate/mask、Cube2 QH、Cube3 AV、Vec2 输出调度，以及原有 workspace 字段和计算方式。
ArchTag 改为 Ascend960DT，跨核信号 mode=0x4，单 AIV 处理整块。V128 的 L0C→UB 使用 NO_SPLIT。
UB 基础区 [0,71) KiB 不变，四个工作槽改为 64 KiB，依次为 V ping [71,135)、H ping [135,199)、V pong [199,263)、H pong [263,327)。
Vec1 scratch 为 A FP32 [327,343)、out FP32 [343,359)、out B16 [359,367)；Vec2 scratch 为 FP32/B16 ping [327,343)/[343,351)，pong [351,367)/[367,375)。
Vec1/Vec2 串行共享 scratch，最高地址 375 KiB，小于 376 KiB 可用预算；连续空闲 1 KiB。
Vec1/Vec2 每段最多 32 行，单 AIV 最多四段；段号映射为 rowStart=stage*32，末段 min(32,m-rowStart)。
Vec1 在下一段搬入前等待前段 MTE3 结束；Vec2 在覆盖输出前等待前段 MTE3，最后一段才归还工作槽。
V256 仍经 GM workspace，按原有 16 行 tile 处理，不新增 dtype、layout 或维度。
L1/L0 保留 A5 分配及 block_mmad 容量断言。所有兼容路径 ready 在 PIPE_S 消费以保留 A5 下发顺序，set 保留原生产 pipe，mode 改为 0x4。

### 稳定性与复用审查

优化路径只在有效因果域计算 gate_A：无效差值在 Exp 前置零，Exp 后置零；S3 在乘法前屏蔽无效 A_raw，避免 padding 的 NaN/Inf 传播。
有效域不改变 A5 的指数、舍入和累加顺序；任意有限 g 并不保证指数有限，输入值域沿用现有用例，不能宣称全值域精度。
gScratch 的 BF16 输入先加载到寄存器再扩展到 FP32，扩展结果只驻留寄存器，尾行显式归零；每槽 V_MTE2 free 防止下一工作项提前覆盖。
Q/K free 在该 hk 最后一次 QH 的 MTE1 后归还；H/V 共享事件顺序复用；AB/C 的初始 free 在退出时全部 drain。
AIC 和 AIV 用同一工作项迭代器解析 chunk、起始 hv 和有效 head 数。当前组 S1 读完各自的 gScratch 后，AIV 的 MTE2 即可预取下一组的 g，与本组 S3/S5 重叠。ROUND_DONE 由本组最后一笔输出 MTE3 发布，AIC 可预取下一组首个 Q/K/H，在首个 S2 的 L1→L0 前或退出时消费该信号。

兼容路径最后按实际使用过的 streamId 收回 vec2Done，不能假定单任务使用 slot 0；scheduler 初始 currStage=1。公共 block_mmad 的 CopyL0CToDst 选择为 9201 补齐，3510/2201 保持原分支。
