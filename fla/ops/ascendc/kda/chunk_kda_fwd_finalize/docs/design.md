# ChunkKdaFwdFinalize 设计

方案设计规则版本：`V2`。公开接口与支持范围以 [api.md](api.md) 为准。

## 阶段边界

Prepare 已生成按 HV 展开的 `qg_scaled` 和 `Aqk`，FwdH 已生成
`v_new` 和每个 chunk 的前态 `h`。Finalize 只消费这四项并写出
`attn_out`，不读取原始 `q/k/v/g/beta`，不输出反向检查点或
`h/final_state`。公开约束见[算子 README](../README.md#输入输出)。

## 数学与精度

对每个 batch、value head、序列及其第 `c` 个 chunk，设有效长度为
`M<=64`，物理 `h_c` 按 `state_v_first` 解释为逻辑 `[K,V]`：

```text
P_c = qg_scaled_c[M,128] @ h_c[128,128]       // BF16 输入，FP32 累加
R_c = Aqk_c[M,M] @ v_new_c[M,128]            // BF16 输入，FP32 累加
attn_out_c = bf16(P_c + R_c)
```

两个矩阵乘均以 BF16 输入、FP32 累加，只有最终和落盘时转 BF16。
`qg_scaled` 和 `Aqk` 在 Prepare 边界已经舍入且已应用 scale，
本阶段不能重新缩放，也不能从未舍入的原始 q/k 重算。

`Aqk` 的物理列数为 64，但尾 chunk 只读前 M 列；`v_new` 同样只读
有效 token。不同序列不会共享 chunk，`h` 的 chunk 轴按规范的
sequence-major 顺序映射，不按 tensor 的 T 维简单除以 64。

## 调度与存储

本节保留已有 A5 调度、搬运和同步描述。A2/A3 的 chunk × head group
调度与单 Stage 0 详设见后文；它替代 A2/A3 原先的 chunk 优先分核，
继续使用本节定义的 BF16 操作数、L0C 原位累加与 Fixpipe 输出路径。

各 chunk 不存在跨 chunk 数据依赖，先按 chunk 分核；仅当 chunk
任务少于可用核时增加完整 value head 的分区。输出写回地址按
`(batch,chunk,head)` 分区，不与其他 AIC 重叠。

A5 dense 的主体按完整 chunk 均分；最后不足一轮核数的 chunk，再按
4 个完整 head 的组均摊。AIC/AIV 使用相同映射和组内消费顺序。例如
128 个 chunk、96 个 head、28 核时，先每核分 4 个完整 chunk，余下
16 个 chunk 的 384 个 head 组按每核 13/14 组分配，避免原来每核
4/5 个 chunk 的负载差。每个 `(chunk,head)` 只由一个核组处理，
不增加计算、workspace 或同步 flag。变长路径保持原有任务划分。

Host 根据平台和每核工作量选择 `USE_AIV_INPUT_MOVER` 编译期模板参数：

- `false`：A2/A3 始终使用，A5 在 Host 已划分 head 分区，或每核完整
  chunk 数不足阈值时使用（dense 为 4、变长为 8）。模板组合指定为
  AIC-only，四个输入由 AIC MTE2 直接从 GM
  搬到 L1。
- `true`：仅 A5 使用，要求 Host 未划分 head 分区，且每核至少有
  4 个 dense chunk 或 8 个变长 chunk。dense 尾部负载均衡由 Kernel
  进一步划分完整 head 组。模板组合指定为 MIX AIC 1:2，两个 AIV 负责 GM 到
  UB 再到 L1 的格式转换，AIC 只消费已经就绪的 L1 操作数。

TilingKey 由 `GET_TPL_TILING_KEY(USE_AIV_INPUT_MOVER)` 生成；Kernel 全局
入口直接以同名模板参数实例化，不做数值 key 的运行时
二次分派。两种模板组合通过 `ASCENDC_TPL_KERNEL_TYPE_SEL` 分别
声明核类型，因此 AIC-only 路径不会启动空闲 AIV。

`USE_AIV_INPUT_MOVER=false` 只有一个 Cube stage：

| Stage | 核 | 本轮计算 | 阶段结果 |
| --- | --- | --- | --- |
| C0 | Cube | MTE2 搬入四个输入；第一次 MMAD 以 `initC=true` 计算 `qg_scaled@h`；第二次以 `initC=false` 将 `Aqk@v_new` 直接累加到同一 L0C | 一次 Fixpipe 以 `F322BF16` 转换并写出 `attn_out` |

`USE_AIV_INPUT_MOVER=true` 按一个 head 的两组依赖拆成下列流水。
V0 发布 Q/H 后即可继续
搬 Aqk/V，C1 不必等待四个输入全部到达；同理，C3 消费 L1 后立即
发布 free，AIV 不等待 Fixpipe 才复用该 head 槽。

| Stage | 核 | 本轮计算 | 阶段结果 |
| --- | --- | --- | --- |
| V0 | Vector | AIV MTE2 将 `qg_scaled/h` 从 GM 搬入 UB，MTE3 跳搬到 L1 并完成 ND2NZ | 发布 Q/H ready |
| C1 | Cube | 等待 Q/H ready，MTE1 搬入 L0A/L0B，MMAD 以 `initC=true` 计算 `qg_scaled@h` | 第一项保留在 L0C |
| V2 | Vector | AIV MTE2 将 `Aqk/v_new` 从 GM 搬入 UB，MTE3 跳搬到 L1 并完成 ND2NZ | 发布 Aqk/V ready |
| C3 | Cube | 等待 Aqk/V ready，第二次 MMAD 以 `initC=false` 累加到同一 L0C；发布 L1 free；Fixpipe 转换并写出 | `attn_out` |

每个 head 的 L1 操作数地址固定：`qg_scaled` 16 KiB、`h` 32 KiB、
`Aqk` 8 KiB、`v_new` 16 KiB，共 72 KiB；四个 head 为 288 KiB，
小于每 AIC 的 512 KiB。C0 读完对应 head 的操作数后才能复用其
L1 槽。`h` 的 32 KiB 是 BF16 `[128,128]`，不能和两个 FP32
计算结果的空间混同。

两项矩阵乘的 L0A 区域分别为 `qg_scaled` 16 KiB 和 `Aqk` 8 KiB，
共 24 KiB；
L0B 分别为 `h` 32 KiB 和 `v_new` 16 KiB，共 48 KiB。单个 head
只产生一份 FP32 `[64,128]` 结果；两块 32 KiB L0C 是相邻 head 的
MMAD/Fixpipe ping-pong 槽，不是两项乘积的独立保存区。第二次 MMAD
直接读取第一次的 L0C 累加结果，随后每个 head 只提交一次 Fixpipe。
Arch35 以 PIPE_MTE1/PIPE_M/PIPE_FIX 的 Mutex 约束对应槽位，Arch22
使用对应的 HardEvent。

A2/A3 的尾块不足 16 行时，L1 左操作数按每个 K 分形只填零无效的
M 行，MMAD 的物理 M 补到 16；Fixpipe 与输出仍只写有效行。
填零和输入搬运都由 MTE2 完成，现有 MTE2 到 MTE1 的事件覆盖两者，
不会读取未初始化的 L0A 行，也不会改变有效行的计算语义。

`USE_AIV_INPUT_MOVER=true` 的 UB 采用两个静态槽，基址为 0 和
128 KiB。每槽固定分配
`qg_scaled` 18 KiB、`h` 36 KiB、`Aqk` 10 KiB、`v_new` 18 KiB，
共 82 KiB；第二槽末端为 210 KiB，小于 A5 的 248 KiB UB。128 列
BF16 数据的 UB 行 pitch 为 9 个 32B datablock，64 列数据为 5 个，
避免连续物理行反复命中同一组 bank。两槽语义固定，不做 UB 内位置
移动；Q/H 与 Aqk/V 分别使用 Mutex 0/1 和 2/3 保护同槽 MTE2/MTE3
并发访问。一次性 GM 输入关闭 L2 cache，避免污染后续可复用数据。

`USE_AIV_INPUT_MOVER=true` 时，AIV0 负责组内 local head 0/2，AIV1
负责 1/3。mode 0x4 下
Q/H ready 在 AIV 侧使用 flag 0/1，AIC 侧映射为 0/16/1/17；Aqk/V
ready 使用 2/3，映射为 2/18/3/19；L1 free 使用 4/5，映射为
4/20/5/21。每个 ready 都由对应 AIC 消费，每个在途 L1 槽都在复用
前等待 free，禁止连续无消费地设置同一个 flag。A2/A3 不编译 A5
Vector mover，也不参与上述跨核同步。

两条路径都不把中间结果写入 GM relay，Host 只申请平台库 API
workspace。Fixpipe 将 L0C FP32 结果直接转换为 BF16：BNSD/NTD
输出的 `dstStride=128`，BSND/TND 输出的 `dstStride=HV*128`，因此
四种 layout 都不需要额外 AIV scatter。两个 L0C slot 的复用由 AIC
内 M/FIX 生命周期保护。

## A2/A3 优化目标与任务域

本次只优化 A2/A3 的内部调度及两槽 L1 输入预取，输入、输出、数学边界和 Stable ABI 均沿用
[api.md](api.md)。目标是在现有 `atk_chunk_kda_fwd_finalize_perf.json`
场景中改善相同设备、输入和测量条件下的目标 kernel 耗时；用户未指定
数值性能目标。优化前结果、候选参数与实测结论归档于
[ATK README](../../../../../../tests/atk/chunk_kda_fwd_finalize/README.md)。

已有数学路径满足单 Cube Stage 要求。A2/A3 保留 CG1 的 chunk × head
grid-stride 分核，并用两份独立 L1 输入槽预取本核的下一有效 work。
GM/L1、L1/L0 的数据布局和 MMAD/Fixpipe 的数学顺序保持原定义；
变化是 GM 输入提交顺序与逐槽 ready/free 生命周期。工程调度可参考
`CHUNK_INDEPENDENT V1`（`a78efa9512dfa64c20b2a58a75e84e88a054d694`）及
`chunk_kda_bwd_finalize` 的连续 head
window；该 backward 算子是 arch35 实现，不能作为 A2/A3 数学或硬件参数的依据。

统一符号：`B` 是 batch，`T` 是每个 dense batch 的 token 数或 packed 总 token 数，
`HV` 是 value head 数，`C=ceil(T/64)` 是 dense 每个 batch 的 chunk 数，
`M=1..64` 是当前 chunk 的有效行数。`Nbase=B*C`（dense），或各变长序列
chunk 数之和（varlen）；不同 chunk 没有状态依赖。输入已经沿 `HV` 展开，
本算子没有 `HK` 输入，无新的 q/k source-head 映射或跨 head 归约。

Host 的内部 `headGroupSize=0` 保留既有 A5 调度算法。A2/A3 最终采用统一
固定 CG1，即 `headGroupSize=1`、有效组大小 `G=min(HV,1)=1`；不根据
shape 切换分域策略。固定 CG4/3/2/1 的同条件 profiling 表明 CG1 在本次
十个模型 case 的中位耗时均低于原基线，具体测量配置与结果归档于
[ATK README](../../../../../../tests/atk/chunk_kda_fwd_finalize/README.md)。
本次运行测量来自 A2 `Ascend910B3`；A3 `ascend910_93` 单算子构建已通过，
尚未进行 A3 实机精度或性能测试。A3 构建产物的 generic MIX 元数据不作为
A3 运行时只有 Cube 任务的证据。
`Hgroup=HV/G+(HV%G!=0)=HV`，
`Nwork=Nbase*Hgroup`，`blockDim=min(availableAIC,Nwork)`。乘法和范围检查
在 Host 使用 64 位，合法 tiling 才转为已有 32 位字段。Kernel 按固定 CG1 使用 HV 直接解码 work，无 `HV+G-1` 加法。

一个 work 只含一个连续 value head，在核内仍依次执行两个 GEMM。
该粒度充分展开少量 chunk 的任务域，也消除了完整 head group 与末尾不足
四个 head 的组之间的成本差；不同 M 的 chunk 仍可能有不同计算成本。

```text
work cursor = coreId
nextValid(): decode work/HV and work%HV; advance work by blockDim; skip invalid chunk
current = nextValid()
if current exists: PrefetchHead(current, inputSlot=0)
while current exists:
    next = nextValid()
    if next exists: PrefetchHead(next, inputSlot^1)
    ConsumeHead(current, inputSlot, outputSlot)
    current = next; inputSlot ^= 1; outputSlot ^= 1
drain remaining input-free, L0 operand-free and output-free events
```

Dense：`b=chunkId/C`、`c=chunkId%C`、`tokenBegin=64*c`；
Varlen：`chunkId` 为规范 sequence-major global chunk，复用现有
`ResolveChunk`、`cu_seqlens` 和可选 `chunk_indices` 映射。每个
`(batch,chunk,head)` 只属于一个 work。输出区间不重叠，不使用跨核同步。
固定 G1 总能整除 HV，因此不产生不足 G 的末组。Host 只产生 CG1，
Kernel 按 HV 直接解码一个 head。inputSlot 按本核有效消费序号交替；无效 head
不搬运、不计算、不发 event。
非法或空 tiling 沿用 Host 拒绝路径，Kernel 的零核保护不产生等待。

实例为 `B=1,HV=96,T=8192`，`C=Nbase=128`。上一轮及本轮实机均为
`Ascend910B3`，已查询 `cube_core_num=20`、`vector_core_num=40`、
L2 cache 192 MiB。以下按 20 AIC 代入；A3 运行时仍读取自身的可用核数。

| CG | Hgroup | Nwork | blockDim | 每核 work 数 | wave / 最后一波有效核数 | 最大 head 工作量 |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| 1 | 96 | 12288 | 20 | 614 或 615 | 615 / 8 | 615 |

原 chunk 优先分核为每核 6/7 个完整 chunk，最大为 `7*96=672` 个 head；
CG1 的理论最大为 615，按等成本 head 假设减少约 8.48%。这是尾波负载
估计，不是实测加速比。CG1 分核保持总 GEMM、GM 字节数及旧路径每 head event 次数；两槽预取另改输入事件，
增加 work 解码次数；三个 token 输入按 head-major 存储，grid-stride 的遍历顺序也影响
L2/GM 行为，最终选择以设备测量为依据。

另外两个模型说明固定 CG1 的任务粒度：`B1 HV32 T160` 的 `Nbase=3<20`，
固定 CG4 只有 24 work，每核 4–8 head；G1 有 96 work，每核 4–5 head。
`B6 HV6 T1084` 的 `Nbase=102>=20`，固定 CG4 每核 20–44 head，G1 为
30–31 head。以上按 head 数估计负载，M 尾块的实际 GEMM 成本和 scalar
开销由同口径 profiling 判断。固定 CG1 保持 Stage 0 数学及 L0 资源；
预取候选激活第二个 L1 输入槽并改变输入事件粒度，新增 user workspace 仍为 0，
平台库 API workspace 保留。

## Stage 0：Cube，BF16 双 GEMM 原位累加并直接写出

| Stage | 执行单元 | 功能 | 主要空间 |
| --- | --- | --- | --- |
| 0 | AIC-only | `qg_scaled @ h` 初始化 FP32 L0C，`Aqk @ v_new` 原位累加，Fixpipe 转 BF16 写 GM | 实际 L1 144 KiB，input slot 0/1；L0A 24 KiB；L0B 48 KiB；L0C 64 KiB；UB 0；GM relay 0 |

全部输入来自算子 GM 输入，没有前置 Stage、Vector Stage 或 AIV 任务。
qg_scaled、Aqk 和 v_new 的 GM 物理布局为 head-major ND；h 的 GM 物理布局
为 NT-first `[B,C,HV,128,128]`。每个 head 的公式及实际执行顺序为：

```text
L0C[slot] = qg_scaled[M,128] @ h[128,128]       // BF16 × BF16 -> FP32, initC=true
L0C[slot] += Aqk[M,M] @ v_new[M,128]           // BF16 × BF16 -> FP32, initC=false
attn_out[M,128] = Fixpipe.F322BF16(L0C[slot])  // 仅此处舍入到 BF16
```

这里没有先将输入转为 FP32 的步骤；FP32 仅指 Cube 乘加的累加类型和 L0C
存储类型。P/R 不分别写 GM，也不分别舍入。第二项是矩阵累加，同属 Cube
Stage 0；同一 M pipe 的指令顺序保证先初始化、再累加。

`state_v_first=false` 时 h 使用 RowMajor 逻辑 `[K,V]`，true 时使用
ColumnMajor 视图解释物理 `[V,K]`，由 L1/L0B copy 执行所需布局转换。
两条路径都读取同一 NT-first state 地址，不做 Vector transpose。
Q/A/V 的 L1 是 zN；H 的 L1 在 `state_v_first=false` 时为 zN，true 时为
nZ。L0A 是 arch22 的 zZ，L0B 为 nZ，L0C 是 FP32 分形布局。实证来源为
本次构建使用的 `third_party/catlass/include/catlass/gemm/helper.hpp`
第 154、200、215、266 行，以及 `gemm/tile/tile_copy.hpp` 第 327–331 行；
已安装 `fla_npu_transformer` vendor 的相同 CATLASS 文件映射一致。

### 片上绝对地址与生命周期

容量依据为本算子的既有 CATLASS `AtlasA2` / arch22 资源接口及 A2/A3
片上上限：L1 512 KiB，L0A/L0B 各 64 KiB，L0C 128 KiB。本次构建日志
确认 CATLASS 来源为仓内 `third_party/catlass/include`，容量定义位于
`catlass/arch/arch.hpp` 的 `AtlasA2`；该文件与 CANN 9.1.0 已安装
`fla_npu_transformer` vendor 文件 SHA256 均为
`f9f730ef9dee785d90fb4a824421a3d514eb44d333abc7ec29efc44e40aa44a1`。
本路径不申请 UB。下表所有区间都是绝对半开区间，单位 KiB；地址和 tensor 起始均按
32B 及分形边界对齐。共享地址模板继续保留最多四个 owner 的布局；A2/A3 固定 CG1 激活
两个输入 slot，按本核连续 work 交替使用，每个 work 搬入一份 head 输入。

| 区域 | qg_scaled BF16 `[64,128]` | h BF16 `[128,128]` | Aqk BF16 `[64,64]` | v_new BF16 `[64,128]` |
| --- | --- | --- | --- | --- |
| L1 input slot 0 | `[0,16)` | `[16,48)` | `[48,56)` | `[56,72)` |
| L1 input slot 1 | `[72,88)` | `[88,120)` | `[120,128)` | `[128,144)` |
| L1 owner 2（A2/A3 未激活，仅保留模板地址） | `[144,160)` | `[160,192)` | `[192,200)` | `[200,216)` |
| L1 owner 3（A2/A3 未激活，仅保留模板地址） | `[216,232)` | `[232,264)` | `[264,272)` | `[272,288)` |

每槽共 72 KiB，两槽最大有效地址/峰值均为 144 KiB。MTE2 首次写入
四输入及尾行 Fill；Q/H、A/V 的最后消费分别为两项 MTE1 读取。每个
slot 的 `MTE1_MTE2(slot)` free 在两项读取后发布；下代同槽覆盖前等待。
`L1[144,512)` 空闲、无数据，最大连续空闲区为 368 KiB，包含未激活
owner 2/3。槽生命周期不会因换 chunk 而重置；尾 chunk 只加载有效输入。

| 区域 | 对象、物理布局与份数 | 大小与生命周期 |
| --- | --- | --- |
| L0A `[0,16)` | Q，BF16，QH plane 0，分形 | 16 KiB；MTE1 写，QH MMAD 最后读，`M_MTE1(0)` 后可覆写 |
| L0A `[16,24)` | A，BF16，AV plane 1，分形 | 8 KiB；MTE1 写，AV MMAD 最后读，`M_MTE1(1)` 后可覆写 |
| L0A `[24,64)` | 空闲，无数据 | 40 KiB 连续空闲 |
| L0B `[0,32)` | H，BF16，QH plane 0，分形 | 32 KiB；MTE1 写，QH MMAD 最后读，`M_MTE1(0)` 后可覆写 |
| L0B `[32,48)` | V，BF16，AV plane 1，分形 | 16 KiB；MTE1 写，AV MMAD 最后读，`M_MTE1(1)` 后可覆写 |
| L0B `[48,64)` | 空闲，无数据 | 16 KiB 连续空闲 |
| L0C `[0,32)` | FP32 `[64,128]`，输出 slot 0 | 32 KiB；首 MMAD 初始化，第二次原位累加，Fixpipe 最后读，`FIX_M(0)` 后可覆写 |
| L0C `[32,64)` | FP32 `[64,128]`，输出 slot 1 | 32 KiB；生命周期同 slot 0，以 `FIX_M(1)` 保护 |
| L0C `[64,128)` | 空闲，无数据 | 64 KiB 连续空闲 |

L0A/L0B 的两个 plane 是两个 GEMM 的独立操作数地址，不是 head 双缓冲。
L0C 的两份才是连续 head 的 MMAD/Fixpipe ping/pong，`outputSlot` 在整个
core 的所有 work 上连续交替，不能在换 chunk、head group 或尾组时重置。
尾块 L0A/L0B 按真实 M/K 的紧凑分形布局定义；`M<16` 时 MTE2 Fill
仅将左操作数各 K 分形中的无效 M 行填零，MMAD 物理 M 补到 16，
Fixpipe 仍只写有效 M 行。`MTE2_MTE1` 同时覆盖 copy 和 Fill 完成。

### GM 地址、搬运量与输出

元素 offset 沿用公共函数：

```text
input(dim) = ((b*HV + head)*T + tokenBegin)*dim
state = ((b*C + stateChunk)*HV + head)*128*128
head-major output = input(128)
token-major output = ((b*T + tokenBegin)*HV + head)*128
```

Dense 的 `stateChunk=c`；varlen 的 `b=0,stateChunk=globalChunk`，C 为
本调用的总 chunk 数。Aqk 物理 row stride 为 64，逻辑 reduction 只取前 M 列；
Q/V row stride 为 128，H 是完整 128×128。Token-major 输出 Fixpipe
`dstStride=HV*128`，head-major 为 128，均直接写 BF16 GM。

算子没有 GM 中间 record、bank 或 relay，新增 user workspace 为 0；
Host workspace 总量仍只等于平台库 API 所需 workspace。所有输入均在
一个 head 的 Stage 0 中从 GM 读一次，不进行跨 Stage 重读或 UB 中转。
沿用原有 AIC GlobalTensor 的 L2 cache 策略，此次不改变 cache hint。

| Tensor | 每 head GM 次数 / 有效字节数 | CG1、M=64 的每 work 字节数 | 驻留、复用与批量方式 |
| --- | --- | ---: | --- |
| qg_scaled | 读 1 次，`2*M*128` | 16 KiB | GM ND2NZ 到 L1 Q input slot 0/1，MTE1 后释放 |
| h | 读 1 次，`2*128*128` | 32 KiB | GM ND2NZ 到 L1 H input slot 0/1，MTE1 后释放 |
| Aqk | 读 1 次，`2*M*M` | 8 KiB | 以物理 stride 64 读取有效 M×M，MTE1 后释放 |
| v_new | 读 1 次，`2*M*128` | 16 KiB | GM ND2NZ 到 L1 V input slot 0/1，MTE1 后释放 |
| attn_out | 写 1 次，`2*M*128` | 16 KiB | 一次 Fixpipe F322BF16 写 GM，按 layout 跳行 |

满 work 的输入共 72 KiB、输出 16 KiB，计算量为
`2*M*128*(128+M)` 次浮点操作；CG1、M64 为 3,145,728 FLOP。
BF16 GEMM→FP32 加法→一次 BF16 cast 的序列与基线相同，未引入
指数、除法、归约或额外 cast，数值风险保持基线。中间上界可分别写为
`128*max|Q|*max|H|` 和 `M*max|A|*max|V|`；精度门禁沿用冻结用例的
有限值域、FP64 CPU 标杆和既有阈值，验证包括两项相消及 BF16 舍入边界。
不将已被 BF16 舍入的输入重新恢复成更高精度源数据。

### 事件资源与跨 work 复用

全部事件作用于同一 AIC，按 HardEvent 方向各自编号。继续使用既有静态
ID 0/1，每个方向最多使用 2 个，低于 arch22 每方向 8 个 event ID 的上限。
该上限由已安装 CANN 9.1.0 的
`aarch64-linux/asc/impl/basic_api/kernel_macros.h` 第 97–102 行确认：
`__NPU_ARCH__=2201` 的 `QUE_MAX_EVENT=8`。
共 11 个方向与 ID 组合；无跨核 flag、Mutex、SyncAll 或 AIV barrier。

| Event / ID | 数量 | 生产方 → 消费方 | 初始状态、保护区域与复用 |
| --- | ---: | --- | --- |
| `MTE2_MTE1(inputSlot=0/1)` | 2 | MTE2 → MTE1 | 无初始 credit；每槽四输入 copy/Fill 后 Set，消费对应 head 前 Wait |
| `MTE1_M(0)` | 1 | MTE1 → M | 每项 L0 copy 后 Set/Wait，保护当前项完整操作数 |
| `MTE1_MTE2(inputSlot=0/1)` | 2 | MTE1 → MTE2 | 两项 copy 后 Set，下次同槽加载前 Wait；未复用的尾 credit 退出时 drain |
| `M_MTE1(0/1)` | 2 | M → MTE1 | 首次 plane 无等待；MMAD 后 Set，下次同 plane copy 前 Wait，保护 L0A/B |
| `M_FIX(outputSlot=0/1)` | 2 | M → FIX | 两项 MMAD 后 Set，Fixpipe 前 Wait，保护完整 FP32 和 |
| `FIX_M(outputSlot=0/1)` | 2 | FIX → M | 首次 slot 无等待；Fixpipe 后 Set，下次同 slot 首 MMAD 前 Wait，保护 L0C |

`l1InputInFlight[2]` 表示已消费输入、尚有 free credit 的槽，不表示仍在读取；
首次槽加载不等待 free。`l0OperandInFlight[2]`、`l0OutputInFlight[2]` 沿用原语义。
输入槽获取发生在当前 head 消费前，对下一有效 head 提前提交搬运；MTE2 queue
的 free 等待只依赖已经提交的前代 MTE1 copy，避免循环依赖。消费依次获取
输入 ready、输出 free、QH operand free/ready、AV operand free/ready；L1 free
由 MTE1 发布，独立于 MMAD/Fixpipe 完成。ready 与 free 均按方向和槽独立编号。

```text
init: inputFreePending[0..1]=false; operand/outputInFlight=false
prefetch(head, s):
    if inputFreePending[s]: MTE2: Wait MTE1_MTE2(s); pending=false
    MTE2: load Q/H/A/V into L1[s]; Fill invalid M rows if needed
    MTE2: Set MTE2_MTE1(s)
consume(head, s, o):
    MTE1: Wait MTE2_MTE1(s)
    if outputInFlight[o]: M: Wait FIX_M(o); outputInFlight[o]=false
    for plane in [QH=0, AV=1]:
        if operandInFlight[plane]: MTE1: Wait M_MTE1(plane)
        MTE1: copy this plane from L1[s] to its fixed L0A/B
        MTE1: Set MTE1_M(0); M: Wait MTE1_M(0)
        M: MMAD into outputSlot o, initC=(plane==0)
        M: Set M_MTE1(plane); operandInFlight[plane]=true
    MTE1: Set MTE1_MTE2(s); inputFreePending[s]=true
    M: Set M_FIX(o); FIX: Wait M_FIX(o)
    FIX: F322BF16 output; Set FIX_M(o); outputInFlight[o]=true
first: current=nextValid(); if exists prefetch(current,0)
loop:
    next=nextValid(); if exists prefetch(next,s^1)
    consume(current,s,o)
    if next absent: break
    current=next; s^=1; o^=1
drain for s in [0,1]:
    if inputFreePending[s]: MTE2: Wait MTE1_MTE2(s)
    if operandInFlight[s]: MTE1: Wait M_MTE1(s)
    if outputInFlight[s]: M: Wait FIX_M(s)
```

单任务只加载/消费 slot0；第二任务使用 slot1；第三任务复用 slot0，依次回绕。
全程按有效消费计数轮转，不取全局 work 的奇偶。无效 work 只移动游标，不生成
ready/free；尾部没有无消费者的预取。所有真实事件均由复用或最终 drain 消费。
两份 L1 独立地址和事件允许 MTE2 搬下一 head 与当前 MTE1/Cube/Fixpipe 并行；
具体重叠与收益须由实际 trace 和 profiler 验证，忙碌率不能单独证明等待下降。
MTE2 总搬运字节数不变，新增并发也可能使 L1 端口竞争增加。模型 case、配对
`Task Duration(us)`、噪声范围及采用结论归档在 ATK README。

实现隔离要求是 `ProcessA2A3()`、新 helper/member 和额外 template 参数只在
非 310 分支出现；310 预处理后的 Cube/Vector token 与本次基线一致。
A5 Host/调度/数学/搬运/同步不变；本次仅有 A2 实机验证，A3 只构建验证。

## layout 与元数据

三个 token 输入始终 head-major，h 为 NT-first dense `[B,C,HV,128,128]`，
带 cu_seqlens 的 h 为 `[1,C,HV,128,128]`。
`qg_scaled/Aqk` 为 rank-4 时允许
`BSND/BNSD` 输出，为 rank-3 时允许 `TND/NTD` 输出；FwdH 主路径的
`v_new` 仍保留 rank-4 首维 1，独立调用也允许 rank-3 `v_new`。
输入的物理形状不随 `output_layout` 变化。
公共 `StateOffset` 使用 `((b*C+c)*HV+head)*128*128`；packed 的 c 为
sequence-major global chunk。Cube 与 A5 Vector mover 共享此函数。
KDA V2 直接传入共享 FwdH 的 NT-first h，无额外 chunk/head 转置。
`state_v_first=true` 时，读取 `h` 时交换末两维语义，避免变更公开
输入的物理存储。

变长 `cu_seqlens` 严格递增且覆盖全部 T token。若提供
`chunk_indices`，它必须与 `cu_seqlens` 同时存在，且恰为所有
`(sequence_id, local_chunk_id)` 的规范 sequence-major 枚举。输出
始终保持输入 token 的相对顺序；`BSND/BNSD` 交换 rank-4 的
token/head 轴，`TND/NTD` 交换 rank-3 的 token/head 轴。

## 兼容边界

独立算子不接收 HK；GVA 复用发生在 Prepare，Finalize 只能验证
已展开为 HV 的数据。`h/final_state` 的公开保留策略属于 FwdH 或
完整 forward，Finalize 必须实际收到内部 `h`，即使调用者不请求
公开 intermediate states。稳定 `fla_npu.ops.ascendc` 入口通过 ctypes
直接调用 aclnn，不注册 legacy `torch.ops.npu` 接口。算子私有 ATK
直调只验证设备实现，不能替代稳定 Python 入口验证。
