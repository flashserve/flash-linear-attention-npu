# 当前实现设计

方案设计规则版本：`V2`。接口以 [api.md](api.md) 为准。
A2/A3/A5 保持四 head 工作项；A6 按用户确认的 CG1、四缓冲方案实现，见下一节。

## A6：CG1 与四缓冲

实现依据为 `info/A6 适配详设：chunk_kda_bwd_prepare.md`。A6 Cube 参照 A5，使用
Catlass 的 `Ascend960DT`、`PackedTileCopyTlaToUB`、TLA tensor 与 `TileMmadTla`，
完成 GM→L1、L1→L0、MMAD 和 L0C→UB；`CATLASS_ARCH=9201` 选择目标架构实现。
公开参数、TilingData 布局、CPU 数学语义、输出顺序均不改变。

### 任务域与平台分派

固定长度 `Nchunk=B*ceil(T/64)`，变长 `Nchunk=totalChunkNum`；
`chunkTaskNum=Nchunk`、`headWindowNum=H`、`workTaskNum=Nchunk*H`。
Host 用 `GetCurNpuArch()==DAV_9201` 识别 A6，
按 AIC/AIV 核数相同的约定，只查询 AIC 核数，`blockDim=min(AIC,workTaskNum)`。
例如 B=1、H=5、T=65 时，Nchunk=2、Nwork=10；平台 AIC=24 时启动 10 个核组，
一轮覆盖全部任务。四份缓冲是每个核组的驻留深度，不是每个任务的 head 数。

Kernel 的 9201 分支在 3510 和 arch22 分支之前，在入口通过
`KERNEL_TASK_TYPE_DEFAULT` 选择 MIX_AIC_1_1，其他架构选择 MIX_AIC_1_2。
TilingKey 头文件统一声明业务模板组合，不包含架构条件或 kernel type 选择。
唯一 AIV 的 GetBlockIdx 与 AIC 使用相同逻辑核号，无 AIV1 或加 16 路由。
TilingKey 仍按既有 `(layout, isVariable, doTokenMajor)` 编码；A6 可达 key 为
0（定长 BNSD）和 6（变长 NTD），doTokenMajor 恒为 false。

核组 p 按 `work=p+n*blockDim` 遍历，解码为
`head=work/Nchunk`、`chunkTask=work%Nchunk`。两侧共用 WorkIterator 与 ResolveChunk，
按相同有效任务顺序前进，`owner=s=n%4`。不能用绝对 head 或 work 的模 4 值选槽。
合法 metadata 下每个 work 均有效；防御性的无效 chunk 跳过也在两侧共用的 iterator 中完成。

### Stage 结果与内存

| Stage | 执行单元及功能 | 结果 |
| --- | --- | --- |
| 0 | Cube，按 A→Q→D 顺序执行三次 BF16 输入、FP32 累加的 MMAD | FP32 A_raw/Q、最终一次 RN BF16 D，经 Fixpipe 逐路写 UB |
| 1 | Vector，一次完整 task VF 完成 A 的下三角选择和 FP32 scale；Q/D 不做 VF | FP32 d_aqk、FP32 dq_raw、BF16 dv，经 MTE3 写 GM |

每份输入 owner 的相对地址：

| L1 区间 | 数据 | 容量 | 最后消费者 |
| --- | --- | ---: | --- |
| [0x00000,0x04000) | dO，BF16 [64,128] | 16 KiB | D 的 L1→L0B |
| [0x04000,0x08000) | vNew，BF16 [64,128] | 16 KiB | A 的 L1→L0B |
| [0x08000,0x10000) | h，BF16 [128,128] | 32 KiB | Q 的 L1→L0B |
| [0x10000,0x12000) | aqk，BF16 [64,64] | 8 KiB | D 的 L1→L0A |

四份 owner 的基址为 `owner*0x12000`，总区间 [0,0x48000)，288 KiB。
按 512 KiB 预算，剩余连续 224 KiB。每份完整输入仅发布一次 MTE2_MTE1 ready。
首先提交最多四个有效 task 的 Load，随后 Compute(n)→Load(n+4) 滚动推进，
不存在“四份输入全部 ready”或“四个任务全部结束”的整体 barrier。

每组 UB 的相对地址：

| UB 区间 | 数据 | 容量 | 最后消费者 |
| --- | --- | ---: | --- |
| [0x00000,0x04000) | FP32 A_raw [64,64] | 16 KiB | 当前 task VF |
| [0x04000,0x08000) | FP32 A_out [64,64] | 16 KiB | d_aqk 的 MTE3 |
| [0x08000,0x10000) | FP32 Q [64,128] | 32 KiB | dq_raw 的 MTE3 |
| [0x10000,0x14000) | BF16 D [64,128] | 16 KiB | dv 的 MTE3 |

四组基址为 `s*0x14000`，共 320 KiB；共享 mask 在 [0x50000,0x54000)，16 KiB。
AIV 使用一个连续 TBuf，AIC 的 Fixpipe 按相同 UB 绝对偏移寻址，不依赖多个分配的排列。
UB 共 336 KiB，376 KiB 可用预算内剩余连续 40 KiB，物理末尾 8 KiB 不参与分配。
mask 每核只构造一次。L1、UB 的实际容量和 API 支持仍须目标 SDK 确认。

L0A/B/C 各保留 [0,0x8000)、[0x8000,0x10000) 两个 32 KiB 槽，各使用 64 KiB。
按 L0A/B 64 KiB、L0C 256 KiB 预算，A/B 无剩余，C 剩余连续 192 KiB。
`slot=matmulStep++%2` 在 task 间连续推进。

### Stage 0：Cube，三次矩阵乘与逐路发布

A=`dO[M,128] @ vNew[M,128].T`，Q=`dO[M,128] @ h.T`，
state_v_first 为 true 时 Q=`dO @ h`；D=`aqk[:M,:M].T @ dO[M,128]`。
aqk 的有效上三角保留，不能把用于 d_aqk 的 tril 错用于 D 的输入。
A/Q/D 分别定义 TileCopyA、TileCopyQ、TileCopyD：A 使用 RowMajor/ColumnMajor，
Q 按 state_v_first 选择 RowMajor/RowMajor 或 RowMajor/ColumnMajor，D 使用
ColumnMajor/RowMajor。TLA 的 ColumnMajor 视图表达 vNew.T、h.T 和 aqk.T，
不在 GM 中生成转置副本。L1 采用对应的 zN/nZ 视图，L0A 为 zN、L0B 为 nZ。

`CopyInput` 用 Catlass `TileCopyTla` 完成 GM→L1；`ComputeAndCopyToUb` 参照 A5
集中完成 L1→L0、MMAD、Fixpipe 和结果通知。L1 tensor 按完整物理 owner 尺寸建立，
之后切取 align16(M) 的计算 tile，不能用 M 重新建立紧凑的 L1 分形行距。
L0 tensor 按实际 MMAD 尺寸建立，输出 UB tensor 则保留 64/128 的固定行距。
`PackedTileCopyTlaToUB::CopyL0CToDst` 使用 NO_SPLIT，显式指定 subBlockId=0、
unitFlag=0；A/Q 的输出为 FP32，D 的输出为 BF16。

Cube 的 `Process` 按 A5 写法只构造一份 `Catlass::Arch::Resource<Ascend960DT>`，
通过 `l1Buf/l0ABuf/l0BBuf/l0CBuf/ubBuf.GetBufferByByte<T>(offset)` 获取缓冲视图。
Resource 为各存储位置分配架构定义的完整 arena；上表描述当前算法实际使用的区间，
不将 arena 的完整容量计作新增数据槽。Fixpipe 的目标 UB 偏移与 AIV 的连续 TBuf 一致。
AIC 入口和 Cube Init 不再接收或另建 TPipe；AIV 保留自身的 TPipe。

Resource 内部 TPipe 在成员缓冲初始化之后调用 `Destroy()`，释放其初始化同步；
随后才调用 Cube `InitEvents` 建立固定方向内 ID 的 ready/free 闭环。
整个任务循环结束后调用 `DrainEvents` 消费剩余 free credit 和所有 UB_FREE，
再离开 Resource 的作用域。不能在循环内反复构造 Resource，也不使用已经 Destroy 的
`resource.pipe` 分配或释放事件。

A 消费 aFree/bFree 后加载 dO 和 vNew；MMAD 后只归还 bFree。
Q 复用 A 的 L0A dO，不再等待 aFree、不重新加载 dO，只加载 h；MMAD 后归还 A/B。
D 重新获得 A/B 槽、加载 aqk 转置与 dO；MTE1 读完即发布 ownerFree，
让下一代输入的 MTE2 与 D 的 MMAD/Fixpipe、AIV 写回独立推进。

每次 MMAD 前等待 cFree，结果通过 M_FIX 通知 Fixpipe；
Fixpipe 结束读取后归还 FIX_M 的 cFree。A/Q 保持 FP32，D 仅最终一次 F322BF16。
每个 task 只在 A 的 Fixpipe 前获取 UB_FREE，Q/D 不重复获取。

M 不是 16 的倍数时按 align16(M) 的 MMAD 粒度计算。尾块的 dO/vNew/aqk
物理 L1 tile 先用 Fill 写 BF16 零位型，再由 ND2NZ 写有效矩形；h 全量搬入。
填充和搬运由 MTE2 barrier 排序，禁止使用旧 owner 的归约 padding。
Fixpipe 的 mSize 仅取 M，A 的 dstStride=64，Q/D 的 dstStride=128；GM 不写 padding 行。

### Stage 1：Vector，下三角选择、缩放和整组释放

BuildTriScaleMask 沿用 A5 的先写零、再写下三角 scale 的方式，zero/scaled 在循环外初始化；
按 VECTOR_REG_WIDTH 分段覆盖每行，所有读写限制在当前行内。
ApplyTriScaleMask 对 M 行各覆盖全部 64 列，先 Select 排除上三角/尾列，再乘 scale，
最后 Select 保证无效元素严格为零。有效位置不跳过 scale=0 的乘法；
因此有效非有限值仍遵循原有数学传播，未新增 clamp 或输入值域调整。
一次 VF 包含当前 task 全部有效行，不拆成多个 Stage。

三条输出依次通过 MTE3 写出 A_out、Q、D；Q/D 只搬运、不再次转 dtype。
三份公开输出设置 L2 bypass，输入保持基础缓存策略；输入每 task 各读一次，
dO 在 L1 供 A/D 使用、在 L0A 供 A/Q 复用。没有 GM 中间结果，用户 workspace=0，
总 workspace 仅为平台查询的系统 workspace。
满 task 的有效 GM 读 72 KiB、写 64 KiB；尾块只读取有效 token 行和 aqk 有效方形。

### 同步闭环

| 跨核 mode 0x4 flag | ID | 方向 | 初始化 |
| --- | --- | --- | --- |
| UB_FREE[s] | 0+s | AIV MTE3→AIC FIX | 每组 1 份 |
| A_READY[s] | 4+s | AIC FIX→AIV V | 0 |
| Q_READY[s] | 8+s | AIC FIX→AIV MTE3 | 0 |
| D_READY[s] | 12+s | AIC FIX→AIV MTE3 | 0 |

UB_FREE 只能在 D 的 MTE3 写出后发布；A_raw 被 VF 消费后也不提前释放整组。
同组下一代先取 UB_FREE，保证上一代的三个 READY 已被消费。

AIC 核内事件参照 A5 使用方向内固定 ID；不同 HardEvent 方向的相同 ID 相互独立：

| 事件 | HardEvent | ID |
| --- | --- | --- |
| ownerFree[s] | MTE1_MTE2 | 0..3 |
| inputReady[s] | MTE2_MTE1 | 0..3 |
| aFree[s] | M_MTE1 | 0..1 |
| bFree[s] | M_MTE1 | 2..3 |
| operandReady[s] | MTE1_M | 0..1 |
| resultReady[s] | M_FIX | 0..1 |
| cFree[s] | FIX_M | 0..1 |

这些 ID 在 Resource 完成内部 TPipe 的初始化同步清理后才初始化，避免与其保留事件重叠。
AIV 的 V_MTE3/MTE3_V 各 4 个事件仍由自身 TPipe 分配、释放。
ownerFree、A/B/C free、outFree 初始各有一份 credit，其余 ready 初始为零。
ready 在正常操作中逐一消费；最终 drain 消费所有 ownerFree、A/B/C free 与 UB_FREE，
包括未分配到任务的空核及未使用槽的初始 credit。

```text
AIC: 预取最多4个Load
     当前inputReady → A(保留dO) → Q(复用dO并释放) → D(加载后释放owner)
     每次MMAD → resultReady → Fixpipe → 对应A/Q/D_READY → cFree
     A的Fixpipe前额外等待一次UB_FREE
     提交Load(n+4)，推进下一个task
AIV: A_READY + outFree → 一次VF → V_MTE3 → 搬A_out → outFree
     Q_READY → 搬Q → D_READY → 搬D → UB_FREE
结束: AIC消费全部最终ownerFree、A/B/C free及四个UB_FREE；AIV消费全部outFree
      ready按task消费一次，drain不重复等待；未用槽保留初始化credit
```

### 验证边界

覆盖两种支持布局、两种状态方向、奇数 head、多 batch、变长空序列和规范 metadata，
M=1/15/16/17/63/64、T=65/127/128/129；每核任务深度 1/2/3/4/5/7/8/9
检查不足四份预取、首次回绕、重复代次和最后一个槽。h 与 aqk 使用非对称输入，
aqk 有效上三角非零；上三角/尾列的非有限值不得泄漏到 d_aqk 无效区。
Host contract 检查 CG1 任务数、按 AIC 核数限制 blockDim、0 核拒绝、workspace 和布局拒绝；
设备 smoke 复用独立 CPU 标杆，正式 ATK 验收仍须替换既有零输出占位标杆后按测试规范完成。
实际执行结果与尚未验证项记在 [validation.md](validation.md)，不以静态分析代替设备结论。

## A5 四布局适配设计

基线为主线 `67dc99d7` 的 arch35 实现，公开接口对齐 `8c630459`。
本次只调整输入地址、GM tensor 的物理行距、平台路由与调用参数。
继续沿用现有整 head AIV 分工，不在接口适配中重排流水或变更缓冲所有权。

Stage 0（Cube）仍按每工作项最多四个 head，依次计算
`A = dO @ v_new.T`、`Q = dO @ h.T`（state_v_first 时为 `dO @ h`）、
`D = Aqk.T @ dO`。输入 BF16、累加 FP32；A/Q 以 FP32，D 以 BF16
经一次 Fixpipe 进入对应 AIV 的 UB。A/Q 复用 dO 的 L0A。
输入起点在 BNSD/NTD 为 `((b*H+h)*T+t)*D`，在 BSND/TND 为
`((b*T+t)*H+h)*D`；相邻 token 行距分别为 D 和 H*D。
packed 的 b 恒为 0；h 仍用原 chunk-major 寻址。
GM tensor 的 stride 与逻辑 tile 大小分开，L1/L0 的紧密布局不变。
每 pair 的两个 head 仍分别落入原有 L1 owner；不增加 NPU transpose。
超过批量搬运字段范围的 head 间距改用两次单矩阵搬运，ready 在两次完成后发布。
固定依赖 CATLASS `41bf90da` 的 A5 TileCopy 接收 uint32_t 行距和矩阵间距；
矩阵间距使用上述拆分，超限的 token 行距在 host 明确拒绝，不做窄化截断。

Stage 1（Vector）保持原 head 交错分工与两份 A raw/output slot；
先用下三角有效 mask 排除无效位置，再缩放 A，保证非有限上三角和尾列也为零。
Q/D 只做 MTE3 写回。三路输出始终按原 head-major offset 写出。

每 head L1 区间为 `[0,16)` KiB dO、`[16,32)` v_new、`[32,64)` h、
`[64,72)` Aqk；四份共 288 KiB。L0A/B/C 各两份 32 KiB，共各 64 KiB。
每 AIV 的 A raw `[0,32)`、A output `[32,64)`、mask `[64,80)`、
Q `[80,112)`、D `[112,128)` KiB，共 128 KiB，保持原 192 KiB UB 上限。
所有 MTE2_MTE1 pair-ready、MTE1_MTE2 owner-free、M_MTE1、M_FIX/FIX_M
及跨核 mode 4 ready/free 不变，奇数 head 仍只发布实际 head 的通知。
A5 不使用 A2/A3 的 GM 中转 scratch，仅申请 SDK 系统 workspace。

共享 host 按 `layout + 4 * isVariable` 生成 key 0..7，编译架构选择 arch22 或 arch35。
layout 决定 rank，cu_seqlens 决定变长；四维变长要求 B=1，三维定长按单序列处理。
h 始终 chunk-major，rank 随 Token rank；变长缺省 chunk_indices 由 L0 生成。
删除 standalone 的旧 Phase A 及 FP16 fallback，统一使用优化 BF16 实现。
A5 暂只开放定长 BNSD、变长 NTD，新增组合明确拒绝。
旧 A5 单体 backward 仍引用的共享 Phase A 源码不在本次删除范围。
旧完整 backward 和 V2 都显式传入 BNSD/NTD，保持其输出消费方式。

本次无新增性能目标；需对四布局、两种状态方向、tail、奇数 head、多 batch、
packed 空序列及同核多任务做回归。原有路径性能、设备精度与 sanitizer 仍需跑卡验证。

实现采用 arch22 的 A2/A3 GM 中转路径。
接口由 [api.md](api.md) 固定；验收用例及结果记录在
[ATK README](../../../../../../tests/atk/chunk_kda_bwd_prepare/README.md)。

Stage 0 顺序计算 A、Q、D；A 通过每个 AIC 的四条 FP32 workspace record 交给两个 AIV，
Q/D 直接写公开输出。Stage 1 负责三角掩码、scale 和尾列清零。每份 L0C 只有一次 Fixpipe。

资源为 L1 288 KiB、L0A/L0B/L0C 各 64 KiB、每个 AIV UB 80 KiB、每个实际 AIC
user workspace 64 KiB。系统 workspace 单独由 SDK 查询并计入总量。

目标 SDK 查询得到 A2/A3 的 L1 可用值为 524032 字节、L0A/L0B=64 KiB、L0C=128 KiB、
UB 可用值为 196352 字节，实现使用量均在容量内。系统 workspace 为 16 MiB，
实际 AIC 数由平台查询并与任务数取最小值。ND2NZ 使用 arch22 硬件搬运路径，
SDK 实现直接调用 GM→L1 指令，不使用软件临时 UB。

CANN 9.1.0 的 TPipe 在 AIC 内部占用三个 M_MTE1 事件；算子通过 AllocEventID
另申请四个，该方向总计七个。MTE2_MTE1 申请六个，其余方向分别申请二或四个。
全部事件按同一方向成对使用并在退出前 drain/release；不手工占用内部事件编号。

CHUNK_INDEPENDENT V1 开发参考提供独立任务跨步调度和同步闭环的工程组织。
本实现的四 head 窗口、四 record、双 slot、分形布局和同步计数均取自本算子的设计。

## 批量搬运优化

以 arch35 的 pair 预载、四份 L1 owner、A/Q 共用 dO 和消费者完成后释放资源为参考。
arch22 的 ND2NZ 源行距和矩阵间距均为 16 位：两者不超过 65535 元素时，
相邻两个 head 通过 ndNum=2 搬入相隔 36864 个 BF16 元素的独立 owner；
矩阵间距超限时分别搬 head，行距超限时沿用逐 token 行搬运。
每对 head 的 A/Q/D READY 只在对应两份输入的最后一次搬运之后设置。

L1→L0 使用 LoadData 的 repeatTimes/srcStride/dstGap 批量搬运 K 条带。
L0A 每个 M 分形行发一条指令；L0B 每个 N 分形列发一条指令，dstGap=N 分形数-1。
每个 repeat 的源、目的地址和分形内转置与原单分形路径一致，M/K/N 及尾块语义不变。
当转置视图的 N 分形数等于 L1 的物理 row pitch / 16 时，NZ 到 ZN 的分形顺序相同，
整块 L0B 进一步用一次连续 LoadData 完成；其它情况沿用 K 条带搬运。
A 的 dO 继续驻留 L0A，Q 仅搬 H 到另一 L0B；D 完成 L1→L0 后释放当前 L1 owner。
两个 L0C slot 仍分别由 M_FIX/FIX_M 保护，每份结果只 Fixpipe 一次。

A 的结果保持 FP32 L0C→GM→UB，mode 2 广播由两个 AIV 同时确认 READY/FREE；
非目标 AIV 仅确认，目标 AIV 在 MTE2 完成 GM 读取后释放 workspace record。
完整 64 行 chunk 使用一次 Select 去除上三角，再一次 Muls 缩放；Select 防止无效区
NaN/Inf 通过零乘法泄漏。三角 bit mask 初始化经 S_V 配对事件后供 Vector 读取。
尾块仍只读取已初始化的有效三角区，严格清零上三角和尾列。raw UB 在 V 消费完成后
释放，独立 output UB 在 MTE3 写回完成后释放，workspace、UB、L1 和 L0 容量保持不变。

沿用 arch35 的流式 cache 策略：四份输入只从 GM 读取一次并在 L1 复用，三份公开输出
按流式写回，均设置 CACHE_MODE_DISABLE；小容量 GM 中转 record 保留默认 cache 策略。

BSND/TND 的相邻 head 使用 pair 内 SoA：dO[2]、v_new[2]、H[2]、Aqk[2]，
每 pair 为 144 KiB。dO/v_new 和完整块 Aqk 一次搬运两个 head 的连续列，
再按独立 NZ 视图参与计算；Aqk 尾块保留逐 head 的有效方形读取。
输入逻辑字节数、owner 生命周期、同步与总 L1 容量不变。

## NZ 尾行初始化

GM→L1 ND2NZ 只写逻辑行，L1→L0 的 LoadData2D 则读取完整 16×16 分形。
当有效行数 `rows` 未对齐 16 时，在每个 NZ 列块中将
`[rows, Align16(rows))` 补零；列块的起点间隔为 `pitch × 16` 个 BF16 元素，
不能把这一区域当作紧密连续矩阵清零。每份矩阵从 `rows × 16` 元素偏移开始，
InitConstValue 的 `repeatTimes=Align16(cols)/16`、`blockNum=Align16(rows)-rows`、
`dstGap=pitch-blockNum`，两份独立矩阵分别按原有 head stride 处理。

零填充通过 uint16 视图写入 BF16 的零位型 `0x0000`，使用目标 SDK 支持的
L1 填充指令及其 sanitizer 插桩路径。填充与后续 ND2NZ 均在 MTE2 pipe，
以 MTE2 barrier 保序，最终仍由原有 MTE2_MTE1 READY 事件交给消费者。
16 对齐行不触发补零；L1/UB/workspace 容量、GM 读取次数、pair READY 事件数
以及完整 chunk 搬运路径均不改变。

## 调用层与诊断

公开入口优先选择 `_stable.py`，通过单独的 C++ Stable-ABI adapter 分配三路 head-major
输出，并以同一 ACLNN 参数顺序调用；四布局名由统一枚举顺序转换。
packed 元数据在 host 侧传递。ctypes 保留为参考与显式回退路径。
ACLNN 参数检查使用 opbase 的 Error Message 宏；Stable-ABI 读取失败线程上的
最近错误详情并将其附加到异常中，保留原始 ACLNN 返回码。
按 SDK 宏是否存在选择错误上报：新 SDK 保留原生 `OP_LOGE_FOR_INVALID_*`；
旧 SDK 使用已有的属性/类型结构化上报，保留参数名、实际值与期望值。
缺失参数直接使用 `REPORT_PREDEFINED_ERR_MSG` 上报 EZ0004，绕过 CANN 9.0.0
`OP_LOGE_WITH_INVALID_INPUT` 中的形参笔误，不改写 SDK 宏或头文件。
950 注册继续保留；A5 适配设计见本文前节，其验证不属于原 A2/A3 结果。

调用层与错误上报回归：32 组正常输入的 96 路输出逐位通过，Stable 后端无回退；
29 项 ACLNN 非法输入检查及公开 Stable 入口的错误详情检查通过。
单算子 A2/A3 编译通过，A2 原始 ATK 用例 0..19 精度全部通过，未改变输入范围和精度标准。

本轮最终精度回归按用户指定以原始用例 id 0..19 为范围；其余历史验收结果不代表优化版本
已重新完成全量、确定性或内存验收。模型性能继续使用 case 198/199，原输入和种子不变，
每条预热 5 次后采集一次正式调用，读取 msprof 的 Task Duration(us)。
