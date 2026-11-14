# 当前实现设计

当前规则版本：`V2`。采用四 head 工作项及整 head 的 AIV 映射。

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
[ATK README](../../../../../../tests/atk/chunk_kda_bwd_prepare_a3/README.md)。

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
950 注册继续保留；A5 适配设计见本文前节，其验证不属于原 A2/A3 结果。

调用层与错误上报回归：32 组正常输入的 96 路输出逐位通过，Stable 后端无回退；
29 项 ACLNN 非法输入检查及公开 Stable 入口的错误详情检查通过。
单算子 A2/A3 编译通过，A2 原始 ATK 用例 0..19 精度全部通过，未改变输入范围和精度标准。

本轮最终精度回归按用户指定以原始用例 id 0..19 为范围；其余历史验收结果不代表优化版本
已重新完成全量、确定性或内存验收。模型性能继续使用 case 198/199，原输入和种子不变，
每条预热 5 次后采集一次正式调用，读取 msprof 的 Task Duration(us)。
