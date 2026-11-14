# ChunkKdaBwdRecompute A6 实现设计

方案设计规则版本：V2。接口以 [api.md](api.md) 为准。
本次采用用户确认的 A6 简化方案：1 AIC : 1 AIV、CG=1、直接 UB→L1、独立双输入/双结果区。
这是用户指定的 A6 映射与数据通路，对应 V2 中双 AIV 切片、Vector→Cube 经 GM 的硬件假设采用上述替换。

## 1. 目标

在 `__NPU_ARCH__==9201` 下新增 arch92，实现 safe-gate、chunk cumsum 和重计算，保留原有公开接口、
四种 g/beta dtype key、A2/A3 默认入口与 arch35 入口。实现参考 GDN Prepare 的 arch92
LoadData、MMAD、Fixpipe 和 mode 0x4 写法；不引入其三 head 分包。
实现保留四种 g/beta dtype key，并按 A6 的 1:1 MIX 资源模型组织 Vector 与 Cube 阶段。

## 2. V0 / C1 完整详设

`M` 为当前 chunk 有效行数，`1<=M<=64`；K=V=128，`hv=work%HV`，`hk=hv/(HV/HK)`。
定长 `Nbase=B*ceil(T/64)`；变长 Nbase 为 `chunk_indices` 中 pair 的数量。
`Nwork=Nbase*HV`。A6 核数按 MergeFwdBwdKernel 的 CompileInfo 传递方式处理：
TilingParse 在识别 DAV_9201 后设置默认 AIC/AIV=64/64，分别用平台查询的非零值覆盖，
保存到 CompileInfo。64/64 是 `Ascend960DT_968B7` 的调度默认值，用于运行期平台查询缺省时的核数配置。
运行期 Tiling 先读取 context 的 `GetCoreNumAic()/GetCoreNumAiv()`，仅对为 0 的一侧
读取 `GetCompileInfo` 对应字段；CompileInfo 不存在或该字段仍为 0 时使用对应的 A6 默认值。
两侧取较小值形成 1:1 逻辑核对，不再按固定 SoC 名调用 GetPlatformInfos。
TilingParse 缺少平台对象时保留既有 28/56 初值，无法据此识别 A6；A2/A3/A5 分支不变。
`usedAic=min(上述逻辑核对数,Nwork)`；每核
`work=coreIdx+ordinal*usedAic`，`slot=ordinal%2`。slot 必须根据每核 ordinal 轮转，
不能使用 `work%2`，否则偶数 grid stride 下只有一个 slot 活跃。
空调用 usedAic=0，启动一个只排空本地初始事件的 block，不产生跨核通知或数据访问。

Host 以 `isAscend960` 标识 A6，架构判断仍为 `NpuArch::DAV_9201`。TilingProcessor 沿用公共
`CheckSpec`，不增加 A6 专有 rank、shape、变长元数据或 UB/L1 容量校验；保留原有变长元数据指针获取和传递。
A6 的 `SetVecRow` 固定设置 vecRow=64、用户 workspace=0。片上布局沿用现有常量定义；arch92 struct 头文件按用户要求删除静态校验。

例如 B=1、T=129、HV=3，Nwork=9；若可用 AIC/AIV 配对数为 4，则 blockDim=4，
core0 处理 work=0/4/8，对应 slot=0/1/0；其它核各处理两份 work。没有 head group 尾分支。

| Stage | 执行单元 | 功能 | 主要空间 |
| --- | --- | --- | --- |
| V0 | 唯一 AIV | gate/cumsum、qg/kg/kbg/vb | UB 356 KiB，两份 IN/OUT/META |
| C1 | AIC | U=A@vb、W=A@kbg | L1 144 KiB；L0A 8、L0B 16、L0C 32 KiB |

### V0：Vector

```text
x = FP32(g) + dt_bias[hv]              # bias 可选
gate = lower_bound * sigmoid(exp(A_log[hv]) * x)   # gate 开启
gate = FP32(g)                        # gate 关闭
gk = cumsum_chunk(gate) * (use_exp2 ? 1/ln2 : 1)
E(z) = use_exp2 ? exp2(z) : exp(z)
qg = BF16(q[hk] * E(gk))
kbg = BF16(k[hk] * E(gk) * beta[hv])
vb = BF16(v[hv] * beta[hv])
kg = BF16(k[hk] * E(gk[M-1]-gk))
```

实际 FP32 运算沿用 arch35 快路径的次序：gate 的 numerator 为 `lower_bound*scale`，
先除以 `1+exp(-exp(A_log)*x)`，再累加；关闭 gate 时先乘 scale 再累加。
g/beta 的 BF16→FP32 转换在寄存器内进行。完整 64 行使用一次 VF，gk 末行保留在寄存器；
尾块先完成 cumsum VF，再通过 V barrier 调用结果 VF，并读取已存储的 gk 末行。
这避免短循环中末行读取被编译器提前的问题。

所有模式的 ExpPair 均对低/高两个寄存器直接执行 FP32 Exp，
gate+exp2 路径仍将指数乘 ln2，但删除原 [-11,11] 裁剪及 half 往返转换，避免深衰减值被该裁剪抬高。
FAST 模板参数保留以兼容现有调用；不新增整块 UB scratch。
GateCumsum、ProduceResults 的有效行计算与 RHS padding 写零均使用固定 [0,64) 循环：
计算循环跳过 row>=count，padding 循环跳过 row<count。尾块不读取无效输入或未初始化 gk；
该写法保持固定循环边界，兼容 A6 的 Vector 指令实现。
gk 始终以 FP32 存储；qg/kbg/kg/vb 以 BF16 舍入。gate 开启且 lower_bound<=0 时 gate、cumsum<=0，
qg/kbg 的指数及 kg 的末行差值均<=0；非正门控避免正指数溢出。对于原始 gate、正 lower_bound 或
极端 A_log，沿用输入公式的有限值边界，不额外扩大数值域。
padding 行直接赋 kbg/vb=0，不读取无效 g，也不对无效数据计算 Exp。

单 AIV 的 UB 物理规格为 384 KiB，以 Prepare 相同的 376 KiB 可用预算核算。
下面是绝对半开 byte 地址，均至少 32 B 对齐。

| 地址区间 | 对象、dtype/layout | 大小 | 生产、最后消费与复用 |
| --- | --- | ---: | --- |
| [0x00000,0x14000) | IN0，ND | 80 KiB | MTE2 写，V 最后读取后 INPUT_FREE |
| [0x14000,0x28000) | IN1，ND | 80 KiB | 同上 |
| [0x28000,0x40000) | OUT0，ND | 96 KiB | V 写，MTE3 全部读取后 OUTPUT_FREE |
| [0x40000,0x58000) | OUT1，ND | 96 KiB | 同上 |
| [0x58000,0x58800) | META0 | 2 KiB | MTE2 写，随 IN0 释放 |
| [0x58800,0x59000) | META1 | 2 KiB | MTE2 写，随 IN1 释放 |
| [0x59000,0x5E000) | 空闲，无数据 | 20 KiB | 可用预算内最大连续空闲区 |
| [0x5E000,0x60000) | 保留，无数据 | 8 KiB | 不分配 |

每份 IN 相对 offset：g=0（最大 FP32 [64,128]，32 KiB），q=0x8000、k=0xC000、v=0x10000
（各 BF16 [64,128]，16 KiB）。每份 OUT：gk=0（FP32，32 KiB），qg=0x8000、kbg=0xC000、
vb=0x10000、kg=0x14000（各 BF16，16 KiB）。输入/输出没有别名。
每份 META：beta BF16 位于 [0,0x80)，beta FP32 位于 [0x80,0x180)，按当前 dtype 使用其中一块；
bias FP32 [128] 位于 [0x180,0x380)，A_log FP32 [1] 在 [0x380,0x3A0) 的 32 B 预留中，
其余为 slot 内对齐余量。A_log 只搬当前 head 的 4 B，不对 GM 做对齐越界读取。

### C1：Cube

```text
U = A @ vb      # [64,64] @ [64,128] -> [64,128]
W = A @ kbg     # [64,64] @ [64,128] -> [64,128]
```

A/RHS 为 BF16，A 与 RHS 的 L1 为 zN/NZ，RHS 经带 transpose 的 LoadData 进入 L0B。
FP32 累加，Fixpipe F322BF16、NZ→ND 直接写 GM。A 只加载到 L0A 一次供两次 MMAD 使用。
两次乘法没有结果依赖；同一 L0 资源按 U→Fixpipe→W→Fixpipe 顺序复用。

| 地址区间 | 对象、dtype/layout | 大小 | 生产、最后消费与复用 |
| --- | --- | ---: | --- |
| L1 [0x00000,0x02000) | A0，BF16 [64,64] zN | 8 KiB | MTE2 写，MTE1 读完后 L1A_FREE0 |
| L1 [0x02000,0x04000) | A1，同上 | 8 KiB | 同上 |
| L1 [0x04000,0x0C000) | kbg0，BF16 [64,128] zN | 32 KiB 预留 | MTE3 写，MTE1 最后读取后 RHS_FREE0 |
| L1 [0x0C000,0x14000) | vb0，同上 | 32 KiB 预留 | 同上 |
| L1 [0x14000,0x1C000) | kbg1，同上 | 32 KiB 预留 | RHS_FREE1 后复用 |
| L1 [0x1C000,0x24000) | vb1，同上 | 32 KiB 预留 | 同上 |
| L1 [0x24000,0x80000) | 空闲，无数据 | 368 KiB | 无消费者 |
| L0A [0,0x2000) | 当前 A，BF16 [64,64] | 8 KiB | MTE1 写，U/W MMAD 读取 |
| L0B [0,0x4000) | 当前 RHS，BF16 [64,128] | 16 KiB | MTE1 写，当前 MMAD 读取 |
| L0C [0,0x8000) | 当前累加结果，FP32 [64,128] NZ | 32 KiB | MMAD 写，Fixpipe 读取 |

每个 RHS 实际载荷 16 KiB，其余为当前 slot 预留。L0A/B 预算各 64 KiB，L0C 预算 256 KiB；
剩余连续空间分别 56/48/224 KiB，不用于其它数据。L0 的下一次使用由 FIX_MTE1 证明上次 Fixpipe 已完成。
L1 A 的 MTE1_MTE2 释放与 RHS 的跨核释放相互独立，下一 work 可以在另一个 L1 slot 预取。

按用户确认的候选实现，CopyA 删除 InitConstValue 和关联 MTE2 barrier，只搬入有效 M×M，
GM 行宽仍为 64；MMAD 保留 M=64、N=128、K=64，Fixpipe 只写有效 M 行。
kbg/vb 在 V0 中对 [M,64) 行置零，A 的其余 L1 区域不主动清零。
有效输出仍会累加 A 的无效列与 RHS 零值的乘积；若未初始化的 A 包含 NaN/Inf，
可能污染有效结果，因此本候选不具备尾块安全保证，不能将 RHS 补零或 Fixpipe 限行当作完整证明。
满块 M=64 的 A 区域全部被搬入，本次删除不改变满块执行路径。

### 搬运与 workspace

用户 workspace=0；所有中间量都在 UB/L1/L0 内，各核有独立片上存储。系统 workspace 由平台提供。
每个 work（M 有效行）的 GM 逻辑字节数与次数如下，bias/A_log 未提供时跳过。

| tensor | GM 读 | GM 写 | 驻留、复用与搬运 |
| --- | --- | --- | --- |
| g | 1 次，M×128×sizeof(g) | 0 | MTE2→IN，VF 最后读取后复用 |
| q/k/v | 各 1 次，各 M×128×2 B | 0 | 同上，按当前 hk/hv 寻址 |
| beta | 1 次，M×sizeof(beta) | 0 | META，支持非 32 B 对齐的尾长 |
| bias/A_log | 各 1 次，512/4 B | 0 | META，随输入释放 |
| A | 1 次，M×M×2 B | 0 | GM→L1 ND2NZ，GM stride=64，L0A 复用 |
| kbg/vb | 0 | 0 | UB→L1 zN→L0B；尾块传完整零填充 tile |
| gk | 0 | 存在输出时 1 次，M×128×4 B | OUT→GM |
| qg/kg | 0 | 各 1 次，各 M×128×2 B | OUT→GM |
| u/w | 0 | 各 1 次，各 M×128×2 B | L0C→GM Fixpipe |

沿用 SDK 默认 L2 策略，不继承 A5 专属指针高位假设。bank 访问布局使用固定连续 K 通道，
通过不同 IN/OUT slot 分离读写生命周期。

### 同步与连续 work

| 资源 | 作用核、方向与数量 | 初始状态 | 复用条件 |
| --- | --- | --- | --- |
| INPUT_FREE[s] | AIV，V_MTE2，2 个动态 event | 每 slot 一次 free | VF 全部读取 IN/META 后 |
| INPUT_READY[s] | AIV，MTE2_V，2 个动态 event | 无 ready | 当前输入完成 MTE2 后 |
| OUTPUT_READY[s] | AIV，V_MTE3，2 个动态 event | 无 ready | 所有结果 VF 完成后 |
| OUTPUT_FREE[s] | AIV，MTE3_V，2 个动态 event | 每 slot 一次 free | UB→L1 和 UB→GM 全部读取后 |
| RHS_READY[s] | AIV→AIC，mode 0x4，flag 6/8 | 无通知 | MTE3 写完两个 RHS 后 |
| RHS_FREE[s] | AIC→AIV，mode 0x4，flag 7/9 | 无通知，首次发布不等待 | MTE1 最后读完两个 RHS 后 |
| L1A_FREE[s] | AIC，MTE1_MTE2，ID 0/1 | 每 slot 一次 free | L1 A 已载入 L0A 后 |
| L1A_READY[s] | AIC，MTE2_MTE1，ID 0/1 | 无 ready | A 的有效 M×M 从 GM→L1 完成后 |
| L0_FREE | AIC，FIX_MTE1，ID 0 | 一次 free | 当前 U 或 W 的 Fixpipe 完成后 |
| OPERANDS_READY | AIC，MTE1_M，ID 0 | 无 ready | 本次 A/B 搬入 L0 后 |
| RESULT_READY | AIC，M_FIX，ID 0 | 无 ready | 当前 MMAD 完成后 |

AIV 每种 HardEvent 使用 2 个 ID，AIC 每种使用最多 2 个 ID；跨核 flag 共 4 个，小于 Prepare 使用的 16 ID 空间。
准确数量上限、event 分配和 A6 SDK 重载仍需目标工具链核对。

```text
AIV:
  初始化 IN0/1 free、OUT0/1 free
  CopyIn(work0)                  # MTE2 等 INPUT_FREE，完成后 INPUT_READY
  for ordinal = 0..本核任务数-1:
    若有下一 work: CopyIn(next)  # 预取到另一 IN，可等待前一 VF 的 INPUT_FREE
    Compute(current)            # 等 INPUT_READY / OUTPUT_FREE
                                # 完成后 INPUT_FREE / OUTPUT_READY
    若 ordinal>0: Publish(previous)
                                # 等 OUTPUT_READY；同 slot 再发布前等 RHS_FREE
                                # 写 L1 RHS，发 RHS_READY；写 GM；发 OUTPUT_FREE
  Publish(last)
  排空 IN0/1 free、OUT0/1 free，以及每个发布过的 L1 slot 的末次 RHS_FREE
  释放动态 event

AIC:
  初始化 A0/1 free、L0 free；CopyA(work0)
  for ordinal = 0..本核任务数-1:
    若有下一 work: CopyA(next)   # MTE2 等 A_FREE，完成后 A_READY
    等 L0_FREE、当前 A_READY
    A→L0A；发 A_FREE
    等 RHS_READY；vb→L0B；OPERANDS_READY→MMAD U→RESULT_READY→Fixpipe U→L0_FREE
    等 L0_FREE；kbg→L0B；发 RHS_FREE
    OPERANDS_READY→MMAD W→RESULT_READY→Fixpipe W→L0_FREE
  排空 A0/1 free 和 L0_FREE
```

固定获取顺序保证没有环形等待：IN 的复用不依赖 OUTPUT_FREE；结果复用只依赖上一轮 MTE3；
MTE3 的 RHS_FREE 由按同一 work 顺序执行的 AIC 发出。每个有效 work 恰好一对 RHS ready/free，
最后一次 free 也被消费；空核只消费本地初始 free，不操作跨核 flag。
VF(n+1) 先于 Publish(n) 排队，避免将上一结果的搬运等待放到下一 VF 的发射之前。
