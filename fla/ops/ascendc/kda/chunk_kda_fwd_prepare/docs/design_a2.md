# ChunkKdaFwdPrepare A2 实现设计

方案设计规则版本：V2；按仓内 [阶段3工作流](../../../../../../docs/agents/03-方案设计.md)
的目标、Stage、资源、搬运和同步结构整理。实现基点为 `02f1424b`；本文记录现有代码，
不把工作流推荐结构或侧线候选写成已实现。公开接口以 [api.md](api.md) 为准。
A5 原设计保留在 [design.md](design.md)，本次不修改 arch35。

## 1. 目标

融合边界为 Q/K norm、gate/beta、chunk-local score、三角求逆和 W/U。
调用链为 Prepare → FwdH → Finalize；跨 chunk 状态递推与最终 attention 不在本算子内。
单次公开调用启动一个 MIX kernel，13个输出槽位不变。

### 1.1 当前是4-head组，不是2-head组

`Shape::kHeadsPerGroup=4`。每个 AIC 与两个 AIV 协作，组内分两轮 pair：

| pair | AIV0 完整处理 | AIV1 完整处理 | AIC |
| --- | --- | --- | --- |
| 0 | local head 0 | local head 1 | 顺序消费 head 0、1 |
| 1 | local head 2 | local head 3 | 顺序消费 head 2、3 |

代码为 `localHead=pair*2+aiv`，每个 AIV 有两个 private slot、一个共享 G/scratch 区。
尾组只执行有效 head，两个 AIV 仍发送配对 ready。两个 pair 是同步粒度，
两个32阶叶子是单个 head 内的求逆分块，都不能称为 CG=2。

这与 V2 R01 的“两个 AIV 横切同一 head，分别遍历所有 head”不同；当前也不是
R20 的 `Nchunk*ceil(HV/CG)` grid-stride 调度。保留此映射是本次文档与代码一致的要求，
不是声称已完成 V2 全部结构整改。将其改成横切映射或参数化 CG=2/3/4 需要另行实现和验证。
同样，R07 的 Cube 中间量 L1 驻留仅部分实现：FP32 T 仍按 R14 经 GM 中转。
R10 的一次读入也不完全满足：GVA 仍按 value head 重读 Q/K，V6 重读保存的上下文。

### 1.2 任务域与一个模型的代入

记有效 chunk 长度 M≤64、维度 D=128、value head 为 hv、Q/K head 为 hk，
G=HV/HK，`hk=floor(hv/G)`。共享 Q/K 检查点仅由该 hk 的 owner 写，
gate、beta、score、求逆和 workspace 仍属于各自 hv。

Generic 的基础 chunk 数为 dense 的 `B*ceil(T/64)` 或 varlen 各段 chunk 数之和。
仅 chunk 数少于可用 AIC 且 HK>1 时，host 才按完整 GVA 组拆 head；
否则一个 work 包含一个 chunk 的所有 head，内部每4 head处理一组。
`Nwork=Nchunk*headPartitionCount`，`usedCoreNum=min(Nwork,availableAIC)`。
两种核使用相同连续 `[workBegin,workEnd)` 区间，不是 grid-stride。
规范 varlen 且未拆 head 时，按 token 位置 lower_bound 调整区间端点。

模型 V01/H4：T=131046、579段、2292 chunk；20 AIC 时不拆 head，
Nwork=2292，组执行总数2292，每组4 head。按 chunk 均分的算术对照为114/115 work每核；
实际 token-balanced 边界取决于完整 cu_seqlens，不能把114/115写成实测分配。
H8 同边界仍有2292 work，每 work 两个4-head组，共4584组。
尾 head/尾行只掩蔽访问和计算，不移除对侧所期待的同步。

### 1.3 性能目标与证据口径

主要32模型为 safe gate、FP32 gate、gate-in-kernel、seed3；扩展精度矩阵保留
32模型×普通/safe gate×FP32/BF16 gate共128配置。完整边界在外部
[A2 ATK交付件](https://gitcode.com/chen-linxin4/work_tool/pull/32) 的128条 `_a2_perf.json`。
不是把两个不同布局或输入配方当作同 case 比较。

逐模型目标为 A5 Task Duration/A2 Task Duration≥0.5，另须自身基线无显著退化。
A2 对照是 current495 加 Target 依赖修复的499冻结安装，候选为02f1424b。
同输入 ABBA、warmup5、3组×50、两个独立窗口；差异小或漂移项追加200。
统一判定值为目标 kernel 的 Task Duration中位数，Event单列。
正式采集可用 `msprof --application='<固定case的单进程入口>' --output=<新目录>`，
审核 op_summary 对应 kernel；开发期 torch_npu profiler 的 kernel_details 原始时间序列
用于配对审计，不能用 PipeUtilization 百分比或仿真周期替代模型耗时。

当前32模型两窗未观察到回退，但跨注册入口 A/A 负对照失败，收益仍是描述性数据；
A5同输入健康设备基线未补齐。因此逐项50%目标及完整性能验收均未证明。
资源上界只能说明方案合法：不能从减少 Exp/搬运直接推出目标可达。
V0/V1/V3/V6 的 Vector/Scalar 发射、T relay、C7及等待是需要实测拆分的候选关键路径。

### 1.4 key 与模式分发

| GateT/BetaT | Generic | Target |
| --- | ---: | ---: |
| BF16/BF16 | 2570 | 68106 |
| FP32/BF16 | 2590 | 68126 |
| BF16/FP32 | 7690 | 不适用 |
| FP32/FP32 | 7710 | 不适用 |

共享 CANN 声明生成 key，无 key-filter。pipelineVariant 与 safe_gate 独立。
normMode/betaMode/gateMode/outputMode/useExp2 由共享字段序列化为72字节 tiling。
AIC/AIV 使用同一个 `RunArch22Selected`，优先 BF16 relay，再 CommonMath，最后运行时泛化。
CommonMath 只匹配 BF16 beta、L2、Sigmoid、exp2、Save，gateMode仍动态；
非匹配输入不隐式强制这些数学模式，不展开全部旧模板组合。

BF16 relay额外要求 FP32 gate、SafeSigmoid、head-major、epsilon=1e-6、lower_bound=-5，
及合法整数GVA映射。scale是host校验后的有限运行时值，不要求1/√128。
coarse 的实际启用条件是 `BF16_RELAY && coarseEligible`；当前并未与 relay 完全解耦。
其条件函数虽然允许部分更宽组合，外层 relay合同仍限制最终可达范围。
不能仅凭2590认定命中BF16 relay。

## 2. Stage 0–N 完整详设

以下为 Generic；Target 的资源与附加 Cube Stage 单独列于2.11。
地址均为字节半开区间，UB为每个AIV独立空间，L1为每个AIC空间。

| Stage | 单元 | 结果 | 主要存储 |
| --- | --- | --- | --- |
| V0 | Vector | Qhat/Khat、betaEff、累计G | private Q/K、shared G、GM上下文 |
| V1 | Vector | Qplus/Kplus/Kminus | 同址private、GM payload |
| C2 | Cube | 四band的两种 raw score | 四head L1 lane、L0、GM raw relay |
| V3 | Vector | Aqk、两个精确叶子逆、B、Akk对角 | private换义、GM payload/Akk |
| C4 | Cube | T=B@X0 | 四份X0/negX1/T、FP32 GM T relay |
| C5 | Cube | Akk左下块=negX1@T | L0C、L1 Akk、可选公开Akk |
| V6 | Vector | qg/kg/qg_scaled、两个C7 RHS | private、other-private临时、GM RHS |
| C7 | Cube | W/U | 专用RHS L1、Akk lane、L0、公开GM |

### 2.1 固定地址与所有权

每个AIV的private基址 `P=0` 或 `0x1C000`，每份0x12000=72KiB。

| UB区间 | 内容及布局 | 首次生产、最后消费、复用条件 |
| --- | --- | --- |
| [0,0x12000) | pair0 private，详见各Stage | ioFree[0]保护 |
| [0x12000,0x1A000) | G `[64,128]` FP32；V3时前部betaEff | sharedFree保护，G在V1后退出，V6重新加载 |
| [0x1A000,0x1C000) | 8KiB共享scratch，含beta offset表 | sharedFree；offset表[0x1BF00,0x1C000)不得破坏 |
| [0x1C000,0x2E000) | pair1 private | ioFree[1]保护 |
| [0x2E000,0x30000) | head-major V1/V6的8KiB临时stage | 各Stage分时使用；sequence-major不分配 |

head-major分配192KiB、sequence-major184KiB，前者连续空闲0、后者未分配尾部8KiB。
private中Q为[P,P+0x4000)、K为[P+0x4000,P+0x8000)，均BF16 `[64,128]`。
beta/gate状态位于P+0x10000之后；不是全部private都能作为自由临时区。

下文按工作流的“地址区间、对象与slot、dtype/layout、大小、生产与最后消费”展开。
地址都是**单个AIV内的绝对字节地址**，不是两AIV容量相加；1KiB=1024B。
两private分别对应本AIV承包的两个head，不能解释成同一head的两份完整ping/pong。
表中同时列pair0/pair1地址，实际Stage只激活当前pair；另一pair中仍未完成的
MTE3源保持原址，直到其ioFree许可被消费。共享区只允许一个pair持有。
表内均为32B对齐基址，DMA尾行使用原DataCopyPad参数；小标量有效字节另列，
不能把一个256B地址槽误认为256个标量。

| L1区间 | 内容、份数 | 生命周期 |
| --- | --- | --- |
| [0,0x48000) | 4×72KiB head lane，基址h*0x12000 | C2 score输入→C4 B；每lane+0x4000存Akk，活到C7末次LoadData |
| [0x48000,0x4C000) | X0，4×4KiB FP32 NZ | C4加载，C4 MMAD末次消费 |
| [0x4C000,0x50000) | negX1，4×4KiB FP32 NZ | C4加载，C5末次消费 |
| [0x50000,0x54000) | T，4×4KiB FP32 NZ | C4 GM重载，C5消费 |
| [0x54000,0x5C000) | 公共常量预留区，Generic Akk不在此驻留 | 当前Generic不作独立Akk池 |
| [0x5C000,0x6C000) | C7第0个head的两个FP32 RHS容量 | BF16仅占各lane有效部分；rhsLaneFree保护 |
| [0x6C000,0x7C000) | C7第1个head的两个FP32 RHS容量 | 两pair分时复用同两槽 |
| [0x7C000,0x80000) | 未分配16KiB | 非已实现额外bank |

Generic L1分配496KiB。L0A/L0B/L0C各64KiB，普通流水使用同一池，
由M_MTE1/FIX_M约束复用；没有Generic双L0槽或任意两代work常驻的实现。
BF16与FP32 Akk分别占每head lane+0x4000起8/16KiB。

每workgroup固定4个GM record，`recordBase=wg*(4*S)+localHead*S`，Generic S=0x33400=205KiB。
同record跨work复用，不是两个workspace bank。host总量为`libApiWorkspace+usedCoreNum*4*S`。

| record内绝对offset | 大小 | 生产→消费 |
| --- | ---: | --- |
| [0,0x4000)、[0x4000,0x8000) | 各16KiB Qhat/Khat | V0→V6，独立于公开输出 |
| [0x8000,0x8400) | 1KiB betaEff容量 | V0→V3/V6，有效64×4B |
| [0x8400,0x1A400) | 72KiB payload | V1→C2后换义V3→C4 |
| [0x1A400,0x1F400) | 20KiB compact raw；C4复用前4KiB为T | C2→V3后，C4→C4重载 |
| [0x1F400,0x27400)、[0x27400,0x2F400) | 各32KiB RHS容量 | V6→C7；BF16各只用前16KiB |
| [0x2F400,0x33400) | 16KiB FP32 Akk | V3→C4/C5；fallback重载至L1 |

### 2.2 Stage V0：Vector，归一化和gate/beta

读取本head Q/K、gate、beta及按需a_log/dt_bias；Q/K按hk映射加载，GVA暂不跨hv复用。
L2路径为BF16→FP32→平方→行ReduceSum→epsilon→Sqrt→Div(1,x)→Mul→CAST_RINT BF16；
关闭norm时保留输入，rstd=1。FP32 gate使用norm32批量归约，BF16 gate保留逐行实现。
norm32临时values=[P+0x8000,P+0xC000)，squares=[P+0xC000,P+0x10000)。
共享scratch的repeat/Brcb/ones分别占[0x1A000,0x1A400)、[0x1A400,0x1A800)、
[0x1A800,0x1A880)，reduceTmp传入0x1B000。
此处依赖已核对CANN ReduceSum AR/reuseSource原位归约行为，不宣称所有SDK均适用。

beta按Raw/Sigmoid/2*Sigmoid模式得到FP32 betaEff。
gate step按预计算、稳定Softplus或SafeSigmoid计算，累加至G；exp2仅改变保存指数域。
Softplus使用exp(-abs(x))及log1p补偿避免直接exp(x)溢出。
safe gate step有lower_bound范围约束；预计算/普通gate不能由safe范围推断精度。
norm的epsilon保护零向量分母；不为提高性能取消有效行判断或扩展已验收值域。
计算后写内部Qhat/Khat/betaEff、公开gk及请求的检查点；输出Q/K检查点只允许hk owner写。
MTE2_V→Vector→V_MTE3→MTE3_V闭环后V1才覆盖Q/K。

V0的UB地址图如下，所有大张量为ND行主序；每行只列一份当前head数据。

| pair0绝对区间 | pair1绝对区间 | 对象、shape、dtype及容量 | 首次生产→最后消费/复用 |
| --- | --- | --- | --- |
| [0,0x4000) | [0x1C000,0x20000) | Q→Qhat `[64,128]` BF16，16KiB | MTE2→norm/V0写回；V1原址生成Q+ |
| [0x4000,0x8000) | [0x20000,0x24000) | K→Khat，同上16KiB | MTE2→norm/V0写回；V1生成K+ |
| [0x8000,0xC000) | [0x24000,0x28000) | FP32 gate分支：norm values `[32,128]` FP32，16KiB | Cast→归一化Cast回Q/K；Q、K分时复用 |
| [0xC000,0x10000) | [0x28000,0x2C000) | FP32 gate分支：squares `[32,128]` FP32，16KiB | Mul→ReduceSum；随后归一化下一tile复用 |
| [0x8000,0xC000) | [0x24000,0x28000) | BF16 gate分支：原gate `[64,128]` BF16，16KiB | MTE2→gate Cast；此分支norm逐行用共享scratch，不同时放norm32 values |
| [0x10000,0x10800) | [0x2C000,0x2C800) | sequence-major beta padded `[64,32B]`，2KiB | MTE2→Gather；head-major不激活 |
| [0x10800,0x10A00) | [0x2C800,0x2CA00) | beta `[64]` BF16/FP32，有效128/256B，预留512B | DMA/Gather→beta变换；V1覆盖前已退出 |
| [0x10A00,0x10B00) | [0x2CA00,0x2CB00) | betaEff `[64]` FP32，256B | beta变换→V0内部及公开写回；V3/V6从GM重读 |
| [0x10C00,0x10E00) | [0x2CC00,0x2CE00) | dt_bias `[128]` FP32，512B | 按需MTE2→gate VF；V1覆盖前退出 |
| [0x10E00,0x10F00) | [0x2CE00,0x2CF00) | a_log `[1]` FP32，有效4B，预留256B | 按需MTE2→gate VF；DMA按32B落点 |
| [0x10F00,0x11000) | [0x2CF00,0x2D000) | qRstd `[64]` FP32，256B | norm→请求的公开检查点写回 |
| [0x11000,0x11100) | [0x2D000,0x2D100) | kRstd `[64]` FP32，256B | 同上；MTE3最后读取后可覆盖 |
| [0x10B00,0x10C00)、[0x11100,0x12000) | [0x2CB00,0x2CC00)、[0x2D100,0x2E000) | 当前Generic V0无独立对象 | 私有槽保留空洞，不作为跨pair无条件可借区域 |
| [0x12000,0x1A000) | 同一共享区 | FP32原gate→G `[64,128]`，32KiB；BF16 gate也转入此区 | MTE2/Cast→gate/cumsum→V0写gk→V1读G；尚未退出时不能借给另一pair |
| [0x1A000,0x1C000) | 同一共享区 | 下列分时scratch及Gather表，共8KiB | sharedFree持有期内使用 |
| [0x2E000,0x30000) | 同一尾区 | V0无数据；head-major已分配、sequence-major未分配 | V1才使用；不是额外32行FP32 tensor |

norm32的scratch细分为repeatScalars `[0x1A000,0x1A400)` FP32 `[32,8]` 1KiB、
blockScalars `[0x1A400,0x1A800)` 同容量、ones `[0x1A800,0x1A880)` FP32 `[32]` 128B。
reduceTmp传入对齐地址0x1B000；已核对的AR/reuseSource实现不读写外部临时区，
不能据此为其它归约实现推导零scratch。gate VF随后复用共享scratch低7KiB，
`[0x1BC00,0x1BF00)`未供这两个FP32临时块使用，sequence-major
`[0x1BF00,0x1C000)`为64个uint32 Gather offset，256B，Init生产、所有V0 Gather最后读取。
FP32原gate在norm32期间仍活在共享G，values/squares不能挪到G中。
BF16 gate原输入未退出前，同址private区也不能作为norm32临时区。

### 2.3 Stage V1：Vector，指数因子和score输入

普通路径每16行取一个锚点r：`Q+=Qhat*E(G-G[r])`、`K+=Khat*E(G-G[r])`，
`Kminus[s]=Khat[:p_s]*E(G[r]-G[:p_s])`，p_s=16/32/48/64。
E按域换成自然底Exp，V1裁剪使用等价base2范围[-126,120]。
Q/K在原private地址覆盖，Kminus占[P+0x8000,P+0x12000)。
普通payload为16KiB Q+、16KiB K+、40KiB Kminus，共72KiB。
coarse按32行的两个锚点生成32/64行prefix，共56KiB payload；C2仍有四个16行band。
这与L³/L⁷求逆无关，也没有集成侧线“两个锚点指数比值复用”补丁。
head-major tile16、sequence-major tile7；共享scratch和尾部stage分时使用。
计算最后读取G后发布sharedFree，MTE3写完payload后发布ioFree；跨核ready由Process发布。

V1地址图保留两个private槽的独立所有权；下面的区间会覆盖已经写回完成的V0状态。

| pair0绝对区间 | pair1绝对区间 | 对象/容量 | 生命周期 |
| --- | --- | --- | --- |
| [0,0x4000)、[0x4000,0x8000) | [0x1C000,0x20000)、[0x20000,0x24000) | Qhat/Khat→Q+/K+，各BF16 `[64,128]` 16KiB | V0保留输入；先完成所有Kminus读取，再原址Cast；MTE3 payload末次读后退出 |
| [0x8000,0x9000)、[0x9000,0xB000)、[0xB000,0xE000)、[0xE000,0x12000) | [0x24000,0x25000)、[0x25000,0x27000)、[0x27000,0x2A000)、[0x2A000,0x2E000) | 普通Kminus `[16/32/48/64,128]` BF16，4/8/12/16KiB | V1生产→payload MTE3；合计40KiB，四份前缀不是同一份in-place矩阵 |
| [0x8000,0xA000)、[0xA000,0xE000) | [0x24000,0x26000)、[0x26000,0x2A000) | coarse替代布局：Kminus `[32/64,128]` BF16，8/16KiB | 仅coarse启用；不能同时套用上一行四前缀布局 |
| [0xE000,0xE400) | [0x2A000,0x2A400) | coarse references `[2,128]` FP32，1KiB | 在G被局部换义前保存→两锚点因子末次读取；不随56KiB payload写GM |
| [0x12000,0x1A000) | 同一共享区 | G `[64,128]` FP32，32KiB | V0保持原址→最后因子读取；coarse可换义已消费tile，参考行先保存 |
| [0x1A000,0x1C000) | 同一共享区 | head-major factor `[16,128]` FP32，8KiB | clamp/Exp→乘法；无有效beta Gather表 |
| [0x2E000,0x30000) | 同一尾区 | head-major stage `[16,128]` FP32，8KiB | Cast→Mul→Cast；一个共享tile，非每private一份 |

sequence-major替代最后两行为factor `[0x1A000,0x1AE00)`、stage
`[0x1AE00,0x1BC00)`，均FP32 `[7,128]`、3584B；保留
`[0x1BF00,0x1C000)` Gather表及中间768B空洞。尾8KiB未分配。
coarse的剩余private `[0xE400,0x12000)`或`[0x2A400,0x2E000)`本Stage无数据，
仍在private所有权内；并非可同时容纳另一完整head的新增槽。

### 2.4 Stage C2：Cube，分带原始score

按localHead把payload ND→NZ写入对应72KiB L1 lane。每band：
`[rawQ_s;rawK_s] = [Q+_s;K+_s] @ Kminus_s^T`，操作数BF16、FP32累加。
完整band的M=32、K=128，N依次16/32/48/64；满chunk4次MMAD提交，对应8个数学乘积。
L1→L0使用LoadData，M_MTE1保护L0AB，FIX_M保护L0C复用。
Fixpipe NoQuant写GM compact raw ND，满chunk为20KiB。
最后band只有有效行时，分开写上下两半有效行，不能写公开张量尾部外。
C2完成后归还该pair，使V3可加载raw；C4改写head lane前有MTE2 barrier。

### 2.5 Stage V3：Vector，mask和完整叶子逆

从GM读compact raw和betaEff。Aqk只施加因果mask及scale；
`L=strict_lower(rawK*betaEff_row)`，`A=(I+L)^-1`。
切成32阶叶子，`X0=(I+L00)^-1`、`X1=(I+L11)^-1`、B=L10。
Generic使用完整右向消元，两个独立叶子交错，系数Gather为8 lane，
保留Newton倒数修正；没有L³截断、raw reciprocal或64-lane合并Sub侧线改动。
非有效行/上三角先置安全值，避免先做高风险计算再依赖最终mask消错。

private换义：Aqk[P,P+0x2000)、Lkk[P+0x2000,P+0x6000)、
叶子/系数区[P+0x6000,P+0x8000)、B/X0/X1/negX1依次位于
P+0x8000/0x9000/0xA000/0xB000，各4KiB；compact raw[P+0xD000,P+0x12000)。
betaEff借用已退休共享G前256B；band广播scratch位于[0x1A000,0x1B300)。
BF16 Akk pack从P+0xE000开始，只在raw末次消费后建立；GM store源读完后，
raw区[P+0xD000,P+0x11000)再换义为FP32 Akk。MTE3_V保护这些覆盖。
写公开Aqk、内部完整零填充64×64 Akk、B/X0/negX1及FP32 Akk；
关闭公开Akk仍保留内部scratch，不能借公开输出地址作中转。

V3地址图按“解包→求逆→打包→写回后重建FP32 Akk”分相位，避免将别名相加或并发使用：

| pair0绝对区间 | pair1绝对区间 | 对象/shape/dtype/大小 | 首次生产→最后消费和换义条件 |
| --- | --- | --- | --- |
| [0,0x2000) | [0x1C000,0x1E000) | Aqk `[64,64]` BF16，8KiB | raw解包/scale→公开MTE3写回 |
| [0x2000,0x6000) | [0x1E000,0x22000) | Lkk `[64,64]` FP32，16KiB | rawK×beta解包→两个叶子所有Gather及B提取 |
| [0x6000,0x6400) | [0x22000,0x22400) | offsets `[32,8]` uint32，1KiB有效容量 | CreateVecIndex等→最后Gather；之后才可转BF16 Cast缓冲 |
| [0x6000,0x6800)、[0x7000,0x7800) | [0x22000,0x22800)、[0x23000,0x23800) | cast0/cast1 `[32,32]` BF16，各2KiB | 求逆完Cast→二维DataCopy进Akk pack；与offsets分时 |
| [0x8000,0x9000) | [0x24000,0x25000) | B `[32,32]` FP32，4KiB | Lkk左下提取→payload MTE3 |
| [0x9000,0xA000) | [0x25000,0x26000) | X0 `[32,32]` FP32，4KiB | identity/精确消元→pack、payload、FP32 Akk构造 |
| [0xA000,0xB000) | [0x26000,0x27000) | X1，同上4KiB | identity/精确消元→negX1、pack、FP32 Akk构造 |
| [0xB000,0xC000) | [0x27000,0x28000) | negX1 `[32,32]` FP32，4KiB | Muls(-1)→payload MTE3 |
| [0xC000,0xD000) | [0x28000,0x29000) | 无数据，4KiB空洞 | 当前V3不借用；仍受private许可约束 |
| [0xD000,0x12000) | [0x29000,0x2E000) | compact raw最多5120 FP32，20KiB | MTE2→Aqk/Lkk解包；全部解包后才可作求逆scratch |
| [0xD000,0xD400)、[0xE000,0xE400) | [0x29000,0x29400)、[0x2A000,0x2A400) | coeffA/B `[32,8]` FP32有效上界，各1KiB | Gather→对应Mul；raw已退出，下一轮Gather可覆盖 |
| [0xF000,0x10000)、[0x10000,0x11000) | [0x2B000,0x2C000)、[0x2C000,0x2D000) | prodA/B `[32,32]` FP32，各4KiB | Mul→独立Sub；两叶未采用64lane合并Sub |
| [0xE000,0x10000) | [0x2A000,0x2C000) | Akk pack `[64,64]` BF16，8KiB | coeff/prod最后消费后清零和Cast拼装→内部及可选公开MTE3 |
| [0xD000,0x11000) | [0x29000,0x2D000) | Akk FP32 `[64,64]`，16KiB | 等MTE3读完重叠pack区后建立→独立FP32 GM relay |
| [0x12000,0x12100) | 同一共享区 | betaEff `[64]` FP32，256B | MTE2→band广播；旧G已退出 |
| [0x12100,0x1A000) | 同一共享区 | 当前V3无数据，余31.75KiB | 持有sharedFree，不等于另一pair可提前装入G |
| [0x1A000,0x1B000)、[0x1B000,0x1B200)、[0x1B200,0x1B300) | 同一共享区 | betaRepeat预留4KiB、betaBrcb512B、betaPadded256B，均FP32 | band处理生产/消费；与offsets、beta输入物理分离 |
| [0x1B300,0x1BF00)、[0x1BF00,0x1C000) | 同一共享区 | 3KiB无数据；后256B为保留Gather表 | 不破坏下一次sequence-major V0的offsets |
| [0x2E000,0x30000) | 同一尾区 | V3无数据；head-major已分配/sequence-major未分配 | 非第三个private槽 |

offsets与cast0的别名、raw与coeff/prod/pack/FP32 Akk的别名都有明确相位边界；
尤其“Vector已结束pack计算”不等于“MTE3已读完pack”，后者必须经MTE3_V再换义。
当前pair的最高实际触及地址为0x12000或0x2E000；这不是将两pair和所有相位
占用量相加的峰值。另一private未取回ioFree时继续保留其上一Stage的数据。

### 2.6 Stage C4：Cube，跨叶子第一次乘法

先ND2NZ加载当前RelayT的Akk到本head lane+0x4000。
M≤32时无跨叶子乘法；仍完成输入搬运依赖并按原协议归还pair。
M>32：B/X0/negX1均FP32，加载到上表四份独立L1位置，
`T[32,32]=B[32,32] @ X0[32,32]`，FP32输入/累加，HF32关闭。
FP32 L0A用16×8分形LoadData2D，L0B使用LoadData3Dv2及其转置配置。
T通过NoQuant、channelSplit的FP32 NZ Fixpipe写4KiB GM，然后原布局读回L1。
FIX_MTE2→MTE2_MTE1 tReady[h]保证C5消费；不是FP32 L0C直接写L1。
收齐所有head的C4后再执行C5，独立head可提供排队空间，但不等同于已隐藏全部relay成本。

### 2.7 Stage C5：Cube，完成Akk

`A10[32,32]=negX1[32,32] @ T[32,32]`，negX1=-X1；FP32输入/累加，HF32关闭。
完整A由X0/X1和左下块组成，上三角保持零。仅M>32执行。
BF16分支将新左下块经`F322BF16`、NZ Fixpipe直写本head的Akk L1，
FIX_MTE1通知C7；对角块已由C4加载。fallback将左下块写FP32 GM Akk并ND2NZ重载，
MTE2_MTE1通知C7。公开Akk只写有效bottomRows，且只在请求时写出。
BF16舍入发生在求逆块完成之后；C4的T和两次跨叶子乘法保持FP32。

### 2.8 Stage V6：Vector，后半段RHS

从内部GM重读Qhat/Khat/betaEff、公开gk，再读原V。
计算qg、qg_scaled、kg及K_beta_g/V_beta，指数输入以等价base2[-80,80]裁剪。
qg先BF16舍入，qg_scaled由该舍入值再乘scale；不能把两次舍入合成一次。
原有BF16中间舍入点按V6Vf/V6BatchVf保留；relay dtype变化只在已选择合同内启用。
Qg/Kg/VBeta/KBetaG依次占private的0/0x4000/0x8000/0xC000起16KiB。
G正向读取后才将共享低16KiB换义为qg_scaled；betaEff保留于private状态区。

普通FP32 relay用16行head-major或7行sequence-major tile。
BF16 relay用tile32：factor借另一pair已退休private前16KiB，stage借其后16KiB。
先获取other ioFree，再MTE2_V把许可传给Vector；VF末次使用后才发布sharedFree。
另一pair的ioFree不能替代共享区依赖。KBeta写出后可发射独立factor计算，
覆盖其打包源之前仍等待MTE3_V。整head各写一次两个独立GM RHS lane，
BF16共32KiB、FP32共64KiB；公开qg只在Save写，其他必选输出照常写。

V6地址图首先列仍活跃的整head数据，随后列两条实现的临时区；所有矩阵为ND。

| pair0绝对区间 | pair1绝对区间 | 对象/shape/dtype/大小 | 首次生产→最后消费 |
| --- | --- | --- | --- |
| [0,0x4000)、[0x4000,0x8000) | [0x1C000,0x20000)、[0x20000,0x24000) | Qhat→qg、Khat→kg，各`[64,128]` BF16，16KiB | context MTE2→VF原址变换→请求的qg及必选kg写回 |
| [0x8000,0xC000) | [0x24000,0x28000) | V→VBeta `[64,128]` BF16，16KiB | 原V MTE2→beta乘法；BF16分支保持至整head RHS写回，FP32分支分tile转存 |
| [0xC000,0x10000) | [0x28000,0x2C000) | 仅BF16 relay：KBetaG `[64,128]` BF16，16KiB | VF逐tile生产→整head RHS MTE3；最后读取前不能借作stage |
| [0x10A00,0x10B00) | [0x2CA00,0x2CB00) | betaEff `[64]` FP32，256B | context MTE2→所有tile beta广播 |
| [0x12000,0x1A000) | 同一共享区 | G `[64,128]` FP32，32KiB | gk MTE2→因子与last-row差；逐行退出 |
| [0x12000,0x16000) | 同一共享区 | qg_scaled `[64,128]` BF16，16KiB，与G低半别名 | 当前FP32 G行末次读取后按正序压缩写；最后MTE3_V后sharedFree |

BF16 relay/head-major的两份**临时tile**借用otherPair：

| 当前pair | factor地址 | stage地址 | shape/dtype/大小及复用 |
| --- | --- | --- | --- |
| 0 | [0x1C000,0x20000) | [0x20000,0x24000) | 各`[32,128]` FP32、16KiB；覆盖已退休pair1 Q/K |
| 1 | [0,0x4000) | [0x4000,0x8000) | 同上；覆盖已退休pair0 Q/K |

每次先获取other ioFree，再经MTE2_V桥接给Vector；重发ioFree本身不允许下一次MTE2
立即覆盖，因为下一StageV0还必须获取sharedFree，直到借用VF退出才归还它。
当前pair的betaBrcb为`[0x10000,0x10800)`或`[0x2C000,0x2C800)`，
FP32 `[64,8]`、2KiB；betaRepeat为`[0x10800,0x10A00)`或`[0x2C800,0x2CA00)`，
FP32 `[16,8]`为前512B；wide tile32实际写`[32,8]`共1KiB，地址扩展至
0x10C00或0x2CC00。实际MulTile读取当前rows对应的8lane scalar block。
此时CopyRowScalarBlocks的目标会继续覆盖已消费的betaEff及相邻空洞；
beta原始值已预先Brcb到独立2KiB区，后续不再从betaEff读取，不能把二者写成同时存活。
此路径尾8KiB与共享scratch没有额外整tile对象；不为借用虚构第三private。

FP32 relay/head-major的临时区完全不同：factor在`[0x1A000,0x1C000)`，
FP32 `[16,128]` 8KiB；stage在`[0xD800,0xF800)`或`[0x29800,0x2B800)`，
同为8KiB，位于已取消的BF16 KBetaG输出区内。betaBrcb在`[0xC000,0xC800)`
或`[0x28000,0x28800)`，2KiB；betaRepeat从0xC800或0x28800起按tile读取。
FP32 RHS逐tile由stage写GM，下一次Cast覆盖stage前必须等待MTE3_V；
不能套用BF16整head写回或other-private借用的说明。
sequence-major FP32分支用tile7，factor/stage仍是V1的两块3584B共享scratch，
Gather表保持在`[0x1BF00,0x1C000)`；BF16 relay当前不会路由到此布局。

静态容量核算：两private `2*72KiB`＋共享G `32KiB`＋scratch `8KiB`＝184KiB；
head-major另分配尾stage `8KiB`＝192KiB，恰至0x30000。V0/V3不使用尾stage，
V1使用它，BF16 V6改借已退休other-private。这里分别说明了**分配上界、实际对象、
相位别名与空洞**；不是声称192KiB在每条指令都同时存活，也不把已退休区算成额外容量。

### 2.9 Stage C7：Cube，W/U

每pair先等rhsLaneFree，两个head RHS各自ND2NZ加载至专用两槽。
RHS离开workspace后即发free，使AIV可准备下一代，不要求W/U已写完。
`W[M,128]=A[M,m] @ K_beta_g[m,128]`、`U[M,128]=A[M,m] @ V_beta[m,128]`，
m为32或64，A/RHS均为所选RelayT；累加FP32、Fixpipe输出BF16。
最后LoadData读取RHS后从MTE1发布rhsLaneFree；pair1同时发布akkLaneFree，
下一组C2 pair1改写旧Akk lane前必须等待它。
workspace free、L1 free和L0 free是三个独立生命周期，不能互相替代。

### 2.10 搬运汇总与同步闭环

下表以满chunk、单value head为单位，未计按模式可选的公开检查点。
实际尾块只传有效行；固定零填充矩阵按Stage说明传满块。

| 数据 | GM写/读 | 位置、批量方式与原因 |
| --- | --- | --- |
| 原Q/K | 各读16KiB一次 | V0→UB；GVA每hv仍重读 |
| 原gate/beta/V | gate读16/32KiB、beta读128/256B、V读16KiB | sequence-major用stride DMA，不先全量转置 |
| Qhat/Khat/betaEff内部context | 写32KiB+256B；V6读相同量，V3另读256B | V1覆盖private后仍需复用，走GM |
| gk | 写32KiB、V6读32KiB | 公开输出也是已定义的后续输入，不能关闭 |
| score payload | 写/读各72KiB或coarse56KiB | V1→C2，ND2NZ；非两head批量ND2NZ实现 |
| raw score | 写/读各20KiB | C2→V3，FP32 ND |
| B/X0/negX1 | 写/读各12KiB | V3→C4，FP32 ND2NZ |
| BF16/FP32对角Akk | V3各写8/16KiB，C4选择一路读 | BF16包和FP32内部矩阵均实际生成，不能声称另一份写回已删除 |
| T | 写/读各4KiB | C4 FP32 NZ经GM中转 |
| C5结果 | BF16内部无GM；fallback写左下4KiB并重读完整16KiB | 两分支不混算节省 |
| RHS | 写/读各32KiB或64KiB | V6→C7专用record，BF16整head写回、FP32逐tile写回 |
| W/U | 各写16KiB | C7 Fixpipe，尾行受限 |

Generic保留代码默认缓存及已有C4 DCCI/DDR操作，没有整体移植A5的流式cache hint。
不能从静态字节数表推断实际带宽、cache命中或重叠收益。

核内HardEvent各方向有独立ID池，用户申请数如下；不是把所有方向合成一个ID池：

| 核 | event方向 | 用户数量 | 初始许可/保护 |
| --- | --- | ---: | --- |
| 每AIV | V_S、S_V | 各1 | 无；Scalar读写与Vector交接 |
| 每AIV | V_MTE2 sharedFree | 1 | 初始1；共享G/scratch |
| 每AIV | MTE3_MTE2 ioFree | 2 | 各初始1；private输入覆盖 |
| 每AIV | MTE2_V、V_MTE3、MTE3_V | 各2 | 无；每次Stage显式set/wait |
| AIC | MTE2_MTE1 | 5 | 通用1+tReady4；无初始许可 |
| AIC | MTE1_M、M_FIX | 各1 | 无；LoadData/MMAD/Fixpipe |
| AIC | M_MTE1、FIX_M | 各1 | 各初始1；L0复用 |
| AIC | FIX_MTE2、FIX_MTE1 | 各4 | 无；每head T/Akk就绪 |
| AIC | MTE1_MTE2 | 2 | 各初始1；RHS和旧Akk lane保护 |

最大用户单方向为5；另有TPipe在M_MTE1池占用0/1/2，普通用户为3。
按已验证A2/CANN的每方向8个ID检查容量，不把注释预期ID当所有SDK的固定ABI。
核间mode0x2固定ready=0/1、free=2/3；一轮ready汇合两个AIV。

```text
init: AIC发布free[0/1]；每AIV发布sharedFree和ioFree[0/1]
for 每个本核work、每个最多4-head group:
  AIV pair0/1: wait free → 有效head V0,V1 → set ready（无效head也set）
  AIC pair0/1: wait ready → 有效head C2 → set free
  AIV pair0/1: wait free → 有效head V3 → set ready
  AIC pair0/1: wait ready → C4搬入本pair全部输入 → set free
  AIC head0..3: C5（M≤32跳过，不制造额外tReady等待）
  AIV pair0/1: wait free → 有效head V6 → set ready
  AIC pair0/1: wait ready → C7读完GM RHS即set free → W/U
  下一group/work只有取得对应许可才能覆盖相应区域
drain:
  AIV消费C7最后free、每pair ioFree及sharedFree，再ReleaseEventID
  AIC消费最终M_MTE1、FIX_M、rhsLaneFree、akkLaneFree，再释放全部事件
empty-core: 两核一致跳过跨核循环；只消费自身初始许可并释放
```

这是四record按head重复使用，不是跨work双bank轮换；早释放只证明允许重叠，
实际重叠比例必须由msprof op simulator的pipe时间线及PC映射确认。

### 2.11 Target variant：保留的独立算法与资源

仅host `Arch22TargetConfig::Matches` 的定长B1/H96/T8192、128满chunk、
head-major、beta BF16、L2、Sigmoid beta、预计算gate、exp2、Save合同命中。
Target仍为最多4-head组和相同AIV 0/2、1/3映射；不是32模型的通用算法。
host另选最多4份head partition；20 AIC时2份×48head、256work，
24 AIC时3份×32head、384work，不能挪用Generic任务数公式的选择条件。

Target在上述阶段之间增加Cube norm-sum lookahead和C3叶子多项式，
一轮到L³，C4/C5保持FP32输入/累加，C7为BF16输入、FP32累加。
Generic完整消元不继承这项近似。
每AIV UB192KiB；L1仅[0,0x5C000)共368KiB，末尾144KiB未分配。
L1四head lane及X0/negX1/T起点与Generic相同，但Akk在[0x54000,0x5C000)，
不能把Generic RHS区域表套在此分支。
record S=0x32400=201KiB，四份；保留公共context/payload/raw区域，
两个BF16 RHS位于[0x1F400,0x27400)，norm-sum[0x27400,0x2B400)，
叶子辅助[0x2B400,0x32400)。均由target_common和target_config共同约束。

C3有独立两个L0逻辑槽，但物理池与普通MMAD共享。进入C3前消费普通M_MTE1/FIX_M，
最后使用后从M/FIX重新归还普通许可，不能声称两套池物理互不相交。
on-demand MTE1_MTE2 l1ReadersDone在C2/norm-lookahead/C3覆盖head lane前，
等待先前LoadData读完；它无初始许可，调用处set/wait闭环。
V0最后gk Muls之后才发V_MTE3，V3 identity复制源读完后才覆盖X0。
Target另有两个restReady和两槽C3事件；以InitTarget/ReleasePairEvents的逐方向分配为准，
不复用Generic事件ID或继承其工具结论。

Target UB也为每AIV独立的两个72KiB private、32KiB G、8KiB scratch、8KiB尾stage，
分配上界192KiB。Q/K、beta/dt/a_log/rstd、Aqk/Lkk/B/X0/X1/identity和G的绝对地址
与前面对应张量一致，但**生命周期及以下别名不同**；不能继承Generic norm32或V6借用图：

| Stage与对象 | pair0绝对区间 | pair1绝对区间/共享地址 | shape、大小、生产与最后消费 |
| --- | --- | --- | --- |
| V0 FP32 gate时normSum Q/K | [0x8000,0xA000)、[0xA000,0xC000) | [0x24000,0x26000)、[0x26000,0x28000) | 各FP32 `[2,32,32]`，8KiB；AIC Gram经GM→MTE2→读取对角rstd；gate仍在共享G |
| V0 BF16 gate时normSum Q/K | [0x12000,0x14000)、[0x14000,0x16000) | 同一共享区 | 同上；BF16 gate仍活在当前private `[0x8000,0xC000)`或`[0x24000,0x28000)`；提取rstd后才覆盖为FP32 G |
| V0 Gather offsets，仅sequence-major辅助代码 | [0x11300,0x11400) | [0x2D300,0x2D400) | `[64]` uint32，256B；每次V0 Scalar生产→Gather，V1前退出；当前Target路由不命中该布局 |
| V1 bandDelta | [0x11400,0x11A00) | [0x2D400,0x2DA00) | `[3,128]` FP32，1.5KiB；参考行差值Exp→Kminus因子递推；与普通Kminus末段分时使用 |
| V1/V6 factor | [0x1A000,0x1C000) | 同一共享区 | `[16,128]` FP32，8KiB；VF生产和消费；没有Generic sequence-major常驻Gather表 |
| V1/V6 stage | [0x2E000,0x30000) | 同一尾区 | `[16,128]` FP32，8KiB；Cast/Mul/Cast临时；不借另一private |
| V3 X0/X1 | [0x9000,0xA000)、[0xA000,0xB000) | [0x25000,0x26000)、[0x26000,0x27000) | FP32 `[32,32]`各4KiB；先建I，再原位I−L，写invAux供C3；不是Generic完整消元后叶子逆 |
| V3 identity | [0xB000,0xC000) | [0x27000,0x28000) | FP32 `[32,32]`，4KiB；复制X0的I→invAux写回，复制完成后才允许X0做Sub |
| V3 Akk零填充 | [0xE000,0x10000) | [0x2A000,0x2C000) | BF16 `[64,64]`，8KiB；raw解包后Duplicate→公开/内部初始化；C3/C5后续补最终块 |
| V6 KBetaG | [0xC000,0x10000) | [0x28000,0x2C000) | BF16 `[64,128]`，16KiB；VF生产→独立RHS GM；不存在Generic FP32 relay的stage换义 |

其余private空洞、两个完整Q/K或qg/kg张量和共享G均按当前Stage重新列明的生命期使用；
Target的normSum不能与尚未读取的同址gate共存。V6通过inputReady/restReady把
前32行Q/K与后续输入分开通知，G仍一次装入32KiB，beta/V及后32行按restReady消费。
这不是两份完整UB ping/pong；只有尾stage一份，最高已分配地址仍为0x30000。

### 2.12 设计与实现的核对入口

- 任务/边界：`chunk_kda_fwd_prepare_utils.h`、host `tiling_processor.h`。
- 资源/地址：`chunk_kda_fwd_prepare_policy.h`、`arch22/*target_common.h`。
- Generic时序：`arch22/*vec.h::Process`、`arch22/*cube.h::Process/StageC7`。
- 分支：`chunk_kda_fwd_prepare_kernel.h::RunArch22Selected`。
- Target：`arch22/*vec_target.h`、`arch22/*cube_target.h`、`*target_config.h`。
- 路由、coarse条件、六key和真实CANN序列化已通过独立开发期合同测试；
  新增测试与原始证据保留在独立验证归档，不随本优化PR提交。
  上游已有`op_host/tests`保持原样；ATK交付件维护于work_tool。

V2映射/分核整改、跨GVA Q/K复用、Generic双bank/双L0、mask设置复用、
侧线锚点因子复用和packed-leaf均不属于本冻结版本。记录这些差异用于后续设计评审，
本次不据此修改kernel，也不把它们计入已有收益。
