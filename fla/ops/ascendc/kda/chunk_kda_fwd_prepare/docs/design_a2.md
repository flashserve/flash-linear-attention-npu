# ChunkKdaFwdPrepare 设计

方案设计规则版本：V2。公开合同与 13 槽输出掩码见 [api.md](api.md)。

## 1. 目标

融合 Q/K norm、gate prefix、因果 score、完整块三角逆及 W/U，chunk 间不传播状态。
FwdH 和 Finalize 的递推与 attention 输出不属于本算子。A2 只有统一 BF16 resident relay；
必要的归一化、gate、prefix、score 后处理和 Cube 累加保持 FP32。
BF16 输入/relay 不代表全程 FP32 求逆，精度以冻结的 FP64 truth / FP32-BF16 control 比较为准。

使用用户确认的 value-head pair：一个 AIC 的两个 AIV 各负责一个完整 value head，
不改成横切同一 head。此映射沿用已验证的资源、协议和性能方案，是 R01/R20 的本算子既有映射。
开发参考为 CHUNK_INDEPENDENT V1，来源 a78efa9512dfa64c20b2a58a75e84e88a054d694；
采用 grid-stride、resident 生命周期及 ready/free 方法，具体 lane 映射和地址由本算子定义。

性能固定使用实际 32 模型输入，seed3、safe gate、gate-in-kernel、Save，
对照任务启动时 PR994 的源码安装与相同实际输入的 A5，目标逐例 A5/A2 ≥ 0.5。
warmup5、repeat50、3 groups，A2 ABBA；Task Duration 决定性能，Event 独立列出。
已验收版本、逐模型结果和实验取舍收录在独立测试交付件，不能由窄合同 fast 的结果替代。
此处不把阶段候选或尚未完成的最终测试标为通过。

## 2. Stage 完整详设

### 2.0 依赖图、实现来源与 Stage 合同

文档组织参考 [PR833](https://github.com/flashserve/flash-linear-attention-npu/pull/833)
所引用 CANNBot PR1274 的 linear-attention stage-design 工作流，参考版本
`8a2d4d34e0deae6c249495e9c5c425dfc70c5eef`。这里记录实际 AscendC 原语、地址和事件；
没有将本实现描述为尚未接入的 CATLASS BlockMmad，也不声称运行过缺席的工作流工具。
A5 设计保留在原 `design.md`，本文只描述 A2 实现。

```text
q/k --norm--> qHat/kHat --四band锚点--> score payload --Cube--> Raw
g --gate/prefix--> G -------------+                         |
beta --activation--> betaEff ----|-------------------------+
                               +--> 当前RHS/公开qg/kg      |
                                                          v
                                                    Lower/Aqk
                                                          |
                             Identity/Precomputed: 局部FP32求逆（Precomputed为64阶，其余为两份32阶） ---+
                             L2非Precomputed: BF16叶内16阶Cube计算 -----+--> BF16驻留合并逆
                                                               |
                                        BF16 RHS --------------+--> W/U
```

|Stage|实际实现函数/原语|输入→输出|发布与退休点|
|---|---|---|---|
|0|Vector::Prepare，DataCopyPad/Gather/归约/非线性|外部布局→UB Q/K/G/beta|有效行DMA完成；checkpoint源保持活跃|
|1|ScorePayload，FP32 Exp/乘法及BF16 Cast|Q/K/G→GM Payload|score-ready覆盖其MTE3|
|2|Cube::Scores，ND2NZ/LoadData/Mmad/Fixpipe|Payload→FP32 Raw|raw-ready；scoreReleased释放L1读取租约|
|3|RhsAndOutputs|G/K/V/beta→BF16 RHS及公开输出|rhs-ready；DMA退休后复用UB|
|4|LowerAndAqk；Identity/Precomputed另含LeafInverse32/64|Raw→Lower/Aqk及可选BF16叶逆|lower-ready覆盖全部相关store|
|5|InversePair/PackedLeavesPair/MergePair|BF16 Lower/叶逆→resident X|每次FIX_MTE1后才读X；末次读取后复用|
|6|PreloadWu/Wu|共享resident X、独立RHS→W/U|MTE1释放L1；FIX结束释放GM slot|

`reference_source`：A5 Prepare 的 `arch35/chunk_kda_fwd_prepare_vec.h` 中
StageV0 及 V 输入阶段的 DataCopyPad 直接按 dense BSND 行 stride 读取 q/k/v/g，完整 fwd 的 L2
另行适配 beta；TND 分支有独立转换，不能从 dense 路径推断它也没有转换。
最新 main 的 A2 KDA bwd_finalize 通过公开 `output_layout` 选择输出布局；
`arch22/chunk_kda_bwd_finalize_vector.h` 中 `outputTokenMajor` 分支直接使用
带目标 stride 的 DataCopyPad 写出 BSND/TND，输入仍是连续 Head First。
这里借鉴的是按消费 tile 读入、按公开布局直接写出的方法，不能混淆输入与输出 layout。
Prepare 的 q/k/v/g/beta 均按原公开 layout 读取，进入相同连续 UB tile；使用下文的分块读取，不产生全量 transpose 临时张量。

### 2.1 符号、工作域和选择

M=64，D=128，r∈[1,64] 是有效行数，G=HV/HK，pairs=ceil(HV/2)。
Nbase=fixed 时 B×ceil(T/64)，varlen 时为规范 chunk_indices 的记录数。
Nwork=Nbase×pairs；blockDim=min(AIC_count,Nwork)，job=core+n×blockDim。
每个 job 描述一个 chunk 和一对 value heads，slot=n&1。
lane l 的 hv=2×pair+l，active=(hv<HV)，hk=floor(hv/G)。
Q/K 检查点 owner 仅为 hv%G==0；不同 hk 的两个 AIV 各自准备自己的 Q/K。
当前实现不跨 lane 共享 Q/K prepare；同一 hk 时仍独立计算。
例如模型31 B=1、T=131072、HK=HV=4：Nbase=2048、pairs=2、Nwork=4096。
实际 blockDim 取平台 AIC 数，尾 wave 使用同一 grid-stride 和完整协议。

六个 tiling key 为：

|key|norm|gate|
|---|---|---|
|0|Identity|Precomputed|
|1|L2|Precomputed|
|2|Identity|Softplus|
|3|L2|Softplus|
|4|Identity|SafeSigmoid|
|5|L2|SafeSigmoid|

Norm/Gate 为静态策略。Raw/Sigmoid/TwoSigmoid、输出掩码及实际标量是 immutable tiling。
现有 CANN dtype 注册分别生成 g/beta 四种输入对象，每对象仍只有六个数学 key。

共享 ACLNN/L0 的参数校验和错误处理沿用 main；本优化不重写跨平台诊断。
不按模型编号、精确 shape、epsilon、scale 或 lower_bound 开优化资格。
Exp/Exp2 均按原接口支持；每个 Stage 选择一次指数域辅助函数，不增加数学 key。
A5 保留原模板声明、arch35 实现和字段；下面的资源和协议专指 A2。
六个静态key同时选择叶内计算组织：Identity或Precomputed保留FP32 Lower；key0/1在UB内完成64阶逆，
key2/4在UB内解两个32阶叶块。只在交接时转BF16；key3/5保留四个16阶BF16 Cube叶逆。各策略仍用统一BF16 inverse/RHS relay、
相同工作描述符、地址上界和事件协议。此选择不依赖模型、layout、精确shape或标量常量。
依据为冻结泛化用例中Identity及Precomputed的叶内量化误差和无差别FP32叶内计算的可复现指令开销；
不增加第七个key，也没有FP32 GM/L1 relay实现族。每个可达key必须独立通过冻结比较。

### 2.2 唯一资源账本

地址常量由 `arch22/chunk_kda_fwd_prepare_resource.h` 同时供 host/device 消费。
每 AIV 一个 0x2f800=190KiB UB，硬件上限192KiB，未增加整套 UB 双缓冲：

|半开字节区间|物理用途|活跃与复用条件|
|---|---|---|
|[0,0x8000)|q_：低16KiB Q，另16KiB临时|Q checkpoint、score payload、RHS 完成消费后用于 Lower|
|[0x8000,0x10000)|k_：低16KiB K，另16KiB临时|同上；不能在 MTE3 未排空前改写|
|[0x10000,0x18000)|g_ FP32 64×128|prefix/RHS及gk store退休后，Identity/Precomputed复用前16KiB保存FP32负Lower|
|[0x18000,0x28000)|work_ FP32 64KiB|gate暂存→norm/factor/raw；key2/4在raw后复用前8KiB作两份32×32逆，key0/1使用前16KiB作64×64逆|
|[0x28000,0x2c000)|v_ BF16 16KiB|Kminus→qg_scaled/kg→Aqk，按末次 DMA 消费复用|
|[0x2c000,0x2f000)|small_ 12KiB|rstd、beta、bias、广播量和 scalar DMA 分区|
|[0x2f000,0x2f800)|mask 2KiB|640B strict/inclusive packed triangle，初始化一次，只读|

small_ 内 rstd 为 [0x2c000,0x2c200)，FP32 beta 为 [0x2c200,0x2c300)，
beta DMA [0x2e000,0x2e800)，BF16 pack [0x2e800,0x2e880)。
sequence-major beta的有界tile按2.4读取，不再每标量启动一行DMA。
只读索引：beta Small[2816,2944)即UB[0x2ec00,0x2ee00)，完整定义128个uint32索引；
前64项为实际row stride，额外masked项为零，覆盖BF16 Gather的完整索引读取范围。
Identity/Precomputed叶内Gather使用Small[2944,3008)即UB[0x2ee00,0x2ef00)。
二者与ALog缓存/临时量不重叠，均在现有Small UB内，不增加190KiB分配。
资源头的static_assert同时检查128项beta范围、64项叶内范围与Mask边界。
Identity/Precomputed叶内阶段复用Small[1536,1600)为两份当前行。
Lower store经MTE3_V退休后，UB[UbQ,UbG)整64KiB用于最多31列的系数广播；
work_[2048,4096)为最多1984个int32索引，[4096,6144)为对应FP32系数，
与前8KiB叶逆及g_中FP32 Lower互不重叠。
qr/kr/beta/bias已完成本work末次读取才能覆盖；ALog和两个只读索引始终保留。
FP32 beta 直接保留低位，不经 BF16 中转；FP32 beta_eff 按真实值写回。
BF16 beta每work在DMA之前将原始/打包转换区Small[2048,2592)初始化为零，
不覆盖ALog缓存、只读Gather索引或mask。Lower阶段将v_完整16KiB初始化，
Eye DMA只覆盖其前8KiB，其余转换非活跃区域保持定义值；公开写回仍仅有效行。
持久UB tensor通过同一TBuf分配派生有界TBuffAddr，逐字段初始化并显式落栈，
ShapeInfo置零；地址、容量与账本一致，不依赖未初始化的描述符字节。

每 AIC 的 L1 为192KiB，两 lane 各96KiB。每 lane 相对地址：

|区间|用途|生命周期|
|---|---|---|
|[0,0x12000)|score Q+/K+/分带 K- payload|score MTE1 最后读取完成后释放|
|[0,0x2000)|P，下三角 BF16|score 释放后开始，到最后 merge 读完|
|[0x2000,0x4000)|X，BF16完整逆|eye 初始化→叶子逆→merge→W/U 最后 LoadA|
|[0x4000,0x8000)|Tmp，merge临时及空隙|按连续 GEMM 的 FIX_MTE1 闭环复用|
|[0x8000,0xa000)|Power，BF16叶子幂|key3/5叶子迭代期间活跃；其余key不使用|
|[0xc000,0x10000)|RHS K 16KiB|scoreReleased 后提前载入，到W/U LoadB结束|
|[0x10000,0x14000)|RHS V 16KiB|同上，不能被下一 work score 覆盖|
|[0x14000,0x16000)|I，BF16 identity|key3/5每lane首次载入，跨work常驻；其余key不使用|

score 阶段与逆/RHS阶段的地址重叠有 scoreReleased 事件，表中同时活跃的区域不重叠。
L0A/L0B/L0C 各64KiB；A/B 两 lane 各有32KiB独立 bank，C通过 FIX_M 保护。
W/U 共用一次逆的 LoadA，同时准备两份 RHS 的 LoadB；下一 head bank 装载与当前 MAC 交叠。

GM workspace 每 lane152KiB，record字段如下；每 slot 两lane=304KiB，每core两slot=608KiB：

|相对地址|大小|生产→消费|
|---|---|---|
|0x00000 Payload|72KiB|score payload→AIC；raw-ready后key0/1/2/4复用低8KiB交接BF16叶逆|
|0x12000 Raw|32KiB|AIC FP32 score→AIV Lower/Aqk|
|0x1a000 Lower|8KiB|AIV BF16 strict lower→AIC inverse|
|0x1c000 Eye|8KiB|AIV identity→AIC X/I|
|0x1e000 RhsK|16KiB|AIV当前 RHS→AIC W|
|0x22000 RhsV|16KiB|AIV当前 RHS→AIC U|

lane base=core×0x98000+slot×0x4c000+lane×0x26000。
workspace总量=平台libApiWorkspace+usedCoreNum×0x98000，由host checked multiply/add产生。
关闭的公开输出不充当 scratch。Norm/Gate 六个key使用相同资源上界和事件集合。

### Stage 0：Vector，输入、norm、gate prefix

仅 DMA 有效 r 行，内部矩阵先补零。Identity 保留 Q/K，rstd=1；L2 在 FP32 执行
sum(q²)+epsilon、sqrt、div、mul，再 CAST_RINT 到 BF16 q_hat/k_hat。
仅 owner 写 HK checkpoint；每 HV 独立写 beta_eff。Raw beta 跳过非线性；Sigmoid 为
1/(1+exp(-beta))，TwoSigmoid 在结果上乘2，不修改用户 allow_neg。
g BF16 先扩展到 FP32；FP32 g 直接载入同一 FP32 区。
Precomputed 直接使用 step；Softplus 用 max(x,0)+log1p(exp(-abs(x))) 的补偿实现；
SafeSigmoid 使用 lower_bound/(1+exp(-exp(a_log)×(g+bias)))。
norm和 gate 一直保持各自明确的 FP32 中间量。prefix 按8行段扫描并加carry，仅当前chunk，
不跨序列；Exp2 的域变换沿用当前实现的舍入顺序，不能交换成另一种变换后宣称等价。

Precomputed 且 g=FP32 时，有符号 step 的前缀抵消需要保留加法舍入残差。
同一8行层级使用 TwoSum：g_ 保存高位，work_[0,8192) 保存低位；
work_[8192,11264) 保存三份8×128 FP32临时量，总活跃44KiB，位于既有64KiB work_内。
低位与高位在层级合并后相加，再按原顺序做 Exp2 域缩放；不改变 BF16 relay。
该分支静态限定 GateMode=Precomputed、GateT=FP32，不以 shape、模型或标量常量选路。
其余 gate/dtype 沿用既有扫描。此处只复用 norm 结束后的 work_，在 factor/RHS 复用前
完成全部 Vector 读写并执行既有 barrier；没有新增 UB、workspace、key 或跨核事件。
gk 在仍活跃时写回，checkpoint DMA 的源在 score payload 消费期间保持不变。

### Stage 1：Vector，四个16行 band 的 score payload

每band选有效区域中点作为 FP32 Gref。Qplus=q_hat×E(G-Gref)，
Kplus=k_hat×E(G-Gref)，Kminus=k_hat×E(Gref-G)，操作数按现有位置转BF16。
第s band的Kminus只保留16(s+1)行，完整payload72KiB。
指数输入按既有base2界[-126,120]限制；padding使用完整64行内部格式，公开输出仅有效行。
保留四band；不混入未通过冻结标准的单锚点或32行band数值实验。
MTE3排空后 q/k/v 的高半区才能复用给当前 RHS，随后发布 score-ready。

### Stage 2：Cube，score

对每个有效lane、每个band分别计算 Qplus @ transpose(Kminus) 和
Kplus @ transpose(Kminus)，形状为[16,128]@[128,16(s+1)]，
BF16 ND/NZ操作数、FP32 MMAD累加，Fixpipe以FP32写Raw。
两lane由同一 AIC 依次消费。最后一次 L1 MTE1读取结束后发布scoreReleased，
Raw实际完成后发布raw-ready，二者职责不同。

### Stage 3：Vector，当前 RHS、公开输出

此Stage在score发布后立即开始，与Stage2交叠，不等待inverse。
FP32中按既有[-80,80]界计算E(G)；kgRHS先将k_hat×E(G)转BF16，再扩展FP32乘beta，
最终BF16舍入到RhsK。RhsV由V在FP32乘beta后转BF16。
qg先BF16舍入，再FP32乘scale并BF16输出qg_scaled；Save在qg活跃时写回。
kg使用最后有效行G[r-1]，不取补零行。两个RHS的内部64行均零填充，写入独立GM区。
最后DMA发布rhs-ready，AIC在scoreReleased后即可提前装入L1。

### Stage 4：Vector，strict Lower、Aqk与局部FP32叶内逆

等待raw-ready及前序work_ Vector消费结束。FP32 score乘scale或-beta，
分别施加inclusive/strict三角mask。Aqk在原位置Cast BF16；完整64×64 BF16负Lower
写入私有GM。640B packed mask一次初始化；padding负Lower=0、inverse对角=1。

Identity或Precomputed key将未Cast的FP32负Lower另存已经退休的g_[0:4096]。
key2/4的LeafInverse32从私有Eye加载完整64行padding模板至v_，两个32阶对角块Cast为FP32，
按64行×32列打包至work_[0:2048]。依次k=0..min(r-1,31)：
`X[i,:] += (-L)[i,k] * X[k,:]`，两叶互相独立；strict mask令i<=k系数为0。
先统一生成各k列索引，一次Gather所有64×steps个系数，再一次Brcb到已退休的Q/K区；
steps<=31、Brcb最多248个repeat，广播区最大62KiB。
第k次仅更新行k+1..31，跳过精确零系数；第二叶仅在r>32时执行对应更新。
非零更新保持原k顺序和FP32 MulAddDst；每步两个PIPE_V屏障，无部分覆盖。
FP32叶内结果一次CAST_RINT为BF16并填回v_对角块。
其余零/单位padding保留，写私有Payload低8KiB；该区最后score消费者已由raw-ready释放。
启用Akk时同时写全部有效行的对角叶块和零区，后续Cube写跨32行下块。

Precomputed（key0/1）使用LeafInverse64：完整Eye在work_[0,4096)展开为FP32，
g_[0,4096)继续保存未量化负Lower；按k=0..r-2执行完整64列行更新。
每批至多16列系数，work_[4096,5120)存1024个Gather索引，work_[5120,6144)存1024个FP32系数。
退休Q的32KiB UB别名保存16×64×8广播系数，Small[1536,1600)保存当前64列逆矩阵行。
每批Gather、广播、每步行复制和MulAddDst之间均有PIPE_V依赖；复用Q前排空MTE3。
padding负Lower为0，完整Eye保证无效行对角为1。完成后一次CAST_RINT到BF16，写Payload及完整有效Akk。
Cube仅将此BF16完整逆载入resident X，不再加载P或执行32+32合并；W/U共用X的路径保持不变。
该策略只由已有Norm/Gate key选择，g/beta dtype、shape、layout与标量不决定优化资格。
没有FP32跨核或GM/L1 relay、额外UB分配或新事件；完整64阶局部计算的性能成本独立测量。

L2非Precomputed（key3/5）不进行上述FP32叶内计算；启用Akk时先清零有效区域，Cube随后写完整叶对角及下块。
两种策略都在本阶段排空Aqk、Lower、叶逆/检查点的全部MTE3，再发布同一lower-ready。
不得提前改写Q/K、g_/work_或checkpoint DMA来源；关闭输出不作为scratch。

### Stage 5：Cube，BF16驻留逆和分层合并

key0/1只加载一次完成的BF16完整逆X；不加载P或执行跨叶合并。
key2/4读取BF16 P与BF16叶逆X到同一L1地址，只有一次32+32合并：
`Tmp=X_lower @ P_cross`，`X_cross=Tmp @ X_upper`。
两次GEMM均FP32累加后Fixpipe舍入为BF16；P已包含负号。
Precomputed的有符号输入可能放大跨叶BF16中间舍入，因此其64阶计算在UB内保留FP32，
只将最终逆量化为BF16 relay；不按精确shape、layout或输入值域选择求逆精度。
X_cross的目标区域不与X_lower输入重叠；Tmp沿用原有地址和FIX_MTE1事件，不增加资源或事件。
key3/5读取BF16 P/Power、Eye到X及首次常驻I；四个16阶叶子通过4轮幂倍增生成完整叶逆，
随后两次16+16及一次32+32合并。

两者所有Cube操作数和resident X均BF16，MMAD累加FP32，Fixpipe再BF16舍入回L1；
没有FP32 inverse/RHS relay。叶内32阶FP32计算也不意味着整个64阶逆全程FP32。
每次Fixpipe写回经FIX_MTE1发布后才读取X/Tmp；临时地址复用等待末次MTE1。
需要公开Akk时在对应叶/跨叶块真实活跃期间写出；完整64列、仅有效r行。
完整阶数不能代替对每处BF16舍入及FP32运算顺序的冻结双标杆验收。

### Stage 6：Cube，复用 inverse 计算 W/U

从同一 resident X 一次LoadA；W=X @ RhsK，U=X @ RhsV，
[m,m]@[m,128]，m=ceil(r/16)×16，BF16操作数、FP32累加、BF16 Fixpipe输出有效r行。
不同lane的A/B bank独立，当前lane MMAD时预载下一lane。
L1 lane free只在最后LoadA/LoadB完成后发布，不能用GM slot free代替。


### 2.3 事件、复用与退出

跨核flag分别为每slot：score-ready=slot，raw-ready=2+slot，
lower-ready=4+slot，work-free=6+slot，rhs-ready=8+slot。
AIC启动时为两个slot各发布work-free credit；两个AIV统一消费同一事件序列：

```text
AIV: wait work-free → Prepare/ScorePayload → publish score-ready
     → current RHS/output → publish rhs-ready
     → wait raw-ready → Lower/Aqk及可选FP32叶内逆 → publish lower-ready → next work
AIC: wait score-ready → score → scoreReleased → publish raw-ready
     → wait rhs-ready → preload RHS → wait lower-ready
     → inverse/merge → W/U → publish work-free → next work
exit AIV: consume both work-free credits
exit AIC: drain C-free, A/B-free and L1-lane-free, then release local events
```

inactive奇数lane不访存、不写公开输出，但完整参与每个跨核等待和通知。
scoreReleased=MTE1_MTE2，L1 laneFree=MTE1_MTE2，A/B free=M_MTE1，
C free=FIX_M，copyReady=MTE2_MTE1，loadReady=MTE1_M，matrixReady=M_FIX，
resident写回=FIX_MTE1。TPipe按各HardEvent类别分配/释放，ID上限由实际编译及simulator检查。
Vector DMA发布使用MTE3，UB覆盖前显式V_MTE2、MTE2_V、MTE3_V等本地依赖。
超过ALog缓存容量的路径在Small[384]的GetValue之后以S_V及S_MTE2退休Scalar读，
再允许Vector或MTE2覆盖该区间；不能仅依赖Vector侧barrier。
work-free保护GM两代record；独立的RHS-ready不会推迟当前RHS到inverse之后。
每core连续多work、slot回卷、最后active/inactive lane和最终credit排空由小shape实际binary捕获验证。
正常性能测量关闭执行记录，不以131k大模型仿真替代小shape生命周期证据。

### 2.4 搬运与数值边界

每HV工作首次读取Q/K/g/beta各一次、V在RHS阶段一次；GVA当前重复读取Q/K，
仅owner写checkpoint。payload、Raw、Lower、RHS按表中生产/消费；key0/1/2/4在score租约结束后另外交接8KiB BF16叶逆，
key0/1/2/4的Eye由Vector每work加载；key3/5的I在L1每lane首次载入。两者完整成本计入性能。
RHS与resident inverse不经公开输出中转，完整Save的所有写出成本属于算子性能。
layout只改变输入地址/strided DMA，内部矩阵统一；varlen由序列起止和规范索引限定。
host拒绝非法head比、chunk/K/V、dtype、标量、metadata及DMA stride溢出；
关闭输出仅返回None，不改变内部必须的矩阵生产。
极端gate/beta、norm epsilon、tail padding及高阶逆的数值路径遵守冻结比较器，
失败合同明确标失败，不新增FP32 relay回退族，也不更改truth/control或容差。

### 2.4.1 四种公开布局的局部搬运

实际A2入口为 `CHUNK_KDA_PREPARE_RUNTIME_KEYS` 下的 `Arch22::Run`。
`QkvInputOffset`、`GateInputOffset` 与 `BetaInputOffset` 均按已有 inputSequenceMajor
计算来源地址；packed 使用 b=0。head-major 的 g 地址为 `((b*HV+head)*T+token)*K`，
sequence-major 为 `((b*T+token)*HV+head)*K`；beta 去掉末尾 K。

`Read` 用 DataCopyPad 只读取有效行。BNSD/NTD 连续读取每个 head 的 Q/K/V/g；
BSND/TND 行间隙为 `(heads-1)*width*sizeof(T)`，到达相同连续 UB tile。
不增加整份 q/k/v/g transpose 或全输入临时张量，不增加布局 key 或 tiling 字段。

head-major beta 在既有 raw scratch 一次连续读取有效 r 个元素，MTE2_V 后转为 FP32 beta；
BF16沿用原Cast，FP32仅拷贝，不做Gather。sequence-major beta 保留八列以内tile：
HV<=8连续读取r×HV个元素，HV>8仅读当前head所属的至多8列，末tile取剩余列并pad到32B。
Gather索引HV<=8为row×HV×sizeof(BetaT)，否则row×32字节；baseAddr为tile内head偏移。
FP32直接Gather，BF16按uint16位模式Gather后Cast，保持原舍入顺序。

仅sequence-major初始化完整128项BetaGather索引，masked上半区为中性零；不修改inverse索引。
初始化经S_V发布，MTE2_V保护DMA→Gather/Cast，PIPE_V保护后续beta激活。
两条读取策略使用相同既有scratch和事件边界，inactive lane完整参与跨核协议。
尾块只读取有效行；不跨head/序列边界，不增加UB容量、key或跨核事件。

### 2.4.2 一次性输入L2 cache策略

raw g、v提示CACHE_MODE_DISABLE；HK=HV时q/k也使用该提示，GVA的q/k保留默认策略。
beta、a_log、dt_bias、全部workspace及输出保持原cache策略。
cache hint仅影响复用策略，不代替ready/release，不改变布局、数学或六key选择。
效果必须由同输入ABBA确认，不能从提示本身推断硬件cache命中率或带宽收益。

### 2.4.3 每个活跃 lane 的逻辑 GM 搬运量

下面按一个有效 value head、一个 chunk 计数，字节数不包括 cache line、总线对齐放大、
L1/L0 内部搬运。令 b=ceil(r/16)、m=16b、s=16×(1+…+b)，g/beta 元素大小为 sg/sb。
读写地址与释放条件分别对应 Stage 0–6 和 2.2/2.3；表中容量不能替代实际有效搬运量。

|对象|GM 生产/消费|每活跃 lane 字节数；r=64 示例|驻留、复用与搬运方法|
|---|---|---|---|
|Q、K 输入|各读一次|各 r×128×2；各16KiB|UB Q/K；BSND/TND 分行 stride DMA，其余连续；GVA 当前仍逐 value head 重读|
|raw g 输入|读一次|r×128×sg；FP32 为32KiB|BNSD/NTD连续、BSND/TND分行DMA到UB，FP32 prefix/RHS末次消费后复用|
|beta 输入|head-major连续；sequence-major矩形tile|head-major r×sb；sequence-major r×c×sb，c≤min(HV,8)|复用UB raw区；只在sequence-major有邻head冗余读取/Gather，计入性能|
|V 输入|读一次|r×128×2；16KiB|RHS 阶段读入，不用公开输出作中转|
|score Payload|AIV 写、AIC 读各一次|512m+256s；各72KiB|Q+/K+ 与四 band K-；scoreReleased 后释放其 L1 区|
|Raw score|AIC 写、AIV 读各一次|128s；各20KiB|32KiB 是分配容量，四 band 只搬实际矩形；raw-ready 后读|
|RHS K/V|AIV 写、AIC 读各一次|各16KiB，合计各32KiB|尾行在 UB 补零后仍搬完整矩阵；AIC 提前载入 L1，W/U 共用 inverse|
|Lower|AIV 写一次|8KiB|所有 key 当前均写完整 BF16 Lower；key0/1 的 AIC 不读该区域；其余 key 见下文|
|key0/1/2/4 叶内逆|AIV 写、AIC 读各一次|各8KiB|复用已退休 Payload 低地址，统一 BF16 relay；key0/1 为完整64阶结果|
|Eye|每 lane 两 slot 各初始化一次|启动时共16KiB写入|immutable GM；key0/1/2/4 每 work 读8KiB到UB；key3/5 每 work 读8KiB到L1 X，并在 lane 首次工作额外读8KiB到L1 I|
|ALog、dt_bias|ALog按容量缓存，bias逐 work读|缓存最多128×4B/每AIV；有bias时512B/work|使用Small区；不随qkv layout改变地址语义|

key2/4 的 AIC 另读8KiB Lower；key3/5 的 AIC 分别将同一 Lower
读入 P 与 Power，各8KiB，不能因逻辑内容相同而只计一次。key3/5 的 inverse 更新与分层合并
常驻 L1，不逐轮写回 GM；W/U 在 paired L0 中共享 inverse 的装载。

完整 Save13 的有效输出按上述 r 计算；r=64 且该 value head 是 Q/K checkpoint owner 时，
唯一输出区域共160.75KiB：gk 32KiB，Aqk/Akk各8KiB，W/U/qg/kg/qgScaled/qHat/kHat各16KiB，
qRstd/kRstd/betaEff各256B。GVA 非 owner 不写 qHat/kHat/qRstd/kRstd，减去32.5KiB。
这只是唯一输出区域大小：Akk 的完整初始化/叶内结果之后，key3/5 Cube 还覆盖写5KiB下三角块，
key2/4还覆盖写2KiB的32×32合并块；key0/1不再由Cube覆盖Akk。
这些重复写回、RHS padding、Eye初始化和全部启动成本均包含在算子 Task/Event 中。
尾块只向公开输出写有效行；被关闭的槽不写回，不能从其节省量扣除仍必要的内部生产。

### 2.5 Gate 参数缓存

非 Precomputed gate 的 exp(a_log) 在每个 AIV 开始工作前计算一次。
HV≤128 时按实际 value-head 索引读取 Small UB 中的缓存，不随 grid-stride
work 切换而混淆 head；超过容量时仍使用逐 work 读取/计算，数学语义和六个 key 不变。
该缓存占用现有 Small UB 内 FP32 索引 [2688,2816)，不增大 UB 分配，
与 FP32 beta DMA、BF16 pack 及 triangle mask 区域均不重叠。
缓存初始化经 MTE2→V→S 依赖完成后才可消费，不增加跨核事件。
Vector/Cube 在构造时绑定同一份只读 PrepareKernelArgs 引用，避免再次复制包含padding的参数对象；
引用覆盖整个 Run 生命周期，Init 不改写参数，也不改变工作和slot选择。

## 3. 覆盖、验收和代码边界

|维度|验证原则|
|---|---|
|key与dtype|六个可达数学key，各自覆盖g/beta BF16/FP32四对象；不把对象数当新数学key|
|布局/head|BNSD/BSND/NTD/TND、HK=HV、GVA、奇数head、跨QK pair、head tile末列|
|有效行|1..64、r=1/63/64、多个序列、连续多代slot；chunk_indices与cu_seqlens一致|
|输出|none/forward/recompute/save，正确比较None及全部启用槽，Save完整13槽|
|数学|合法gate、beta模式、真实scale/epsilon/lower_bound及可选bias；不可达合同由host拒绝|
|数值|原FP64 TRUTH、FP32/BF16 CONTROL和原比较器/容差冻结；首个失败Stage单独定位|
|生命周期|小shape、当前编译binary/key的simulator及内存/同步检查；不仿真131k模型|
|确定性|同输入重复、跨slot、多序列、odd lane和关闭输出，不借用公开输出作scratch|
|性能|32模型逐例及同规模合法非模型组合；ABBA 5/50/3，Task与Event分列，独立第二窗口|

所有报告记录真实40位source commit、host/kernel/wrapper与包hash、实际输入hash、物理卡和
独占guard、运行tag。日常可只读引用已验证双参考缓存，DUT须新算；交付ATK另库保持独立运行。
代码设计文档不混入某次实验的临时“通过”声明；最终验证与性能原始证据由独立交付件绑定。
未通过合同明确记录，不以模型平均收益抵消逐例回退，不以原窄合同fast替代本实现验收。
