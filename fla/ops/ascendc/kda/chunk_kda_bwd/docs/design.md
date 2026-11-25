# KDA Chunk 前反向维度与 FP16 泛化设计

版本 D1.3，状态为可开始首版功能 kernel 开发的设计。实现基线 fc0bc08e；分支 feat/fla-compat。工作流参考 PR833 / 9ec4253b。本文冻结首版算法、缓存、调度和同步；实现后的编译与设备验证记录见 [validation.md](validation.md)；本文不将 A3 结果视为 A2 验收或性能结论。

## 1. 范围与实施决策
目标为A2/A3、C64、HK=HV，K/V为60×60、64×64、100×100、128×128、256×256、64×32。Q/K/V同类型FP16/BF16，dq/dk/dv回写原类型。支持已有前向norm和beta sigmoid组合、A2/A3的两倍sigmoid，训练沿raw safe gate/exp2支持域实现保存与重算。
A3首先开发与实测，A2独立验收。新增GVA、C32、Graph协议、组合h0/dht反向另列需求。所有公开规格校验在ACLNN，Python只路由。

首版新增通用路径采用纯Vector、内部FP32数据流。这样能直接消费FP32 score和状态，无需假设混合精度Cube指令支持；全部新规格先获得相同的计算与训练契约。既有匹配的BF16/128热路径保留。预计通用路径显著慢于专用Cube路径，首版目标是可验证的泛化与精度，后续优化必须以本版数学和缓存契约为基线。

本轮选择改变了前稿的“全链64×64 Cube候选”。独立C/C++算子保留原入口并增加通用模板，ACLNN根据实际dtype/shape分派。新路径的各阶段均为Vector任务，无跨AIC/AIV共享结果槽。旧混合路径不借用新模板的缓冲和同步。

## 2. 接口和缓存契约
### 2.1 输入与输出
公开参数顺序、默认值和返回槽位沿现有接口。ACLNN完成shape/dtype/layout、组合开关、索引和大小溢出检查。新通用路径各公开缓存允许FP32描述符；旧类型继续用于旧路径。

|数据|新通用路径存储|逻辑末维|
|---|---|---|
|q/k/v，最终o，dq/dk/dv|输入类型T|K/K/V|
|q_hat/k_hat，qg/qg_scaled/kg/W|FP32|K|
|U/Vnew|FP32|V|
|G，rstd，beta_eff|FP32|K/标量/标量|
|Aqk/Akk|FP32|C|
|h快照、运行状态、反向状态|FP32|K×V|
|d_aqk、dq_raw、dv_local和反向其它缓存|FP32|C/K/V|
|dA_log/ddt_bias|FP32|H/H×K|
|dg/db|各自原输入dtype|K/标量|

新增内部FP32缓存不是把整个用户接口改成FP32主输入。前后向必须消费同一条路径的缓存，ACLNN拒绝混合不兼容缓存集合。更新涉及的算子def、InferShape、tiling实例、ACLNN及launcher输出描述；依赖新缓存的wheel/OPP必须配套。旧直接调用的合法缓存仍路由旧模板。

final_state按现有StateT约定转换。新路径从最后FP32运行状态直接转换，默认无初态时为FP32。独立入口支持的物理状态布局使用真实K/V stride；组合反向仍按原入口边界拒绝非空h0/dht。

### 2.2 beta反向接口类型
|调用层|beta输入|sigmoid开启时输出db|写回责任|
|---|---|---|---|
|独立Finalize|FP32 beta_eff|FP32，已应用激活导数|独立算子按其输入描述符分配|
|组合反向|原始beta/logits及FP32 beta_eff|原始beta/logits dtype|消费独立Finalize FP32结果后cast一次|
|sigmoid关闭|有效beta|跟随对应入口beta dtype|导数因子1|

### 2.3 保存与重算
保存模式保留Aqk/Akk、G/W/qg/kg/Vnew/h，以及开启norm时的q_hat/k_hat/rstd、开启sigmoid时的beta_eff与原始beta信息。
重算模式必须保留Aqk/Akk、q_hat/k_hat/rstd及对应beta信息；BwdRecompute.A直接引用前向Akk。重算恢复G/W/U/qg/kg，FwdH恢复h/Vnew。
重算不再激活q_hat/k_hat或beta_eff，不重新求逆。共享前向派生量函数，使同一数据在两条模式具有相同运算次序。
无norm时q_hat/k_hat由原输入提升FP32；有norm时使用保存FP32结果。

## 3. 数学与数值定义
以下为新路径精确的运算图，所有中间算术FP32；按显式顺序执行，最终输出才cast_T。数学reference使用高精度递推独立核对，control模拟FP32节点。

### 3.1 gate和归一化
qhat=q*rsqrt(sum_K(q²)+epsilon)，khat同理；sum只包括逻辑K。norm关闭时直接提升输入。
x=raw_g+dt_bias，a=exp(A_log)。
safe模式delta=ell*sigmoid(a*x)，ell为负下界。
softplus前向delta=-a*softplus(x)。
预激活前向delta=输入log-step。
G=c*cumsum(delta)，每chunk从零开始；exp2时c=1/ln2，否则c=1。E为对应底数指数。

beta_eff=sigmoid(beta_raw)或2*sigmoid(beta_raw)，关闭时为输入有效beta。激活在FP32完成。safe sigmoid采用稳定正负分支，softplus采用max(x,0)+log1p(exp(-abs(x)))等价稳定形式，实际API组合按现有实现复用。

### 3.2 score与三角逆
对有效i,j逐对直接计算指数差。未进入因果域的(i,j)不执行指数。
R_qk(i,j)=sum_K(qhat_i*khat_j*E(G_i-G_j))。
R_kk(i,j)=sum_K(khat_i*khat_j*E(G_i-G_j))。
Aqk(i,j)=FP32(scale*R_qk(i,j))，j≤i，其余为0。
L(i,i)=1，L(i,j)=beta_eff_i*R_kk(i,j)，j<i，其余为0。
物理L为diag(L_valid,I_(64-M))，交叉块为0。

新路径统一用FP32前代求单位下三角逆，取消新路径内部快慢求逆分支。按列j、行i递增
A(i,i)=1；
A(i,j)=-sum_(k=j..i-1)L(i,k)*A(k,j)，i>j；
A(i,j)=0，i<j。
所有累加以固定k升序FP32执行，写Akk时不降精度。M外的物理对角为1，交叉块为0。Akk与旧分支不要求逐位相等，但须满足数学误差策略。

此定义同时解决Akk舍入未决项和190KiB旧求逆arena依赖。旧快速模板仍使用原计算方式，其接口回归独立验证。

### 3.3 派生量和前向
qg=FP32(qhat*E(G))；
qg_scaled=FP32(qg*scale)；
kg=FP32(khat*E(G_last-G))；
Rk=FP32(FP32(khat*E(G))*beta_eff)；
Rv=FP32(v*beta_eff)；
W=Akk@Rk，U=Akk@Rv。
Vnew=U-W@H；
O=qg_scaled@H+Aqk@Vnew；
Hnext=diag(E(G_last))*H+kg.T@Vnew。
各矩阵乘用第5节Vector外积累加模板，运行次序固定。

### 3.4 状态精度
新路径Hrun、公开h快照和矩阵乘读取的H均为FP32。chunk入口复制Hrun到h；WH读取同一FP32入口状态，产生FP32 Vnew；Hnext以FP32写入另一个运行槽，交换后作为下一chunk输入。final_state仅在返回时按StateT转换。
因此新路径不存在FP16 h快照在WH前的隐式舍入。保存与重算均用同一FwdH模板。旧路径仍沿其BF16快照和StateT运行规则，不能以旧输出dtype推导新路径。

### 3.5 反向
BwdPrepare输出
d_aqk=scale*tril(dO@Vnew.T)，FP32；
dv_local=Aqk.T@dO，FP32；
dq_raw为FP32。K-first物理h[K,V]时dq_raw=dO[M,V]@h.T；V-first物理h[V,K]时dq_raw=dO[M,V]@h，结果均为M×K。
d_aqk是未缩放score的梯度，dv_local使用已含scale的Aqk。

Dhu接收未乘scale的qg和独立scale，与保存及重算输出一致。从Dnext开始倒序处理每个chunk c：
先写dh[c]=Dnext，保存decay及状态更新之前的FP32值；
dVnew=dv_local+kg@Dnext；
termQ=qg.T@dO；
Dprev=FP32(scale*termQ)+diag(E(G_last))*Dnext-W.T@dVnew。
termQ先完成FP32归约，再乘scale；control与kernel使用同一顺序，不改为预先缩放qg。
完成整个K域依赖后以Dprev作为下一轮Dnext。终端取零或独立入口支持的dht。Finalize的dkg=Vnew@dh[c].T，读取更新前快照。支持dh0的独立入口在完成最早chunk后返回Dprev。

Finalize按反向计算图执行
dU=dVnew；dW=-dVnew@h.T；
dkg=Vnew@Dnext.T；
Finalize先按前向FP32节点重建Rk=FP32(FP32(khat*E(G))*beta_eff)、Rv=FP32(v*beta_eff)，按tile临时生成并在最后消费者结束后复用；Rk/Rv不作为额外autograd保存量。保存与重算两条链都使用同一重建函数。随后计算dA=dW@Rk.T+dU@Rv.T；
dL=-Akk.T@dA@Akk.T，取严格下三角；
dRk=Akk.T@dW；dRv=Akk.T@dU。
最终dv=cast_T(beta_eff[:,None]*dRv)，乘法在FP32执行；这里使用有效beta，sigmoid链式导数只应用于dbeta。beta_eff为0时对应dv为0，而dbeta仍可非零。
由直接差值score公式求qhat/khat/G贡献，对每有效(i,j,d)，指数差梯度对G_i加、G_j减。合并qg、kg、Rk及状态decay贡献后执行本chunk gate逆扫。
qg相关梯度按独立dq_raw与scale定义应用一次，禁止再次给dv_local乘scale。

在所有score/qg/Rk贡献之外，显式合并kg及状态decay的末行贡献。令b=ln(2)（exp2域）或1（自然指数域），逐元素
t_i=b*dkg_i*kg_i；
dG_i -= t_i；
dG_last += sum_i(t_i)；
dG_last += b*E(G_last)*sum_V(h*Dnext)。
i=last时kg的加减贡献数学上相消，实现可跳过该项的成对更新以避免不必要舍入。其余末行贡献使用累加，不能覆盖已有dG。scratch由当前task单owner顺序更新。
以上全部合并后才执行逆向cumsum。

G的实际存储域梯度记dG：
dDelta_j=c*sum_(i=j..M-1)dG_i，每chunk初始化suffix为0。求E导数时对应底数系数在dG形成处保留，禁止重复换底。
safe模式的数学导数含sigmoid(a*x)*(1-sigmoid(a*x))，实际FP32实现固定用稳定形式：
t=exp(-abs(a*x))，sigma_prime=t/((1+t)*(1+t))；
dRaw=dDelta*ell*a*sigma_prime；
禁止从已舍入为1的sigmoid值计算1-z。kernel和FP32 control采用同一稳定计算顺序。
dA_log=sum_(token,K)(dRaw*x)；
ddt_bias=sum_token(dRaw)。
softplus数学导数为dRaw=-dDelta*a*sigmoid(x)、dA_log=sum(dDelta*delta)，仅用于已支持路径；本轮训练能力边界见第1节。

每个有效token i的beta_eff梯度明确为
dBetaEff_i = sum_d(dRk_id * FP32(khat_id*E(G_id)))
          + sum_v(dRv_iv * v_iv)
          + sum_(j<i)(dL_ij * R_kk(i,j))。
R_kk(i,j)按3.2直接差值指数重算，使用有效(i,j)且固定K归约顺序。不能用L_ij/beta_eff_i恢复R_kk，因为beta_eff为0时该导数仍可非零。
以上三项均在FP32内归并完成，再按三个分支生成dbeta：
use_beta_sigmoid_in_kernel=False时dbeta=dBetaEff，链式因子为1；
sigmoid开启且allow_neg_eigval=False时dbeta=dBetaEff*beta_eff*(1-beta_eff)；
sigmoid开启且allow_neg_eigval=True时dbeta=dBetaEff*beta_eff*(1-beta_eff/2)。
ACLNN拒绝sigmoid关闭而allow_neg_eigval开启的组合。最后按调用层次处理类型。独立Finalize的db跟随其beta入参dtype，sigmoid模式该入参是FP32 beta_eff，所以db保持FP32；组合入口持有原始logits的描述符，在独立Finalize结果返回后才cast到原始beta的dtype。数学上sigmoid模式db已是logits梯度，组合入口只做dtype转换，不能再乘一次导数。
L2Norm关闭时dq=cast_T(dqhat)、dk=cast_T(dkhat)，不读取rstd，不做投影。
L2Norm开启时按完整逻辑K归约，dq=rstd*(dqhat-qhat*sum_K(qhat*dqhat))，k同理，再cast_T；rstd和qhat使用同一前向保存值。返回dq/dk/dv与各自原输入dtype一致，dg跟随raw_g，db遵循上述独立/组合入口分层契约。

### 3.6 有限性承诺
M16接近-5 gate的band反例使用直接差值计算，不产生人为Inf*0。
M1、q=k=256、scale1、norm关闭时Aqk=65536以FP32保存，v0/h00时输出必须有限且为0。
所有中间缓存FP32，避免这些half relay溢出。对于真实数学结果或FP32中间量超出FP32范围，遵循FP32算术语义；不承诺任意有限输入均返回有限结果，不通过裁剪改变数学定义。验收值域和误差阈值由reference/precision policy明确，不能只以有限性替代精度。

## 4. 调度和Tiling
新路径为Vector-only kernel，各AIV独立拥有任务与UB，不存在两个AIV共同读取一个结果slot。任务编号使用实际Vector核数分配。
独立chunk阶段为(seq,chunk,head)，grid-stride；计算score/逆矩阵时单owner处理完整64阶矩阵，避免参数或矩阵写入冲突。
FwdH/Dhu为(seq,head,Vtile)，TV=32。每owner按序列顺序/逆序处理完整K×32状态，K256时该状态为32KiB。
独立梯度stage按(seq,chunk,head)处理，按输出列片32遍历，参数梯度写每任务独立partial，再启动独立Vector归约kernel按固定任务顺序归并。
TK=64用于直接差值score沿K的分段；归约固定tile次序。K/V行距使用逻辑尺寸，内部tile不足32/64时补零。
新tiling分派条件由ACLNN选择general-fp32模板并按输入T设置最终cast。代码实现时使用未占用的key值；旧key含义保持原定义。未匹配已有或新增模板的组合由ACLNN报错。

## 5. Kernel Stage和FP32消费者实现
所有新Stage为Vector，矩阵乘统一采用行外积累加：对输出行i和32列片，初始化acc[32]=0，按归约r递增，加载B[r,col:col+32]，广播A[i,r]，Mul后Add到acc，最后一次写回。输入以FP32读；用户半精度输入在入口Cast升精度。转置访问通过分行DataCopyPad收集到UB，禁止将不连续GM当作连续tile。
该实现只依赖FP32向量Add/Mul/Muls、Exp、Cast和归约，避免混合Cube输入限制。

|kernel|Stage次序|关键依赖|
|---|---|---|
|FwdPrepare|输入升精度/norm/激活 → 直接差值score → scale/mask → FP32前代求逆 → 派生量 → W/U|score全K归约结束后写Aqk；Akk在求逆结束后供两次乘法使用|
|FwdH|跨K求WH → Vnew → kg.T@Vnew → decay状态更新|先完整WH，后更新所有K行|
|FwdFinalize|qg_scaled@h → Aqk@Vnew → 合并/最终cast|两消费者直接读FP32 h/Aqk|
|Recompute|读取保存qhat/khat/beta_eff/Akk → gate与派生量 → W/U|不重做norm/sigmoid/求逆|
|BwdPrepare|dO@Vnew.T → scale/mask；Aqk.T@dO；按state_v_first取dO@h.T或dO@h|三路全FP32暂存|
|Dhu|保存dh[c]=Dnext → kg@Dnext → dVnew → termQ及termW → scale/decay/状态交换|输入为qg及独立scale；快照在更新之前写出|
|Finalize|矩阵梯度 → score逐对梯度 → gate逆扫 → norm/beta链式导数 → cast|每task独占输入梯度区，参数写partial|
|参数归并|按head/K片读取全部partial求和|最后写FP32参数梯度一次|

Finalize中score梯度逐行/列串行累加至GM任务私有FP32 scratch；其每个输出元素由同一task owner更新，不使用跨核原子。当q/k等头时无需GVA归约。

## 6. 内存与workspace
### 6.1 UB地址
新模板每AIV固定128KiB arena。以下为字节半开区间。
[0,32768) A，FP32，最大K256×TV32状态或64×128矩阵输入；
[32768,65536) B，同大小，第二状态或第二操作数；
[65536,81920) C，64×64 FP32 L/A/score之一；
[81920,98304) D，64×64 FP32逆矩阵或结果；
[98304,106496) E，输入DataCopyPad/TQue区，8192 bytes；
[106496,114688) F，输出TQue区，8192 bytes；
[114688,122880) G，FP32乘法/归约scratch；
[122880,131072) J，FP32统计、cast临时和索引小表。
同一Stage内A/B/C/D的角色以调用次序固定。Stage边界先将后续需保留的缓存写GM并等待完成，再复用区域。前代求逆只用C/D与G，所有64阶对象可完全驻留；原190KiB求逆arena不进入新模板。

H/Dhu阶段A/B容纳FP32运行状态双槽，C/D用于当前32列输出和暂存，score按行流入E。最大K256时双状态共64KiB。跨chunk状态留在A/B；不同seq/head任务结束后才重用。
TQue实际初始化分配必须保持上述E/F各8192byte预算，其余采用TBuf；总分配不超过128KiB。源码arch22已有190KiB使用事实支撑预算，Host仍查询实际UB大小并在不足时拒绝该模板。新路径L1/L0分配为0。

### 6.2 GM缓存公式
P为有效token总数×H，N为chunk总数×H。每条分配按32byte对齐并做64位溢出检查。
Aqk/Akk合计8PC；
G、W、qg、kg、qg_scaled每份4PK；
U、Vnew每份4PV；
qhat/khat每份4PK；
rstd每份4P，beta_eff为4P；
h为4NKV，反向dh为4NKV；
d_aqk为4PC，dq_raw为4PK，dv_local/dv_scan各4PV。
各量是否分配由输出档位与最后消费者决定；autograd返回缓存直到反向末次使用才释放。运行状态位于UB，没有每owner GM双状态scratch。

Finalize task私有scratch最保守预算为dA/dL两份64阶矩阵、dQ/dK/dG三份C×K、dV/dU两份C×V以及dW/dKG两份C×K：
4*(2*C*C+5*C*K+2*C*V) bytes/task。
实际按活动Vector核数分配并跨grid-stride work复用；输出参数partial单独按N*(K+1)*4，归约kernel最终写H*(K+1)*4。存在bias时才分配对应K列，dA_log按每head归并。
前向峰值、反向留存和executor暂存按上述集合的实际生命周期分别求和。SDK系统workspace另计。

## 7. 搬运、同步和边界
新模板采用深度1的输入/输出TQue，CopyIn后EnQue、计算前DeQue；计算结束释放输入，CopyOut使用输出EnQue/DeQue并在搬运完成后释放。API内部管理MTE2/V/MTE3事件，禁止同时手工复用这些事件。
A/B/C/D驻留区只有当前AIV访问，连续Vector依赖显式PipeBarrier<PIPE_V>；通过E/F队列完成所有GM读写。循环中下一结果必须先Alloc到空闲输出队列，前一次MTE3未完成时无法复用F。这是新模板结果槽的唯一信用机制。
无跨核ready/free、无跨AIC/AIV flag、无共享L0C/Fixpipe结果。各kernel之间通过executor顺序和正常kernel完成依赖传递GM数据。旧混合模板保留原同步并单独回归。

采用源码已使用的DataCopyPad/DataCopyExtParams接口，单行blockCount=1，blockLen=有效元素数*元素字节宽，源/目的stride=0；多行逻辑stride在外层行循环计算。输入补零只发生在UB物理tile，GM输出长度保持有效byte数。100维half末片36元素只写72bytes；FP32缓存同片只写144bytes。
M外score为0，求逆对角为1；空序列不生成chunk，状态owner按实际入口契约处理，不能在Python静默压缩索引。

## 8. 标杆与验证
验收使用 fla-org/flash-linear-attention 原始测试，不修改其输入和精度断言，不使用本地 ATK。固定上游提交 c55dc553edf44701bf703715ed052d811f8a6376，记录 pytest node ID、实际 AscendC 路由及结果；回退 Triton 或被 skip 的用例不计为本次算子通过。FP64递推/autograd仅为独立公式自检，FP32逐节点control仅用于定位，不替代上游测试。
本轮不接通组合反向 h0/dht。上游 test_chunk/test_chunk_varlen 中依赖非零初态、末态梯度及 dh0 断言的用例记为未覆盖，不置零输入、不删除断言，也不计为通过。test_chunk_varlen_prefill 的前向初态支持保持原契约；无 h0/dht 依赖的 test_chunk_varlen_fused_training_contract 用于保存/重算训练验证。
现有precision policy优先复用；新增FP32缓存精度规则在测试资产中明确，不用扩大阈值掩盖非有限值。进入04前先运行CPU矩阵公式与递推一致性、反向autograd一致性检查。

必测
1. M16、接近-5 safe gate、单位基向量，所有有效score有限。
2. M1、q=k=256、scale1、norm关闭、v0/h00，Aqk65536、o0。
3. K64/V32、T65、scale0.37、FP16，四组norm/sigmoid开关、保存/重算。
4. allow_neg_eigval=True、logit0、非零beta梯度，beta_eff1、导数因子0.5。
5. T65分别检查第二chunk FP32入口状态、WH、Vnew和最终状态转换。反向另取dht=0，仅最后token的dO非零，检查末chunk的dh快照为0、首chunk的dh等于末chunk计算的Dprev；使用scale0.37逐chunk比较独立reference。
6. sigmoid关闭时以有效beta=1、dBetaEff=1做局部导数检查，期望dbeta=1；组合梯度另与autograd比较，不能把不可控组合dBetaEff当作恒1。
7. 独立BwdPrepare用K64/V32的同一逻辑状态，分别以[K,V]及[V,K]物理布局输入，比较dq_raw及其余两输出一致。
8. M1/15/63检查物理求逆单位对角。
9. packed cu=[0,0,1,66,66,129]与逐序列独立执行比较；只扰动65序列，其余输入梯度和输出不变，参数梯度按独立结果求和。入口不支持空段时先验证独立阶段并记录组合能力依赖。
10. norm关闭的局部导数用q=e1、dqhat=e1，检查dq=e1；保存/重算均覆盖，rstd缺省时不得读取。
11. raw safe gate使用a*x=+20/-20与非零dDelta，核对sigma_prime约2.0611536e-9及FP32 dA_log/ddt_bias；dg输出低精度时同时核对cast前值和目标dtype参考，避免把允许的最终舍入误判为内部提前归零。
12. sigmoid开启使用低精度原始logits和FP32 beta_eff，分别验证独立Finalize db为FP32、组合db为原始dtype，并检查导数只应用一次。
13. M2、sigmoid关闭、beta_eff[1]=0，构造严格下三角dL贡献非零，比较显式三项dBetaEff与独立autograd；禁止除以beta恢复score。
14. beta_eff为0的token显式检查dv为0并允许dbeta非零；逐元素核对逆扫前dG，特别是kg末行抵消与状态decay贡献。
15. 六组K/V×FP16/BF16，T1/15/63/64/65/127及长序列，布局、保护区、重复运行。
A3直调Stage → 模板full → 总链 → wheel + fla-org 原始测试；A2独立验收。共享Host改动回归旧A5及旧BF16路径。

CPU数学验证记录：check_kda_design.py使用NumPy float64手写块前向和完整基础反向，独立逐token递推由autograd自动求导。固定seed20261125，共8例，M/K/V为1/4/3、2/4/3、5/7/3、16/64/32，每组覆盖末token beta为零与非零，scale0.37，非零初态和末态梯度。比较q/k/v/beta_eff/delta/h0六类梯度，最大绝对差1.7763568394002505e-15；块前向与递推最大差8.326672684688674e-17。beta零的dv=0且dbeta非零检查通过。结果见check_kda_design.results.json。
该验证只覆盖单chunk自然对数域、高精度基础公式，未覆盖FP32舍入、norm/raw-gate激活链、packed跨chunk、完整规格矩阵或设备执行。非零状态用于数学验证，不扩大组合入口支持域。

## 9. 性能与后续优化
本首版为FP32 Vector通用实现，计算复杂度包含O(C²K)直接score、O(C³)前代求逆，以及O(CKV)状态乘法。预计吞吐低于旧Cube路径，不作性能提升承诺。
同机记录msprof各kernel Task Duration固定求和、总调用耗时和GM峰值。功能精度通过后，逐个将范围可证明的矩阵乘替换为Cube或优化Vector归约；每次保持FP32缓存契约与基准，不一次更改缓存、算法和调度。

## 10. 开发映射与进入条件
FwdPrepare新增general-fp32模板、资源和tiling；共享FwdH/Dhu新增FP32缓存输入实例；两个Finalize、BwdPrepare和Recompute增加FP32消费者；ACLNN统一描述符检查/输出推导；launcher按新路径分配缓存，Python保持原始参数透传。

开发顺序为唯一标杆 → FwdPrepare的M1/M16 → FwdH/Finalize → BwdPrepare/Dhu/Finalize → 保存/重算 → 完整矩阵。每个模板编译时验证实际API签名和资源报告，设备运行失败按首个偏离Stage回流。

本版已经选定消费者算法、Aqk/Akk和状态精度、求逆运算顺序、Vector调度与队列同步；可以进入首版实现。尚未完成的是实现产物的编译、资源报告、精度和性能验收。CANNBot运行状态未执行，不能把本文状态等同于工作流工具已标记complete。

## 11. 证据
- PR833 head 9ec4253b 的算子自动开发工作流。
- fc0bc08e FwdPrepare arch22 resource.h及vector.h，提供190KiB UB、DataCopyPad和FP32 Vector算术的现有使用证据。
- FwdH arch22 vec.h区分stateFp32、stateBf16与h/final_state写回；新路径明确采用FP32运行和快照。
- BwdPrepare docs/api.md明确三路scale与dtype约定，BwdRecompute和总反向实现明确Akk输入依赖。
- BwdFinalize和组合反向docs/api.md明确beta_eff及两倍sigmoid导数。
- 本地Ascend C API资料中的DataCopyPad与TQue EnQue/DeQue同步用法；编译阶段仍需对目标CANN头文件核验。
