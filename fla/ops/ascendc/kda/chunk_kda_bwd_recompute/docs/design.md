# chunk_kda_bwd_recompute Atlas A2系列产品和Atlas A3系列产品设计方案

## 1. 目标

主模型 case F: B=1,HK=16,HV=128,T=11264,K=V=128,C=64；V: packed T=16384，明确使用16条1024的参考长度列表。
性能目标：用户2026-11-19明确要求仅报告实测耗时，豁免数值性能目标准入与达标结论，不将未确认的0.4x作为目标。
原始v2的技术方案作为完整推演存入 design.full.md。A5/A6路径保持原行为，本次只修改通用A2/A3路径。

### 1.1 数据类型

| 范围 | 数据 | 数据类型 |
|---|---|---|
| API输入 | q/k/v/A | bf16 |
| API输入 | g/beta | bf16或fp32，独立选择 |
| API输入 | A_log/bias | fp32 |
| API输出 | gk;w/u/qg/kg | fp32;bf16 |
| Vector计算 | gate/cumsum/exp/products | fp32 |
| Cube输入与累加 | A/vb/kbg;acc | bf16;fp32 |
| 跨Stage交接 | GM页vb/kbg | bf16 |

### 1.2 理论性能上限分析

满chunk/head两次GEMM共2097152 FLOPs。F为47.24 GFLOPs，V为68.72 GFLOPs。
F必要输出2112MiB加页交接1408MiB，V必要输出3072MiB加页交接2048MiB，另外计算真实输入读流量。
理论下界为max(FLOPs/Pcube, VectorWork/Pvector, GMBytes/BW, dependencyCriticalPath)。
采用实际910B3可用核数与资源查询；没有同SoC实测有效算力、带宽或native baseline时不伪造数值下界。
数值性能目标准入按用户明确豁免，理论可达性不作已证实结论。构建与正确性通过后只报告msprof实测耗时。

## 2. 变量与任务映射

| 符号 | 含义 |
|---|---|
| C,D,G | 64,128,HV/HK |
| CG | host动态选择1..4，HV小于4时等于HV；G=3选择3，其余初版4，并不声称最优 |
| Nh,Nbase | ceil(HV/CG)，所有序列有效chunk数 |
| usedAic | min(可用AIC,Nbase*Nh)，空任务不发页事件 |
| L,p | chunk有效长度、ordinal%2页号 |

```text
work=core+ordinal*usedAic
chunk=work/Nh; head_begin=(work%Nh)*CG
AIV a遍历所有有效head并处理d=[64*a,64*a+64)
DIRECT模板hk=hv，GVA模板hk=hv/G；Q/K在同work同hk时驻留复用
qk=((batch*HK+hk)*T+token_begin)*128+d
out=((batch*HV+hv)*T+token_begin)*128+d
```
定长与packed起点、长度使用64位；packed空序列无work，序列tail不跨边界。
HEAD_MODE编译轴0/1，两种模式与四dtype组合形成8个逻辑key；CG留在TilingData运行时字段。

## 3. Stage 总览

| Stage | 执行单元 | 主要动作 | 并行与流水关系 | 结束条件 |
|---|---|---|---|---|
| S0 | 两AIV | 同head通道分片，safe-gate/cumsum及所有Vector结果 | 生产work(i+1)与Cube消费work(i)并行，GM双页流水 | 所有有效head半片MTE3完成，再ready |
| S1 | AIC | 每head两GEMM u/w | 与下一work的Vector并行，等完整页ready后消费，最后释放页以供i+2复用 | 最后page读取完成、本地Cube/FIX排空 |

### 3.1 芯片规则应用检查

| 规则 | 结论 | 设计证据/不适用原因 |
|---|---|---|
| R01 | 已应用 | S0仅Vector，S1仅Cube |
| R02 | 已应用 | u/w无互相数据依赖，合并S1 |
| R03 | 已应用 | 每work CG head，两个GM页独立生产消费 |
| R04 | 已应用 | vb/kbg经GM页交接 |
| R05 | 已应用 | S0保留K/Gacc，Q/K跨同hk复用，S1同head A供两GEMM |
| R06 | 已应用 | 两页用于下一work生产与当前消费并行 |
| R07 | 已应用 | S0每AIV显式缓冲，S1单head工作集，编译核验物理容量 |
| R08 | 已应用 | 原始输入和GM交接默认NORMAL缓存，实测前不猜测bypass收益 |
| R09 | 已应用 | FP32稳定sigmoid，kg使用后缀差指数，无效行exp前排除 |

## 4. Stage 详细设计

### S0 · Vector：重建检查点

```text
z=exp(A_log)*(g+bias)
gate=lower_bound*exp(min(z,0))/(1+exp(-abs(z))) if enabled else g
Gacc=cumsum(gate)*(RCP_LN2 if exp2 else 1)
qg=bf16(q*E(Gacc)); kbg=bf16(k*beta*E(Gacc))
vb=bf16(v*beta); kg=bf16(k*E(Gacc_last-Gacc))
```
Gacc和K不回读GM。每AIV连续半行64元素用2D DMA，行stride128。
为MemBase API显式预留原始输入及转换scratch，本轮从原v2的100KiB容量计划修正为132KiB上限，不能把额外scratch藏在保留区。

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
|---|---|---|---|---|
| vb/kbg | GM（用户 workspace） | page_base+r*32768B+0/16384B，单record32768B | 使能 | 每AIV写不同通道半片，tail行清零，ready前写完成 |
| gk/qg/kg | GM（API 输出） | tensor_base+((batch*HV+hv)*T+token_begin)*128*elementBytes+d*elementBytes，每行128B(bf16)或256B(fp32) | 使能 | 实际L行；gk关闭无需写全量占位 |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
|---|---|---|---|---|
| q/k | GM（API 输入） | tensor_base+((batch*HK+hk)*T+token_begin)*256B+d*2B，单半片8192B | 使能 | 同work同hk驻留跨value head复用 |
| g/v/beta/bias/Alog | GM（API 输入） | tensor_base+head_token_offset，最大每head半片16384B | 使能 | 各AIV独立搬入，公共参数按需要复制 |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
|---|---|---|---|---|
| q,k,v原始与bf16输出scratch | UB | [0,32) KiB，每份8KiB，共4份单缓冲 | 不涉及 | q/k驻留，输出scratch每次MTE3完成后复用 |
| g原始 | UB | [32,48) KiB，16KiB单缓冲 | 不涉及 | 按FP32最大容量，bf16单独转换 |
| Gacc,tmp,Kraw,work,Qfp | UB | [48,128) KiB，5份16KiB单缓冲 | 不涉及 | 同head所有64token保留，Qfp复用于V计算 |
| beta/carry/bias/Alog | UB | [128,132) KiB，4KiB容量 | 不涉及 | beta原始/FP32、Brcb、bias、标量及carry |

### S1 · Cube：两次GEMM

```text
u[L,128]=A[L,L]@vb[L,128]
w[L,128]=A[L,L]@kbg[L,128]
```
按head顺序消费页。每head A仅同head复用，不跨value head复用。尾块内部补零，输出仅L行。
当L<64时，使用目标A2 SDK InitConstValue(blockNum=768个32B块,repeatTimes=1)清零完整24KiB L1，并使用PIPE_MTE2屏障确保清零在ND2NZ搬运前完成。每head构造独立resident block，避免B页地址相同但内容已回绕时沿用旧数据。
S0 beta FP32扩展区在Brcb前显式清零，避免尾部对齐lane读取未初始化值。

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
|---|---|---|---|---|
| w/u | GM（API 输出） | tensor_base+((batch*HV+hv)*T+token_begin)*256B，L*256B | 使能 | FP32累加，Fixpipe为bf16 |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
|---|---|---|---|---|
| A | GM（API 输入） | A_base+((batch*HV+hv)*T+token_begin)*128B，L*128B | 使能 | 同head两GEMM共用 |
| vb/kbg | GM（用户 workspace） | page_base+r*32768B+0/16384B，每tile16384B | 使能 | ready等待后访问，最终读取完成后free |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
|---|---|---|---|---|
| A/B | L1 | [0,8),[8,24) KiB，24KiB单缓冲 | 不涉及 | Catlass ENABLE_L1_RESIDENT；同head A保留，两份B顺序复用同16KiB；每head重建对象避免GM页回绕误判驻留 |
| A操作数 | L0A | L0A[0,8) KiB，8KiB单缓冲 | 不涉及 | 64x64 bf16，MMAD完成后复用 |
| B操作数 | L0B | L0B[0,16) KiB，16KiB单缓冲 | 不涉及 | 64x128 bf16，MMAD完成后复用 |
| acc | L0C | L0C[0,32) KiB，32KiB单缓冲 | 不涉及 | u的Fixpipe完成后覆盖为w |

## 5. 用户额外 workspace

| 区域 | 数量与单份大小 | 主模型 case 大小 |
|---|---|---|
| GM双页 | usedAic*2*CG*32768B | CG4、24核示例6.291456 MB；实际按设备核数计算 |

```text
workspace_size=usedAic*2*CG*32768
page_base=workspace+(core*2+page)*CG*32768
```
只统计用户额外GM页。API 输入、最终输出不属于该数值；系统 workspace单列。
适配连续化和gate关闭占位必须另外计入峰值，不以页预算替代实际临时内存。

## 6. 跨 Stage 同步契约

| 数据边 | Producer | Consumer | ready | free/复用 | 本地 pipe | 回绕/退出 |
|---|---|---|---|---|---|---|
| 双页vb/kbg | 两AIV的所有有效head半片 | AIC全部head两GEMM | mode2集合set/wait flag，两个AIV全部MTE3完成 | 复用前wait free，AIC最后读取完成set free | V_MTE2，MTE2_V，V_MTE3，MTE3_V；AIC MTE2/MTE1/M/FIX事件 | 首两页不等free，仅有后续使用才通知，空work不发事件，退出drain |

```text
AIV ordinal: if ordinal>=2 wait FREE[p]; load own half; wait MTE2;
compute S0; store public outputs and page; wait MTE3; set READY[p]
AIC ordinal: wait READY[p] from both producers; execute S1 all valid heads;
wait last page read; if ordinal+2<count set FREE[p]
exit: drain local DMA and Fixpipe before resources released
```
所有SetFlag有配对WaitFlag。独立页跨work重叠，不能用逐work串行生产、计算、消费替代。
