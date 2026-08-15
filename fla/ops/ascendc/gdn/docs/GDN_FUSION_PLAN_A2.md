# A2 GDN 全融合计划归档

本文档是 A2 平台 GDN 全融合的稳定路线和冻结 Phase 边界归档。当前进度只见
`GDN_CURRENT_STATUS_A2.md`，最终数据只见对应 Phase 验收报告；只有路线或边界变化时才更新本文档。

### 性能基线命名澄清

当前对外性能报告采用以下唯一口径：正式 Phase0 是“一 ACLNN executor 调度六个未融合 kernel”，
实现符号为历史版本化入口 `phase1_one_aclnn_six_kernels`；最初的“六个独立 ACLNN 调用”属于
Phase0 之前的 legacy 基线，标识为 `legacy_six_aclnn`。早期验收快照中的 Phase0/Phase1 名称按当时
冻结的历史语义保留，不回写归档数据；新的跨 Phase 性能结论必须明确写出这两个 identity，禁止将
`legacy_six_aclnn` 的收益简称为正式 Phase0->Phase6 收益。

## 1. 最终目标

在 A2 平台将 Demo 使用的完整 GDN 路径逐步融合为大融合算子，最终覆盖：

```text
输入投影/布局处理
    -> causal_conv1d
    -> local_cumsum
    -> KKT
    -> solve_tri
    -> recompute_w_u
    -> fwd_h
    -> fwd_o
    -> RMSNorm + gated SiLU
```

最终要求来自 `gdn.md`：

- 硬件：A2
- 数据格式：ND
- 数据类型：FP16/BF16
- `bs`、头数泛化
- GVA 支持，即 `Hk <= Hv`，通常要求 `Hv % Hk == 0`
- `k_dim = 128`
- `v_dim = 128/256`
- `chunk_size = 64/128`
- 额外显存以 50 MB 为优化目标并完整报告；当前不作为优先于性能的绝对硬门槛
- 融合版本精度达到 L1，并与六个 AscendC 小算子拼接版本比较
- 全场景性能不劣于小算子版本

当前第一阶段先锁定 `K == V == 128`，但接口和内部数据结构不能阻碍后续支持 `K=128, V=256`。

## 2. 两个层次必须区分

### 2.1 统一 ACLNN 入口

`aclnnGdnCoreFwd` 的第一阶段定位是稳定的复合入口：

```text
一个 ACLNN executor
    -> local_cumsum
    -> KKT
    -> solve_tri
    -> recompute_w_u
    -> fwd_h
    -> fwd_o
```

它解决的是 Python 侧统一调用、输入输出契约、元数据校验、测试基线和后续替换边界。内部仍可以调用六个已有 AscendC kernel，因此它不等于最终的单 kernel 大融合。

### 2.2 真正的 kernel 融合

真正融合是减少阶段之间的 GM 中间结果、算子启动和同步边界。融合实现应当在不改变 `aclnnGdnCoreFwd` 高层 ABI 的前提下，逐阶段替换已有小算子。

**首个真正融合目标固定为：**

```text
chunk_scaled_dot_kkt + solve_tri
```

不能把“六算子串成一个 ACLNN”误称为“六算子已经融合”，也不能把当前探索性的 `local_cumsum + KKT` 实现改称为首个正式融合目标。

## 3. 正式实施阶段

### Phase 0：六算子基线

固定以下顺序和语义，作为所有后续方案的唯一基线：

```text
local_cumsum -> KKT -> solve_tri -> recompute_w_u -> fwd_h -> fwd_o
```

基线必须覆盖：

- dense 定长
- 等长和不等长 varlen
- FP16/BF16
- `chunk_size=64/128`
- 尾块
- 普通多头和 GVA
- 输出、必要的中间结果和完整 GDN 精度

每轮记录：

- 输出精度和中间结果精度
- ACLNN 调用次数
- NPU 端 kernel 数量
- 端到端 latency
- 每个阶段 workspace 和峰值显存
- 尾块、varlen、GVA 的通过情况

### Phase 1：统一复合 ACLNN

保留六个已有 kernel 的计算实现，在 `aclnnGdnCoreFwd` 内完成统一调度。此阶段的验收重点是：

- Python GDN 可以切换到统一入口
- 与六算子 Python 拼接路径结果一致
- dense/varlen、FP16/BF16、尾块均可运行
- transpose 和 layout 转换位置固定、可追踪
- 复合入口成为后续所有融合实现的共同调用入口

### Phase 2：首个真正融合，KKT + solve_tri

新增独立的 fused kernel 或等价内部实现，计算边界为：

```text
输入 k、g_cumsum、beta
    -> 生成 KKT/A
    -> 直接完成 solve_tri
    -> 输出求解后的 A
```

第一轮不同时改外部布局，不同时融合 transpose，不同时引入 `K != V`。目的是隔离变量，清楚回答：

1. KKT 和 solve_tri 的中间 GM 往返能否消除。
2. 融合后精度是否满足 L1。
3. A2 上性能是否优于两个独立 kernel。
4. workspace 和峰值显存是否下降或至少不恶化。

### Phase 3：吸收 local_cumsum

在 Phase 2 稳定后，评估将 `local_cumsum` 与 KKT 前处理放入同一 kernel。此阶段重点检查：

- gate 累加的 FP32 精度边界
- varlen 序列边界
- 尾块和无效 padding 区域
- KKT 输入的数值顺序是否改变

Phase 3 早期以独立 `ChunkCumsumKkt` 验证这个方向，但最终 core 路由按下文修正卡收敛为累积融合。

#### Phase 3 启动卡（2026-07-25 冻结）

> 以下是 Phase 3 启动时的历史冻结卡，不代表当前仍处于 Phase 3 启动阶段。

| 项目 | 冻结内容 |
| --- | --- |
| 基线 | 局部基线为同一完整包中的 `aclnnChunkLocalCumsum -> aclnnChunkScaledDotKkt`；端到端基线为不可变 `aclnnGdnCoreFwdPhase2` |
| 首个融合边界 | 仅 `raw g -> chunk-local FP32 forward cumsum` 与 `KKT`；首版独立输出 `g_cumsum` 和未求解 `A_raw` |
| 输入 | `k=[B,H,T,128]` FP16/BF16；`g,beta=[B,H,T]` FP32；dense 时无可选元数据，varlen 时同时提供 INT64 `cu_seqlens` 与 `[seq,local_chunk]` `chunk_indices` |
| 输出 | `g_cumsum=[B,H,T]` FP32；`A_raw=[B,H,T,C]` FP32，`C=64/128` |
| 物理布局/头契约 | head-first ND；探索实现的 tiling 明确要求物理 `k.H == g.H == beta.H`，首版不实现原生 `Hk != Hv` |
| 保留项 | 保留公开 `g_cumsum` GM 输出、KKT score workspace、现有 transpose/contiguous、外部 GVA 扩头和 Phase 2 `ChunkKktSolveTri` 路径 |
| 不做项 | 不接 `solve_tri`，不创建 Phase 3 core 调度，不改 Phase 1/2，不做 reverse/scale cumsum，不扩 `V=256`、原生 GVA、causal conv、RMSNorm/gate 或 workspace 别名 |
| 独立入口 | 新增 `aclnnChunkCumsumKkt`、torch/Python wrapper 和专项测试；现有探索实现只有 OpDef/tiling/kernel/L0，尚不能直接做独立 ACLNN A/B |

静态执行语义按源码冻结为：每个物理 head、每个序列、每个 chunk 内从 raw FP32 `g` 按 token
顺序做 `values[row] += values[row-1]`，同一 FP32 UB 结果既写出公开 `g_cumsum`，也直接进入 KKT
gate epilogue。验收时必须把公开输出和内部 compute view 分开记录；只有源码和逐位证据同时证明
二者一致后，才允许以公开 `g_cumsum` 代表内部视图。

路由身份不能只写算子名。A2/CANN `9.1.0.beta1` 下首轮冻结 8 个完整身份：

```text
(FP16/BF16 k) × (C64/C128) × (dense/varlen optional-input state)
```

源码 tiling key 为 FP16 C64=`10`、BF16 C64=`20`、FP16 C128=`778`、BF16 C128=`788`；
dense/varlen 共用 key 但走不同 `isVarlen` 子路径，因此仍分开验收。实际二进制必须用 profiler 再
确认 `ChunkCumsumKkt` kernel 名、tiling key 和 block dim，不能只由源码公式推定。

最低验收矩阵和成功标准：

1. 先做 1 个 dense FP16/C64 最小 smoke，只证明独立 ACLNN 可调用、两个输出 shape/dtype 正确且有限。
2. 再覆盖 dense/varlen、FP16/BF16、C64/C128、整 chunk/尾块、多 batch/多头；每个完整路由身份至少 `10` 个 exact case。
3. `g_cumsum` 全有效元素对两小算子 NPU 拼接基线逐位一致；`A_raw` 严格下三角有效值逐位一致，并单独断言对角、上三角和尾块 padding 为零。基线非有限时改用 CPU FP64 参考，不把 NaN/Inf 当 golden。
4. profiler 证明局部 ACLNN 数 `2 -> 1` 且实际 NPU kernel 数减少；生产计时关闭 launch blocking，固定 warmup/iteration，AB/BA 测 median、P90、min。
5. 局部融合 median 不劣于两小算子基线；随后新增不可变 `aclnnGdnCoreFwdPhase3`，端到端 median 不劣于 Phase 2。若局部无收益，停止接入 core并保留 Phase 2。
6. 记录 fused/baseline workspace max/sum、peak delta 和绝对值；Phase 3 当时以相对基线额外显存不超过 `50 MB` 为门禁，绝对 `50 MB` 口径继续单列。该历史门禁不自动继承到后续 Phase，后续按当前“性能优先、workspace 持续优化并完整报告”的规则执行。

Phase 3 的第一个实现小步只补 `ChunkCumsumKkt` 独立 ACLNN、torch/Python wrapper、ABI/静态测试
和最小 smoke 脚本，不修改现有 kernel、tiling、数学顺序或 Phase 2 调度。该入口通过最小 smoke
前，不开始完整 exact 矩阵，更不接入 `solve_tri` 或 `GdnCoreFwdPhase3`。

#### Phase 3 core 累积融合修正卡（2026-07-25 冻结）

局部 `ChunkCumsumKkt` 已完成 frozen 8 identities 共 `80/80` exact，随后探索性 Phase 3 core
使用 `ChunkCumsumKkt -> Cast -> SolveTri`。生产 pilot 和完整 core profiler 证明该路由相对
Phase 2 median 回退 `3.682%`，且将 core NPU kernel 数从 `9` 增至 `10`；根因是吸收 cumsum 时
丢失了 Phase 2 已完成的 KKT + solve_tri 融合。因此 Phase 3 最终验收边界修正为累积融合：

```text
raw FP32 g -> chunk-local FP32 cumsum -> KKT -> low-precision hand-off -> solve_tri
```

| 项目 | 冻结内容 |
| --- | --- |
| Phase 2 不可变基线 | `aclnnGdnCoreFwdPhase2` 保持 `ChunkLocalCumsum -> ChunkKktSolveTri`，不原地修改 |
| Phase 3 专用 kernel | 新增 `ChunkCumsumKktSolveTri`；不改写已完成独立精度证据的 `ChunkCumsumKkt` contract |
| 输入 | 与 Phase 3 局部算子一致：`k=[B,H,T,128]` FP16/BF16，raw `g/beta=[B,H,T]` FP32，dense/varlen canonical metadata |
| 输出 | 公开 `g_cumsum=[B,H,T]` FP32；已求解 `A=[B,H,T,C]` 与 `k` 同 dtype，`C=64/128` |
| workspace | 复用 Phase 2 已验证布局：FP32 score、低精度 A hand-off、每个 MIX core group 的 solve workspace；不做输出/workspace 别名 |
| 同步 | AIC 生成 score；配对 AIV 使用已 `80/80 exact` 的 cumsum/KKT epilogue并写低精度 A；AIV 以 `PIPE_MTE3` 发出 `KKT_READY`，配对 AIC 只等待一次后进入 solve |
| public/compute cumsum | 继续使用已验证的 `MTE2_V -> V_MTE3 -> MTE3_MTE2 -> MTE3_V` 顺序；公开 FP32 输出写回后再读作 KKT compute view |
| core 调度 | `aclnnGdnCoreFwdPhase3` 直接消费该 kernel 的两个输出，不再调度独立 Cast 或 SolveTri |
| 保留边界 | head-first ND、外部 GVA 扩头、`K==V==128`、现有 transpose、dense/varlen metadata 和 Phase 1/2 固定入口不变 |
| 性能门槛 | profiler 预期完整 core kernel 数 `9 -> 8`；生产 median 必须不劣于 Phase 2，否则 Phase 3 不收口 |

实现顺序固定为：新增 op/kernel 静态注册与 L0 -> Phase 3 core 改接 -> 完整包最小 smoke -> 独立
精度/主路由/state 回归 -> 单点生产性能门禁 -> 全矩阵 -> profiler/workspace -> 验收与归档。任何一层
失败先刷新 `GDN_CURRENT_STATUS_A2.md`，不跨层继续。

### Phase 4：优先融合 fwd_h + fwd_o

令 `A/B/C/D/E/F` 分别表示 `local_cumsum/KKT/solve_tri/recompute_w_u/fwd_h/fwd_o`。
Phase 4 的正式候选边界调整为：

```text
(A+B+C) + D + (E+F)
```

优先选择 `E+F`，是因为 `h` 和 `v_new` 仅为 core 内部中间量；生产路径不需要公开保存它们，
因此有机会同时消除写回和再次读取。相比之下，`A` 和 `g_cumsum` 仍属于公开输出，先做
`(A+B+C)+D` 只能减少已求解 A 的再次读取，边际 GM 收益更小。

Phase 4 先做一个受限 `E+F` pilot，不直接铺开完整矩阵：

- 不可变基线为 `aclnnGdnCoreFwdPhase3`，局部基线为独立 `fwd_h -> fwd_o`；
- 首个 pilot 仍只验收 `K==V==128` 和外部扩头 GVA，不同时修改 transpose、原生 GVA 或 workspace 别名；
- tiling、调度和中间结构不得把 `V=128` 或 `Hk==Hv` 固化为后续无法扩展的永久契约；
- profiler 必须证明目标段由两个 NPU kernel 降为一个，并证明 `h/v_new` 的 GM 生命周期确实缩短；
- 单点精度或有限性不通过立即停止；生产性能超出噪声回退或没有获得预期收益时，只在该冻结 case
  上最多做三轮有明确假设的单变量优化，仍无解再反馈决策，不进入完整矩阵。

Phase 4 正式收口仍要求完整冻结矩阵相对 Phase 3 不回退，并同时记录 kernel 数、workspace max/sum、
peak delta 和公开输出/state 精度；矩阵必须覆盖非空 initial/真实 final state、dense/varlen 元数据、
尾块、FP16/BF16 与 C64/C128。不能仅凭中间张量理论字节数宣称收益。

### Phase 4 后续规格扩展

`V=256` 和原生 GVA 是后续独立规格，不是 Phase 5 的前置门槛。当前 Phase 4/P1 只验收
冻结的 `V=128` 口径，同时保留扩展所需的 tiling/调度表达能力：

1. `K=128, V=256`；
2. 原生 GVA，即物理 `Hk != Hv` 且 `Hv % Hk == 0`，不依赖外部复制 q/k。

两个规格在产品计划要求正式支持时一次只引入一个变量，分别填写启动卡并使用版本化入口验收。若需要改变已验收 Phase 的
tiling、数据布局或性能行为，必须新增不可变 checkpoint，不能把规格扩展悄悄并入 Phase 4 或后续
suffix 融合。Phase 4 首版虽只验收窄范围，但实现结构不得阻碍这两个闸门。

### Phase 5：在 E+F 上吸收 recompute_w_u

在冻结 `V=128` 验收口径下，Phase 5 的候选边界为：

```text
(A+B+C) + (D+E+F)
```

目标是进一步消除 `w/u` 的 GM 写回和读取。`D` 当前按 chunk 分配、在 chunk 内遍历 value head，
而 `E` 按 `(batch, value_head)` 持有 stream 并顺序推进 chunk；因此不能直接拼接模板，必须先证明
统一 scheduler 后仍满足并行度、state 顺序、MIX 同步和 L1/UB 容量要求。Phase 5 从已验收 Phase 4
和规格 checkpoint 线性追加，不为不同规格预设 `E+F` / `D+E+F` 生产路由分支。

Phase 5 的第一小步是 P0 安全融合：新增版本化入口和 `D + (E+F)` fused kernel，保留 `w/u`
GM hand-off，先验证 D 完成到 HO 消费之间的阶段同步；不把 P1 的 chunk-ready 流水、workspace
别名、`IBSet/IBWait` 或 `ABC` 合并同时带入。P0 通过 smoke、精度和有限性后，才评估是否在同一
边界内继续做流水优化。

Phase 5 P0 checkpoint（2026-07-30）已通过：dense FP16/BF16 C64、dense FP16 C128、
varlen FP16 C64、state FP16 C64 均与 Phase 4 bit-exact 且 finite；`T=128` 配对 median
延迟下降约 4.7%~10.7%，workspace 同步下降约 19~20%，FP16/C64 profiler 任务数为
`7 -> 6`。P1 preflight 进一步确认 fused suffix 为 `64.92 us`，Phase 4 对应 suffix
为 `26.94 + 43.12 us`；当前 `w/u` 仍由 GM workspace 承载并在全核同步后供 H 消费。
P1 随后按三轮规则完成单变量实验：Round1 移除外层冗余 barrier；Round2 将 Phase 5 的 D
展平为 `chunk x value_head` 任务，5 个冻结合同全部 bit-exact/finite，目标 kernel 约改善
`1.8%~2.1%`；Round3 尝试双 scratch 和移除 recompute 内部 barrier，虽然 kernel 更快，
却在 BF16/C64 产生 `8.98e35` 量级输出误差，已拒绝并回退。Round2 的正式完整性能矩阵随后
通过：相对 P0 为 `7/8` case 改善，矩阵 median `-0.581%`，唯一正差 `+0.264%` 判为噪声，
workspace 逐 case 不变。因此 Round2 接受为 Phase 5 P1，不开启 Round4。
后续局部 profiler 补测进一步确认，`(D+E+F)` 相对 Phase 4 的 `D+(E+F)`
在 `T=128/1025` 的中位任务时间分别下降 `21.462%/6.142%`，融合边界本身的收益已被独立证明。

### Phase 6：条件性合并 ABC 与 DEF

只有 profiler 证明剩余 A 重读、launch 和 workspace 生命周期值得继续时，才尝试：

```text
(A+B+C+D+E+F)
```

由于 `A` 和 `g_cumsum` 仍需作为公开输出落盘，`ABC + DEF` 合成单 kernel 的边际 GM 收益天然小于
`D/E/F` 内部融合，而且两侧 scheduler 和同步模型不同。最终生产结构允许停在更快的
`ABC + DEF` 两 kernel 路径；“单 kernel”不是独立验收目标，完整 core latency、显存和稳定性才是。

2026-07-31 使用 `gdn-a2-phase5` clean 归档包完成 profiler 闸门。固定 A2 device 4、dense
FP16、`B=1,Hk=4,Hv=8,K=V=128`，分别测 `C=64/128`、`T=128/1025`，每点三个独立进程。
以下为目标任务的三轮中位数；边界收益只采用稳定 trace，拒绝把共享卡调度空洞算作 launch 收益：

| shape | ABC | 公开 `g_cumsum` transpose | DEF | 设备任务和 | 若吸收 transpose 的任务侧上限 |
| --- | ---: | ---: | ---: | ---: | ---: |
| C64/T128 | 32.861 us | 9.120 us | 54.461 us | 116.482 us | 约 9.8%（含约 2.25 us 稳定边界） |
| C64/T1025 | 104.642 us | 13.060 us | 162.163 us | 303.486 us | 约 5.0%（含约 2.25 us 稳定边界） |
| C128/T128 | 53.761 us | 12.020 us | 51.661 us | 141.063 us | 至少约 8.5%（边界空洞不计） |
| C128/T1025 | 139.723 us | 17.220 us | 130.063 us | 306.826 us | 至少约 5.6%（边界空洞不计） |

四点现有单 ACLNN workspace 依次为 `73,155,072 / 79,010,304 / 78,149,632 /
82,169,856 B`。这些数值只作为 Phase 6 P0 的同设备基线；实际可复用量必须由新入口
实测，不能把 ABC user workspace 的理论字节数直接当作最终降幅。

闸门结论为**带约束 GO**：

1. 仅做 `ABC+DEF`、仍把公开 `g_cumsum` transpose 留在外侧时，稳定 trace 只剩约
   `1~2 us` 的设备边界可消除，性能价值不足，直接判 no-go。
2. Phase 6 P0 必须在一次 launch 内完成 `ABC+DEF`，并把公开 `g_cumsum` 的 BHT->BTH
   写出纳入 kernel；公开输出仍需落 GM，不能把其字节数虚报为已消除。2026-07-31
   独立探针已排除多 AIV 对同一 BTH 行的 4-byte 分 lane 写回，P0a 只能使用 chunk/row
   owner 生成完整 BTH 行后写回。
3. 分 launch 的 ABC 与 DEF workspace 生命周期不重叠；合入单 kernel 后，各核可能异步
   进入 DEF，因此 P0 只能复用已证明安全的 task/core 私有区，不能先验整段别名。绝对
   workspace 继续作为收益项而非独立硬门槛。
4. `T=128` 时 ABC 与 DEF 的活跃核数不同，`T=1025` 时两侧都更接近 24 AIC。
   P0 必须让未参与 ABC 的核也能安全进入跨阶段同步；不得复用已知有共享卡活性风险的
   朴素外层 `SyncAll<false>()` 方案。
5. 第一小步只做 dense FP16、`B=1,logical Hk=4,Hv=8,physical core H=8,T=128,
   K=V=128,C=64` smoke；先验证无卡死、
   `o/A/g_cumsum` bit-exact/finite 和任务数目标，再决定是否进入 BF16、varlen 与性能矩阵。

#### Phase 6 公开 BTH 写回探针

`g_cumsum` 公开布局的主要风险不是模板数，而是多核对同一 32-byte GM block 的
非对齐分 lane 写所有权。A2 device 7 上的 DAV220 探针结果为：

| 模式 | 精度结果 | 结论 |
| --- | --- | --- |
| 每 head 一个 AIV，每 BTH 行 4-byte `DataCopyPad` | `17/137` exact，payload 不一致 `557400`，guard 不一致 `0` | **拒绝** |
| 已是 BTH 的完整行写对照 | `117/117` exact | 可用对照 |
| owner 在 UB 内 BHT->BTH 后写完整行 | `117/117` exact，guard 不一致 `0` | **P0a 选定** |

独立 kernel 在 `H=8, rows=128/1025` 时，owner 相对完整行对照的差值为
`-0.339/-0.149 us`；`rows=4096` 时因每 tile 需要 owner 内部 `PIPE_ALL` 排空，差值为
`+14.694 us`。集成时必须按 chunk 并行分配 owner，不得引入 ABC/DEF 外层全核 barrier；
完整 core 收益仍由 P0c 复验。原始结果、产物哈希和适用边界见
[`evidence/phase6_lane_write_probe_20260731/README.md`](evidence/phase6_lane_write_probe_20260731/README.md)。

#### Phase 6 P0 启动卡

| 项目 | 冻结决定 |
| --- | --- |
| 版本边界 | 新增 `aclnnGdnCoreFwdPhase6`，Phase 5 tag 和入口保持不可变；生产默认入口在 Phase 6 验收前不切换 |
| L0/kernel 边界 | 新增单个 `ChunkGdnCoreFwd` / `chunk_gdn_core_fwd`，输入 raw `g` 与 FP32 `beta` 的内部 BHT view，一次 launch 产出 `o/final_state/A` 和公开 BTH `g_cumsum` |
| 首轮规格 | 仅 A2、dense FP16、`B=1,logical Hk=4,Hv=8,physical core H=8,T=128,K=V=128,C=64`；core 内先冻结外部扩头后的 `Hk==Hv` |
| 调度 | 统一采用 `global_chunk x value_head` 任务编号；同一 MIX core group 完成本组 ABC 后直接消费本组 A 做 D，不新增 ABC/DEF 全核 barrier |
| 同步 | ABC 内只使用配对 AIC/AIV flag；D/H/O 沿用 Phase 5 已验收协议。未分到 ABC 实算任务的核仍走空任务握手并进入后续 DEF，禁止提前 return |
| `g_cumsum` | kernel 同时保留 DEF 所需的 BHT 内部 view；由 chunk/row owner 在 UB 内组织完整 BTH 行并写回公开输出，禁止多 AIV 分 lane 直写；P0 性能候选中不得保留独立 transpose task |
| workspace | 首个无卡死 smoke 允许暂不别名；随后只复用已证明生命周期不重叠的 task/core 私有区。跨核异步阶段不得整段别名，最终 P0 性能/显存验收前必须给出实测 workspace |
| 模板 | 沿用 ABC 的 `dtype x chunk_size` 四个静态选择和 DEF 的 V tiling key；不为 shape/dtype 新增 ACLNN 运行时生产分支 |

执行仍按小步跑，不把所有风险塞进第一次构建：

1. **P0a 编译/活性步：** 新入口、新 op、统一任务编号和 ABC->D producer affinity；由
   chunk/row owner 直接写出完整公开 BTH `g_cumsum` 行。只跑单点超时保护 smoke，
   不测性能，不做跨阶段 workspace 别名。
2. **P0b 正确性步：** P0a 无卡死后，对 Phase 5 检查 `o/A/g_cumsum` bit-exact、finite，
   再补 state 单变量；失败停在该点定位。
3. **P0c 机理步：** profiler 必须看到 core 计算 `ABC + transpose + DEF` 由三任务变为一个
   `ChunkGdnCoreFwd`，完整 ACLNN 设备任务目标 `6 -> 4`；同时核对新 kernel body 是否吞掉
   transpose 收益，不能只看 launch 数。
4. **P0d workspace/性能步：** 只对已证明安全的私有区做别名，然后跑冻结短/长代表点。
   单点没有可复现收益时按三轮规则优化；未过单点前不扩 BF16、C128 或 varlen 矩阵。

#### Phase 6 P0 执行结果

P0a~P0d 已于 2026-07-31 按上述顺序完成，结论为 **P0 接受、继续扩规格**：

1. `ChunkGdnCoreFwd` 与 `aclnnGdnCoreFwdPhase6` 已进入完整 A2 run 包；owner 完整行
   BTH 写回替代了独立公开 transpose task。
2. `T=1025` 首探针暴露 ABC 连续生产与 recompute 跨核步进消费之间的 `A` 竞争。
   边界 `SyncAll` 只用于因果验证；最终改为 `kAbcTaskOrder=true` 时复用 ABC 连续核内
   分片，同核生产/消费，不保留 ABC/DEF 全核 barrier。
3. dense FP16/C64 `T=128/1025` 无 state 与 state 共 `4/4` 对 Phase 5 bit-exact、
   `max_abs=0` 且全部有限。最终 profiler 设备任务数 `6 -> 4`，目标段
   `102.262 -> 84.162 us`。
4. 正式同进程配对性能每规格三个独立进程、每进程 `20 warmup + 200 iterations`；短点
   三轮收益中位数 `3.023%`，长点 `2.413%`，六轮全部改善。workspace 分别为
   `73,155,072 -> 74,371,584 B` 和 `79,010,304 -> 83,140,608 B`，按性能优先原则接受。
5. 2026-08-01 按 `dense BF16/C64 -> C128 -> varlen -> state` 小步顺序完成完整规格：
   dense/varlen 的 FP16/BF16 x C64/C128 均对 Phase 5 bit-exact/有限；varlen 主 state
   合同 `4/4` 及两个 state 边界合同均通过。varlen 的 H/O 可见性问题通过仅 varlen 的
   H 末尾同步修复，不重新引入 ABC/DEF 边界全核同步。
6. 正式性能使用 A2 device 2、同进程 AB/BA、每 identity 三轮稳定轮次和 600 个 pooled 样本；
   8/8 identity 改善，矩阵 pairwise median `-2.733%`。workspace 逐 case 上升，按性能优先、
   不新增运行时规格分支的既定规则接受。
7. clean 源树、run 包隔离安装、wheel 隔离安装、wheel ABI/runtime 和 Demo composite 回归全部通过；
   最终 `ChunkGdnCoreFwd` 包含 `4 .o + 4 .json`。用户确认后已以 `gdn-a2-phase6` 完成
   Git 归档；默认入口仍保持 Phase 2。详细证据见 [`GDN_PHASE6_ACCEPTANCE_A2.md`](GDN_PHASE6_ACCEPTANCE_A2.md)。

### Phase 7：transpose/layout 收敛

核心计算边界稳定后，再单独评估输入 `g/beta`、公开 `g_cumsum` 和输出 `o` 的布局适配。每次只移动
一条 layout 边界，并用 profiler 证明 transpose kernel 或 GM 流量真实减少，不在同一步同时修改数学
计算、原生 GVA 或 `V=256`。

### Phase 8：纳入 causal_conv1d、RMSNorm 和门控

完整 Demo 的顶层融合还包括：

```text
causal_conv1d -> GDN core -> RMSNorm -> gated SiLU
```

纳入顺序由公开中间量、producer/consumer 调度兼容性和完整 Demo profiler 决定；不为了算子数量
破坏已经更快的 core 拓扑。输出侧 RMSNorm/gate 与输入侧 causal conv 必须分别启动和验收。

### Phase 9：完整 Demo/模型收口

最后在同一安装包、同一输入和同一 device 上比较完整 Demo/模型路径，关闭功能、精度、生产性能、
绝对/相对 workspace、peak 和稳定性口径。Backward 暂不并入 forward mega-kernel，但必须保留现有
反向链路回归，不能把 forward 收益建立在破坏保存张量或梯度契约之上。

## 4. transpose 策略

第一轮真正融合不跳过 transpose，也不把 transpose 和 KKT/solve_tri 同时修改。

暂时采用：

```text
外部输入布局
    -> ACLNN wrapper 中完成必要 transpose/contiguous
    -> fused KKT + solve_tri 使用已验证的内部 head-first 布局
    -> 保持现有输出布局
```

这样可以隔离早期融合变量。Phase 4--6 保持现有边界；只有 suffix/core 计算稳定并关闭规格闸门后，
Phase 7 才单独评估：

- fused kernel 是否直接接受外部布局
- 是否可以消除 transpose
- 是否可以把 transpose 与前后阶段合并

## 5. 消融和验收规则

开始任何新 Phase 前，必须先按 `GDN_FUSION_DEVELOPMENT_PLAYBOOK_A2.md` 填写阶段启动卡，并执行其规定的分层构建、版本哈希、精度和验收流程。

### 5.1 Phase 版本固化规则

从 2026-07-25 起，每个进入验收的 Phase 必须保留独立、可构建、可调用的版本化 ACLNN，禁止在同一个 ACLNN 实现上原地覆盖上一 Phase：

| Phase | 固定 ACLNN | 固定计算边界 |
| --- | --- | --- |
| Phase 1 | `aclnnGdnCoreFwdPhase1` | 一个 executor 内调度六个独立 kernel |
| Phase 2 | `aclnnGdnCoreFwdPhase2` | 用 `ChunkKktSolveTri` 替换独立 KKT + solve_tri |
| Phase 3 | `aclnnGdnCoreFwdPhase3` | 用 `ChunkCumsumKktSolveTri` 替换 local_cumsum + KKT + solve_tri |
| Phase 4 | `aclnnGdnCoreFwdPhase4` | 在 Phase 3 上用 `ChunkGatedDeltaRuleFwdHO` 替换独立 fwd_h + fwd_o |
| 当前默认 | `aclnnGdnCoreFwd` | 兼容入口，当前指向 Phase 2，可随验收阶段前移 |

Phase 5 及后续规格 checkpoint 必须继续新增版本化入口，不能修改 Phase 1/2/3/4
的阶段调度来承载新边界。版本化入口
必须同时存在于同一个完整验收包中，使用同一输入、同一设备和同一 benchmark 直接 A/B；不得用
不同安装包的先后测试冒充同环境对照。

### 5.2 Phase 2 性能对照的进程隔离例外

`V_BF16_C64` 的因果 A/B 已证明：Phase 1/2 各自在干净进程中连续 `10/10 PASS`，但 Phase 2
释放 workspace 后，Phase 1 若在同一进程复用同址的非零 workspace 会触发 `507015`/AIV MTE
越界；迫使 Phase 1 换址或清零 Phase 1/2 workspace 重叠区后故障消失。这是 Phase 1 对 workspace
初值的未声明依赖，继续用同进程交替会把旧路径生命周期缺陷混入 Phase 2 性能门禁。

因此 Phase 2 剩余性能收口采用以下受控例外：Phase 1/2 仍使用同一个完整安装包、同一输入、
同一 device、同一环境和同一 benchmark 参数，但每个 variant 在干净进程中分别测量；case 级
执行顺序按 `Phase 1 -> Phase 2` 与 `Phase 2 -> Phase 1` 进程级换序，固定 warmup/iteration，
分别记录 median、P90 和 min，用重复批次检查设备漂移。该例外只改变进程生命周期，不允许
更换构建产物、输入 contract 或计时方法。旧基线非有限的用例只使用 Phase 1 latency，Phase 2
精度必须独立检查有限性并与 CPU FP64 或其他已验证高精度参考比较。

版本化入口只允许修复已证明影响该版本正确性、构建性或 ABI 的缺陷。修复前后必须归档原因和回归结果；性能优化或新融合边界一律进入新 Phase。源代码提交或归档点、安装包 SHA256、验收报告共同构成 Phase 快照，只有函数改名而没有测试和版本证据不算归档完成。

每个融合阶段都必须在同一个 benchmark 中增加命名 variant，并与 Phase 0 六算子基线比较。不能为每个版本创建互不兼容的测试脚本。

除版本化 ACLNN 外，每个 Phase 还必须在 `chw/gdn-a2-phase-archive` 上形成一个只追加、不改写的里程碑 commit。下一 Phase 新增 commit，不 amend/rebase/force-push 已归档 Phase；commit SHA、验收 wheel/OPP SHA256 和报告一起登记到 `GDN_PHASE_VERSION_ARCHIVE_A2.md`。Phase 1/2 因规则建立前共处同一未提交工作区，由首个 Phase 2 归档 commit 共同固化；Phase 3 起严格一 Phase 一新增 commit。

最低验收维度：

| 维度 | 要求 |
| --- | --- |
| 数值 | 输出和关键中间结果达到 L1；有效三角区域比较，忽略尾块无效 padding |
| 功能 | dense、varlen、尾块、FP16、BF16、GVA |
| 规格 | `K=128`，第一阶段 `V=128`；后续 `V=256` |
| chunk | `64/128` |
| 性能 | 全场景不劣于六算子基线，重点看端到端 latency |
| 显存 | 完整记录 workspace max/sum、相对增量和 peak；额外 50 MB 为优化目标，不是压过性能的绝对硬门槛 |
| 稳定性 | 至少 warmup 后多次测量，记录均值、P90 和异常情况 |

每次融合只有在以下条件同时满足后才能进入下一阶段：

1. 精度通过。
2. 目标场景功能通过。
3. 性能没有全场景回退，或回退有明确且获批准的取舍。
4. workspace、同步和尾块行为已解释清楚。
5. benchmark 结果和变更说明已归档。

后续 Phase 默认只有一条生产路线，不按 dtype、chunk、layout 或 shape 预设分支。版本化 ACLNN 只用于同包 A/B 和不可变归档。性能门禁不预设统一百分比：冻结用例相对上一已验收 Phase 的稳定基线若出现超出测量噪声的回退，或没有达到启动卡预期收益，则在同一路线上最多执行三轮有 profiler 假设的单变量优化。单纯复测和确认噪声不计轮次；三轮仍无解时，带齐基线差距、瓶颈、已尝试方案和继续优化成本反馈决策，不自动扩展分支。

## 6. 当前边界和不混淆事项

- 平台是 A2，不是 A5。
- 第一阶段先做 `K == V == 128`，这是阶段性范围，不是 GDN 数学上的永久限制。
- `V=256` 需要后续扩展状态、w/u、h、final state 和 fwd_o 的 shape/tiling。
- 统一 ACLNN 入口不代表单 kernel 融合。
- `GVA` 当前复合入口可以依赖外部 q/k head 扩展，但最终融合应原生处理 `Hk != Hv`，避免复制带来的额外流量。
- TND 是变长逻辑布局；当前 wrapper 可能使用物理 `B=1` 的 head-first rank-4 张量并通过 `cu_seqlens` 表达逻辑 batch。
- RMSNorm 不在当前六个 GDN core 小算子列表中，但属于最终完整 Demo 融合目标。
- backward 暂不作为首轮融合目标；先稳定 forward，保持现有 backward 链路。

## 7. 冻结里程碑摘要（非当前进度源）

本节只保留路线与冻结 Phase 的对应关系。当前 Phase、下一小步和 blocker 统一由
`GDN_CURRENT_STATUS_A2.md` 维护。

截至 2026-07-24，Phase 1 已通过 A2 验收：

- GDN core 复合入口 A/B：`6/6 PASS`，覆盖 dense/varlen、FP16/BF16、尾块、`chunk_size=64/128`
- 完整 Demo 复合入口 A/B：`2/2 PASS`，覆盖 dense 与等长 varlen 的输出和梯度
- Phase 1 因此关闭，后续默认从 Phase 2 开始
- Phase 1 的六独立 kernel 复合调度已恢复并固化为 `aclnnGdnCoreFwdPhase1`；不再由通用入口的当前实现隐式代表

截至 2026-07-25，Phase 2 已按冻结范围完成实现、功能/精度和生产性能门禁：

- 独立 `ChunkKktSolveTri` A/B：`6/6 PASS`，覆盖 dense/varlen、FP16/BF16、`chunk_size=64/128`、多 batch/头数和尾块
- 4 组端到端 GDN core 消融场景通过；输出、`g_cumsum` 和有效 `A` 区域与基线一致
- 4 个代表 case 中，Phase 2 局部阶段 median latency 下降 `45.2%–57.3%`，完整 core 五调用路径相对六调用 median 改善 `1.2%–27.2%`
- profiler 确认 KKT + solve_tri 的 NPU kernel 数由 3 降为 1，完整 core 路径由 12 降为 10
- 局部融合 ACLNN 最大 workspace 为 `24,252,928` bytes；完整 core 峰值分配较基线下降 `8.2%–9.0%`
- 关闭 `ASCEND_LAUNCH_BLOCKING` 后，dense/varlen 的 FP16/BF16 × C64/C128 交叉点和 `B=4,H=4,T=4096` 扩展点的 Phase 2 median 均无可复现回退；varlen standalone 的 Phase 2 输出/state 均通过独立有限性检查
- `T=32768` varlen 上 Phase 2 单路径输出有限且 median 为 `6.911 ms`；Phase 1 首次同步 MTE 越界，故该点的相对性能/显存不可判定，不扩入 Phase 1 修复
- 生产可比较 case 的最大相对增量为 `8.86 MB` workspace / `8.39 MB` peak；单 ACLNN 绝对 workspace `<=50 MB` 口径未满足，需与相对增量口径区分
- 完整结果见 `GDN_PHASE2_ACCEPTANCE_A2.md`

截至 2026-07-26，Phase 3 已按累积融合修正卡完成实现和 A2 验收：

- 最终累积边界为单 `ChunkCumsumKktSolveTri`：raw FP32 cumsum + KKT + low-precision hand-off + solve_tri；
- 独立 `ChunkCumsumKkt` 的最终共享 helper 完成 `80/80 exact`，core dense/varlen `8/8` 加 state
  `1/1` 均对不可变 Phase 2 bit-exact/有限；
- 共享 helper 局部微基准 `8/8` median 改善；作为最终结论的完整 GDN core Phase 2/3 生产性能也是 `8/8` median 改善；
- profiler 证明完整 core NPU kernel 数 `9 -> 8`，目标段由两个 kernel 合并为一个；
- 完整 run 包 SHA256 为 `837742c6731143ec0ea55c517338e0daca3d3f295b9b7a71f079805b9b62bfdb`，
  安装 host 库 SHA256 为 `645336f9f65c522a06d74cf8b3df0ca6db2ae493fcd6fec980719eca7c8af07f`；
- `aclnnGdnCoreFwdPhase1/2/3` 与默认入口可在同一包内并存，默认入口仍保持 Phase 2 兼容行为；
- 完整结果见 `GDN_PHASE3_ACCEPTANCE_A2.md`。

Phase 3 的正式文件清单、归档前门禁、实现与验收里程碑 commit、不可变 annotated tag、归档
branch 和远端逐 SHA 回查均已完成。`gdn-a2-phase3^{commit}` 固定为
`7fb8f05b59ab56a8392e0f6c9bef071714894826`，tag object 固定为
`b26159171f8aa0b1340f3f927412795853dc72e9`；Phase 3 已标记为 Git 归档完成。Phase 4 只能从该
不可变里程碑追加新 commit 和新版本化入口，不得原地改写 Phase 3。

截至 2026-07-28，Phase 4 已按冻结 `K==V==128` / 外部 GVA 范围完成 A2 验收：

- 最终边界为 `(A+B+C) + D + (E+F)`，`E+F` 由单 `ChunkGatedDeltaRuleFwdHO` MIX kernel 实现；
- core dense/varlen `8/8` 加 state `1/1` 与 Phase 3 bit-exact/有限；
- dense/varlen `8/8` 生产性能点无可复现实质回退，varlen 四点和 dense C64 改善；
- 密集 C128 配对 median 差异小于 `0.7%`，同时 C128 目标 kernel 由 `115.52 us`
  降至 `112.24 us`，判为测量噪声范围；
- core NPU 任务数 `8 -> 7`，所有性能点 workspace 下降 `24.11%~27.89%`；
- 完整 run 包 SHA256 为 `a297168f3b5d14a09afd23acd060d3ab546bab9cbd71e7c8d301ebe3ce9b9206`，
  安装 host 库 SHA256 为 `6f67757282030b90f95f92f403beaf1a3bdeeee9cd52ccf9de87f58226c5a23d`；
- 完整结果见 `GDN_PHASE4_ACCEPTANCE_A2.md`。

截至 2026-07-29，Phase 4 流水 P1 在不改变上述融合边界的前提下完成独立 checkpoint：

- dense/varlen `8/8` 加 state `1/1` 对 Phase 3 bit-exact/有限，额外压力点通过；
- 8 点相对 Phase 3 的 median 变化为 `+0.94%/-0.88%/-1.63%/-2.14%/+0.45%/-0.34%/-1.81%/+0.20%`；
- workspace 8 点全部下降 `24.11%~27.89%`，代表 profiler 目标段均更快；
- 完整结果见 `GDN_PHASE4_PIPELINE_P1_ACCEPTANCE_A2.md`，P1 使用新 tag，Phase 4 原始快照不变。

Phase 4/P1 归档后的 Phase 5 P0 已在冻结 `V=128` 口径下完成 `D+E+F` 安全融合验收。
同一生产边界内的 P1 三轮优化也已结束：Round2 已通过功能、精度和完整性能矩阵并接受为
Phase 5 P1，Round3 因 BF16 正确性失败已回退。Phase 5 已以 `gdn-a2-phase5`
完成 Git/交付归档。Phase 6 随后完成 `ABC+DEF` 单 launch、owner 内置公开 `g_cumsum`
写回、ABC/recompute 同核连续 task 分片、dense/varlen/state 全合同、正式性能矩阵和 clean
wheel 交付回归；用户确认后已以 `gdn-a2-phase6` 完成独立 Git 归档，默认入口保持不变。
`V=256` 和原生 GVA 在产品计划要求时分别启动独立规格闸门，不与当前 Phase 6 规格扩展混合进行。

## 8. 相关文件

- 总体背景与规格：工作区外部任务说明 `gdn.md`（不纳入仓库）
- 统一 ACLNN 接口说明：`fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/docs/aclnnGdnCoreFwd.md`
- 消融 benchmark 说明：`fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/docs/gdn_core_ablation.md`
- 消融脚本：`torch_custom/fla_npu/test/benchmark_gdn_core_ablation.py`
- GDN Python 入口：`examples/flash_gated_delta_rule.py`
- Phase 2 A2 验收报告：`fla/ops/ascendc/gdn/docs/GDN_PHASE2_ACCEPTANCE_A2.md`
- Phase 3 A2 验收报告：`fla/ops/ascendc/gdn/docs/GDN_PHASE3_ACCEPTANCE_A2.md`
- Phase 6 P0 A2 验收报告：`fla/ops/ascendc/gdn/docs/GDN_PHASE6_ACCEPTANCE_A2.md`
- Phase 4 流水 P1 A2 验收报告：`fla/ops/ascendc/gdn/docs/GDN_PHASE4_PIPELINE_P1_ACCEPTANCE_A2.md`
- A2 GDN 融合开发手册：`fla/ops/ascendc/gdn/docs/GDN_FUSION_DEVELOPMENT_PLAYBOOK_A2.md`
