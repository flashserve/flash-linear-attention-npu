# Chunk KDA Forward Prepare 内核伪代码

本目录是 A5 `chunk_kda_fwd_prepare` 的可检视、可用 host C++17 做语法检查的设计伪代码。
文件命名和 kernel 侧分层参考 `chunk_fwd_h`，但本目录**不参与构建**，不会创建算子定义、
Host Tiling、CMake target、aclnn/Python API 或设备 kernel ABI。

`VectorOps`、`CubeOps`、`SyncLedger` 中类似 Ascend C API 的名字都只是符号化数据流标记，
不代表已经确认目标 CANN 的函数重载、内存通路、event mode、flag ID 或同步能力。
所有这类边界均明确标为 **PROPOSED**。

## 文件结构

```text
pseudocode/
|-- README.md
|-- chunk_kda_fwd_prepare_tiling_key.h  # 固定维度与候选模板轴
|-- chunk_kda_fwd_prepare_policy.h      # A5 UB/L1/workspace 固定资源账本
|-- chunk_kda_fwd_prepare_struct.h      # task、buffer、同步公共类型
|-- chunk_kda_fwd_prepare_utils.h       # 分核、head owner、代际映射
|-- chunk_kda_fwd_prepare.cpp           # MIX AIC/AIV 符号入口与调度
|-- arch35/
|   |-- chunk_kda_fwd_prepare_vec.h     # V0/V1/V3/V6
|   `-- chunk_kda_fwd_prepare_cube.h    # C2/C4/C5/C7
`-- arch22/
    |-- README.md
    |-- chunk_kda_fwd_prepare_vec.h     # 明确不可调用的占位声明
    `-- chunk_kda_fwd_prepare_cube.h    # 明确不可调用的占位声明
```

## 八个物理 Stage

一个 `(sequence, chunk, head)` 事务固定为八个物理 Stage：

| Stage | 类型 | Stage 入口完整输入 | 计算与提交结果 |
| --- | --- | --- | --- |
| `V0` | Vector | UB 中完整 `q/k/beta`，以及所选 gate 模式要求的预计算 step，或必需 `A_log` 与可选 `dt_bias` | 一次 VF 完成 Q/K norm、beta、gate/cumsum、`G_ref[4]`；workspace context 只导出 `Qhat/Khat/G`，`betaEff` 常驻 AUX |
| `V1` | Vector | 仍驻留在同一 UB 的 V0 local source | 一次 VF 先按 `ScoreStorage` 截断 base-2 指数，再物化全部 `S=4` 的 `Qplus/Kplus/Kminus`，一次 MTE3 drain 发布 packed score |
| `C2` | Cube | 一次把完整 72 KiB score 搬入 L1 | 四个 row band，每个 band 两个互不依赖 MMAD，输出 `rawAqk/rawAkk` |
| `V3` | Vector | UB 中完整 raw score 与保留的 `betaEff` | 一次 VF 完成无效区清零、causal mask、`Aqk/Lkk` 和 VCS 两个叶子 `B/X0/X1` |
| `C4` | Cube | L1 中完整 VCS payload | 一个 MMAD 计算 `T = B @ X0`；`X1` 与稳定 Akk quadrant 直接放最终 resident 地址 |
| `C5` | Cube | 前一 Stage 的 `T`、`X1`、Akk prepack 均 ready | 一个 MMAD 计算 `Y = -X1 @ T`，直接提交到最终 Akk quadrant |
| `V6` | Vector | `Qhat/Khat/G` context reload、AUX 常驻 `betaEff`、`V` 以及三条 data-ready 边 | 一次 VF 以 `[-80,80]` 截断 direct base-2 指数并生成 `Qg/qg`、`kg`、`K_beta_g`、`V_beta`，一次 drain 发布 RHS |
| `C7` | Cube | 完整 Akk resident 与完整 RHS | 一个逻辑 MMAD 计算 `Akk @ [K_beta_g | V_beta]`，提交 `[W | U]` |

每个 Stage 只包含 Cube 或 Vector 之一。`V0/V1/V3/V6` 各只允许一次 VF，不按 token、
score block 或硬件 tile 分 pass。Cube 的独立逻辑 MMAD 可在编译期展开为硬件 tile，但本 Stage
全部操作数必须在 Stage 入口前就绪，不能在同一 Cube Stage 内读取本 Stage 新产生的结果。

八 Stage 是当前约束下的下界：

- `V0+V1` 必须同时保留 context 和 72 KiB score，而单 local head 只有固定
  `112 KiB MAIN + 12 KiB AUX`，容量不成立。
- `C4` 产生 `T`，`C5` 才能消费；二者不能合并。
- `C7` 既依赖 C5 新完成的 Akk，又要汇聚独立的 V6 RHS；`C5/C7` 不能合并。
- `V6` 必须等 V3 归还 local bank 且 C4 归还 payload owner，不能提前覆盖任一存活区。

V0 的核心数学不能被一个含糊的 `ApplyFrozen*` 隐藏。候选 TilingKey 必须显式携带以下
**PROPOSED** 语义轴，但当前伪代码不冻结它们的数值编码：

- `QkNormMode::{Identity,L2}`；L2 使用 runtime `epsilon`。
- beta 是每 token 一个 FP32 标量；L2 公开入口即使接受 BF16/FP32，也必须在下发 Prepare 前统一
  cast 到 FP32，所以 beta storage 不是 kernel 模板轴。full chunk 逻辑输入为 256 Byte，可放入
  0x200 Byte 的对齐 AUX region；剩余字节只是 hard pad。
  `BetaMode::{Raw,Sigmoid,TwoSigmoid}` 只决定该 FP32 标量的数学变换。
- `GateMode::{PrecomputedStep,Softplus,SafeSigmoid}`；后两种按冻结公式消费必需 `A_log`、
  runtime `lowerBound` 和可选 `dt_bias`。`hasDtBias=false` 必须把最终 AUX 地址显式置零，且不能
  构造或读取虚假的 GM source。
- `InputStorage::{Fp16,Bf16}` 显式选择 `q/k/v` 的共同公开 2-byte dtype；三者不能混用。
  `ScoreStorage::{Fp16,Bf16}` 单独描述内部 `Qplus/Kplus/Kminus` 的真实 dtype，不能用含糊的
  `TwoByte` 代替。独立的 `safeGate` 模板属性描述现有 SAFE_GATE specialization，它与决定 V0
  gate 数学公式的 `GateMode` 正交，不能用 `GateMode::SafeSigmoid` 推导。Host 必须校验允许的
  映射：`safeGate=false` 时 `ScoreStorage` 必须等于 `InputStorage`；`safeGate=true` 且 input 为
  FP16 时 score 必须提升为 BF16，即使 `GateMode::PrecomputedStep` 也一样；BF16 input 始终只能
  配 BF16 score。`(Bf16,Fp16)` 和 SAFE_GATE `(Fp16,Fp16)` 均拒绝。
- V1 的 `Exp2` 输入是 base-2 指数，必须先做 dtype 相关截断：BF16 score 为 `[-126,120]`，
  FP16 score 为 `[-80,80]`。V6 的 direct `G`、`G_last-G` 与输出 dtype 无关，固定截断到
  `[-80,80]`。单 VF 内的 symbolic `Exp2Clamped` 只冻结此数学次序；具体 `Mins/Maxs/Exp` API 与必要的
  `PIPE_V` 顺序仍是 **PROPOSED**，不能省略截断或把两个范围混为一个。
- 每次写入 2-byte storage 前都使用 RINT 语义。目标是 FP16 时，先把 FP32 中间值饱和到
  `[-65504,65504]` 再舍入；目标是 BF16 时只做 BF16 舍入，不做这一步有限幅值饱和。V1 的目标
  由 `ScoreStorage` 选择；V3/V6 以及 C5/C7 的 Fixpipe 目标由 `InputStorage` 选择。symbolic
  `RoundToScoreStorage`、`RoundToInputStorage` 与 `CubeOps::StoreRounded` 必须保留该区别，普通
  `Store` 只表示同 dtype 搬运，不能暗含 cast、饱和或舍入。
- V6 的 `K_beta_g` 保留现有 Prepare 的两个真实存储舍入点，不能合并为末尾一次 cast：
  `kGateRounded = RoundToInputStorage(ClampForInputStorage(kHat * exp2(G)))`，随后把
  `kGateRounded` 转回 FP32 与 `betaEff` 相乘，再执行第二次
  `RoundToInputStorage(ClampForInputStorage(betaEff * fp32(kGateRounded)))`。FP16 的两次
  `ClampForInputStorage` 都是上面的 `+/-65504` 饱和；BF16 两次 clamp 都是 no-op，但两次 BF16
  舍入仍必须存在。
- FUSED `Qg_scaled` 同样保留两个存储边界：先得到
  `qgStorage = RoundToInputStorage(ClampForInputStorage(qHat * exp2(G)))`；Current Qg 输出消费该
  已舍入值。FUSED 再把 `qgStorage` 转回 FP32 做 scale，并产生
  `qgScaledStorage = RoundToInputStorage(ClampForInputStorage(scale * fp32(qgStorage)))`。禁止直接
  scale 未舍入的 FP32 `qHat * exp2(G)`，否则 FUSED 与现有 `StorePreparedQG` 数值顺序不一致。
- runtime `scale` 在 V3/V6 的既定 cast/scale 边界使用，不能被某个 ABI 重复应用。

Host 必须在 launch 前验证 `epsilon/lowerBound/scale` 有限，L2 的 epsilon 为正，并按所选
gate/beta 模式检查 tensor presence、shape/dtype/layout 是否匹配。StageArgs 直接透传这三个标量和
`hasDtBias`；它们不是 runtime 分支猜测出来的常量，也不进入尚未冻结的数值 TilingKey 编码。

每个实际 chunk 的 `validRows` 必须在 `[1,64]`；空 sequence 不产生 chunk。Host 遇到非法
descriptor 必须拒绝，符号入口也会在进入 V0 前跳过，避免 V6 的 `validRows-1` 下溢。

## S=4 causal-prefix 72 KiB

full chunk 的 `b_s = {16, 32, 48, 64}`，score payload 固定为：

```text
Qplus[64,128]                  16 KiB
Kplus[64,128]                  16 KiB
Kminus_0[16,128]                4 KiB
Kminus_1[32,128]                8 KiB
Kminus_2[48,128]               12 KiB
Kminus_3[64,128]               16 KiB
total                           72 KiB
```

`C2` 对每个 active band `s` 执行两次互不依赖的
`[16,128] x [128,b_s]` MMAD，分别生成对应行的 `rawAqk` 和 `rawAkk`。
每份 prefix 恰好包含 full `Kminus[64,128]` 公式在有效列读取的 key，因此有效 raw 结果完全一致。
`[b_s,64)` 没有存储，也不是隐式零；`V3` 必须在任何 mask/VCS reader 之前直接清零这些区域，
再处理当前 16-row block 内的 `j > i` causal mask。

`C2` 在 Stage 入口只搬一次完整 72 KiB。遍历四个编译期 band 只是同一 Cube Stage 内的独立
MMAD 宏格，不构成 GM 重读或语义 pass。

## 固定物理布局

下表中的地址都是 owner 内的 byte half-open range。hard pad/reserve 不能借给另一个 head、
generation 或未登记 scratch；Stage overlay 只能在上一语义的最后异步 reader 完成后原址改名，
不能通过 UB/L1 内搬位整理碎片。

### UB：每个 AIV 248 KiB

| AIV 绝对 UB range | 大小 | 固定 owner |
| --- | ---: | --- |
| `[0x00000,0x1C000)` | 112 KiB | local head 0 `MAIN` |
| `[0x1C000,0x38000)` | 112 KiB | local head 1 `MAIN` |
| `[0x38000,0x3B000)` | 12 KiB | local head 0 `AUX` |
| `[0x3B000,0x3E000)` | 12 KiB | local head 1 `AUX` |

每个 local head 的 MAIN 相对布局按 Stage 改变语义，但地址不移动：

| Stage/template | `[0x0000,0x1C000)` 的完整 MAIN 分段 |
| --- | --- |
| `V0/GATE_2B` | `Qhat[0000,4000) Khat[4000,8000) gate->G[8000,C000) G[C000,14000) work[14000,1C000)` |
| `V0/GATE_FP32` | `Qhat[0000,4000) Khat[4000,8000) gate->G[8000,10000) work[10000,18000) hard[18000,1C000)` |
| `V1/GATE_2B` | `Q+[0000,4000) K+[4000,8000) K-0[8000,9000) K-1[9000,B000) hard[B000,C000) live-G[C000,14000) K-2[14000,17000) K-3[17000,1B000) hard[1B000,1C000)` |
| `V1/GATE_FP32` | `Q+[0000,4000) K+[4000,8000) live-G[8000,10000) K-0[10000,11000) K-1[11000,13000) K-2[13000,16000) K-3[16000,1A000) hard[1A000,1C000)` |
| `V3` | `rawAqk/Aqk[0000,4000) rawAkk[4000,8000) leaf0[8000,9000) leaf1[9000,A000) B[A000,B000) X0[B000,C000) X1[C000,D000) work[D000,E000) optional-pack[E000,10000) hard[10000,1C000)` |
| `V6` | `Qhat->Qg[0000,4000) Khat->kg[4000,8000) V->Vbeta[8000,C000) G-input[C000,14000) KbetaG[14000,18000) VF-scratch[18000,1C000)`；`Qg_scaled[C000,10000)`只在该行全部 G reader 完成后原位覆盖 |

每个 local head 的 AUX 相对布局固定为：

| AUX range | 大小 | 语义与生命周期 |
| --- | ---: | --- |
| `[0x0000,0x0200)` | 512 B | FP32 `betaRaw` 有效 256 B，其余 hard pad |
| `[0x0200,0x0400)` | 512 B | FP32 `betaEff` 有效 256 B；V0 产生后原址常驻到 V6 最后读取 |
| `[0x0400,0x0C00)` | 2 KiB | 四个 512 B `G_ref`；V0 早期可依次解释为 `dt_bias/A_log`，最后 reader 后才 overlay |
| `[0x0C00,0x0E00)` | 512 B | scan carry |
| `[0x0E00,0x1000)` | 512 B | `G_last` |
| `[0x1000,0x3000)` | 8 KiB | 连续 VF scratch |

### L1：每个 AIC 512 KiB

| AIC L1 range | 大小 | 固定 owner |
| --- | ---: | --- |
| `[0x00000,0x12000)` | 72 KiB | group-local head 0 current lane |
| `[0x12000,0x24000)` | 72 KiB | group-local head 1 current lane |
| `[0x24000,0x36000)` | 72 KiB | group-local head 2 current lane |
| `[0x36000,0x48000)` | 72 KiB | group-local head 3 current lane |
| `[0x48000,0x5C000)` | 80 KiB | 四 head 的跨 Cube Stage resident |
| `[0x5C000,0x80000)` | 144 KiB | hard reserve；未登记 tensor 不得占用 |

每条 72 KiB current lane 在 C2 中是完整 score：`Q+[0000,4000)`、`K+[4000,8000)`、
四个 causal prefix `K-0[8000,9000)`、`K-1[9000,B000)`、`K-2[B000,E000)`、
`K-3[E000,12000)`。C4 将同一 lane 改名为 VCS payload，仅使用
`X0[0000,1000) X1[1000,2000) B[2000,3000) X0tau[3000,3800)
X1tau[3800,4000) q01zero[4000,4800)`；`[4800,12000)` hard unused。C7 再把 lane
改名为 `KbetaG[0000,4000) Vbeta[4000,8000)`，`[8000,12000)` hard unused。

当前可达 `TwoByteAbi` 的 resident 为四份 `X0[48000,4C000)`、四份
`X1[4C000,50000)`、四份 `T[50000,54000)` 和四份 `AkkTau[54000,5C000)`；单 head
stride 分别为 4/4/4/8 KiB。`Fp32Internal` 的容量候选把同一区间解释成四份
`AkkFp32[48000,58000)` 与四份 `T[58000,5C000)`，但该模板是 BLOCKED，不可 dispatch。

单 head 的 8 KiB `AkkTau` 是 2x2 tight quadrant pack：相对 resident base 的
`q00[0x0000,0x0800)`、`q01[0x0800,0x1000)`、`q10[0x1000,0x1800)`、
`q11[0x1800,0x2000)` 各为紧凑 32x32 两字节矩阵。它不是一个 row-major `64x64` BufferSpan。
C7 的 `MmadQuadrantPackedLhs` 只符号化以下合同：MTE1 从四个最终 L1 象限地址直接装入逻辑
L0A 四象限后参与 MMAD，不先在 L1 拼成 row-major，也不做任何 L1 位置移动。目标 A5 是否支持
该装载/计算形式及其精确 API、事件和 tile 描述仍是 **PROPOSED** 硬门禁。

### L0C：按 Stage overlay 的候选账本

L0C 不跨 Stage 常驻，但四个 group-local head 可以同时处于同一 Cube Stage；每个异步 MMAD
结果在对应 Fixpipe 最后读取前必须有独立物理 span。因此固定四条互不重叠的 head lane：

```text
headLaneBase(h) = h * 0x10000, h in [0,4)
headLaneBytes   = 0x10000       # 64 KiB
requiredBytes   = 4 * 0x10000   # 256 KiB
```

C2 在每条 lane 内使用紧凑 `16 x N` FP32 结果；`N={16,32,48,64}` 时每个结果真实为
1/2/3/4 KiB。下面都是相对 `headLaneBase(h)` 的 half-open range：

| C2 lane-relative range | 大小 | owner |
| --- | ---: | --- |
| `[0x0000,0x0400)` | 1 KiB | rawAqk，prefix 16 |
| `[0x0400,0x0C00)` | 2 KiB | rawAqk，prefix 32 |
| `[0x0C00,0x1800)` | 3 KiB | rawAqk，prefix 48 |
| `[0x1800,0x2800)` | 4 KiB | rawAqk，prefix 64 |
| `[0x2800,0x2C00)` | 1 KiB | rawAkk，prefix 16 |
| `[0x2C00,0x3400)` | 2 KiB | rawAkk，prefix 32 |
| `[0x3400,0x4000)` | 3 KiB | rawAkk，prefix 48 |
| `[0x4000,0x5000)` | 4 KiB | rawAkk，prefix 64 |

C2 每 head 同时存活 20 KiB，四 head 合计 80 KiB；但不能把它们密排成单个 80 KiB owner，
因为同一条物理 lane 还要服务后续 C7。C4 把本 head lane 的 `[0x0000,0x1000)` 用作 `T`，
C5 保留该输入并把 `[0x1000,0x2000)` 用作 `Y`；只有前一 Stage 的最后 Fixpipe reader 完成后，
后续 Stage 才能 overlay C2 的旧语义。

C7 在每条 lane 内同时保留 `W[0x0000,0x8000)` 与 `U[0x8000,0x10000)` 两个独立 32 KiB
目标，避免 U MMAD 覆盖仍被 W Fixpipe 读取的 source。四个 head 的 C7 峰值因此是完整
`0x40000 = 256 KiB`，这是本候选的 L0C 容量下界。相邻 head、相邻 Stage 或下一 head-group
复用任何 L0A/L0B/L0C 地址前，仍必须等待对应 MTE1/Cube/Fixpipe 最后 reader；本表不提供隐式
执行顺序。目标 A5 是否提供至少 256 KiB 可用 L0C、上述 MMAD/Fixpipe layout 表达能力及所需
HardEvent 组合，尚未由目标 CANN 头文件和最小设备编译证明，均为 **PROPOSED** API gate。

### Workspace：每个 workgroup 8 slot

| slot-relative range | 大小 | owner/lifetime |
| --- | ---: | --- |
| `[0x00000,0x04000)` | 16 KiB | `Qhat`，V0 MTE3 -> V6 MTE2 |
| `[0x04000,0x08000)` | 16 KiB | `Khat`，V0 MTE3 -> V6 MTE2 |
| `[0x08000,0x08200)` | 512 B | hard pad；不存 `betaEff` |
| `[0x08200,0x10200)` | 32 KiB | FP32 `G`，V0 MTE3 -> V6 MTE2 |
| `[0x10200,0x10400)` | 512 B | alignment pad |
| `[0x10400,0x22400)` | 72 KiB | score/VCS/Post-RHS 三代 payload overlay |

`slotStride=0x22400`，八槽区间为 `[0,0x112000)`；control 为
`[0x112000,0x113000)`，所以 `workgroupStride=0x113000`。payload 的 score 语义与上面的
C2 lane 相同；VCS 使用低 `[0,0x4800)`；Post-RHS 使用
`KbetaG[0,0x4000)+Vbeta[0x4000,0x8000)`，其余始终 hard unused。

Host 必须调用等价于 `CheckedWorkspaceSizing(N, workgroupId)` 的 checked-u64 计算，先验证
`N > 0`、`workgroupId < N` 和 `N <= UINT64_MAX / 0x113000`，再得到：

```text
workgroupBase = workgroupId * 0x113000
totalBytes    = N           * 0x113000
```

还要验证 `totalBytes` 能被实际 allocator 的 size 类型表达且不小于传入 workspace；任何检查失败
都在 Host 拒绝 launch，不能在 device 侧用截断 offset 继续执行。公开输出是独立 GM tensor，
不计入这段 workspace，也不能假装成 `WorkspaceRegion`。

## Tail 与 GM 有效区

令 `M=validRows`，必须满足 `1 <= M <= 64`。token-row 输入 `q/k/gate/v/beta` 只允许从 GM
读取 `[0,M)`：向量输入的有效元素分别为 `M*K`、`M*K`、`M*K`、`M*V`、`M`；
`A_log/dt_bias` 是按 ABI 校验的 head/key 属性，不用 64-token padding 伪造。V0/V1 必须在固定 UB
和 72 KiB payload 内显式生成 MMAD 会读取的 padding，禁止从 GM 读 `[M,64)` 补满物理槽。

tail 仅执行 `S_active=ceil(M/16)` 个非空 band。active band 的 Q/K 对齐 padding 必须为零；C2
未写的 raw row 可以保留上一代值，但定义为 undefined。V3 的一次 VF 必须在任何 mask/VCS reader
之前直接清零 raw 的无效行列，并为求逆构造 identity padding，不能先读取旧值再乘零。

所有 token-row 公开 GM 输出只写 `[0,M)`，不得把本地 `[M,64)` 行映射到下一个 sequence/chunk。
`Aqk/Akk` 的公开物理 row width 固定为 64：每个有效行只允许写 `[0,64)`，其中逻辑有效区
`[0,M)x[0,M)` 按公式产生，列 `[M,64)` 写确定的零；内部求逆矩阵的本地行 `[M,64)` 才使用
identity padding，且这些行不得映射到当前公开 GM。具体写回量是 `Aqk: M x 64`；Akk 的
`q00/q01` 只写 `min(M,32)` 行，`q10/q11` 只写 `max(M-32,0)` 行，四个 quadrant 的 GM
destination leading dimension 都是 64。若后续 ABI 为 Akk 分配完整 `[64,64]` chunk tensor，
必须先单独冻结其 padding 值并补 golden，不能沿用本伪代码推断。GM copy 的 byte count 和
最后地址都必须由 checked stride 计算，
验证落在 Host 提供的 tensor extent 内；固定 UB/L1/workspace 容量不因 tail 缩小。

## 分核与 AIV owner

令 `Ctot` 为总 chunk 数、`H` 为 head 数、`G=ceil(H/4)`、`N` 为可用 AIC workgroup 数。
`BuildCorePlan` 使用无乘法溢出的 64-bit balanced half-open range：

```text
if Ctot >= N:
    ChunkOnly
    把 Ctot 个 chunk 均衡分给 N 个 workgroup
    每个已分配 chunk 在本 workgroup 内依次处理全部 G 个 head group
else:
    ChunkHeadGroup
    把 Ctot * G 个 (chunk, head-group) 展平任务均衡分配
```

因此只有 chunk 数不足以铺满机器时才按 head group 补充分核，正常路径始终 chunk-first。

一个四 head group 内的 AIV 映射在所有 Vector Stage 保持不变：

```text
AIV0: group-local head 0, 1 -> local slot 0, 1
AIV1: group-local head 2, 3 -> local slot 0, 1
```

相邻两个 head group 构成一个 wavefront，八个 head 映射到 `headId % 8` 的 workspace slot。
workspace owner、物理 UB owner、AIC L1 owner 和 AIC L0C owner 是四套独立 credit，不能只用
一个 slot/generation 表示。每个 workgroup 的 AIC 与两个 AIV 按完全相同的 `WorkItem` 顺序维护：

```text
OwnerTicketState:
    workspaceNext[8] = {0}
    localNext[4] = {0}
    l1Next[4] = {0}
    l0cNext[4] = {0}

for each active head in deterministic WorkItem order:
    workspaceSlot       = headId % 8
    workspaceGeneration = workspaceNext[workspaceSlot]++
    localBankId         = aivId * 2 + aivLocalSlot
    localGeneration     = localNext[localBankId]++
    l1BankId            = groupLocalHead
    l1Generation        = l1Next[l1BankId]++
    l0cBankId           = groupLocalHead
    l0cGeneration       = l0cNext[l0cBankId]++
```

例如相邻 group 的 head0 与 head4 分别占 workspace slot 0/4，但都复用 AIV0 local bank 0；
head4 必须等待 head0 发布的
`LocalBankFree(localBankId=0, localGeneration=head4.localGeneration)`，不能用 slot4 的初始
workspace credit 绕过。两者还会复用 L0C lane 0，head4 的 C2 必须独立等待
`L0cBankFree(l0cBankId=0, l0cGeneration=head4.l0cGeneration)`，不能把上一组 C7 尚在读取的
W/U source 当成已释放。
inactive head 不递增任何数组，因此任意 `H % 4`、任意 workgroup balanced begin、ChunkOnly 与
ChunkHeadGroup fallback 都从每个物理 owner 的 ticket0 开始且严格连续。`H=1..17`、两种分核模式、
不同非零 begin 已由 host `constexpr static_assert` 覆盖。

四种 owner credit 都采用点对点 ticket：初始化只预置 `Free[0]`；当前 owner 先
`Wait(Free[currentGeneration])`，最后消费者只发布一次 `Set(Free[currentGeneration+1])`，下一 owner
的 generation 恰好加一。禁止 Set 同代，否则同一个 token 会同时充当 acquire 前置和重复发布。
V0/V6 负责 local UB ticket，C2/C7 负责 L1 和 L0C 两套独立 ticket，C7 的最终 W/U Fixpipe 完成
负责 workspace slot ticket。L1 在 C7 最后 MTE1 reader 完成后即可独立归还；L0C 必须等最终 U
Fixpipe reader 完成后才能归还。其余边按真实资源域选择当前 head 的 ticket：context/payload/transaction 使用
`(workspaceSlot, workspaceGeneration)`，beta/raw/local-source 使用
`(localBankId, localGeneration)`，T/Akk/C2-L1 使用 `(l1BankId, l1Generation)`，所有 L0C
MMAD/Fixpipe source 使用 `(l0cBankId, l0cGeneration)`。V6/C7 这类跨域
汇聚必须逐条 Wait，不能拿单一 ID 的 Join 伪装成同一计数器。

八个 GM slot 不增加 UB：每个 AIV 仍只有两份固定 local bank，每份
`112 KiB MAIN + 12 KiB AUX`。上一 workspace generation 的 output drain 与全部 ready/free 握手未闭环前，
禁止新 generation 复用同一 GM slot；上一 local generation 未归还前也禁止覆盖同一物理 UB bank。
上一 L0C generation 未由最终 Fixpipe reader 归还前，同一 `groupLocalHead` 的下一组 C2 也不得
写入该 64 KiB lane。

## Ready/free 合同

伪代码同时具名 data-ready 与 storage-free，后续设备实现必须使用有界 credit/反向 free，
不能依赖 block 启动顺序，也不能对同一 flag 连续 set 而没有消费。

| Producer -> consumer | 必须保留的边 |
| --- | --- |
| slot owner -> `V0` | `SlotFree(workspaceSlot, workspaceGeneration)` 与独立的 `LocalBankFree(localBankId, localGeneration)`；V6 最后使用 UB 后发布 `LocalBankFree(localBankId, localGeneration+1)` |
| L1 owner -> `C2` | `L1BankFree(l1BankId, l1Generation)`；C7 最后读取本 lane/Akk resident 后发布 `L1BankFree(l1BankId, l1Generation+1)` |
| L0C owner -> `C2` | `L0cBankFree(l0cBankId, l0cGeneration)`；C7 最后一个 U Fixpipe reader 完成后发布 `L0cBankFree(l0cBankId, l0cGeneration+1)` |
| `V0` -> `V1` | local `V0ExportDone`；V1 只能在 V0 的所有 MTE3 source reader 完成后 overlay 同一 MAIN source |
| `V0` -> `V3` | `V0BetaReady` 保留 `betaEff`；V3/V6 的后续 local 生命周期最终由 `LocalBankFree` next-ticket 闭环 |
| `V0` -> `V6` | workspace `V0ContextReady`；它只表示 `Qhat/Khat/G` context drain 完成，只由 V6 消费，不是 V1 的 local 前置 |
| `V1` UB source -> `C2` raw destination | score 的最后一个 MTE3 source reader 完成后发布 `V1MainSourceFree`；C2 写 raw 前还要持有对应 generation 的 `C2RawDstFree` credit |
| `V1` -> `C2` | `V1ScoreReady`；C2 一次 MTE2 读完 GM payload 后返回 `C2ScorePayloadFree`，最后一个 MTE1 reader 完成后另发 `C2ScoreL1Free` |
| `C2` -> `V3` | 来自真实 raw-score producer pipe 的 `C2RawReady`；`C2RawDstFree` 只由当前 V1 source drain 发布并被当前 C2 消费，跨代物理 UB 复用统一由 `LocalBankFree` 管理 |
| `V3` -> `C4` | 所选 VCS payload drain 完成后的 `V3VcsReady` |
| `V3` -> `V6` | 所有 V3 MTE3 source reader 完成后的独立 `V3LocalSourceFree` |
| `C4` -> `C5` | Fixpipe 产生的 `C4TReady` 与 resident producer 产生的 `C4AkkPrepReady` 两条边；跨 head-group 的 resident 复用由 `L1BankFree` next-ticket 闭环 |
| `C4` -> `V6` | C4 一次读完全部所选 payload 段后的 `C4PayloadFree` |
| `C5` -> `C7` | 最终 resident quadrant 提交后的 `C5AkkReady` |
| `V6` -> `C7` | 完整 RHS drain 后的 `V6RhsReady` |
| `C7` -> owner | C7 最后一个 MTE1 reader 完成后发布 `L1BankFree(l1BankId, l1Generation+1)`；最终 U Fixpipe reader 完成后发布 `L0cBankFree(l0cBankId, l0cGeneration+1)`，并直接发布 `SlotFree(workspaceSlot, workspaceGeneration+1)` |

`V6` 必须三路汇聚 `V0ContextReady + V3LocalSourceFree + C4PayloadFree`；`C7` 必须两路汇聚
`C5AkkReady + V6RhsReady`。context ready 不等于 local bank free，VCS ready 也不等于 payload free，
即使 profiling 中某个生产者总是先完成，也不能合并这些状态。

当前实现不引入额外的 output-drain coordinator 或 C7 中间完成状态。V0/V3/V6 的 ready 边必须在
各自启用的公开 GM 输出完成后才能发布，C7 对 `C5AkkReady/V6RhsReady` 的 wait 因而传递覆盖此前
drain；C7 自身的最终 U Fixpipe 完成后直接发布 L0C 与 workspace 两个 next-ticket。若以后增加不能被这些 ready
边覆盖的独立输出，必须重开 coordinator 合同，不能直接沿用本闭环。

本目录禁止 `PipeBarrier<PIPE_ALL>()`，也没有任何真实 barrier 调用。未来实现只有在证明是同核
V-pipe RAW/WAR/WAW 时才能考虑 `PipeBarrier<PIPE_V>()`；它不能替代 MTE/Cube/Fixpipe 的硬事件，
更不能替代核间 ready/free。

## 资源和 API 边界

- UB 只给 Vector 使用。一个 Vector Stage 的完整输入、输出、scratch 和异步 source 生命周期必须
  同时装入固定布局，不能靠隐藏 spill、搬位或分 tile 绕过。
- L1 只给 Cube 使用。跨 Cube Stage resident 必须预留四份并直接写最终地址，禁止 L1 内整理或搬位。
- Vector 结果供 Cube 使用时必须经过 GM/workspace；Cube 结果供 Vector 使用时必须有两份匹配 UB
  destination，或显式选择并登记 GM relay fallback。
- AIC/AIV 无依赖操作没有隐式执行顺序，live region 不能因时间线“看起来错开”就重叠。
- workspace context 固定为 `Qhat[0x0000,0x4000) + Khat[0x4000,0x8000) + hard pad
  [0x8000,0x8200) + G[0x8200,0x10200)`；`betaEff` 不写 workspace，而是在同一 AUX region
  常驻到 V6，最终由 `LocalBankFree(localBankId, localGeneration+1)` 保护复用。
- `AkkStorage::Fp32Internal` 为 **BLOCKED**，目前只是通过容量推导的 L1 布局候选；FP32 Akk 与 2-byte RHS 的
  C7 Cube operand 组合及对应目标 CANN API 尚未证明。当前 `RunV0/V1/V3/V6` 与
  `RunC2/C4/C5/C7` 伪 dispatch 全部只接受 `AkkStorage::TwoByteAbi`，不能把 FP32 candidate
  当作已支持模板。
- `InputStorage` 与 `ScoreStorage` 是两个独立的必要语义轴，都不是已冻结的数值 key。前者
  选择 q/k/v 的共同 FP16/BF16 dtype，后者选择 score dtype；Host 必须按上面的允许映射拒绝
  未实例化组合。V1 必须据 `ScoreStorage` 选择 BF16
  `[-126,120]` 或 FP16 `[-80,80]` 的 base-2 指数范围；`GateStorage::TwoByte` 只描述 gate
  载荷宽度，不能替代 score dtype，也不能驱动该分支。
- workspace 精确 offset、公开输出集合、Akk cast 边界、Fixpipe 直写配对 AIV UB 的能力、flag/event
  分配、Matmul/VF API、TilingKey 编码和 launch ABI 均为 **PROPOSED**，必须结合目标 CANN 官方文档、
  随包头文件和实现源码确认后才能编码。

## Host 语法检查

本目录可作为普通 C++17 做静态语法检查，不需要 CANN：

```sh
g++ -std=c++17 -Wall -Wextra -Werror -fsyntax-only \
  fla/ops/ascendc/kda/chunk_kda_fwd_prepare/op_kernel/pseudocode/chunk_kda_fwd_prepare.cpp
```

该命令通过只说明设计伪代码内部是合法 C++17；不代表 A5 编译、device link、功能、精度、
sanitizer 或性能验证通过。
