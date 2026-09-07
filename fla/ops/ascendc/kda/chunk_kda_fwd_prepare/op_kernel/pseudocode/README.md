# Chunk KDA Forward Prepare 内核伪代码

本目录是 A2/A3 Arch22 与 A5 Arch35 `chunk_kda_fwd_prepare` 的可检视、可用 host C++17
做语法检查的设计伪代码。
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
|-- chunk_kda_fwd_prepare_policy.h      # 分架构 UB/L1/workspace 资源账本
|-- chunk_kda_fwd_prepare_struct.h      # task、buffer、同步公共类型
|-- chunk_kda_fwd_prepare_utils.h       # 分核、head owner、代际映射
|-- chunk_kda_fwd_prepare.cpp           # MIX AIC/AIV 符号入口与调度
|-- chunk_kda_fwd_prepare_contract_test.cpp # 跨架构同步与操作轨迹 host 合同测试
|-- arch35/
|   |-- chunk_kda_fwd_prepare_vec.h     # V0/V1/V3/V6
|   `-- chunk_kda_fwd_prepare_cube.h    # C2/C4/C5/C7
`-- arch22/
    |-- chunk_kda_fwd_prepare_vec.h     # A2/A3 V0/V1/V3/V6 候选
    `-- chunk_kda_fwd_prepare_cube.h    # A2/A3 C2/C4/C5/C7 候选
```

## 八个物理 Stage

一个 `(sequence, chunk, head)` 事务固定为八个物理 Stage：

| Stage | 类型 | Stage 入口完整输入 | 计算与提交结果 |
| --- | --- | --- | --- |
| `V0` | Vector | HK cohort owner 从公开 GM 一次装入 raw `q/k`；其余 HV head 从 owner 的只读 Q/K cache 装入已归一化值；每个 HV 另完整装入 `beta` 及所选 gate 输入 | 每个 HV 仍只有一次 VF：仅 `owner+L2` 对该 HK 做唯一一次 Q/K norm，Identity owner 和其余 head 直接保留已在最终 UB 地址的 2-byte MTE2 结果，不做重复 convert/round；同一次 VF 完成本 HV 的 beta、gate/cumsum。只有 owner 写一份 `Qhat/Khat` cache，非 owner 不复制到各自 context；Fused ABI 另存每 HV 的 G context，Current ABI 只写一次公开 `gk` 并由 V6 复用；Arch35 另物化 `G_ref[4]`，Arch22 不物化 |
| `V1` | Vector | 同一 head 仍驻留在 UB 的 V0 local source | 一次 VF 先按 `ScoreStorage` 截断 base-2 指数，再按 `useExp2` 选择 `exp2(x)` 或 `exp(x*ln2)`，物化全部 `S=4` 的 `Qplus/Kplus/Kminus`；Arch22 必须 `V0(head)->V1(head)` 后才能复用 shared arena，score drain 只读取该 head 的 private bank |
| `C2` | Cube | 一次把完整 72 KiB score 搬入 L1 | 四个 row band；每个 band 将 `Qplus/Kplus` 沿行堆叠为 `[32,128]`，通过一次 `MmadRowStackedLhs`（packed TileMmad）同时生成 `rawAqk/rawAkk`；Arch35 直达配对 AIV UB，Arch22 先写 compact raw GM relay |
| `V3` | Vector | UB 中完整 raw score；Arch35 从 AUX 取保留的 `betaEff`，Arch22 从 workspace 重载 | 一次 VF 完成无效区清零、causal mask、`Aqk/Lkk` 和 VCS 两个叶子 `B/X0/X1`；已知为零的 q01 只在 Current ABI 作为公开 Akk 输出写回，Arch22 Fused full 因完整 ND 转换限制保留一次 relay，Fused top 不物化 |
| `C4` | Cube | 所需 GM/workspace source 已 ready；`M>32` 装入 `B/X0/X1`，Arch35 另把 stable Akk 装入最终 L1 resident | `M>32` 用一个 MMAD 计算 `T = B @ X0`；Arch35 在最终 L1 地址直接清零 q01，Arch22 把 FP32 `T` 写 GM relay；`M<=32` 时 Arch35 只装 q00、Arch22 走 control，两者都不启动 MMAD |
| `C5` | Cube | `M>32` 时前一 Stage 的 `T`、`X1`、Akk prepack 均 ready | `M>32` 用一个 MMAD 计算 `Y = -X1 @ T`；Arch35 提交 resident quadrant，Arch22 把 2-byte `q10` 写入 row-major Akk GM relay；`M<=32` 只做 control pass-through |
| `V6` | Vector | `Qhat/Khat` context、Fused 的 G context 或 Current 的公开 `gk`、`V` 以及三条 data-ready 边；Arch35 从 AUX 取常驻 `betaEff`，Arch22 从 workspace 重载 | 一次 VF 以 `[-80,80]` 截断 direct base-2 指数，按同一 `useExp2` 轴求 `2^x`，并生成 `Qg/qg`、`kg`、`K_beta_g`、`V_beta`；两个 RHS plane 固定 base，`M<=32` 各 drain 32 行，`M>32` 各 drain 64 行 |
| `C7` | Cube | 有效 Akk 与对应 32/64 行 RHS | 一个逻辑 MMAD 计算 `Akk @ [K_beta_g \| V_beta]`，提交 `[W \| U]`；两架构的 top-only tail 都把有效 q00 直接装成 tight 32x32 Cube operand 并做 `K=32`，Arch22 full 才从 GM relay 做完整 ND 到 Cube-ready 转换 |

每个 Stage 只包含 Cube 或 Vector 之一。`V0/V1/V3/V6` 各只允许一次 VF，不按 token、
score block 或硬件 tile 分 pass。Cube 的独立逻辑 MMAD 可在编译期展开为硬件 tile，但本 Stage
全部操作数必须在 Stage 入口前就绪，不能在同一 Cube Stage 内读取本 Stage 新产生的结果。

八 Stage 是冻结的跨架构物理合同。Arch35 的容量/依赖下界与 Arch22 的 relay 依赖分别为：

- `V0+V1` 必须同时保留 context 和 72 KiB score，而单 local head 只有固定
  `112 KiB MAIN + 12 KiB AUX`，容量不成立。
- `C4` 产生 `T`，`C5` 才能消费；二者不能合并。
- `C7` 既依赖 C5 新完成的 Akk，又要汇聚独立的 V6 RHS；`C5/C7` 不能合并。
- `V6` 必须等 V3 归还 local bank 且 C4 归还 payload owner，不能提前覆盖任一存活区。
- Arch22 的 V0/V1 虽按同一 head 相邻调度，但仍是两次独立 VF；本伪代码没有证明把两者折成一次
  VF 的 CANN 接口、寄存器压力或数值顺序，因此不能据此把八 Stage 降为七 Stage。
- Arch22 的 C2 raw，以及 `M>32` 时的 C4 `T`、C5 `q10`，分别形成 GM relay producer/consumer 边；不能把相邻 Cube
  Stage 合并后读取本 Stage 的 Cube/Fixpipe 新输出。

V0 的核心数学不能被一个含糊的 `ApplyFrozen*` 隐藏。候选 TilingKey 必须显式携带以下
**PROPOSED** 语义轴，但当前伪代码不冻结它们的数值编码：

- `QkNormMode::{Identity,L2}`，Prepare 公共默认值为 `Identity`；L2 明确采用仓内 FLA kernel/reference 的
  `x_hat = x * rsqrt(sum_d(x_d^2) + epsilon)`，不是
  `x / max(sqrt(sum_d(x_d^2)), epsilon)`。Arch22 的 reduction work 区只改变实现方式，不改变
  分母；`epsilon` 是 runtime 正有限标量，公共默认值为 `1e-6`。
- beta 是每 token 一个 FP32 标量；L2 公开入口即使接受 BF16/FP32，也必须在下发 Prepare 前统一
  cast 到 FP32，所以 beta storage 不是 kernel 模板轴。full chunk 逻辑输入为 256 Byte，可放入
  0x200 Byte 的对齐 AUX region；剩余字节只是 hard pad。
  `BetaMode::{Raw,Sigmoid,TwoSigmoid}` 只决定该 FP32 标量的数学变换。
- `GateMode::{PrecomputedStep,Softplus,SafeSigmoid}`；后两种按冻结公式消费必需 `A_log`、
  runtime `lowerBound`（公共默认 `-5`）和可选 `dt_bias`。`hasDtBias=false` 必须把最终 AUX 地址显式置零，且不能
  构造或读取虚假的 GM source。
- `InputStorage::{Fp16,Bf16}` 选择 `q/k` 及其派生输出的公开 2-byte dtype；独立的
  `valueStorage::{Fp16,Bf16}` 选择 `v/u` dtype，二者允许不同且不改变容量。`ScoreStorage::{Fp16,Bf16}`
  单独描述内部 `Qplus/Kplus/Kminus` 的真实 dtype，不能用含糊的
  `TwoByte` 代替。独立的 `safeGate` 模板属性描述现有 SAFE_GATE specialization，它与决定 V0
  gate 数学公式的 `GateMode` 正交，不能用 `GateMode::SafeSigmoid` 推导。Host 必须校验允许的
  映射：`safeGate=false` 时 `ScoreStorage` 必须等于 `InputStorage`；`safeGate=true` 且 input 为
  FP16 时 score 必须提升为 BF16，即使 `GateMode::PrecomputedStep` 也一样；BF16 input 始终只能
  配 BF16 score。`(Bf16,Fp16)` 和 SAFE_GATE `(Fp16,Fp16)` 均拒绝。
- `useExp2` 是与 gate、dtype、ABI 正交的模板轴，Prepare 公共默认值为 true，且不改变
  `gk` 的 log2 单位：true 选择
  `exp2(x)`，false 选择 `exp(x * ln(2))`。两条路径必须先截断同一个 base-2 指数 `x`；V1 的
  BF16 score 范围为 `[-126,120]`，FP16 score 范围为 `[-80,80]`，V6 的 direct `G` 与
  `G_last-G` 固定为 `[-80,80]`。共享 symbolic `EvaluatePow2<useExp2>` 冻结
  `clamp(x) -> exp2(x)` 或 `clamp(x) -> FP32 multiply ln(2) -> exp(x*ln2)` 的顺序；具体
  `Mins/Maxs/Exp/Exp2` API 与必要的 `PIPE_V` 顺序仍是 **PROPOSED**。
- 每次写入 2-byte storage 前都使用 RINT 语义。目标是 FP16 时，先把 FP32 中间值饱和到
  `[-65504,65504]` 再舍入；目标是 BF16 时只做 BF16 舍入，不做这一步有限幅值饱和。V1 的目标
  由 `ScoreStorage` 选择；V3、V6 的 Q/K 路径、C5 和 C7-W 由 `InputStorage` 选择，V6 的
  V/V-beta 路径与 C7-U 由 `valueStorage` 选择。symbolic `RoundToScoreStorage`、
  `RoundToInputStorage` 与 `CubeOps::StoreRounded` 必须保留该区别，普通
  `Store` 只表示同 dtype 搬运，不能暗含 cast、饱和或舍入。
- V6 的 `K_beta_g` 保留现有 Prepare 的两个真实存储舍入点，不能合并为末尾一次 cast：
  `kGateRounded = RoundToInputStorage(ClampForInputStorage(kHat * Pow2(G,useExp2)))`，随后把
  `kGateRounded` 转回 FP32 与 `betaEff` 相乘，再执行第二次
  `RoundToInputStorage(ClampForInputStorage(betaEff * fp32(kGateRounded)))`。FP16 的两次
  `ClampForInputStorage` 都是上面的 `+/-65504` 饱和；BF16 两次 clamp 都是 no-op，但两次 BF16
  舍入仍必须存在。
- FUSED `Qg_scaled` 同样保留两个存储边界：先得到
  `qgStorage = RoundToInputStorage(ClampForInputStorage(qHat * Pow2(G,useExp2)))`；Current Qg 输出消费该
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

`C2` 对每个 active band `s` 将两个互不依赖的左操作数沿行堆叠：

```text
stack(Qplus_s, Kplus_s) [32,128] x Kminus_s^T [128,b_s]
    -> stack(rawAqk_s, rawAkk_s) [32,b_s]
```

并通过一次 `MmadRowStackedLhs`（设备侧映射为一次 packed TileMmad API 提交）完成。
结果的逻辑上 16 行是 `rawAqk_s`，逻辑下 16 行是 `rawAkk_s`；两部分必须通过原生
L0C layout-aware tile 分别写入原有 raw destination，不能换算成 row-major 连续字节半区。
每个 head 的 full chunk 共提交四次 packed TileMmad API，但数学上仍是
四个 `Qplus_s @ Kminus_s^T` 和四个 `Kplus_s @ Kminus_s^T`，即八个独立乘积；按
`16 x 16` 输出 tile 展开仍为 `2 x (1 + 2 + 3 + 4) = 20` 个 tile，不能把四次 API
提交误写成四条硬件 MMAD 指令。
这一 lowering 与仓内 A5 `ComputeRawAqkAkkCubeStableBlockDirectUbArch35` 已采用的
`packedRows + L0C top/bottom tile` 模式一致；该路径使用 32 行 score block，而本设计仍按
`S=4` 固定 16 行 band，只复用行堆叠方法，不改变分块语义。
每份 prefix 恰好包含 full `Kminus[64,128]` 公式在有效列读取的 key，因此有效 raw 结果完全一致。
`[b_s,64)` 没有存储，也不是隐式零；`V3` 必须在任何 mask/VCS reader 之前直接清零这些区域，
再处理当前 16-row block 内的 `j > i` causal mask。

`C2` 在 Stage 入口只搬一次完整 72 KiB。遍历四个编译期 band 只是同一 Cube Stage 内的独立
packed TileMmad 宏格，不构成 GM 重读或语义 pass。

## 固定物理布局

下表中的地址都是 owner 内的 byte half-open range。hard pad/reserve 不能借给另一个 head、
generation 或未登记 scratch；Stage overlay 只能在上一语义的最后异步 reader 完成后原址改名，
不能通过 UB/L1 内搬位整理碎片。

### Arch35 UB：每个 AIV 248 KiB

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

### Arch22 UB：每个 AIV 184 KiB

Arch22 按 Basic API 的普通 Vector local 上限 `0x2E000` 记账，不使用硬件总 UB 与保留区之间的
8 KiB。两个 72 KiB private bank 与一个 40 KiB shared arena 恰好占满该上限：

| AIV 绝对 UB range | 大小 | 固定 owner |
| --- | ---: | --- |
| `[0x00000,0x12000)` | 72 KiB | local slot 0 private bank |
| `[0x12000,0x1C000)` | 40 KiB | 本 AIV 唯一 shared arena |
| `[0x1C000,0x2E000)` | 72 KiB | local slot 1 private bank |

slot0 的完整计算窗是 `[0x00000,0x1C000)`，slot1 的完整计算窗是
`[0x12000,0x2E000)`，均为 112 KiB；二者只在 shared arena 重叠，禁止并发写。每个 private bank
的相对 `[0000,4000)`、`[4000,8000)` 固定放 Q/K。Gate2B V0 把 raw 放
`[8000,C000)`，把 `[10000,12000)` 的高 8 KiB 留给小对象；gate/cumsum 的全部 raw reader 完成后，
`[8000,10000)` 才改名为 32 KiB norm work。V1 等 context export source drain 后，把 private 后
40 KiB 直接原址改名为 `Kminus0/1/2/3` 的 `[8000,9000)`、`[9000,B000)`、
`[B000,E000)`、`[E000,12000)`。最终 72 KiB score 完全位于 private bank，MTE3 不读取
shared arena，也没有 UB 内搬位。

shared arena 的相对布局固定为 `G[0000,8000)` 与 VF scratch `[8000,A000)`。Gate2B raw 从不
进入 shared，也不做重叠 widening；single VF 先从 private raw 计算 gate/cumsum 并写 shared G，
在 raw last-read 后复用 private norm work 完成 Q/K norm。只有 GateFP32 与 G 同 shape 时才允许在
shared G 地址逐 row 原位更新。gate source、G 与 scratch 的 last-reader/overlay 次序必须在单次 VF
内部显式表达，不能把两个存活语义排到同一地址。

同一 AIV 的调度固定为 `V0(local slot)->V1(same slot)->V0(next slot)->V1(next slot)`。
这样 V1 直接消费同一 shared G；若改成 all-V0 后 all-V1，只能把 G 写 GM 后再次读取，属于规则14
fallback，不是本主路径。V0/V1 共持有一张 shared ticket；只有 V0 的 context MTE3 source 和 V1
最后一个 shared reader 都完成后才能发布 next-ticket。V3 的 compact raw、输出、work 和 beta
合计小于72 KiB，完全在 private bank 运行，不取得 shared ticket；V6 才取得后一张 shared ticket。
V1 score、V3 VCS/公开输出和 V6 RHS/公开输出都必须排到 private bank 后再异步 drain，不能把
shared arena 留作下一 local slot 的异步 source。

### L1：每个 AIC 512 KiB

| AIC L1 range | 大小 | 固定 owner |
| --- | ---: | --- |
| `[0x00000,0x12000)` | 72 KiB | group-local head 0 current lane |
| `[0x12000,0x24000)` | 72 KiB | group-local head 1 current lane |
| `[0x24000,0x36000)` | 72 KiB | group-local head 2 current lane |
| `[0x36000,0x48000)` | 72 KiB | group-local head 3 current lane |
| `[0x48000,0x5C000)` | 80 KiB | 四 head 的跨 Cube Stage resident |
| `[0x5C000,0x7FF00)` | 143.75 KiB | 两架构 hard reserve；未登记 tensor 不得占用 |
| `[0x7FF00,0x80000)` | 256 Byte | Arch35 hard reserve；Arch22 Basic API 不可分配 |

每条 72 KiB current lane 在 C2 中是完整 score：`Q+[0000,4000)`、`K+[4000,8000)`、
四个 causal prefix `K-0[8000,9000)`、`K-1[9000,B000)`、`K-2[B000,E000)`、
`K-3[E000,12000)`。C4 不把 VCS 放回 current lane：Arch35 只在该 lane 的
`B[0000,1000)` 暂存一个 MMAD 输入，`X0/X1` 与 stable Akk 直接装入下面的四份 resident；
Arch22 的 current lane 同样只暂存 `B[0000,1000)`，其 `X0/X1` 直装 resident，stable Akk
已单独写入 row-major GM relay。C7 再把 current lane 改名为
`KbetaG[0000,4000) Vbeta[4000,8000)`，`[8000,12000)` hard unused。

当前可达 `TwoByteAbi` 的 resident 为四份 `X0[48000,4C000)`、四份
`X1[4C000,50000)`、四份 `T[50000,54000)` 和四份 `AkkTau[54000,5C000)`；单 head
stride 分别为 4/4/4/8 KiB。`Fp32Internal` 的容量候选把同一区间解释成四份
`AkkFp32[48000,58000)` 与四份 `T[58000,5C000)`，但该模板是 BLOCKED，不可 dispatch。

Arch35 可按上表把矩阵中间量保存在 resident；Arch22 只复用四份 head owner 与地址上界，不能
据此声明同样的数据通路。目标 CANN 9.1 Arch2201 的已查接口不支持 L0C->UB，也不支持
FP32 L0C->FP32 L1，因此 Arch22 C2 raw 与 C4 `T` 必须经 GM relay。为降低接口风险，当前 Arch22
候选连 C5 `q10` 也写入 row-major 2-byte Akk GM relay；C7 full 路径再把完整 Akk 装入四份 L1
resident，top-only 路径则直接把 q00 装成 tight Cube operand。Arch22 full 的 Fused ABI 仍需
relay 已知零 q01，因为把多个 ND 子矩形拼入同一个最终 NZ operand 尚未通过接口门禁；这是规则14
例外，不是数学 payload。该路径不复用下述 Arch35 tight-quadrant resident。三次 GM roundtrip 是满足规则14的具名例外，不能
静默改成未证明的 L0C->L1/L0C->UB 通路。Arch22 可用 L1 上界按 `0x7FF00` 保守计算，登记区
只到 `0x5C000`，其余 `[0x5C000,0x7FF00)` 保持 hard reserve。

Arch35 单 head 的 8 KiB `AkkTau` 是 2x2 tight quadrant pack：相对 resident base 的
`q00[0x0000,0x0800)`、`q01[0x0800,0x1000)`、`q10[0x1000,0x1800)`、
`q11[0x1800,0x2000)` 各为紧凑 32x32 两字节矩阵。它不是一个 row-major `64x64` BufferSpan。
C7 的 `MmadQuadrantPackedLhs` 只符号化以下合同：MTE1 从四个最终 L1 象限地址直接装入逻辑
L0A 四象限后参与 MMAD，不先在 L1 拼成 row-major，也不做任何 L1 位置移动。目标 A5 是否支持
该装载/计算形式及其精确 API、事件和 tile 描述仍是 **PROPOSED** 硬门禁。

Arch35 还冻结 64 KiB L0A/L0B 的 stage overlay。C2 的 `[32,128]` 左操作数拥有完整
L0A owner `[0,0x2000)`；`Qplus/Kplus` 分别写入 zN 布局的逻辑 `rows[0,16)` 和
`rows[16,32)` tile，物理上会按分形块交织，不能解释成两个连续 4 KiB 半区。
一次 `MmadRowStackedLhs` 只读同一份 L0B `Kminus[0,0x4000)`。C7 只读一份
L0A `Akk[0,0x2000)`，并把 `Kbeta/Vbeta` 分别放在 L0B `[0,0x4000)`/`[0x4000,0x8000)`。
完整 C2 owner 与 C7 的两个不重叠 RHS 区间，是 C2 单次 packed TileMmad、以及 C7
两次 MMAD 完成后分别只发布一次
`CubeToMte1OperandReuse` 的前提；若正式 API 不能维持该映射，必须按实际最后 reader
重新设计释放和装载，不能沿用一次 release。

L0A/L0B operand 另有独立于 L1 resident 和 L0C result 的 owner epoch。Arch35 的所有 head
共享一个物理 operand bank；每个 active C2 band、C4、C5、C7 各取得一个严格递增的
`l0OperandGeneration`。同一 C2 band 的 stacked-QK-L0A、Kminus-L0B 与单次 packed TileMmad 共用
同一个 epoch，C7 的 Akk-L0A、Kbeta-L0B、Vbeta-L0B 与 W/U 两次 MMAD同理。只有该 epoch
最后一次 Cube reader 完成后，`CubeToMte1OperandReuse` 才释放地址供下一 epoch 的 MTE1 覆盖。
这个同 AIC 核内 release 不能由 L1 generation、L0C generation 或跨核 ready/free 代替。

### Arch35 L0C：按 Stage overlay 的候选账本

L0C 不跨 Stage 常驻，但四个 group-local head 可以同时处于同一 Cube Stage；每个异步 MMAD
结果在对应 Fixpipe 最后读取前必须有独立物理 span。因此固定四条互不重叠的 head lane：

```text
headLaneBase(h) = h * 0x10000, h in [0,4)
headLaneBytes   = 0x10000       # 64 KiB
requiredBytes   = 4 * 0x10000   # 256 KiB
```

C2 在每条 lane 内按 band 保存紧凑 `32 x N` FP32 stacked 结果；上 16 行为 `rawAqk`，
下 16 行为 `rawAkk`。`N={16,32,48,64}` 时每个 stacked 结果真实为 2/4/6/8 KiB。
下面都是相对 `headLaneBase(h)` 的 half-open range：

| C2 lane-relative range | 大小 | owner |
| --- | ---: | --- |
| `[0x0000,0x0800)` | 2 KiB | band 0 的完整 `[32,16]` L0C owner |
| `[0x0800,0x1800)` | 4 KiB | band 1 的完整 `[32,32]` L0C owner |
| `[0x1800,0x3000)` | 6 KiB | band 2 的完整 `[32,48]` L0C owner |
| `[0x3000,0x5000)` | 8 KiB | band 3 的完整 `[32,64]` L0C owner |

C2 每 head 同时存活 20 KiB，四 head 合计 80 KiB。表中的 byte range 只描述完整物理
owner；其中 `rawAqk/rawAkk` 分别是 `MakeLayoutL0C(32,N)` 上的逻辑
`rows[0,16)`/`rows[16,32)` tile。除 `N=16` 的特例外，两者按 L0C 分形布局交织，
不得为它们声明连续 byte subspan。每个 band 的单次 packed TileMmad 完成后，Fixpipe 通过
layout-aware tile view 分别写入原有目标；在两个 reader 都完成前不得覆盖该 owner。
四个 head 仍不能密排成单个 80 KiB owner，
因为同一条物理 lane 还要服务后续 C7。C4 把本 head lane 的 `[0x0000,0x1000)` 用作 `T`，
C5 保留该输入并把 `[0x1000,0x2000)` 用作 `Y`；只有前一 Stage 的最后 Fixpipe reader 完成后，
后续 Stage 才能 overlay C2 的旧语义。

C7 在每条 lane 内同时保留 `W[0x0000,0x8000)` 与 `U[0x8000,0x10000)` 两个独立 32 KiB
目标，避免 U MMAD 覆盖仍被 W Fixpipe 读取的 source。四个 head 的 C7 峰值因此是完整
`0x40000 = 256 KiB`，这是本候选的 L0C 容量下界。相邻 head、相邻 Stage 或下一 head-group
复用任何 L0A/L0B/L0C 地址前，仍必须等待对应 MTE1/Cube/Fixpipe 最后 reader；本表不提供隐式
执行顺序。目标 A5 是否提供至少 256 KiB 可用 L0C、上述 MMAD/Fixpipe layout 表达能力及所需
HardEvent 组合，尚未由目标 CANN 头文件和最小设备编译证明，均为 **PROPOSED** API gate。

### Arch22 L0C：两个 stage-use lane

Arch22 只有 `0x20000` L0C，按两条 64 KiB lane 分两轮处理 `(head0,head1)` 与
`(head2,head3)`。`l0cBankId=groupLocalHead%2`，但 generation 不能像 Arch35 一样从 C2 持有到
C7；C2/C4/C5/C7 每个 Stage 都取得独立 `l0cStageGenerations[stage]`，本 Stage 的最后 Fixpipe
reader 发布 next-ticket 后，下一 pair 或下一 Stage 才能复用同一 lane。每 Stage 的具体 L0C
offset 可 overlay，但不得在 reader 完成前重叠。普通 A2/A3 MMAD、Fixpipe 到 GM、GM ND->L1 NZ
及其 HardEvent 组合仍为 **PROPOSED**，正式代码必须按 CANN 9.1 头文件做最小编译确认。
其中 C2 从同一个 `MakeLayoutL0C(32,N)` owner 选择上下两个逻辑 row tile 并分别
Fixpipe 到 GM 的精确 API、source layout、stride 和 mode 也都是 **PROPOSED**；必须在
CANN 9.1/Arch2201 上完成最小编译与设备验证后才能落地，不能用 row-major 字节切半替代。

Arch22 的 L0A/L0B 同样按两条 physical lane 建立独立 operand epoch，
`l0OperandBankId=groupLocalHead%2`。每条 lane 分别按实际 AIC 发射顺序递增，不能因为 head0/head2
或 head1/head3 使用相同 offset 就共享同一代。C2 在每条 lane 内同样使用 zZ 布局的
stacked-QK L0A owner `[0,0x2000)`，Qplus/Kplus 仍由上下逻辑 row tile 区分，而不是连续
byte 半区；Kminus L0B 使用 `[0,0x4000)`，packed L0C 使用上表的 `[0,0x5000)`；每个 C2 band、
C4、C5、C7 的复用边界与 Arch35 相同。差别只是两个 bank 各自从 ticket0 连续推进，
不需要跨两条 lane 做同步。

### Arch35 Workspace：每个 workgroup 8 slot

| slot-relative range | 大小 | owner/lifetime |
| --- | ---: | --- |
| `[0x00000,0x04000)` | 16 KiB | 仅 cache owner slot 保存 `Qhat`，V0 MTE3 -> 所有 mapped-HV V0/V6 MTE2；非 owner slot 的该区 hard unused |
| `[0x04000,0x08000)` | 16 KiB | 仅 cache owner slot 保存 `Khat`，V0 MTE3 -> 所有 mapped-HV V0/V6 MTE2；非 owner slot 的该区 hard unused |
| `[0x08000,0x08200)` | 512 B | hard pad；不存 `betaEff` |
| `[0x08200,0x10200)` | 32 KiB | Fused ABI 的 FP32 `G` context；Current ABI hard unused，V6 改读公开 `gk` |
| `[0x10200,0x10400)` | 512 B | alignment pad |
| `[0x10400,0x22400)` | 72 KiB | score/VCS/Post-RHS 三代 payload overlay |

`slotStride=0x22400`，八槽区间为 `[0,0x112000)`；control 为
`[0x112000,0x113000)`，所以 `workgroupStride=0x113000`。payload 的 score 语义与上面的
C2 lane 相同；VCS 的 `[0,0x3000)` 为 `X0/X1/B`。Fused ABI 在 `[0x3000,0x4000)` 只保存 tight
q00/q11，q01 由 C4 在最终 L1 地址直接清零；`[0x4000,0x4800)` hard unused。Current ABI 由
公开 AkkOut 的 q00/q11 直接装入 L1，同样不回读公开 q01；Post-RHS 的两个 plane 固定为
`KbetaG[0,0x4000)+Vbeta[0x4000,0x8000)`。`M<=32` 时每个 plane 只写回并回读前
`0x2000`（32 行），base 仍为 `0/0x4000`；`M>32` 时各传完整 `0x4000`。其余始终 hard unused。

control page 的前 `0x100` Byte 是八个 32 Byte Q/K cache state record，物理 cache slot `i`
对应 `[0x112000+i*0x20, 0x112020+i*0x20)`；其余 `[0x112100,0x113000)` 保留。每条 record
固定为：

| record-relative range | 字段 | 初始化与访问合同 |
| --- | --- | --- |
| `[0x00,0x08)` | `freeGeneration` | Host 初始化为 0；AIC C7 汇聚 cohort 全部 `V6RhsReady` 后以 release 语义发布下一代 |
| `[0x08,0x10)` | `readyGeneration` | owner 的 Qhat/Khat MTE3 完成后写当前代；reader 只在 valid 后比较 |
| `[0x10,0x18)` | `logicalKey=(globalChunk,HK)` | 防止同物理 slot 的错误 cohort 被接受 |
| `[0x18,0x20)` | `validState` | Host 初始化为 0；owner 取得 free 后先失效，写完 key/generation 后以 release 语义置有效；reader acquire 后读取 |

这张表冻结物理空间和发布顺序，不宣称某个具体 atomic/flag API。`QkCacheReady` 是上述
level-triggered 状态的符号名，不是会被第一个 waiter 消耗的一次性 flag。

### Arch22 Workspace：每个 workgroup 4 slot

Arch22 仍使用 `slotStride=0x22400`，但每个四-head group 只占 `headId%4` 的四个 slot：slot 区间
`[0,0x89000)`，control `[0x89000,0x8A000)`，所以 `workgroupStride=0x8A000`。context 中
`Qhat/Khat/G` offset 与 Arch35 相同，但 Qhat/Khat 只在 cache owner slot 物化，非 owner slot
对应范围 hard unused；G region 同样只供 Fused ABI 使用；Current ABI 的 V6
直接重读 V0 唯一写出的公开 `gk`。`betaEff` FP32 的 256 Byte 放
`[0x8000,0x8100)`，`[0x8100,0x8200)` 保持 hard pad。它在 V0 drain 后供 V3/V6 重载，不能沿用
Arch35 的 AUX 常驻假设。

Arch22 control page 复用同一 32 Byte record 格式。四个物理 cache slot 使用
`[0x89000,0x89080)`，`[0x89080,0x89100)` 是为统一表格式保留的 pad，
`[0x89100,0x8A000)` 为其余 control reserve；初始化和 acquire/release 次序与 Arch35 相同。

Arch22 payload 的同一地址按 ready/free 顺序 overlay，所有范围均为 slot payload-relative：

| 生命周期 | payload range | 语义 |
| --- | --- | --- |
| C2 -> V3 | `[0x0000,0x5000)` | compact raw Aqk/Akk GM relay |
| V3 -> C4 | `[0x0000,0x3000)` | `X0/X1/B` VCS；C4 MTE2 drain 后才可 overlay |
| reserve | `[0x3000,0x4800)` | Arch22 不物化额外 Akk quadrant，保持 hard unused |
| C4 -> C5 | `[0x4800,0x5800)` | FP32 `T` GM relay |
| V3/C5 -> C7 | `[0x5800,0x7800)` | 仅 Fused ABI 的 row-major 2-byte Akk：V3 总写 q00，full 另写 q01/q11，C5 写 q10；top-only 不物化无人消费的 q01；Current ABI hard unused，直接复用公开 AkkOut |
| V6 -> C7 | `[0x7800,0xF800)` | 固定 base 为 `KbetaG[0,0x4000)`、`Vbeta[0x4000,0x8000)`；top-only 各传前 8 KiB，full 各传 16 KiB |
| reserve | `[0xF800,0x12000)` | hard unused |

C2、C4、C5 的内部 relay 结果都写 workspace/GM，后续 Stage 再读取；只有在 producer ready、consumer
MTE2 drain 和 payload overlay credit 都闭环后才允许改名。它们是 Arch22 的显式规则14 fallback。
Current ABI 的 G 与 Akk 不另建内部副本，分别复用公开 `gk` 与 AkkOut；Fused ABI 因无对应公开
输出才使用 context G 与上述 Akk relay。
特别是 V3 必须先用核内 MTE2->MTE3 source-free event 确认 compact raw 的最后 MTE2 reader 完成，
才能让 MTE3 覆写同一 payload 低地址为 VCS；跨核 `C2RawReady` 只表示 producer 已写好 raw，不能
替代这条反向 source-free 保护。

Host 必须调用等价于 `CheckedWorkspaceSizing(architecture,N,workgroupId)` 的 checked-u64 计算，
先验证 `N > 0`、`workgroupId < N` 和
`N <= UINT64_MAX / WorkspaceWorkgroupStrideFor(architecture)`，再得到：

```text
stride       = architecture == Arch22 ? 0x8A000 : 0x113000
workgroupBase = workgroupId * stride
totalBytes    = N           * stride
```

还要验证 `totalBytes` 能被实际 allocator 的 size 类型表达且不小于传入 workspace；任何检查失败
都在 Host 拒绝 launch，不能在 device 侧用截断 offset 继续执行。符号入口要求
`WorkspaceView::backingBytes >= totalBytes`，随后按架构策略绑定 `workgroupBase`、slot stride、context
和 payload offset/size；禁止沿用默认零值或由调用者任意拼装这些派生字段。公开输出是独立 GM tensor，
不计入这段 workspace，也不能假装成 `WorkspaceRegion`。

## Tail 与 GM 有效区

令 `M=validRows`，必须满足 `1 <= M <= 64`。token-row 输入 `q/k/gate/v/beta` 只允许从 GM
读取 `[0,M)`：向量输入的有效元素分别为 `M*K`、`M*K`、`M*K`、`M*V`、`M`；
`A_log/dt_bias` 是按 ABI 校验的 head/key 属性，不用 64-token padding 伪造。V0/V1 必须在固定 UB
和 72 KiB payload 内显式生成 MMAD 会读取的 padding，禁止从 GM 读 `[M,64)` 补满物理槽。

tail 仅执行 `S_active=ceil(M/16)` 个非空 band。active band 的 Q/K 对齐 padding 必须为零；C2
未写的 compact raw 段定义为 undefined。Arch22 V3 的 MTE2 load phase只搬 active band 的 Aqk/Akk
compact 矩形，单次 VF 对无效单元走零值分支，禁止先读取 inactive 段或上一代值再乘零；随后为求逆
构造 identity padding。Arch35 直写 UB 路径同样只能读取 C2 已定义的矩形。

`M<=32` 没有 q10，两架构都不执行无消费者的 T/q10 计算。Arch35 Fused ABI 的 V3 向内部
payload 提交 tight q00；Current ABI 只写公开 ld64 AkkOut，C4 再用带 stride 的 2-D copy 将 q00
直接装入 tight L1。Arch22 Current 保留公开 q01 零输出，Fused 不写内部 q01；C7 从 ld64
row-major relay 的有效 q00 直接生成 tight 32x32 Cube-ready operand，不能先构造不完整的 64x64
NZ 再用 row-major 偏移切片。两架构的 V6/C7 都只传每个 RHS plane 的前 32 行，C7 执行
`q00[32,32] @ RHS_top[32,128]` 并只回写 `M` 行。

Arch22 V3 不把 `X0/X1/B` drain 到 GM，C4 不加载 VCS、不计算/写回 T，C5 也不启动 q10 MMAD。
C4/C5 仍分别消费并以 Control pipe 传递各自 stage-use L0C credit，且 C4 先等待
`C2ScoreL1Free`，所以 C7 不能越过仍在读取 score 的 MTE1。

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

令 `Ctot` 为总 chunk 数、`HV` 为 value/gate head 数、`HK` 为 Q/K head 数、
`R=HV/HK`、`N` 为可用 AIC workgroup 数。Host 必须验证 `HV % HK == 0`；value head
`hv` 固定读取 `hk = hv / R`。

head fallback 的最小分区不是任意四个 HV，而是完整 HK cohort。若 `R<=4`，一个分区最多装
`floor(4/R)` 个完整 cohort；若 `R>4`，一个 cohort 在同一 workgroup 内连续展开成
`ceil(R/4)` 个四-head group。由此一个 HK 对应的 HV 永远不会被两个 workgroup 拆开。
`BuildCorePlan` 使用无乘法溢出的 64-bit balanced half-open range：

```text
if Ctot >= N:
    ChunkOnly
    把 Ctot 个 chunk 均衡分给 N 个 workgroup
    每个已分配 chunk 在本 workgroup 内依次处理全部 HK-cohort partition
else:
    ChunkHeadGroup
    把 Ctot * P 个 (chunk, complete-HK-cohort-pack) 展平任务均衡分配
```

其中 `P` 是上述 cohort pack 数。partition 内再按最多四个 HV 发出连续编号的 local group；
`R>4` 时一个 partition 对应 `ceil(R/4)` 个 local group，但这些 group 仍属于同一 workgroup。
因此只有 chunk 数不足以铺满机器时才按 head 补充分核，
正常路径始终 chunk-first；fallback 也不允许为了铺满核而拆开同一 HK cohort。

每个 cohort 的第一个 HV 是 Q/K cache owner。owner 等待 cache-slot free 后只从公开 GM 搬运
一次 raw Q/K，并在其唯一 V0 VF 中执行所选 norm：L2 模式只在 owner 做一次 L2Norm，Identity
模式保持 MTE2 后的两字节值不变。其他 HV 等待该 generation 的 level-triggered ready，从
workspace cache 重读已舍入 `Qhat/Khat` 到最终 UB 地址，不再执行归一化、FP32 convert 或二次
舍入。V6 也直接回读这份 owner cache，不给非 owner 建 context 副本。每个
HV 在 V0/V1 与 V6 各需要一次 workspace MTE2，是规则14例外；它避免了重复 GM raw input、重复
Vector 归一化、额外 context MTE3、跨 AIV UB 搬位和新增 Stage。cohort 最后一个
HV 的 V6 完成 Q/K MTE2 source read 后发布各自的 `V6RhsReady`；AIC C7 按 cohort 顺序观察
全部 mapped HV 的 ready 后，由 coordinator 归还一次 cache free credit。后续 VF/C7 只消费
UB/RHS 副本，下一 HK 才可以覆盖相同物理 cache slot。禁止让逻辑最后一个 HV 自行 free：
不同 AIV 没有完成顺序，该 head 先完成时，另一 AIV 仍可能正在读取 cache。

一个四 head group 内的 AIV 映射由架构策略冻结，在所有 Vector Stage 保持不变：

```text
Arch35 AIV0: group-local head 0, 1 -> local slot 0, 1
Arch35 AIV1: group-local head 2, 3 -> local slot 0, 1
Arch22 AIV0: group-local head 0, 2 -> local slot 0, 1
Arch22 AIV1: group-local head 1, 3 -> local slot 0, 1
```

Arch35 相邻两个 head group 构成一个八-slot wavefront；Arch22 每个 group 使用 `headId%4` 的四槽
wavefront。workspace 的 Q/K cache 子区与同 slot 的每-HV G/beta/payload 子区也是两个独立 owner：
前者只由 `QkCacheFree/Ready` 管理，允许在 owner 自身 `SlotFree` 后继续存活；后者由
`SlotFree` 管理。private UB、shared UB、AIC L1 与 AIC L0C 同样是独立 owner，不能只用一个
slot/generation 表示。每个 workgroup 的 AIC 与两个 AIV 按完全相同的 `WorkItem` 顺序维护。
Arch35 保持原四域 transaction ticket；Arch22 增加 shared stage-use，并把 L0C 改为 stage-use：

```text
OwnerTicketState:
    workspaceNext[max(4,8)] = {0}
    qkCacheNext[max(4,8)] = {0}
    qkCacheCurrentGeneration[max(4,8)] = {0}
    qkCacheCurrentKey[max(4,8)] = {0}
    qkCacheValid[max(4,8)] = {false}
    localNext[4] = {0}
    sharedNext[2] = {0}
    collectiveNext[2] = {0}       # Arch22 pair0/pair1 mode-0x2 ticket
    l1Next[4] = {0}
    l0cNext[4] = {0}
    l0OperandNext[2] = {0}         # Arch35 only uses bank0

for each active head in deterministic WorkItem order:
    qkHeadId          = headId / (HV/HK)
    qkCacheSlot       = firstHvOf(qkHeadId) % workspaceSlotCount
    qkCacheGeneration = owner ? qkCacheNext[qkCacheSlot]++
                              : currentGeneration(qkCacheSlot, globalChunk, qkHeadId)
    workspaceSlot       = headId % (architecture == Arch22 ? 4 : 8)
    workspaceGeneration = workspaceNext[workspaceSlot]++
    localBankId         = aivId * 2 + aivLocalSlot
    localGeneration     = localNext[localBankId]++
    l1BankId            = groupLocalHead
    l1Generation        = l1Next[l1BankId]++
    l0cBankId           = architecture == Arch22 ? groupLocalHead % 2
                                                   : groupLocalHead
    l0OperandBankId     = architecture == Arch22 ? groupLocalHead % 2 : 0

Arch35 only:
    l0cGeneration       = l0cNext[l0cBankId]++       # C2 through C7

Arch22 in actual execution order:
    sharedGenerations[V01,V6] = sharedNext[aivId]++
    l0cStageGenerations[C2,C4,C5,C7] = l0cNext[l0cBankId]++

Both architectures in actual AIC execution order:
    for each active C2 band of each active head:
        l0OperandGenerations[C2Band[s]] =
            l0OperandNext[l0OperandBankId]++
    if validRows > 32, for each active head:
        l0OperandGenerations[C4] = l0OperandNext[l0OperandBankId]++
    if validRows > 32, for each active head:
        l0OperandGenerations[C5] = l0OperandNext[l0OperandBankId]++
    for each active head:
        l0OperandGenerations[C7] = l0OperandNext[l0OperandBankId]++
```

Arch22 的 shared generation 按 `V01 slot0, V01 slot1, V6 slot0, V6 slot1` 的每-AIV真实顺序
分配；L0C generation 按 Stage-major、pair-wave 次序分配。它们不能在
`HeadTask` 上各存一个 transaction-long generation，否则 head2 会拿到错误的初始 credit 并覆盖
head0 的异步 reader。Arch35 仍由 C7 归还 transaction-long L0C ticket。
L0 operand generation 也按 AIC 的 Stage-major 顺序分配，但只约束 L0A/L0B：Arch35 在唯一
bank 上连续，Arch22 在两条 physical lane 上分别连续。未执行的 C2 tail band、`M<=32` 时跳过的
C4/C5 都不占 generation；一个 epoch 内所有 L0 operand descriptor 必须具有相同 bank 和 generation。
Arch22 的 `pairCollectiveGenerations[2]` 则按 pair owner 独立连续：partial pair 仍分配一张 ticket，
完全空 pair 不分配；inactive head 不递增 per-head owner，但其 AIV 必须参与 partial pair dummy arrive。
inactive head 不递增 workspace/QK-cache/local/shared/L1/L0C 等 per-head/use 数组；partial pair 只递增一次
pair collective ticket，完全空 pair 不递增。因此任意 `HV % 4`、任意合法 `HV/HK`、任意
workgroup balanced begin、ChunkOnly 与 ChunkHeadGroup fallback 都从每个物理 owner 的 ticket0
开始且严格连续。`HV=1..17` 的 ratio-1 路径以及 `R=2/3/4/8` cohort、两种分核模式、不同非零
begin 由 host 合同覆盖。

除 level-triggered `QkCacheReady` 外，owner credit 都采用点对点 ticket：初始化只预置
`Free[0]`；当前 owner 先
`Wait(Free[currentGeneration])`，最后消费者只发布一次 `Set(Free[currentGeneration+1])`，下一 owner
的 generation 恰好加一。禁止 Set 同代，否则同一个 token 会同时充当 acquire 前置和重复发布。
Arch35 由 V0/V6 负责 local UB ticket，C2/C7 负责 L1 和 transaction-long L0C ticket。Arch22 的
V0/V1 与 V6 分别闭环 shared ticket，V3 只使用 private bank；每个 Cube Stage 的最后 Fixpipe reader分别闭环本 Stage
L0C ticket。C7 最终 W/U Fixpipe 完成负责 workspace slot ticket。其余边按真实资源域选择 ticket：context/payload/transaction 使用
`(workspaceSlot, workspaceGeneration)`，beta/raw/local-source 使用
`(localBankId, localGeneration)`，T/Akk/C2-L1 使用 `(l1BankId, l1Generation)`；Arch22 shared
使用 `(sharedArenaId, sharedGenerations[use])`，其 L0C 使用
`(l0cBankId,l0cStageGenerations[stage])`。V6/C7 这类跨域
汇聚必须逐条 Wait，不能拿单一 ID 的 Join 伪装成同一计数器。

GM slot 不增加 UB：Arch35 是两份 `112 KiB MAIN + 12 KiB AUX`；Arch22 是两份72 KiB private
bank加一份40 KiB shared arena。上一 workspace generation 的 output drain 与 `SlotFree` 未闭环前，
禁止新 generation 复用同一 slot 的每-HV G/beta/payload；即使 `SlotFree` 已归还，只要
`QkCacheFree` 未闭环，任何后续事务仍不得覆盖该 slot 的 Qhat/Khat cache 子区。上一 local
generation 未归还前也禁止覆盖同一物理 UB bank。
上一 L0C generation 未由对应架构定义的最后 Fixpipe reader 归还前，不得写入同一物理 lane。

## Ready/free 合同

伪代码同时具名 data-ready 与 storage-free，后续设备实现必须使用有界 credit/反向 free，
不能依赖 block 启动顺序，也不能对同一 flag 连续 set 而没有消费。

下面第一张表是 Arch35 合同：

| Producer -> consumer | Arch35 必须保留的边 |
| --- | --- |
| slot owner -> `V0` | `SlotFree(workspaceSlot, workspaceGeneration)` 与独立的 `LocalBankFree(localBankId, localGeneration)`；V6 最后使用 UB 后发布 `LocalBankFree(localBankId, localGeneration+1)` |
| L1 owner -> `C2` | `L1BankFree(l1BankId, l1Generation)`；C7 最后读取本 lane/Akk resident 后发布 `L1BankFree(l1BankId, l1Generation+1)` |
| L0C owner -> `C2` | `L0cBankFree(l0cBankId, l0cGeneration)`；C7 最后一个 U Fixpipe reader 完成后发布 `L0cBankFree(l0cBankId, l0cGeneration+1)` |
| `V0` -> `V1` | local `V0ExportDone`；V1 只能在 V0 的所有 MTE3 source reader 完成后 overlay 同一 MAIN source |
| HK cache owner -> mapped HV `V0/V6` | owner 等待 `QkCacheFree(slot,generation)`，MTE3 完成 Qhat/Khat 后发布一次 level-triggered `QkCacheReady`；同 cohort 的非 owner V0 可多次 acquire，但不能消费或清除 ready；每个 HV 的 `V0ContextReady` 传递其 V0 已 acquire cache-ready 的先行关系，V6 直接回读 owner cache |
| `V0` -> `V3` | `V0BetaReady` 保留 `betaEff`；V3/V6 的后续 local 生命周期最终由 `LocalBankFree` next-ticket 闭环 |
| `V0` -> `V6` | workspace `V0ContextReady`；它表示本 HV 的所选 G source 已写完，且其 V0 已 acquire owner Q/K cache：Fused 为 context G，Current 为公开 `gk`；只由 V6 消费，不是 V1 的 local 前置 |
| `V1` UB source -> `C2` raw destination | score 的最后一个 MTE3 source reader 完成后发布 `V1MainSourceFree`；C2 写 raw 前还要持有对应 generation 的 `C2RawDstFree` credit |
| `V1` -> `C2` | `V1ScoreReady`；C2 一次 MTE2 读完 GM payload 后返回 `C2ScorePayloadFree`，最后一个 MTE1 reader 完成后另发 `C2ScoreL1Free` |
| `C2` -> `V3` | 来自真实 raw-score producer pipe 的 `C2RawReady`；`C2RawDstFree` 只由当前 V1 source drain 发布并被当前 C2 消费，跨代物理 UB 复用统一由 `LocalBankFree` 管理 |
| `V3` -> `C4` | `M>32` 为 VCS 与稳定 Akk source 完成后的 `V3VcsReady`；`M<=32` 只保证 q00 source 已提交。Fused 的 stable Akk source 是 tight payload，Current 是公开 ld64 AkkOut |
| `V3` -> `V6` | 所有 V3 MTE3 source reader 完成后的独立 `V3LocalSourceFree` |
| `C4` -> `C5` | `M>32` 保留 Fixpipe `C4TReady` 与 resident `C4AkkPrepReady` 两条边；`M<=32` 不产生 T，C5 以 Control wait/set 把 Akk ready 继续传给 C7 |
| `C4` -> `V6` | C4 一次读完全部所选 payload 段后的 `C4PayloadFree` |
| `C5` -> `C7` | `M>32` 为最终 resident quadrant 提交后的 `C5AkkReady`；`M<=32` 为 C4 tight q00 ready 的 Control 传递 |
| `V6` -> `C7` | 对应 32/64 行 RHS drain 后的 `V6RhsReady` |
| all mapped `V6` -> AIC C7 -> next HK owner | 每个 HV 在 Qhat/Khat MTE2 source read、VF 和 RHS MTE3 完成后发布 `V6RhsReady`；Arch35 C7 依次等待 cohort 全部 head，Arch22 C7 依次等待覆盖 cohort 的全部 pair collective；AIC coordinator 观察完整集合后只发布一次 `QkCacheFree(slot,generation+1)`。逻辑最后 HV 不能自行 free；ready/free 是 generation 状态，不能用一次性单消费者 flag 冒充广播 cache |
| `C7` -> owner | C7 最后一个 MTE1 reader 完成后发布 `L1BankFree(l1BankId, l1Generation+1)`；最终 U Fixpipe reader 完成后发布 `L0cBankFree(l0cBankId, l0cGeneration+1)`，并直接发布只覆盖每-HV G/beta/payload 子区的 `SlotFree(workspaceSlot, workspaceGeneration+1)`；Q/K cache 子区只走上面的 cohort free |

Arch22 保留相同的数学 data-ready 名称，但资源 owner 和 relay 语义不同：

| Producer -> consumer | Arch22 必须保留的边 |
| --- | --- |
| owner -> `V0/V1` | 两个 AIV 各对本 pair 调一次 `AivWaitPair(SlotFree)`，active head 再等自己的 `LocalBankFree + SharedArenaFree(V01)`；V0/V1 共持有 shared ticket，V1 最后 reader 后发布 next-ticket |
| HK cache owner -> mapped HV `V0/V6` | 与 Arch35 相同，使用独立 `QkCacheFree/Ready` generation；各 mapped HV 的 V6 MTE2 是 cache reader；Arch22 pair collective 同时汇聚两个 AIV，C7 按 pair 顺序覆盖完整 cohort 后才发布 free。cache ready 仍是跨 pair 的 level-triggered state，不能由单个 pair collective 替代 |
| owner -> `V3` / `V6` | V3 只使用 private bank，不等待 shared credit；V6 等待 `SharedArenaFree(V6)`，最后 shared reader 后发布 next-ticket；最后一次 private MTE3 drain 后发布 `LocalBankFree+1` |
| owner -> each Cube Stage | C2/C4/C5/C7 分别等待本 Stage 的 `L0cBankFree(l0cStageGeneration)`，本 Stage 最后 Fixpipe reader 立即发布 next-ticket；Arch22 C4/C5 的 `M<=32` 空计算分支改由 Control wait/set 传递 credit；两条物理 lane 不能跨 Stage 长持有 |
| `V0` -> `V1/V3/V6` | `V0ExportDone` 保护 private Q/K overlay；workspace `V0ContextReady/V0BetaReady` 保护本 HV 的所选 G source（Fused context 或 Current `gk`）、betaEff，并传递 owner Q/K cache ready 的先行关系；V6 直接回读 owner cache |
| `V1` -> `C2` | score drain 后两个 AIV 各 `AivArrivePair(V1ScoreReady)` 一次，inactive partner 发 dummy；AIC 对该 pair `AicWaitPair` 一次，再读取各 active head 的 72 KiB score |
| `C2` -> `V3` | pair 内所有 active raw Fixpipe->GM 完成后，AIC `AicPublishPair(C2RawReady)` 一次；两个 AIV 各 wait 一次。Arch22 不使用 local `C2RawDstFree` 或 L0C->UB |
| `V3` -> `C4/C5` | VCS/Akk drain 后两个 AIV 各 `AivArrivePair(V3VcsReady)` 一次，inactive partner发 dummy；C4 每 pair wait 一次，读完 pair 内 VCS 后 `AicPublishPair(C4PayloadFree)` 一次，两个 AIV各wait一次 |
| `C4` -> `C5` | `M>32` 时 FP32 T Fixpipe->GM 完成后发布 workspace-ticketed `C4TReady`，X1 MTE2 完成后发布 L1-ticketed `C4AkkPrepReady`；C5 等待两者并重载 T；`M<=32` 只通过 Control 传递 owner |
| `C5` -> `C7` | `M>32` 时 q10 Fixpipe 写入 row-major 2-byte Akk relay 后发布 `C5AkkReady`；`M<=32` 的 Control ready 表示 V3 已完成的 top-only Akk 可读 |
| `V6` -> `C7` | RHS drain 后两个 AIV 各 `AivArrivePair(V6RhsReady)` 一次，inactive partner发 dummy；C7 每 pair wait 一次 |
| `C7` -> owner | pair 内两个 active head 的最终 U Fixpipe 都完成后，AIC `AicPublishPair(SlotFree,next pair ticket)` 一次并由两个 AIV各wait一次；per-head L1 与 stage-use L0C credit仍各自闭环 |

Arch22 的逻辑 `SyncPoint` 不逐一占物理 flag。**PROPOSED** adapter 只设三类有界、带 reverse ACK
的有序 channel：AIV->AIC phase（score/VCS/RHS ready）、AIC->AIV phase（raw ready/payload free）和
C7->V0 slot credit。score/VCS/RHS 是两个 AIV 各 arrive 一次、AIC 每 pair wait 一次；raw/payload/slot
是 AIC 每 pair publish 一次、两个 AIV 各 wait 一次。禁止同一 pair 的两个 head 分别操作同一
collective flag。每条 channel 只在上一逻辑 token 已 wait/ACK 后复用；reverse ACK 只防计数器
溢出，不能替代 workspace/shared/L0C 的 storage-free ticket。mode-0x2 collective 的两个 AIV
必须具有完全相同的 set/wait 次数；tail pair 的 inactive AIV 发 dummy token，完全不存在的 pair wave
则三方一致跳过。物理 flag ID、counter depth、与 Matmul 内部 flag 的冲突仍需 CANN 9.1 最小编译验证。

`V6` 必须三路汇聚 `V0ContextReady + V3LocalSourceFree + C4PayloadFree`；`C7` 必须两路汇聚
`C5AkkReady + V6RhsReady`。context ready 不等于 local bank free，VCS ready 也不等于 payload free，
即使 profiling 中某个生产者总是先完成，也不能合并这些状态。

当前实现不引入额外的 output-drain coordinator 或 C7 中间完成状态。每条 ready 只覆盖其具名
consumer 数据：`V3VcsReady` 覆盖 VCS 与 stable Akk source，Fused ABI 不必等待独立的公开 Aqk
drain；`V3LocalSourceFree` 在 V3 全部公开输出完成后保护 V6 的 local bank 复用；`V6RhsReady` 与
`C5AkkReady` 分别在 RHS 和 q10 的所有启用写出完成后发布。C7 等待后两条边，再在最终 U
Fixpipe 完成后发布 L0C 与 workspace 两个 next-ticket。若以后增加不受这些具名 ready 覆盖的
输出，必须重开 coordinator 合同，不能直接沿用本闭环。

本目录禁止 `PipeBarrier<PIPE_ALL>()`，也没有任何真实 barrier 调用。未来实现只有在证明是同核
V-pipe RAW/WAR/WAW 时才能考虑 `PipeBarrier<PIPE_V>()`；它不能替代 MTE/Cube/Fixpipe 的硬事件，
更不能替代核间 ready/free。

每个 Vector Stage 还显式保留命名的本核 `MTE2 -> V` input-ready 与 `V -> MTE3`
output-ready 边；每个有数据搬运的 Cube Stage 保留 `MTE2 -> MTE1` operand-ready、
`Cube/M -> MTE1` L0A/L0B operand-release 和 `Cube/M -> Fixpipe` result-ready 边。每条 operand
release 必须晚于当前 `(l0OperandBankId,l0OperandGeneration)` 的最后 MMAD reader。C2 每个 band
的单次 `MmadRowStackedLhs` 完成后发布一次 operand-release，C4/C5 各一次，C7 的两个 RHS product 都完成
后发布一次，才能让下一 band/head/stage 的 MTE1 覆盖同一 lane。Arch22 另有 payload overlay 的
MTE2/Fixpipe 反向保护、条件化 Fixpipe->MTE2 relay 和 C7 fill/load WAW。它们目前都是
**PROPOSED** helper，正式编码时必须用目标 CANN 版本支持的成对 HardEvent 落地，不能因源码
调用顺序或跨核 ticket 已存在而省略。

## 资源和 API 边界

- UB 只给 Vector 使用。一个 Vector Stage 的完整输入、输出、scratch 和异步 source 生命周期必须
  同时装入固定布局，不能靠隐藏 spill、搬位或分 tile 绕过。
- L1 只给 Cube 使用。跨 Cube Stage resident 必须预留四份并直接写最终地址，禁止 L1 内整理或搬位。
- Vector 结果供 Cube 使用时必须经过 GM/workspace；Cube 结果供 Vector 使用时必须有两份匹配 UB
  destination，或显式选择并登记 GM relay fallback。
- AIC/AIV 无依赖操作没有隐式执行顺序，live region 不能因时间线“看起来错开”就重叠。
- workspace context 固定预留
  `Qhat[0x0000,0x4000) + Khat[0x4000,0x8000) + G[0x8200,0x10200)`。G region 只在
  Fused ABI 存有效行；Current ABI 不写该区，V6 复用公开 `gk`，避免同一 Vector 数据重复
  GM write/read。Qhat/Khat 的槽位容量固定为 64 行，但 V0 写回和 V6 回读都只传
  `validRows * 128 * 2 Byte`，tail padding 不产生无消费者的 GM 流量。Arch35 的
  `[0x8000,0x8200)` 全为 hard pad，`betaEff` 在 AUX 常驻到 V6；Arch22 用
  `[0x8000,0x8100)` 保存有效 FP32 betaEff，剩余 `[0x8100,0x8200)` 仍为 hard pad。
- `AkkStorage::Fp32Internal` 为 **BLOCKED**，目前只是通过容量推导的 L1 布局候选；FP32 Akk 与 2-byte RHS 的
  C7 Cube operand 组合及对应目标 CANN API 尚未证明。当前 `RunV0/V1/V3/V6` 与
  `RunC2/C4/C5/C7` 伪 dispatch 全部只接受 `AkkStorage::TwoByteAbi`，不能把 FP32 candidate
  当作已支持模板。
- `InputStorage`、`valueStorage` 与 `ScoreStorage` 是三个独立的必要语义轴，都不是已冻结的
  数值 key。前两者分别选择 q/k 与 v/u 的 FP16/BF16 dtype，后者选择 score dtype；Host 必须
  按上面的允许映射拒绝
  未实例化组合。V1 必须据 `ScoreStorage` 选择 BF16
  `[-126,120]` 或 FP16 `[-80,80]` 的 base-2 指数范围。raw gate 使用独立的
  `GateStorage::{Fp16,Bf16,Fp32}`；前两者只共享 2-byte UB 布局，V0 仍按准确枚举解码并转
  FP32，不能用一个 `TwoByte` 宽度标记混淆 FP16/BF16，也不能让 gate dtype 驱动 score 分支。
- `useExp2` 必须随公开 Prepare 参数进入 Host dispatch，并分别实例化两架构 V1/V6 的
  `EvaluatePow2<true/false>`；不得只在接口或 TilingKey 留一个未被 kernel 消费的字段。
- workspace 精确 offset、公开输出集合、Akk cast 边界、Arch35 Fixpipe 直写配对 AIV UB、Arch22
  mode-0x2 collective/GM relay/ND2NZ、split L0C tile Fixpipe、flag/event 分配、Matmul/VF API、
  TilingKey 编码和 launch ABI
  均为 **PROPOSED**，必须结合目标 CANN 官方文档、
  随包头文件和实现源码确认后才能编码。

## Host 语法检查

本目录可作为普通 C++17 做静态语法检查，不需要 CANN：

```sh
g++ -std=c++17 -Wall -Wextra -Werror -fsyntax-only \
  fla/ops/ascendc/kda/chunk_kda_fwd_prepare/op_kernel/pseudocode/chunk_kda_fwd_prepare.cpp
```

本目录另有可执行 host 合同测试。Arch22 部分运行真实伪调度，覆盖 `H=1..17` 的三个连续
chunk、`M=16/32/33/64` 边界、Current/Fused ABI，以及 chunk-only/head fallback 的多 workgroup
映射。测试为两个 AIV 和 AIC 分别记录完整事件流，按实际源码顺序逐项检查普通 owner
wait/set、pair arrive/wait/publish、inactive dummy、SyncPoint、stage、pipe、owner 与 generation；
独立 local-edge trace 同时检查 `MTE2/V/MTE1/Cube/Fixpipe/MTE3` 的命名依赖位置。Arch22/Arch35
都记录 Current/Fused 在 `M=16/32/33/64` 下的 Vector/Cube 外部操作轨迹，检查 G source、
Qhat/Khat 有效行搬运、Akk quadrant writer、q01 fill/relay 分支、`ld64 -> tight`、partial fill、
top-only RHS 收缩、MMAD 操作数/模式，以及 ready 晚于对应最后写入。VF 内部逐元素循环仍由
共享的语义伪代码描述；合同另用可执行 probe 验证 `useExp2=true` 只走截断 Exp2、false 只走
截断后 FP32 `x*ln2` 再 Exp。外部 OperationTrace 不把一次 VF 展开成第二条执行流：

```sh
g++ -std=c++17 -Wall -Wextra -Werror -pedantic \
  fla/ops/ascendc/kda/chunk_kda_fwd_prepare/op_kernel/pseudocode/chunk_kda_fwd_prepare.cpp \
  fla/ops/ascendc/kda/chunk_kda_fwd_prepare/op_kernel/pseudocode/chunk_kda_fwd_prepare_contract_test.cpp \
  -o chunk_kda_fwd_prepare_contract_test
./chunk_kda_fwd_prepare_contract_test
```

该命令通过只说明设计伪代码内部是合法 C++17；不代表 A2/A3/A5 编译、device link、功能、精度、
sanitizer 或性能验证通过。
