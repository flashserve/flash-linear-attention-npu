# Chunk KDA Forward Prepare 内核伪代码

本目录描述拆分后的 `chunk_kda_fwd_prepare`，计算范围是：

```text
L2 norm + gate cumsum + prepare + post-WU
```

它是面向 A2/A3（Arch22）和 A5（Arch35）的设备代码设计稿，尚未接入算子定义、
Host Tiling、CMake、aclnn 或 Python API。伪代码直接使用真实 Ascend C API 的写法，
目的是让搬运、同步和计算顺序能够从代码现场读出；未由目标 CANN 版本确认的参数在调用旁
保留 `TODO`，不能据此声称当前目录已经可构建或可运行。

## 代码结构

```text
pseudocode/
|-- README.md
|-- chunk_kda_fwd_prepare_tiling_key.h  # 编译期语义轴和 runtime tiling
|-- chunk_kda_fwd_prepare_policy.h      # 静态 UB/L1/workspace 地址
|-- chunk_kda_fwd_prepare_struct.h      # GM 参数和 chunk 索引
|-- chunk_kda_fwd_prepare_utils.h       # chunk-first 分核和地址计算
|-- chunk_kda_fwd_prepare.cpp           # MIX AIC/AIV 编译期分架构入口
|-- chunk_kda_fwd_prepare_layout_test.cpp
|-- arch35/
|   |-- chunk_kda_fwd_prepare_vec.h
|   `-- chunk_kda_fwd_prepare_cube.h
`-- arch22/
    |-- chunk_kda_fwd_prepare_vec.h
    `-- chunk_kda_fwd_prepare_cube.h
```

代码只保留 Kernel 类、八个 Stage 函数以及少量任务/地址函数；每个 Stage 内直接出现
`DataCopy/DataCopyPad`、同步、一次向量函数（VF，Vector Function）或
`LoadData/Mmad/Fixpipe`。

架构由编译器宏静态选择：

```cpp
#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
// Arch35 / A5
#else
// Arch22 / A2/A3
#endif
```

TilingData 只承载 shape、调度和标量参数；设备架构不进入运行时分支。

`USE_EXP2` 是独立的编译期布尔轴，`false` 和 `true` 都必须生成实例。正式
TilingKey 声明应与 `chunk_fwd_h` 一致使用
`ASCENDC_TPL_BOOL_DECL(USE_EXP2, 0, 1)`；`q/k/v`、score 操作数及
`Aqk/Akk/w/u/qg/kg/qg_scaled/q_hat/k_hat` 固定为 BF16，
`gk/q_rstd/k_rstd/beta_eff` 固定为 FP32，gate/beta 只允许 FP32 或 BF16。
当前 gate/beta dtype 编码与其他模式轴尚未冻结，
因此本伪代码只冻结该轴的 `0/1` 编码，不提交一个参数不完整的注册声明。
正式 selector 的笛卡尔积中，每个合法的 dtype/模式组合都必须由
`SEL_EXP` 一类宏同时展开 `USE_EXP2=0` 和 `USE_EXP2=1`，op_host 再把公开
`use_exp2` 属性原样传给 `GET_TPL_TILING_KEY`。当前 host 小测试只验证
`PrecomputedStep` policy 能选择两份 `ExpDomainTraits`，以及其中
step 缩放、base-2 边界换算和 `Exp` 输入倍率的纯数值合同；它不编译架构头，也不验证
设备 `Muls/Exp/Cast`、BF16 舍入或尚未接入的 op_host 可达性。
现有 `chunk_gated_delta_rule_fwd_prepare` 参考实现仍拦截 `false`，不能作为双分支
依据；本设计新增的 `false` 路径以已支持双值的 `chunk_fwd_h` TilingKey 方式为准。

## 数学定义

对每个长度不超过 64、维度为 128 的 chunk：

```text
q_rstd[i] = 1/sqrt(sum_d(Q[i,d]^2) + epsilon), L2
             1,                                      Identity
k_rstd[i] = 1/sqrt(sum_d(K[i,d]^2) + epsilon), L2
             1,                                      Identity
Qhat = round_BF16(Q * q_rstd), L2
       Q,                       Identity
Khat = round_BF16(K * k_rstd), L2
       K,                       Identity
beta_eff = beta,               Raw
           sigmoid(beta),      Sigmoid
           2*sigmoid(beta),    TwoSigmoid

x[i,d] = raw_gate[i,d] + optional_dt_bias[d]
deltaG_ln[i,d] = raw_gate[i,d],                         PrecomputedStep
               = -exp(a_log) * softplus(x[i,d]),        Softplus
               = lower_bound / (1+exp(-exp(a_log)*x)),  SafeSigmoid
gateScale = 1/ln(2), USE_EXP2=true
          = 1,       USE_EXP2=false
G[i,d] = sum(t=0..i, deltaG_ln[t,d] * gateScale)
E(x) = exp(x * ln(2)), USE_EXP2=true
     = exp(x),         USE_EXP2=false

qg = round_BF16(Qhat * E(G))
qg_scaled = round_BF16(float(qg) * scale)
kg = round_BF16(Khat * E(Glast - G))
```

公开输出固定为
`gk/Aqk/Akk/w/u/qg/kg/qg_scaled/q_hat/k_hat/q_rstd/k_rstd/beta_eff`。
`qg` 与 `qg_scaled` 是两块独立输出，后者必须从已经写成 BF16 的 `qg` 回读到
FP32 后再乘 `scale`，不能合并两次舍入。`K_beta_g/V_beta` 只在 V6 到 C7
之间中转，不是公开输出。

`QkNormMode::Identity` 不允许跳过五个新增输出的写回：`q_hat/k_hat` 分别等于
BF16 输入 Q/K，`q_rstd/k_rstd` 的所有有效 token 固定写 FP32 `1`。
`BetaMode::Raw` 也不表示直接透传输入地址，而是把 beta 转为 FP32 后写入独立的
`beta_eff` 输出。尾 chunk 只写真实 token 对应的行，不把 UB 中的补零行写出张量边界。

`PrecomputedStep` 的输入是自然对数域、尚未累计的单 token `deltaG_ln`。该模式
不读取或应用 `dt_bias/A_log`，但 V0 仍按 `USE_EXP2` 缩放每个 step，并执行
cumsum；禁止把已经累计的 `gk` 或已经换到 log2 域的 step 再传入此模式。

S 固定为 4，四个参考行是每个 16 行子块的中点：

```text
rows(s) = [16*s, 16*(s+1))
a_s = 16*s
b_s = min(validRows, a_s+16)
r_s = floor((a_s+b_s)/2)
Gref[s,d] = G[r_s,d]

Qplus[i,d] = Qhat[i,d] * E(G[i,d] - Gref[s(i),d])
Kplus[i,d] = Khat[i,d] * E(G[i,d] - Gref[s(i),d])
Kminus[s,j,d] = Khat[j,d] * E(Gref[s,d] - G[j,d])
```

`Gref` 只有 `4 x 128 x FP32 = 2 KiB`。它在 VF 中逐行广播，不物化为
`[64,128]`。额外空间来自四个不同参考点对应的 `Kminus` 有效前缀，而不是
`Gref/Glast` 广播。

有效前缀按因果依赖保存：

| 数据 | 物理 shape | 大小 |
| --- | --- | ---: |
| `Qplus` | `[64,128]` | 16 KiB |
| `Kplus` | `[64,128]` | 16 KiB |
| `Kminus[0]` | `[16,128]` | 4 KiB |
| `Kminus[1]` | `[32,128]` | 8 KiB |
| `Kminus[2]` | `[48,128]` | 12 KiB |
| `Kminus[3]` | `[64,128]` | 16 KiB |
| 合计 | | 72 KiB |

这不是四份完整 `Kminus[64,128]`。每个 `s` 只保留该 16 行 query band
能够访问的 key 前缀。

## 输出分类

本伪代码固定把以下 13 个结果全部写回 GM：

```text
gk, Aqk, Akk, w, u, qg, kg, qg_scaled,
q_hat, k_hat, q_rstd, k_rstd, beta_eff
```

“公开输出”只表示它们都位于算子边界；按实际消费者分类时，各类允许重叠：

| 类别 | 数据 | 实际用途 |
| --- | --- | --- |
| 后续正向使用 | FwdH：`gk/w/u/kg`；Finalize：`Aqk/qg_scaled` | FwdH 计算 `v_new` 与 chunk 状态递推；Finalize 计算 `attn_out=qg_scaled@h+Aqk@v_new` |
| 反向使用或保存 | `q_hat/k_hat/q_rstd/k_rstd/beta_eff/Aqk/Akk/gk/w/qg/kg` | 前五项是反向保存量；`Aqk/Akk` 始终保留，其余按 gate 与重计算策略保存或重算；启用 L2Norm 时反向消费 `q_rstd/k_rstd` |
| 用户可选状态结果 | `hOut/final_state` | 由后续 FwdH 产生，不属于 Prepare 的 13 个输出 |

其中 `qg_scaled` 只服务正向 Finalize，`Akk/qg` 在 Prepare 已包含 Post-WU 的边界
之后不再被正向消费。内部 head-major `hCompute` 即使不导出也必须存在；它供 Finalize
使用，不能与可选公开的 `hOut` 混为一类。`u` 服务 FwdH，并随禁用重计算路径兼容
保留，但当前反向不读取。

所有 Prepare 输出均固定为 head-major；输入的 `inputSequenceMajor` 不改变输出布局：

| 输出 | dense shape | varlen shape | dtype |
| --- | --- | --- | --- |
| `gk/w/qg/kg/qg_scaled` | `[B,H_v,T,128]` | `[H_v,T,128]` | `gk` 为 FP32，其余为 BF16 |
| `u` | `[B,H_v,T,128]` | `[H_v,T,128]` | BF16 |
| `Aqk/Akk` | `[B,H_v,T,64]` | `[H_v,T,64]` | BF16 |
| `q_hat/k_hat` | `[B,H_k,T,128]` | `[H_k,T,128]` | BF16 |
| `q_rstd/k_rstd` | `[B,H_k,T]` | `[H_k,T]` | FP32 |
| `beta_eff` | `[B,H_v,T]` | `[H_v,T]` | FP32 |

GVA 中 `q_hat/k_hat/q_rstd/k_rstd` 按 `H_k` 编址。每个 Q/K head 映射到一组
连续 value head，只有该 QK 头组的首个 value head 是这四个公开输出的 owner；其他
value head 仍可为 V1/V6 生成各自的内部 Qhat/Khat context，但不得写同一公开地址。
`beta_eff` 按 `H_v` 编址，每个 value head 都独立写回。Host Tiling 仍必须保证完整
QK 头组不跨 workgroup。

## 八个 Stage

每个 Stage 只能包含 Cube 或 Vector 操作之一。Vector Stage 只调用一次 VF，不按 token、
子块或硬件 tile 分 pass；Cube Stage 可以在编译期展开独立 MMAD，但所有输入必须在 Stage
入口前 ready。

| Stage | 核 | 计算 |
| --- | --- | --- |
| `V0` | Vector | Q/K L2 norm、beta 变换、gate 变换、cumsum，生成 `Qhat/Khat/qRstd/kRstd/G/Gref/Glast/betaEff`，并写回五个新增公开输出 |
| `V1` | Vector | 一次 VF 生成 S=4 的 `Qplus/Kplus/Kminus`，写 72 KiB payload |
| `C2` | Cube | 四个 band 的 `rawAqk=Qplus@Kminus^T` 和 `rawAkk=Kplus@Kminus^T` |
| `V3` | Vector | 因果 mask、beta、两个 32x32 叶子逆，生成 `Aqk/B/X0/X1/negX1` |
| `C4` | Cube | `M>32` 时计算 `T=B@X0` |
| `C5` | Cube | `M>32` 时计算下左象限 `q10=negX1@T` |
| `V6` | Vector | 生成 `qg/qg_scaled/kg/K_beta_g/V_beta` |
| `C7` | Cube | `W=Akk@K_beta_g`、`U=Akk@V_beta` |

Arch35 的 C4/C5 按两轮提交：

```text
C4(head0), C4(head1), C4(head2), C4(head3)
C5(head0), C5(head1), C5(head2), C5(head3)
```

C5 仍通过同一 head 的 L1 Mutex 等待 C4 的 Fixpipe 把 `T` 写入 L1，但不会先于
其他 head 的独立 C4 排进 MTE1 队列。这样下一 head 的 C4 MTE1/MMAD 可以与上一 head
的 C4 Fixpipe 排空重叠；AIC 仍只有一条 M 管线，不表示多个 MMAD 同时执行。C4 在
MTE2 搬完 `B/X0/negX1/Akk` 后即发布 workspace 可复用信号，C4/C5 随后只读各 head
独立的 L1 常驻区，因此 AIV 的 V6 可以并行覆盖原 payload。Arch22 原本就是先完成
全部 C4、再完成全部 C5，并用每 head 的 `tReady` EventID 保证相同依赖。

八段不能压成六段：

- `V0` 的 112 KiB 输入/上下文与 `V1` 的 72 KiB score 输出不能同时保持完整生命周期。
- `C2` 的结果必须先交给 `V3`，Cube 不能消费同一 Cube Stage 的新结果。
- `C4` 产生 `T`，`C5` 才能消费。
- `C7` 同时依赖 `C5` 的最终 `Akk` 和独立 `V6` 的两个右操作数。

### C2 的四次 MMAD

每个 band 把两个独立左操作数沿行堆叠：

```text
[ Qplus[band] ] [32,128]
[ Kplus[band] ]          @ Kminus[s]^T [128,Ns]

    = [ rawAqk[band] ] [32,Ns]
      [ rawAkk[band] ]
```

| API 次序 | Q/K 输出行 | `Ns` | 覆盖的两个数学乘积 |
| ---: | --- | ---: | --- |
| 1 | `0:16` | 16 | `Qplus[0:16]@Kminus[0]^T`、`Kplus[0:16]@Kminus[0]^T` |
| 2 | `16:32` | 32 | `Qplus[16:32]@Kminus[1]^T`、`Kplus[16:32]@Kminus[1]^T` |
| 3 | `32:48` | 48 | `Qplus[32:48]@Kminus[2]^T`、`Kplus[32:48]@Kminus[2]^T` |
| 4 | `48:64` | 64 | `Qplus[48:64]@Kminus[3]^T`、`Kplus[48:64]@Kminus[3]^T` |

所以一个 full head 是 4 次 `Mmad` API 提交、8 个数学矩阵乘积。它不是
`(64/16)^2=16` 次独立计算；每次的 N 直接覆盖当前 band 所需的完整因果 key 前缀。
若目标版本不能把上下两个 16 行左操作数装入同一 32 行 L0A，则退化为 8 次提交，
不能在未验证前把“四次提交”写成正式实现结论。

## V0 后的释放点

V0 的一次 VF 和回写完成后：

| 数据 | V0 后是否保留 | 原因 |
| --- | --- | --- |
| raw Q/K | 否 | 已生成 `Qhat/Khat` |
| raw gate | 否 | 已生成 `G` |
| norm reduction work | 否 | 已生成 FP32 `qRstd/kRstd`；其临时归约区可以释放 |
| `dt_bias/A_log` | 否 | gate 变换完成 |
| scan scratch/carry | 否 | `G/Glast/Gref` 已完成 |
| UB 中的 `Qhat/Khat` | 是到 V1 | V1 原址消费并覆盖为 Qplus/Kplus |
| workspace 中的 `Qhat/Khat` context | 是到 V6 | V6 回读；公开 `q_hat/k_hat` 则作为反向保存量持续存在 |
| UB 中的 `qRstd/kRstd` | 否 | V0 的 MTE3 写回完成后没有本算子内部消费者 |
| `G/Gref` | 是到 V1 | S=4 score 需要 |
| 公开 `gk` | 是 | V6 从 GM 回读，且供后续算子使用 |
| 内部 `betaEff` | 是到 V6 | Arch22 放 workspace，Arch35 放每 head 状态区；公开 `beta_eff` 作为反向保存量持续存在 |

释放表示该静态地址在最后一个异步 reader 完成后可以换义，不表示在 UB 内移动数据。

## 分核

Prepare 没有 chunk 间依赖，优先只按 chunk 分核：

```text
totalChunks >= usedCoreNum:
    每个 workgroup 取得一段连续 chunk，遍历该 chunk 的全部 head
```

只有 chunk 数不足时才增加 head partition：

```text
totalChunks < usedCoreNum:
    workItem = chunk x headPartition
```

`headsPerPartition` 由 Host 以完整 HK（Q/K head）对应的 QK 头组为单位生成，不能切开共享同一 Q/K
源的 HV（value/gate head）集合。每个 AIC wave 最多处理 4 个 HV。Arch35 中 AIV0
处理 group-local head 0/1，AIV1 处理 2/3；Arch22 中 AIV0 处理 0/2，
AIV1 处理 1/3，以两次 pair wave 复用 40 KiB 共享区。

当前 GVA 伪代码仍按 HV slot 独立搬运并归一化对应的 Q/K。同一 HK 对应的 QK 头组跨越
两个 AIV 时没有可共享的 UB，同时还要保持单次 VF、静态 slot 和不做 UB 位置移动，
因此这里明确采用“条件无法同时闭合时允许 GM 重读”的兜底规则。正式性能实现若要
去掉这部分重复，必须在不破坏单次 VF 和静态地址合同的前提下增加跨 AIV 数据共享与
成对同步；当前设计只冻结 GM 重读路径。

`chunk_fwd_h` 有 chunk 间状态依赖，所以采用 head 分核；这个理由不能反推到
Prepare/Finalize。

## 静态 UB

两架构都使用编译期固定地址，不使用 `TQue` 动态分配。不同 Stage 可以让同一地址换义，
前提是上一语义的最后一个异步 reader 已经完成。任何 Stage 都不在 UB 内移动数据。

### Arch35

每个 AIV 的 248 KiB：

| 地址 | 大小 | 所有者 |
| --- | ---: | --- |
| `[0x00000,0x1C000)` | 112 KiB | local head 0 主计算区 |
| `[0x1C000,0x38000)` | 112 KiB | local head 1 主计算区 |
| `[0x38000,0x3B000)` | 12 KiB | local head 0 向量状态与临时区 |
| `[0x3B000,0x3E000)` | 12 KiB | local head 1 向量状态与临时区 |

每个 12 KiB 区固定保存 `betaRaw/betaEff/Gref[4]/qRstd/kRstd/Glast` 和
8 KiB 连续 VF scratch。`qRstd/kRstd` 分别位于区内 `0x0C00/0x0D00`，各占
256 B，只保留到 V0 的 MTE3 写回完成；cumsum carry 保持在同一次 VF 的寄存器中，
不另占静态 UB。每个 112 KiB 区的 Stage 语义及 offset 直接定义在
`chunk_kda_fwd_prepare_policy.h`，代码用
`resource.ubBuf.GetBufferByByte<T>(offset)` 绑定 `LocalTensor`。
V6 的 112 KiB 正好由 `qg/kg/V_beta/qgScaled/K_beta_g` 五块 16 KiB BF16
矩阵和一块 32 KiB FP32 `G` 组成，不再保留含义不明的 post scratch。

### Arch22

普通 Vector API 可用 184 KiB：

| 地址 | 大小 | 所有者 |
| --- | ---: | --- |
| `[0x00000,0x12000)` | 72 KiB | pair 0 私有区 |
| `[0x12000,0x1C000)` | 40 KiB | 两个 pair 分时复用的 G/qgScaled + VF scratch |
| `[0x1C000,0x2E000)` | 72 KiB | pair 1 私有区 |
| `[0x2E000,0x30000)` | 8 KiB | CANN 保留，不使用 |

同一 AIV 必须执行 `V0(pair0)->V1(pair0)->V0(pair1)->V1(pair1)`，并用
`V_MTE2` shared-free 事件保证前一个 pair 的共享区消费者完成后，下一个 pair 的
MTE2 才能覆盖共享 G/scratch；V6 还要先等 qgScaled 的 MTE3 读完共享区，不能仅
依赖源码调用顺序。
当前 `q/k/v` 只支持 BF16。V6 按 token 行正序读取 FP32 `G[r]` 后，把同一行
BF16 `qgScaled[r]` 写到共享 G 起始地址的 `256*r` 字节处；该地址始终位于下一条
尚未读取的 G 行之前，因此无需移动 UB 数据。
每个 72 KiB 私有区还固定预留 `qRstd=0x10800`、`kRstd=0x10900`，各占
256 B；两块区域只在 V0 使用，并在对应公开输出的 MTE3 写回完成后释放，不与
`betaRaw/betaEff/dtBias/aLog/Glast` 或 Kminus payload 重叠。

## L1 和 workspace

L1 只供 Cube：

| 地址 | 大小 | 用途 |
| --- | ---: | --- |
| `[0x00000,0x48000)` | 4 x 72 KiB | 四个 head 的当前 payload |
| `[0x48000,0x4C000)` | 16 KiB | 四份 X0 |
| `[0x4C000,0x50000)` | 16 KiB | 四份 X1/negX1 |
| `[0x50000,0x54000)` | 16 KiB | 四份 T |
| `[0x54000,0x5C000)` | 32 KiB | 四份 BF16 Akk 象限包 |
| `[0x5C000,0x80000)` | 144 KiB | 保留 |

每个 workspace slot 固定为 `0x1A400` 字节：

| slot 内 offset | 内容 |
| --- | --- |
| `0x00000` | Qhat，16 KiB |
| `0x04000` | Khat，16 KiB |
| `0x08000` | Arch22 的 betaEff context，512 B；Arch35 不写该区 |
| `0x08200` | 对齐保留，512 B |
| `0x08400` | Stage payload，72 KiB |

`Qhat/Khat/betaEff` context 都只搬运 `validRows`。`G` 始终写入公开 `gk`，
V6 从该公开输出回读，不再在 workspace 保留第二份 G context。
因此每个 slot 从 137 KiB 降为 105 KiB，四个 slot 共减少 128 KiB workspace。

Arch35 和 Arch22 每个 workgroup 都只分配 4 个 slot，对应一个 AIC wave 的四个
group-local head。下一组 head 必须先消费上一组的 C7 free，再原址复用这 4 个 slot，
因此不额外分配不可达的第二组 slot。payload 在 V1、V3、C4/C5、V6 间原址换义；
只有生产者确认旧 reader 完成后才能覆盖。

Arch22 的 C4 还把 payload 的 `[0x4000,0x5000)` 用作一份 4 KiB `T[32,32]`
NZ relay。目标 C220 不支持 FP32 L0C 直接写 FP32 L1，因此 C4 先由 Fixpipe 写入
这段 GM，再通过成对的 `FIX_MTE2` 事件等待写出完成，最后由 MTE2 原样搬入每个
head 的 L1 T 常驻区。每个 head 使用独立的 `MTE2_MTE1` ready 事件交给 C5；
Arch35 支持 FP32 L0C 直写 L1，不经过这段 relay。

V3 会把完整补零的 `Akk[64,64]` 固定写到 payload 内 `0x5800`，C4 始终读取这份
workspace relay。公开 `Akk` 同时只写 `validRows` 行，不能把尾 chunk 的公开输出
当成 64 行 relay。

## 同步

### Arch35

Arch35 核内没有 HardEvent EventID，ping-pong 使用 `AscendC::Mutex`。ID 在各
Stage 的 `Lock/Unlock` 前直接计算，不再通过公共 helper 隐藏：
接口约束见 Ascend C
[`Mutex`](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/latest/API/ascendcopapi/docs/zh/api/SIMD-API/basic_api/sync_control/intra_core_sync/Mutex_ISASI.md)。

| Core | Mutex ID | 资源 |
| --- | --- | --- |
| AIV | `localSlot=0/1` | 两个 UB local slot，即 `0/1` |
| AIC | `localHead=0..3` | 四个 L1 head lane，即 `0/1/2/3` |
| AIC | `4` | 共享 L0A/L0B operand |
| AIC | `5+localHead` | 四个 `W=Akk@K_beta_g` 的 L0C 结果区，即 `5/6/7/8` |
| AIC | `9+localHead` | 四个 `U=Akk@V_beta` 的 L0C 结果区，即 `9/10/11/12` |

`13..27` 不用，`28..31` 为系统保留。相同物理区跨 pipe 使用同一 ID：

```cpp
AscendC::Mutex::Lock<PIPE_MTE2>(id);
AscendC::DataCopy(...);
AscendC::Mutex::Unlock<PIPE_MTE2>(id);

AscendC::Mutex::Lock<PIPE_V>(id);
StageV1Compute(...); // 本 Stage 唯一一次 VF
AscendC::Mutex::Unlock<PIPE_V>(id);
```

Mutex 只处理同核 pipe 交接，不是核间同步。AIC/AIV 仍用 mode `0x4` 的
`CrossCoreSetFlag/CrossCoreWaitFlag` 建立 ready/free 双向协议。下表是调用现场
直接写出的固定 ID；AIV1 的 `+16` 只出现在 AIC 视角，AIV1 本身仍使用本地
`0/1/4/5`：

- `payload ready` 表示 AIV 已把当前阶段的 payload 写入该 head 的 workspace slot，AIC
  可以开始读取；由 AIV `set`，由 AIC `wait`。
- `slot reusable` 表示 AIC 已把当前 payload 搬离 workspace，AIV 可以复用同一 slot 写入
  下一阶段的数据；由 AIC `set`，由 AIV `wait`。
- C2 发布 `slot reusable` 时还同时保证 raw score 已通过 Fixpipe 到达对应 AIV 的
  UB，因此它也是 V3 的输入 ready 信号。
- 这些编号是 AIV/AIC 间的 CrossCore flag，不是核内 Mutex ID，也不是 Arch22
  `AllocEventID` 返回的 HardEvent ID。

| local head | AIV / local slot | AIV payload ready / slot reusable | AIC peer flagId |
| ---: | --- | --- | --- |
| 0 | AIV0 / 0 | `0 / 4` | `0 / 4` |
| 1 | AIV0 / 1 | `1 / 5` | `1 / 5` |
| 2 | AIV1 / 0 | `0 / 4` | `16 / 20` |
| 3 | AIV1 / 1 | `1 / 5` | `17 / 21` |

同一个 slot 的 ready/free 按以下顺序复用，任何一次 set 都有唯一的后续 wait：

| 轮次 | AIV | AIC |
| --- | --- | --- |
| 初始化 | 等待 free `4/5` | 发布 free `4/5/20/21` |
| V0+V1 -> C2 | 发布 ready `0/1` | 等待 ready `0/1/16/17`，C2 后发布 free `4/5/20/21` |
| V3 -> C4/C5 | 等待 free `4/5`，发布 ready `0/1` | 等待 ready `0/1/16/17`，C4 搬完后发布 free `4/5/20/21` |
| V6 -> C7 | 等待 free `4/5`，发布 ready `0/1` | 等待 ready `0/1/16/17`，C7 搬完 RHS 后发布 free `4/5/20/21` |
| 收尾 | 等待最后一次 free `4/5` | 无额外 set |

### Arch22

核内仍通过 `TPipe::AllocEventID` 申请，不能改成不登记占用的裸数字。下表的
“预期 ID”是当前 CANN 9.1 分配器按本文件申请顺序返回的值，实际调用始终使用
`AllocEventID` 的返回值，并在结束时用同一值 `ReleaseEventID`；设备 kernel 不用断言
校验它。不同 HardEvent 是独立事件池，因此多个池都出现 ID 0 不冲突。
申请/释放规则见 Ascend C 的
[`AllocEventID`](https://www.hiascend.com/document/detail/en/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0114.html)
和
[`ReleaseEventID`](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/ascendcopapi/atlasascendc_api_07_0115.html)；
`M_MTE1` 的预占值依据 CANN 9.1.0
[`TPipe::Init`](https://gitcode.com/cann/asc-devkit/blob/v9.1.0/impl/basic_api/kernel_tpipe_impl.h)。

| 核 | HardEvent | 变量 | 预期 ID |
| --- | --- | --- | --- |
| AIV | `V_S` | `scalarRead_` | `0` |
| AIV | `S_V` | `scalarWrite_` | `0` |
| AIV | `V_MTE2` | `sharedFree_` | `0` |
| AIV | `MTE3_MTE2` | `ioFree_[0/1]` | `0/1` |
| AIV | `MTE2_V` | `inputReady_[0/1]` | `0/1` |
| AIV | `V_MTE3` | `outputReady_[0/1]` | `0/1` |
| AIV | `MTE3_V` | `mte3ToV_[0/1]` | `0/1`，V0 写 context 后通知 V1；V6 搬出 qgScaled 后通知 V 释放共享区 |
| AIC | `MTE2_MTE1` | `mte2ToMte1_` | `0` |
| AIC | `MTE2_MTE1` | `tReady_[0..3]` | `1/2/3/4` |
| AIC | `MTE1_M` | `mte1ToM_` | `0` |
| AIC | `M_MTE1` | `mToMte1_` | `3`，`0/1/2` 由 AIC `TPipe::Init` 预占 |
| AIC | `M_FIX` | `mToFix_` | `0` |
| AIC | `FIX_M` | `fixToM_` | `0` |
| AIC | `MTE2_FIX` | `mte2ToFix_` | `0` |
| AIC | `FIX_MTE2` | `fixToMte2_[0..3]` | `0/1/2/3` |
| AIC | `FIX_MTE1` | `fixToMte1_[0..3]` | `0/1/2/3` |

两个 pair 复用共享 G/scratch 时，`V_MTE2` ID 0 是 shared-free：V0/V3/V6 的
MTE2 写前 wait，V1/V3 在最后一次 V 读取后 set。V6 把 BF16 qgScaled 原址写入
FP32 G 的低 16 KiB，并在 MTE3 搬完 qgScaled 后通过 `MTE3_V -> V_MTE2` 传递
shared-free。两个私有 UB slot 的内部 ping-pong 分别使用各 HardEvent 池的 ID 0/1。

核间使用 mode `0x2`，固定 ID 已在 AIV/AIC 两侧主循环直接列成数组：

| pair | 覆盖的 local head | ready ID | free ID |
| ---: | --- | ---: | ---: |
| 0 | 0/1 | `0` | `2` |
| 1 | 2/3 | `1` | `3` |

| 轮次 | AIV | AIC |
| --- | --- | --- |
| 初始化 | 等待 free `2/3` | 发布 free `2/3` |
| V0+V1 -> C2 | 发布 ready `0/1` | 等待 ready `0/1`，C2 后发布 free `2/3` |
| V3 -> C4/C5 | 等待 free `2/3`，发布 ready `0/1` | 等待 ready `0/1`，C4 搬完后发布 free `2/3` |
| V6 -> C7 | 等待 free `2/3`，发布 ready `0/1` | 等待 ready `0/1`，C7 搬完 RHS 后发布 free `2/3` |
| 收尾 | 等待最后一次 free `2/3` | 无额外 set |

每个 pair 的 inactive AIV 只参与 collective，不计算地址、不搬 GM。生产者覆盖 slot
前必须收到反向 free，禁止连续无消费地 set 同一 flag。C7 归还 slot 后，W/U 的
MMAD 与 Fixpipe 不再读取 workspace，可以和 AIV 对下一组 slot 的生产重叠。

## 精度顺序

- `USE_EXP2=true` 时 V0 对每个自然对数域 step 先乘 `1/ln(2)`，再做 FP32
  cumsum，`gk` 保存 log2 值；V1/V6 先在 base-2 域截断，再执行
  `Exp(x * ln(2))`。这个顺序不能改成累计完成后整体除法，否则 FP32 舍入顺序不同。
- `USE_EXP2=false` 时 `gk` 保存自然对数值；V1/V6 使用乘过 `ln(2)` 的等价
  截断边界并直接执行 `Exp(x)`。两条分支输出同一数学量。
- beta sigmoid、`exp(A_log)`、softplus 和 safe-sigmoid 内部的 `Exp` 是激活
  计算，始终使用自然底，与门控累计量选择 `USE_EXP2=true/false` 无关。
- `gk` 的单位随该属性变化；消费它的 `chunk_fwd_h` 必须使用相同的
  `use_exp2`，禁止把自然对数 `gk` 交给 exp2 分支，或反向错配。
- V1 的 BF16 base-2 等价截断范围为 `[-126,120]`；V6 固定为 `[-80,80]`。
- 目标 CANN 9.1 的 SIMD/Reg API 只提供自然底 `Exp`，因此两条分支都调用
  `Exp`，不能把 SIMT `Exp2` 混入单次 VF。
- BF16 写回执行 BF16 RINT。
- L2Norm 的 `q_rstd/k_rstd` 在 FP32 中按
  `1/sqrt(sum(x^2)+epsilon)` 生成并直接写回；不能把暂存的 `sqrt(...)` 当作 rstd。
  Identity 分支写 FP32 `1`，保证固定输出始终初始化。
- `beta_eff` 始终以 FP32 写回；Raw 分支等于 beta 的 FP32 值，Sigmoid 与
  TwoSigmoid 分支分别写 `sigmoid(beta)` 与 `2*sigmoid(beta)`。
- `K_beta_g` 保留两次 BF16 舍入：
  `round(Khat*E(G))`，转回 FP32 乘 beta，再次 round。
- `qg_scaled` 先生成并舍入 `qg`，再转回 FP32 乘 scale 并二次舍入；两者都写入
  独立公开输出。
- C5 不使用不存在的 `Mmad negate` 参数。V3 直接生成 `negX1`，C5 做普通
  `Mmad(negX1,T)`。

## 正式实现前的编译门禁

下列项目必须按目标 CANN 版本头文件和最小设备用例确认：

1. Arch35/Arch22 把 72 KiB packed score 从 L1 装入堆叠 L0A、转置 L0B 的
   `LoadData2DParamsV2` 参数和 stride 单位。
2. C2 每次把完整 `[32,Ns]` L0C 以一条 Fixpipe 写成
   `[rawAqk_s; rawAkk_s]` 时的源布局和 stride。
3. Arch35 `FixpipeParamsArch3510` 的 NZ/L1 与 row-major/UB 配置。
4. Arch22 `FixpipeParamsV220` 的 BF16 quant 模式及 workspace 原址覆盖前
   可用的 MTE2/FIX 事件组合。
5. C7 的四象限 Akk pack 到 L0A 的真实 NZ 排列。
6. 四个单次 VF 的寄存器、mask、repeat/stride 和自然底 `Exp` 参数。
7. 已冻结的 CrossCore flag ID 在目标版本上的 mode 映射、计数深度，以及与
   Catlass/Matmul 内部占用是否冲突。
8. A2、A3、A5 的目标编译、精度、性能和 sanitizer 验证。

设备 kernel 内不使用断言。容量和静态 offset 由 host 小测试、编译资源报告和
sanitizer 分别验证。
