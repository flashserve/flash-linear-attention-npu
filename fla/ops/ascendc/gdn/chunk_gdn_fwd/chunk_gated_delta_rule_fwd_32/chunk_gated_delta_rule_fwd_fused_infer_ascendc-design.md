# chunk_gated_delta_rule_fwd_32 融合推理 Ascend C 设计

本文描述当前已落地的 `ChunkGatedDeltaRuleFwd32`。入口是 `GdnKdaPipe::Kernel`，源码在 `op_kernel/arch35/chunk_gated_delta_rule_fwd_32.h`。

一次 launch 完成原来拆开的 prepare、`fwd_h`、`fwd_o`。GM 上只读写 `q/k/v/g/beta`、可选的 `initial_state`，以及 `o` 和 `final_state`。`g_cumsum`、`A`、`w`、`u`、`v_new`、`H` 只活在 UB、L1、L0。

## 1. 数学

一个 value head 上，状态 \(R \in \mathbb{R}^{K \times V}\)。\(q_t,k_t \in \mathbb{R}^{K}\)，\(v_t,o_t \in \mathbb{R}^{V}\)，\(g_t,\beta_t\) 是标量。\(K = V = 128\)。没有 `initial_state` 时 \(R_0 = 0\)。`scale` 缺省为 \(1/\sqrt{K}\)。

### 1.1 逐 token

\(\varepsilon = 10^{-6}\)，\(\sigma\) 是 sigmoid。\(v_t\) 不做 L2。

\[
\hat q_t = \frac{q_t}{\sqrt{\lVert q_t \rVert_2^2 + \varepsilon}}, \qquad
\hat k_t = \frac{k_t}{\sqrt{\lVert k_t \rVert_2^2 + \varepsilon}}, \qquad
\beta_t = \sigma(\beta_t^{\mathrm{raw}})
\]

\[
\begin{aligned}
R_t^{-} &= R_{t-1}\, e^{g_t} \\
\delta_t &= \beta_t \bigl( v_t - \hat k_t^{\mathsf T} R_t^{-} \bigr) \\
R_t &= R_t^{-} + \hat k_t \, \delta_t^{\mathsf T} \\
o_t &= \mathrm{scale} \cdot \hat q_t^{\mathsf T} R_t
\end{aligned}
\]

### 1.2 一个 32-token chunk

记本 chunk 的 token 为 \(i = 0,\ldots,31\)。\(H\) 是进入本 chunk 之前的 \(R\)。离开时的 \(R\) 是下一个 chunk 的 \(H\)。状态每个 chunk 都更新，下一段读的是更新后的 \(H\)。

\(Q,K,V_{\mathrm{in}}\) 按行堆成 \([32,128]\)。预处理与 1.1 相同。cumsum 保持自然单位，指数是 `exp`：

\[
e_i = \exp\Bigl(\sum_{s=0}^{i} g_s\Bigr), \qquad
G_{ij} = \frac{e_i}{e_j} = \exp(g_i^{\mathrm{csum}} - g_j^{\mathrm{csum}})
\]

\(\mathrm{decay}\) 是 `eg[31]`，也就是本 chunk 32 个 \(g\) 全部加完再 `exp`。token \(i\) 走到 chunk 末尾还要再乘 \(\mathrm{decay}/e_i\)。kernel 用 `DecayOf` 读出这个数。

\[
\mathrm{decay} = \exp\Bigl(\sum_{s=0}^{31} g_s\Bigr)
\]

\[
L_{ij} =
\begin{cases}
\beta_i \, G_{ij} \, (KK^{\mathsf T})_{ij} & i > j \\
0 & i \le j
\end{cases}
\qquad
A = (I + L)^{-1}
\]

\[
\begin{aligned}
U &= A \bigl( V_{\mathrm{in}} \odot \beta \bigr) \\
W &= A \bigl( K \odot (\beta \odot e) \bigr) \\
V^{\mathrm{new}} &= U - WH \\
V^{\mathrm{new}}_{g,i} &= \dfrac{\mathrm{decay}}{e_i}\, V^{\mathrm{new}}_{i} \\
R &= \mathrm{decay}\, H + K^{\mathsf T} V^{\mathrm{new}}_{g}
\end{aligned}
\]

输出用的是本 chunk 入口的 \(H\)，以及没有乘 \(\mathrm{decay}/e_i\) 的 \(V^{\mathrm{new}}\)：

\[
\mathrm{Attn}_{ij} =
\begin{cases}
G_{ij} \, (QK^{\mathsf T})_{ij} & i \ge j \\
0 & i < j
\end{cases}
\]

\[
O = \mathrm{scale} \Bigl( \bigl(Q \odot e\bigr) H + \mathrm{Attn}\, V^{\mathrm{new}} \Bigr)
\]

比值 \(e_i/e_j\) 用乘法完成：`CumsumExp2VF<true>` 把 \(1/e\) 写在 `eg` 后面的 32 个 float 里。差分不另做 `exp`，也不裁剪到 \([-50,50]\)。

## 2. 接口与约束

算子名 `ChunkGatedDeltaRuleFwd32`，只注册 `ascend950`。Python 入口是 `npu_chunk_gated_delta_rule_fwd_32`。ACL 两段式是 `aclnnChunkGatedDeltaRuleFwd32GetWorkspaceSize` 和 `aclnnChunkGatedDeltaRuleFwd32`。

```python
o, final_state = npu_chunk_gated_delta_rule_fwd_32(
    q, k, v, g, beta,
    initial_state=None,   # BF16 或 FP32，可空；空时 final_state 为 FP32
    cu_seqlens=None,      # int64 [N+1]，与 chunk_indices 同时给出
    chunk_indices=None,   # int64，按序列的 (seqId, localChunkId)
    scale=None,           # None 时为 1/sqrt(K)
)
```

| 张量 | shape | dtype | 方向 |
| --- | --- | --- | --- |
| q, k | `[B, Hk, T, K]` | BF16 | 入 |
| v | `[B, Hv, T, V]` | BF16 | 入 |
| g | `[B, Hv, T]` | FP32 | 入，原始门控 |
| beta | `[B, Hv, T]` | BF16 | 入，kernel 内 sigmoid |
| initial_state | `[N, Hv, K, V]` | BF16 或 FP32 | 可选。BF16 进入递推时扩成 FP32 一次，FP32 原样进入 |
| cu_seqlens | `[N+1]` | INT64 | 可选，与 chunk_indices 同时给出 |
| chunk_indices | `[2·Nc]` | INT64 | 可选，规范的 (seqId, localChunkId) |
| o | `[B, Hv, T, V]` | BF16 | 出 |
| final_state | `[N, Hv, K, V]` | 与 initial_state 同 dtype，无 initial_state 时为 FP32 | 出 |

Tiling 拒绝这些情况：`T <= 0`，`K` 或 `V` 不是 128，`Hv` 不是 `Hk` 的正倍数，dtype 或 rank 不符，`o` / `final_state` 的 shape 不符。`T` 可以不是 32、64、128 的倍数，最后一块在 UB 里补零到 32，只把有效行写出。`scale` 是必选属性，缺省 `0.0883883476`（\(1/\sqrt{128}\)）。`blockDim = min(B·Hv, AIC 核数)`，按 V 头分核。GVA 时多个 V 头共用一个 K 头，每个核各自重读 q/k。tiling key 为 0，schedule mode 为 1。

kernel 类型是 `KERNEL_TYPE_MIX_AIC_1_2`，符号 `chunk_gated_delta_rule_fwd32`。`Init` 收到 workspace 后不使用。host 只申请 lib 工作区，`workspaceOffset = 0`。

GM 下标。v/g/beta/o 的 token 基址是 `(b·Hv + hv)·T`，q/k 的 token 基址是 `(b·Hk + hv / (Hv/Hk))·T`。`Hv == Hk` 时两个基址相同。

```text
q/k 的一行         (q_base + t) * 128
v/o 的一行         (v_base + t) * 128
g / beta           v_base + t
state 的一个 AIV   ((b·Hv + hv) * 128 + sub * 64) * 128
```

每个 AIV 只持有状态的 64 行：sub 0 是行 `0..63`，sub 1 是行 `64..127`。一个核上的下一个 `(b, hv)` 在写 final state 之前发出自己的 PrepareLoad，MTE2 和这次写出重叠。

不带 `cu_seqlens` 时 `N = B`，每个 batch 行是一条等长序列，状态槽就是这个 batch。带上 `cu_seqlens` 时物理 `B = 1`，`T` 里按顺序排着 N 条序列，`N = len(cu_seqlens) - 1`。`cu_seqlens[0] = 0`，`cu_seqlens[N] = T`，相邻值可以相等（空序列）。`chunk_indices` 必须是按序列、按块的 `(seqId, localChunkId)`，块长 32。每条序列的 chunk 从它自己的起始 token 算起，起始 token 不必 32 对齐。`o` 仍是整段 `[1, Hv, T, V]`。`final_state` 和 `initial_state` 是 `[N, Hv, 128, 128]`，一条序列一个槽。序列边界上先发出下一条序列的 PrepareLoad，再写本条的 final state，让这段读和这次写重叠。空序列不跑 cube：有 initial state 就把它抄到对应槽，否则写 0。定长路径仍走原来的 `ProcessAiv` / `ProcessAic`。

## 3. 舍入

精确公式在第 1 节。片上实际舍入如下。

| 量 | 驻留 | 舍入 |
| --- | --- | --- |
| beta | GM BF16，sigmoid 在 FP32 | 先 cast 再 `1/(1+exp(-x))` |
| \(e\) | UB FP32 | `exp(cumsum)`，cumsum 不乘 \(1/\ln 2\) |
| \(\hat q,\hat k\) | L1 BF16 NZ | L2 在 FP32，`CAST_RINT` 后写入 NZ |
| \(q \odot e\) | L1 BF16 NZ | FP32 相乘后一次 `CAST_RINT`，再做 `q_exp @ H` |
| \(k \odot \beta \odot e\) | L1 BF16 NZ | 同上，并且多乘 \(-1\)。见第 4 节 |
| \(KK^{\mathsf T}, QK^{\mathsf T}\) | UB FP32 | Cube BF16×BF16、FP32 累加，Fixpipe 不量化 |
| \(L\) | UB FP32 | 已取负，供求解的 `vmula` 做减法 |
| \(A\) | UB FP32，再 L1 BF16 | `Solve32VF` 之后 `CAST_RINT` |
| Attn | L1 BF16 | FP32 相乘后 `CAST_RINT` |
| \(V^{\mathrm{new}}\) | L0C FP32，再 UB/L1 BF16 | `u` 与 \(-WH\) 在 L0C 累加，Fixpipe `F322BF16` |
| \(V^{\mathrm{new}}_g\) | L1 BF16 | 把 BF16 的 \(V^{\mathrm{new}}\) cast 回 FP32，乘 \(\mathrm{decay}/e_i\)，一次 `CAST_RINT` |
| \(D = K^{\mathsf T} V^{\mathrm{new}}_g\) | UB BF16 NZ | Fixpipe 直接写成 BF16，再在 saxpy 里 cast 回 FP32 |
| \(R\) | UB FP32 NZ | `decay·R + FP32(D)` 不舍入。BF16 镜像只给 Cube 的 H；BF16 的 `final_state` 在写出时按 `CAST_RINT` 生成 |
| \(o\) | GM BF16 | `q_exp@H + Attn@V^{new}` 留在 FP32 L0C，Fixpipe `QF322BF16` 乘 `scale` 后一次舍入。Vector 不再乘 scale |

递推状态从 head 开始到结束都是 FP32。Cube 仍吃 BF16 的 H，那份镜像每个 chunk 由当前 FP32 状态收一次，不把舍入结果写回递推。`initial_state` 读入后转成 NZ panel。`final_state` 与 `initial_state` 同 dtype，不传 `initial_state` 时为 FP32：FP32 输出由 FP32 panel 转回 ND，BF16 输出由 BF16 镜像转回 ND。

## 4. 计算怎么落在 Cube 和 Vector 上

\(k\beta e\) 在 L2 的尾部乘了 \(-1\) 再 `CAST_RINT`。因此 L1 里的 `kbg` 是 \(-(K \odot \beta \odot e)\)，`A @ kbg` 得到的是 \(-W\)。`w@H` 以 `initC = false` 累加进已经放着 \(U\) 的 L0C，Fixpipe 出来的就是 \(V^{\mathrm{new}} = U - WH\)。Vector 不再做 `u - P`。代码里的 `ubU_` 装的是这份已经相减过的 \(V^{\mathrm{new}}\)。

\(L\) 在 `GateAttnVF<1>` 里同样取负。`Solve32VF` 的更新是 `y += x · L`，负号把它变成三角减法，解的仍是 \((I+L)X = I\)。

一个 chunk 的 Cube 顺序：

```text
K@K、Q@K          FP32 Fixpipe 回 owner AIV
A @ (v⊙β)         L0C bank 0，initC
A @ kbg           L0C bank 1，得到 -W，Fixpipe BF16 到 L1
(-W) @ H          按 N 拆成两段 32×64，累加进 bank 0 上的 U
                  FixpipeHalfBf16 → 两个 AIV 各 16 行 Vnew
q_exp @ H         整块 32×128×128，initC，结果留在 l0C0
K^T @ v_new_g     两段 64×128，BF16 NZ Fixpipe 到对应 AIV
Attn @ Vnew       initC = false，累加进 q_exp@H
o                 FixpipeHalfBf16，带 scale
```

Vector 侧与之对应的是：L2、sigmoid、cumsum、`vb = v⊙β`、门控、`Solve32`、\(V^{\mathrm{new}}_g\) 的 FP32 门控、状态 saxpy、把 o 的 16 行写回 GM。

### 4.1 一个 32-token chunk 的数据流

下面按数据依赖写 chunk \(c\)（token \([32c, 32c+32)\)）从 GM 进到 `o` 和状态。owner 是 \(c\) 为偶数时的 AIV0、奇数时的 AIV1。两步之前的迭代里，owner 已经把本 chunk 的 L2 和 L1 做完；本迭代消费它们。状态本身不在每个 chunk 写 GM，只有最后一个 chunk 的 saxpy 之后才写成 `final_state`。

```text
GM  q[32,128] k[32,128] v[32,128] g[32] beta[32]     上一 chunk 留下的 H[128,128]
        │ MTE2（owner，在迭代 c-2；c=0、1 在前奏）
        ▼
┌ Vector  预处理，只在 owner 上 ─────────────────────────────┐
│ beta = sigmoid(cast(beta))                         FP32   │
│ e    = exp(cumsum(g))，1/e 写在 e 后面 32 个数      FP32   │
│ decay = e[31] = exp(g[0]+...+g[31])                       │
│ k_hat = k / sqrt(||k||^2+eps)                     BF16 NZ │
│ kbg   = -(k_hat * beta * e)                        BF16 NZ │
│ q_hat = q / sqrt(||q||^2+eps)                      BF16 NZ │
│ q_exp = q_hat * e                                  BF16 NZ │
│ vb    = v * beta                                   BF16 NZ │
└────────────────────────────────────────────────────────────┘
        │ MTE3 → L1 槽 (c & 1)
        ▼
┌ Cube  K@K、Q@K ────────────────────────────────────────────┐
│ kkt = k_hat @ k_hat^T                              FP32   │
│ qk  = q_hat @ k_hat^T                              FP32   │
│ c=0、1 在前奏；c>=2 插在 cube 迭代 c-1 的 q_exp 装载之后   │
└────────────────────────────────────────────────────────────┘
        │ FIX → owner 的 UB
        ▼
┌ Vector  L、求逆、Attn，只在 owner 上 ──────────────────────┐
│ G_ij = e_i * (1/e_j)                                       │
│ L    = -(beta_i * G_ij * kkt) ，i>j，否则 0        FP32   │
│ A    = Solve32(I+L)，8 行一组，再 CAST_RINT        BF16   │
│ Attn = G_ij * qk ，i>=j，否则 0，CAST_RINT         BF16   │
└────────────────────────────────────────────────────────────┘
        │ A、Attn → L1
        ▼
┌ Cube  Vnew = U - WH ───────────────────────────────────────┐
│ U   = A @ vb                     留在 l0C0                 │
│ -W  = A @ kbg                    Fixpipe BF16 到 L1       │
│ Vnew = U + (-W)@H                H 按列拆成 0:64 和 64:128 │
│      Fixpipe F322BF16，两个 AIV 各 16 行                  │
└────────────────────────────────────────────────────────────┘
        │ 两个 AIV 都做自己的 16 行
        ▼
┌ Vector  门控 ──────────────────────────────────────────────┐
│ 行内已是 Vnew（BF16）                                      │
│ v_new_g = CAST_RINT( FP32(Vnew) * decay / e_i )           │
│ 拼成 L1 上的 [32,128] v_new 和 v_new_g                    │
└────────────────────────────────────────────────────────────┘
        │
        ├──────────────────────────────────────────┐
        ▼                                          ▼
┌ Cube  o ────────────────────────────────┐  ┌ Cube  D ─────────────────────────┐
│ acc = q_exp @ H                         │  │ D = K^T @ v_new_g                │
│ acc += Attn @ v_new                     │  │ 两段 64×128，BF16 NZ             │
│ o = Fixpipe(acc * scale)  QF322BF16     │  │ AIV0 拿 K 行 0:64                │
│ 两个 AIV 各 16 行                       │  │ AIV1 拿 K 行 64:128              │
└─────────────────────────────────────────┘  └──────────────────────────────────┘
        │ MTE3                                         │ 两个 AIV
        ▼                                              ▼
GM  o[32,128]                          ┌ Vector  状态 ─────────────────────────┐
token [32c, 32c+16) 由 AIV0 写         │ state = decay * state + D，FP32 不舍入 │
token 行 16:32 由 AIV1 写              │ 各持 64×128，8 个 [64,16] panel       │
                                       │ c<NT-1：panel 0:4 置 H0，4:8 置 H     │
                                       │ 这份 H 是 chunk c+1 的入口             │
                                       └────────────────────────────────────────┘
                                                      │ 仅 c = NT-1
                                                      ▼
                                       panel → ND，各写 final_state 的 64 行
```

`o` 用的 \(H\) 是进入本 chunk 之前的状态。saxpy 用的 \(\mathrm{decay}\) 和 \(D\) 属于本 chunk，写回去的状态给下一个 chunk 做 \(H\)。`final_state` 是 chunk `NT-1` 这次 saxpy 的结果，不是每个 chunk 的快照。

## 5. 分核

MIX 是 1 个 AIC 带 2 个 AIV。AIV 的 `GetBlockIdx()` 先除以 `GetSubBlockNum()`，折回同一个 AIC。head 按 AIC 均分：`head = aicIdx, aicIdx+coreNum, ...`，`coreNum` 就是 `blockDim`。

一个 head 内，chunk `c` 的 owner 是 `c` 的奇偶：

- AIV0 拥有偶数 chunk：这份 chunk 的 L2、sigmoid、cumsum、\(L\)、求逆、Attn。
- AIV1 拥有奇数 chunk，做同样的事。
- 两个 AIV 都参加每一个 chunk。\(V^{\mathrm{new}}\) 和 `o` 按行拆成 16/16（sub 0 拿行 `0..15`）。状态按 K 拆成 64/64。每个 AIV 把自己的 64 行 \(D\) 加进自己的状态。

L1 上 `q/k/vb/kbg/q_exp/A/attn/W` 有两套槽，下标是 `chunk & 1`。`H`、`v_new`、`v_new_g` 各一份，所有 chunk 共用。

## 6. Vector 循环

`GenAuxMatrixVF` 每个核只做一次：5 行 Sklansky 下标，以及 32×32 单位阵。然后按 head 循环。

前奏，设本 AIV 的 sub 为 `s`，且 `s < NT`：

1. `PrepareLoad(s)`：把该 chunk 的 `q/k/v/g/beta` 从 GM 拉进 UB。
2. `InitState`：有初值则拷自己的 64×128 并 `NdToPanels`，否则把 panel 状态清零。置 `V_MTE3(1)`，供第一次 `UploadH` 等待。
3. `PublishStaged(s)`：K 的 L2（含取负的 `kbg`）并拷进 L1，置 `kFlagQk`；再做 Q 的 L2、`q_exp`、`vb`，一并拷进 L1，置 `kFlagQ`。
4. chunk 0 的 owner 等到 `kFlagKkt` 后做 `InverseL1`：先在 UB 里求 L 和逆，再把 BF16 的 `A` 送进 L1 并置 `kFlagInv`。主循环再等 `kFlagAttn`，`AttnL1` 做完 Attn 后拷进 L1。
5. 两个 AIV 都 `UploadH`。初值已经完整，8 个 panel 都拷完后同时置 `kFlagH0` 和 `kFlagH`。
6. chunk 1 的 owner 在这次上传之后求逆，和 chunk 0 的 `A@v` 重叠。

稳态，对 `c = 0 .. NT-1`。owner 是拥有 chunk `c` 的那个 AIV，另一侧是 non-owner。`nxt = c+2`。

1. non-owner `LoadGate(c)`：只拉 `g`，在 `ubEg2_` 上做 cumsum 和 `exp`。owner 的 \(e\) 已经在 `ubEg_`。
2. owner 若还有 `nxt`，先把当前 `eg`（含倒数）拷到 `ubScratch_`。后面的 L2 会覆盖 `ubEg_`。
3. owner 若还有 `nxt`，发出 `PrepareLoad(nxt)`。这次 GM 读盖住后面的 `v_new`、o 和 saxpy。
4. 等 `kFlagP`。`DecayOf` 取 `decay`（`eg[31]`）。`EmitV` 对本 AIV 的 16 行做 FP32 门控，写成高度 16 的 NZ，再按行拼进高度 32 的 L1，置 `kFlagV`。
5. owner 若还有 `nxt`，做 `PublishK(nxt)`：等 GM 读完，sigmoid、cumsum、K 的 L2。L1 拷贝要等本 chunk 的 \(D\) 把 `l1K` 读完，所以放在 saxpy 之后。
6. non-owner 在 `c > 0` 且 `c+1 < NT` 时，等 `kFlagKkt`，求 chunk `c+1` 的逆。chunk 1 已在循环外求过。求逆和 `v_new` 抢同一条 Vector，所以它排在 `EmitV` 之后，避免拖住 `kFlagV`。
7. 主循环等 `kFlagD`，然后 `ApplyD`。先 saxpy panel `0..3`（K 的列 `0..63`），拷进 L1，置 `kFlagH0`；再 saxpy panel `4..7`，置 `kFlagH`。最后一个 chunk 只 saxpy，不上传。
8. owner 若还有 `nxt`：`PublishKFlag`、`PublishQ`、`PublishQFlag`。Q 的 L2 和四份 L1 拷贝发生在 \(H\) 已经排队之后。
9. 主循环等 `kFlagO`，然后 `StoreO(c)` 把本 AIV 的 16×128 写到 GM，置 `kFlagOd`。

head 收尾：等 saxpy 的 `V_MTE3`，必要时再等 NZ 缓冲上的 `MTE3_V(3)`。FP32 的 `final_state` 把 FP32 panel 转成 ND 写出；BF16 的 `final_state` 用 BF16 镜像做 `PanelsToNd`。

`nzArmed_` 记录上一轮 `PublishQFlag` 是否还在读 `q/k/kbg/q_exp/v` 的 NZ 缓冲。下一次 `WaitLoads` 会先等 `MTE3_V(3)`。

## 7. Cube 循环

进入每个 head 时，`M_MTE1(0/1)` 和 `FIX_M(0/1)` 已经置位。

前奏把 chunk 0 和 chunk 1 的 `K@K` 先做完，再做这两段的 `Q@K`。chunk 0 用 L0 bank 0，chunk 1 用 bank 1，第二段 MMAD 和第一段 Fixpipe 重叠。两段 `K@K` 都结束之后才做 `Q@K`，AIV1 的求逆不用等 AIV0 的 `Q@K`。chunk 2 及以后的 `K@K/Q@K` 插在前一个 chunk 的 `q_exp` 装载之后，见下面第 8 步。两段都放在 bank 1，bank 0 上的 \(H\) 保持不动。

每个 chunk `c`，槽 `slot = c & 1`，owner 是 `c` 的奇偶：

1. 等 owner 的 `kFlagInv`。
2. `A @ vb` 在 bank 0，`initC`，结果留在 `l0C0`，先不发 Fixpipe。装载时把 bank 1 的 `M_MTE1` 借走，`A @ kbg` 的 L1→L0 和这次 MMAD 重叠。`kbg` 已取负。
3. `A @ kbg` 在 bank 1。Fixpipe BF16 到 `l1W_[slot]`，再装回 `l0A_[0]`。这一步不读 \(H\)。
4. 等两个 AIV 的 `kFlagH0`。`LoadL1NzHCols(l1H, l0B[0], colFrac=0)` 取 \(H\) 的列 `0..63`。`W` 已经在 `l0A_[0]`，MMAD `32×64×128`，`initC = false`，写入 `l0C0`。
5. 等两个 AIV 的 `kFlagH`。列 `64..127` 装进 `l0B_[1]`，MMAD 写入 `l0C0 + 2048` 个 float。L0C 按 N 主序，`32×64` 正好是 2048。
6. `FixpipeHalfBf16(ubU_)`，两个 AIV 各得 16 行。置 `kFlagP` 两次，原始 id 和 `id+16`，两个 sub 都能等到。
7. 把整份 \(H\) 重新装进 `l0B_[0]`，`q_exp` 装进 `l0A_[0]`。一次 `MTE1_M(0)` 盖住这两次装载。
8. 若 `c >= 1` 且 `c+1 < NT`，在 bank 1 上做 chunk `c+1` 的 `KktQk`。它和 `q@H` 的等待重叠，且不碰 bank 0 里的 \(H\)。
9. 等两个 AIV 的 `kFlagV`。
10. `q_exp @ H`，`32×128×128`，`initC = true`，结果留在 `l0C0` 的低 16KB。
11. \(D\) 使用另一套 bank。`v_new_g` 装进 `l0B_[1]`，`K[:, 0:64]` 转置装进 `l0A_[1]`。`q@H` 释放 bank 0 之后，`K[:, 64:128]` 装进 `l0A_[0]`，`v_new_g` 留在 `l0B_[1]`。两段 MMAD 都是 `64×128×32`，`initC = true`，L0C 起点是 `l0C0 + 16×1024` 个 float，第二段紧接在 `64×128` 之后。第一段的 BF16 NZ Fixpipe 和第二段 MMAD 重叠。sub 0、sub 1 各收到自己的 64 行，并各置一次 `kFlagD`。
12. `Attn @ v_new`，`initC = false`，累加进 `l0C0` 上的 `q@H`。
13. 等两个 AIV 上一 chunk 的 `kFlagOd`（`c = 0` 除外）。`FixpipeHalfBf16(ubO_, scale)`，量化模式 `QF322BF16`。置 `kFlagO` 给两个 sub。

head 结束时等两个 AIV 的 `kFlagOd`，避免下一 head 覆盖还没写走的 `ubO_`。

`LoadL1ToL0AB`、`LoadL1ToL0A`、`LoadL1ToL0B` 只发 LoadData，`MmadL0ToL0C` 只发 Mmad，四个 Fixpipe 只发 Fixpipe。`M_MTE1`、`MTE1_M`、`M_FIX`、`FIX_M` 写在 `Kkt`、`Qk` 和各段 matmul 里。`A@v` 不置 `M_FIX(0)`，`u` 留在 `l0C0` 上继续累加。

### 7.1 T=384 的流水

`T = 384` 时 `NT = 12`，chunk `c` 的 token 是 `[32c, 32c+32)`，`c = 0..11`。下表的一行是一段重叠窗口，顺序由跨核 flag 决定，不是按微秒切开的测量。同一行里的 AIC、AIV0、AIV1 同时在跑。

偶数 chunk 的 owner 是 AIV0，奇数是 AIV1。一行里的「前」收到 `kFlagP` 为止：owner 发出 chunk `c+2` 的 GM 读，两边做 16 行门控。 「后」从 K 的 L2 和求逆开始，盖住 AIC 的 `q@H`、\(D\) 和 `o`：owner 先做 chunk `c+2` 的 K L2，non-owner 在 `c = 1..10` 求 chunk `c+1` 的逆，然后两边 saxpy，owner 再把 K/Q/`vb`/`kbg`/`q_exp` 拷进 L1，最后写 o。chunk 0、1 的求逆在前奏里做完。

chunk 0 的 `w@H` 用的是循环前上传的初值 \(H\)，`kFlagH0` 和 `kFlagH` 一起置上。从 chunk 1 起，前 64 列的 `w@H` 吃上一 chunk saxpy 前半留下的 \(H\)，后 64 列等后半 saxpy。

| 阶段 | AIC | AIV0 | AIV1 |
| --- | --- | --- | --- |
| 前奏 | `K@K` 0，`K@K` 1，再 `Q@K` 0，`Q@K` 1 | 读入 chunk 0；状态行 0:64；L2 和 L1；求逆 0 | 读入 chunk 1；状态行 64:128；L2 和 L1 |
| 前奏尾 | 可以开始 chunk 0 的 `A@v` | 上传初值 \(H\)（8 个 panel，H0 和 H 都置上） | 上传初值 \(H\)；求逆 1，和 chunk 0 的 `A@v` 重叠 |
| 0 前 | 等逆 0；`A@vb`、`A@kbg`；`-W`；`w@H`；Fixpipe `Vnew`；装 \(H\) 和 `q_exp` | 读入 chunk 2；等 `Vnew`；16 行门控 | 只读 chunk 0 的 `g` 并 `exp`；16 行门控 |
| 0 后 | 等 `v_new`；`q@H`；两段 \(D\)；`Attn@v`；`o` Fixpipe（不看上一份 o） | K 的 L2（chunk 2）；saxpy 行 0:64；上传下一份 \(H\)；chunk 2 的 L1 和 Q L2；写 o 行 0:16 | saxpy 行 64:128；上传 \(H\)；写 o 行 16:32 |
| 1 前 | 等逆 1；`A@v` 到 `Vnew`；装 `q_exp`；`K@K` 2 和 `Q@K` 2 | 读 chunk 1 的 `g`；门控 | 读入 chunk 3；门控 |
| 1 后 | `q@H`、\(D\)、`Attn@v`；等 chunk 0 的 o 写完；`o` Fixpipe | 求逆 2；saxpy 并上传 \(H\)；写 o 行 0:16 | K 的 L2（chunk 3）；saxpy 并上传 \(H\)；chunk 3 的 L1 和 Q L2；写 o 行 16:32 |
| 2 前 | 等逆 2；`A@v` 到 `Vnew`；`K@K` 3、`Q@K` 3 | 读入 chunk 4；门控 | 读 `g`；门控 |
| 2 后 | `q@H`、\(D\)、`o` | K 的 L2（chunk 4）；saxpy、上传 \(H\)、chunk 4 进 L1、写 o 行 0:16 | 求逆 3；saxpy、上传 \(H\)、写 o 行 16:32 |
| 3 前 | 等逆 3；`A@v` 到 `Vnew`；`K@K` 4、`Q@K` 4 | 读 `g`；门控 | 读入 chunk 5；门控 |
| 3 后 | `q@H`、\(D\)、`o` | 求逆 4；saxpy、上传 \(H\)、写 o 行 0:16 | K 的 L2（chunk 5）；saxpy、上传 \(H\)、chunk 5 进 L1、写 o 行 16:32 |
| 4 前 | 等逆 4；`A@v` 到 `Vnew`；`K@K` 5、`Q@K` 5 | 读入 chunk 6；门控 | 读 `g`；门控 |
| 4 后 | `q@H`、\(D\)、`o` | K 的 L2（chunk 6）；saxpy、上传 \(H\)、chunk 6 进 L1、写 o 行 0:16 | 求逆 5；saxpy、上传 \(H\)、写 o 行 16:32 |
| 5 前 | 等逆 5；`A@v` 到 `Vnew`；`K@K` 6、`Q@K` 6 | 读 `g`；门控 | 读入 chunk 7；门控 |
| 5 后 | `q@H`、\(D\)、`o` | 求逆 6；saxpy、上传 \(H\)、写 o 行 0:16 | K 的 L2（chunk 7）；saxpy、上传 \(H\)、chunk 7 进 L1、写 o 行 16:32 |
| 6 前 | 等逆 6；`A@v` 到 `Vnew`；`K@K` 7、`Q@K` 7 | 读入 chunk 8；门控 | 读 `g`；门控 |
| 6 后 | `q@H`、\(D\)、`o` | K 的 L2（chunk 8）；saxpy、上传 \(H\)、chunk 8 进 L1、写 o 行 0:16 | 求逆 7；saxpy、上传 \(H\)、写 o 行 16:32 |
| 7 前 | 等逆 7；`A@v` 到 `Vnew`；`K@K` 8、`Q@K` 8 | 读 `g`；门控 | 读入 chunk 9；门控 |
| 7 后 | `q@H`、\(D\)、`o` | 求逆 8；saxpy、上传 \(H\)、写 o 行 0:16 | K 的 L2（chunk 9）；saxpy、上传 \(H\)、chunk 9 进 L1、写 o 行 16:32 |
| 8 前 | 等逆 8；`A@v` 到 `Vnew`；`K@K` 9、`Q@K` 9 | 读入 chunk 10；门控 | 读 `g`；门控 |
| 8 后 | `q@H`、\(D\)、`o` | K 的 L2（chunk 10）；saxpy、上传 \(H\)、chunk 10 进 L1、写 o 行 0:16 | 求逆 9；saxpy、上传 \(H\)、写 o 行 16:32 |
| 9 前 | 等逆 9；`A@v` 到 `Vnew`；`K@K` 10、`Q@K` 10 | 读 `g`；门控 | 读入 chunk 11；门控 |
| 9 后 | `q@H`、\(D\)、`o` | 求逆 10；saxpy、上传 \(H\)、写 o 行 0:16 | K 的 L2（chunk 11）；saxpy、上传 \(H\)、chunk 11 进 L1、写 o 行 16:32 |
| 10 前 | 等逆 10；`A@v` 到 `Vnew`；`K@K` 11、`Q@K` 11 | 门控。没有 chunk 12，不再读入 | 读 `g`；门控 |
| 10 后 | `q@H`、\(D\)、`o` | saxpy、上传最后一份 \(H\)、写 o 行 0:16 | 求逆 11；saxpy、上传 \(H\)、写 o 行 16:32 |
| 11 前 | 等逆 11；`A@v` 到 `Vnew`。不再做下一段 `K@K` | 读 `g`；门控 | 门控。不再准备下一段 |
| 11 后 | `q@H`、\(D\)、`o` | saxpy，不上传 \(H\)；写 o 行 0:16，token `[352,368)` | saxpy，不上传；写 o 行 16:32，token `[368,384)` |
| 收尾 | 等两个 AIV 把 chunk 11 的 o 写完 | panel 转 ND，写 `final_state` 行 0:64 | panel 转 ND，写 `final_state` 行 64:128 |

`o` 的 12 段在各自阶段的后半写出。`final_state` 只有收尾这一次，内容是 chunk 11 的 saxpy 结果。

同一窗口里固定叠在一起的是这三件事：

- AIC 做 chunk \(c\) 的 `q@H`、\(D\)、`o` 时，owner 在做 chunk \(c+2\) 的 K L2，non-owner 在求 chunk \(c+1\) 的逆。
- 两个 AIV 的前半 saxpy 把 \(H\) 的列 `0:64` 送上 L1，AIC 用它做 chunk \(c+1\) 的前半 `w@H`，后半 saxpy 还在走。
- chunk \(c+1\) 的 `K@K`/`Q@K` 插在 chunk \(c\) 的 `q_exp` 已经装进 L0、`q@H` 还没开始的那段，用的是 L0 bank 1，bank 0 上的 \(H\) 不动。

## 8. 片上内存

UB 248KB，L1 512KB，L0A/L0B 各 64KB，L0C 256KB。下表是字节偏移。

### 8.1 UB

| 偏移 | 名字 | 类型 | 实际占用 | 内容 |
| --- | --- | --- | --- | --- |
| 0 | `ubState_` | BF16 | 16KB | 本 AIV 的 64×128 状态，8 个 `[64,16]` panel |
| 16KB | `ubQ_` | BF16 | 8KB | 原始 q，ND |
| 24KB | `ubK_` | BF16 | 8KB | 原始 k，ND |
| 32KB | `ubV_` | BF16 | 8KB | 原始 v，ND |
| 40KB | `ubQnz_` | BF16 | 8KB | \(\hat q\) NZ |
| 48KB | `ubKnz_` | BF16 | 8KB | \(\hat k\) NZ |
| 56KB | `ubKbg_` | BF16 | 8KB | \(-(k \odot \beta \odot e)\) NZ |
| 64KB | `ubQexp_` | BF16 | 8KB | \(q \odot e\) NZ |
| 72KB | `ubVnz_` | BF16 | 8KB | \(v \odot \beta\) NZ |
| 80KB | `ubG_` | FP32 | 128B | 本 owner chunk 的 g |
| 81KB | `ubBeta_` | BF16 | 64B | 原始 beta |
| 82KB | `ubBe_` | FP32 | 128B | sigmoid 之后的 beta |
| 84KB | `ubEg_` | FP32 | 256B | `e[32]` 和 `1/e[32]` |
| 88KB | `ubKkt_` | FP32 | 4KB | \(KK^{\mathsf T}\) |
| 92KB | `ubQk_` | FP32 | 4KB | \(QK^{\mathsf T}\) |
| 96KB | `ubL_` | FP32 | 4KB | 已取负的 L |
| 100KB | `ubA_` | FP32 | 4KB | \((I+L)^{-1}\) |
| 104KB | `ubAbf_` | BF16 | 2KB | A 的 `CAST_RINT` |
| 106KB | `ubAttn_` | BF16 | 2KB | Attn |
| 108KB | `ubU_` | BF16 | 8KB | Fixpipe 下来的 \(V^{\mathrm{new}}\)，本 AIV 只用其中 16 行 |
| 124KB | `ubVn_` | BF16 | 4KB | 本 AIV 16 行 \(V^{\mathrm{new}}\) 的 NZ |
| 128KB | `ubVg_` | BF16 | 4KB | 本 AIV 16 行 \(V^{\mathrm{new}}_g\) 的 NZ |
| 132KB | `ubO_` | BF16 | 4KB | 本 AIV 16 行 o |
| 144KB | `ubEye_` | FP32 | 4KB | 32×32 单位阵 |
| 148KB | `ubSh_` | U32 | 1.25KB | 5×64 的 Sklansky 下标 |
| 152KB | `ubScratch_` | FP32 | 256B | owner 保存的 `eg` |
| 160KB | `ubG2_` | FP32 | 128B | non-owner 的 g |
| 162KB | `ubEg2_` | FP32 | 256B | non-owner 的 `e` 和 `1/e` |
| 176KB | `ubD_` | BF16 | 16KB | 本 AIV 的 64×128 \(D\)，NZ。head 首尾也当 ND 暂存 |

高水位在 `ubD_` 末尾，192KB。`108KB` 到 `124KB`、`136KB` 到 `144KB` 是空档，不挪作它用。

状态在 chunk 之间保持 panel。只有 `InitState` 和写出 `final_state` 时做 ND 与 8 个 `[64,16]` panel 的互转。panel `fc` 的元素起点是 `fc·1024`。

### 8.2 L1

| 偏移 | 名字 | 占用 | 内容 |
| --- | --- | --- | --- |
| 0 | `l1H_` | 32KB | `[128,128]` BF16 NZ。每个 AIV 写自己的 64 行 |
| 40KB + `s`·56KB | slot `s` | 52KB 有效 | `s = chunk & 1` |
| slot+0 | `l1Q_` | 8KB | \(\hat q\) |
| slot+8KB | `l1K_` | 8KB | \(\hat k\) |
| slot+16KB | `l1Vb_` | 8KB | \(v \odot \beta\) |
| slot+24KB | `l1Kbg_` | 8KB | 取负后的 \(k \odot \beta \odot e\) |
| slot+32KB | `l1Qexp_` | 8KB | \(q \odot e\) |
| slot+40KB | `l1A_` | 2KB | A，ND 按列 panel 拷入 |
| slot+42KB | `l1Attn_` | 2KB | Attn |
| slot+44KB | `l1W_` | 8KB | \(-W\)，BF16 NZ |
| 160KB | `l1Vn_` | 8KB | `[32,128]` \(V^{\mathrm{new}}\)，两个 AIV 各写 16 行 |
| 168KB | `l1Vg_` | 8KB | `[32,128]` \(V^{\mathrm{new}}_g\) |

两套 slot 结束于 152KB。`l1Vn_` 从 160KB 起。L1 用到 176KB。

`H` 的一行 panel 高 128。AIV `sub` 写 panel `fc` 的目的地址是 `fc·128·16 + sub·64·16`，每次拷 64×16。`CopyHalfNz` 把 16 行 NZ 写进高度 32 的 panel，另一半行留给另一个 AIV，`dstGap` 是这 16 行的空洞。

### 8.3 L0

L0A、L0B 各两套 32KB bank。\(H\) 是 32KB，只能放在 bank 0。bank 1 用来放 \(D\) 的 `v_new_g`、`A@kbg`，以及插进来的下一 chunk `K@K/Q@K`。

`l0C0_` 在 0。`l0C1_ = l0C0_ + 32·128` 个 float，即 +16KB，给 `A@kbg`。`q@H` 和随后累加的 `Attn@v_new` 占用 `l0C0_` 的低 16KB。\(D\) 从 `l0C0_ + 16·1024` 个 float 开始，两段 `[64,128]` 共 64KB，结束于 +128KB。

`w@H` 的第二段 N=64 写在 `l0C0_ + 2048` 个 float。

## 9. 同步

跨核 flag 用 `CrossCoreSetFlag<0x4>` / `CrossCoreWaitFlag<0x4>`。AIV0 用旗标原号，AIV1 用原号加 16。一个 id 在一个 chunk、一个 AIV 上只置一次、等一次。

| id | 名字 | 谁置 | 谁等 | 含义 |
| --- | --- | --- | --- | --- |
| 0 | `kFlagQk` | owner，`PublishKFlag` | AIC `Kkt` | \(\hat k\) 已在 L1 |
| 1 | `kFlagKkt` | AIC `Kkt` | owner 求逆 | \(KK^{\mathsf T}\) 已在 UB |
| 2 | `kFlagInv` | owner，`InverseL1` | AIC，`A@v` 之前 | A 已在 L1 |
| 3 | `kFlagH` | 两个 AIV，panel 4..7 落地后 | AIC，`w@H` 的第二段 | \(H\) 列 `64..127` 已在 L1 |
| 4 | `kFlagP` | AIC，`Vnew` Fixpipe 之后 | 两个 AIV | 16 行 \(V^{\mathrm{new}}\) 已在 UB |
| 5 | `kFlagV` | 两个 AIV，`EmitV` | AIC，`q@H` 之前 | `v_new` 和 `v_new_g` 已在 L1 |
| 6 | `kFlagO` | AIC，o 的 Fixpipe | 两个 AIV，`StoreO` 之前 | 16 行 o 已在 UB，scale 已乘完 |
| 7 | `kFlagD` | AIC，每段 \(D\) 一次 | 对应 AIV，`ApplyD` 之前 | 64×128 BF16 \(D\) 已在 UB |
| 8 | `kFlagOd` | AIV，o 写入 GM 之后 | AIC，下一次 o Fixpipe 之前 | `ubO_` 可以复用 |
| 9 | `kFlagQ` | owner，`PublishQFlag` | AIC `Qk` | \(\hat q\) 已在 L1。`vb`、`kbg`、`q_exp` 在这个 flag 之后、同一条 MTE3 上接着拷 |
| 10 | `kFlagAttn` | AIC `Qk` | owner，`AttnL1` 之前 | \(QK^{\mathsf T}\) 已在 UB，可以做 Attn |
| 11 | `kFlagH0` | 两个 AIV，panel 0..3 落地后 | AIC，`w@H` 的第一段 | \(H\) 列 `0..63` 已在 L1 |

`kFlagP` 和 `kFlagO` 由 AIC 对原始 id 和 `id + 16` 各置一次，因为两个 AIV 都要等。`kFlagD`、`kFlagQk`、`kFlagKkt`、`kFlagQ`、`kFlagAttn`、`kFlagInv`、`kFlagOd` 带 owner 或 sub，用 `id + aiv * 16`。`kFlagH0` 和 `kFlagH` 由每个 AIV 置自己的原始 id，AIC 把两个 sub 都等完。

`InitState` 只置 `V_MTE3(1)`。前奏的 `UploadH` 等它。稳态里 saxpy 前半置 `V_MTE3(1)`，后半置 `V_MTE3(2)`；上传时分别等这两个事件再拷 panel。`EmitV` 用 `V_MTE3(4)`。`PublishKFlag` / `PublishQFlag` / `InverseL1` 用 `V_MTE3(0)`。NZ 缓冲的生产者和消费者之间是 `MTE3_V(3)`。

Cube 侧 `FIX_M(0)` 的链：前奏或上一 chunk 的 o 把它留下；`A@v` 的 MMAD 消耗它；两段 `w@H` 不碰它；u 的 Fixpipe 再置上；`q@H` 消耗它；\(D\) 的第一段 Fixpipe 再置上；o 之前的等待消耗它；o 的 Fixpipe 再置上，留给下一 chunk 的 `A@v`。`FIX_M(1)` 给 `A@kbg` 和 \(D\) 的第二段。`M_MTE1(0/1)` 是两套 L0 bank 的装载令牌。`w@H` 两段之间不置 `M_MTE1(0)`，因为 `W` 还留在 `l0A_[0]`。

## 10. VF 与装载

`chunk_gated_delta_rule_fwd_32_vf.h`：

| 函数 | 作用 |
| --- | --- |
| `GenAuxMatrixVF` | 辅助矩阵：Sklansky 下标和单位阵 |
| `BetaSigmoidVF` | BF16 beta → FP32 sigmoid |
| `CumsumExp2VF<true>` | 5 次 gather-add，`exp`，并写出 `1/e`。名字保留，计算是自然底 |
| `L2NormKAndKbgVF` | NZ 的 \(\hat k\)，以及 `kbg = -\hat k \odot \beta \odot e` |
| `L2NormQAndQexpVF` | NZ 的 \(\hat q\)，以及 `q_exp = \hat q \odot e` |
| `ScaleRowsVF` | \(v \odot \beta\)，直接写成 NZ |
| `GateAttnVF<1>` | 只写取负的 L |
| `GateAttnVF<2>` | 只写 BF16 Attn |
| `Solve32VF` | 4 组、每组 8 行活寄存器，解 \((I+L)^{-1}\) |
| `InvertAVF` | `Solve32VF` 后把 A `CAST_RINT` 成 BF16 |

L2 一次处理两行。平方和在 FP32，加 \(\varepsilon\) 后做 `1/sqrt`，再乘回行。`StoreNzEvenOdd` 把 even/odd 两个 FP32 寄存器 `CAST_RINT`、`Or`，按 panel 高 32 做 block-copy。

`chunk_gated_delta_rule_fwd_32.h` 里的 Vector：

| 函数 | 作用 |
| --- | --- |
| `VnewHalfVF` | 16 行。BF16 行直接收成 \(V^{\mathrm{new}}\) 的 NZ；同一行 cast 到 FP32，乘 \(\mathrm{decay}/e_i\)，`CAST_RINT` 成 \(V^{\mathrm{new}}_g\) |
| `SaxpyStateVF` | 一组 512 个元素，四行同时活着。先 FP32 乘 `decay`，再 FP32 加 cast 出来的 \(D\)，`CAST_RINT` 写回。`i0/nIter` 选择 16 组里的一段 |
| `NdToPanels` / `PanelsToNd` | 只在 head 首尾使用 |
| `CopyStatePanels` | 一个 panel 一次 `DataCopy`，`burst = 64` |
| `CopyHalfNz` | 16 行 NZ 写入高度 32 的 L1 |

`chunk_gated_delta_rule_fwd_32_matmul.h`：

| 函数 | 作用 |
| --- | --- |
| `LoadL1ToL0AB` | 一对 L1 NZ 装进 L0A 和 L0B，只发 LoadData。`transposeA/B` 为真时源矩阵按转置后的形状存放 |
| `LoadL1NzHCols` | \(H\) 的 64 列。`srcStride = 8`，因为 L1 里每个列 panel 仍是 128 行；`colFrac` 为 0 或 4 |
| `LoadL1ToL0A` / `LoadL1ToL0B` | 只装一边，只发 LoadData。`D` 的两段共用 `v_new_g` |
| `MmadL0ToL0C` | 只发 Mmad。BF16×BF16，FP32 累加。`initC` 为真时清 L0C |
| `FixpipeFp32Ub` | `K@K`、`Q@K` 的 FP32 ND |
| `FixpipeBf16UbNz` | \(D\) 的 BF16 NZ |
| `FixpipeBf16L1` | \(-W\) 写回 L1 |
| `CopyNzToL1` | 已经是 NZ 的 UB 一次拷完 |
| `CopyNdToL1` | A 和 Attn 这种 32 宽矩阵，按 16 列 panel 从 ND 拷进 L1 |

`FixpipeHalfBf16` 在 kernel 头文件里。`[m,n]` 且 `m` 是 32 的倍数时，每次写 `m/2` 行到一个 AIV。`scale == 0` 用 `F322BF16`；`o` 传入非零 scale，用 `QF322BF16`，乘在 Fixpipe 的 FP32 上。

## 11. 源码

```text
op_kernel/chunk_gated_delta_rule_fwd_32.cpp          入口，调用 GdnKdaPipe::Kernel
op_kernel/arch35/chunk_gated_delta_rule_fwd_32.h     调度、UB/L1 地址、跨核 flag
op_kernel/arch35/chunk_gated_delta_rule_fwd_32_vf.h  L2、门控、求逆
op_kernel/arch35/chunk_gated_delta_rule_fwd_32_matmul.h
op_kernel/arch35/chunk_gated_delta_rule_fwd_32_common.h
op_kernel/arch35/chunk_gated_delta_rule_fwd_32_tiling_data.h
op_host/chunk_gated_delta_rule_fwd_32_tiling.cpp
op_host/chunk_gated_delta_rule_fwd32_def.cpp
op_host/op_api/aclnn_chunk_gated_delta_rule_fwd_32.h
```

CPU 标杆在同目录 `cpu_golden_fused_infer.py`。

| 函数 | 和本 kernel 的关系 |
| --- | --- |
| `run_infer_golden` | 逐 token FP64，`exp`。精确算术下与第 1 节相同 |
| `chunk32_reference(..., use_exp2=False)` | 每 32 个 token 更新一次 \(H\)，指数是 `exp`。这是本 kernel 的 chunk 形式 |
| `kda_aligned_reference` | 两个 32-token 段共用同一份入口 \(H\)，第二段的 \(V^{\mathrm{new}}\) 不读第一段更新后的状态。那是另一条递推 |

`test_precision.py` 同时打印逐 token 标杆和 `kda_aligned_reference(use_exp2=True)`。对拍本 kernel 的公式时，以 `run_infer_golden` 和 `chunk32_reference(use_exp2=False)` 为准。`g` 的幅度在 `0.02` 附近时，`exp` 和 `exp2` 的 BF16 结果通常重合，这不改变 kernel 使用自然底 `exp` 这一事实。

`o` 和 BF16 的 `final_state` 比较时把 FP64 标杆 cast 成 BF16 再转回 FP32。FP32 的 `final_state` 直接和 FP64 标杆比。`o` 的舍入点是最后一次 Fixpipe。片上递推不再每个 chunk 把状态舍入回 BF16；BF16 的 `final_state` 只在写出时收一次。

## 12. 未启用：D 与 u−P 直接落 FP32

当前 kernel 仍按第 3 节：\(D\) 用 `FixpipeBf16UbNz`（`F322BF16`），\(u-P\) 用 `FixpipeHalfBf16`。下面这版在 2026-10-27 编过、对过精度、量过 T=16K，然后退回。后续可以再改，现在不要启用。

改动只动这两处 Fixpipe 的量化，Cube 的 matmul 仍是 BF16×BF16。

| 量 | 这版的落点 | Vector 接着做什么 |
| --- | --- | --- |
| \(u-P\) | `FixpipeHalfFp32`，`ROW_MAJOR`，`NoQuant`。一个 AIV 16×128 个 float，8KB | 用这份 FP32 做一次 `CAST_RINT` 得到给 Cube 的 \(V^{\mathrm{new}}\)。门控乘在 FP32 上，再一次 `CAST_RINT` 得到 \(V^{\mathrm{new}}_g\) |
| \(D\) | `FixpipeFp32UbNz`，`NoQuant`，`isChannelSplit = false`，`dstStride = m*16`。一个 AIV 64×128 个 float，32KB。元素顺序与现在的 BF16 NZ 相同 | saxpy 用 `DIST_DINTLV_B32` 直接加，不再把 \(D\) cast 回 FP32 |

UB：`kUbU` 改成 float，占到 124KB。`kUbD` 从 176KB 挪到 160KB，32KB，尽头是 192KB 的 `kUbStateF`。`kUbG2`、`kUbEg2` 挪到 224KB、226KB。L0C 上 \(D\) 仍在 `l0C0_ + 16*1024` 个 float。

精度：`test_initial_fp32.py` 在 NPU 4 上 `failed=0`。T=64、B=1、Hk=Hv=2、无初态、FP32 `final_state` 的 state max/mean ratio 从退回后的 0.7375 / 0.1084 降到 0.6025 / 0.0851。`o` 几乎不动。

性能：NPU 4，B=1，H=28，T=16384，无初态，FP32 `final_state`。对照是现在这版，Task Duration **868.80 µs**（对象 sha256 `90a4d99b…`）。

| 调度 | Task Duration | 说明 |
| --- | --- | --- |
| 整块 FP32 Fixpipe，两个 AIV 的 \(D\) 都到齐再 saxpy | **905.67 µs** | `aic_cube` 仍是 168.80 µs。`aic_fixpipe` 从 348 µs 升到 488 µs |
| 每个 AIV 先送 \(D\) 的 0:64 列（16KB，与现在整块 BF16 一样大）并置 `kFlagD`，64:128 列复用已经消费过的 `kFlagP`，saxpy 中间再等 | **905.68 µs** | 精度与整块相同。墙钟没动 |

每个 chunk、每个 AIV，\(D\) 从 16KB 变成 32KB，\(u-P\) 从 4KB 变成 8KB。512 个 chunk 把这段 Fixpipe 流量加倍。多出来的约 140 µs 活跃时间里，大部分和原来的 MTE1 叠上了，露在关键路径上的是 37 µs。拆半只让 AIV 更早开始 saxpy；Cube 仍要把后一半搬完才能做下一个 matmul，所以墙钟不变。超过相对 872.32 µs 的 +20 µs 线，已退回。源码和已安装的 `ChunkGatedDeltaRuleFwd32_af0e2452…o` 都是 868.80 µs 那份。

同一天下午又把「只改 \(D\)」「只改 \(u-P\)」「两处都改」各跑了一遍 ATK 混合容差（26.9.8，GM 初始化关，102 条）和 T=16K。性能都在 NPU 6、1650 MHz，B=1、H=28、T=16384、无初态、FP32 `final_state`。对照仍是 868.80 µs 那份。`o` 全部通过。失败的都是 `final_state` 的 `matched_ratio` 低于 0.99。

| Fixpipe | 混合容差 | state 失败条的 matched_ratio | Task Duration | aic_fixpipe |
| --- | --- | --- | --- | --- |
| \(D\) 与 \(u-P\) 都落 FP32 | 24/102 | 0.2438–0.9844，均值 0.352 | **916.06 µs** | 493 µs |
| 只 \(D\) 落 FP32 | 24/102 | 0.2236–0.9844，均值 0.349 | **903.44 µs** | 455 µs |
| 只 \(u-P\) 落 FP32 | 23/102 | 0.2235–0.9869，均值 0.333 | **900.86 µs** | 396 µs |

多出来的那 1 条是 `0030`（BNSD、T=1、FP32 初态、seed 0）。只改 \(u-P\) 时它的 matched_ratio 是 0.9869，停在阈值下，通过数回到退回前的 23/102。长序列的 `final_state` 仍在 0.3 附近。三组都超过 +20 µs，源码已退回，已安装对象 sha256 恢复为 `90a4d99b…`。
