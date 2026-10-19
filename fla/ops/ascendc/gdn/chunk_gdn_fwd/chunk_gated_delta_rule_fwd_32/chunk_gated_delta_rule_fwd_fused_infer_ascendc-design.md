# chunk_gated_delta_rule_fwd 推理融合 Ascend C 算子设计

## 1. 目标

本文设计 GDN 推理前向全融合算子。它把现在 A5 上拆开调度的三条 kernel 收进一次 launch：

```text
    -> ChunkGatedDeltaRuleFwdPrepare
    -> ChunkFwdH          
    -> ChunkFwdO
```

融合之后，GM 上只发布两个结果，与
`flash-linear-attention/fla/ops/gated_delta_rule/chunk.py` 前向返回值一致：

```text
return o.to(q.dtype), final_state
```

`g_cumsum`、`A`、`w`、`u`、`v_new`、`h` 都只活在 UB / L1 / L0，不写 GM。

实现逻辑对齐独立 kernel `ops/kda_32_32`：

- SoC 只做 Ascend 950，一个 AIC 带两个 AIV；
- 软件流水深度 2，UB / L1 双缓冲；
- 三角求解在 Vector 上按 16+16 做：先 VCS 解两个 16×16，再 MBH 合成 32×32；
- Cube 只做 BF16×BF16、FP32 累加，`mmad` 清零累加器，`mmad_acc` 在已有累加器上加；

数学按 GDN 的标量门控，不按 KDA 的逐通道 `gk`。片上 chunk 宽度固定 `BT = 32`，与 `kda_32_32` 的求解宽度相同

CPU 标杆是独立文件：

```text
tests/atk/chunk_gated_delta_rule_fwd/cpu_golden_fused_infer.py
```


## 2. 数学语义

一个 value head 上，状态 \(R \in \mathbb{R}^{K \times V}\)。\(q_t, k_t \in \mathbb{R}^{K}\)，\(v_t, o_t \in \mathbb{R}^{V}\)，\(g_t, \beta_t\) 是标量。\(K = V = 128\)。序列结束时的 \(R\) 就是 `final_state`；没有 `initial_state` 时 \(R_0 = 0\)。`scale` 是入参，缺省为 \(1/\sqrt{K}\)。

下面两小节在精确算术下是同一条递推。第 2.2 节把连续 \(M\) 个 token 按行堆成矩阵，第一版 \(M = 32\)。

### 2.1 逐 token

先对当前 token 做归一化和 sigmoid，再更新状态。\(v_t\) 不归一化。\(\varepsilon = 10^{-6}\)，\(\sigma\) 是 sigmoid。

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

\(\hat k_t^{\mathsf T} R_t^{-}\) 和 \(\hat q_t^{\mathsf T} R_t\) 都是 \(1 \times V\)。\(\delta_t \in \mathbb{R}^{V}\)，外积 \(\hat k_t \delta_t^{\mathsf T}\) 是 \(K \times V\)。\(g_t\) 是原始门控。

### 2.2 一个 chunk

记这个 chunk 的 token 为 \(i = 0,\ldots,M-1\)。\(H\) 是进入本 chunk 之前的 \(R\)，离开时的 \(R\) 就是下一个 chunk 的 \(H\)。

把本 chunk 的向量按行堆起来：\(Q, K \in \mathbb{R}^{M \times K}\)，\(V_{\mathrm{in}} \in \mathbb{R}^{M \times V}\)。第 \(i\) 行就是 \(\hat q_i^{\mathsf T}\)、\(\hat k_i^{\mathsf T}\)、\(v_i^{\mathsf T}\)。预处理与 2.1 相同。

\[
g'_i = \Bigl(\sum_{s=0}^{i} g_s\Bigr) \cdot \frac{1}{\ln 2}, \qquad 2^{g'_i} = \exp\Bigl(\sum_{s=0}^{i} g_s\Bigr)
\]

\[
G_{ij} = 2^{g'_i - g'_j}
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

\(\beta\)、\(2^{g'}\) 按行乘到矩阵上：

\[
\begin{aligned}
U &= A \bigl( V_{\mathrm{in}} \odot \beta \bigr) \\
W &= A \bigl( K \odot (\beta \odot 2^{g'}) \bigr) \\
V^{\mathrm{new}} &= U - WH \\
V^{\mathrm{new}}_{g,i} &= 2^{g'_{M-1} - g'_i} \, V^{\mathrm{new}}_{i} \\
R &= 2^{g'_{M-1}} H + K^{\mathsf T} V^{\mathrm{new}}_{g}
\end{aligned}
\]

\(V^{\mathrm{new}}_{i}\) 是 \(V^{\mathrm{new}}\) 的第 \(i\) 行。\(g'_{M-1}\) 记为 \(g'_{\mathrm{last}}\)。

输出用的是进入本 chunk 时的 \(H\)，以及没有乘 \(2^{g'_{\mathrm{last}}-g'_i}\) 的 \(V^{\mathrm{new}}\)：

\[
\mathrm{Attn}_{ij} =
\begin{cases}
G_{ij} \, (QK^{\mathsf T})_{ij} & i \ge j \\
0 & i < j
\end{cases}
\]

\[
O = \mathrm{scale} \Bigl( \bigl(Q \odot 2^{g'}\bigr) H + \mathrm{Attn}\, V^{\mathrm{new}} \Bigr)
\]

\(O \in \mathbb{R}^{M \times V}\)，第 \(i\) 行是 \(o_i^{\mathsf T}\)。

## 3. 范围与 shape

第一版：

```text
SoC                  Ascend 950 only（arch35 / dav-c310）
布局                 kernel 入口为 BNSD，head 主序
q, k                 [B, HK, T, K]     BF16
v                    [B, HV, T, V]     BF16
g                    [B, HV, T]        FP32，原始逐 token 门控，未做 cumsum
beta                 [B, HV, T]        BF16
initial_state        [N, HV, K, V]     BF16，可选
o                    [B, HV, T, V]     BF16
final_state          [N, HV, K, V]     BF16
K                    128
V                    128
BT                   32
G = HV / HK          {1}
use_exp2             true
use_qk_l2norm        固定打开，q/k 每行做 L2Norm
use_beta_sigmoid     固定打开，beta 做 sigmoid
use_gate_in_kernel   false（不读 A_log / dt_bias）
output_final_state   true
```

`N` 在定长时等于 `B`。第一版只接受 `T` 已经是 32 的倍数，host 拒绝其它长度。变长不进 kernel。

目标 case 与 ATK 推理模型对齐，只是本 kernel 的片上 chunk 是 32：

```text
B=1, HK=32, HV=32, T=11264, K=V=128, G=1, BF16
scale = 1/sqrt(128)    # 入参。本 case 取缺省值 1/sqrt(K)
```

公开接口若拿到 `[B, T, HV]` 的 `g/beta` 或 BSND 的 `q/k/v`，由 host 转成上面的 BNSD 再启动。kernel 内部不再转置。

符号：

```text
B          batch
T          本序列 token 数，kernel 看到的值已按 32 对齐
HK         q/k head 数
HV         value head 数
G          HV / HK
K          128
V          128
BT         32，一个数学 chunk，也是一个流水级的 token 数
NT         ceil(T / 32)
M          当前 chunk 的有效 token 数，M <= 32
hk         hv / G
RCP_LN2    1.4426950408889634     # 1/ln(2)
eps        1e-6                   # L2Norm
E(x)       exp2(x)                # use_exp2=true 时的指数
```

`g'` 是 chunk 内 cumsum 再乘 `RCP_LN2`。因此 `E(g') = exp2(g'/ln(2) 的倒数已经乘过) = exp(cumsum(g_raw))`。下文写 `E(g')` 时，指数的底已经和原始 `exp(cumsum)` 对齐。

主要张量：

| 张量 | Shape | 驻留 | 说明 |
| --- | --- | --- | --- |
| `q, k` | `[B,HK,T,K]` | GM 入 | 进 kernel 后每行 L2Norm |
| `v` | `[B,HV,T,V]` | GM 入 | 不做 L2Norm |
| `g` | `[B,HV,T]` | GM 入 | 原始标量门控 |
| `beta` | `[B,HV,T]` | GM 入 | 进 kernel 后做 sigmoid |
| `initial_state` | `[N,HV,K,V]` | GM 入 | BF16，没有则状态从 0 开始 |
| `o` | `[B,HV,T,V]` | GM 出 | 唯一的激活输出 |
| `final_state` | `[N,HV,K,V]` | GM 出 | 序列末尾的 BF16 状态 |
| `g'` | `[32]` | UB | chunk 内 cumsum × RCP_LN2 |
| `L` | `[32,32]` | UB FP32 | 严格下三角 |
| `A` | `[32,32]` | UB→L1 BF16 | `(I+L)^{-1}` |
| `w` | `[32,128]` | L1 BF16 | 本 AIV 这一整块 chunk 的 `A @ kbg`，K 维是 128 |
| `u` | `[32,128]` | UB FP32 | `A @ (v*beta)` |
| `v_new` | `[32,128]` | L1 BF16 | `u - w@H` |
| `H` | `[128,128]` | L1 BF16 | 本 chunk 入口 R 的快照 |
| `R` | `[128,128]` | UB BF16 | 常驻状态。更新时寄存器里 FP32，`CAST_RINT` 写回 |


## 4. 接口设计

# aclnnChunkGatedDeltaRuleFwd

签名沿用现有 `npu_chunk_gated_delta_rule_fwd` / `aclnnChunkGatedDeltaRuleFwd`。第一版不新增算子名。注释里写的是第一版实际接受的值。

```python
from fla_npu.ops.ascendc import npu_chunk_gated_delta_rule_fwd

o, final_state = npu_chunk_gated_delta_rule_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    *,
    initial_state: Tensor | None = None,          # 可空；非空时为 BF16
    output_final_state: bool = True,              # 第一版只支持 True
    chunk_size: int = 32,                         # 第一版只支持 32
    cu_seqlens: Tensor | None = None,             # 第一版必须为 None
    chunk_indices: Tensor | None = None,          # 第一版必须为 None
    scale: float | None = None,                   # 入参；None 时取 1/sqrt(K)
    use_exp2: bool = True,                        # 第一版只支持 True
    use_qk_l2norm_in_kernel: bool = True,         # 第一版只支持 True
    use_gate_in_kernel: bool = False,             # 第一版只支持 False
    use_beta_sigmoid_in_kernel: bool = True,      # 第一版只支持 True
    allow_neg_eigval: bool = False,               # 第一版只支持 False
    state_v_first: bool = False,                  # 第一版只支持 False
    a_log: Tensor | None = None,                  # 第一版必须为 None
    dt_bias: Tensor | None = None,                # 第一版必须为 None
    layout: str = "BNSD",                         # 第一版 kernel 看到的是 BNSD
)
```

第一版只返回 `(o, final_state)`。`g_cumsum`、`A`、`beta_eff`、`h`、`q_hat`、`k_hat`、`q_rstd`、`k_rstd` 不分配、不写 GM。

`scale` 是入参。不传时取 `k.shape[-1] ** -0.5`。目标 case 的 `1/sqrt(128)` 就是这个缺省值。

### 4.1 输入

布局为 BNSD。`BSND` 由 host 转成 BNSD 再启动，kernel 内部不再转置。


| 参数 | 必选 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| q | 是 | `[B, Hk, T, K]` | `bfloat16` | 原始 query |
| k | 是 | `[B, Hk, T, K]` | `bfloat16` | 原始 key |
| v | 是 | `[B, Hv, T, V]` | `bfloat16` | value，不做 L2Norm |
| g | 是 | `[B, Hv, T]` | `float32` | 原始 gate，尚未 cumsum |
| beta | 是 | `[B, Hv, T]` | `bfloat16` | 原始 beta，kernel 内做 sigmoid |
| initial_state | 否 | `[N, Hv, K, V]` | `bfloat16` | 可空。空则状态从 0 开始 |
| output_final_state | 是 | 标量 | `bool` | 第一版只支持 `True` |
| chunk_size | 是 | 标量 | `int` | 第一版只支持 `32` |
| scale | 否 | 标量 | `float` | 入参。`None` 时为 `1/sqrt(K)` |
| use_exp2 | 是 | 标量 | `bool` | 第一版只支持 `True` |
| use_qk_l2norm_in_kernel | 是 | 标量 | `bool` | 第一版只支持 `True` |
| use_gate_in_kernel | 是 | 标量 | `bool` | 第一版只支持 `False` |
| use_beta_sigmoid_in_kernel | 是 | 标量 | `bool` | 第一版只支持 `True` |
| allow_neg_eigval | 是 | 标量 | `bool` | 第一版只支持 `False` |
| state_v_first | 是 | 标量 | `bool` | 第一版只支持 `False`，末两维 `[K, V]` |
| a_log | 否 | — | — | 第一版必须 `None` |
| dt_bias | 否 | — | — | 第一版必须 `None` |
| cu_seqlens | 否 | — | — | 第一版必须 `None`，不做变长 |
| chunk_indices | 否 | — | — | 第一版必须 `None` |
| layout | 是 | 标量 | `str` | kernel 入口为 `BNSD` |


`T` 必须是 32 的倍数。`N` 定长时等于 `B`。`Hv % Hk == 0`，第一版 `Hv / Hk = 1`，`K = V = 128`。

### 4.2 输出

| 返回值 | shape（BNSD） | dtype | 说明 |
| --- | --- | --- | --- |
| o | `[B, Hv, T, V]` | `bfloat16` | 前向输出 |
| final_state | `[N, Hv, K, V]` | `bfloat16` | 序列末尾的状态 |


公开接口里的 `g_cumsum`、`A`、`beta_eff`、`h`、`q_hat`、`k_hat`、`q_rstd`、`k_rstd` 第一版不作为返回值。它们若在计算中出现，只留在 UB / L1。

---

### 4.3 ACLNN 接口

两段式：先 `aclnnChunkGatedDeltaRuleFwdGetWorkspaceSize` 拿 workspace 与 executor，再 `aclnnChunkGatedDeltaRuleFwd` 在指定 stream 上执行。头文件保持 `aclnn_chunk_gated_delta_rule_fwd.h`，不改参数顺序。

```cpp
aclnnStatus aclnnChunkGatedDeltaRuleFwdGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    const aclTensor *aLogOptional,
    const aclTensor *dtBiasOptional,
    const aclTensor *initialStateOptional,
    const aclIntArray *cuSeqlensOptional,
    const aclIntArray *chunkIndicesOptional,
    const char *layout,
    double scale,
    int64_t chunkSize,
    bool useExp2,
    bool useQkL2norm,
    bool allowNegEigval,
    bool stateVFirst,
    const aclTensor *oOut,
    const aclTensor *finalStateOutOptional,
    const aclTensor *qHatOutOptional,
    const aclTensor *kHatOutOptional,
    const aclTensor *qRstdOutOptional,
    const aclTensor *kRstdOutOptional,
    const aclTensor *betaEffOutOptional,
    const aclTensor *gCumsumOutOptional,
    const aclTensor *aOutOptional,
    const aclTensor *hOutOptional,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnChunkGatedDeltaRuleFwd(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

#### 4.3.1 `aclnnChunkGatedDeltaRuleFwdGetWorkspaceSize`

布局固定 BNSD。第一版不用的 tensor 传 `nullptr`。


| 参数 | 输入/输出 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| q | 输入 | `[B, Hk, T, K]` | BF16 | |
| k | 输入 | `[B, Hk, T, K]` | BF16 | |
| v | 输入 | `[B, Hv, T, V]` | BF16 | |
| g | 输入 | `[B, Hv, T]` | FLOAT | 原始 gate |
| beta | 输入 | `[B, Hv, T]` | BF16 | 原始 beta，kernel 内 sigmoid |
| aLogOptional | 输入 | — | — | 第一版 `nullptr` |
| dtBiasOptional | 输入 | — | — | 第一版 `nullptr` |
| initialStateOptional | 输入 | `[N, Hv, K, V]` | BF16 | 可 `nullptr` |
| cuSeqlensOptional | 输入 | — | — | 第一版 `nullptr` |
| chunkIndicesOptional | 输入 | — | — | 第一版 `nullptr` |
| layout | 输入 | 标量 | `char *` | `"BNSD"` |
| scale | 输入 | 标量 | `double` | 入参，缺省 `1/sqrt(K)` |
| chunkSize | 输入 | 标量 | int64 | 第一版 `32` |
| useExp2 | 输入 | 标量 | bool | 第一版 `true` |
| useQkL2norm | 输入 | 标量 | bool | 第一版 `true` |
| allowNegEigval | 输入 | 标量 | bool | 第一版 `false` |
| stateVFirst | 输入 | 标量 | bool | 第一版 `false` |
| oOut | 输出 | `[B, Hv, T, V]` | BF16 | 必选 |
| finalStateOutOptional | 输出 | `[N, Hv, K, V]` | BF16 | 第一版非空 |
| qHatOutOptional | 输出 | — | — | 第一版 `nullptr`，不写 GM |
| kHatOutOptional | 输出 | — | — | 第一版 `nullptr` |
| qRstdOutOptional | 输出 | — | — | 第一版 `nullptr` |
| kRstdOutOptional | 输出 | — | — | 第一版 `nullptr` |
| betaEffOutOptional | 输出 | — | — | 第一版 `nullptr`。sigmoid 仍在 kernel 内做 |
| gCumsumOutOptional | 输出 | — | — | 第一版 `nullptr` |
| aOutOptional | 输出 | — | — | 第一版 `nullptr` |
| hOutOptional | 输出 | — | — | 第一版 `nullptr` |
| workspaceSize | 输出 | 标量 | `uint64_t *` | Device workspace 字节数 |
| executor | 输出 | — | `aclOpExecutor **` | 执行器 |


`Hv / Hk = 1`，`K = V = 128`，`T % 32 == 0`。`use_beta_sigmoid` 第一版固定打开，不靠 `betaEffOutOptional` 非空来推断。

#### 4.3.2 `aclnnChunkGatedDeltaRuleFwd`


| 参数 | 输入/输出 | shape | dtype | 说明 |
| --- | --- | --- | --- | --- |
| workspace | 输入 | `[workspaceSize]` | 设备字节缓冲 | 第一段返回的大小；为 0 时可空 |
| workspaceSize | 输入 | 标量 | uint64_t | |
| executor | 输入 | — | `aclOpExecutor *` | 第一段返回，执行前保持有效 |
| stream | 输入 | — | `aclrtStream` | 执行流 |


## 5. 计算流程与 stage 设计

这一章按一个 chunk 的数据流写。图上 Stage 1 到 Stage 7 各用一个矩形框住本段操作，后面各节展开公式、任务、内存和同步。

单个 AIV 的这一份就是一个完整 chunk。

```text
GM  q[32,128]  k[32,128]  v[32,128]  g[32]  β[32]
    │ MTE2
    ▼
┌ Stage 1  Vector ───────────────────────────────────┐
│ q_hat = q * rsqrt(sum(q^2)+eps)          [32,128]  │
│ k_hat = k * rsqrt(sum(k^2)+eps)          [32,128]  │
│ beta  = sigmoid(β)                       [32]      │
│ g'    = Sklansky(g) * RCP_LN2            [32]      │
│ e_g   = exp2(g')                         [32]      │
│ vb    = v * beta                         [32,128]  │
│ kbg   = k_hat * beta * e_g               [32,128]  │
│ q_exp = q_hat * e_g                      [32,128]  │
└────────────────────────────────────────────────────┘
    │ MTE3
    ▼
L1  q_hat  k_hat  vb  kbg  q_exp
    │ MTE1
    ▼
┌ Stage 2  Cube ─────────────────────────────────────┐
│ 每个 chunk 两次 BlockMmad，各 1 次                 │
│ kkt = k_hat @ k_hat^T                              │
│ qk  = q_hat @ k_hat^T                              │
│ FIX 送回该 chunk 自己的 AIV                        │
└────────────────────────────────────────────────────┘
    │ FIX
    ▼
┌ Stage 3  Vector ───────────────────────────────────┐
│ g_ij = exp2(g'[i] - g'[j])                         │
│ L    = beta * g_ij * kkt     i>j 时，否则 0        │
│ attn = g_ij * qk             i>=j 时，否则 0       │
│ X00  = (I+L[0:16,0:16])^{-1}                       │
│ X11  = (I+L[16:32,16:32])^{-1}                     │
│ A10  = -X11 @ L[16:32,0:16] @ X00                  │
│ 每个 AIV 只解自己那一个 chunk 的 [32,32]           │
└────────────────────────────────────────────────────┘

R 仍在 UB。首 chunk 是 initial_state 或 0，其后是上一 chunk 写回的 R。
Stage 2、Stage 3 都不搬 R。
    │
    ▼
┌ 搬 H  Vector ──────────────────────────────────────┐
│ VecStoreH：CAST_RINT，然后 MTE3                    │
│ R 进 L1，这份叫 H                                  │
└────────────────────────────────────────────────────┘
    ▼
┌ Stage 4  Cube ─────────────────────────────────────┐
│ A 来自 Stage 3，kbg、vb 来自 Stage 1，H 刚搬上来   │
│ u = A @ vb                                         │
│ w = A @ kbg                                        │
│ P = w @ H                                          │
│ u、P FIX 回 UB；w 收成 BF16 后给 P                 │
└────────────────────────────────────────────────────┘
    ▼
┌ Stage 5  Vector ───────────────────────────────────┐
│ v_new   = u - P                      [32,128] FP32 │
│ v_new_g = exp2(g'_last - g') * v_new [32,128] BF16 │
│ MTE3  v_new_g、v_new、attn → L1                    │
└────────────────────────────────────────────────────┘
    ▼
┌ Stage 6  Cube ─────────────────────────────────────┐
│ attn 来自 Stage 3，Stage 5 才 MTE3 进 L1           │
│ D       = k_hat^T @ v_new_g                        │
│ o_inter = q_exp @ H                                │
│ o_acc   = o_inter + attn @ v_new                   │
│ FIX  D、o_acc → 该 AIV 的 UB                       │
└────────────────────────────────────────────────────┘
    ▼
┌ Stage 7  Vector ───────────────────────────────────┐
│ R = exp2(g'_last) * R + D                          │
│ 写回 UB 的 R，不改 L1 的 H                         │
│ o = o_acc * scale，CAST_RINT                       │
│ MTE3 → GM o                                        │
└────────────────────────────────────────────────────┘
```

跨 chunk 留下来的只有 BF16 状态。`final_state` 在最后一个 chunk 更新完之后，由每个 AIV 把自己的那一份 BF16 写回 GM。

### 5.1 Stage 1：Vector，L2Norm / sigmoid / cumsum / vb / kbg

Stage 1 是一个 chunk 从 GM 进 UB 之后、第一批 Cube 之前的全部 Vector 工作。两个 VF 都在 AIV 上，中间不经过 Cube，因此这一段内部没有跨核 flag。跨核 flag 只出现在 `CopyUbToL1` 之后的 `CrossCoreSetFlag<0x4, PIPE_MTE3>(l1Ready + offset)`。

一个 AIV、一个 32-token chunk。`q/k/v` 的 K 或 V 都是 128。`g` 和 `beta` 是每个 token 一个标量。

#### Stage 1 计算公式

`eps = 1e-6`，`RCP_LN2 = 1/ln(2)`。`i` 是 chunk 内的 token，`0 <= i < 32`。

```text
q_hat[i,:]   = q[i,:] * rsqrt(sum_d q[i,d]^2 + eps)     [32,128]
k_hat[i,:]   = k[i,:] * rsqrt(sum_d k[i,d]^2 + eps)     [32,128]
beta_eff[i]  = sigmoid(beta[i])                          [32]
g'[i]        = Sklansky(g)[i] * RCP_LN2                  [32]
e_g[i]       = exp2(g'[i])                              [32]
vb[i,:]      = v[i,:] * beta_eff[i]                     [32,128]
kbg[i,:]     = k_hat[i,:] * beta_eff[i] * e_g[i]        [32,128]
q_exp[i,:]   = q_hat[i,:] * e_g[i]                      [32,128]
```

`g'` 的前缀和用 prepare 里的 `CumsumScaleVF`（`chunk_gated_delta_rule_fwd_prepare_vf.h`）。它是原地 Sklansky inclusive scan，加法树深度 \(\log_2 32 = 5\)，然后整段乘 `RCP_LN2`。32 个 FP32 在一个向量里：步长 \(s=1,2,4,8\) 用 `DeInterleave` 在寄存器里加；\(s=16\) 把下标 15 的值广播到下标 16..31。逐个 `g'[i] = g'[i-1] + g[i]` 会在最后一个 token 上叠 31 次舍入，这里不用。

`beta_eff[i]` 和 `e_g[i]` 按行广播到 128 列。`allow_neg_eigval=true` 时 `beta_eff` 再乘 2；第一版该 flag 为 false。无效行 `i >= M`：`q/k/v` 按 0，`beta` 按 0，`g` 按 0，`g_last` 取最后一个有效 token。

`q_hat/k_hat/vb/kbg/q_exp` 写出 BF16 时用 `CAST_RINT`。`beta_eff`、`g'`、`e_g` 保持 FP32。

#### Stage 1 的两个 VF

| VF | 计算 | 读 | 写 |
| --- | --- | --- | --- |
| `VecCumsumGate` | L2Norm、sigmoid、cumsum、`exp2(g')`、`vb` | `q,k,v,g,beta` | `q_hat,k_hat,beta_eff,g',e_g,vb` |
| `VecPack` | `kbg`、`q_exp` | UB 里的 `k_hat,q_hat,beta_eff,e_g` | `kbg,q_exp` |

`VecCumsumGate` 内部顺序：

```text
1. beta_eff = sigmoid(fp32(beta))
2. 对每一行求 q、k 的 128 维平方和，rsqrt(sum+eps)，乘回得到 q_hat、k_hat
3. g' = Sklansky(g) * RCP_LN2，e_g = exp2(g')
4. vb[i,:] = v[i,:] * beta_eff[i]
5. SetFlag<HardEvent::V_MTE3> / WaitFlag，CopyUbToL1 把 q_hat、k_hat、vb 的 NZ 送进 L1
```

平方和在本 AIV 的 128 列上完成，不拆给另一个 AIV。

`VecPack` 内部顺序：

```text
1. 等待 q_hat、k_hat、beta_eff、e_g 已在 UB
2. kbg[i,:]  = CAST_RINT(k_hat[i,:] * beta_eff[i] * e_g[i])
3. q_exp[i,:] = CAST_RINT(q_hat[i,:] * e_g[i])
4. 按 16×16 NZ 基本块存放
5. SetFlag<HardEvent::V_MTE3> / WaitFlag，CopyUbToL1 把 kbg、q_exp 的 NZ 送进 L1
```

L2Norm 之后的 BF16 写回载入用的那块 ND 缓冲，一行 128 个元素。NZ 是另一块目的 UB。VF 从 ND 读进寄存器，按 16×16 分形写入目的 UB，每个 32 字节块后面留 1 个 32 字节的缝。紧排是 8 KiB，加上缝是 8704 B。`q_hat`、`k_hat`、`vb`、`kbg`、`q_exp` 各有一块这样的目的 UB。`SetFlag<HardEvent::V_MTE3>` / `WaitFlag` 之后，`CopyUbToL1` 把目的 UB 拷进 L1，L1 上没有这条缝，每个矩阵 8 KiB。`g'`、`e_g`、`beta_eff` 不到 L1，继续以 ND 留在 UB。

#### Stage 1 内存

容量按 A5 的实际大小算：每个 AIV 自己一份 UB，**248 KiB**；两个 AIV 和 AIC 共用一份 L1，**512 KiB**。下面的字节数都是一个 AIV、一个 32-token chunk。两个 AIV 的 Stage 1 任务相同，所以每边 UB 占用一样，L1 要按两份相加。

元素大小：BF16 为 2 字节，FP32 为 4 字节。一行向量 `32 × 128 × 2 = 8192` 字节，记为 **8 KiB**。标量 `32 × 4 = 128` 字节，`32 × 2 = 64` 字节。

每个 Tensor 一份的大小：

| Tensor | 形状 | dtype | 一份 | pingpong后的 UB | 是否进 L1 |
| --- | --- | --- | --- | --- | --- |
| `q` / `q_hat` ND | `[32,128]` | BF16 | 8 KiB | 16 KiB（ping+pong）。ND 就地覆盖当前 ping | NZ 目的 UB 见下一行 |
| `k` / `k_hat` ND | `[32,128]` | BF16 | 8 KiB | 16 KiB。ND 就地覆盖 | NZ 目的 UB 见下一行 |
| `q_hat`、`k_hat`、`vb`、`kbg`、`q_exp` 的 NZ 目的 UB | `[32,128]` 加 32 字节缝 | BF16 | 8704 B | 5 × 8704 B = 43520 B。每块单独分配 | `CopyUbToL1` 去掉缝，各 8 KiB |
| `v` | `[32,128]` | BF16 | 8 KiB | 16 KiB | 不进 |
| `g` | `[32]` | FP32 | 128 B | 256 B | 不进 |
| `g'` | `[32]` | FP32 | 128 B | 128 B，单份，本 chunk 留着 | 不进 |
| `e_g` | `[32]` | FP32 | 128 B | 128 B，单份，本 chunk 留着 | 不进 |
| `beta` | `[32]` | BF16 | 64 B | 128 B | 不进 |
| `beta_eff` | `[32]` | FP32 | 128 B | 128 B，单份，本 chunk 留着 | 不进 |

`q_hat`、`k_hat` 的 ND 不另开缓冲，写回当前这份 `q`、`k`。五个进 L1 的矩阵各有一块 NZ 目的 UB，一起活到各自的 MTE3 结束。`g'`、`e_g`、`beta_eff` 留到本 chunk 的 `VecTril`，Stage 1 结束不释放。原始 `q/k/v/g/beta` 的另一半 pingpong 给下一 chunk 的 MTE2。

UB 峰值出现在 `VecPack` 刚写完、`kbg/q_exp` 还没搬完，同时下一 chunk 的 MTE2 已经把另一半pingpong填上。这时一个 AIV 上同时活着：

| 同时活着 | 份数 × 大小 | 小计 |
| --- | --- | --- |
| 当前 `q_hat`、`k_hat` 的 ND | 2 × 8 KiB | 16 KiB |
| 五个 NZ 目的 UB | 5 × 8704 B | 43520 B |
| 下一 chunk 的 `q`、`k`、`v` | 3 × 8 KiB | 24 KiB |
| `g'`、`e_g`、`beta_eff`、下一 chunk 的 `g` | 4 × 128 B | 512 B |
| 下一 chunk 的 `beta` | 64 B | 64 B |
| **Stage 1 工作集** | | **85056 B** |

248 KiB = 253952 B。Stage 1 工作集 85056 B，约占一个 UB 的 **33.5%**，剩下约 165.0 KiB。

`R` 在进 Stage 1 之前就已经放在 UB 里，Stage 1 不写它，但峰值占用要算上。`R` 是本 AIV 的 64 行：`64 × 128 × 4 = 32768 B = 32 KiB`。工作集加上 `R` 是 **115.1 KiB**，约占 UB 的 **46.4%**，剩下约 132.9 KiB。`mask` 和 `eye16` 合计约 1.2 KiB，可以忽略。

L1 是两 AIV 共享的。一个 AIV、一个流水槽写出 5 个 8 KiB：`q_hat + k_hat + vb + kbg + q_exp = 40 KiB`。

| L1 上同时活着 | 计算 | 占用 |
| --- | --- | --- |
| 一个 AIV、一个槽 | 5 × 8 KiB | 40 KiB |
| 两个 AIV、一个槽 | 2 × 40 KiB | 80 KiB |
| 两个 AIV、两个流水槽都还没被 Cube 释放 | 2 × 80 KiB | **160 KiB** |

512 KiB 里 Stage 1 的峰值是 160 KiB，约占 **31.3%**，剩下 352 KiB。另一个 AIV 的起点是 `+ 32×128×2 = 8192` 字节，那是另一个 chunk 的完整 `[32,128]`。

#### Stage 1 的搬运和同步

`CopyUbToL1` 按张量启动，不等两个 VF 全部结束：

```text
q_hat、k_hat、vb    VecCumsumGate 写完
                    SetFlag<HardEvent::V_MTE3> / WaitFlag
                    CopyUbToL1
kbg、q_exp          VecPack 写完，同样的 HardEvent 之后再 CopyUbToL1
g'、e_g、beta_eff   留在 UB，Stage 1 不进 L1
```

这一阶段要给 Cube 的 L1 都写完后，发一次 `CrossCoreSetFlag<0x4, PIPE_MTE3>(l1Ready + offset)`。AIC 用 `CrossCoreWaitFlag<0x4, PIPE_MTE1>(l1Ready + offset)` 之后才 `LoadData`。这一段还没有 Fixpipe，不发 `ubReady`。

### 5.2 Stage 2：Cube，kkt 与 qk

Stage 2 是 Stage 1 把 `k_hat`、`q_hat` 写入 L1 之后、`VecTril` 之前的 Cube。每个 chunk 两件逻辑乘，各一次 `BlockMmad`。两个 AIV 的 chunk 分开算，结果 Fixpipe 回各自的 UB。

一个 128-token 迭代有 4 个 chunk。sub0 拿第 0、2 个，sub1 拿第 1、3 个。同一次 Stage 2 调用只算当前流水槽上的那一个 chunk；另一个槽是这个 AIV 的下一个 chunk，稍后再算一次。

#### Stage 2 计算公式

```text
kkt = k_hat[32,128] @ k_hat^T[128,32]     BlockMmad tile 32×128×32    结果 [32,32] FP32
qk  = q_hat[32,128] @ k_hat^T[128,32]     BlockMmad tile 32×128×32    结果 [32,32] FP32
```

左矩阵、右矩阵是 BF16，L0C 累加是 FP32。`BlockMmad` 清零累加器，这里不在已有 L0C 上累加。相对门控不在这一步乘，留到 Stage 3。

#### Stage 2 的 Cube 任务

没有 VF。AIC 上两个任务：

| 任务 | 计算 | 读 | 写 |
| --- | --- | --- | --- |
| `CubeKkt` | `BlockMmad` tile 32×128×32 | L1 里该 chunk 的 `k_hat` | 该 AIV 的 UB，`kkt[32,32]` FP32 |
| `CubeQk` | `BlockMmad` tile 32×128×32 | L1 里该 chunk 的 `q_hat` 和同一份 `k_hat^T` | 该 AIV 的 UB，`qk[32,32]` FP32 |

顺序：

```text
1. CrossCoreWaitFlag<0x4, PIPE_MTE1>(l1Ready + offset)
2. LoadData：k_hat → L0A，k_hat^T → L0B。前后用 HardEvent::MTE1_M、M_MTE1
3. BlockMmad：CubeKkt
4. WaitFlag<HardEvent::M_FIX>，CrossCoreWaitFlag<0x4, PIPE_FIX>(ubFree + offset)
   Fixpipe kkt → 该 AIV 的 UB
   CrossCoreSetFlag<0x4, PIPE_FIX>(ubReady + offset)
   SetFlag<HardEvent::FIX_M>
5. LoadData：q_hat → L0A。k_hat^T 仍在 L0B 则不重载
6. BlockMmad：CubeQk
7. 同样 Fixpipe qk → 同一 AIV 的 UB，再发一次 ubReady
8. 不发 l1Free。k_hat 还要给 Stage 6 的 CubeD，kbg、vb、q_exp 还留在 L1
```

sub0、sub1 各走一遍。先到的 chunk 可以先做，不必等另一个 chunk 再把两份结果加进同一个 L0C。

#### Stage 2 内存

UB 仍是每个 AIV 248 KiB，L1 仍是共享的 512 KiB。L0 是 AIC 私有：L0A、L0B 各 64 KiB，L0C 128 KiB。下面先算一个 chunk。

| Tensor | 位置 | 形状 | dtype | 一份 |
| --- | --- | --- | --- | --- |
| `k_hat` | L1，Stage 1 留下 | `[32,128]` | BF16 | 8 KiB |
| `q_hat` | L1，Stage 1 留下 | `[32,128]` | BF16 | 8 KiB |
| `k_hat` | L0A | `[32,128]` | BF16 | 8 KiB |
| `k_hat^T` | L0B | `[128,32]` | BF16 | 8 KiB |
| `q_hat` | L0A，覆盖上一份 | `[32,128]` | BF16 | 8 KiB |
| `kkt` 累加器 | L0C | `[32,32]` | FP32 | 4 KiB |
| `qk` 累加器 | L0C，`kkt` fixpipe 之后覆盖 | `[32,32]` | FP32 | 4 KiB |
| `kkt` | 该 AIV 的 UB | `[32,32]` | FP32 | 4 KiB |
| `qk` | 该 AIV 的 UB | `[32,32]` | FP32 | 4 KiB |

`32×32×4 = 4096` 字节，记为 4 KiB。`32×128×2 = 8 KiB`。

L0 峰值是一件乘正在算、另一件的左矩阵还没覆盖时：L0A 8 KiB + L0B 8 KiB + L0C 4 KiB = **20 KiB**。64 KiB 的 L0A/L0B 各用掉 12.5%，128 KiB 的 L0C 用掉 3.1%。

一个 AIV 上 Stage 2 新写下的 UB 是 `kkt + qk = 8 KiB`。第二个流水槽再来一个 chunk，就是另一份 8 KiB。两槽同时活着是 **16 KiB**，约占 248 KiB 的 **6.5%**。

Stage 2 结束时 UB 里还必须留着 Stage 3 要读的量：`g'`、`e_g`、`beta_eff` 共 384 B，以及后面还要用的 `vb`、`kbg`、`q_exp` 共 24 KiB。加上两槽的 `kkt/qk` 16 KiB 和常驻 `R` 32 KiB，这一时刻一个 AIV 大约 **72.4 KiB**，约占 UB 的 **29.2%**。`q_hat`、`k_hat` 的 UB 副本在 Stage 1 的 MTE3 完成后就可以复用，不计入这个峰值。

L1 在 Stage 2 不新增。正在被读的是两个 AIV、当前槽的 `q_hat` 和 `k_hat`：`4 × 8 KiB = 32 KiB`。两个槽都还没释放时是 **64 KiB**。`vb`、`kbg`、`q_exp` 如果也还留在 L1，就仍是 Stage 1 的 160 KiB 峰值，Stage 2 不把它抬高。512 KiB 里这 160 KiB 占 **31.3%**。

#### Stage 2 的同步

```text
AIV  CopyUbToL1 完成 k_hat、q_hat
     CrossCoreSetFlag<0x4, PIPE_MTE3>(l1Ready + offset)
AIC  CrossCoreWaitFlag<0x4, PIPE_MTE1>(l1Ready + offset)
     BlockMmad：CubeKkt、CubeQk
     Fixpipe 到这个 AIV，不是把两个 chunk 的结果写进同一块 UB
     CrossCoreSetFlag<0x4, PIPE_FIX>(ubReady + offset)
```

`kbg`、`vb`、`q_exp` 这一步不读，也不释放。

### 5.3 Stage 3：Vector，构造 L 与求逆

Stage 3 接在 Stage 2 的 fixpipe 之后。每个 AIV 只解自己 UB 里的那一个 `[32,32]`。sub0 和 sub1 同时各解一个 chunk，两份 `A` 不相加。一个 128-token 迭代里，每个 AIV 在另一个流水槽上再解自己的下一个 chunk，那也是一次独立的 32×32 求逆。

#### Stage 3 计算公式

`i,j` 是本 chunk 内的 token 下标。

```text
g_ij[i,j] = exp2(g'[i] - g'[j])                         [32,32]
L[i,j]    = beta_eff[i] * g_ij[i,j] * kkt[i,j]         i > j
L[i,j]    = 0                                           i ≤ j
attn[i,j] = g_ij[i,j] * qk[i,j]                         i ≥ j
attn[i,j] = 0                                           i < j

L00 = L[0:16, 0:16]      L11 = L[16:32, 16:32]      L10 = L[16:32, 0:16]
X00 = (I + L00)^{-1}     X11 = (I + L11)^{-1}       都是 16×16，对角为 1，前向替换无除法
A00 = X00                A11 = X11                  A01 = 0
A10 = - X11 @ L10 @ X00
```

`i >= M` 或 `j >= M` 的 `L`、`attn` 保持 0。`A` 的无效行写成单位行。`L` 保持 FP32。`A` 在 MBH 的行累加里用 FP32，整行写完再 `CAST_RINT` 成 BF16。`attn` 写成 BF16，留给后面的 `attn @ v_new`。

#### Stage 3 的三个 VF

三个 VF 都在本 AIV 上，不调用 Cube。

| VF | 计算 | 读 | 写 |
| --- | --- | --- | --- |
| `VecTril` | `g_ij`、严格下三角 `L`、含对角的 `attn` | 本 chunk 的 `kkt`、`qk`、`g'`、`beta_eff` | `L[32,32]` FP32，`attn[32,32]` BF16 |
| `VecTrilInv1` | 两个 16×16 叶子逆 `X00`、`X11` | `L00`、`L11` | `X00`、`X11`，FP32 |
| `VecTrilInv2` | 合成 32×32 的 `A` | `X00`、`X11`、`L10` | `A[32,32]`，先 FP32，写出 BF16 |

`VecTril`：

```text
1. CrossCoreWaitFlag<0x4, PIPE_V>(ubReady + offset)
2. 用 g' 做 g_ij，乘到 kkt、qk 上，按 i>j / i≥j 填 L 和 attn
3. kkt、qk 的 UB 此后可以复用
```

`VecTrilInv1` 对 `L00`、`L11` 各做一次前向替换。右侧是单位阵，对角是 1：

```text
for col in 0..15:
    for row in 0..15:
        acc = (row == col) ? 1 : 0
        for t in 0..row-1:
            acc = acc - Lii[row,t] * X[t,col]
        X[row,col] = acc
```

`VecTrilInv2` 用 FP32 的 `X00`、`X11` 算 `A10`。行与行之间不读刚写出的 BF16。一整行算完才 `CAST_RINT`。然后 `SetFlag<HardEvent::V_MTE3>` / `WaitFlag`，`CopyUbToL1` 把 `A` 搬进 L1。

#### Stage 3 内存

一个 chunk、一个 AIV：

| Tensor | 形状 | dtype | 大小 | 生命周期 |
| --- | --- | --- | --- | --- |
| `kkt` | `[32,32]` | FP32 | 4 KiB | Stage 2 留下，`VecTril` 读完即一次性 |
| `qk` | `[32,32]` | FP32 | 4 KiB | 同上 |
| `g'` | `[32]` | FP32 | 128 B | Stage 1 留下，本 stage 继续驻留 |
| `beta_eff` | `[32]` | FP32 | 128 B | 同上 |
| `L` | `[32,32]` | FP32 | 4 KiB | `VecTril` 写出，`Inv2` 读完即一次性 |
| `X00`、`X11` | 各 `[16,16]` | FP32 | 各 1 KiB | `Inv1` 写出，`Inv2` 读完即一次性 |
| `attn` | `[32,32]` | BF16 | 2 KiB | 留到 `attn @ v_new`，然后一次性 |
| `A` FP32 | `[32,32]` | FP32 | 4 KiB | `Inv2` 的行累加，写出 BF16 后一次性 |
| `A` BF16 | `[32,32]` | BF16 | 2 KiB | UB 副本在 MTE3 进 L1 后一次性；L1 副本留到 `A @ vb`、`A @ kbg` |

`16×16×4 = 1024` 字节。`32×32×2 = 2048` 字节。

一个 chunk 上这些缓冲如果在 `Inv2` 写完、旧输入还没释放时叠在一起：`kkt + qk + L + X00 + X11 + attn + A_fp32 + A_bf16 = 4+4+4+1+1+2+4+2 = 22 KiB`，再加 `g'` 和 `beta_eff` 的 256 B，是 **22.25 KiB**。

第二个流水槽是另一个独立 chunk，缓冲不和这一份共用。两槽同时活着是 **44.5 KiB**，约占一个 UB 的 **17.9%**。加上仍要留着的 `vb/kbg/q_exp` 24 KiB 和 `R` 32 KiB，大约 **100.5 KiB**，约占 **40.5%**。248 KiB 还剩约 147.5 KiB。

L1 新增的是每个 chunk 的 `A`，2 KiB。两个 AIV、两个槽是 `4 × 2 KiB = 8 KiB`。相对 512 KiB 可以忽略。`k_hat` 要留到 Stage 6，这块 L1 不在这里释放。

#### Stage 3 的同步

Stage 3 内部没有 Cube，没有新的 CV 事件。入口和出口是：

```text
入口   CrossCoreWaitFlag<0x4, PIPE_V>(ubReady + offset)     只等自己这个 chunk
       读完 kkt、qk 后 CrossCoreSetFlag<0x4, PIPE_V>(ubFree + offset)
出口   SetFlag<HardEvent::V_MTE3> / WaitFlag
       CopyUbToL1：A → L1 的本 chunk 槽位，偏移按 AIV 和流水槽分开
       CrossCoreSetFlag<0x4, PIPE_MTE3>(l1Ready + offset)
```

sub0 不等 sub1 的 `A`，sub1 也不等 sub0 的 `A`。下一阶段的 `A @ vb`、`A @ kbg` 用的是本 chunk 这份 `A`。

### 5.4 Stage 4：Cube，u、w 与 P

Stage 4 是 Stage 3 把 `A` 写入 L1、并且本 chunk 入口的 `R` 已经收成 `H` 之后的三件 Cube 乘。每个 chunk 单独算，fixpipe 回这个 chunk 所属的 AIV。

#### Stage 4 计算公式

```text
u = A[32,32] @ vb[32,128]          BlockMmad tile 32×32×128     [32,128]
w = A[32,32] @ kbg[32,128]         BlockMmad tile 32×32×128     [32,128]
P = w[32,128] @ H[128,128]         BlockMmad tile 32×128×128    [32,128]
```

`A`、`vb`、`kbg`、`H`、`w` 进 Cube 时是 BF16，L0C 累加是 FP32。`u` 和 `P` 以 FP32 fixpipe 回 UB，因为下一步是 `v_new = u - P`。`w` 只作为 `P` 的左矩阵，收成 BF16 留在 L0A 或 L1，不必再回 UB。

`H` 的来源：

```text
UB 的 R（本 chunk 入口的 FP32 状态）
    │ VecStoreH：CAST_RINT
    ▼
UB 的 H（BF16，[128,128]）
    │ MTE3
    ▼
L1 的 H
    │ MTE1
    ▼
L0B，供 P = w @ H
```

第一个 chunk 的 `R` 来自 `initial_state`，没有初态就是 0。后面的 chunk 用上一 chunk 写回的 `R_next`。同一 value head 上，后一个 chunk 必须等前一个 chunk 的 `R` 更新完，再做自己的 `VecStoreH`。不能拿前一个 chunk 的 `H` 去算后一个 chunk 的 `P`。

#### Stage 4 的任务

`VecStoreH` 在 AIV 上，三件乘在 AIC 上。

| 任务 | 计算 | 读 | 写 |
| --- | --- | --- | --- |
| `VecStoreH` | `H = CAST_RINT(R)` | UB 里本 chunk 入口的 `R` | L1 的 `H[128,128]` BF16 |
| `CubeU` | `BlockMmad` tile 32×32×128 | L1 的 `A`、`vb` | 该 AIV 的 UB，`u[32,128]` FP32 |
| `CubeW` | `BlockMmad` tile 32×32×128 | L1 的 `A`、`kbg` | L0A 或 L1 的 `w[32,128]` BF16 |
| `CubeP` | `BlockMmad` tile 32×128×128 | 刚得到的 `w`，L1 的 `H` | 该 AIV 的 UB，`P[32,128]` FP32 |

顺序：

```text
1. VecStoreH：R → H。SetFlag<HardEvent::V_MTE3> / WaitFlag，CopyUbToL1
   CrossCoreSetFlag<0x4, PIPE_MTE3>(l1Ready + offset)
   A、vb、kbg 的 l1Ready 在 Stage 1 / Stage 3 已经发过，并且已经被消费
2. BlockMmad：CubeU 与 CubeW 都读 A，先后做。谁先做都可以
3. CubeU 的 u Fixpipe 回该 AIV 的 UB。先 CrossCoreWaitFlag<0x4, PIPE_FIX>(ubFree + offset)
   写完 CrossCoreSetFlag<0x4, PIPE_FIX>(ubReady + offset)
4. CubeW 的 w 收成 BF16 后，CubeP 才能开始
5. CubeP，P Fixpipe 回同一 AIV 的 UB，再发一次 ubReady
6. H 还要给 Stage 6 的 o_inter，这一步不发 l1Free
```

`u` 不依赖 `w` 和 `H`，可以和 `CubeW` 错开复用 L0A 上的 `A`。`P` 依赖 `w`，不能和 `CubeW` 并行。

#### Stage 4 内存

一个 chunk：

| Tensor | 位置 | 形状 | dtype | 大小 | 生命周期 |
| --- | --- | --- | --- | --- | --- |
| `A` | L1，Stage 3 留下 | `[32,32]` | BF16 | 2 KiB | `CubeU`、`CubeW` 读完释放 |
| `vb` | L1，Stage 1 留下 | `[32,128]` | BF16 | 8 KiB | `CubeU` 读完释放 |
| `kbg` | L1，Stage 1 留下 | `[32,128]` | BF16 | 8 KiB | `CubeW` 读完释放 |
| `R` | UB，跨 chunk 常驻 | `[128,128]` | FP32 | 64 KiB | 本 stage 只读，不在这里更新 |
| `H` | L1 | `[128,128]` | BF16 | 32 KiB | 本 chunk 的 `P` 和后面的 `q_exp @ H` 用完释放 |
| `A` | L0A | `[32,32]` | BF16 | 2 KiB | `CubeU`、`CubeW` 期间 |
| `vb` 或 `kbg` | L0B | `[32,128]` | BF16 | 8 KiB | 对应的那次 BlockMmad |
| `w` | L0A 或 L1 | `[32,128]` | BF16 | 8 KiB | 只服务 `CubeP`，`P` 算完即一次性 |
| `H` | L0B | `[128,128]` | BF16 | 32 KiB | `CubeP` 期间 |
| `u` 累加器 / `P` 累加器 | L0C | `[32,128]` | FP32 | 16 KiB | fixpipe 后覆盖 |
| `u` | 该 AIV 的 UB | `[32,128]` | FP32 | 16 KiB | 留到 `v_new = u - P` |
| `P` | 该 AIV 的 UB | `[32,128]` | FP32 | 16 KiB | 同上，减完即一次性 |

`128×128×2 = 32768` 字节，即 `H` 的 32 KiB。`32×128×4 = 16384` 字节，即 `u` 或 `P` 的 16 KiB。`R` 按整份 `[128,128]` FP32 计是 64 KiB；Stage 1 里若仍按每个 AIV 常驻 64 行来放，那是 32 KiB，拼到 L1 时必须是同一时刻同一份状态的两半，不能把两个 chunk 的 `R` 拼成一个 `H`。

L0 峰值在 `CubeP`：L0A 的 `w` 8 KiB，L0B 的 `H` 32 KiB，L0C 的 `P` 16 KiB，合计 **56 KiB**。L0B 的 64 KiB 放得下这份 `H`。

一个 AIV、一个 chunk 新占的 UB 是 `u + P = 32 KiB`。第二个流水槽是另一个 chunk，要等这个 chunk 的 `R` 更新完才有自己的 `H`，因此同一 head 上不会同时存在两份有效的 `u/P`。UB 峰值按一份算是 **32 KiB**，约占 248 KiB 的 **12.9%**。加上还留着的 `q_exp` 8 KiB、`attn` 2 KiB 和 `R` 64 KiB，大约 **106 KiB**，约占 **42.7%**。

L1 上一个 chunk 的 `A + vb + kbg + H = 2+8+8+32 = 50 KiB`。同一 head 同时只放一份 `H`。两个 AIV 若正在算同一个 head 的前后两个 chunk，后一个的 `H` 还不能写。512 KiB 里这 50 KiB 占 **9.8%**。

#### Stage 4 的同步

```text
AIV   VecStoreH
      CrossCoreSetFlag<0x4, PIPE_MTE3>(l1Ready + offset)
AIC   CrossCoreWaitFlag<0x4, PIPE_MTE1>(l1Ready + offset)
      BlockMmad CubeU，Fixpipe 后 CrossCoreSetFlag<0x4, PIPE_FIX>(ubReady + offset)
      BlockMmad CubeW
      BlockMmad CubeP，再发一次 ubReady
AIV   CrossCoreWaitFlag<0x4, PIPE_V>(ubReady + offset)
      下一步才做 v_new = u - P
```

`H` 的 `l1Ready` 没到，`CubeP` 不能开始。`CubeU` 不读 `H`，只等 `A` 和 `vb`。

### 5.5 Stage 5：Vector，v_new 与 v_new_g

Stage 5 只做 Vector。输入是 Stage 4 fixpipe 回来的 `u` 和 `P`，输出给后面的 Cube。

```text
v_new[i,:]   = u[i,:] - P[i,:]                         [32,128] FP32
v_new_g[i,:] = exp2(g'_last - g'[i]) * v_new[i,:]      [32,128] BF16
```

`g'_last` 是本 chunk 最后一个有效 token 的 `g'`。`VecVnew` 做完后，MTE3 把三份 Cube 要用的矩阵搬进 L1：`v_new_g`、没乘门控的 `v_new`、Stage 3 留在 UB 的 `attn`。`k_hat`、`q_exp`、`H` 已经在 L1，这一步不重搬。

一个 chunk 新占 UB：`v_new` 16 KiB FP32，`v_new_g` 8 KiB BF16，加上还在的 `u`、`P` 各 16 KiB，同时活着是 **56 KiB**。`v_new_g` 进 L1 后 UB 副本可复用。`attn` 是 2 KiB BF16。

同步：`CrossCoreWaitFlag<0x4, PIPE_V>(ubReady + offset)`。算完 `SetFlag<HardEvent::V_MTE3>` / `WaitFlag`，三份 `CopyUbToL1` 之后 `CrossCoreSetFlag<0x4, PIPE_MTE3>(l1Ready + offset)`，并 `CrossCoreSetFlag<0x4, PIPE_V>(ubFree + offset)`。

### 5.6 Stage 6：Cube，D 与 o_acc

Stage 6 只做 Cube。三件乘之间没有 Vector。`o_inter` 用 `BlockMmad` 清零 L0C，`attn @ v_new` 加进同一块 L0C。

```text
D       = k_hat^T[128,32] @ v_new_g[32,128]     BlockMmad tile 128×32×128     [128,128] FP32
o_inter = q_exp[32,128]  @ H[128,128]           BlockMmad tile 32×128×128     [32,128] FP32
o_acc   = o_inter + attn[32,32] @ v_new[32,128] 累加进同一块 L0C，tile 32×32×128   [32,128] FP32
```

`H` 仍是本 chunk 入口的快照，不是正在更新的 `R`。`k_hat`、`q_exp` 来自 Stage 1，`v_new_g`、`v_new`、`attn` 来自 Stage 5 的 MTE3。

入口 `CrossCoreWaitFlag<0x4, PIPE_MTE1>(l1Ready + offset)`，三件乘用 `BlockMmad`。`D` 和 `o_acc` 以 FP32 Fixpipe 回该 AIV 的 UB：先 `CrossCoreWaitFlag<0x4, PIPE_FIX>(ubFree + offset)`，写完 `CrossCoreSetFlag<0x4, PIPE_FIX>(ubReady + offset)`。整槽不再被读之后，`CrossCoreSetFlag<0x4, PIPE_MTE1>(l1Free + offset)`。`D` 是 64 KiB，`o_acc` 是 16 KiB。L0C 上这两份不同时保留：先做 `D` 并 Fixpipe，再做 `o_inter` 和累加。L0 峰值是 `D` 那次，L0C 64 KiB，落在 128 KiB 以内。

### 5.7 Stage 7：Vector，更新 R 与写出 o

Stage 7 只做 Vector。状态和输出互不依赖，都先 `CrossCoreWaitFlag<0x4, PIPE_V>(ubReady + offset)`。写 GM 之前 `SetFlag<HardEvent::V_MTE3>` / `WaitFlag`，再用 `CopyUbToGm`。

```text
R = exp2(g'_last) * R + D          寄存器里 FP32，CAST_RINT 写回 UB 的 BF16 R
o = CAST_RINT(o_acc * scale)       [32,128] BF16，MTE3 到 GM
```

写回的是 UB 里的 `R`。L1 上的 `H` 保持本 chunk 入口值，下一个 chunk 再从这份新 `R` 做 `VecStoreH`。最后一个 chunk 结束后，把这份 BF16 `R` MTE3 到 `final_state`。

`o` 一份 8 KiB。`R` 原地更新，不另开缓冲。

## 6. 分核

### 6.1 block 与 head

对照 `kda32_aiv.cce` 的 `vec_prepare_g3`。一个 AIV 一次 `gm_ub_single<32, 128, 2>` 搬的是 32 行 × 128 列 × 2 字节，K 维不切开。`subid` 乘的是 `32*128`，那是下一个 32-token chunk 的起点。

```text
launch: blockDim = head 数

每个 block
┌──────────────── AIC (core = head) ───────────────────                 ┐
│  Cube：本 head 上两个 chunk 的矩阵乘，K 维都是 128                    │
│  ┌────────────────────┐  ┌────────────────────                        ┐      │
│  │ AIV sub0           │  │ AIV sub1           │                       │
│  │ 自己的 chunk       │  │ 紧挨着的下一个 chunk│                      │
│  │ q,k,g [32, 128]    │  │ q,k,g [32, 128]    │                       │
│  │ beta [32]          │  │ beta [32]          │                       │
│  └────────────────────┘  └────────────────────┘                       │
└──────────────────────────────────────────────────────                 ┘
```

`kda32_aiv.cce` 第 443–446 行，每个 AIV 自己的 `gqkdb_cnt` 从 0 递增：

```text
offset = 64*128*gqkdb_cnt + 32*128*subid

sub0, cnt=0    token [0:32)      [32, 128]
sub1, cnt=0    token [32:64)     [32, 128]
sub0, cnt=1    token [64:96)     [32, 128]
sub1, cnt=1    token [96:128)    [32, 128]
```

L1 上第二个 subblock 的起点是 `+ 32*128*2` 字节（`kda32_aic.cce` 里 `kd_addr2`）。那是另一份完整的 `[32, 128]`，不是同一份矩阵的 K `[64:128)`。

Cube 上每个 chunk 的 `kkt`、`qk` 各一次 `mmad<32,128,32>`。`kda32_aic.cce` 第 790 行的 `mmad<32,128,16>` 加第 793 行的 `mmad<16,128,16>` 是 KDA 的 `kd/kd2` 拆法，GDN 不使用。

GM 偏移（元素个数）：

| 张量 | 单个 AIV 一次搬入 | 地址 |
| --- | --- | --- |
| `q, k, g` | `[32, 128]` BF16 | head 基址 + `64*128*cnt + 32*128*subid` |
| `beta` | `[32]` BF16 | head 基址 + `64*cnt + 32*subid` |
| `v` | 每次 `[16, 128]` BF16 | head 基址 + `32*128*v_cnt + 16*128*subid` |
| `o` | 每次写出 `[16, 128]` BF16 | head 基址 + `32*128*out_cnt + 16*128*subid` |

### 6.2 token 方向

```text
T tokens
└─ NT = T/32 个 chunk，严格串行更新 R
   └─ 软件流水看连续两个 chunk
      槽 0 = chunk 2i
      槽 1 = chunk 2i+1
      槽 1 的 Cube 读的 H，必须是槽 0 的 vec_state 写回之后的 R
```

两个 AIV 各处理自己那一个完整 chunk，chunk 的 `q/k` 都是 `[32, 128]`。64-token 窗口里 sub0 拿前 32 个 token，sub1 拿后 32 个 token。16 只出现在这个 chunk 内部：求逆按 16+16 分块，`v` 和 `o` 的 GM 搬运按 16 行。K 维在这个 chunk 里始终是 128。

### 6.3 软件流水深度

`iterN = 2`，迭代变量 `c` 走过 chunk：

```text
c = 0                 前奏。只做槽 0 的加载和 Vector 准备，Cube 还没有上一个 chunk
c = 1 .. NT-1         稳态。槽 (c%2) 做本 chunk 的准备，槽 ((c+1)%2) 做上一 chunk 的 Cube 和状态写回
c = NT                收尾。把最后一个 chunk 的 o 和 R 写完
```

状态槽不pingpong。`R` 在整个 `NT` 里占用固定 UB，两个流水槽的矩阵缓冲才分开。

### 6.4 尾块

`T` 已经对齐到 32 时，只有逻辑上的有效长度 `M<32` 会走进尾块规则，见第 10 节。两个 AIV 看到同一个 `M`。

## 7. 片上内存

A5 上两个 AIV 各有一份 UB，和 AIC 共享 L1。下面的预算按每 AIV 可用 192 KiB UB、共享 L1 512 KiB 来排，地址都留在这个上限里面。

UB 地址单位沿用 `kda_32_32` 的 512 字节。L1 / L0 地址单位 1024 字节。

### 7.1 UB 常驻

两级流水共用，不随槽号切换。

| 名字 | 起点 | 大小 | dtype | 谁持有 |
| --- | --- | --- | --- | --- |
| `R` | 0 | 128×128×2 = 32 KiB | BF16 | 常驻状态，更新后 `CAST_RINT` 写回 |
| `mask` | 32 KiB | 32×32 bit = 128 B | bit | sub0，严格下三角 |
| `eye16` | 32.5 KiB | 16×16×4 = 1 KiB | FP32 | sub0，VCS 的单位阵 |

`R` 从 0 到 32 KiB。更新是原地的：读出 BF16，在寄存器里升成 FP32，乘 `E(g'_last)` 再加上 `D`，`CAST_RINT` 写回同一地址。

### 7.2 UB 流水区

UB 计算区只有一套，从 48 KiB 起。`u`、`v_new`、`L`、`A` 不按槽再复制一份，192 KiB 放不下两套 FP32 `[32,128]`。深度 2 的重叠放在两个地方：输入pingpong在第 7.3 节，L1 的两套槽在第 7.4 节。下一 chunk 的 MTE2 可以盖住本 chunk 的求逆；下一 chunk 的 `vec_tril` 必须等本 chunk 释放 `gram/L/A`。

槽 0 流水区，起点 48 KiB：

| 缓冲 | 相对起点 | 大小 | dtype | 生命周期 |
| --- | --- | --- | --- | --- |
| `k` | 0 | 32×128×2 = 8 KiB | BF16 | 本 chunk 的完整 K |
| `q` | 8 KiB | 8 KiB | BF16 | 同上 |
| `g_raw` | 8 KiB | 32×4 = 128 B | FP32 | cumsum 期间 |
| `beta` | 8.5 KiB | 32×2 = 64 B | BF16 | 整个 chunk |
| `g'` | 9 KiB | 32×4 = 128 B | FP32 | 整个 chunk |
| `e_g` | 10 KiB | 32×4 = 128 B | FP32 | `E(g')` |
| `gram` | 12 KiB | 32×32×4 = 4 KiB | FP32 | KKT fixpipe 之后，tril 之前 |
| `L` | 16 KiB | 4 KiB | FP32 | 求逆期间 |
| `A_fp32` | 20 KiB | 4 KiB | FP32 | MBH 行累加 |
| `A_bf16` | 24 KiB | 32×32×2 = 2 KiB | BF16 | 交给 Cube 之前 |
| `v` | 28 KiB | 32×128×2 = 8 KiB | BF16 | sub0 |
| `vb` | 36 KiB | 32×128×2 = 8 KiB | BF16 | sub0 |
| `u` | 44 KiB | 32×128×4 = 16 KiB | FP32 | sub0 |
| `v_new` | 60 KiB | 16 KiB | FP32 | sub0，随后铸 BF16 进 L1 |
| `o_acc` | 76 KiB | 16 KiB | FP32 | sub0，scale 之后铸 BF16 |

`q`、`k` 各 8 KiB，后面的 `g'` 到 `o_acc` 整体后移 8 KiB，流水区终点约 48+100 = 148 KiB。sub1 同样持有自己那个 chunk 的 `q[32,128]`、`k[32,128]`。

### 7.3 UB 静态区：输入pingpong

从 144 KiB 起，两个槽共用这一对缓冲做 MTE2/V 重叠：

```text
144 KiB
  k_ping, k_pong     各 32×128 BF16 = 8 KiB，合计 16 KiB
  q_ping, q_pong     合计 16 KiB
  g_ping, g_pong     各 32 FP32 = 128 B
  beta_ping/pong     各 64 B
  v_ping, v_pong     各 32×128 BF16 = 8 KiB，合计 16 KiB，仅 sub0
```

sub0 这一段合计约 33 KiB，终点约 177 KiB，低于 192 KiB。sub1 不放 `v` 的pingpong。

`q/k/g/beta/v` 的双缓冲用 `PIPE_MTE2` 与 `PIPE_V` 的 flag 0/1：

```text
set_flag(PIPE_MTE2, PIPE_V, slot)
wait_flag(PIPE_MTE2, PIPE_V, slot)
set_flag(PIPE_V, PIPE_MTE2, slot)
wait_flag(PIPE_V, PIPE_MTE2, slot)
```

下一 chunk 的搬运可以盖住当前 chunk 的求逆。

### 7.4 L1

L1 512 KiB，两套槽。槽 0 用前 240 KiB，槽 1 用后 240 KiB，最后 32 KiB 留给常驻的 16×16 单位阵和归约信箱。

一个槽里、一个 chunk 的布局（字节，相对该槽基址）：

| 缓冲 | 偏移 | 大小 | 说明 |
| --- | --- | --- | --- |
| `k[2]` | 0 | 2×32×128×2 = 16 KiB | 两个 AIV 各一份完整 chunk，NZ |
| `q[2]` | 16 KiB | 16 KiB | 同上 |
| `kbg[2]` | 32 KiB | 16 KiB | 各是 `[32, 128]` 的 `k * β * E(g')` |
| `H_rows[2]` | 24 KiB | 2×64×128×2 = 32 KiB | R 的 BF16 快照，按 K 行 |
| `A` | 56 KiB | 32×32×2 = 2 KiB | 求逆结果 |
| `vb` | 58 KiB | 32×128×2 = 8 KiB | |
| `w[2]` | 66 KiB | 2×32×128×2 = 16 KiB | 每个 AIV 自己 chunk 的 `A @ kbg`，`[32, 128]` |
| `v_new` | 74 KiB | 32×128×2 = 8 KiB | |
| `v_new_g` | 82 KiB | 8 KiB | |
| `q_exp[2]` | 90 KiB | 2×32×128×2 = 16 KiB | 每个 AIV 自己 chunk 的 `q_hat * E(g')` |
| `attn[2]` | 106 KiB | 2×32×32×2 = 4 KiB | 含对角的因果块，各一份 `[32,32]` |

一个槽里两份 `[32, 128]` 挨着放：sub1 的基址是 sub0 加上 `32*128*2` 字节。Cube 看到的每一份 `k` 都是 `[32, 128]`。

NZ 分型沿用 Cube BF16 的 16×16。`[32, 128]` 写成 2×8 个 16×16 基本块。写入时按基本块交错，Cube `LoadData` 才能直接进 L0。

### 7.5 L0A / L0B / L0C

| 存储 | 容量预算 | 用途 |
| --- | --- | --- |
| L0A | 64 KiB | 左矩阵 BF16：`k[32,128]`、`A`、`w`、`q_exp`、`k^T` |
| L0B | 64 KiB | 右矩阵 BF16：`k^T`、`H`、`vb`、`v_new` |
| L0C | 128 KiB | FP32 累加器 |

`mmad<M,K,N>` 表示 `C[M,N] = A[M,K] @ B[K,N]`。本算子一个 chunk 上的 Cube 调用：

```text
mmad<32, 128, 32>   kkt = k_hat @ k_hat^T
mmad<32, 128, 32>   qk  = q_hat @ k_hat^T
mmad<32, 128, 128>  k_hat @ H          状态读出，可按 N 再切
mmad<32, 32, 128>   w = A @ kbg        K 维是 32
mmad<32, 128, 128>  P = w @ H
mmad<128, 32, 128>  D = k_hat^T @ v_new_g
```

每个 AIV 的 chunk 单独做上面这些乘，结果 fixpipe 回这个 AIV 的 UB。不把两个 chunk 用 `mmad_acc` 加在一起。KDA 把 `kkt` 拆成 `mmad<32,128,16>` 和 `mmad<16,128,16>`，用的是两套左矩阵 `kd` 和 `kd2`。GDN 没有 `kd2`。

`o_inter = q_exp @ H` 与 `P` 同一分型。`attn @ v_new` 是 `mmad<32,32,32>` 四段。输出那次用 `mmad_acc` 把 `o_inter` 和 `attn@v_new` 加在同一个 L0C 里，fixpipe 成 FP32 回 sub0，再乘 `scale`。

Fixpipe 三条路，事件名与 `kda_32_32` 的 `gen_utils_*.h` 一致：

```text
L0C FP32 ──► UB FP32     gram、P、D、o_acc          FIXtoV_UBReady
L0C FP32 ──► UB BF16     不作为主路径
L0C FP32 ──► L1 BF16     A 已在 Vector 收好；w、v_new、H 由 Vector 写入 L1
```

`H`、`A`、`w`、`v_new` 的 BF16 舍入发生在 Vector，不发生在 fixpipe。这样舍入点和 `vec_tril_inv2` 的「寄存器里 FP32、写出时 ROUND_R」一致。


## 8. 流水排布与同步

实现走 CATLASS，目标是 A5（`CATLASS_ARCH=3510`，`Arch::Ascend950`）。数据交接用 `SYNC_L1_HANDOFF`：AIV 的 `CopyUbToL1` 直接写 AIC 的 L1，AIC 的 Fixpipe 直接写对应 AIV 的 UB。第 5 章的同步用的就是本节的 `HardEvent` 和 mode `0x4` 的 `CrossCore*Flag`。

参考 `chunk_kda_fwd` 的 arch35 路径（`arch35/chunk_kda_fwd_prepare_arch35.h` 的 `CrossCore*Flag<0x4, ...>`，以及 `catlass/arch/kda_cross_core_sync.hpp`）。

### 8.1 事件

两类指令，不要混用。

核内同一条 pipe 的先后，用 `AscendC::SetFlag` / `WaitFlag<HardEvent::*>`。入口 `pipe->AllocEventID<HardEvent::...>()`，出口先 `WaitFlag` 排空再 `ReleaseEventID`。`chunk_kda_fwd` 的 `BlockMmad` 在进循环前先 `SetFlag` 把空闲额度发出去，析构里再 `WaitFlag` 收回。

| HardEvent | 核 | 含义 |
| --- | --- | --- |
| `MTE2_V` | AIV | GM 进 UB 之后，VF 可以读 |
| `V_MTE2` | AIV | VF 用完这块 pingpong，下一次 MTE2 可以覆盖 |
| `V_MTE3` | AIV | VF 写完，MTE3 可以搬去 L1 或 GM |
| `MTE3_V` | AIV | MTE3 搬完，VF 可以复用这块 UB |
| `MTE3_MTE2` | AIV | MTE3 搬完，下一次 MTE2 可以覆盖同一块 UB |
| `MTE2_MTE1` | AIC | L1 装好，可以 LoadData 到 L0 |
| `MTE1_M` | AIC | L0A/L0B 装好，可以 BlockMmad |
| `M_MTE1` | AIC | BlockMmad 不再读这块 L0，可以装下一块 |
| `M_FIX` | AIC | BlockMmad 写完 L0C，可以 Fixpipe |
| `FIX_M` | AIC | Fixpipe 不再占 L0C，下一次 BlockMmad 可以复用 |

跨 AIC/AIV 用 `CrossCoreSetFlag` / `CrossCoreWaitFlag`。本算子两个 AIV 各持有一个 chunk，AIC 要单独等其中某一个，所以用 mode `0x4`，flag 按 `subBlockIdx` 和槽号分开。这是 arch35 `chunk_kda_fwd` 里 Fixpipe 直达某个 AIV UB 的写法：

```text
offset = slot + subBlockIdx * 16        slot = c % 2，取值 0 或 1
```

`subBlockIdx * 16` 和 arch35 `chunk_kda_fwd` 的 `DIRECT_UB_FLAG_STRIDE` 一样，两个 AIV 的 flag 错开。id 8、9、10 留给 `CrossCoreBarrier`，这里不用。

| flag | id | 生产者 | 消费者 | 含义 |
| --- | --- | --- | --- | --- |
| `l1Ready` | `0 + offset` | AIV `CrossCoreSetFlag<0x4, PIPE_MTE3>` | AIC `CrossCoreWaitFlag<0x4, PIPE_MTE1>` | 这个 chunk 本阶段要给 Cube 的 L1 已经写完 |
| `l1Free` | `2 + offset` | AIC `CrossCoreSetFlag<0x4, PIPE_MTE1>` | AIV `CrossCoreWaitFlag<0x4, PIPE_MTE3>` | AIC 不再读这块 L1，AIV 可以覆盖 |
| `ubReady` | `4 + offset` | AIC `CrossCoreSetFlag<0x4, PIPE_FIX>` | AIV `CrossCoreWaitFlag<0x4, PIPE_V>` | Fixpipe 的 FP32 已经进这个 AIV 的 UB |
| `ubFree` | `6 + offset` | AIV `CrossCoreSetFlag<0x4, PIPE_V>` | AIC `CrossCoreWaitFlag<0x4, PIPE_FIX>` | Vector 不再读这份 UB，AIC 可以再 Fixpipe |

同一个 id 上，下一次 `Set` 之前必须已经有配对的 `Wait`。一个 chunk 里 Stage 1、Stage 3、Stage 5 都会写 L1，它们复用同一个 `l1Ready[offset]`，靠上面的 free 额度把次数压在 2 以内。`Catlass::Arch` 里单向 flag 连续 `Set` 超过 `MAX_REVERSE_DEPTH`（15）会卡死；ready/free 成对之后用不到 `CrossCoreFlagWithReverse`。`chunk_kda_fwd` 的 score 队列才用 `CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>`，那是 mode `0x2`、两个 AIV 合起来给 AIC 一次通知。本算子的 `kkt` 按 chunk 分开算，不走那条聚合。

同一 head 上，后一个 chunk 的 `VecStoreH` 要看见前一个 chunk 写回的 `R`。两个 AIV 的 UB 不共享。更新 `R` 的 AIV 在 `CAST_RINT` 之后做 `V_MTE3`，把这份 BF16 `R` 写到本 block 共用的 L1 状态槽；另一个 AIV 从这里取走，再做自己的 `VecStoreH`。两边用 `Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>()` 会合，flag 是头文件里的 `AIV_INTER_SUBBLOCK_BARRIER`。每次调用内部是一次 `Set` 加一次 `Wait`，两个 AIV 调用次数相同。

kernel 入口先把 free 额度发出去，和 `BlockMmad` 进循环前的 `SetFlag` 一样：

```text
AIC   每个 offset：CrossCoreSetFlag<0x4, PIPE_MTE1>(l1Free)
      每个用到的 L0 槽：SetFlag<HardEvent::M_MTE1>、SetFlag<HardEvent::FIX_M>
AIV   每个 offset：CrossCoreSetFlag<0x4, PIPE_V>(ubFree)
      每个输入槽：SetFlag<HardEvent::V_MTE2>、SetFlag<HardEvent::MTE3_MTE2>
出口  按入口的相反顺序 WaitFlag / CrossCoreWaitFlag，再 ReleaseEventID
```

### 8.2 源码里的循环

`kda32_aiv.cce` 的 `Process` 和 `kda32_aic.cce` 的 `Process` 都是 `iterN = 2`，三段循环：

| 循环 | AIV 起点 | AIC 起点 | 做什么 |
| --- | --- | --- | --- |
| 前奏 | 3187 行，`iter = 0`，只跑一轮 | 1255 行 | 只有 `iter-0` 成立，`iter-1` 的收尾全部跳过 |
| 稳态 | 3686 行，`iter = 1 .. N-1` | 1864 行 | `iter-0` 的新 chunk 和 `iter-1` 的上一 chunk 插在同一次迭代里 |
| 收尾 | 4185 行，`iter = N` | 2473 行 | 不再 `vec_prepare`，只把最后一槽的收尾做完 |

一个 AIV 一次迭代碰两个 chunk。`vec_prepare_g3` 每次调用把 `gqkdb_cnt` 加 1（449 行），GM 起点是 `64*128*gqkdb_cnt + 32*128*subid`（443–446 行）。同一次迭代里先调 `vec_prepare_g1`（模板 id 0），后调 `vec_prepare_g1_auto_unroll_id_1`（模板 id 1，3941 行）。id 0 和 id 1 是这个 AIV 自己的两块 UB 槽，装的是它名下相邻的两个 cnt，中间隔着另一个 AIV 的 chunk。

`vec_iter_half` 只改本 AIV 的 `s_ub`（2545 行起从 `s_ub` 读、写回 `s_ub`）。源码里没有两个 AIV 之间的屏障。GDN 的后一个 chunk 在另一个 AIV 上，所以在源码「上一 chunk 把状态送进 L1、下一 chunk 拿它做 `P`」的位置加 `CrossCoreBarrier<0x1, PIPE_MTE3>()`。

### 8.3 稳态一次迭代的真实顺序

下面按 `kda32_aiv.cce` 3686 行那个循环的调用顺序写。左边是这次迭代新推进的 chunk（`iter-0`），右边是上一 chunk（`iter-1`）。GDN 没有 `kd2`，也没有把状态拆成 `s1/s2` 两半；表里用 Stage 名对应过去。flag 用第 8.1 节的 CATLASS 名字，`offset = slot + subBlockIdx * 16`。

| 源码顺序 | 新 chunk，槽 0 | 上一 chunk，槽 1 | GDN 在这里发的同步 |
| --- | --- | --- | --- |
| 3694–3695 | `vec_prepare_g1`：`WaitFlag<V_MTE2>`，MTE2 搬 `q/k/g/beta`，`SetFlag<MTE2_V>`，开始 Stage 1 | `ub_to_l1_v_kds1`：上一 chunk 的 `v_new` 进 L1 | 新 chunk `CrossCoreWaitFlag<0x4,PIPE_MTE3>(l1Free)` 之后才 `CopyUbToL1`。上一 chunk `CopyUbToL1` 之后 `CrossCoreSetFlag<0x4,PIPE_MTE3>(l1Ready)` |
| 3711 | | `ub_to_l1_qdkd`：上一槽已经打好的 NZ `q_hat/k_hat` 进 L1。进这次拷贝之前先 `Wait MTE1toMTE3_L1Release`（3703–3706） | 上一 chunk 的 `l1Free` 到了才覆盖 L1；写完 `l1Ready` |
| 3737–3738 | `vec_prepare_g2`：Stage 1 的后半，L2Norm、Sklansky、`vb` | 上一槽的 `kbg` 一类向量进 L1 | 新 chunk 仍在 VF，不发 `l1Ready` |
| 3762–3777 | | `vec_iter_partial1`、`vec_iter_final1`：等 `FIXtoV_UBReady(s1_delta)` 之后更新 `s_ub` 的前半 | GDN 把两半合成一次 Stage 7。入口 `CrossCoreWaitFlag<0x4,PIPE_V>(ubReady)` |
| 3788–3790 | | `ub_to_l1_s2`：刚更新的状态进 L1，随后 `MTE3toMTE1_L1Ready(s2)` | GDN 这里是 `CopyUbToL1` 把新 `R` 写成下一 chunk要用的 `H`，然后 `CrossCoreBarrier` |
| 3808 | | `vec_tril`：等 `FIXtoV_UBReady(kdki/qdki)` 之后做上一槽的 Stage 3 | `CrossCoreWaitFlag<0x4,PIPE_V>(ubReady)`，用完 `CrossCoreSetFlag<0x4,PIPE_V>(ubFree)` |
| 3821–3836 | | `vec_v_kds2` 和 `ub_to_l1_v_kds2`，与 `vec_tril_inv1` 叠在一起 | Stage 5 的 `CopyUbToL1` 和 Stage 3 的求逆可以重叠，它们不读同一块 UB |
| 3841–3843 | | `ub_to_gm_out1`：等 `FIXtoMTE3_UBReady(out)` 后把 `o` 写 GM | `SetFlag<HardEvent::V_MTE3>` / `WaitFlag`，`CopyUbToGm` |
| 3852 | | `vec_tril_inv2`，然后 3887 行 `ub_to_l1_mm` 把 `A` 送进 L1 | `CopyUbToL1` 之后 `CrossCoreSetFlag<0x4,PIPE_MTE3>(l1Ready)` |
| 3902–3903 | `vec_prepare_qdkd_kikr`：Stage 1 的 NZ 打包 | 同时 `ub_to_l1_s1` 把上一 chunk 的另一半状态送进 L1 | 打包只动本槽 UB。状态那次 `CopyUbToL1` 用的是上一槽 |
| 3941–3942 | `vec_prepare_g1` 的槽 1：这个 AIV 的下一个 chunk 开始 MTE2 | 上一槽剩余的 `v_new` 进 L1 | 和 3694 行同一对重叠，只是槽号对调 |
| 3959 | `ub_to_l1_qdkd(iter-0)`：本槽刚打好的 NZ 进 L1。前面 3951–3954 行先等这四块 L1 的 `L1Release` | | `CrossCoreWaitFlag(l1Free)`，`CopyUbToL1`，`CrossCoreSetFlag(l1Ready)` |
| 3985–3986 | 槽 1 的 Stage 1 后半，并把它的向量送进 L1 | | 同 3737 行 |
| 4009–4023 | | 槽 1 对应的上一 chunk 做 `vec_iter_partial1/final1` | Stage 7 |
| 4053 | `vec_tril(iter-0)`：本槽的 `kkt/qk` 已经 Fixpipe 回来 | | Stage 3。AIC 的 `kkt` 在更早的迭代里做完，不在这个 VF 前面 |
| 4068–4080 | | 上一槽 `vec_v_kds2` 进 L1，和本槽 `vec_tril_inv1` 重叠 | 同 3821 行 |
| 4085–4087 | | `ub_to_gm_out1` 写上一槽的 `o` | `CopyUbToGm` |
| 4098 | `vec_tril_inv2(iter-0)` | | Stage 3 结束 |
| 4135 | `ub_to_l1_mm(iter-0)`：本槽的 `A` 进 L1 | | `l1Ready` |
| 4150–4151 | 槽 1 的 `vec_prepare_qdkd` 和本槽 `ub_to_l1_s1` 重叠 | | 同 3902 行，槽号对调 |
| 4169 | | `ub_to_gm_out2` | 上一槽的另一半 `o` |
| 4177 | `vec_v_kds1(iter-0)` | | 本槽 Stage 5，结果留到下一次迭代开头再 `CopyUbToL1` |

AIC 稳态在 `kda32_aic.cce` 1864 行，吃的是 AIV 已经 `MTE3toMTE1_L1Ready` 的 L1，不跟 AIV 的 VF 串行。一次迭代里 Cube 的顺序是：

```text
1881  Wait L1Ready(v_kds1)
1885  l1_to_l0ab_y1，1892 matmul_y1          上一 chunk 的 u/P 这一类，GDN 的 Stage 4 / Stage 6 前半
1900  Wait L1Ready(mqk)、L1Ready(kr)
1916  matmul_iter_out1                       D 和 o_acc。Fixpipe 回对应 AIV，发 ubReady
1951  Wait L1Ready(kd)、L1Ready(ki)          另一个槽的 k_hat 到了才做
1979  matmul_kdki                            GDN 的 CubeKkt。不等状态
1996  matmul_qdki                            GDN 的 CubeQk
1989  Wait L1Ready(s2)                       状态进 L1 之后才
2020  matmul_kds_qds2                        读 H 的那次乘，GDN 的 P = w @ H 和 o_inter = q_exp @ H
2046  Wait L1Ready(v_kds2)，2057 matmul_y2，2082 matmul_iter_out2
```

`matmul_kdki` 的等待对象是 `kd/ki` 的 `L1Ready`，不是 `s2`。`s2` 的 `Wait` 在 1989 行，夹在 `qdki` 和 `kds_qds` 之间。所以 `kkt` 可以和上一 chunk 的 `v_new`、写 `o` 重叠；`P = w @ H` 必须等上一 chunk 的状态已经 `CopyUbToL1`。

### 8.4 前奏和收尾

前奏就是 3187 行那个循环，`iter-1` 的条件全假：

```text
只跑 vec_prepare_g1 / g2 / qdkd、ub_to_l1_qdkd、vec_tril、inv、vec_v_kds。
没有 vec_iter，没有 ub_to_gm_out，没有上一 chunk 的 L1Release 可等。
H 来自 InitState 写进本 AIV s_ub 的 R。第一个 chunk 不做 CrossCoreBarrier。
```

收尾是 4185 行，`iter-0` 的 `vec_prepare_*` 条件为假，只留下 `iter-1` 的 `vec_iter`、`ub_to_l1_s*`、`ub_to_gm_out`、`vec_v_kds` 的 L1 搬运。`final_state` 的 `CopyUbToGm` 接在最后一次 `vec_iter_half` 的 `SetFlag<HardEvent::V_MTE3>` / `WaitFlag` 之后。两个 AIV 的 `CrossCoreBarrier` 调用次数要一样，否则后一个 chunk 的 `VecStoreH` 等不到。

### 8.5 流水示意图

槽 0、槽 1 是同一个 AIV 的两块 UB。`c0` 和 `c2` 都落在 sub0，`c1` 落在 sub1，所以 `c1` 的加载不在 sub0 的这次迭代里。下图只画 sub0，重叠关系来自 3694 与 3695、3902 与 3903、3835 与 3843 这几对调用。

```text
sub0 的迭代:   iter0（前奏）          iter1（稳态）
槽:            槽0 = c0               槽0 = c2          槽1 = c0 的收尾

AIV prepare    [====]                 [====]
AIV pack         [==]                   [==]
AIC KKT            [==]                   [==]     不等 R，1989 行之前就做完
AIV tril             [====]
AIV inv                [====]               [====]  inv 与上一槽写 o 重叠（3835/3843）
AIC P,o                  [====]
AIV vnew                   [==]
AIC D                        [=]
AIV state                    [=]
AIV store o                  [=]
CopyUbToL1 v_new               [=]          下一次迭代开头才搬（3695 行的 iter-1）

                           [====]          c2 的 prepare 盖住 c0 的 v_new→L1
                             [==]          c2 的 pack 盖住 c0 的状态→L1（3902/3903）
                               [==]        c2 的 KKT
```

`c1` 在 sub1 上跑同一份 `Process`。sub1 的 `VecStoreH` 要等 sub0 把 `c0` 的 `R` 写进共用 L1，这个等待是 `CrossCoreBarrier<0x1, PIPE_MTE3>`。源码的 `vec_iter_half` 没有这一次屏障，因为它只更新本 AIV 的 `s_ub`。

## 9. VF 设计

VF 都在 AIV 上，每个 AIV 对自己那一个 32-token chunk 跑同一套函数。`q/k/v` 都是 `[32,128]`，`g` 是这个 chunk 自己的 `[32]` FP32，`beta` 是 `[32]` BF16。状态 `R` 和 `H` 都是 `[128,128]` BF16。16×16 只出现在求逆的分块和 NZ 分形里。

外层按 CATLASS 的 mix kernel 写：`ASCEND_IS_AIV` 里按第 5 章的 Stage 调用下面这些 VF，`ASCEND_IS_AIC` 里用 `BlockMmad` 做 9.12 的乘。搬运用带 layout 的 `CopyGmToUb` / `CopyUbToL1` / Fixpipe 到该 AIV 的 UB。同步用第 8 章的 `HardEvent` 和 mode `0x4` 的 `CrossCore*Flag`，`offset = slot + subBlockIdx * 16`。

舍入：`CAST_RINT` 是 BF16 的 round-to-nearest-even。求逆时行与行之间用 FP32 寄存器里的值。v1 的 `T` 已经是 32 的倍数，下面的循环上界都是 32。

### 9.1 `InitConst`

每个 AIV 在进 chunk 循环前做一次。求逆在持有该 chunk 的 AIV 上做，所以两份 UB 都要有自己的常数。

```text
mask[i,j] = 1 if i>j else 0          32×32 bit
eye16     = I_16                     FP32
```

留在 UB 常驻区，不读 GM。大约 1.2 KiB。

### 9.2 `InitState`

把本 AIV 的常驻 `R[128,128]` BF16 设成初值。

```text
若没有 initial_state：
    R 写成 0
否则：
    CopyGmToUb，把这个 value head 的 initial_state[128,128] BF16 搬进 R
    SetFlag<HardEvent::MTE2_V>
    WaitFlag<HardEvent::MTE2_V>
```

两个 AIV 都持有完整的 `[128,128]`。第一个 chunk 的 AIV 用这份 `R` 做 `VecStoreH`。后一个 chunk 的 AIV 在 `CrossCoreBarrier` 之后，改用前一个 chunk 写到共用 L1 的那份 `R`。

### 9.3 `LoadQkvg`

把本 chunk 的原始输入搬进 UB pingpong。这是搬运，计算在后面的 VF 里。

```text
q, k, v     [32,128] BF16
g           [32]     FP32
beta        [32]     BF16
```

GM 起点按第 6 章：这个 AIV 的 chunk 从 `32 * subBlockIdx` 个 token 起，下一次循环再加 64。搬完 `SetFlag<HardEvent::MTE2_V>(slot)`，VF 前 `WaitFlag`。覆盖这块 pingpong 之前 `WaitFlag<HardEvent::V_MTE2>(slot)`。

### 9.4 `VecCumsumGate`

L2Norm、beta sigmoid、标量 cumsum、`e_g`、`vb`。门控是每个 token 一个标量。`eps = 1e-6`。

```text
beta_eff[i] = sigmoid(fp32(beta[i]))
rstd_q[i]   = rsqrt(sum_{d=0}^{127} q[i,d]^2 + eps)
rstd_k[i]   = rsqrt(sum_{d=0}^{127} k[i,d]^2 + eps)
q_hat[i,:]  = q[i,:] * rstd_q[i]
k_hat[i,:]  = k[i,:] * rstd_k[i]

g' = CumsumScaleVF(g, RCP_LN2)     # Sklansky，深度 5，原地 FP32
e_g[i] = exp2(g'[i])
vb[i,:] = v[i,:] * beta_eff[i]
```

平方和在这 128 列上做完。`beta_eff`、`g'`、`e_g` 保持 FP32，留在 UB，给 `VecTril` 和 `VecVnew` 用。`q_hat`、`k_hat`、`vb` 按 16×16 NZ 写成 BF16，`CAST_RINT`。写完 `SetFlag<HardEvent::V_MTE3>`，`CopyUbToL1` 之后由第 8.3 步发出 `l1Ready`。

### 9.5 `VecPack`

用已经在 UB 里的 `k_hat`、`q_hat`、`beta_eff`、`e_g` 做出 `kbg` 和 `q_exp`。

```text
for i in 0..31:
    kbg[i,:]   = CAST_RINT(k_hat[i,:] * beta_eff[i] * e_g[i])
    q_exp[i,:] = CAST_RINT(q_hat[i,:] * e_g[i])
```

形状都是 `[32,128]` BF16，NZ。相对门控 `exp2(g'[i]-g'[j])` 留到 `VecTril`，乘在已经算完的 `kkt` 和 `qk` 上。`kbg`、`q_exp` 随 Stage 1 的其余矩阵一起 `CopyUbToL1`。

### 9.6 `VecTril`

把本 chunk 的 FP32 `kkt[32,32]`、`qk[32,32]` 收成严格下三角 `L` 和因果 `attn`。入口是 `CrossCoreWaitFlag<0x4, PIPE_V>(ubReady + offset)`，对应 Stage 2 的 Fixpipe。

```text
for i in 0..31:
    for j in 0..31:
        d = exp2(g'[i] - g'[j])
        L[i,j]    = beta_eff[i] * d * kkt[i,j]    if i > j else 0
        attn[i,j] = d * qk[i,j]                   if i >= j else 0
```

`g'` 已经乘过 `RCP_LN2`，相减后再 `exp2` 等于 `exp(cumsum_i - cumsum_j)`。`attn` 含对角，`L` 不含对角。`L` 保持 FP32。`attn` 先留在 UB，到 Stage 5 结束才 `CAST_RINT` 并 `CopyUbToL1`。用完 `kkt/qk` 之后 `CrossCoreSetFlag<0x4, PIPE_V>(ubFree + offset)`。

### 9.7 `VecTrilInv`

`A = (I+L)^{-1}`。在持有这个 chunk 的 AIV 上做，输入是该 AIV 自己的 `L[32,32]`。输出 `A[32,32]` BF16 NZ，`CopyUbToL1`。

16×16 是这一个 `[32,32]` 内部的分块：

```text
L = | L00   0  |
    | L10  L11 |
```

`VecTrilInv1` 对 `L00`、`L11` 做前向替换，解 `(I+L00) X = I`。对角是 1，不除。

```text
for col in 0..15:
    for row in 0..15:
        acc = (row==col) ? 1 : 0
        for k in 0..row-1:
            acc = acc - L00[row,k] * X[k,col]
        X[row,col] = acc
```

`VecTrilInv2` 合成 32×32。两次 16×16 乘在 Vector 上用 FP32 做。

```text
A00 = X00
A11 = X11
A01 = 0
A10 = - X11 @ L10 @ X00
```

一行的 `X00/X11/L10` 用 FP32。整行 `A` 写出时 `CAST_RINT` 到 `A_bf16`。下一行不读刚写出的 BF16。写完 `SetFlag<HardEvent::V_MTE3>`，再 `CopyUbToL1`。

### 9.8 `u` 不在 VF 里

`vb = v * beta_eff` 已经在 `VecCumsumGate` 里完成，并在 Stage 1 进了 L1。`u = A @ vb` 是 9.12 的 `CubeU`，`mmad<32,32,128>`，Fixpipe 成 FP32 回到这个 AIV 的 UB。

### 9.9 `VecVnew`

`v_new = u - P`，再乘 chunk 内相对门控得到 `v_new_g`。入口 `CrossCoreWaitFlag<0x4, PIPE_V>(ubReady + offset)`，`u` 和 `P` 都是 FP32 `[32,128]`。

```text
g'_last = g'[31]
for i in 0..31:
    v_new[i,:]   = u[i,:] - P[i,:]
    v_new_g[i,:] = CAST_RINT(v_new[i,:] * exp2(g'_last - g'[i]))
```

`attn @ v_new` 用没乘门控的 `v_new`，`CAST_RINT` 后和 `v_new_g`、`attn` 一起 `CopyUbToL1`。搬之前 `SetFlag<HardEvent::V_MTE3>` / `WaitFlag`。三份都进 L1 后 `CrossCoreSetFlag<0x4, PIPE_MTE3>(l1Ready + offset)`，并 `CrossCoreSetFlag<0x4, PIPE_V>(ubFree + offset)`。

### 9.10 `VecState`

更新这个 head 的整份状态。输入是 `CubeD` fixpipe 回来的 `D[128,128]` FP32，以及 `exp2(g'_last)`。入口同样是 `CrossCoreWaitFlag<0x4, PIPE_V>(ubReady + offset)`。

```text
decay = exp2(g'[31])
for r in 0..127:
    for c in 0..127:
        R[r,c] = CAST_RINT(decay * fp32(R[r,c]) + D[r,c])
```

乘加在 FP32 寄存器里做，写回同一块 BF16 `R`。写完 `SetFlag<HardEvent::V_MTE3>` / `WaitFlag`，把这份 `R` 拷到本 block 共用的 L1 状态槽，供下一个 chunk 的 `VecStoreH` 使用。最后一个 chunk 再把同一份 BF16 写到 `final_state`。

### 9.11 `VecStoreH` 与 `VecEpilogueO`

`VecStoreH` 在本 chunk 的 `CubeP` 之前，把入口状态收成 L1 上的 `H[128,128]` BF16。第一个 chunk 的源是 `InitState` 放在 UB 里的 `R`。后面的 chunk 先 `CrossCoreBarrier<0x1, PIPE_MTE3>()`，再使用上一 chunk 写到共用 L1 的那份 `R`。`H` 写进 L1 之后，本 chunk 的 `w@H` 和 `q_exp@H` 都用它。随后 `VecState` 更新 UB 里的 `R`，不改这块 `H`。

`VecEpilogueO` 在持有该 chunk 的 AIV 上做。`o_acc` 是 `o_inter + attn @ v_new` 的 FP32 `[32,128]`。

```text
o[i,:] = CAST_RINT(o_acc[i,:] * scale)
```

`scale` 是输入，只乘这一次。`SetFlag<HardEvent::V_MTE3>` / `WaitFlag` 之后，按第 4 章的 `o` layout 把 `[32,128]` 写到本 chunk 的 token 起点。

### 9.12 Cube 任务

在 AIC 上，用 CATLASS `BlockMmad`。操作数 BF16，L0C 累加 FP32。每个 chunk 单独做，Fixpipe 回到持有该 chunk 的 AIV。开始前 `CrossCoreWaitFlag<0x4, PIPE_MTE1>(l1Ready + offset)`，Fixpipe 前 `CrossCoreWaitFlag<0x4, PIPE_FIX>(ubFree + offset)`，写完 `CrossCoreSetFlag<0x4, PIPE_FIX>(ubReady + offset)`。L0 的复用用 `M_MTE1`、`M_FIX`、`FIX_M`。

| 名字 | 公式 | 分型 | 结果去向 |
| --- | --- | --- | --- |
| `CubeKkt` | `k_hat @ k_hat^T` | `mmad<32,128,32>` | 该 AIV 的 UB，`kkt[32,32]` FP32 |
| `CubeQk` | `q_hat @ k_hat^T` | `mmad<32,128,32>` | 该 AIV 的 UB，`qk[32,32]` FP32 |
| `CubeU` | `A @ vb` | `mmad<32,32,128>` | 该 AIV 的 UB，`u[32,128]` FP32 |
| `CubeW` | `A @ kbg` | `mmad<32,32,128>` | L1 `w[32,128]` BF16 |
| `CubeP` | `w @ H` | `mmad<32,128,128>` | 该 AIV 的 UB，`P[32,128]` FP32 |
| `CubeD` | `k_hat^T @ v_new_g` | `mmad<128,32,128>` | 该 AIV 的 UB，`D[128,128]` FP32 |
| `CubeOInter` | `q_exp @ H` | `mmad<32,128,128>` | 留在 L0C |
| `CubeAttnV` | `attn @ v_new` | `mmad_acc<32,32,128>` | 加进同一块 L0C，再 Fixpipe 成 `o_acc` |

`CubeW` 的 Fixpipe 直接写成 L1 上的 BF16，`w` 没有 FP32 消费者。`P`、`D`、`o_acc` 以 FP32 回 UB。`CubeOInter` 和 `CubeP` 用的是同一份 `H`。Stage 6 结束、这块 L1 槽不再被读之后，AIC 才 `CrossCoreSetFlag<0x4, PIPE_MTE1>(l1Free + offset)`。


## 11. 精度与标杆

### 11.1 标杆文件

独立 CPU 标杆：

```text
tests/atk/chunk_gated_delta_rule_fwd/cpu_golden_fused_infer.py
```

两个入口：

| 函数 | 作用 |
| --- | --- |
| `run_infer_golden` | 逐 token FP64 递推，验收基准。调用时 `use_qk_l2norm=True`，`use_beta_sigmoid=True` |
| `chunk32_reference` | 第 5.2 节的 BT=32 WY 形式，两个开关与上面相同 |

两者都只返回 `o` 和 `final_state`。`g_layout="BTH"` 时 `g/beta` 是 `[B,T,HV]`，与 ATK 公开布局一致；`g_layout="BHT"` 时是 kernel 看到的 `[B,HV,T]`。

同目录的 `gdn_reference.py` 是公开融合算子的双标杆，返回值还包含 `g_cumsum` 和 `A`，chunk 可以是 64 或 128。本 kernel 不拿它当输出合同。它的逐 token 递推和 `run_infer_golden` 是同一条数学定义，可以用来核对标杆本身有没有写歪。

### 11.2 比较方式

1. 同一份冻结输入，CPU 跑 `run_infer_golden`，NPU 跑本 kernel。
2. `o` 比较前把 FP64 golden cast 成 BF16 再转回 FP32，与 kernel 的 BF16 读回值比。
3. `final_state` 两边都先按 BF16 比较。golden 是 FP64 递推后再 `CAST_RINT` 成 BF16。kernel 每个 chunk 都把状态舍入一次，终态允许比 `o` 更大的误差。
4. 阈值先沿用 ATK 双标杆的量级：最大相对误差比例 5，平均相对误差比例 1.5，均方根误差比例 1.5。若 BF16 状态快照让终态超阈，单独报 `final_state`，不要把 `o` 的阈值一起放宽。

中间量对拍（不作为上库门禁，调试用）：在 `chunk32_reference` 上把 `A`、`v_new`、每 chunk 出口的 `R` 打出来，与仿真 UB 比较。`A` 允许 BF16 舍入，`R` 的 FP32 路径应接近 golden 的 chunk 出口，差别只来自 `H` 和 `A` 的 BF16。



