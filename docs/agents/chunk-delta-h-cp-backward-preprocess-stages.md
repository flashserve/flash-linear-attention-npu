# `pre_process_bwd_kernel_merged` Stage 划分（Ascend C）

## 1. 文档定位与范围

本文按 `kda-pipeline-execution-contract.drawio` 三页合同（`01 Prepare` / `02 FwdH` / `03 Finalize`）的写法，给出
CP 反向状态预处理 Kernel `pre_process_bwd_kernel_merged` 的 Stage 划分。

- 数学语义、入参含义、跨 rank 合并方式见同目录
  [`chunk-delta-h-cp-backward-preprocess.md`](chunk-delta-h-cp-backward-preprocess.md)，本文不重复推导，只定义
  Ascend C 侧的 Stage 边界、引擎归属、入口/出口合同和分核规则。
- 上游一次 launch 用 `grid = (cdiv(V, BS) + cdiv(K, BS), HV)` 把"V 分支"和"K 分支"映射成同一个 Kernel 的不同
  program。本文把这两支显式拆成两组 Stage，并给出共享操作数的复用方式。
- 本文只描述 Stage 合同，不描述通信编排，也不描述后续 merge（`merge_fwd_bwd_kernel`）。
- 物理 Stage 划分采用合并后的 **6 Stage** 版本：`V0 → C1 → V2 → C3 → V4 → C5`。原八 Stage 版本
  （`C1`/`C5` 分开、`V4`/`V6` 分开）数学等价，仅在需要把 P 链从 E 链解耦时回退使用。

### 1.1 本次必须支持

| 维度 | 必须支持的取值 |
| --- | --- |
| gate 模式 | `USE_G`（GDN 标量 gate）、`USE_GK`（KDA/GDN2 逐 K gate）、无门控（`g`/`gk` 均为空） |
| 序列形态 | dense；varlen（`IS_VARLEN`，一次 launch 一个 `[bos,eos)`） |
| 状态行维 | `K <= 256`，K 方向按 64 行分组，最多 4 组（对应上游 `b_dh1..b_dh4`） |
| 状态列维 | `V` 任意，列方向按 `BLOCK_SIZE` tile，`BLOCK_SIZE = 32 if K <= 64 else 64` |
| 尾块 | chunk 内有效行 `M < BT`、K/V 方向尾 tile 都必须支持 |
| head 映射 | `H_qk == HV` 与 `HV > H_qk`（`hk = hv // (HV / H_qk)`，host 校验 `HV % H_qk == 0`） |
| 状态布局 | `state_v_first` 只影响 `dht`/`initial_state` 的 GM 布局，不改变 `dhm` 的 `[K, V+K]` 逻辑布局 |
| dtype | `q/k/w/do/dv` 为 FP16 或 BF16；`dhm`、矩阵链累计、状态累计固定 FP32 |

### 1.2 本次明确不支持

| 排除项 | 含义 |
| --- | --- |
| `USE_BG`（DPLR） | 不做用 `bg` 顶替 `k` 入参、不做 `+ W^T K̄` 加号链、不做不带 `scale` 的 query 项；`w` 固定按 GDN/KDA 的 `[B,T,HV,K]` 寻址 |
| `AFFINE_CHAIN_PRECISION` | P 链固定 FP32 语义，不引入 `tf32x3`/`ieee` 等输入精度开关，也不新增该 tiling 字段 |

Host 侧门禁：`USE_G` 与 `USE_GK` 互斥，两者同时为真直接拦截；上游没有显式 assert，NPU 侧不能沿用这个缺口。

## 2. 符号

| 符号 | Shape | 含义 |
| --- | --- | --- |
| `BT` | 标量 | chunk size，必须与 gate cumsum、WY 生成、正式 backward 一致 |
| `M` | 标量 | 当前 chunk 的有效 token 数，`0 < M <= BT` |
| `NT` | 标量 | 本 segment 的 chunk 数 `cdiv(T, BT)`，反向扫描 |
| `BS` | 标量 | `BLOCK_SIZE`，本 Stage 合同中所有列 tile 的宽度 |
| `K` / `V` | 标量 | 状态行维 / 列维，也是 `q/k` 单 head 维与 `do/dv` 单 head 维 |
| `ns` | 标量 | `HV / H_qk`（每个 q/k head 对应的 value head 数），GVA 下 `hk = hv // ns` |
| `dH` | `[K, V]` | 状态梯度 accumulator，本 Kernel 内初值为 0，末值即 `E_r` |
| `P` | `[K, K]` | 反向状态转移矩阵，初值为单位阵，末值即 `P_r` |
| `dhm` | `[HV, K, V+K]` FP32 | 唯一输出，`[..., :V] = E_r`，`[..., V:] = P_r` |

## 3. 两个分支的数学目标

按 chunk 从最后一个（`i_t = NT-1`）向第一个扫描。定义：

```text
Q̄_c  = q·2^{g}                      (USE_G)
     = qg                            (USE_GK)
     = q                             (无门控)
K̄_c  = k·2^{g_last - g}              (USE_G)
     = kg                            (USE_GK)
     = k                             (无门控)
decayK_c = 2^{g_last}                (USE_G，对标量，沿 K 广播)
         = 2^{gk_last}               (USE_GK，逐 K)
         = 1                        (无门控)
Q̄s_c = scale · Q̄_c                   (scale 折进操作数)
```

### 3.1 E 分支（`dH`，写 `dhm[..., :V]`）

```text
dV_pre = K̄_c @ dH_old                (Cube)
dV̂'    = -(dV_pre + dv_local)        (Vector，取负号让 C3 只做累加)
inc    = Q̄s_c^T @ do_c + W_c^T @ dV̂' (Cube，两个 MMAD 累加进同一 L0C)
dH_new = decayK_c ⊙ dH_old + inc     (Vector)
E_r    = dH 在 i_t = 0 处理完后的值
```

关键等价关系：上游把 `2^{g_last - g}` 乘在 `K̄ @ dH` 的乘积上（`b_dv *= where(m_t, ...)`）。
由于该衰减只沿 token 轴逐行作用，先乘在 `K̄` 上与后乘在乘积上完全等价。本设计统一把衰减折进 `K̄`
（K 分支本来就需要带衰减的 `K̄`），因此：

- `USE_G` 与 `USE_GK` 共用同一套 Cube 公式，不再为 `USE_G` 单独引入一个 token 衰减 Stage；
- 代价是 `V0` 必须把 `[M, BT)` 的无效 token 行在 `K̄` 上写零，才能复现上游 `where(m_t, ..., 0)` 的
  掩码语义。

### 3.2 K 分支（`P`，写 `dhm[..., V:]`）

```text
T1     = W_c^T @ K̄_c                 (Cube，[K,K])
P_c    = diag(decayK_c) - T1         (Vector，只在 row == col 处加对角项)
P_new  = P_c @ P_old                 (Cube，FP32 链，初值 P = I)
P_r    = P 在 i_t = 0 处理完后的值
```

链的方向固定为 `P_new = P_c @ P_old`，最终得到 `P_r = P_0 P_1 ... P_{NT-1}`。禁止写成
`P_old @ P_c`。

## 4. 与已发布 Stage 合同的对齐

| 本合同 | 参考对象 | 复用关系 |
| --- | --- | --- |
| `V0` 门控/衰减/操作数准备 | `01 Prepare` 的 `V0`/`V6` | 同样是"每 chunk 一次 VF 准备，Vector 结果经 GM/workspace 交给 Cube" |
| `C1`/`V2`/`C3`/`V4` | `02 FwdH` 的 `stage0..stage3` | 完全同构：状态交替作为 Cube 操作数和 Vector 累加器，跨引擎经 GM/workspace 往返 |
| `C1` 单 Stage 两路 MMAD | `01 Prepare` 的 `C2` | 与 C2 的"S=4 一次入口、内部多路 MMAD"同类：入口操作数一次备齐，内部展开多路并分别回写 |
| `C3` 一个 L0C 累加两个 MMAD | `03 Finalize` 的 `stage0` | 沿用"两个乘积累加到同一 L0C、只做一次 Fixpipe"的做法 |
| `V4` 一次 VF 两路输出 | `01 Prepare` 的 `V6` | 与 V6 的"同一次 VF 产出多个操作数平面"同类 |
| `C5`（原 `C7`） | 本页新增 | `P` 链没有正向对应物；`P_c` 的对角注入与 FwdH 的 `S-1` 初值处理同级 |
| 尾块与 padding | `01 Prepare` 的 Tail/GM 段 | token 行尾块与列尾 tile 都必须在 UB/payload 内生成，不能把 padding 写回 GM |

## 5. 公共定义与分核规则

```text
tileV = cdiv(V, BS)          每个 head 的 V 方向列 tile 数
tileK = cdiv(K, BS)          每个 head 的 K 方向列 tile 数
TILE  = tileV + tileK        每个 head 的列 tile 总数
NT    = cdiv(T, BT)          本 segment 的 chunk 数
N     = 本阶段可接收任务的 AIC 工作组数
```

1. **禁止按 chunk 分核。** 两个分支都是沿 chunk 的反向链（`dH` 与 `P` 都是跨 chunk 状态），同一 head 的全部
   chunk 必须留在同一工作组内按依赖顺序执行。这一点与 `02 FwdH` 的"有 chunk 依赖"规则相同。
2. **默认（`HV >= N`）：仅按 head 连续分核。** `groupHeads = ceil(HV/N)`，工作组 `i` 的范围为
   `[i*groupHeads, min((i+1)*groupHeads, HV))`，最后一核可以少于 `groupHeads`，空核直接退出。工作组内部
   遍历该 head 的全部 `TILE` 个列 tile。
3. **补充分核（`HV < N`）：按 `(hv, 列 tile)` 展平。** 把 `HV * TILE` 个 task 按 balanced half-open range
   均分，分配顺序先 `tileV` 再 `tileK`。仍禁止切 chunk：**列 tile 之间相互独立，chunk 之间不独立**。
   - V 方向列 tile：`dH[:, v_off:v_off+BS]` 的递推只用到 `dH` 的同一列区间，`do`/`dv_local` 的同一列区间；
   - K 方向列 tile：`P[:, k_off:k_off+BS]` 的递推只用到 `P` 的同一列区间；
   - 该列独立性是补充分核唯一的合法依据，不能用"chunk 数够多"作为切 chunk 的理由。
4. **task 的 Stage 子集不同。** 只拿到 `tileV` 的 task 只需要 `V0 → C1(路 A) → V2 → C3 → V4(路 A)`；
   只拿到 `tileK` 的 task 只需要 `V0 → C1(路 B) → V4(路 B) → C5`。`K̄` 与 `decayK` 两支都要，
   `Q̄s` 只有 E 分支要。合并 Stage 的两路输出互相独立，展平分核时 task 只执行属于自己分支的那一路，
   另一路既不参与计算，也不需要为它准备或发布 ready/free。
5. **多 segment。** Kernel 一次 launch 只处理一个 `[bos,eos)`（上游 `i_n` 固定为 0）。varlen 下需要多个跨
   rank segment 时，由 host 多次 launch，或显式增加 segment 维 grid 与对应 offset metadata；不能让一个
   工作组隐式遍历 `cu_seqlens` 的全部 sequence。
6. **头映射。** `hk = hv // ns`，`ns = HV / H_qk`；`HV % H_qk != 0` 必须 host 拦截。不允许要求物理 repeat
   `q/k`。

## 6. Stage 总表

编号规则：`V*` 为 Vector Stage，`C*` 为 Cube Stage，数字是执行序号。

合并说明：原 `C5`（`T1`）并入 `C1`，原 `V6`（`P_c`）并入 `V4`，物理 Stage 数由 8 降为 6。
被合并的两路输出互相独立，合并只共享同一个 Stage 入口和同一份操作数装载，不改变数学语义；合并后的
名称按 `V/C` 交替重新编号为 `V0 → C1 → V2 → C3 → V4 → C5`。

编号对照（六 Stage ← 八 Stage）：`V0 ← V0`；`C1 ← C1 + C5`；`V2 ← V2`；`C3 ← C3`；`V4 ← V4 + V6`；
`C5 ← C7`。原 `C5`（`T1`）与原 `V6`（`P_c`）已并入前序 Stage，编号不复用。

| 阶段 | 分支 | 引擎 | 公式 | 入口必须 ready | 出口发布 |
| --- | --- | --- | --- | --- | --- |
| `V0` | 共享 | Vector | `Q̄s = scale·Q̄_c`；`K̄_c`；`decayK_c`；`[M,BT)` 行写零 | 本 chunk 的 `q/k/g`（或 `qg/kg/gk`）GM 可见 | `V0ExportReady`（每 chunk slot，多读者） |
| `C1` | E+P（原 C1+C5） | Cube | 路 A：`dV_pre = K̄_c @ dH_old`；路 B：`T1 = W_c^T @ K̄_c` | `V0ExportReady` + `dH` 旧值可见（首 chunk 为 0）+ `W_c` GM 可见 | `C1DvPreReady`、`C1T1Ready` |
| `V2` | E | Vector | `dV̂' = -(dV_pre + dv_local)` | `C1DvPreReady` + `dv_local` GM 可见 | `V2DvHatReady`、`C1DvPrePayloadFree` |
| `C3` | E | Cube | `inc = Q̄s_c^T @ do_c + W_c^T @ dV̂'` | `V2DvHatReady` + `Q̄s` 在 L1、`do_c` GM 可见 | `C3IncReady`、`V2SourceFree` |
| `V4` | E+P（原 V4+V6） | Vector | 路 A：`dH_new = decayK_c ⊙ dH_old + inc`；路 B：`P_c = diag(decayK_c) - T1` | `C3IncReady` + `C1T1Ready` + `dH` 旧值可见 + `decayK` | `V4DhtReady`（下一 chunk 的 `C1`，末 chunk 写 `dhm[..., :V]`）、`V4PcReady`（本 chunk 的 `C5`）、`C3PayloadFree`、`C1T1PayloadFree`、`DhtPrevFree` |
| `C5`（原 `C7`） | P | Cube | `P_new = P_c @ P_old`（按 K 归约展开硬件 tile，L0C 累加） | `V4PcReady` + `P` 旧值可见（首 chunk 为 `I`） | `C5PReady`（下一 chunk 的 `C5`，末 chunk 写 `dhm[..., V:]`）、`V4PcPayloadFree`、`PPrevFree` |

初值不需要独立 Stage：

- `dH` 初值 0：第一次 `C1` 的 `dH_old` 用零填充，等价于首轮 `V4` 直接取 `inc`；
- `P` 初值 `I`：第一次 `C5` 用单位阵初始化 L0C 累加器，等价于 `03 Finalize` 的 `L0C` 初始化做法。

## 7. 各 Stage 详细合同

### 7.1 `V0` · 门控与衰减准备（每 chunk 一次，两个分支共用）

```text
USE_G   : Q̄s = scale·(q ⊙ 2^{g})，K̄ = k ⊙ 2^{g_last - g}，decayK = 2^{g_last}
USE_GK  : Q̄s = scale·qg，          K̄ = kg，                   decayK = 2^{gk_last}
无门控   : Q̄s = scale·q，           K̄ = k，                    decayK = 1
```

- 一次 VF 覆盖整个 `[BT, K]`；`K > 64` 不拆 VF，但 K 方向 caster/循环按现行 Vec 模板处理。
- `[M, BT)` 的无效 token 行必须在 `K̄` 上写零；`Q̄s` 的无效行也必须为零（`Q̄s^T @ do` 的规约维是 token）。
- base-2 指数先按 `ScoreStorage`/`GateStorage` 截断再求值：FP16 走 `[-80, 80]`，BF16 走 `[-126, 120]`，
  与 `01 Prepare` 的 `V6` 规则一致，不允许在一个 Kernel 内换另一套截断区间。
- `scale` 只乘 `Q̄`，不乘 `K̄`、`W`、`do`；如果调用方传入的是已经带 `scale` 的 `qg`，host 必须显式约定，
  不能"两者都乘"。
- 输出经 MTE3 写 workspace slot；`Arch35` 下允许考虑 Fixpipe/直通到配对 AIV UB 的等价路径，`Arch22` 走具名
  GM relay。`Q̄s/K̄/decayK` 的实际布局与对齐按目标 CANN 版本核对后再定稿。
- `V0` 是唯一读原始 GM 输入的 Vector Stage；`K̄`、`Q̄s`、`decayK` 的消费者是 `C1`（两路）、`C3` 和
  `V4`（两路），因此 slot 必须是**多读者 ready + 最后一个读者 free** 的生命周期。

### 7.2 `C1` · `dV_pre = K̄_c @ dH_old` 与 `T1 = W_c^T @ K̄_c`（合并原 `C1` 与 `C5`）

- 两路 MMAD 共用同一次 `K̄_c` 的 L1 装载，所以它们是同一个 Cube Stage 的两个输出，而不是两个 Stage：
  入口只需要 `V0ExportReady` + `dH` 旧值 + `W_c`，两路操作数在 Stage 入口前全部 ready。
- 路 A（E 分支）：`K̄_c`（`[M, K]`）@ `dH_old`（`[K, BS]`）→ `dV_pre`（`[M, BS]`），FP32 累加。
  `K > 64` 时按 64 行分组做 K 方向规约，全部累加进同一 L0C，与上游 `b_dh1..b_dh4` 一一对应。
- 路 B（P 分支）：`W_c^T`（`[K, M]`）@ `K̄_c`（`[M, K]`）→ `T1`（`[K, K]`）。`T1` 与输出列 tile
  无关，每 chunk 只需一次，列 tile 复用；这是上游"每个 program 各算一遍 `b_kw`"在单工作组持有整头
  之后的优化。
- **发布顺序：先路 A、后路 B。** 路 A 的 Fixpipe 完成后立刻发布 `C1DvPreReady`，让
  `V2 → C3 → V4(路 A)` 先跑；路 B 的 `T1` 只需在合并后的 `V4` 入口前就绪，因此不得把 `dV_pre`
  拖到 `T1` 之后才发布。
- `dH_old` 在本 chunk 内被两个读者读取：`C1`（经 L1）与 `V4`（经 UB，做衰减项）。这是一个显式的
  **双读者**合同，不能假设"`C1` 读完就 free"。
- `W_c` 只被路 B 使用；`do_c` 不经过本 Stage。两路读完 `K̄_c` 后一起归还 `V0` 的 slot reader 计数。
- `K = 128` 时 `T1` 为 64 KiB FP32，`K = 256` 时 256 KiB，已经超出单个 L1，因此 `T1` 建议驻留
  workspace/GM 平面，由合并后的 `V4` 按块 MTE2。具体切块与对齐按目标 CANN 版本核对后定稿。
- 尾块：`M < BT` 的行由 `K̄` 的零行保证两路都为 0；`BS` 尾 tile 由 `dH` 的列 mask 保证。

### 7.3 `V2` · `dV̂' = -(dV_pre + dv_local)`

- 一次 VF 覆盖 `[BT, BS]` 的当前列 tile。
- 取负号是为了让 `C3` 只做"两个 MMAD 相加"，避免为 `-W^T dV̂` 再引入取负操作数或第二个 L0C 累加方向。
- 由于衰减已经折进 `K̄`，`USE_G` 与 `USE_GK` 在这一 Stage 上完全一致，没有分支。
- 有效行之外的 `dv_local` 依赖 GM 零 padding（上游 `boundary_check` 的 `other=0` 语义）；如果上游 buffer 的
  padding 不保证为零，必须在本 Stage 或 MTE2 侧显式清零。

### 7.4 `C3` · `inc = Q̄s_c^T @ do_c + W_c^T @ dV̂'`

- 两个 MMAD 累加到同一 L0C，完成后只做一次 Fixpipe（对齐 `03 Finalize` 的 `S/R` 做法）。
- `Q̄s` 参与的是**转置**左操作数（`[K, M]`），与 `FwdH` 的 `stage2`（`kg^T @ vNew`）是同一类加载形态。
- `W_c` 与 `do_c` 都直接从 GM 取（`do_c` 只被本 Stage 消费一次，不需要 `V0` 预搬），`W_c` 在本 chunk 被
  `C3` 与 `C1` 路 B 读取两次，reader 统计要按 2 计。
- 输出 `inc` 为 `[K, BS]` FP32；`K > 64` 时按 64 行分组，`Q̄s/W` 的行也随分组切。

### 7.5 `V4` · `dH_new = decayK_c ⊙ dH_old + inc` 与 `P_c = diag(decayK_c) - T1`（合并原 `V4` 与 `V6`）

- 一次 VF 同时产出两路结果，共用同一份 `decayK`（`[K]`）装载：
  - 路 A（E 分支）：读 `dH_old`（`[K, BS]`）+ `inc`（`[K, BS]`）+ `decayK`，写 `dH_new`；
  - 路 B（P 分支）：读 `T1` 的一个 block + `decayK`，写 `P_c` 的同一个 block。
- 入口同时需要 `C3IncReady`（路 A）与 `C1T1Ready`（路 B）。`T1` 由合并后的 `C1` 在很早的阶段发布，
  因此本 Stage 的实际入口条件仍是 `C3IncReady`。
- `USE_G` 是标量广播，`USE_GK` 是逐 K 行，`无门控` 是恒 1；三者在 Vec 侧只是同一模板的不同
  `calcMode`，两路共用同一份 decay 计算。
- 对角项按全局行列索引 `row == col` 放置：只在 `row` 落在本 block 的列区间内时写入 `decayK[row]`，
  其余位置保持 `-T1`。禁止把 `decayK` 当作整列广播。
- `dH` 是跨 chunk 状态，必须 FP32，且必须**逐 chunk 回写** workspace/GM（Vector→Cube 必须经
  GM/workspace）；不允许把 `dH` 的链保持在 UB 里跨 chunk。`P_c` 是本 chunk 内的一次性载荷，按 block
  写 workspace 后由 `C5` 消费。
- 双缓冲：`dH_new` 与 `dH_old` 使用 ping-pong，`DhtPrevFree` 只有在下一轮 `C1` 与 `V4` 都完成对旧缓冲的
  读取后才能发布。
- **合并代价：** `P_c` 必须等 `inc` 就绪，因此 P 链的进度与该 chunk 的 E 链绑定。由于 E 链（4 段）比
  P 链长，且两条链在下一轮 `C1`/`C5` 之前都必须完成，正常路径下关键路径不变；若出现 P 链成为关键路径的
  形态（例如 `tileV` 极少而 `tileK` 极多），可以把路 B 拆回独立的 `V6`，即回退到八 Stage 版本。
- 末 chunk 结束时把 `dH` 写 `dhm[..., :V]`；本 Kernel 内 `dhm` 不做 `StoreRounded`，直接按 FP32 写。

### 7.6 `C5` · `P_new = P_c @ P_old`（原 `C7`）

- 输出列 tile 为 `[K, BS]`，K 方向规约需要遍历全部 `P_c` 的行；`K > 64` 时按 K 归约分块，多个 MMAD
  累加进同一 L0C。这属于"Cube 展开硬件 tile"，不改变本 Stage 的边界。
- `P_old` 是跨 chunk 状态，必须 FP32，ping-pong 双缓冲；`PPrevFree` 在下一轮 `C5` 完成读取后发布。
- 末 chunk 结束时把 `P` 写 `dhm[..., V:]`。
- `C5` 与 `C1`/`C3` 共用 Cube 流水，因此 `C5` 的 MMAD 序列与 E 分支的 MMAD 序列按 Cube pipe 串行
  排布；两者之间只有 `V0ExportReady`/`V4PcReady` 这些数据边，没有额外的执行顺序要求。

## 8. 为什么合并后是 6 个 Stage

两组可以合并的 Stage 都满足"同一计算引擎 + 入口操作数在同一时刻 ready"，合并只共享入口和操作数装载，
不改变任何数学语义：

1. **原 `C1` + 原 `C5` 合并（都是 Cube，都吃 `K̄`）。** 两路的入口操作数 `K̄_c`、`dH_old`、`W_c` 都在 `V0` 之后
   同一时刻 ready；两路只是两个不同的 MMAD，共用同一次 `K̄_c` 的 L1 装载。合并后 `dV_pre` 与 `T1`
   分别 Fixpipe 发布，先发 `dV_pre`，所以 `V2` 不会被 `T1` 阻塞。
2. **原 `V4` + 原 `V6` 合并（都是 Vector，都吃 `decayK`）。** 路 B 的输入 `T1` 由合并后的 `C1` 很早就发布，
   合并后本 Stage 的实际入口条件仍然是 `C3IncReady`；`decayK` 一次装载同时服务两路输出。
3. **不能继续合并的部分：**
   - `V0` 不能并入 Cube Stage：它是唯一做门控/指数/截断/掩码的 Stage，两个分支都依赖它的输出，
     数值规则（截断区间、掩码零行）集中在这里才能一处保证；
   - `C1` 与 `C3` 之间必须有 `V2`：`C3` 的右操作数是 `V2` 的乘积结果，Cube 不消费同 Stage 新产生的数据；
   - `C3` 与下一轮 `C1` 之间必须有 `V4` 路 A：`dH` 的衰减+累加是 Vector 语义（逐 K 行），且必须回写
     workspace 才能被下一轮 `C1` 当 L1 操作数读取；
   - `V4` 路 B 与 `C5` 之间必须有对角注入：`diag(decayK)` 是单位阵形式的注入，必须在链乘之前落入
     `P_c`，留到 `C5` 的 L0C 初始化就无法按 `row == col` 的位置语义注入。
4. **6 = 1 共享 + 1 合并 Cube + 1 + 1 + 1 合并 Vector + 1**，即 `V0 → C1 → V2 → C3 → V4 → C5`，是在
   "每个 Stage 只含一种计算引擎、入口操作数必须全部 ready"约束下的最小划分。
5. **合并收益：** Stage 数由 8 降到 6，每 chunk 少两次 Stage 入口 handshake；`K̄` 每 chunk 只做一次
   L1 装载（原 `C1`/`C5` 各一次），`decayK` 每 chunk 只做一次 UB 装载（原 `V4`/`V6` 各一次）。
6. **合并代价：** P 链的 `P_c` 与本 chunk 的 E 链绑定（见 §7.5）。E 链更长时关键路径不变；如果实测
   `tileK` 远多于 `tileV` 导致 P 链成为关键路径，回退到八 Stage 版本即可解耦。

## 9. ready/free 依赖 DAG

数据边（data-ready）：

```text
V0 ──V0ExportReady──▶ C1, C3, V4
C1 ──C1DvPreReady──▶ V2 ──V2DvHatReady──▶ C3 ──C3IncReady──▶ V4 ──V4DhtReady──▶ 下一轮 C1
C1 ──C1T1Ready─────▶ V4                                     V4 ──V4PcReady──▶ C5 ──C5PReady──▶ 下一轮 C5
```

跨 chunk 的反向边（必须显式建模，不能依赖 core 启动/完成顺序推断）：

```text
V4(i_t) ──▶ C1(i_t - 1)      dH 链
C5(i_t) ──▶ C5(i_t - 1)      P 链
```

存储边（storage-free，与 data-ready 分开）：

```text
V2 读完后发布 C1DvPrePayloadFree    释放 dV_pre
V4 读完后发布 C3PayloadFree          释放 inc
C5 读完后发布 V4PcPayloadFree        释放 P_c
本轮 C1（两路）、C3、V4 全部读完后发布 V0SlotFree   释放 chunk slot（多读者计数）
下一轮 C1、V4 都读完后发布 DhtPrevFree          释放 dH 旧缓冲
下一轮 C5 读完后发布 PPrevFree                   释放 P 旧缓冲
```

- data-ready 不能替代 storage-free。上游 profiling 里偶然先完成的边不能被省略或合并。
- 合并 Stage 的两路必须各有独立的 ready/free：`C1` 的 `dV_pre` 与 `T1` 不能共用一个 ready，`V4` 的
  `dH` 与 `P_c` 也不能共用一个 ready，否则先完成的一路会被另一路拖住。
- `V0` 的 slot 是多读者资源，必须用"最后一个读者归还"的计数语义，不能让某个读者自行 free。
- 反向链的 free 边（`DhtPrevFree`/`PPrevFree`）必须由**生产者+消费者**共同确认，否则会出现
  生产者覆盖消费者仍在读取的 `dH`/`P`。
- 如果按第 5 节第 3 条把 `tileV` 与 `tileK` 拆到不同工作组，`V0` 的输出就变成跨核共享资源，需要按
  `01 Prepare` 的 `QkCacheReady`/`QkCacheFree` 模式补一套具名 owner 协议；同核内则只需本地 slot
  计数。此时合并 Stage 只执行属于本 task 的那一路（见 §5 第 4 条）。

## 10. 驻留与资源账本（推导值）

以下数字由 shape 直接推出，用于判断哪些平面必须离开 UB/L1；**具体放在哪一级存储、用什么地址方案，
属于需要按目标 CANN 版本核对后再定稿的 PROPOSED 项**。

取 `BT = 64`、`BS = 64`、`K = V = 128`、输入 BF16：

| 平面 | 逻辑 Shape | 单个大小 | 生产者 → 消费者 |
| --- | --- | --- | --- |
| `Q̄s` | `[64,128]` BF16 | 16 KiB | `V0` → `C3` |
| `K̄` | `[64,128]` BF16 | 16 KiB | `V0` → `C1`（路 A + 路 B，共用一次装载） |
| `decayK` | `[128]` FP32 | 0.5 KiB | `V0` → `V4`（两路共用一次装载） |
| chunk slot 小计 | — | ≈ 32.5 KiB | 每 chunk 一份，按流水深度做多份 |
| `dV_pre` | `[64,64]` FP32 | 16 KiB | `C1` → `V2` |
| `dV̂'` | `[64,64]` FP32 | 16 KiB | `V2` → `C3` |
| `inc` | `[128,64]` FP32 | 32 KiB | `C3` → `V4` |
| `dH` tile | `[128,64]` FP32 | 32 KiB（ping-pong ×2 = 64 KiB） | `V4` → `C1`/`V4` |
| `T1` 平面 | `[128,128]` FP32 | 64 KiB | `C1` 路 B → `V4` 路 B |
| `P_c` 平面 | `[128,128]` FP32 | 64 KiB | `V4` 路 B → `C5` |
| `P` tile | `[128,64]` FP32 | 32 KiB（ping-pong ×2 = 64 KiB） | `C5` → `C5` |

结论：

- `K` 方向越长，`T1`/`P_c` 增长最快：`K = 256` 时各为 256 KiB，**必须**驻留 workspace/GM 并按块搬运，
  不能假设能整体进 L1；这也是 `C5` 按 K 归约分块的根本原因。
- `dH`/`P` 的 ping-pong 双份加上 `T1`/`P_c` 平面后，单工作组的常驻需求已经超过单核 UB 容量，因此必须
  按"UB 只放 Vector 平面、L1 只放 Cube 当前 tile、跨 Stage payload 走 workspace"来分工，不能把
  `Q̄s/K̄/W` 长期留在 UB 里当全局缓冲。
- `W`/`do`/`dv_local` 直接从 GM 取（各被消费 1~2 次），不进入 chunk slot，避免 slot 随 K 线性膨胀。
- 合并 Stage 只减少装载次数，不改变平面大小：`K̄`（16 KiB/chunk）由 `C1` 两路共用，`decayK`
  （0.5 KiB/chunk）由 `V4` 两路共用，`T1`/`P_c` 的大小与拆分版本相同。

## 11. 模式矩阵（本次必须覆盖）

| gate 模式 | `V0` 的输入 | `K̄` 构造 | `decayK` | 额外分支 |
| --- | --- | --- | --- | --- |
| `USE_G` | `q,k,g` | `k ⊙ 2^{g_last - g}` | 标量 `2^{g_last}` 沿 K 广播 | 无（衰减已折进 `K̄`） |
| `USE_GK` | `qg,kg,gk` | `kg` | 逐 K `2^{gk_last}` | 无 |
| 无门控 | `q,k` | `k` | 恒 1 | 无 |

其余必须覆盖的组合：

| 维度 | 取值 |
| --- | --- |
| `K` | 64（`BS=32`）、128（`BS=64`）、256（4 个 64 行组） |
| `V` | 能被 `BS` 整除；不能被整除的尾 tile |
| chunk | `M = BT` 满块；`M < BT` 尾块；`NT = 1` 与 `NT > 1` |
| head | `H_qk == HV`；`HV > H_qk`（GVA） |
| 序列 | dense；varlen；单 sequence 跨一个 segment；空/零长尾 segment |
| dtype | FP16 输入；BF16 输入；`dhm`/链/accumulator 全 FP32 |
| 状态布局 | `state_v_first = false` 与 `true`（只影响 `dht`/`initial_state`，不影响 `dhm`） |

## 12. 验证与门禁

1. **数值基线**：先用纯 PyTorch/NumPy 按 §3 公式逐 chunk 复算 `E_r`、`P_r`，再用非零 `dht` 校验
   `P_r @ dht + E_r` 与整段 backward 的 `dh0` 一致。只测 `dht = 0` 无法验证 `P`。
2. **Stage 边界**：在每个 Stage 出口抽 `dV_pre`、`dV̂'`、`inc`、`T1`、`P_c`、`dH`、`P`，与参考实现逐块比对。
   出现偏差时先判断是"某一 Stage 的公式错误"还是"ready/free 边缺失导致的旧值/新值混用"。
3. **链方向**：单独验证 `P_new = P_c @ P_old` 与 `dH_new = decayK ⊙ dH_old + inc` 的 chunk 顺序；把 `NT`
   取到 3 以上，避免 `NT = 1` 时方向错误被掩盖。
4. **尾块**：`M < BT`、K/V 尾 tile 必须走与满块相同的 Stage 路径，只靠 mask 生效；padding 不允许写回 GM。
5. **多读者/多写者**：构造 `K̄` 双读者（`C1` 两路）、`W` 双读者（`C1` 路 B 与 `C3`）、`dH` 双读者
   （`C1` 与 `V4`）同时出现的用例，确认 free 计数与 ping-pong 没有提前复用。
6. **合并的两路独立性**：构造"路 A 已就绪、路 B 尚未就绪"和反过来的用例，确认 `C1` 的
   `dV_pre`/`T1`、`V4` 的 `dH`/`P_c` 各自独立发布与释放，没有并成一条 ready 或提前 free。
7. **分核**：覆盖 `HV >= N`（仅 head 分核）与 `HV < N`（`(hv, 列 tile)` 展平）两条路径，验证列 tile 独立性
   结论；展平路径下每个 task 只执行合并 Stage 中属于自己分支的那一路，结果必须与单 task 全量执行逐位一致
   （同一 dtype 下允许的差异仅来自规约顺序）。
8. **内存检查**：按仓库规范对 UB/L1 复用、跨核 slot、ping-pong 做 sanitizer 验证，并确认运行时命中的是
   sanitizer 版本对象。
9. **跨 rank 一致性**：最终仍要以"同一全局输入、同一 loss，只改 CP 切分"的方式对比 `dq/dk/dv/dg`（或
   `dgk/dbeta`）与单 rank 基线；缩小 shape 只用于定位，不作为通过依据。

## 13. 待确认项（PROPOSED）

- `V0` 输出的 slot 布局、对齐与多读者 ready/free 的具体 flag/event 方案；
- `Arch35` 下 `C1` 两路与 `C3`/`C5` 的 Fixpipe→配对 AIV UB 直通路径，以及 `Arch22` 的 GM relay 方案；
- `T1`/`P_c` 在 `K = 128/256` 时的分块粒度、驻留层级与 `C5` 的 K 归约 MMAD 切分；
- `USE_G` 与 `USE_GK` 在 `V4` 两路中的模板分发方式（`calcMode`/`tilingKey`）以及 host 侧互斥拦截位置；
- 合并 Stage 内部的 MMAD/VF 排布（`C1` 路 A→路 B 的先后、`V4` 两路的 UB 峰值）是否需要按 UB 预算
  再切子阶段，以及是否保留"P 链解耦时回退八 Stage"的编译期开关；
- `scale` 折进 `Q̄` 与调用方 `qg` 是否已含 `scale` 的接口约定；
- 列 tile 展平分核时 `V0` 输出的跨核 owner 协议（是否需要 `QkCacheReady`/`QkCacheFree` 同款机制）。

以上均需在编码前按目标 CANN 版本核对官方文档、随包头文件和实现源码，并做最小设备编译与精度验证。
