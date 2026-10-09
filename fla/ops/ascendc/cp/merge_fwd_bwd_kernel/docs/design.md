# MergeFwdBwdKernel 设计

公开接口、shape、属性和支持范围以 [api.md](api.md) 为准。本文只记录计算怎么拆、数据放在哪、A2/A3 与 A5 的流水差在哪里。

A3 是 `ascend910_93` 和 A2（`ascend910b`）走 `__CCE_AICORE__ != 310` 的同一套 Cube/Vector。A5（`ascend950`）走 `arch35` Cube，并把 GEMM 结果直接 Fixpipe 进 Vector UB。

## 1. 目标

一次调用把若干 rank 上的 `(M, He)` 收成调用方的一块状态 `h`。`h` 是 inplace：OpDef 里输入和输出同名，kernel 只写输出地址，两处地址是同一块用户内存。计算不读 `h` 的旧值。

```text
h ← He_0
h ← M_i @ h + He_i    i = 1 .. N-1
```

`He` 是 `[128, 128]`，`M` 是 `[128, 128]`。`ag_hm` 每个 head 的物理行是 `[128, 256]`：前 128 列 `He`，后 128 列 `M`。内部累加器是 FP32。`state_v_first=true` 只改变最后写给用户的物理顺序，等价于

```text
(M @ h + He)^T = h^T @ M^T + He^T
```

源 rank：

```text
forward:  src(i) = rank - N + i
backward: src(i) = rank + N - i
S = 1 且 N = 1 且 rank = 0: src = 0
```

`N = 1` 没有矩阵乘，结果就是 `He_0`。调用链是一次 Mix kernel：`KERNEL_TYPE_MIX_AIC_1_2`，Vector 打包和加 `He`，Cube 做 GEMM。没有第二轮 kernel，也没有芯片级 `SyncAll`。

本算子没有单独约定的性能达标线。测量用的是 CP 最后一档合并（`N = cp - 1`，`rank = cp - 1`，FP32），统计方式是预热后对单次调用做 `synchronize` 再取中位数。这些数只说明量级，不作为设计门禁。

## 2. Stage 详设

### 符号与任务

| 符号 | 含义 |
| --- | --- |
| `S` | `ag_hm` 的 rank 维 |
| `HV` | value head 数 |
| `N` | `preOrPostNumRanks`，链上参与合并的段数 |
| `K`, `V` | 固定 128 |
| `h` | head 下标，`0 .. HV-1` |
| `i` | 链上的步，`0 .. N-1`。`i = 0` 只产生初值 `He_0` |
| `CG` | 实现按 head 奇偶把一个 Mix 核的两个 AIV 拆开，每个 AIV 独占若干完整 head，不是把一个 head 再切成上下半片。文档里的 `CG` 记为 1：一个逻辑任务是一个 head |
| owner | `(h & 1) == subblock` 的那个 AIV |

`ag_hm` 中 head `h`、源 rank `src` 的元素起点：

```text
(src * HV + h) * 128 * 256
```

Head 分到 Mix 核：`usedAic` 个核按 `CoreTaskRange` 切 `[0, HV)`。`usedAic` 先取 `min(aicNum, HV)`，再至少提到 `min(aicNum, ceil(HV / 4))`，让每个 Mix 核大约不超过 4 个 head。`blockDim = CalcTschBlockDim(usedAic * 2, aicNum, aivNum)`，不足 `usedAic` 时抬到 `usedAic`。`scheduleMode = 1`。

代入 `HV = 32`、`aicNum = 20`（核数以运行时平台查询为准；20 只用来看分核）：`ceil(32/4) = 8`，`usedAic = min(20, 32) = 20`。20 个核分 32 个 head，12 个核 2 个 head，8 个核 1 个 head。`Nwork = HV`（每个 head 一条依赖链，不能把链拆到不同核）。wave 数为 1。没有跨核 tail head：区间按整除余数摊开，每个 head 只属于一个 Mix 核。

空区间的核不发 flag。A3 上一个核的区间非空时，两个 AIV 都要参加该区间里每一个 head 的 flag，即使它不是 owner。

### Stage 结果

| Stage | 执行单元 | 功能 | 主要空间 |
| --- | --- | --- | --- |
| S0 | Vector | `N = 1`：读 `He_0` 并写用户 `h`。`N > 1`：把 `He_0` 和 `M_1` 写入 scratch | UB `fp32Buf` / `hmmBuf`；GM scratch bank 0 和 M bank 0 |
| S1 | Cube | `C = M @ H`，上下各 64 行 | L1 的 H 与两段 M；L0A/L0B/L0C；结果在 GM scratch（A3）或 UB（A5） |
| S2 | Vector | `H ← C + He`。还有下一步就写回 scratch 并打包下一个 `M`；最后一步写用户 `h` | UB `fp32Buf`、`hmmBuf`；用户 `h` |

`N > 1` 时 S1、S2 对 `i = 1 .. N-1` 交替进行。S0 只跑一次。

### Workspace

不使用算子输入以外的 GM 张量。用户 workspace 起点是 FP32 scratch `[3, HV, 128, 128]`：

```text
bytes = 3 * HV * 128 * 128 * 4
```

| bank | 偏移（FP32 元素） | 内容 |
| --- | --- | --- |
| 0 | `h * 16384` | 当前 `H`，`[128, 128]`，行主序 `[K, V]` |
| 1 或 2 | `(1 + bank) * HV * 16384 + h * 16384` | `M`。step `i`（从 1 计）用 bank `(i - 1) & 1` |

`H` 和 `M` 的 L2 cache hint 都关掉。A3 的 scratch 基址是 `GetUserWorkspace()`；失败时用 `workspace + sysWorkspaceSize`。A5 直接用 kernel 入参 `workspace`。`N = 1` 不写 scratch。

`HV = 32` 时 scratch 为 `3 * 32 * 64 KiB = 6 MiB`，再加平台系统 workspace。

### Stage S0：Vector，装入链的起点

`N = 1`：owner 从 `src(0)` 把 `He` 的 128×128 按行搬入 `fp32Buf`。BF16 在 UB 里 cast 成 FP32。然后 `StoreUserH`。Cube 不启动。非 owner 不参与，也没有 flag。

`N > 1`：owner 做两件事。

1. `He_0` 写入 scratch bank 0。BF16 先 cast。
2. `src(1)` 的后 128 列是 `M_1`，写入 M bank 0。`M` 是 128×128，不是 128×256。

A5 只让 owner `Set` VC。A3 上 owner 搬完之后，两个 AIV 都 `Set` VC（flag 4），Cube 才会开始读 GM。

UB 在这条路径上的绝对区间（两个 SoC 相同，`InitBuffer` 顺序固定）：

```text
UB[0, 65536)       fp32Buf；FP32 [128, 128] 行主序；64 KiB；He 或累加结果
UB[65536, 131072)  hmmBuf；FP32 [128, 128]；64 KiB；本步 He，或 state_v_first 的转置结果
UB[131072, 163840) bf16Buf；BF16 [128, 128]；32 KiB；仅输入为 BF16 时的搬运/写出暂存
UB[163840, UB容量) 空闲
```

A2/A3 每个 AIV 的 UB 容量按 192 KiB 计，峰值 160 KiB，连续空闲约 32 KiB。A5 UB 更大，同一布局仍放得下。`fp32Buf` 与 `hmmBuf` 同时活着：一个是 `H`/`C`，一个是 `He`。最后一步若要转置，`He` 已经加完，`hmmBuf` 改作转置目的地。

`ag_hm` 每个 head 在 S0 最多读两次：一次 `He_0`（128×128），一次 `M_1`（128×128）。FP32 时各 64 KiB，BF16 时各 32 KiB。写 scratch：`H` 64 KiB FP32，`M` 64 KiB FP32。

### Stage S1：Cube，`C = M @ H`

只在 `N > 1` 时执行。对每个 head、每个 `step = 1 .. N-1`：

```text
C = M @ H
```

旁注：`M` 与 `H` 都是 FP32、行主序、不转置。`M` 分成上下两块 `[64, 128]`，`H` 是 `[128, 128]`。每个 64 行块再沿 K 切成两段 64 做 MMAD，FP32 累加到 L0C。`initC` 在每个 K 段的第一段为真。

A3 的 L1 是单缓冲，绝对字节区间：

```text
L1[0, 65536)         H；FP32 [128, 128] 行主序；64 KiB；本步 MTE2 写入，两段 MMAD 读完后释放
L1[131072, 163840)   M 上半；FP32 [64, 128]；32 KiB；对应 C 的前 64 行
L1[196608, 229376)   M 下半；FP32 [64, 128]；32 KiB；对应 C 的后 64 行
L1[65536, 131072)    空闲；无数据
L1[163840, 196608)   空闲；无数据
```

已用最高地址 229376，A2/A3 L1 按 512 KiB 计，后面约 282 KiB 连续空闲。`H` 与两段 `M` 同时在 L1，直到下半 MMAD 结束才允许下一轮 MTE2 覆盖（`MTE1_MTE2`）。

L0：一段 `L0A` 是 `[64, 64]` FP32（16 KiB），`L0B` 是 `[64, 128]` FP32（32 KiB）。A3 的两块 L0C 各 64 KiB（`kL0CTileBytes`），放一个 `[64, 128]` FP32 累加结果。K 向两段在同一块 L0C 上累加。

A3 写出：`FixpipeParamsV220`，`CFG_ROW_MAJOR`，无量化。上半 C 写到 scratch bank 0 的前 `64 * 128` 个 FP32，下半写后一半。写完上半 `Set` CV0（flag 2，`PIPE_FIX`），写完下半 `Set` CV1（flag 3）。

A5 不写这半块到 scratch。它在 Fixpipe 前 `Wait` `CUbFree`，用 `FixpipeParamsC310` 把半块 C 按行主序送进对应 AIV 的 `fp32Buf`。`subBlockId = h & 1`。

Cube 不读、不写用户 `h`。

每个 head、每个 step 从 scratch 读 `H` 64 KiB、读 `M` 64 KiB，各一次。A3 再把 C 64 KiB 写回 scratch bank 0，覆盖旧 `H`。A5 的 C 留在 UB，scratch 上的 `H` 要等 S2 写回。

### Stage S2：Vector，`H ← C + He`

公式：

```text
H = C + He
```

`C` 与 `He` 都是 FP32 `[128, 128]`，按元素加。`He` 来自 `src(step)` 的前 128 列。BF16 输入在进 UB 时已经 cast，加法不再 cast。

A5：CV0/CV1 的 wait 在 `PIPE_V` 上，等到时 C 已经在 `fp32Buf` 的对应 64 行里，直接加。

A3：同样在 `PIPE_V` 上 wait。wait 返回后用 `V_MTE2` 把 scratch 里刚写好的 64×128 搬进 `fp32Buf`，再加。不用 `PIPE_ALL`，否则会去等同一个 Mix 块里仍在算另一半的 Cube。

还有下一步时，owner 把完整 `H` 写回 scratch bank 0（64 KiB FP32），并把 `M_{step+1}` 写入另一 M bank（64 KiB FP32）。最后一步调用 `StoreUserH`：

- `stateVFirst = 0`：按 `[K, V]` 写出。FP32 直接 `DataCopy` 64 KiB。BF16 先 `CAST_RINT` 再写 32 KiB。
- `stateVFirst = 1`：`TransDataTo5HD` 把 FP32 `[K, V]` 转到 `hmmBuf` 的 `[V, K]`，再按上面的 dtype 规则写出。寻址与 `recurrent_kda` 的 K-first 到 V-first 相同，固定 128×128。

用户 `h` 每个 head 只写一次。L2 cache hint 关闭。

### 数据搬运汇总

下列字节数是一个 head、FP32、`N > 1` 的量。BF16 的 `ag_hm` 读和最终 `h` 写减半，scratch 仍是 FP32。

| 数据 | 每个 head 的 GM 访问 | 驻留 | L2 |
| --- | --- | --- | --- |
| `He_0` | 读 1 次，64 KiB，来自 `ag_hm` | 写入 scratch bank 0 | 关闭 |
| `M_i`，`i = 1 .. N-1` | 读 `N-1` 次，每次 64 KiB | 写入 M bank，Cube 再读 1 次 | 关闭 |
| `He_i`，`i = 1 .. N-1` | 读 `N-1` 次，每次 64 KiB | 只进 UB `hmmBuf`，不进 scratch | 关闭 |
| scratch `H` | A3：Cube 每步读 64 KiB、把 C 写回 64 KiB；Vector 再读 C 64 KiB，非末步写新 `H` 64 KiB。A5：Vector 写 `H`，Cube 只读，C 不经 scratch | bank 0 | 关闭 |
| 用户 `h` | 写 1 次。FP32 64 KiB，BF16 32 KiB | 无 | 关闭 |

`HV = 32`、`N = 8`、FP32、A3：`ag_hm` 读 `(1 + 2 * 7) * 32 * 64 KiB = 30 MiB`（`He_0`、7 次 `M`、7 次 `He`）。用户 `h` 写 `32 * 64 KiB = 2 MiB`。scratch 容量 6 MiB，每步被覆盖，不是按步累加。

搬运都是连续 `DataCopy` 或按 128 行、行长 128、源行距 256 的 `DataCopyPad`。没有把同一输入再搬进 L1 和 UB 各一份：Vector 先把 `M` 和 `H` 收成紧凑 FP32，Cube 只读 scratch。

### 同步

Flag 的 mode 放在 `kCrossCoreModeAiv`。A5 是 `0x4`，AIV1 的 id 为 base+16。A3 是 `0x2`：一个 AIC 对两个 AIV 的集合同步，id 不按 head 奇偶偏移。两边都用这三面 flag：

| flag | id | 方向 | 初始 | 含义 |
| --- | --- | --- | --- | --- |
| VC | 4 | AIV `PIPE_MTE3` → AIC `PIPE_MTE2` | 未置位 | `H` 和当前 `M` 已在 scratch |
| CV0 | 2 | AIC `PIPE_FIX` → AIV `PIPE_V` | 未置位 | 上 64 行 C 可见 |
| CV1 | 3 | AIC `PIPE_FIX` → AIV `PIPE_V` | 未置位 | 下 64 行 C 可见 |
| CUbFree | 5 | 仅 A5。AIV `PIPE_V` → AIC `PIPE_FIX` | prologue 置一次 | `fp32Buf` 已腾出，Cube 可以 Fixpipe |

A3 不使用 `CUbFree`。mode `0x2` 要求两个 AIV 对同一 id 都 set、都 wait。非 owner 不搬数，但每个 head 都要走同一面 flag，否则 AIC 的 wait 不返回。owner 的 VC set 排在自己的 MTE3 之后，AIC 要等两个 set 都完成，因此不会在 `H`/`M` 写完前开读。

`N = 1` 没有 flag。`N > 1` 的一个 head、一步如下。A3 的 “双方” 指两个 AIV；A5 只有 owner 执行 set/wait。

```text
S0:
  owner: 写 H、写 M_1
  双方 Set VC

每个 step = 1 .. N-1:
  AIC Wait VC
  AIC: L1 装入 H 与 M，MMAD 上半，写出上半 C，Set CV0
  AIC: MMAD 下半，写出下半 C，Set CV1
  双方 Wait CV0
  owner: 取上半 C，加 He 的上半
  双方 Wait CV1
  owner: 取下半 C，加 He 的下半
  owner: 末步则 StoreUserH，否则写回 H 与下一步 M
  非末步: 双方 Set VC
  A5: owner Set CUbFree，供下一步 Fixpipe
```

A3 取 C 是 `DataCopy` scratch。A5 取 C 是 UB 里已经 Fixpipe 好的半块，没有这次 GM 读。

无效 head 不在 `[hBegin, hEnd)` 里，对应 AIC/AIV 都不碰它的 flag。一个核的区间为空时，该核全程不 set、不 wait。
