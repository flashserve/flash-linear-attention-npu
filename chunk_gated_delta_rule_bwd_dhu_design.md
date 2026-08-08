# chunk_gated_delta_rule_bwd_dhu 算子设计方案

## 1. 背景与目标

`chunk_gated_delta_rule_bwd_dhu` 负责 Gated Delta Rule 反向链路中 `dh/dh0/dv2` 的递推计算。功能参考上游 `fla/ops/common/chunk_delta_h.py` 中的 `chunk_gated_delta_rule_bwd_dhu` 和 `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64`。

与 `prepare_wy_repr_bwd` 不同，本算子的核心状态 `dH` 在同一个序列的相邻 chunk 之间存在倒序依赖：

```text
dH_in(chunk i) = dH_out(chunk i + 1)
```

当前已落地的主流程按 `prepare_wy_repr_bwd` 风格保持统一调度 helper，但任务粒度不是单 chunk，而是 `seq + 4-head window`。kernel 主流程不拆定长/变长两套分支，通过统一的 `GetSeqInfo(cu_seqlens, seqIdx)` 和 `GetChunkInfoBySeqChunk(chunk_indices, seqInfo, localChunkIdx)` 计算当前序列的 chunk 数、token 起点、实际长度和 `dh` 输出 chunk 下标。每个 task 固定覆盖 1 个序列和最多 4 个 value head，分核粒度为：

```text
fixed length:  B * ceil_div(HV, 4)
varlen:        seqNum * ceil_div(HV, 4)
```

每个 core 内部负责一个或多个 `(seqIdx, hvBase)` task，其中 `hvBase` 取 `0, 4, 8, ...`，当前 round 的 `headCnt = min(4, HV - hvBase)`。A2/A5 路径中 task 内两个 Vector subblock 按 `headOffset % subBlockNum` 交替承包完整 head：subblock0 处理 `headOffset=0/2`，subblock1 处理 `headOffset=1/3`；两个 subblock 仍共同参与每个 head 的 raw flag 同步配平。随后按当前 seq 的 chunk 倒序串行执行所有阶段。定长和变长的差异只在 chunk offset helper 内处理：定长按 `seqIdx/localChunkIdx` 计算 batch 内 token 区间；变长先按 `cu_seqlens` 计算当前 seq 的 packed token 区间、chunk 数和 canonical flattened chunk 前缀，chunk 内用 `outputChunkBase + localChunkIdx` 作为 `dh` 输出下标，并校验 `chunk_indices` 对应 pair；若输入不是 canonical 顺序，再 fallback 线性反查。

阶段性目标：

1. stage0 已调通，固定/变长使用同一套 kernel 主流程。
2. 当前三阶段首版已经接入 Vector/Cube 跨核 ready 边界、workspace 段布局、Cube 侧 `dvState/termQ/termW` 三个 GEMM，以及 Vector 侧 `dv2` 写回和 `dhState` 倒序更新。
3. Cube 侧直接对齐 `prepare_wy_repr_bwd` 的 tile 级 resident/双缓冲写法：使用 `GM->L1->L0A/L0B->TileMmadTla->L0C->Dst`，其中 Dst 按路径选择 GM 或 AIV UB。
4. 后续仍需继续对齐 `dht/h0/dh0` 初始/输出语义、`state_v_first` 布局，以及必要的 AIC/AIV UB 直连性能优化。

说明：完整递推版本的 `dH` 在相邻 chunk 间存在倒序依赖。当前代码已经让每个 `(seq,hv)` 的 `dhState` 由负责该 head 的 Vector subblock 以 row tile 形式在 workspace state 段和 UB 间跨 chunk 倒序流转，chunk 主循环按 seq 内倒序执行；定长和变长差异仍只在 chunk offset helper 内处理，不在 kernel 主流程拆两套分支。

## 2. 接口与形状

### 2.1 输入

| 名称 | Shape | dtype | 说明 |
|---|---|---|---|
| `q` | `[B, HK, T, K]` | fp16/bf16 | Query |
| `k` | `[B, HK, T, K]` | fp16/bf16 | Key |
| `w` | `[B, HV, T, K]` | fp16/bf16 | Delta rule 中的 w |
| `d_o` | `[B, HV, T, V]` | fp16/bf16 | 输出梯度 |
| `dv` | `[B, HV, T, V]` | fp16/bf16 | 输入的 value 分支梯度，会与 state 贡献相加得到 `dv2` |
| `g` | `[B, HV, T]` | fp16/bf16/fp32 | token-wise gate，对应上游 `[B, T, HV]`；与 `gK` 二选一；影响 `dv2` token gate、`dH` 的统一 decay 和 `q` 项 |
| `gK` | `[B, HV, T, K]` | fp16/bf16/fp32 | key-wise gate，对应上游 `gk=[B, T, HV, K]`，不是 `[B, HV, T]`；与 `g` 二选一；只按 K 行 decay `dH` |
| `h0` | optional | fp16/bf16/fp32 | 仅用于表示是否需要输出 `dh0`，反向计算本身不读取 `h0` |
| `dht` | `[N, HV, K, V]` | fp16/bf16/fp32 | final state 梯度，可选；为空时初始 `dH=0` |
| `cu_seqlens` | `[seqNum + 1]` | int64 | 变长模式累计长度 |
| `chunk_indices` | `[totalChunkNum * 2]` | int64 | 变长模式 chunk 索引，flatten 后为 `[seqIdx, chunkIdx, ...]` |

属性：

| 名称 | 说明 |
|---|---|
| `scale` | `q^T @ d_o` 分支缩放系数 |
| `chunk_size` | `BT`，支持 `64/128` |

约束：

```text
K = 128
V = 128 or 256
BT = 64 or 128
HV % HK == 0
varlen 模式下 B = 1
第一版 state_v_first = false
g 和 gK 必须刚好传入一个，按参数是否传入选择分支
g 或 gK 使用低精度类型时必须与 q/k 同为 fp16 或同为 bf16；也支持 fp32 gate
```

说明：上游 Triton 实现支持 `K <= 256`、`state_v_first` 等更宽配置。本仓当前相邻算子和测试主要围绕 `K=128`、head-major 布局展开，第一版先收敛到该范围。

### 2.2 输出

| 名称 | Shape | dtype | 说明 |
|---|---|---|---|
| `dh` | fixed: `[B, HV, chunkNumPerB, K, V]`; varlen: `[1, HV, totalChunkNum, K, V]` | fp16/bf16 | 每个 chunk 开始处的 `dH` |
| `dh0` | `[N, HV, K, V]` | fp32 | 如果 `h0` 非空则输出最终递推到序列开头的 `dH`；否则返回空 tensor |
| `dv2` | `[B, HV, T, V]` | fp16/bf16 | `dv + K @ dH` 加 gate 后的结果 |

注意：当前 fast-kernel launch 占位 meta 如果把 `dh0` 做成 `[B, HV, chunkNum, K, V]` 且 dtype 跟随 `q`，建议实现时修正为上表语义，和上游功能保持一致。

## 3. 数学语义

上游 `fla/ops/common/chunk_delta_h.py` 的 common kernel 使用 token-major 形状：

```text
q/k      [B, T, H, K]
w/do/dv  [B, T, HV, K or V]
g        [B, T, HV]
gk       [B, T, HV, K]
dh       [B, NT, HV, K, V]     // state_v_first=false
```

本仓接口只做 shape 置换，数学语义不变：

```text
q/k      [B, HK, T, K]
w/do/dv  [B, HV, T, K or V]
g        [B, HV, T]
gK       [B, HV, T, K]
dh       [B, HV, NT, K, V]
```

注意：上游 `gk` 带 K 维，是 key-wise gate；本仓如果传 `[B,HV,T]` 的逐 token gate，应走 `g`，不能作为 `gK` 的上游对齐语义。

下面公式以本仓 head-major shape 书写。`g` 与 `gK` 使用不同的指数函数：

```text
E_g(x)  = exp(x)
E_gK(x) = exp2(x)
```

当前代码对齐状态：`g` 分支直接调用 `AscendC::Exp` 实现自然指数，CPU golden 使用 `torch.exp`；`gK` 分支通过 `x * ln2` 后调用 `AscendC::Exp` 实现 `exp2(x)`，CPU golden 使用 `torch.exp(x * ln2)`。两个 optional 输入的公式、实现和 golden 必须分别保持一致。

对单个 `(seq, hv)`，令：

```text
hk = hv / (HV / HK)
BT = chunk_size
M  = 当前 chunk 实际长度，M <= BT
BV = V 方向核内 tile，建议第一版 BV=64
```

当前 V tile 的递推状态：

```text
dH      [K, BV]    fp32
K_blk   [M, K]
Q_blk   [M, K]
W_blk   [M, K]
DO_blk  [M, BV]
DV_blk  [M, BV]
g_blk   [M]        // optional token-wise gate
gK_blk  [M, K]     // optional key-wise gate
```

`g` 和 `gK` 是互斥 optional 输入：必须刚好传入一个。传 `g` 时走 token-wise gate 分支；传 `gK` 时走 key-wise gate 分支。二者都传或二者都不传均视为非法入参。

从最后一个 chunk 向第一个 chunk 递推。进入每个 chunk 时先保存当前状态：

```text
dH_old = dH
dh[seq, hv, chunk] = dH_old
```

计算 `dv2`：

```text
dv_state = K_blk @ dH_old                   // [M, BV]

if g is provided:
    g_last = g_blk[M - 1]
    dv_state[t, :] *= E_g(g_last - g_blk[t])
else if gK is provided:
    // gK 不参与 dv2 token gate

dv2_blk = dv_state + DV_blk
```

更新 `dH` 时，`g` 和 `gK` 是两条不同分支：

- `g` 是 token-wise gate：参与 `dv2` 的 token gate，参与旧 `dH` 的统一 decay，也参与 `q` 项。
- `gK` 是 K-wise gate：只参与旧 `dH` 的 K 行 decay，不参与 `dv2`，也不参与 `q` 项。
- `g` 和 `gK` 不叠加；二者都存在或都不存在时不进入 kernel 计算。

```text
dH_decay = dH_old

if g is provided:
    dH_decay *= E_g(g_last)
    Q_gate[t, k] = Q_blk[t, k] * E_g(g_blk[t])
else if gK is provided:
    gK_last[k] = gK_blk[M - 1, k]
    dH_decay[k, :] *= E_gK(gK_last[k])
    Q_gate = Q_blk

term_q = Q_gate^T @ DO_blk                  // [K, BV]
term_w = W_blk^T  @ dv2_blk                 // [K, BV]

dH = dH_decay + term_q * scale - term_w
```

注意：`gK` 只读取当前 chunk 最后一个有效 token 的 `[K]` 向量，即本仓 shape 下的 `gK[seq, hv, global_last_idx, :]`。它不需要读取整个 `gK_blk[M,K]`，除非后续其它输出分支需要逐 token 的 key-wise gate。

## 4. 分核设计

### 4.1 统一 seq-head task 调度

```text
chunkNumPerB = ceil_div(T, BT)
headWindowNum = ceil_div(HV, 4)

if cu_seqlens is None:
    seqNum = B
    totalChunkNum = chunkNumPerB           // dh 的每 batch chunk 维
else:
    seqNum = len(cu_seqlens) - 1
    totalChunkNum = len(chunk_indices) / 2 // dh 的 flattened chunk 维

taskNum = seqNum * headWindowNum
blockDim = platformAicCoreNum

for task = coreIdx; task < taskNum; task += blockDim:
    seqIdx = task / headWindowNum
    headWindowIdx = task % headWindowNum
    hvBase = headWindowIdx * 4
    headCnt = min(4, HV - hvBase)
    for headOffset in [0, headCnt):
        InitDhState(seqIdx, hvBase + headOffset)

    seqInfo = GetSeqInfo(cu_seqlens, seqIdx)
    for localChunkIdx = seqInfo.chunkCnt - 1 downto 0:
        chunkInfo = GetChunkInfoBySeqChunk(chunk_indices, seqInfo, localChunkIdx)
        ProcessChunkHeadWindow(chunkInfo, hvBase, headCnt)

    for headOffset in [0, headCnt):
        StoreDh0IfNeeded(seqIdx, hvBase + headOffset)
```

`chunkTaskNum` 当前只保留为 tiling 中的 chunk 总量统计/输出规模辅助，不再驱动 kernel 主任务数。正式 `taskNum` 必须是 `seqNum * ceil_div(HV, 4)`，这样同一个 `(seq,hv)` 的 chunk 才能在一个 task 内倒序串行并共享同一份 fp32 workspace `dhState` carry。

### 4.2 GetSeqInfo/GetChunkInfoBySeqChunk 语义

```text
if cu_seqlens is None:
    seqInfo.chunkCnt = chunkNumForT if 0 <= seqIdx < B else 0
    seqInfo.outputChunkBase = 0
    seqInfo.tokenStart = 0
    seqInfo.tokenEnd = T
    bIdx = seqIdx
    chunkIdx = localChunkIdx
    tokenStart = localChunkIdx * BT
    tokenEnd = min(tokenStart + BT, T)
    outputChunkIdx = localChunkIdx
else:
    seqInfo.outputChunkBase = sum(ceil_div(cu_seqlens[i+1] - cu_seqlens[i], BT) for i < seqIdx)
    seqInfo.chunkCnt = ceil_div(cu_seqlens[seqIdx + 1] - cu_seqlens[seqIdx], BT)
    seqInfo.tokenStart = cu_seqlens[seqIdx]
    seqInfo.tokenEnd = cu_seqlens[seqIdx + 1]
    chunkIdx = localChunkIdx
    tokenStart = seqInfo.tokenStart + localChunkIdx * BT
    tokenEnd = min(tokenStart + BT, seqInfo.tokenEnd)
    bIdx = 0
    outputChunkIdx = seqInfo.outputChunkBase + localChunkIdx
    if chunk_indices[2*outputChunkIdx:2*outputChunkIdx+2] != [seqIdx, localChunkIdx]:
        outputChunkIdx = find index j where chunk_indices[2*j:2*j+2] == [seqIdx, localChunkIdx]

chunkLen = tokenEnd - tokenStart
```

这与 `prepare_wy_repr_bwd` 的 `GetTaskInfo/GetChunkOffset` 风格一致：kernel 主循环不出现定长/变长两套处理入口。变长 canonical 输入下不再在每个 chunk 内从头线性扫描 `chunk_indices`，而是每个 task 先计算当前 seq 的 flattened chunk 前缀，再用 O(1) 下标校验；只有遇到非 canonical `chunk_indices` 顺序时才 fallback 线性反查，保留兼容语义。

### 4.3 递推依赖说明

当前三阶段首版让同一 `(seq,hv)` 的 chunk 在同一个 task 内倒序串行。每个 head 的 `dhState` carry 固定为 fp32，按 row tile 在 GM workspace 的 fp32 state 段和 Vector 专用 `stateFp32` ping/pong UB 之间搬运；每个 chunk 的 `stage_0` 将当前 `dhState=dH_old` 保存为公开输出 `dh`，Cube 的 `dvState=K@dh[chunk]` 因而读取的是更新前状态；Vector 随后按 `g/gK` 分支对 fp32 state row tile 做 decay，`stage_1/stage_2` 再用 `termQ/termW` 更新 fp32 state，作为下一个倒序 chunk 的输入。

当前递推调度已经满足 `chunk i` 依赖 `chunk i+1` 的串行顺序；仍未完整对齐的是 `dht` 初始化和 `dh0` fp32 输出语义。定长/变长差异必须继续收敛在统一 offset helper 中，不允许在 kernel 主流程拆两套分支。

### 4.4 4-head round 内部顺序

每个 task 必须以 4-head round 为单位调度：

```text
for headOffset = 0; headOffset < headCnt; ++headOffset:
    hv = hvBase + headOffset
    ProcessOneHeadInRound(chunkInfo, hv)
```

Cube 侧一个 task 内顺序处理最多 4 个 head；A2/A5 Vector 侧按 `headOffset % subBlockNum` 交替承包完整 head，两个 subblock 仍参与同一 raw flag 同步配平。workspace slot 对齐 `prepare_wy_repr_bwd` 的 window 方式：每个 AIC core 固定保留 8 个 per-head GM workspace slot，四个相邻 slot 组成一个 4-head window，slot 地址为 `coreIdx * 8 + windowStartSlot + headOffset`，`windowStartSlot` 按当前 core 的 task 轮次在 `0/4` 间切换；不能用 `hv & 1` 或 `subBlockIdx` 推导 workspace slot。

### 4.5 V tile 处理

完整递推版本中，`V` 不参与 block 级分核：

```text
for headOffset = 0; headOffset < headCnt; ++headOffset:
    hv = hvBase + headOffset
    for vBase = 0; vBase < V; vBase += BV:
        InitDHTile(hv, vBase)
        for chunk = chunkNum - 1; chunk >= 0; --chunk:
            ProcessOneChunk(hv, chunk, vBase)
        StoreDH0Tile(hv, vBase)
```

当前实现按 chunk 倒序处理，A2/A5 中每个 head 由 `headOffset % subBlockNum` 选中的 AIV 负责完整 `dH` row tile，fp32 state 在 workspace state 段和 `stateFp32` ping/pong UB 之间流转。这样 state 本身不经过 q dtype round-trip，同时 row tile 的计算写法继续贴近 `prepare_wy_repr_bwd`。

## 5. 阶段划分与依赖关系

本节先描述完整递推版本的目标阶段划分；当前代码已经推进到三阶段递推首版，具体已实现范围见本节后面的“当前代码 stage 对齐状态”。

每个 `(seq, hvBase, headOffset, vBase, chunk)` 的处理只拆为 `stage_0/stage_1/stage_2` 三个阶段。V tile 开始前的 `dH` 初始化、kernel 入口的可选 `dh0` 整体清零，以及 V tile 全部 chunk 结束后的 `dh0` 写回都只是循环入口/出口动作，不作为 stage。Cube/Vector 中间结果主要落 GM workspace，但 `dh` 和 `dv2` 这两个公开输出也作为 Cube 后续 GEMM 的输入，避免把同一份 DT 数据再镜像写一份 workspace。A5 `qgDt` 是仅供当前 Cube stage1 消费的内部量，当前通过四块 L1A slot 从 Vector UB 直接送入 Cube，不再经过 GM workspace；`g` 按 headOffset 独立映射，`gK` 在当前 window 内按共享 `hk` 映射到组首 slot。A2 继续使用 qg workspace 路径。4-head round 内每个 head 使用当前 window 的一个独立 workspace slot。A2/A5 中同一个 head 的 Vector 计算由 `headOffset % subBlockNum` 选中的 subblock 完整负责，另一个 subblock 只参与该 head 的跨核同步配平。

完整递推目标中，`stage_0` 的 Cube 和 Vector 都只依赖原始输入或当前 fp32 state row tile，不互相 wait。Vector 先把旧 `dH` 保存到公开输出 `dh`，再安全地预计算 `qgDt` 并对 `stateFp32` 中的 `dhState` 做 gate decay；后续 Cube 读取的是 GM 中已保存的 `dh[chunk]`，因此 `dvState=K@dH_old` 仍使用更新前的 `dH_old`，不会受到 state 原地 decay 影响。

| 阶段 | Cube 做什么 | Vector 做什么 | 缓存/驻留产物 |
|---|---|---|---|
| `stage_0` | 不等待 Vector；读取或预取当前 chunk 的 `K_blk=k[seq,hk,bos:eos,:]`、`W=w[seq,hv,bos:eos,:]`、`dO=d_o[seq,hv,bos:eos,vBase:vBase+BV]` 到 Cube 侧 resident，或至少完成 GM offset/layout 准备 | 从 fp32 workspace state 段读取当前 AIV 的 `dhState=dH_old` row tile 到专用 `stateFp32` UB；写公开输出 `dh[chunk]=cast(dH_old)`；若传 `g`，将当前 chunk 的 `g[M]` 搬入当前 head 的 `gRaw` UB resident，向量化生成 `gateFactor=E_g(g[t])` 和 `dvGateFactor=E_g(g_last-g[t])` 并驻留，`qgDt=cast(q*gateFactor)`，并做 `dhState *= E_g(g_last)`；若传 `gK`，整行读取 `gK_last[K]` 并生成 `gkLastFactor=E_gK(gK_last[K])`，按 K 行做 state decay；A5 当前 window 内同一 `hk` 只由首个 V head 生成一次 `qgDt=cast(q)`，后续 V heads 复用该 L1A slot，`g` 分支仍按 head 独立生成；A2 继续写各 head 的 qg workspace；更新后的 fp32 state row tile 写回 workspace；每个 head 保持原有 `vecToCube` 通知 | A5 四块 qg L1A slot，`g` 按 head 独立使用，`gK` 按 window 内 `hk` 共享；A2 qg workspace；公开输出 `dh`；fp32 `dhState[K,V]` workspace 段；gate UB resident |
| `stage_1` | 等 `dh/qgDt`；读取公开输出 `dh[chunk]` 作为 `dH_old`；A5 `g` 分支从 `l1AScratch[headOffset]` 读取，`gK` 分支从当前 window 内该 `hk` 首个 head 对应的共享 slot 读取；A2 从 qg workspace 搬入。复用或读取 `K_blk/dO`，计算 `dvState = K_blk @ dH_old`、`termQ = qgDt.T @ dO`；A5 bf16 将 `dvState` 按 row tile 写矩阵 CV，A2 和 fp16/fp32 写 GM workspace；`termQ` 写 GM workspace；A5 BF16 `V=128` 时，同一 `hk` 的连续 V heads 复用 K 的 L0A slot0，`termQ` 使用独立 slot1 | 等 `dvState` ready；A5 bf16 从矩阵 CV 消费，A2 和 fp16/fp32 从 workspace 消费；若传 `g`，复用 stage0 驻留的 `dvGateFactor` 计算 `dv2=cast(dvState*dvGateFactor+dv)`；若传 `gK`，计算 `dv2=cast(dvState+dv)`；只写公开输出 `dv2`，写完后通知 Cube | 矩阵 CV 或 workspace 中的 `dvState[M,BV]`，workspace `termQ[K,BV]`；A5 BF16 V128 的 K L0A resident；公开输出 `dv2` |
| `stage_2` | 等 `dv2` ready；读取公开输出 `dv2`，并复用或读取 `W`；计算 `termW = W.T @ dv2`；A5 bf16 在首个 CV tile 前通知 Vector 可启动，再按 row tile 写矩阵 CV，A2 和 fp16/fp32 写 GM workspace 后通知 Vector | 等 stage2 可启动；读取 workspace `termQ`，A5 bf16 按 tile 等待矩阵 CV `termW`，A2 和 fp16/fp32 读取 workspace `termW`；同时把 fp32 workspace state row tile 搬入 `stateFp32`；计算 `dhState = dhState + termQ * scale - termW` 并写回 workspace | 矩阵 CV 或 workspace 中的 `termW[K,BV]`；fp32 `dhState[K,V]` workspace 段 |

### 5.1 当前代码 stage 对齐状态

当前 `op_kernel` 已从 stage0 推进到三阶段递推首版，已经和代码行为对齐如下：

| 代码位置 | 当前已实现 | 还未实现 |
|---|---|---|
| `GetSeqInfo` / `GetChunkInfoBySeqChunk` | 参考 `prepare_wy_repr_bwd`，用 helper 同时处理定长和变长 offset；定长由 `seqIdx/localChunkIdx` 计算 batch 内 token 区间，变长由 `cu_seqlens` 得到 packed token 区间，并在 task 粒度计算 canonical flattened chunk 前缀，chunk 内 O(1) 校验 `chunk_indices`，非 canonical 输入才 fallback 线性反查 | 后续仍可继续减少 `cu_seqlens` 前缀计算的重复读取，但不能把主流程拆成定长/变长两套分支 |
| Kernel 主循环 | Cube/Vector 都按相同 `taskIdx -> seqIdx/headWindowIdx/hvBase/headCnt` 映射处理；若 `hasDh0`，Vector 在进入 task 循环前用 CANN `AscendC::Fill` 对完整 `dh0` GM 做一次性清零，并用 `SyncAll<true>()` 做全 Vector 同步，不需要 Cube 参与；一个 task 最多覆盖四个 value head；task 内负责当前 head 的 AIV 初始化该 head 的完整 fp32 workspace state，再按 seq 内 chunk 倒序串行执行 stage0/stage1/stage2 | 当前 `dhState` 入口仍固定清零，尚未从 `dht` 初始化；`dh0` 仍未按上游 fp32 shape/dtype 完整输出 |
| `ChunkGatedDeltaRuleBwdDhuVector` 调度 | A2/A5 中对每个 `headOffset`，`headOffset % subBlockNum` 选中的 subblock 执行当前 head 的完整 stage0/stage1/stage2 计算；另一个 subblock 不读写该 head 的 workspace/output，只参与同一 raw flag 的 set/wait 配平。一个 4-head window 内 subblock0 处理 `headOffset=0/2`，subblock1 处理 `headOffset=1/3` | 后续可继续优化四个 head 的跨阶段 overlap |
| `ChunkGatedDeltaRuleBwdDhuVector::ProcessChunkStage0` | 负责当前 head 的 AIV 从 fp32 workspace state 段按 row tile 读取 `stateFp32`，写公开 `dh` 对应行，Cube stage1 直接从 `dh` 读取 state；stage 内直接按 token 生成 `qgDt`；`g` 分支整段搬入 `g[M]` 到当前 head 的 `gRaw`，用向量 `Exp` 生成 `gateFactor=exp(g_t)` 和 `dvGateFactor=exp(g_last-g_t)` 并驻留，复用当前 chunk 最后 token 的向量 gate 结果做统一 decay；`gK` 分支生成 `qgDt=q`，读取 `gK_last[K]` 并用向量 `Muls/Exp` 生成 K 行 `gkLastFactor=exp2(gK_last)`；A5 将 `qgDt` 从 output UB 通过 MTE3 直接写入 L1A，`g` 按当前 `headOffset` 映射，`gK` 按当前 window 内共享 `hk` 的组首 head 映射，省去 UB->GM workspace 和 Cube GM->L1 两段搬运，A2 保持 qg workspace 路径；decay 后的 fp32 state 写回 workspace | 还没有从 `dht` 初始化真实 `dhState`，当前入口仍从 0 初始化 |
| `ChunkGatedDeltaRuleBwdDhuVector` copy helpers | 对齐 `prepare_wy_repr_bwd`：q 行输入、gate 输入、dv 行输入和输出写回拆成 `CopyIn/Cast/CopyOut` helper。q 输入、gate 输入和输出各自使用独立 ping-pong 事件组 | 当前调用顺序较直，但事件设计不要求 q/gate 固定串行 |
| stage0 的 `USE_GK=0` 分支 | stage 内读取 `q/g`，`gRaw/gateFactor/dvGateFactor` 按当前 headOffset 在 UB 中驻留；用向量 `Exp` 生成自然指数系数，再用 `Brcb+Mul` 写出 `qgDt = q * exp(g_t)`；随后按当前 chunk 最后 token 做 `dhState *= exp(g_last)`；stage1 直接复用 `dvGateFactor=exp(g_last-g_t)` | `g` 分支代码和 golden 都使用自然指数；不通过 scalar 读 gate 系数生成向量 |
| stage0 的 `USE_GK=1` 分支 | 每个 head 仍独立读取 `gK_last[K]`，生成 `exp2(gK_last[K])` 并完成 state decay；A5 当前 window 内同一 `hk` 只由首个 V head 读取和转换 q、写一次共享 qg L1A slot，后续 V heads 不重复 q 的 GM->UB、cast 和 UB->L1；`gK` 不参与 token gate 或 `qgDt` | 已对齐上游 gate 指数语义；继续保持 `gK` dtype/shape 与上游一致 |
| `ChunkGatedDeltaRuleBwdDhuCube` | 先搬 `K/dO`，等待当前 `headOffset` 的 `vecToCube` flag；A5 `g` 分支按 headOffset 消费独立 qg slot，`gK` 分支按当前 window 内 `hk` 首 head 映射消费共享 qg slot，A2 从 qg workspace 搬入；用 tile 级 resident/双缓冲计算 `dvState = K @ dh[chunk]` 和 `termQ = qgDt.T @ d_o`；bf16 路径将 `dvState` 按 `vecRow` row tile 通过 CV 发给负责当前 head 的 Vector subblock；fp16/fp32 路径写 GM workspace；随后发布 `cubeToVec`；`K` 使用 L1 resident ping/pong 并在同 chunk shared-key heads 间复用；A5 BF16 `V=128` 进一步让同组 K 驻留 L0A slot0，`termQ` 使用 slot1 | 其它 dtype/V 档位保持 L1 resident 到 L0 ping/pong 路径 |
| CrossCore 同步 | 当前使用 `prepare_wy_repr_bwd` 风格的两套 raw flag：`vecToCube=2`、`cubeToVec=4`。每个 head 的每个同步点两个 Vector subblock 都参与一次 `CrossCoreSetFlag<0x2, PIPE_MTE3>` 或 `CrossCoreWaitFlag`，Cube 侧按 head 串行等待/发布；A2/A5 head 交替分摊只改变实际 producer，不改变 raw flag 次数 | 后续可继续优化四个 head 的跨阶段 overlap |
| `ChunkGatedDeltaRuleBwdDhuVector::ProcessChunkStage1` | 负责当前 head 的 subblock 消费 `dvState` 和输入 `dv`。A5 bf16 路径先发起 `dv` 的 GM->UB 搬运，再等待矩阵 CV ready、cast `dvState` 并立即归还 free；fp16/fp32 路径从 GM workspace 读回 `dvState`；`g` 分支复用 stage0 的 `dvGateFactor` UB resident，`gK` 分支直接相加；按 token row tile 只写公开 `dv2` | 后续精度对齐仍需确认 dht/h0 语义 |
| `ChunkGatedDeltaRuleBwdDhuCube::ProcessChunkStage2` | 等待 Vector 的 `dv2` ready 后，用 tile 级 W resident/双缓冲计算 `termW = w.T @ dv2`；A5 bf16 在首个矩阵 CV tile 前发布 `cubeToVec`，逐 tile 使用 ready/free；fp16/fp32 写 GM workspace 后发布 `cubeToVec` | `W` 当前按 head resident，不做跨 chunk 复用 |
| `ChunkGatedDeltaRuleBwdDhuVector::ProcessChunkStage2` | 等待 stage2 可启动后，按 K 行读取 workspace `termQ` 和 fp32 state；A5 bf16 从矩阵 CV 读取 `termW`，fp16/fp32 从 workspace 读取；计算 `dhState = dhState + termQ * scale - termW`，再以 fp32 写回 workspace 供下一个倒序 chunk 使用 | 还没有支持从 `dht` 初始化和 `dh0` fp32 输出语义 |

因此，当前代码已经完成的是 `g/gK` 二选一、shape/tilingKey 分支、固定/变长统一 seq-head task 调度、seq 内 chunk 倒序串行、Vector fp32 `dhState` workspace carry 和专用 `stateFp32` ping/pong UB、`g` 自然指数与 `gK` exp2 的分支语义、stage0 的 `qgDt/dh`、stage1 的 `dvState/termQ/dv2`、stage2 的 `termW` 和 state update，以及 A5 bf16 `dvState/termW` 共用矩阵 CV 双缓冲传递。继续缺口主要是 `dht/h0/dh0` 完整语义、`state_v_first` 布局和进一步性能流水。

### 5.2 当前代码与完整方案不一致项

| 项目 | 当前代码 | 完整目标/需要更新的方向 |
|---|---|---|
| gate 指数函数 | `g` 分支直接用 `AscendC::Exp`，golden 用 `torch.exp`；`gK` 分支用 `x * ln2` 后调用 `AscendC::Exp`，golden 用 `torch.exp(x * ln2)` | 两个模板分支分别保持对应指数语义 |
| chunk 调度 | 当前已经按 `seqIdx/headWindowIdx` 作为 task，task 内按 `headOffset % subBlockNum` 选中的 AIV 初始化该 head 的 fp32 workspace state，再按 seq 内 chunk 倒序串行；定长/变长都通过统一 helper 拿 offset | 后续如果引入 V tile 并行、head 间 overlap 或 chunk 级流水，必须保持每个 `(seq,hv,vTile)` 的 chunk 倒序依赖，不允许退回 chunk 并行 |
| workspace slot | 当前对齐 prepare 的 8-slot 方式：每个 AIC core 保留 8 个 per-head slot，大小为 `blockDim * 8 * workspaceElemsPerSubBlock * sizeof(DT)`，slot 地址为 `coreIdx * 8 + windowStartSlot + headOffset`，布局保留 `qgDt/dhState/dvState/termQ/termW` 五段；A5 的 `qgDt` 及 bf16 `dvState/termW` 段仅预留，运行时有效 workspace 数据为 `dhState/termQ`；`dhState` 按 fp32 字节数预留，`stateDt/dv2Dt` 不占 workspace | 后续如果其它段也改 fp32，应继续按 byte 对齐折算到对应 dtype 指针 |
| `dhState` | 当前使用 fp32 workspace carry 和单独 `stateFp32` ping/pong UB；负责当前 head 的 AIV 按 `vecRow` 连续 row tile 搬入、更新和写回完整 `[K,V]`，入口仍固定清零 | 后续要从 `dht` 或 0 初始化 fp32 `dhState[K,V]`，最后按需输出真实 `dh0` |
| `gK` 分支 | 当前整行读取当前 chunk 最后有效 token的 `gK_last[K]` 到 UB，用向量指令生成 K 个 `exp2` decay 系数，并保持 `qgDt=q`、`dv2` 不加 token gate | 需要继续保持上游 dtype/shape 行为 |
| `dv2` | 当前 stage1 按 `dvState(+gate)+dv` 路径只写公开 `dv2`；`g` 分支加自然指数 token gate，`gK` 分支不加 | 需要和 `dht/h0` 完整语义一起继续做全量精度闭环 |
| Cube 路径 | 当前 Cube 使用 tile 级接口写 `K @ dh[chunk]`、`qgDt.T @ d_o` 和 `w.T @ dv2`；`K/W` 有独立 L1 resident，scratch/L0A/L0B/L0C 按 prepare 风格管理事件；A5 BF16 `V=128` 在共享 `hk` 的 V heads 间复用 K 的 L0A slot0，`termQ` 固定使用 slot1 | 其它模板分支继续使用现有 L0 ping/pong，后续优化保持相同事件闭环 |
| 跨 AIC/AIV 同步 | 当前已经按 `prepare_wy_repr_bwd` 形态接入 stage1/stage2 的 `vecToCube/cubeToVec`。四个 head 串行复用同一对 raw flag；同一 head 内两个 Vector subblock 必须都参与同一个 `<0x2>` 同步点，其中负责该 head 的 subblock 生产数据，另一个 subblock 只做同步配平 | 后续若做 head 间并行，需要先证明 raw flag 顺序协议或设计 ready/free 复用协议 |
| `dh0` | 当前 `hasDh0` 时入口通过 tiling 给出的 `dh0ClearCoreNum/dh0ClearElemsPerCore/dh0ClearTailElems` 切分，使用 `AscendC::Fill` 一次性清零完整当前 5D `dh0` 输出；非最后参与 Vector 核的清零字节数保持 512B 对齐，最后参与核处理尾部。最终每个 seq/head 在出口只写对应首 chunk 的递推 state | 后续应输出 `[N,HV,K,V] fp32`，并在 wrapper/meta/kernel 地址计算中统一 |

当前代码主流程已经按下面的倒序递推框架执行；尚未完整对齐的是 `InitDHTile` 从 `dht` 初始化和 `StoreDh0` 的上游 shape/dtype 语义。单个 head 的每个 V tile 按以下顺序执行：

```text
InitDHTile

for localChunk = chunkNum - 1 downto 0:
    stage_0
    stage_1
    stage_2

StoreDh0
```

其中 `InitDHTile` 是在当前 `hv/vBase` 开始时加载 `dht[seq,hv,:,vBase:vBase+BV]` 到 UB `dhState[K,BV]`；如果 `dht` 为空则清零。`StoreDh0` 是当前 `hv/vBase` 的所有 chunk 处理完后，如果 `h0` 非空，将最终 `dhState` 写 `dh0[seq,hv,:,vBase:vBase+BV]`。它们都不参与 Cube/Vector 跨核同步。

`stage_0` 是双方无跨核等待的准备阶段；`stage_1/stage_2` 才通过 GM workspace ready 串起真实依赖。下一轮 chunk 的 `stage_0` 必须等当前 chunk 的 `stage_2` 完成后才能开始，避免覆盖或提前使用尚未更新完的 `dhState`。

伪代码：

```text
ProcessHeadWindowChunks(chunkInfoListDesc, hvBase, headCnt):
    for headOffset = 0; headOffset < headCnt; ++headOffset:
        hv = hvBase + headOffset
        hk = hv / groupSize

        for vBase in range(0, V, BV):
            dH = LoadDhtOrZero(chunkInfoListDesc[0].seqIdx, hv, vBase)     // UB fp32 [K,BV]

            for chunkInfo in chunkInfoListDesc:
                chunkLen = chunkInfo.chunkLen
                tokenBase = chunkInfo.tokenStart
                outChunkIdx = chunkInfo.outputChunkIdx

                // stage_0: Cube no wait; Vector saves old dH, prepares qgDt, applies selected g or gK decay to UB dH
                Cube_PrepareInputs(k[hk, tokenBase], w[hv, tokenBase], d_o[hv, tokenBase, vBase])
                Vector_SaveStateQgAndDecay(dH, q[hk, tokenBase], g[hv, tokenBase], gK[hv, tokenBase],
                                           outChunkIdx, hv, vBase)
                VecToCubeReady()

                // stage_1: Cube consumes dh/qgDt; Vector consumes dvState
                Cube_WaitVec()
                Cube_DvStateAndTermQ(k[hk, tokenBase], dh[outChunkIdx], qgDt, d_o[hv, tokenBase, vBase])
                CubeToVecReady()

                Vector_WaitCube()
                Vector_Dv2(dvState, dv[hv, tokenBase, vBase], g[hv, tokenBase])
                VecToCubeReady()

                // stage_2: Cube consumes dv2; Vector updates recurrent state
                Cube_WaitVec()
                Cube_TermW(w[hv, tokenBase], dv2)
                CubeToVecReady()

                Vector_WaitCube()
                Vector_UpdateState(dH, termQ, termW, scale)  // dH already contains selected gate decay

            if outputDh0:
                StoreDh0(dH, chunkInfoListDesc[0].seqIdx, hv, vBase)
```

同步次数按单 chunk 统计：

```text
Vector -> Cube: 2 次
    1. dh/qgDt ready
    2. dv2 ready

Cube -> Vector: 2 次
    1. dvState/termQ ready
    2. termW ready
```

这些同步都在单 block 的 AIC/AIV 对之间发生，不涉及不同 block 间通信。

## 6. GM Workspace 设计

### 6.1 当前 stage1/stage2 workspace

当前代码已经按 stage1/stage2 扩展 workspace，slot 对齐 prepare 的 resident 窗口方式：每个 AIC core 保留 8 个 per-head slot，四个相邻 slot 组成一个 4-head window，窗口在当前 core 的连续 task 间按 `0..3 -> 4..7 -> 0..3` 轮转；A2/A5 中同一 head 的 workspace slot 由负责该 head 的 Vector subblock 完整写入。tiling 中仍保留 `qgDt/dhState/dvState/termQ/termW` 五段偏移以保持当前 tiling/workspace 布局和其它路径兼容，其中 `dhState` 是 fp32 carry 段；A5 当前 `qgDt` 通过四块独立 L1A slot 直传，bf16 `dvState/termW` 按阶段复用同一套 CV 双缓冲直传，对应 workspace 段仅保留布局。`stateDt` 直接复用公开输出 `dh`，`dv2Dt` 直接复用公开输出 `dv2`。地址计算为：

```text
qgWorkspaceElems         = chunkSize * K
stateWorkspaceOffset     = align32(qgWorkspaceElems * sizeof(DT)) / sizeof(DT)
stateWorkspaceElems      = align32(K * V * sizeof(float)) / sizeof(DT)
dvStateWorkspaceElems    = chunkSize * V
termQWorkspaceElems      = K * V
dv2WorkspaceElems        = 0
termWWorkspaceElems      = K * V

qgWorkspaceOffset        = 0
dvStateWorkspaceOffset   = stateWorkspaceOffset + stateWorkspaceElems
termQWorkspaceOffset     = dvStateWorkspaceOffset + dvStateWorkspaceElems
dv2WorkspaceOffset       = termQWorkspaceOffset + termQWorkspaceElems
termWWorkspaceOffset     = dv2WorkspaceOffset + dv2WorkspaceElems
workspaceElemsPerSubBlock = termWWorkspaceOffset + termWWorkspaceElems

workspaceSize = sysWorkspaceSize
              + blockDim * 8 * workspaceElemsPerSubBlock * sizeof(DT)

workspaceBase = workspace
              + (coreIdx * 8 + windowStartSlot + headOffset) * workspaceElemsPerSubBlock

stateFloatBase = reinterpret_cast<float *>(workspace)
               + ((workspaceBase + stateWorkspaceOffset) * sizeof(DT)) / sizeof(float)
```

当前 slot 内布局：

| 名称 | Shape | dtype | 当前用途 |
|---|---|---|---|
| `qgDt` | `[BT, K]` | q dtype | 当前保留 workspace 布局；A5 运行时由 Vector UB 直接写四块独立 L1A qg slot，Cube 不读本段 |
| `dhState` | `[K, V]` | fp32 | Vector 递推 carry；按 row tile 在 workspace 和 `stateFp32` UB 之间流转 |
| `dvState` | `[BT, V]` | q dtype | A2 和 fp16/fp32 路径由 Cube stage1 写 `K @ dh[chunk]`；A5 bf16 仅保留布局，运行时走 CV |
| `termQ` | `[K, V]` | q dtype | Cube stage1 写 `qgDt.T @ d_o`；Vector stage2 消费 |
| `dv2Dt` | 0 | q dtype | 不占 workspace；Cube stage2 直接读取公开输出 `dv2` |
| `termW` | `[K, V]` | q dtype | A2 和 fp16/fp32 路径由 Cube stage2 写 `w.T @ dv2`；A5 bf16 仅保留布局，运行时走 CV |

当前代码不分 fixed/varlen workspace 分支；fixed/varlen 差异只来自 `GetChunkInfo` 的 token/chunk offset。workspace slot 必须继续使用 `coreIdx * 8 + windowStartSlot + headOffset`，不能改成 `hv & 1` 或 `subBlockIdx`。

### 6.2 完整递推目标 workspace

完整递推版本中，Cube 和 Vector 交换以下中间产物，`stateDt` 复用公开输出 `dh`，`dv2Dt` 复用公开输出 `dv2`。A5 的 `qgDt` 按当前 UB->L1 方案直传，A2 的 `qgDt` 继续通过 GM workspace；其它中间量按表中 workspace 布局保存。当前 slot 粒度按每个 active head 复用；A2/A5 中同一 head 的 slot 由负责该 head 的 Vector subblock 完整读写。若后续引入同一 head 内 chunk 级流水、V tile 并行或 head 间并行，必须升级为 ring slot 并配套 ready/free 计数。

| 名称 | Shape | dtype | 字节数 |
|---|---|---|---|
| `dvState` | `[BT, BV]` | fp32 | `align32(BT * BV * 4)` |
| `qgDt` | `[BT, K]` | q dtype | `align32(BT * K * sizeof(DT))` |
| `termQ` | `[K, BV]` | fp32 | `align32(K * BV * 4)` |
| `termW` | `[K, BV]` | fp32 | `align32(K * BV * 4)` |

offset 计算：

```text
offset = 0
dvStateOffset = offset; offset += Align32(BT * BV * sizeof(float))
qgDtOffset    = offset; offset += Align32(BT * K  * sizeof(DT))
termQOffset   = offset; offset += Align32(K  * BV * sizeof(float))
termWOffset   = offset; offset += Align32(K  * BV * sizeof(float))

workspaceCoreSize = Align32(offset)
workspaceSlotPerCore = 8
userWorkspaceSize = blockDim * workspaceSlotPerCore * workspaceCoreSize
workspaceSize     = sysWorkspaceSize + userWorkspaceSize
```

完整递推目标建议 `BV=64`。最坏规格 `BT=128,K=128,BV=64,DT=bf16/fp16` 下单 slot workspace 约为：

```text
dvState  32KB
qgDt     32KB
termQ    32KB
termW    32KB
total   128KB
```

## 7. Vector UB 设计

### 7.1 当前 stage1 UB

当前代码已接入 stage1 输出路径，Vector UB 分配如下：

| 名称 | Shape | dtype | 说明 |
|---|---|---|---|
| `matrixCvPing/Pong` | `[vecRow,V]` | bf16 | A5 bf16 矩阵 CV 双缓冲；stage1 承载 `dvState`，stage2 承载 `termW`，Vector cast 后通过专用 free/ready raw flag 归还；位于 UB 最前面的固定偏移 |
| `qInputPing/Pong` | `[vecRow, max(K,V)]` | q dtype | 读取连续 q/dv/workspace 多行，使用 `DataCopy` |
| `gateInputPing/Pong` | `[max(chunkSize,K)]` | active gate dtype | `g` 分支读取当前 chunk 的 `g[M]`；`gK` 分支读取当前 chunk last token 的 `gK_last[K]`；使用 `DataCopyPad`，不走逐元素计算 gate 路径 |
| `outputPing/Pong` | `[vecRow, max(K,V)]` | q dtype | `dh/dv2` 连续多行写 GM；A5 `qgDt` 连续多行按 NZ 分列写当前 head 的 L1A slot，A2 `qgDt` 写 GM workspace |
| `stateFp32Ping/Pong` | `[vecRow,V]` | fp32 | `dhState` 专用 UB 双 buffer；从 fp32 workspace state 段搬入，完成 decay/update 后以 fp32 写回 |
| `qFp32` | `[vecRow, max(K,V)]` | fp32 | q/dv/workspace 多行 cast 后的计算区 |
| `gRawAll` | `[4,max(chunkSize,K)]` | fp32 | `g` 分支按 headOffset 驻留当前 chunk 的 raw `g[M]`；`gK` 分支作为 `gK_last[K]` 临时区 |
| `gateFactorAll` | `[4,max(chunkSize,K)]` | fp32 | `g` 分支按 headOffset 驻留 `exp(g_t)`，stage0 的 `qg/dhState` 复用；`gK` 分支可作为 `gkLastFactor=exp2(gK_last[K])` 临时区 |
| `dvGateFactorAll` | `[4,max(chunkSize,K)]` | fp32 | `g` 分支按 headOffset 驻留 `exp(g_last-g_t)`，stage1 直接复用，不重复搬运或计算小 gate tensor；`gK` 分支不使用 |
| `outFp32` | `[vecRow, max(K,V)]` | fp32 | 零行、`dvState/dv` 合并和 gate 缩放的计算区 |

`vecRow` 由 host 根据 UB 剩余空间计算，至少按 8 行对齐。A2/A5 中两个 Vector subblock 先按 `headOffset % subBlockNum` 交替选择完整 head，负责当前 head 的 subblock 再按 `rowOffset += vecRow` 切连续 tile。这样能把 `dh/state/qg/dv2/termQ/termW` 从大量 128 元素小 DMA 合并为多行连续搬运，同时减少两个 subblock 重复生成 gate resident 小 tensor。A5 qg 预留四块 L1A slot：`g` 分支按 `headOffset` 一一映射；`gK` 分支将当前 window 内同一 `hk` 的 V heads 映射到该组首个 head 的 slot，由首个 head 对应的 subblock 生产一次。共享 slot 保持到该组最后一个 V head 的 `termQ` 消费完成，下一 chunk 在四个 head 的 stage2 `cubeToVec` 握手完成后整体复用，不增加 qg 专用 flag。

A5 Vector 的 regbase 计算按连续依赖合成单趟向量循环：`gateFactor=exp(g)`、`dvGateFactor=exp(g_last-g)`、`dv2=dvState * dvGateFactor + dv`、`state += termQ * scale - termW` 都在各自 row tile 内一趟完成；`gK` 的 `exp2` 仍使用 `x * ln2` 后执行 `Exp`。每个 head/chunk 先计算 `stateBase/dvStateBase/termQBase/termWBase/qBase/dvBase`，row 循环内只保留 `base + rowOffset * stride` 的地址递推；qg 的目的地址由当前 `headOffset` 对应的 L1A slot 和 row offset 直接得到。

当前 A2 bf16/fp32 gate 典型配置下，`vecRow` 由 UB 预算决定，不硬编码：

1. A2 UB 约 192KB。
2. `stateFp32Ping/Pong` 按 `2 * vecRow * V * sizeof(float)` 预算，独立于 `qInput/output` 的 ping/pong buffer。
3. gate ping/pong、per-head gate resident、`stateFp32` ping/pong 和 16KB guard 后，host 会按实际 UB 余量从 `K` 起检查，放不下时每次按 `row /= 2` 折半缩小；token 方向由 kernel 内 `min(vecRow, chunkLen - rowOffset)` 自然截断，不用 `chunkSize` 限制搜索上限；`g` 分支按 `gRaw/gateFactor/dvGateFactor` 三份 resident 预算，`gK` 分支只按 `gkLastFactor` 一份 resident 预算。
4. `vecRow` 变大时，q/dv/workspace 输入、输出、fp32 计算区、fp32 state ping/pong 和 bf16 专用 CV ping/pong 都同步增长，需要继续基于 host UB 预算公式评估。

因此，后续若要提高 `vecRow`，需要继续基于 host UB 预算公式评估，不能直接把常量改大。当前方案固定使用 fp32 workspace carry 和专用 `stateFp32` UB ping/pong，递推 state 不做 q dtype round-trip。

当前主要 Vector 工作：

1. `Process` 主循环
   - 参考 `prepare_wy_repr_bwd`，所有 subblock 使用同一 task 映射。
   - A2/A5 中每个 `headOffset` 由 `headOffset % subBlockNum` 选中的 subblock 执行完整 stage0/stage1/stage2 计算；两个 subblock 都参与当前 `headOffset` 的 `vecToCube`/`cubeToVec` 握手。

2. `ProcessChunkStage0`
   - 按连续 K-row tile 从 fp32 workspace state 段搬入 `stateFp32`，写公开输出 `dh`，并在 decay 后把 fp32 state 写回 workspace。
   - 不在每个 chunk 内清零 `dh0`；若 `hasDh0`，完整 `dh0` 已在 kernel 入口由各 Vector 核按 tiling 切分用 `Fill` 清零。
   - 不预清零 `dvState/termQ/termW` workspace。stage0 写完 `dh/qgDt` 后，通过原有 `CrossCoreSetFlag<..., PIPE_MTE3>` 发布 `vecToCube`，保证同一 MTE3 pipe 上的数据搬运完成后再通知 Cube。
   - 按连续 token-row tile 遍历当前 chunk 有效 token，在 stage 内直接读取 q 多行并生成 `qgDt`；A5 将 cast 后的 q dtype tile 使用 `BLOCK_MODE_VECTOR` 从 UB 直接写 L1 NZ 布局，A2 写 qg workspace。

3. stage0 的 qg/gate 路径
   - 两个分支都先读取 q 行并 cast 到 fp32。
   - `USE_GK=0`：读取当前 chunk 的 `g[M]` 到当前 head 的 `gRaw` resident，用向量指令生成 `gateFactor=exp(g_t)` 和 `dvGateFactor=exp(g_last-g_t)` 并驻留，再用 `Brcb+Mul` 生成 `qgDt = q * gateFactor[t]`；A5 直写当前 head 的 L1A slot，A2 写 qg workspace。
   - `USE_GK=1`：不使用 `gK` 生成 qg；A5 当前 window 内同一 `hk` 只由首个 V head 生成 `qgDt = q` 并写共享 L1A slot，后续 V heads 复用，A2 仍按 head 写 qg workspace。
   - 不把 `qgDt/q` 镜像到 `dv2`；`dv2` 只从 stage1 输出路径写。

4. `ProcessChunkStage1`
   - 等待 Cube stage1 ready。
   - 由负责当前 head 的 subblock 按 token row tile 消费 `dvState` 和输入 `dv`；A5 bf16 从矩阵 CV 通道读取，A2 和 fp16/fp32 从 workspace 读取。
   - `USE_GK=0`：复用当前 head的 `dvGateFactor` resident 对 `dvState` 做 token gate，不再二次搬入 `g[M]`，也不重复计算 `exp(g_last-g_t)`。
   - `USE_GK=1`：不读取 gate，直接 `dvState + dv`。
   - 只写公开输出 `dv2`，不再额外写 workspace `dv2Dt`。

5. `ProcessChunkStage2`
   - 等待 Cube stage2 ready。
   - 由负责当前 head 的 subblock 按 K row tile 读取 `termQ/termW`，把 fp32 workspace state 搬入 `stateFp32`，执行 `dhState += termQ * scale - termW`，再以 fp32 写回 workspace，作为下一个倒序 chunk 的输入。A5 bf16 的 `termW` 从矩阵 CV 通道读取，`termQ/state` 的独立搬入在等待当前 CV tile 前发起。

当前 Vector 内部 HardEvent：

```text
q input ping/pong:
    MTE2_V ready
    V_MTE2 free

gate input ping/pong:
    MTE2_V ready
    V_MTE2 free

output ping/pong:
    V_MTE3 ready
    MTE3_V free

stateFp32 ping/pong:
    MTE2_V ready
    V_MTE2 free
    V_MTE3 ready
    MTE3_MTE2 free
```

q、gate 输入和 stateFp32 参考 `prepare_wy_repr_bwd` 使用独立事件组。当前 qg 调用顺序仍偏直：读取 q、读取 gate、计算 qg；但这是实现简化，不是设计约束。后续可把 q/gate 的 copy-in 调度成先发起各自搬入，再等待二者 ready 后做 `qg`，或进一步预取下一 token；`qg` 计算本身只依赖 q 行和向量 gate 结果都 ready。EventID 在 `Init()` 统一 `AllocEventID` 并初始化 free flag，`Process()` 结束后统一等待并 `ReleaseEventID`。不要在每个 chunk 内重复分配。

输出 copyout 也按 `prepare_wy_repr_bwd` 处理：`dh/dv2` 通过当前 output slot 发起 MTE3 GM 写回；A5 `qgDt` 通过同一 output ping/pong 的 `V_MTE3/MTE3_V` 生命周期直写 L1，A2 `qgDt` 写 GM workspace。stage 末尾直接用 `CrossCoreSetFlag<..., PIPE_MTE3>` 通知 Cube，依赖同一 MTE3 pipe 顺序保证数据可见；A5 不增加 qg 专用同步，也不额外封装全 slot flush helper。

### 7.2 完整递推目标 UB

完整递推版本中，Vector 侧核心是让 `dH[K,BV]` 以 fp32 在 workspace state 段和专用 UB ping/pong 之间流转，跨 chunk 倒序递推。目标 UB 分配：

| 名称 | Shape | dtype | 说明 |
|---|---|---|---|
| `stateFp32Ping/Pong` | `[vecRow, BV]` | fp32 | 当前 row tile 的递推状态；从 fp32 workspace state 段搬入，完成 decay/update 后写回 |
| `inputPing/Pong` | 16KB each | DT | GM 输入 tile 临时区 |
| `outputPing/Pong` | 16KB each | DT | GM 输出 tile 临时区 |
| `calcA` | 32KB | fp32 | fp32 计算区 |
| `calcB` | 32KB | fp32 | fp32 计算区 |
| `calcC` | 32KB | fp32 | fp32 计算区 |
| `gRawBuf` | `[BT]` | fp32 | 当前 chunk 的 raw `g` |
| `gateFactorBuf` | `[BT]` | fp32 | `E_g(g_t)`，用于 `qgDt` 和 `dhState` decay |
| `dvGateFactorBuf` | `[BT]` | fp32 | `E_g(g_last-g_t)`，用于 stage1 `dv2` token gate |
| `gKLastBuf` | `[K]` | fp32 | 可选，`E_gK(gK_last)` 行向量 |
| `rowScalar` | `[max(BT,K)]` | fp32 | reduce/broadcast 临时 |

完整目标的主要 Vector 工作：

1. `LoadDhtOrZero`
   - 如果 `dht` 非空，由负责当前 head 的 AIV 从 `dht[seq,hv,:,vBase:vBase+BV]` 初始化该 head 的 fp32 workspace state。
   - 否则由负责当前 head 的 AIV 将该 head 的 fp32 workspace state 清零。

2. `Vector_SaveStateQgAndDecay`
   - 将 fp32 workspace state row tile 搬入 `stateFp32`，cast 为 DT 写 `dh` 对应 chunk。
   - 不再把同一份 cast 结果镜像写 workspace `stateDt`；Cube 直接从公开输出 `dh` 读取。
   - 若传 `g`，读取 `q/g`，`gRaw/gateFactor/dvGateFactor` 在当前 head 的 UB slot 内驻留，生成 `qgDt = q * E_g(g)`；A5 将 q dtype tile 直写当前 head 的 L1A slot，A2 写 qg workspace。随后对 `stateFp32` 做 `dhState *= E_g(g_last)` 并以 fp32 写回 workspace。
   - 若传 `gK`，读取 `q`，生成 `qgDt = q`；A5 将 q dtype tile 直写当前 head 的 L1A slot，A2 写 qg workspace。再整行读取当前 chunk 最后一个有效 token 的 `gK_last[K]`，用向量指令生成 `E_gK(gK_last[K])`，对 `stateFp32` 按行做 `dhState[k, :] *= E_gK(gK_last[k])` 后以 fp32 写回 workspace；`gK` 不影响 `qgDt`。
   - `dh` 尾列 `vBase + col >= V` 不写。

3. `Vector_Dv2`
   - 读取 `dvState[M,BV]` fp32。
   - 若传 `g`，复用 stage0 驻留的 `dvGateFactor[M]`，`dv2 = dvState * dvGateFactor + dv`，不重复搬运或计算小 gate tensor。
   - 若传 `gK`，`dv2 = dvState + dv`；`gK` 不参与 `dv2` token gate。
   - 写真实输出 `dv2`，不再额外 cast 写 workspace `dv2Dt`。

4. `Vector_UpdateState`
   - 读取 `termQ/termW` 并 cast 到 fp32，同时读取 fp32 workspace state 到 `stateFp32`。
   - `dhState += termQ * scale - termW` 后以 fp32 写回 workspace。

## 8. Cube L1/L0 设计

当前 A5 代码中的 `ChunkGatedDeltaRuleBwdDhuCube` 保留了 stage0 命名，但已经接入 stage1/stage2 三个 Cube GEMM：完成 task 到 seq/head/chunk 的映射、`GetSeqInfo/GetChunkInfoBySeqChunk`、输入 GM offset 和 workspace slot 地址计算；等待 Vector 的 `dh/qgDt` ready 后计算 `dvState = K @ dh[chunk]` 与 `termQ = qgDt.T @ d_o`。`qgDt` 预留四块 `128x128xDT` L1A slot：`g` 分支按 `headOffset` 独立存放，`gK` 分支在当前 window 内按 `hk` 共享该组首个 head 的 slot。Vector UB->L1 的生产和 Cube L1->L0A 的消费继续复用原有每 head `vecToCube` ready。A5 BF16 `V=128` 的 `K @ dh` 将同一 `hk` 的 K 保持在 L0A slot0：首次 L1A->L0A 后即可归还 K resident L1 slot，连续 V heads 各自更新 L0B state 并独立执行 MMAD，最后一个共享 head 完成后才归还 L0A slot0；`termQ` 使用 L0A/L0B slot1。A5 bf16 的 `dvState/termW` 按各自 stage 生命周期复用同一套矩阵 CV ping/pong，从 L0C 按 row tile 直接发给负责当前 head 的 Vector subblock；A2 和 fp16/fp32 使用 GM workspace。Cube 侧保持 `prepare_wy_repr_bwd` 的 tile 级 resident/双缓冲流水。

完整递推 Cube 侧目标只做三个 GEMM：

```text
GEMM0: dvState = K_blk[M,K] @ dh_old[K,BV]
GEMM1: termQ   = qgDt^T[K,M] @ d_o[M,BV]
GEMM2: termW   = w^T[K,M]    @ dv2[M,BV]
```

建议 tile：

| GEMM | M | N | K | 输出 |
|---|---:|---:|---:|---|
| `K @ dH` | `chunkLen` | `BV` | `128` | `[BT,BV] fp32` |
| `qg^T @ d_o` | `128` | `BV` | `chunkLen` | `[K,BV] fp32` |
| `w^T @ dv2` | `128` | `BV` | `chunkLen` | `[K,BV] fp32` |

当前实现必须使用现有相邻算子的 tile 级接口风格。普通 GM operand 和 A5 qg operand 的路径分别为：

```text
普通 GM operand: GM -> L1 -> L0A/L0B -> MMAD -> L0C -> GM workspace/output
A5 qg operand:    Vector UB -> L1A -> L0A -> MMAD -> L0C -> GM workspace
A5 BF16 matrix:     GM/L1 -> L0A/L0B -> MMAD -> L0C -> AIV UB
```

当前 resident/双缓冲是硬约束，不是后续性能优化项；不允许以“保守首版/先跑通”为由退化成 GM 直读、单 scratch 或单缓冲流水。具体要求：

1. `TileMmadTla` 不手传显式 `m/n/k`，实际形状通过 L0A/L0B/L0C tensor layout 表达，调用点只传 `initC/unitFlag`。
2. `K` 使用独立 L1 resident ping/pong，遇到同一 chunk 内多个 value head 共享同一个 key head 时复用已驻留的 `K_blk`。A5 BF16 `V=128` 在首次 L1A->L0A 后归还 L1 resident，并把 K 保持在 L0A slot0，直到同组最后一个 value head 的 `K @ dh` MMAD 完成。
3. `W` 使用独立 L1 resident ping/pong；A5 `qgDt` 预留四块 L1A slot，`g` 按 headOffset 映射，`gK` 按当前 window 内 `hk` 首 head 映射和共享；`d_o`、公开输出 `dh/dv2` 使用 L1B scratch ping/pong。当前 A5 L1 总占用在 `V=128` 时为 320KiB，在 `V=256` 时为 384KiB，均在 512KiB 预算内。A2 `qgDt` 使用 workspace 到 L1A 的路径。
4. L0A/L0B 使用 ping/pong slot 和独立 `M_MTE1/MTE1_M` 事件闭环；A5 BF16 `V=128` 的 K resident 固定使用 slot0，`termQ` 使用 slot1，下一组 K 在 slot0 收到最后一个共享 head 发布的 free 后覆盖；`V_DIM=256` 时 K 方向按 `64` 拆 L0 MMAD 分片，中间分片使用 `unitFlag=0b10` 累加，最后分片使用 `0b11` 完成 copyout 语义。
5. L0C 按架构容量启用 ping/pong：`V=128` 当前最大 tile 为 `128x128xfp32`，可开双槽；`V=256,chunk=128` 最大 tile 为 `128x256xfp32`，单槽已占满 A2/A3 L0C，只能保持单槽并在文档中说明容量原因。
6. 同一个 L0C 结果只 Fixpipe 一次。调试输出不能和正式 workspace 双写同一 L0C。

后续性能优化在上述 tile resident/双缓冲框架内继续推进，可评估跨 4-head window 的 K 生命周期或其它模板分支的 L0A 复用，并保持各 slot 的 ready/free 闭环。

## 9. TilingKey 设计

按全局规范，`tilingKey` 只使用 Ascend C 模板化方案，不手写 bit 编码或 `TILING_KEY_IS` 手动分发。

当前代码使用的模板维度：

```cpp
#define TPL_BF16 10
#define TPL_FP16 20
#define TPL_FP32 30

ASCENDC_TPL_ARGS_DECL(ChunkGatedDeltaRuleBwdDhu,
    ASCENDC_TPL_DTYPE_DECL(D_T_Q, TPL_BF16, TPL_FP16),
    ASCENDC_TPL_DTYPE_DECL(D_T_G, TPL_BF16,
        TPL_FP16, TPL_FP32),
    ASCENDC_TPL_UINT_DECL(V, 1, ASCENDC_TPL_UI_LIST, 128, 256),
    ASCENDC_TPL_UINT_DECL(USE_GK, 1, ASCENDC_TPL_UI_LIST, 0, 1),
);
```

当前组合数：

```text
2(q dtype) * 3(active gate dtype) * 2(V) * 2(USE_GK) = 24
```

`USE_GK=0` 表示当前 launch 传入的是 `g`，`D_T_G` 为 `g` dtype；`USE_GK=1` 表示当前 launch 传入的是 `gK`，`D_T_G` 为 `gK` dtype。定长/变长通过 `cu_seqlens` 是否为空在统一 chunk offset helper 内处理，不作为模板维度。`BT/BV/state_v_first/GATE_EXP_MODE` 暂不进第一版 key；如果后续支持多策略调优，必须新增模板参数，不建议通过运行时 `if` 混在热路径里。

## 10. TilingData 设计

建议新增：

```cpp
#pragma pack(push, 8)
struct ChunkGatedDeltaRuleBwdDhuTilingData {
    int64_t B = 0;
    int64_t HK = 0;
    int64_t HV = 0;
    int64_t T = 0;
    int64_t K = 0;                  // fixed 128
    int64_t V = 0;                  // 128 or 256
    int64_t HRatio = 0;             // HV / HK
    int64_t chunkSize = 0;          // 64 or 128
    int64_t chunkNumForT = 0;       // ceil_div(T, chunkSize)
    int64_t totalChunkNum = 0;      // dh 第三维；fixed: chunkNumForT; varlen: len(chunk_indices) / 2
    int64_t chunkTaskNum = 0;       // chunk 总量统计；fixed: B * chunkNumForT; varlen: totalChunkNum
    int64_t seqNum = 0;             // fixed: B; varlen: len(cu_seqlens) - 1
    int64_t headWindowNum = 0;      // ceil_div(HV, 4)
    int64_t taskNum = 0;            // seqNum * headWindowNum
    int64_t isVariable = 0;
    int64_t hasDh0 = 0;
    int64_t hasGk = 0;
    int64_t workspaceElemsPerSubBlock = 0;
    int64_t qgWorkspaceOffset = 0;
    int64_t stateWorkspaceOffset = 0;
    int64_t dvStateWorkspaceOffset = 0;
    int64_t termQWorkspaceOffset = 0;
    int64_t dv2WorkspaceOffset = 0;
    int64_t termWWorkspaceOffset = 0;
    int64_t qgWorkspaceElems = 0;
    int64_t stateWorkspaceElems = 0;
    int64_t dvStateWorkspaceElems = 0;
    int64_t termQWorkspaceElems = 0;
    int64_t dv2WorkspaceElems = 0;
    int64_t termWWorkspaceElems = 0;
    float scale = 1.0f;
};
#pragma pack(pop)
```

Host tiling 流程：

```text
1. 校验 q/k/w/d_o/dv 以及可选 g/gK 的形状和 dtype；`g` 和 `gK` 必须刚好传一个，`g` 必须为 `[B,HV,T]`，`gK` 必须为 `[B,HV,T,K]`
2. 校验 K/V/BT/HV%HK
3. 判断 fixed/varlen
4. 计算 `seqNum/headWindowNum/chunkNumForT/totalChunkNum/chunkTaskNum`
5. `taskNum = seqNum * headWindowNum`，`blockDim = platformAicCoreNum`
6. 若 `hasDh0`，按 `dh0Elems * sizeof(q dtype)` 计算 `dh0` 一次性 Fill 清零切分：参与清零 Vector 核数、非尾核元素数和尾核元素数；非尾核清零字节数必须 512B 对齐
7. 计算 `qg/dvState/termQ/termW` 的 workspace elems 和 offset，`stateDt/dv2Dt` elems 固定为 0
8. `workspaceSize = sysWorkspaceSize + blockDim * 8 * workspaceElemsPerSubBlock * sizeof(q dtype)`
9. 使用 GET_TPL_TILING_KEY(...) 生成模板化 tilingKey
```

## 11. 地址计算

### 11.1 fixed length

输入地址按 head-major `[B,H,T,D]`：

```text
q/k offset:
    ((b * HK + hk) * T + token) * K

w offset:
    ((b * HV + hv) * T + token) * K

d_o/dv/dv2 offset:
    ((b * HV + hv) * T + token) * V + vBase

g offset:
    (b * HV + hv) * T + token

dh offset:
    (((b * HV + hv) * chunkNumPerB + chunkIdx) * K + kIdx) * V + vBase

dht/dh0 offset:
    ((b * HV + hv) * K + kIdx) * V + vBase
```

### 11.2 varlen

变长模式下 `B=1`，token 使用全局 packed token index：

```text
token = cu_seqlens[seqIdx] + localToken
chunkFlatIdx = chunkTaskIdx
```

地址：

```text
q/k offset:
    ((0 * HK + hk) * T + token) * K

w offset:
    ((0 * HV + hv) * T + token) * K

d_o/dv/dv2 offset:
    ((0 * HV + hv) * T + token) * V + vBase

g offset:
    (0 * HV + hv) * T + token

dh offset:
    (((0 * HV + hv) * totalChunkNum + chunkFlatIdx) * K + kIdx) * V + vBase

dht/dh0 offset:
    ((seqIdx * HV + hv) * K + kIdx) * V + vBase
```

## 12. 同步设计

当前 stage1/stage2 同步状态：

1. Kernel 主流程不区分定长/变长两套分支，Cube/Vector 都通过相同 task 映射拿 chunk/head。
2. 当前只使用两套 raw CrossCore flag：`vecToCube=2`、`cubeToVec=4`；四个 head 复用这对 flag，但每个 head 使用当前 window 内独立的 workspace slot。
3. 负责当前 head 的 subblock 写完 `dh/qgDt` 后，两个 subblock 都参与当前 headOffset 的 `CrossCoreSetFlag<0x2, PIPE_MTE3>(vecToCube)`；非负责 subblock 只做同步配平。
4. Cube stage1 等待 `vecToCube` 并计算 `dvState/termQ`；bf16 CV 路径按 slot 等待矩阵 CV free，Fixpipe 写入后发布 CV ready，完成当前 head 后使用 `CrossCoreSetFlag<0x2, PIPE_FIX>(cubeToVec)` 通知 Vector。
5. Vector stage1 等待 `cubeToVec` 后，负责当前 head 的 subblock 按 token row tile 读取 `dvState/dv` 并写公开 `dv2`，另一个 subblock 只参与 `cubeToVec/vecToCube` 配平。
6. Cube stage2 等待 `vecToCube` 后计算 `termW`；A5 bf16 在首个 CV tile 前使用 `CrossCoreSetFlag<0x2, PIPE_FIX>(cubeToVec)` 发布 stage2 可启动信号，再按 row tile 等待 free、Fixpipe 写矩阵 CV 并发布 ready；A2 和 fp16/fp32 在 workspace copyout 完成后发布 `cubeToVec`。
7. Vector stage2 等待 `cubeToVec` 后，负责当前 head 的 subblock 按 K row tile 读取 `termQ/termW` 和 fp32 workspace state，并更新 fp32 `dhState` workspace carry；A5 bf16 的真实 `termW` tile 可见性由矩阵 CV ready/free 保证。
8. Vector 侧内部使用 `MTE2_V/V_MTE2`、`V_MTE3/MTE3_V` 和 state 专用 `MTE3_MTE2` 事件保护 UB 搬入、计算和写回；gate 系数保持在 UB 中由 `Brcb+Mul` 等向量指令消费，不走 V 到 S 的标量读取路径。

完整递推版本的通用 AIC/AIV 同步目标继续使用两套 raw flag：

```text
vecToCubeFlag: Vector -> Cube
cubeToVecFlag: Cube -> Vector
```

每个 chunk 的同步顺序严格为：

```text
stage_0:
Cube_PrepareInputs
Vector_SaveStateQgAndDecay
Set vecToCube

stage_1:
Wait vecToCube
Cube_DvStateAndTermQ
Set cubeToVec

Wait cubeToVec
Vector_Dv2
Set vecToCube

stage_2:
Wait vecToCube
Cube_TermW
Set cubeToVec

Wait cubeToVec
Vector_UpdateState
```

`InitDHTile` 和 `StoreDh0` 只在 Vector 侧工作，不参与 Cube/Vector 跨核同步，也不算 stage。

完整递推目标中，同一个 `(seq,hv,vBase)` 的 chunk 因 `dH` carry 必须倒序串行；不同 head 使用独立 workspace slot。当前代码按 `coreIdx * 8 + windowStartSlot + headOffset` 使用 prepare 风格的 8 个 per-core head slot，A2/A5 Vector 在 4-head window 内按 `headOffset % subBlockNum` 交替承包完整 head。若后续引入同一 head 内 chunk 级流水、V tile 并行或四个 head 的 AIC/AIV 并行流水，必须重新设计 workspace ring 和 ready/free 计数。

Cube 内部同步遵循：

1. MTE2 写 L1 后，MTE1 读 L1 前使用 `MTE2_MTE1`；不要在 GEMM 入口连续等待两个 operand 的 ready，应把 ready wait 放到各自 L1->L0A/L0B copy 前，让 MTE1 可以先消费已 ready 的一路，同时 MTE2 继续准备另一路。
2. MTE1 写 L0A/L0B 后，M pipe MMAD 前使用 `MTE1_M`。
3. M pipe 写 L0C 后，Fixpipe copy 前使用 `M_FIX`。
4. Fixpipe 完成后，复用 L0C 前使用 `FIX_M`。
5. 若 Fixpipe 写 GM 后同一 Cube 再读该 GM，必须额外使用 `FIX_MTE2`。第一版没有这种同 Cube 写后读依赖。
6. L1 resident (`K/W`) 与 L1 scratch 使用不同 buffer 区域；resident/scratch 的 `MTE1_MTE2` free flag 应在对应 operand 最后一次 L1->L0 搬运完成后尽早恢复，不要拖到整个 GEMM/Fixpipe 结束。scratch ping/pong 只承接短生命周期 operand。
7. L0A/L0B 每个 ping/pong slot 必须有独立 free/ready 事件；L0C 双槽只在容量允许时启用，单槽场景也必须完整闭合 `FIX_M/M_FIX`。

Vector 内部同步遵循：

1. MTE2 写 UB 后，V pipe cast/compute 前使用 `MTE2_V`。
2. V pipe 写 output UB 后，MTE3 copy out 前使用 `V_MTE3`。
3. 当前代码中 q/dv 输入和 gate 输入各自使用独立 ping-pong 事件组；即使当前调用顺序较直，也不把 q/gate/dv 串行作为同步设计约束。
4. 同一 UB 上 V pipe 写后再读，必要时使用 `PipeBarrier<PIPE_V>()`。

## 13. 实现步骤建议

已完成 stage0：骨架和纯功能

1. 新增正式算子目录：
   - `op_kernel/chunk_gated_delta_rule_bwd_dhu.cpp`
   - `op_kernel/chunk_gated_delta_rule_bwd_dhu_struct.h`
   - `op_kernel/chunk_gated_delta_rule_bwd_dhu_common.h`
   - `op_kernel/chunk_gated_delta_rule_bwd_dhu_cube.h`
   - `op_kernel/chunk_gated_delta_rule_bwd_dhu_vector.h`
   - `op_host/op_tiling/...`
   - `op_host/op_api/...`
2. 接入 fast-kernel launch 示例路径当前已有 wrapper。
3. 第一版实现 `g/gK` 二选一的参数分支、`dht=None`、`dh0` 不输出或只做零初始路径。
4. 跑通现有 `test_npu_chunk_gated_delta_rule_bwd_dhu.py` 的 fixed 和 varlen smoke。

当前 stage1/stage2：Cube/Vector 交换 `dvState/termQ/termW`，并复用公开输出 `dh/dv2`

1. workspace 布局保留 `qgDt/dhState/dvState/termQ/termW` 五段，slot 粒度对齐 prepare 的每核 8 个 per-head slot：`coreIdx * 8 + windowStartSlot + headOffset`；A5 的 `qgDt` 通过四块独立 L1A slot 直传，bf16 `dvState/termW` 通过共用矩阵 CV 双缓冲直传，因此三段在 A5 bf16 仅保留布局，运行时 workspace 数据为 `dhState/termQ`。其中 `dhState` 段固定 fp32，`stateDt/dv2Dt` 不占 workspace。
2. Cube 等 `dh/qgDt` ready 后，用 tile 级 resident/双缓冲计算 `dvState = K @ dh[chunk]` 和 `termQ = qgDt.T @ d_o`；A5 `g` 分支从当前 `headOffset` 的 L1A slot 消费 qg，`gK` 分支从当前 window 内 `hk` 首 head 对应的共享 slot 消费，省去 UB->GM workspace 和 GM->L1，并减少 GVA `gK` 重复 q 搬运、cast 和 UB->L1；A2 从 qg workspace 搬入。`K` 使用独立 L1 resident ping/pong，并在 shared-key heads 间复用；A5 BF16 `V=128` 还将 K 保持在 L0A slot0 供同组 heads 复用，`termQ` 使用 slot1。
3. Vector 等 `dvState` ready 后读取 `dvState/dv`，`g` 分支乘 token gate，`gK` 分支直接相加，只写公开 `dv2`；A5 bf16 通过矩阵 CV 消费 `dvState`。
4. Cube 等 `dv2` ready 后，用 tile 级 W resident/双缓冲计算 `termW = w.T @ dv2`；A5 bf16 通过复用的矩阵 CV 把 `termW` 交给 Vector，Vector 更新 fp32 workspace `dhState`。
5. 保持 fixed/varlen 在同一个 `GetChunkInfo` helper 内处理，kernel 主流程不拆两套分支。
6. DHU Cube 禁止 `BlockMmadTla`、`DeviceGemm` 等 block 级接口；所有 GEMM 只允许 `CopyGmToL1A/B`、`CopyL1ToL0A/B`、`TileMmadTla`、`CopyL0CToGm` 这套 tile 级路径。

下一阶段：补齐完整语义

1. 支持 `dht` 作为初始 final state gradient；按 AIV row tile 分片后的 `dhState` 已经在 Vector UB 中跨 chunk 倒序常驻，后续只补真实初始化来源。
2. 写真实 `dh/dh0`；如果 `h0` 非空，输出 `dh0[N,HV,K,V]` fp32，并修正 fast-kernel launch meta 中 `dh0` 的 shape/dtype。
3. 支持 `state_v_first=true` 时，需要输出和 state 地址模板化。

后续性能优化

1. 评估 `termQ` 的 A5 L0C->UB CV 短生命周期传递；`dvState/termW` 已完成共用 CV 双缓冲直传。
2. 在当前 tile resident/双缓冲框架内继续做 head 间 overlap 或更多 GM 输入预取；不能退回 block 级接口。
3. 当 `seqNum * ceil_div(HV, 4)` 太小导致核利用率不足时，评估可证明安全的 `V tile` block 分核增强版本。该增强版必须保持每个 `(seq,hv,vTile)` 的 chunk 倒序串行。

## 14. 验证要求

精度测试：

通用规则：完整精度脚本必须先调用融合 kernel 并 `torch.npu.synchronize()`，打印 `phase_time kernel_sync=...` 后再生成 CPU golden 和做全量比对；如果该行没有出现，优先按 kernel 未返回/疑似卡死定位。全量比对必须覆盖 `dh/dv2` 所有元素，不允许抽样。

1. fixed length:
   - `B=1/2`
   - `HK==HV` 和 `HV>HK`
   - `T` 覆盖整 chunk 和尾 chunk
2. varlen:
   - `B=1`
   - 多 seq，不同 seq 长度不同
   - 每个 seq 覆盖 1 个 chunk、多 chunk、尾 chunk
3. dtype:
   - q/k/w/d_o/dv: fp16、bf16
   - g: fp16、bf16、fp32
   - gK: fp16、bf16、fp32
4. shape:
   - `BT=64/128`
   - `V=128/256`
   - `g=[B,HV,T]` 与 `gK=[B,HV,T,K]` 各覆盖，且每个 case 只传其中一个
   - 反向参数检查覆盖同时传 `g/gK` 和二者都不传
5. 可选状态:
   - `dht=None`
   - `dht` 非空
   - `h0` 非空时校验 `dh0`

大 case 泛化精度规范使用 `test_npu_chunk_gated_delta_rule_bwd_dhu_stage0_accuracy.py --large-suite`。主体固定覆盖 `{非GVA,GVA} x {V=128,V=256} x {无dh0,dh0} x {g,gK} x {varlen,fixed}` 五个维度的笛卡尔组合，共 32 条 bf16 case；另补 fixed 非 GVA `V=128/no-dh0/g` 和 varlen GVA `V=256/dh0/gK` 两条 fp16 case，总计 34 条。`g/gK` 使用 fp32。非 GVA 形状为 `B=1,HK=HV=16,T=4096,K=128,chunk=64`，GVA 形状为 `B=1,HK=8,HV=16,T=4096,K=128,chunk=64`；varlen 非 GVA `cu_seqlens` 使用固定 seed `20260805` 切 3 seq，varlen GVA 使用固定 seed `20260807` 切 5 seq，fixed 不传 `cu_seqlens/chunk_indices`。总控进程为每个 case 启动完整 worker，worker 内先跑 kernel，再只加载当前 case 的 CPU dual golden 并执行 CT，退出后释放 CPU/NPU 内存。该 suite 完整比较 `dh/dv2/dh0`，并显式传入 actual 输出 dtype。

CPU golden 较慢的机器可使用分段验证：目标 NPU 机器逐 case 运行 `--kernel-only --kernel-artifact <case>.pt` 保存 actual，CPU 较快机器用同一脚本、同一 `--large-suite` 和同一 base seed 通过 `--actual-artifact-dir <dir>` 加载 actual 后生成 CPU dual golden 并执行 CT 比较。脚本在 `--case` 模式下按原始 case index 偏移 seed，单 case 和全量 suite 的输入保持一致。

精度失败处理：

1. 先单 case 保存 `dh/dv2/dh0`。
2. 判断是否是结构性错误：chunk 顺序、chunkTaskIdx/chunkFlatIdx、hv->hk、gate last index、尾 chunk mask、V tile 尾列。
3. 结构性错误必须修 kernel，不通过缩小输入 range 或调阈值规避。

性能测试：

1. 使用 `msprof` 的 `op_summary` 中 `Task Duration(us)` 作为性能结论。
2. 默认只跑一轮典型 case。
3. 对比 Python 小算子链路或上游等价链路时，记录总耗时，不用 Python wall time 做正式结论。
4. 变长性能 case 的 `cu_seqlens` 用固定 seed 随机生成，记录 seed、长度 min/max 和 `chunk_pairs`。
5. `B=1,H=32,T=65536,K=V=128,chunk=64,varlen 64 seq,bf16` 当前已把 canonical `chunk_indices` 主路径改为 O(1) 下标校验，删除 `dvState/termQ/termW` 预清零，将 Vector 单行/隔行搬运改为 UB 自适应 `vecRow` 连续多行 tile，去掉 `dh -> stateDt`、`dv2 -> dv2Dt` 两路 workspace 镜像写，让 `g` 分支的 `gRaw/gateFactor/dvGateFactor` 按 headOffset 驻留 UB，避免 stage1 二次搬入 `g[M]` 或重复计算 `exp(g_last-g_t)`；A2/A5 Vector 在 4-head window 内按 `headOffset % subBlockNum` 交替承包完整 head，减少同一 head 内 gate resident 的重复生成；后续性能优化重点应继续转向减少剩余 workspace 行级写回和改善 Vector/Cube 中间结果交换。

内存检查：

1. 疑似 UB/L1/GM 冲突时使用 MindStudio Sanitizer。
2. race 类优先 `racecheck`。
3. 越界类优先 `memcheck`。
4. 未初始化读取优先 `initcheck`。
5. 运行前必须确认实际命中 sanitizer 版本对象。

## 15. 完整递推关键检查清单

1. `chunk` 循环必须倒序：`chunkNum - 1 -> 0`。
2. 分核不能包含 `chunkIdx`；一个 task 必须覆盖 `hvBase..hvBase+3` 的 4-head window，尾部按 `headCnt=min(4,HV-hvBase)` 截断。
3. `hv -> hk` 使用 `hk = hv / (HV / HK)`。
4. `dh` 保存的是进入当前 chunk 前的 `dH`，不是更新后的 `dH`。
5. `dv2` 使用更新前的 `dH` 计算。
6. 传 `g` 时，`dv2` 使用 `E_g(g_last - g_t)`，旧 `dH` 使用 `E_g(g_last)` 做统一 decay，`qg = q * E_g(g_t)` 使用当前 chunk 每个 token 的 gate。
7. 传 `gK` 时，只对旧 `dH` 做 `dH[k,:] *= E_gK(gK_last[k])` 的 K 行 decay；`E_gK(gK_last[K])` 必须由向量指令生成，`gK` 不参与 `dv2` token gate，也不参与 `qg`。
8. `g` 和 `gK` 是互斥关系，二者都传或都不传必须在 wrapper/aclnn/tiling 层拒绝。
9. 尾 chunk 的无效 token 不参与 GEMM、gate、写回。
10. varlen 的 `dh` 使用 flattened `chunkFlatIdx`，不是 local chunkIdx。
11. `dh0` 如果输出，shape 应为 `[N,HV,K,V]`，dtype 应为 fp32。
