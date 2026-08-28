# ChunkFwdH 设计

## 1. 目标与边界

`ChunkFwdH` 是独立的 Ascend C 算子，符号为 `ChunkFwdH`、`aclnnChunkFwdH` 和
`fla_npu.ops.ascendc.chunk_fwd_h`。它不复用、不修改 `ChunkGatedDeltaRuleFwdH`，也不提供
legacy `torch.ops.npu` 兼容入口。

当前支持 A2、A3、A5，固定 `K=V=128`、`chunk_size=64`，`k/w/u` 为 BF16。gate 和
state 分别支持 BF16/FP32，`state_v_first=true/false` 均由 kernel 原生处理。

## 2. 计算语义

记 `E(x)=exp(x)`，`use_exp2=true` 时为 `exp2(x)`。每个 value head、每个 chunk 执行：

```text
H_c = cast_BF16(R_c)
Pacc_c = W_c @ H_c
P_c = cast_PType(Pacc_c)  # StateT=BF16 时 PType=BF16，否则为 FP32
V_new_fp32_c = fp32(U_c) - fp32(P_c)
V_new_c = cast_BF16(V_new_fp32_c)

g-only:
  V_new_g_c[i,:] = cast_BF16(E(g_last-g_i) * V_new_fp32_c[i,:])
  D_c = k_raw_c^T @ V_new_g_c
  R_next = E(g_last) * R_c + D_c

gk-only:
  D_c = kg_c^T @ V_new_c
  R_next[k,v] = E(gk_last[k]) * R_c[k,v] + D_c[k,v]
```

最终 chunk 不请求 `final_state` 时，只生成当前 `H` 和 `V_new`，跳过 Stage2/Stage3。

## 3. head round 规划

规划在 chunk 循环前完成。每个 AIC round 最多四个 head：AIV0 处理 round head 0/2，
AIV1 处理 1/3；两个 AIV 各有 local slot 0/1。

g-only 先按 `group_size=HV/HK` 划分共享 raw K 的 value-head 组：

- `HK:HV=1:3`：每轮三个 value head，一个 kg slot，只读一次 raw K。
- `HK:HV=1:2`：每轮四个 value head，两个 kg slot，各 raw K 只读一次。
- `HK:HV=1:6`：拆为 4+2 两轮；第二轮重新读取同一个 raw K，不跨 round 保留。

gk-only 每个 value head 对应自己的 prepared kg；本轮按 active head 数读取 1..4 份。
`FwdHHeadRoundPlan` 显式记录 `head -> hv -> kh -> kgSlot -> aiv -> localSlot`，Stage2 只遍历
`requiredKhCount`，不预留或加载 16 份 kg。

## 4. 分阶段实现

### S-1：FP32 初态转换

仅 `initial_state` 为 FP32 时执行 `H0=cast_BF16(initial_state)`。A2/A3 逐 tile 处理；A5
每个 head 使用一次完整 RegBase VF，并为两个 local head 分配独立 FP32 input/BF16 output
bank，以“预取当前 head、计算前一 head”的循环实现 ping-pong。全部 active H MTE3 drain
后统一发布本 phase 的 `H_READY`。

### Stage0：入口状态与第一次矩阵乘

AIC 将当前 `W[M,K]` 与 `H[K,V]` 从 GM 搬到四个 round-head L1 槽，随后执行
`P=W@H`。Stage0 不读取 kg/k_raw。tail chunk 先用 MTE2 `InitConstValue` 清零当前 W 槽，
再覆盖有效 ND 行；Cube M 取 `AlignUp(valid_tokens,16)`，最终只写真实 token 行。

- A2/A3：L0C 经 Fixpipe 写 P GM scratch，AIV 再以 MTE2 读取。
- A5：L0C 经 Fixpipe 直接写配对 AIV 的 local UB slot。

### Stage1：向量修正

AIV 读取 `U` 和 P，计算 `V_new`；g-only 同时生成带相对 gate 的 BF16 right，gk-only
直接以 BF16 `V_new` 作为 right。`V_new` 写 GM。right 必须经 MTE3 写 GM workspace，禁止
UB 直搬 L1。

A5 从 MTE2 后开始的全部向量计算由每个 head 一次 RegBase VF 完成。VF 只使用
`if constexpr` 模板分支和明确类型的 `for` 循环，并以两组 FP32 寄存器覆盖 128 列。

### Stage2：第二次矩阵乘

AIC 此时才读取本 round 实际需要的 kg/k_raw，并等待每个 head 独立的 `RIGHT_READY`，把
GM ND right 搬为 L1 Cube 输入。计算 `D=k^T@right`；`state_v_first=true` 时使用等价转置式
`D_physical=right^T@k`，直接生成物理 `[V,K]`。tail chunk 对当前实际 kg/k_raw 槽和各 head
right 槽分别清零后再覆盖有效数据，Cube K 取 `AlignUp(valid_tokens,16)`。

- A2/A3：L0C 经 Fixpipe 写 D GM scratch，AIV MTE2 读取。
- A5：L0C 经 Fixpipe 直接写配对 AIV UB。

### Stage3：递推状态更新

AIV 用 FP32 算术执行 `R_next=decay*R+D`，按 StateT 保存 BF16 或 FP32 rolling state，并按需
写下一 chunk 的 H 或 final_state。A5 使用独立 RegBase VF，不调用 A2/A3 向量实现。

## 5. 存储布局

AIC L1 固定分区：W `[0,64) KiB`，保留空洞 `[64,128) KiB`，H/right `[128,256) KiB`，
kg `[256,320) KiB`。kg 区最多四个 16 KiB slot；每个 round 只占用 `requiredKhCount` 个。

A5 每个 AIV 有两个 64 KiB local slot，Stage0 的 P 与 Stage2 的 D 按生命周期复用；BF16
state、FP32 state、BF16 work 和 gate 区保持固定地址。A2/A3 使用两个 32 KiB tile local slot
和两个 32 KiB BF16 state slot，P/D 使用每核每 round-head 独立的 GM scratch。

## 6. 同步协议

每个 local slot 分别维护 `P_READY/P_FREE`、`D_READY/D_FREE`、
`RIGHT_READY/RIGHT_FREE`、`H_READY`。ready 由真实生产 pipe 发布，free 由最后消费者发布；
同一 slot 的事件复用前必须完成上一代 wait。

Stage 内按核内 head id 统一写一套流程，`headId&1` 选择 ping/pong L0 slot。当前 slot 的
MTE2 完成即可启动该 slot 的 VEC/Cube，不等待另一 slot；Cube->Fixpipe 和 VEC->MTE3 同理。

A2/A3 在无初态、多 chunk 场景为第二个 chunk 的 P scratch 预置一次 free credit，因为首
chunk 没有 Stage0/P；后续 credit 由前一 chunk Stage1 产生。round 结束时 P 与 D 两条独立
scratch 链分别回收，不能用互斥分支漏掉其中一条。

跨 head round 使用双向收口：两个 AIV 等本轮 MTE3 全部完成后发布 `ROUND_DONE`；AIC 收到
两份完成信号并回收 P/D/right 后才回 ACK。下一 round 的 kg/H/W MTE2 必须等待 ACK，因而
不会与上一 round 的未完成写回交叠。
调度上将同一 sequence 的全部 head-round 绑定到同一核并顺序执行；不同 sequence 才允许在不同核并行，
因此上述 ACK 是实际的 round 边界，而不是依赖 block 启动顺序推断。

## 7. 变长序列

变长模式要求 BNSD 容器 `B=1`。`cu_seqlens` 从 0 开始、以 T 结束且严格递增；
`chunk_indices` 若存在，必须是 sequence-major canonical `(seq_id, chunk_id)`。不同 sequence
分配为独立 work unit，state/final_state 的首维是 sequence 数，H 的 chunk 维使用全局 chunk
前缀，GM 输出区间互不重叠。

## 8. TilingData

`ChunkFwdHTilingData` 只保存 host 已校验并由 kernel 直接消费的数据，不在 tiling 中保存每个
head round 的展开数组。kernel 在进入 chunk 循环前，按 `kNumHead/vNumHead` 生成
`FwdHHeadRoundPlan`，因此每个 round 的 active head 数、实际 Hk 数和 head-to-kg-slot 映射
不会随 chunk 改变。

| 字段 | 语义 |
| --- | --- |
| `batch` | dense 时为逻辑 batch 数，varlen 时为 sequence 数 |
| `seqlen` | k/w/u 容器的 token 维长度 |
| `kNumHead` / `vNumHead` | Hk/Hv 数；g-only 要求 `vNumHead % kNumHead == 0` |
| `kHeadDim` / `vHeadDim` | 当前均固定为 128 |
| `chunkSize` | 当前固定为 64 |
| `useInitialState` | 是否读取 initial_state |
| `storeFinalState` | 是否生成并写 final_state |
| `dataType` | k/w/u dtype 枚举，当前为 BF16 |
| `gDataType` | g 或 gk 的 dtype 枚举 |
| `stateDataType` | rolling/final state dtype 枚举；无初态且无 final_state 时为 FP32 |
| `isVariedLen` | 是否启用 varlen 调度 |
| `shapeBatch` | dense 的物理 batch；varlen 固定为 1 |
| `tokenBatch` | varlen 的 sequence 数；dense 固定为 1 |
| `useG` / `useGk` | gate 模式，二者严格一真一假 |
| `useExp2` | gate 指数函数是否使用 exp2 |
| `stateVFirst` | state 物理布局为 `[V,K]` 还是 `[K,V]` |
| `vWorkspaceOffset` | A2/A3 P 的 FP32 GM scratch，形状为 `[AIC,4,64,128]` |
| `vUpdateWorkspaceOffset` | Stage1 BF16 right 的 GM workspace，形状为 `[AIC,4,64,128]` |
| `kDecayWorkspaceOffset` | FP32 rolling state 的 GM workspace，形状为 `[AIC,4,128,128]` |
| `hWorkspaceOffset` | A2/A3 D 的 FP32 GM scratch，形状为 `[AIC,4,128,128]` |
| `numSeqWorkspaceOffset` | varlen sequence 前缀辅助区，长度为 `tokenBatch+1` |
| `numChunksWorkspaceOffset` | varlen chunk 前缀辅助区，长度为 `tokenBatch+1` |

workspace 的各段按 512 Byte 对齐，并在 CANN lib-api workspace 之后额外保留运行时安全区。
A5 的 P/D 走 L0C->AIV UB，不消费对应的 P/D GM scratch；为保持跨架构统一 tiling，offset
仍由 host 生成，但 A5 kernel 不访问这些地址。
