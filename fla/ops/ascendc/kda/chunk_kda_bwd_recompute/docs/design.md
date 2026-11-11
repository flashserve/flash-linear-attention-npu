# ChunkKdaBwdRecompute 设计

方案设计规则版本：`V1`。公开接口与支持范围以 [api.md](api.md) 为准。

## 阶段边界

本算子只恢复反向所需的 chunk-local 检查点，不计算 chunk 间状态、不产生
`h/v_new/final_state`，也不做 Q/K L2 归一化或 beta sigmoid。调用链中的位置：

```text
ChunkKdaBwdRecompute → ChunkFwdH → ChunkKdaBwdPrepare → Dhu → Finalize
```

输入侧消费前向留下的 `A`（三角系统解）以及 raw `q/k/v/g/beta`（及可选
`A_log`/`dt_bias`）。输出 `gk/w/u/qg/kg` 供后续 FwdH / Prepare / Finalize
消费。整体编排约束见
[`../chunk_kda_bwd/docs/A2A3_design.md`](../chunk_kda_bwd/docs/A2A3_design.md)。

## 数学与精度

对每个 value head、每个长度 `M ≤ 64` 的 chunk：

```text
x      = g + dt_bias
gate   = lower_bound * sigmoid(exp(A_log) * x)     # use_gate_in_kernel
gk     = cumsum(gate) * RCP_LN2                      # use_exp2；RCP_LN2 = 1/ln(2)
e2     = exp2(gk)
qg     = q_hv * e2
kbg    = k_hv * beta[..., None] * e2
vb     = v * beta[..., None]
kg     = k_hv * exp2(gk[M-1] - gk)
w      = bf16(A[:M,:M] @ kbg[:M])
u      = bf16(A[:M,:M] @ vb[:M])
```

`q/k` 按 `hvPerHk = HV/HK` 做 GQA 展开到 `HV`。A5 快路径在 RegBase VF 内
把 safe-gate、cumsum、`qg/kg/kbg/vb` 融合；`w/u` 由 Cube 以 BF16 输入、
FP32 累加后 Fixpipe 为 BF16。尾 chunk 只读有效 `M` 行/列；`M=1` 时 Cube
侧将物理 MMAD 的 M 补到 16，Fixpipe 仍只写有效行。

CPU / ATK 标杆与上述公式一致，并对 `A/kbg/vb` 按 Cube 路径先量化到 BF16
再乘，避免把 BF16 GEMM 误差算成实现错误。

## 分核与任务映射

任务单元是 `(chunk, value_head)`，总数 `chunkNum * HV`。Host 按平台
AIC 数与任务数取 `usedAic`，`scheduleMode = 1`，Mix **1 AIC : 2 AIV**。

- A5：`CoreTaskRange` 把任务切成连续 chunk-major 区间，避免
  `task += coreNum` 的跨度跳转打断四 head 旋转。每个 Mix 核内按
  `kHeadRotate = 4` 成组；AIV 按 `h & 1` 独占奇偶 head（slot 0/1）。
- A2/A3：AIC 上限压到 20（完整 1:2 Mix 组）；Vector 按
  `taskIdx += usedAic` 循环，子块用 `vecTaskIdx % subBlockNum` 过滤。

平台查询若只回报 1～2 个 AIC，Host 回退到 `aivNum/2`、`GetCoreNum()`，
再不行落到 A5 的 28。

TilingKey 由 `GET_TPL_TILING_KEY(D_T_G, D_T_BETA, K_DIM)` 生成，覆盖
`g/beta` 的 BF16/FP32 组合与固定 `K=128`。Kernel 入口按模板参数实例化，
不做数值 key 二次分派。

## Stage 划分

### A5 快路径（`useGateInKernel && useExp2`）

Vector 走 `ProcessFastPingPong`：UB 双缓冲搬入 `g/q/k/v/beta`，RegBase
`FusedRecomputeChunk128Regbase` 写出 `gk/qg/kg` 到 GM，并把 `kbg/vb`
以 ND→NZ 形式写入 L1 ping-pong。Cube 驻留读取 `A`（L1 固定 8 KiB），
等待对应 slot 的 ready 后做两次 MMAD，Fixpipe 写出 `u` 再写 `w`。

| Stage | 核 | 本轮计算 | 阶段结果 |
| --- | --- | --- | --- |
| V0 | Vector | 搬入 token；safe-gate + cumsum；生成 `qg/kg`；打包 `kbg/vb` 到 L1 | 公开 `gk/qg/kg`；L1 上 `kbg/vb` ready |
| C1 | Cube | `u = A @ vb`（BF16×BF16→FP32） | Fixpipe `u` |
| C2 | Cube | `w = A @ kbg` | Fixpipe `w`；释放 L1 slot |

同一 chunk 上多个 head 可复用已装入的 `A`（`skipLoadA`），减少 GM→L1。
流式 GM（`q/k/v/g/beta/qg/kg/gk/w/u`）关闭 L2；`A` 保留 NORMAL，避免
AIC MTE2 成为瓶颈。

非 `useGate && useExp2` 时 A5 走 `ProcessFallback`：仍在片上完成 Vector
准备，但不同时启用 RegBase 融合快路径。

### A2/A3 路径

Vector 分两段，中间 `SyncAll`：

1. `ProcessVb`：写出 workspace 中的 `vb`。
2. `ProcessGateAndK`：gate cumsum、`qg/kg`，并把 `kbg` 写到 workspace。

Cube 从 GM workspace 读 `kbg/vb`，从输入读 `A`，写出 `w/u`。跨核用
mode `0x2` 集合 flag（与 A5 的 mode `0x4` 独立 slot 不同）。

## 存储布局

### A5 L1（每 AIC）

| 区域 | 偏移 | 容量 | 内容 |
| --- | --- | --- | --- |
| `A` | `0` | 8 KiB | BF16 `[64,64]`，可跨 head 复用 |
| `kbg` slot0/1 | `8 / 40` KiB | 各 32 KiB | BF16 `[64,128]` ping-pong |
| `vb` slot0/1 | `72 / 104` KiB | 各 32 KiB | BF16 `[64,128]` ping-pong |

合计约 136 KiB，低于 512 KiB L1。L0A/L0B 承载当前 MMAD 的 A/B tile；
两块 L0C 各约 `kL0CTileBytes`，供 `u`/`w` 与相邻 head 流水。

### Workspace

| 平台 | user workspace | 说明 |
| --- | --- | --- |
| A5 | 0（另加 SDK 系统 workspace） | `kbg/vb` 留在 L1，不经 GM 中转 |
| A2/A3 | `2 * B * HV * T * K * sizeof(BF16)` | `[kbg \| vb]` 连续两段 |

`vecRow` 由 Host 按 UB 预算（默认上限约 192 KiB）从 `chunk_size` 向下折半
选择，保证 Vector 侧多份 TQue/TBuf 同时存活。

### 同步（A5）

跨核 mode `0x4`：AIV1 的 flag id 为 base+16。Cube **只 wait 生产该 slot
的那一个 AIV**，不能对同一 head wait 两侧 flag（会死锁）；也不能在无
对应 Wait 的情况下预置超过硬件信用的 free（会卡死芯片）。

| flag | id | 方向 | 含义 |
| --- | --- | --- | --- |
| ready | 6 (+16×slot) | AIV → AIC | 当前 slot 的 `kbg/vb` 已在 L1 |
| free | 7 (+16×slot) | AIC → AIV | Cube 已消费，AIV 可覆盖该 slot |

核内 HardEvent 覆盖 MTE2↔V、MTE3↔MTE2、MTE1↔MTE2、M↔MTE1、FIX↔M。
空任务区间的核不发 flag。

## Host / 属性透传

OpDef 输出登记顺序为 `w/u/qg/kg/gk(optional)`。属性：

| 属性 | 类型 | 说明 |
| --- | --- | --- |
| `chunk_size` | int | 必须为 64 |
| `use_gate_in_kernel` | bool | L0 显式传入；独立 L2 由 `gkOut` 是否非空推导 |
| `use_exp2` | bool | 与快路径选择相关 |
| `lower_bound` | float | 以 bit 模式写入 tiling，避免 host/device 浮点填充差 |
| `input_layout` | string | 仅 `BNSD`/`NTD` |

变长：`isVariable` 由是否提供 `cu_seqlens` 决定；有变长时必须同时提供
`chunk_indices`，`B` 置 1，`chunkNum = len(chunk_indices)/2`。定长时
`chunkNum = B * ceil(T/64)`。

## 实现位置

| 路径 | 职责 |
| --- | --- |
| `op_host/` | OpDef、tiling、手写 aclnn / L0 |
| `op_kernel/arch35/` | A5 Vector RegBase、Cube Catlass MMAD、跨核 flag |
| `op_kernel/*.h`（非 arch35） | A2/A3 Vector/Cube 与 GM workspace 路径 |
| `op_kernel/chunk_kda_bwd_recompute.cpp` | Mix 入口按 `__CCE_AICORE__` 选择实现 |

验收与性能数字维护在
[`tests/atk/chunk_kda_bwd_recompute/README.md`](../../../../../../tests/atk/chunk_kda_bwd_recompute/README.md)。
