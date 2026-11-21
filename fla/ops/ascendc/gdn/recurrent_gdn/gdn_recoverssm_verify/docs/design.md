# GdnRecoverssmVerify Ascend 950PR&950DT系列产品设计方案

方案设计规则版本：V2

## 1. 目标

接口以 [api.md](api.md) 为唯一事实来源。本算子完整实现 GDN speculative window 的
验证递推、BF16 token 输出和 FP32 replay 记录，checkpoint 全程只读。

主模型 case：Qwen3.6-35B-A3B TP4，`HK=4, HV=8, K=V=128, L=5, B=12`，
主 case 有 96 个独立 `(request, HV)` work。先计算
`waves=ceil(Nwork/GetCoreNumAiv())`，再取
`blockDim=ceil(Nwork/waves)`，在不增加 wave 数的前提下减少启动核数。
性能目标：初版以正确性为先，目标为同一 case 的 `msprof Task Duration(us)` 不超过
`100 us`；输入准备和递推均纳入统计。

### 1.1 数据类型

| 范围 | 数据 | 数据类型 |
| --- | --- | --- |
| API 输入 | `A_log,dt_bias` | FP32 |
| API 输入 | `a,b,q,k,v` | BF16 |
| API 输入 | `checkpoint_state` | BF16 或 FP32 |
| API 输出 | `out` | BF16 |
| API 输出 | `replay_cache` | FP32 |
| Vector 计算 | Q/K norm、gate、beta、state、correction、dot、decay | FP32 |
| Cube 输入与累加 | 不适用；算法为逐 value-head 状态递推和向量点积 | 不适用 |
| 跨 Stage 交接 | 归一化 Q/K、V、gate/beta、FP32 state | FP32 UB resident |

### 1.2 理论性能上限分析

主 case 共 96 个 work、60 个有效 token、8 个 value head、`V*K=16384`。每 token-head
约执行 4 次 `V*K` 级向量运算，共约 31.5M 标量乘加；以 32 AIV、1.65 GHz 的非约束
估算仅用于判断目标量级，正式结论由 `msprof` 给出。

| 约束来源 | 工作量/数据量 | 峰值或带宽依据 | 理论最低耗时 | 重叠/依赖关系 | 证据 |
| --- | --- | --- | ---: | --- | --- |
| AIC/Cube | 无 Cube 工作 | 算法为 Matrix-Vector 和 rank-1 更新，AIC 不参与 | 0 us | 不占用 AIC 关键路径 | 公式和参数映射；A5 支持纯 AIV kernel |
| AIV/Vector | 约 31.5M 标量乘加，单 AIV 最坏连续 15 个 token step | 目标 A5 每 AIV 128-lane Vector，按运行时 `GetCoreNumAiv()` 和 `GetCoreMemSize()` 校验 | 8 us | 不同 work 跨核并行；同一请求 token 因 state 依赖串行 | work 数 96 大于典型 32 AIV，满足并行度 |
| GM/MTE | checkpoint、QKV、out、replay 合计小于 5 MiB | 按运行时平台带宽测量；单次流式搬运 | 0.5 us | MTE2 预取下一 work 初始 state，MTE3 与当前递推重叠 | 每 tensor 只搬入一次，tile 输出批量搬出 |
| 必要依赖链 | 每核至少 3 个 work；每 work 至多 5 个 token | 同请求同 HV 的 state 更新是严格链，不能跨 token 并行 | 12 us | 跨 work 并行，组内 token 串行 | `Nwork=96, blockDim=32, wave=3` |

理论最低耗时：`12 us`。
目标耗时：`100 us`。
理论可达性：`理论可达`。

## 2. 变量与任务映射

| 符号 | 含义 |
| --- | --- |
| `B` | `state_indices` 的请求行数 |
| `T` | packed token 总数，`query_start_loc[B]=T` |
| `HK/HV` | TP-local key/value head 数，`G=HV/HK` |
| `K/V` | 输入 key/value 宽度，均为 128 |
| `L` | `replay_cache.shape[2]` 给出的窗口容量，1 到 32 |
| `Nwork` | 独立递推任务数，`B*HV` |
| `blockDim` | 实际启动 AIV 数，`ceil(Nwork/ceil(Nwork/GetCoreNumAiv()))` |
| `slot` | 双缓冲槽，0/1，按跨 work 的全局 tile 顺序取反 |
| `tile` | token 预处理和输出 staging 的 4-token 组 |

```text
work = row * HV + hv
hk   = hv / G
bos  = query_start_loc[row]
eos  = query_start_loc[row + 1]
core = work % blockDim
wave = work / blockDim
tile = global_tile_index % 2
```

同一 AIV 按连续 `wave` 处理任务。四个以上 AIV 组成的相邻 work 同时执行；同一请求的不同
HV 也独立并行。只有同一 `(row,hv)` 内的 token 因 state 依赖保持顺序。state 使用三个完整
bank，按 `bank2(raw/初始) -> bank0 -> bank1 -> bank2` 轮转。

## 3. Stage 总览

| Stage | 执行单元 | 主要动作 | 并行与流水关系 | 结束条件 |
| --- | --- | --- | --- | --- |
| S0 | Vector | 读取当前 work 的 checkpoint、窗口 Q/K/V 和 gate，完成 FP32 转换、Q/K L2 norm、softplus、sigmoid | 多 work 跨 AIV 并行；三 state bank 支持当前 work 内 token0 从 bank2 raw 初始化 | 当前窗口准备量完整写入 UB |
| S1 | Vector | 按 token 顺序完成 decay、residual、state rank-1 更新、output dot 和 replay record | 多 work 跨 AIV 并行；MTE2 预取下一输入 tile，MTE3 搬出上一输出 tile，均与当前 token 递推流水 | 当前 work 的全部有效 token 完成且输出 tile 排空 |

### 3.1 芯片规则应用检查

| 规则 | 结论 | 设计证据/不适用原因 |
| --- | --- | --- |
| R01 | 已应用 | S0、S1 均为 Vector Stage；不使用 Cube，满足每 Stage 单一计算类型 |
| R02 | 已应用 | S1 依赖 S0 的归一化 Q/K、V 和 gate；因需要跨 work state 预取与 S1 长链重叠而保留拆分 |
| R03 | 已应用 | S0 用一次 VF 完成窗口预处理；S1 因 state 跨 token 严格依赖，每 token 一次递推 VF |
| R04 | 已应用 | state 通过 `V_MTE2` 事件预取；全局 `tileIndex` 跨 work 连续递增，使相邻 work 使用不同 ping/pong buffer，当前 MTE3 可与下一 work 的 MTE2 重叠 |
| R05 | 已应用 | 不涉及 Cube/Vector 直连；中间量保留在 UB，不使用 GM 中转 |
| R06 | 已应用 | 归一化 Q/K/V、gate 和 FP32 state 留在 UB；state bank 跨 token 保持原地址 |
| R07 | 已应用 | QKV tile、输出 tile、replay tile 使用 ping/pong；state 使用三个完整 FP32 bank，bank2 首次复用为 checkpoint raw |
| R08 | 已应用 | 第 4 章给出每 AIV 绝对 UB 地址和峰值；三 bank 与双 tile 缓冲峰值约 212 KiB，不超过 248 KiB |
| R09 | 已应用 | checkpoint、QKV 单次读取不使能 L2；当前 work 内重复读取的 gate 和 state resident 不经过 GM |
| R10 | 已应用 | softplus 只在阈值内计算 `exp(z)`，`z<=20`；归一化加 `1e-6`；page 和 token span 由调用方作为前置契约保证，kernel 不增加防御分支 |

## 4. Stage 详细设计

### S0 · Vector：窗口输入准备

```text
DataCopyPad checkpoint[state_idx, hv, :, :] into raw view of stateBank[2]
load a[token,hv], b[token,hv] for token in [bos,eos)
x = a + dt_bias[hv]
z = softplus_beta * x
sp = log1p(exp(z)) / softplus_beta   if z <= softplus_threshold
sp = x otherwise
decay = exp(-exp(A_log[hv]) * sp)
beta = sigmoid(b)
load q[0,token,hk,:] and k[0,token,hk,:]
qh = q / sqrt(sum(q*q) + 1e-6) * scale  if normalization enabled
kh = k / sqrt(sum(k*k) + 1e-6)         if normalization enabled
qh = q * scale otherwise
kh = k otherwise
load v[0,token,hv,:] and cast to FP32
```

S0 对零长度 token span 直接跳过，不读取 checkpoint，也不写 output/replay；非空 work
继续按调用方保证的正整数 page 读取数据。

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| `stateBank[0/1/2]` | UB | `UB[0,65536)` / `UB[65536,131072)` / `UB[131072,196608)` FP32，三个完整 bank，各 64 KiB | 不涉及 | bank2 首次作为 checkpoint raw，首次 token 在统一 VF 中 cast；之后按三 bank 轮转 |
| `qh` | UB | `UB[184320,200704) FP32`，16 KiB | 不涉及 | `[L,K]`，最多 32 个 token，S1 只读 |
| `kh` | UB | `UB[200704,217088) FP32`，16 KiB | 不涉及 | `[L,K]`，最多 32 个 token，S1 只读 |
| `vF32` | UB | `UB[217088,233472) FP32`，16 KiB | 不涉及 | `[L,V]`，最多 32 个 token，S1 只读 |
| `decay,beta` | UB | `UB[233472,233728) FP32`，256 B | 不涉及 | `[L]` 两组 FP32 标量，S1 只读 |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| `checkpoint_state` | GM（API 输入） | `tensor_base + offset`，按 `P,HV,V,K` 实际 stride 读取，单 work 最多 65536 B | 不使能 | 每 work 只读一次初态，S0 消费完成 |
| `q,k` | GM（API 输入） | `tensor_base + offset`，各读取至多 `L*K*2` B | 不使能 | 当前 HK 的窗口 Q/K，S0 消费 |
| `v` | GM（API 输入） | `tensor_base + offset`，读取至多 `L*V*2` B | 不使能 | 当前 HV 的窗口 V，S0 消费 |
| `a,b` | GM（API 输入） | `tensor_base + offset`，各读取至多 `L*4` B | 不使能 | 当前 HV gate/beta 原始值，S0 消费 |
| `A_log,dt_bias` | GM（API 输入） | `tensor_base + hv*4`，各读取 4 B | 不使能 | 当前 HV 两个标量 |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| `stateRaw` | UB | 不适用 | 不涉及 | 不再单独分配；raw 直接使用 `stateBank[2]` 的物理区域 |
| `qkvRaw` | UB | `UB[163840,184320) BF16`，24 KiB | 不涉及 | Q/K/V 原始 BF16，最多 32 token；Cast 后释放 |

### S1 · Vector：逐 token 状态递推与记录

```text
S = stateBank[slot]
for position, token in enumerate([bos,eos)):
    D = S * decay[position]
    residual = beta[position] * (vF32[position] - rowDot(D, kh[position]))
    S = D + residual[:,None] * kh[position][None,:]
    out[0,token,hv,:] = cast_bf16(rowDot(S, qh[position]))
    replay[page,hv,position,0:V] = residual
    replay[page,hv,position,V:V+K] = kh[position]
    replay[page,hv,position,V+K] = decay[position]
```

每个有效 token 先完成 residual 点积，再原位更新已消费的 decayed state；output 使用更新后的
state。tile 内完成 4 个 token 后，`outTile/replayTile` 由一次 MTE3 批量写出。

**本 Stage 产生/写出**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| `outTile[0/1]` | UB | `UB[131072,132096) BF16` / `UB[132096,133120) BF16`，ping/pong 各 1 KiB | 不涉及 | `[4,V]` BF16，最多 4 token；MTE3 后释放 |
| `replayTile[0/1]` | UB | `UB[133120,137216) FP32` / `UB[137216,141312) FP32`，ping/pong 各 4 KiB | 不涉及 | `[4,V+K+1]` FP32；MTE3 后释放 |
| `stateBank[0/1/2]` | UB | `UB[0,65536)` / `UB[65536,131072)` / `UB[131072,196608)` FP32，三个完整 bank，各 64 KiB | 不涉及 | 按 token 顺序轮转；不写回 checkpoint，work 完成后释放 |

**其它 Stage 提供、当前 Stage 读取或需保护**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| `qh,kh` | UB | `UB[184320,217088) FP32`，32 KiB | 不涉及 | S0 产生，S1 全窗口只读 |
| `vF32` | UB | `UB[217088,233472) FP32`，16 KiB | 不涉及 | S0 产生，S1 全窗口只读 |
| `decay,beta` | UB | `UB[233472,233728) FP32`，256 B | 不涉及 | S0 产生，S1 逐 token 读取 |

**本 Stage 临时/固定**

| 对象 | 存储位置 | 分配地址 | L2Cache | 功能 |
| --- | --- | --- | --- | --- |
| `decayPartial,dot` | UB | `UB[233728,234752) FP32`，1 KiB | 不涉及 | 当前 V 行 FP32 临时，VF 内复用 |
| `reduction` | UB | `UB[234752,235008) FP32`，256 B | 不涉及 | rowDot 的 V 行结果和标量广播，VF 内复用 |
| `out` | GM（API 输出） | `tensor_base + token_offset + hv_offset`，每 tile 至多 `4*V*2` B | 不使能 | MTE3 写 BF16 有效 token |
| `replay_cache` | GM（API 输出） | `tensor_base + page_offset + hv_offset + position_offset`，每 tile 至多 `4*(V+K+1)*4` B | 不使能 | MTE3 原地写 FP32 有效记录；padding 不写 |

主 case 的最大 UB 地址为 `235008 B`，即 `229.5 KiB`；计入事件与对齐余量后按 231 KiB
预算，仍低于 A5 的 248 KiB。K/V 缩小时所有缓冲区等比例缩小。

## 5. 用户额外 workspace

本算子不创建 kernel 私有 GM 中间张量。`checkpoint_state` 是 API 输入，`out`（最终输出）和
`replay_cache` 是 API 输出，片上 resident 不进入 workspace 统计。系统 workspace 不包含在
下列用户额外 workspace 内。

```text
workspace_size = 0
```

所有中间 state、归一化 Q/K/V、gate 和输出 tile 均驻留 UB，因此不存在 workspace
对齐、缩放或复用问题。

## 6. 跨 Stage 同步契约

同步范围仅为单个 AIV 内的 MTE2、Vector 和 MTE3；不存在 AIC/AIV 跨核交换。不同 AIV 的
work 和输出区间互不重叠，不增加跨核同步。

| 数据边 | Producer | Consumer | ready | free/复用条件 | 本地 pipe 完成 | 回绕/退出处理 |
| --- | --- | --- | --- | --- | --- | --- |
| checkpoint/state bank | MTE2 | S0 Vector | `STATE_MUTEX[2]` 的 `PIPE_MTE2` unlock | 每个 token 只获取目标 bank 的 `PIPE_V` mutex，并按三 bank 轮转 | Mutex lock/unlock 成对 | 每个 work 从 bank2 raw 重新开始，work 完成后全部释放 |
| S0 QKV resident | S0 Vector | S1 Vector | S0 末次 VF 后按程序顺序就绪 | 当前 work 全部 token 完成后释放 resident | 同 pipe 依赖使用 `PipeBarrier<PIPE_V>()`，不跨硬事件冒充 | 零长度 work 不进入 S0；非空 work 完成后不保留等待 |
| input tile | MTE2 | S1 Vector | `INPUT_MUTEX[tile]` 的 `PIPE_MTE2` unlock | Vector 获取同一 mutex 后消费 tile | Mutex lock/unlock 成对 | 双 slot 回绕；尾 tile 只搬有效行 |
| output/replay tile | S1 Vector | MTE3 | Vector 释放 `OUTPUT_MUTEX[tile]` | MTE3 获取同一 mutex 后写回 | Mutex lock/unlock 成对 | 下一个复用者等待 mutex；work 尾自然释放 |
| 下一 work state 预取 | Vector | 下一 work MTE2 | 单 token work 搬入完成时，或多 token work 倒数第二个 token 完成时发布 `V_MTE2(EVENT_ID0)` | 下一 work MTE2 在 `MTE2_V(EVENT_ID0)` 后写目标 bank | `SetFlag/WaitFlag<HardEvent::V_MTE2/MTE2_V>` | 仅存在同核下一 work 时发布；最后一个 work 不发布 |

执行顺序：

```text
for work in core_work_range:
    wait V_MTE2(EVENT_ID0)
    select next raw bank and lock its PIPE_MTE2 state mutex
    DataCopyPad initial checkpoint into selected bank raw view
    unlock selected PIPE_MTE2 state mutex
    if single-token work and same-core next work exists:
        set V_MTE2(EVENT_ID0)
    S0 Vector uses bank2 raw for the first token
    for each four-token tile:
        lock PIPE_MTE2 input mutex and DataCopyPad tile
        unlock PIPE_MTE2 input mutex
        lock PIPE_V input/output mutexes
        S1 Vector casts bank2 raw on first token, then rotates bank2 -> bank0 -> bank1 -> bank2
        each token locks only its destination bank state mutex
        unlock PIPE_V input/output mutexes
        if this is the penultimate token and same-core next work exists:
            set V_MTE2(EVENT_ID0)
        lock PIPE_MTE3 output mutex
        MTE3 writes out and replay
        unlock PIPE_MTE3 output mutex
```
