# RecurrentKda 设计方案

方案设计规则版本：V2

## 1. 背景

KDA recurrent 用于小步长 decode/MTP 场景，按 token 顺序更新 KDA 状态并生成输出。上游常用 API 支持
raw gate、`A_log/dt_bias/lower_bound/safe_gate`、`beta sigmoid` 和两种 state 布局。该实现的目标
是在一个 KDA 独立算子中完成这些语义，避免把 gate 预处理拆成 `KdaGateCumsum + recurrent` 两段公共实现，也不复用
GDN recurrent 的函数接口。

## 2. 目标与非目标

### 2.1 目标

- 提供 Ascend C 实现类型的 `recurrent_kda`，覆盖 `fla_npu.ops.ascendc`、aclnn 和 `<<<>>>` 入口；legacy 通路不属于支持范围。
- 对齐常用上游 API：`raw gate`、`A_log`、`dt_bias`、`lower_bound`、`safe_gate`、`beta sigmoid`、`allow_neg_eigval`。
- 支持 `BSND` 和 `TND`，支持 device `cu_seqlens` 变长序列、容量化 state pool 和 speculative slot 索引。
- 保持 KDA recurrent 与 GDN recurrent 接口解耦。

### 2.2 非目标

- 不提供内部 state 矩阵任意转置或步长采样的低性能 gather/scatter 路径。
- 不支持把 `KdaGateCumsum` 的输出流水作为公开前置步骤。
- A2/A3 通过第 17 节的时间窗口支持长序列；保持逐 token recurrent 公式和 AIV-only 路径。A5 仍限制每段长度 <=8。

## 3. 能力边界

实现类型：`ascendc`

支持范围以当前公共接口和实现约束为准，Shape 符号引用 [KDA 模型符号表](../../README.md#model-shape-symbols)。

| 维度 | 支持范围 |
| --- | --- |
| SOC | `ascend910b`、`ascend910_93`、`ascend950` |
| Layout | `BSND`、`TND` |
| Q/K/V dtype | BF16 |
| K/V | `K=128,V=128` 或 `K=128,V=256` |
| Heads | `1<=H_k,H_v<=256`，且 `H_v % H_k == 0` |
| Gate/Beta dtype | FP32/BF16/FP16，kernel 在 UB 中转为 FP32 计算 |
| State dtype | FP32/BF16 |
| Gate 模式 | 预计算 step log gate；kernel 内 raw gate |
| State layout | V-first `[state_capacity,H_v,V,K]`；K-first `[state_capacity,H_v,K,V]` |

## 4. 数学与接口语义

若 use_qk_l2norm_in_kernel=True，A2/A3 对每个 token/head 的 q/k 在最后一维
按 FP32 计算 x / sqrt(sum(x*x) + 1e-6)，随后仅 q 乘以 scale。
对应竞品 fused_recurrent.py 的 Q/K 归一化公式。A5 保持现有无 epsilon 的实现，
本轮不修改 A5 kernel 或 host 分派。关闭该属性时不执行归一化。

`use_gate_in_kernel=false` 时，`g` 已经是 step log gate。`use_gate_in_kernel=true` 时，kernel 内转换 raw gate：

```text
gate = -exp(A_log) * softplus(g + dt_bias)                 # safe_gate=False
gate = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))   # safe_gate=True
```

`use_beta_sigmoid_in_kernel=true` 时：

```text
beta = sigmoid(beta)
if allow_neg_eigval:
    beta = 2 * beta
```

每个 token 的 recurrent 更新为：

```text
S = exp(gate_t) * S
delta = beta_t * (v_t - S @ k_t)
S = S + outer(delta, k_t)
o_t = S @ q_t
```

`H_v % H_k == 0`，每个 value head 通过 `floor(h_v / (H_v / H_k))` 映射到一个 query/key head。

## 5. 整体架构

- `op_host/op_api/aclnn_recurrent_kda.cpp`：校验 dtype、layout、shape、可选输入组合；不读取 device metadata 的值；
  对规则非连续 Q/K/V 保留 stride/storage offset，已验证布局回退连续化，token/head 置换或
  正 stride 重叠 descriptor 明确拒绝；非连续 state 以
  `CreateView` 保留真实存储信息；创建 L0 executor。
- `op_host/recurrent_kda_tiling.cpp`：读取 Q/K/V 与输入/输出 state stride、shape、属性和可选输入存在性，
  填充 `RecurrentKdaTilingData`，计算 block dim 与 UB 切分。
- `op_kernel/recurrent_kda.cpp`：单个 AIV kernel 完成 q/k normalize、raw gate 转换、state decay、delta 更新、输出和最终状态写回。
- `torch_custom/fla_npu/csrc/src/stable_recurrent_kda.cpp` + `fla_npu/ops/ascendc/_stable.py`：提供解耦的 Python 入口（Stable-ABI 适配层），不注册 legacy `torch.ops.npu`；`_aclnn_ctypes.py` 保留为参考实现与回退。
- legacy `torch.ops.npu.npu_recurrent_kda` 不属于当前支持范围，不作为公共接口维护。

## 6. Tiling 设计

### 6.1 任务划分

Tiling 按逻辑序列和 value head 拆分任务。提供 `cu_seqlens` 时，`seq_num=len(cu_seqlens)-1`，使用与
fla-org 一致的累计边界语义，第 `i` 条序列范围为 `[cu_seqlens[i], cu_seqlens[i+1])`；未提供时，
BSND 按 batch 行划分序列，TND 视为一条序列。接口面向 decode / MTP 短序列：每个相邻 offset
的差值表示 varlen 每段长度；A2/A3 不再限制为 8，A5 每段与 dense 单序列仍须 <=8。提供 cu_seqlens 时，末项表示有效 token 数，可小于图捕获的
token capacity。

### 6.2 Tiling Data

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `vectorCoreNum` | `uint32_t` | 使用的 AIV core 数 |
| `ubCalSize` | `uint32_t` | 平台 UB 预算上限 |
| `ubRestBytes` | `uint32_t` | A2/A3 命名 scratch 实际字节数；A5 保持原剩余预算语义 |
| `t` / `seqLen` | `uint32_t` | token 总数与输入物理序列轴长度 |
| `nk` / `dk` | `uint32_t` | query/key head 数与 K 维 |
| `nv` / `dv` | `uint32_t` | value head 数与 V 维 |
| `sBlockNum` / `b` | `uint32_t` | state pool 容量与逻辑序列数 |
| `ssmStateStride` | `uint32_t` | 二维 speculative state 索引的行 stride；packed 一维索引时为 0 |
| `query/key/value TokenStride` | `uint64_t` | Q/K/V 相邻逻辑 token 的元素 stride |
| `query/key/value HeadStride` | `uint64_t` | Q/K/V 相邻 head 的元素 stride |
| `vStep` | `uint32_t` | 单次处理的 V 方向步长 |
| `stateOutBufferNum` / `attnOutBufferNum` | `uint32_t` | UB ring buffer 数 |
| `scale` / `lowerBound` | `float` | query scale 与 safe gate 下界 |
| `layout` | `uint32_t` | BSND/TND 编码 |
| `has*` | `uint32_t` | 可选输入存在性 |
| `use*` / `safeGate` / `stateVFirst` | `uint32_t` | 属性分支 |

### 6.3 模板化方案与 tiling key

当前 kernel 通过 `REGISTER_TILING_DEFAULT(RecurrentKdaTilingData)` 使用默认 tiling data，不引入额外 tiling key。
dtype 固定为 BF16 QKV，state dtype 由编译宏 `DTYPE_INITIAL_STATE` 实例化；layout、gate/beta 分支和可选输入存在性由
tiling data 驱动，避免把 runtime shape 组合扩张到 tiling key。

## 7. Kernel 设计

### 7.1 计算流程

1. 解析当前任务对应的逻辑序列、value head、query/key head 和状态槽。
2. 加载初始状态到 UB 工作区；Python 仅在非原位模式下允许 `initial_state=None`，并预先创建全零状态。
3. 按 Q/K/V 各自 token/head stride 计算 GM 首地址并二维搬运到紧凑 UB；随后按属性执行 q/k
   normalize、raw gate 转换和 beta sigmoid。
4. 对状态做 gate decay，计算 `S @ k`、delta 和 outer 更新。
5. 计算 `S @ q` 写回 `out`，并把状态原位写回命中的 state pool 槽。

### 7.2 内存规划

| 内存层级 | Buffer | 生命周期 | 用途 |
| --- | --- | --- | --- |
| GM | 输入/输出 tensor | kernel 全程 | q/k/v/g/beta/state/out 与 device metadata |
| GM | workspace | kernel 调用期间 | 无算法专用 GM 中转；按标准运行时预留 16 MiB system workspace |
| UB | state 工作区 | 单任务内复用 | 保存 `[V,K]` 局部状态片段 |
| UB | q/k/v/g/beta 临时区 | 单 head 内复用 | normalize、gate、beta 与输出计算 |

### 7.3 流水与同步

算子为 AIV-only recurrent 小步长 kernel，单任务内部按 token 串行维护状态依赖。调用侧保证不同活跃序列不共享写槽，
任务分配保证每个 `(seq/value head)` 的输出区域独立。`--cce-auto-sync=off` 保持不变，UB 复用由本 kernel 内顺序和
必要的 vector pipe barrier 维护，不依赖 GDN recurrent 的同步协议。

### 7.4 边界处理

- 提供 `cu_seqlens` 时，首项必须为 0，offset 必须单调不减，末项为有效 token 数且不得超过输入 token
  capacity。host 只检查 rank/dtype，device kernel 校验值。
- 相邻 offset 相等时表示空序列；该序列不产生 token 输出，也不读取 `ssm_state_indices/num_accepted_tokens` 或 state。
- 末项小于 capacity 时，仅 `[0,cu_seqlens[-1])` 的有效输出和 state side effect 有定义；padding tail 输出不作保证。
- 一维 `ssm_state_indices` 按 packed token 偏移读取；二维索引按 `[seq_idx,step]` 读取，第二维必须覆盖对应序列长度。
- `num_accepted_tokens` 只在提供 `ssm_state_indices` 时有效，用于定位 MTP decode 的初始状态槽。

## 8. 平台设计

| 平台 | SOC | 实现策略 | 分支差异 |
| --- | --- | --- | --- |
| A2 | `ascend910b` | 公共 AIV kernel | 无 |
| A3 | `ascend910_93` | 公共 AIV kernel | 无 |
| A5 | `ascend950` | arch35 特化 AIV kernel | RegBase/DataCopy 路径，并对 `K=128` 使用融合归约实现 |

## 9. 精度设计

Q/K/V 公开输入为 BF16，gate/beta 保持 FP32/BF16/FP16 输入 dtype，并在 kernel 的 UB 中转为 FP32 计算。state 可为 FP32 或 BF16；推荐 FP32 状态以降低小步长多 token 更新中的累计误差。

## 10. 性能设计

目标场景是 decode/MTP recurrent 小步长，主要瓶颈是状态矩阵读写和每 token 的 `S @ k`、`outer(delta,k)`、`S @ q`。
当前实现已使用输入/输出双缓冲，并在 arch35 的 `K=128` 路径采用融合归约和 repeat DataCopy；后续优化仍应以
状态搬运、向量流水和同步开销为重点。

## 11. 测试设计

测试工程作为本地开发资产保留，不纳入版本控制，远程仓库文档不依赖其目录结构。本地验收矩阵覆盖：

- BSND raw gate + unsafe gate + beta sigmoid。
- BSND safe gate + `allow_neg_eigval`。
- TND 预计算 log gate + 空 initial_state。
- Kimi/KDA 关键泛化 shape：GVA head 映射、`K=128,V=128` dense raw gate、`K=128,V=256` TND safe gate。
- Kimi K3 TP16：本地 head 数 6、`K=V=128`、BF16 Q/K/V、raw+safe gate、TND packed decode/MTP 1-8；
  覆盖 `[0,1,4,4,5]` 非等长序列、二维 slot 索引、容量化 state pool、非连续槽更新和未命中槽保持不变。
- 空序列覆盖 `[0,1,1,4,4,8]`，确认重复 offset 对应的空序列不读取索引或 state。
- Kimi H96/D128 smoke：`H=H_v=96,K=V=128,safe_gate=True`，运行时生成非等距 `cu_seqlens`，
  覆盖 cu_seqlens 长度和值泛化；该历史测试集每段长度 <=8，新增长序列覆盖见第 17 节及 ATK README。
- Kimi 完整长上下文 stress target：`T_total=12288,H=H_v=96,K=V=128,safe_gate=True` 作为本地 stress case，
  因当前单 kernel 计算量较大，不纳入默认通过矩阵。
- 非连续 state 覆盖 V-first/K-first、FP32/BF16、原位/非原位、slot pool guard 与未命中槽。
- 负向参数组合：长序列、`safe_gate` 与 raw gate 组合、内部矩阵非稠密 stride 与外层重叠 stride。

## 12. 验证修复记录

A2/A5 精度验证过程中修复以下问题：

- `scale/lower_bound` 的 aclnn 公开参数为 `double`，AICore attr 为 `float`；L0 加入显式 cast，避免 tiling 读取到错误标量。
- A5 tiling 对 `GetOriginShape()` 返回对象生命周期更敏感；tiling context 改为持有 shape 副本，避免 optional/required shape 指针悬挂。
- q/k L2 normalize 后从 UB 标量读取 norm 前补充 V->S 同步，避免 scalar 读到旧值导致 raw/safe gate case 精度偏差。

## 13. 已知限制与演进计划

- 当前仅支持 BF16 QKV。
- Q/K/V 直接非连续访问仅支持 feature 致密、head/token 不重叠的正 stride；BSND `B>1` 还要求
  batch 维可按 `T * tokenStride` 扁平化。已验证的 feature-gap、zero-stride、batch-gap view
  自动回退连续化；token/head 置换或正 stride 重叠 descriptor 明确拒绝。
- 当前 `K/V` 仅支持 `K=128,V=128` 或 `K=128,V=256`。
- A2/A3 内部时间窗口容量为 8，支持多窗口序列；A5 每段仍限制为 <=8。
- state 最后两维必须为行主序稠密矩阵；slot/head 维可有间隔但不得重叠。
- 显式 slot 模式要求所有活跃序列的写槽互不冲突，且 slot 值位于 state pool 容量范围内。
- 后续若扩展长序列或更多 dtype，需要同步更新 op_host 校验、tiling、kernel、本地测试 case 和 API 文档。

## 14. A2/A3 内部性能优化

本节落实 A3 优化方案中不改变接口的候选。公共 A2/A3 kernel 使用 packed raw 双槽，
host 仅为 A2/A3 调整 UB 预算和具名 scratch 的实际申请量。A5 仍由
`__CCE_AICORE__ == 310` 选择 `arch35/recurrent_kda.h`，保留原 host UB 计算结果。
ABI、CPU 标杆、L2Norm/gate/Beta 数学序列、state 布局和每段长度限制保持原定义。

### 14.1 任务域与直接寻址

令 C 为实际 AIV blockDim，N=B*HV；C=min(N,可用 AIV 数)。每个
work_id=b*HV+h 的 owner=work_id%C。每条序列的首 head 为
(blockIdx+C-(b*HV)%C)%C，其后以 C 为步长遍历，超出 HV 即结束。
这与旧逐 head 扫描拥有完全相同的任务和处理次序，保持同序列下一 head 的 state 预取、
序列级 Beta 复用和空序列跳过。不同 core 的输出和 state/head 矩阵不重叠，无核间同步。

模型为原四个 base case，T=1、H=HV=96、K=V=128、BF16 QKV/state、
FP32 gate/Beta，开启 L2Norm/raw safe gate/Beta sigmoid。

实测设备的 AIV 数为 40；任务域代入如下：

| B | Nwork=B*96 | blockDim | 最多 wave | 每核 head 数 | 尾波任务 |
| --- | ---: | ---: | ---: | --- | ---: |
| 1 | 96 | 40 | 3 | 2–3 | 16 |
| 4 | 384 | 40 | 10 | 9–10 | 24 |
| 16 | 1536 | 40 | 39 | 38–39 | 16 |
| 64 | 6144 | 40 | 154 | 153–154 | 24 |

例如 B=1 的 core 0 依次处理 head 0/40/80，全部 token 和 V block 均归该 core。
优化目标是每个原始 base case 在相同设备与输入下的设备 kernel 时延下降；
最终性能以 msopprof BasicInfo 的 Task Duration(us) 为准。
ATK performance_data=20,20,10 的 CANN op_summary 时延另作重复采样验证；
两种采集方式分别比较优化与对照，不混合绝对时延或样本。
本轮 packed raw 优化对照为上一轮已验证工作树，已含第 14.3～14.6 节的优化及
B=1 单 token 手工输入预取。对照与候选均通过原四 base 和补充 200-case 精度。
更早的仅修复 EventID 对照记录保留在测试文档，不能与本轮收益混用。

### 14.2 Stage 与资源边界

一个 Vector Stage 完成输入预处理和全部 V block/token 递推，无 Cube Stage。
FP32 state 持续驻留；数学序列、归约、cast、输出 snapshot 和 CPU 标杆不变。
A2/A3 将 Q/K/V/gate 合并为一个 packed raw 输入队列，具有两个独立物理槽；
Beta 仍按序列搬入一次，state 输入双缓冲和原输出队列保持原生命周期。
host 同步计入两个 raw 槽，按实际 NV 计算 generic Beta 容量，tmpBuff 只申请具名分区。
A5 继续使用原单份输入预算、对齐 NV、原 ubRestBytes 余量和 arch35 kernel。

B=1 目标的实际平台预算为 196352 B，vStep=64。每核分配如下，
区间均为核内 UB 字节半开区间，slot 内的 Q/K/V 为 BF16，gate 区按 FP32 最大容量保留：

| 区间 | 对象及物理容量 | 生命周期 |
| --- | --- | --- |
| [0,10240) | packed raw slot 0；Q/K/V 各 2048 B，gate 4096 B | MTE2 写入，V Cast/读取完成后释放 |
| [10240,20480) | packed raw slot 1；布局同 slot 0 | 与 slot 0 交替，不重叠 |
| [20480,23552) | Beta 输入队列，3072 B | 序列级搬入、FP32 转换后释放 |
| [23552,56320) | state 输入两槽，各 BF16 [64,128]、16384 B | MTE2 写入，V Cast 后释放 |
| [56320,89088) | state 输出两槽，各 16384 B | V 写快照，MTE3 搬出完成后复用 |
| [89088,89344) | attention 输出两槽，各 BF16 [64]、128 B | V Cast，MTE3 搬出完成后复用 |
| [89344,174848) | tmpBuff，85504 B；内部切分见下表 | 一个 head 的 FP32 计算驻留 |
| [174848,174912) | scalarBuf，64 B | 前 32 B 为标量运算，后 32 B 为当前 head Beta 缓存 |
| [174912,194304) | 空闲，19392 B | 不属于 tmpBuff，不存活任何数据 |
| [194304,196352) | guard，2048 B | 不分配 |

tmpBuff 的全部分区：

| 区间 | 对象、dtype/shape | 大小 |
| --- | --- | ---: |
| [89344,89600) | delta，FP32 [64] | 256 B |
| [89600,89856) | attention，FP32 [64] | 256 B |
| [89856,93952) | V，FP32 [8,128] | 4096 B |
| [93952,98048) | Q，FP32 [8,128] | 4096 B |
| [98048,102144) | K，FP32 [8,128] | 4096 B |
| [102144,134912) | state，FP32 [64,128] | 32768 B |
| [134912,167680) | broadTmp，FP32 [64,128] | 32768 B |
| [167680,170752) | Beta，FP32 [8,96] | 3072 B |
| [170752,174848) | gate，FP32 [8,128] | 4096 B |

每核申请 174912 B，较本轮对照的 194304 B 减少 19392 B。
tmpBuff 从 115136 B 收紧至 85504 B；原独立 raw bank 随 packed 队列一并替换。
T=1 只搬入、转换一个有效 token，但槽容量仍覆盖最多 8 token。
K-first 另计 stateTransposeBuf；其它 shape 的分区按相同公式重算，不统一减常数。
算法 GM workspace 仍为 0，框架原系统 workspace 保持不变。

### 14.3 手工事件的保留与释放

当前 CANN 的 FetchEventID 只查询空闲编号，不置位 eventOccupy；不能把查询结果
缓存跨队列生命周期复用。A2/A3 的五类手工硬事件改用 AllocEventID 首次分配并保留，
每次同步严格 Set/Wait 配对，kernel 退出时按原 ReleaseEvents 成对释放。
MTE2_V 保护 raw 标量/bias 搬运到 V 读取；V_MTE2 保护原位搬入前的 V 写入；
V_S 保护 norm/exp 标量读取；V_MTE3/MTE3_V 保护转置输出及复用。
每个方向最多保留 1 个手工事件；packed 两槽、Beta 一槽、state 输入两槽，
VECIN 共 5 个物理 buffer，按每槽一个事件的保守上界加手工事件为 6。
VECOUT 最多 4 个物理 buffer，加同方向手工事件不超过 5。
当前 CANN 每方向 8 个 EventID；队列事件随 EnQue/DeQue/Free/Alloc 动态分配释放。
没有跨核事件。手工 buffer 生命周期与队列事件分离，不改变计算顺序或输入值域。

### 14.4 残差 block 复用

沿用 Stage 的 FP32 递推公式，delta[i]=beta*(v[i]-sum_j(state[i,j]*k[j]))。
预测归约完成后，Brcb 将每个 delta 扩为一个 32 B block，
在每个 K=128 的 64 列半行中用 src0BlkStride=0 重复该 block，
src0RepStride=1 切换到下一行，state 的行 repeat stride 为 16，
K 的行 repeat stride 为 0。MulAddDst 的运算仍为 state+=delta*k，
归约和 FP32 累加次序保持不变。合法 V 和 vStep 均为 8 的倍数，
因此有效行数 R 也是 8 的倍数，Brcb 恰好读取 R 个有效残差。

地址账本使用第 14.2 节的实际分配。broadTmp 的所有权顺序为：
预测乘积 -> Add-fold/WholeReduceSum 最后读取 -> V barrier ->
Brcb 写 [134912,136960) 的 [64,8] 残差 block -> V barrier ->
两次 MulAddDst 消费 -> V barrier -> 输出乘积覆盖 -> 归约。
delta 源与目标不重叠，归约与 FP32 累加顺序保持不变。

### 14.5 当前 head 的有效 Beta 标量缓存

CopyInBeta 的序列级 FP32 缓冲不变。每个已拥有的 head 在 V block 循环前，
按短序列或当前窗口的有效 token 顺序调用原 LoadBeta，保存最多 8 个 FP32 标量，放入 scalarBuf_ 的第二个 32 B block。
所有 V block 仅消费该缓存，避免同一 head/token 重复 sigmoid/ExpScalar/V_S 等待；
没有为其他 head 计算 sigmoid。下一个 head 重新覆盖缓存，不与异步搬运共享。
计算 dtype、sigmoid 和 allow_neg_eigval 的运算序列完全相同，没有代数变形。
Beta 缓存本身不增加 UB、GM 或队列分配；FP32 state 仍按 token 持续驻留，head 的全部时间步在单核完成。

### 14.6 A_log 的搬入与 Exp 连续执行

原路径 ReadFloat -> ExpScalar 先把 A_log 搬至 scalarBuf，做 Adds(0)、V_S、
标量读取，再 Duplicate 回同一区、Exp、V_S、标量读取。改为同一块
scalarBuf 的 MTE2 搬入 -> MTE2_V -> FP32 Exp -> V barrier -> V_S -> 标量读取，
保持 exp(A_log) 的指令、dtype、输入和结果相同，消除中间 V/S 往返和冗余 Adds/Duplicate。
scalarBuf 的地址与 64 B 容量不变，不删除或缩小 guard；无新增同步资源。
高风险范围沿用原 Exp，不引入代数变换；A_log 不存在的路径继续使用 1。

Beta 缓存的物理地址为 scalarBuf_ 起点加 32 B；前一个 block 仅供
ReadExpFloat/ExpScalar 使用。两者按 32 B block 完全不重叠。
B=1 的 UB[174848,174880) 为搬运/Exp 标量块，UB[174880,174912) 为
8 个 FP32 Beta 缓存。MTE2 的 4 B DataCopyPad 和 count=1 的 Vector 运算
只访问第一个 block；第二个 block 由 Scalar 写入、读取和跨 head 覆盖，
同一 S pipe 的有序访问不新增跨 pipe 事件。该复用不增加原 64 B scalarBuf 分配。

实际 scalarBuf 起点等于所有前序 queue/buffer 的已分配字节数，加 tiling.ubRestBytes；
Beta 缓存始终取这个起点的 +32 B，长度 32 B。不同平台或 dtype 的绝对起点变化时，
这两个 block 的相对隔离和原 64 B 容量约束保持成立。

### 14.7 Packed raw 双槽与跨 head 预取

本节适用于公共 A2/A3 kernel 的全部合法分支；A5 不包含该 kernel。
每槽大小为 MAX_MTP * ((2*alignK+alignV)*sizeof(BF16)+alignK*sizeof(FP32))。
Q、K、V、gate 分别保留独立 plane；gate 的 BF16/FP16 输入放在相同起点，
只读取有效 seqLen，预处理仍转换至原 FP32 gate 分区。
非连续 Q/K/V 使用原 token/head stride，源地址和有效读取字节数不变。

每条非空、属于本核的序列先提交首 head。处理当前 head 时：
1. DeQue 当前 raw 槽，等待其 MTE2_V ready。
2. Cast/读取当前 Q/K/V/gate 到单份 FP32 驻留区。
3. 当前 raw 槽仍由消费者持有时，为同序列下一拥有的 head AllocTensor。
   两槽中只有另一槽可分配，复用时等待其上次 V_MTE2 free。
4. 对下一槽执行四次 DMA 并 EnQue，之后 FreeTensor 当前槽。
5. 执行当前 head 的 norm/gate/Beta/state 计算；下一 head 的 raw 搬入可与 V 运算重叠。
6. 当前 head 全部 V block/token 完成后，消费下一 raw 槽。

同步伪代码：

    for each nonempty owned sequence:
        produce(first_head)
        while head is valid:
            current = DeQue()                 # wait ready
            Cast current into FP32 buffers
            if next_head exists:
                next = AllocTensor()          # other slot; wait free on reuse
                DMA Q/K/V/gate into next
                EnQue(next)                   # publish ready
            FreeTensor(current)               # V read complete before reuse
            preprocess and compute current head
            head = next_head

B=1 的 core 0 依次处理 head 0/40/80，raw 地址必须表现为 slot 0/1/0，
覆盖首次使用、切换及回绕。core 16–39 只有两个 head，覆盖切换和尾任务。
最后一个 head 不提交新输入，空序列和无任务核不产生 ready/free；
不同序列不互相预取，退出时无未消费的 raw ready。
不复用手工 MTE2_V 作为异步 ready，原标量和 bias 的事件回到立即 Set/Wait。

MTE2 与 V 的实际重叠、同核 slot 运行轨迹、原四个模型的收益均须由设备验证；
双槽本身不作为性能达标证据。FP32 输入仍为单份，不声称两个 V 计算阶段并行。

### 14.8 多 token 尾部同槽 state 写回

UB profile 变化可能使合法 MTP 用例从单 state 输出槽切换为双槽。
双槽在 token 循环内通过再次 AllocTensor 等待旧槽的 MTE3 读取完成，
但循环退出时只剩最后两次待写快照，最后一次写回之前没有下一次队列分配建立等待。
若最后两个 token 指向同一 state 槽，两次 MTE3 写的 GM 区间相同。

A2/A3 仅在 hasPendingState、序列长度大于 1 且最后两个 state 槽相同时，
于尾部最后一次 CopyOutState 前以 MTE3_S 成对 SetFlag/WaitFlag：
先完成已提交的 MTE3 写，再由 S 提交最后快照的写回。
事件通过 AllocEventID 获取，Wait 后立即 ReleaseEventID；该方向最多临时保留 1 个事件。
不改变快照数、状态值或索引语义，不跨核同步，不新增 UB。
T=1 或尾部写入不同槽时不执行等待。A5 arch35 保持原实现。

最终验证：B=1 的全部 40 核 workspace 轨迹确认 raw 双槽切换与回绕，
输出/state 与普通版本逐位一致。最终普通 kernel 的 TimelineDetail 仿真观察到
core 0 的 head 80 raw DMA 与 V 运算交叠，head 40 未观察到交叠，不宣称全部搬运隐藏。
原四 base 精度 4/4、补充 200/200、额外 MTP 2/2；硬件 msopprof 四 case 均提升。
racecheck/memcheck 在记录的代表范围无错误；initcheck PRIVATE 报告和冗余同步
警告仍保留，完整范围与数据见 tests/atk/recurrent_kda/README.md 的本轮记录。


### 14.9 A2/A3 L2Norm epsilon 修复

24b44ae1 中逐行 NormalizeRows 的顺序为 Mul → ReduceSum → Adds(1e-6f) → Sqrt →
V_S → Scalar 倒数 → Muls；各有依赖的 V 指令间保留 PIPE_V barrier。
CPU 标杆以 PyTorch 显式平方和、加 epsilon、开方和相除表达同一公式，
取代 F.normalize 的 norm clamp。Q 的 scale 时机保持在归一化之后。
零向量仍得到零；有限、平方和不溢出的输入，其分母至少 1e-3，
倒数至多 1e3，避免直接除以零。该改动不改变 UB 分区、vStep、队列或事件数，
没有新增跨核依赖；现有 FP32 deltaInUb 标量槽存放加 epsilon 后的平方和。
每个有效 Q/K 行各增加一次标量元素 Adds 和相应 V barrier，性能代价由八模型实测。
数值正确性以竞品公式为依据，不为保持旧时延删除 epsilon。


### 14.10 A2/A3 批量 Q/K L2Norm 与单次 V_S

本节替换公共 kernel 的逐行归一化调度，数学公式和 A5 实现不变。
短路径 L=seqLen，长路径 L 为当前窗口有效行数，均为 1..8。Q/K 仍各占独立的 TOKEN_WINDOW 行平面，
只处理当前 head，不跨 head 合并 FP32 数据。Q 的 scale 保持独立后置。

顺序为批量 Q² → Q 逐行归约 → 批量 K² → K 逐行归约 →
两组平方和加 1e-6 → 两组 Sqrt → 一次 V_S → Scalar 倒数与逐行 Muls →
PIPE_V → 原有 Q scale。K=128 继续使用原 AddFold + WholeReduceSum 行归约树，
调用 rows=L，不合并不同 token/head 的归约维。FP32 dtype、epsilon、倒数和 scale
运算顺序保持不变，零向量和小范数语义继承 14.9。

预处理期间复用 broadTmpInUb[0,L*alignK) 存 Q² 或 K²；
deltaInUb[0,L) 存 Q 范数，attnInUb[0,L) 存 K 范数。
host 保证 vStep>=8，而 L<=8，因此平方 scratch 与两组范数均在现有 slice 内。
Q/K 范数各至多 32 B，按各自 slice 的原 32 B 对齐起点存放。
它们在本 head 的 state 计算前已全部被 Scalar 消费，随后恢复 delta/attn 的原语义。
不复用 scalarBuf 的 Exp 和 Beta 两个 block，不修改 UB 总分配、guard 或 host。

B=1、H=HV=96、K=V=128、BF16 state、vStep=64 时：
UB[89344,89600) delta，其中 [89344,89376) 为 Q 范数；
UB[89600,89856) attn，其中 [89600,89632) 为 K 范数；
UB[134912,167680) broadTmp，其中至多前 4096 B 为当前平方值。
其它 UB 区间与 14.7 保持一致，总分配仍为 174912 B。
其它 profile 的绝对起点按 InitLocalBuffers 变化，以上相对容量与不重叠约束不变。

同步闭环：
1. 已有 packed DeQue/Cast 提供 Q/K ready，不改变 raw 槽释放和下一 head 预取。
2. Q² 写完 PIPE_V 后归约；Q 归约结束 PIPE_V 后才允许 K² 覆盖同一 scratch，
   防止 WholeReduceSum 仍读取 Q² 时被覆盖。
3. K²→归约、归约→Adds、Adds→Sqrt 各保留必要 PIPE_V；Q/K 的独立 norm 操作可并列提交。
4. 两组 Sqrt 完成后使用现有 V_S event 成对 Set/Wait，然后 Scalar 读取两组结果。
   每 head 仅一次 L2Norm V_S，不增加 EventID；其它 gate/Beta 的 V_S 保持原样。
5. Scalar 只读取 norm，不向 UB 写回倒数；逐行 Q/K Muls 的地址互不重叠，
   循环末尾统一 PIPE_V，再执行独立 Q scale 及其原屏障。
6. gate 和 state 阶段之后可复用上述 scratch；无跨核共享或新 GM 搬运。
   空序列沿用 Process 的跳过逻辑，关闭 L2Norm 时不生产或读取范数。

默认 K=128 的源码层统计：L=1/4/8 时，L2Norm V_S 分别由 2/8/16 次变为 1 次，
Vector API 调用分别由 12/48/96 变为 12/18/26（不含 Q scale 和同步）。
收益需以同卡 msopprof 实测确认；不能把 Wait 所含的必要 V 计算直接算作可消除耗时。

## 17. A2/A3 长序列时间窗口

本节扩展序列长度，数学语义、接口、TilingKey、任务归属和 A5 路径保持现状。
短序列 L<=8 沿用原路径。长序列按 sequence -> head -> V tile -> time window -> token
执行；time window 容量为 8，尾窗仅处理有效 token。当前 V tile 的 FP32 stateInUb
覆盖整个逻辑序列，不从已转为 BF16 的状态快照重新载入中间状态。

UB 的 packed raw 两槽、FP32 Q/K/V/gate、Beta、scalarBuf 和计算 scratch 保持原容量；
guard 保持 2048 B，不新增 state workspace，峰值与序列长度无关。不同 V tile 会重复搬入
并预处理输入窗口，这是避免完整 FP32 state 跨时间窗口写回 GM 的代价。
Beta 每窗口最多读取 8*NV，当前 head 的有效标量仍使用 scalarBuf 的第二个 32 B block。

每个窗口的 S0 为 MTE2 输入搬运，S1 为 Vector 转换/批量 QK L2Norm/gate 预处理，
S2 为 Vector 逐 token state 递推与输出快照，S3 为 MTE3 写出。
下个窗口只在上一窗口最后一个 token 的 FP32 state 已产生后进行递推。
现有 TOKEN_WINDOW<=vStep 约束保证归一化 scratch 容量。

packed raw 的预取顺序为同一 V tile 的下一窗口、下一 V tile 的首窗、下一 head 的首窗。
raw 槽按 Alloc/EnQue/DeQue/Free 闭环使用，转换完成后释放；FP32 输入和 scratch
复用前等待上一窗口的 Vector 消费完成。输出队列的 pending 状态跨窗口保留，仅在整个
V tile 末尾排空，沿用最后两个同槽 state 快照的 MTE3 完成等待。
不增加跨核协议。不同活跃 sequence 不得写入相同 state slot 的约束保持不变。

全局 token、窗口起点和状态元数据循环使用 int64；窗口内下标最多为 7。
ssm_state_indices 的二维下标基于原始 sequence 起点，num_accepted_tokens
仅用于选择完整序列的初始 slot。A2/A3 在填充 uint32 tiling 字段前校验容量，
不改变共享 tiling 布局或 A5 的选择与预算。

开发参考采用 CHUNK_DEPENDENT V1，提炼 commit a14961e9dfe8c8264768d67fb447101873da4ecf，
仅采用有依赖窗口的顺序和槽生命周期约束；本算子继续 AIV-only、无 Cube/CrossCore。
