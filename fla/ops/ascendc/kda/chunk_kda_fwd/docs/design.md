# ChunkKdaFwd 设计方案

## 1. 背景

Kimi Delta Attention 正向主算子。它消费已经按 chunk 累加的 key gate `gk`，分阶段生成 chunk 内矩阵项、递推状态和最终输出，并可返回完整中间量用于训练链路与精度定位。 本实现通过统一 aclnn 与 ctypes 稳定入口接入 Kimi Delta Attention (KDA) 链路。

## 2. 目标与非目标

### 2.1 目标

- README 所列 `定长/变长序列、四种显式 layout、可选初始/最终状态、可选中间量` 场景精度与仓内参考实现一致。
- A2/A3/A5 均能构建和执行，`--cce-auto-sync=off`。
- 若本算子用于替换 Triton，同一 shape/dtype/layout 下 Ascend C 性能优于被替换实现。
- aclnn、`fla_npu.ops.ascendc`、`<<<>>>` 使用同一接口语义。

### 2.2 非目标

- 不扩展 README“已知限制”之外的 shape、预留参数或 layout。
- 不以 fallback、放宽阈值或跳过失败 case 替代 kernel 根因修复。

## 3. 能力边界

实现类型：`ascendc`。Dtype：q/k/v 为同一 FP16 或 BF16；gk/beta 为 FP32 或 BF16，由 kernel
编译期模板直接读取；状态为 FP32。Layout：BSND/BNSD/TND/NTD；BNSD/NTD 为内部性能布局，
BSND/TND 通过 KdaLayoutSwap12 转换。模式：定长/变长序列、四种显式 layout、可选初始/最终状态、可选中间量。
Shape 符号统一引用[KDA 模型符号表](../../README.md#model-shape-symbols)。

## 4. 数学与接口语义

对每个 value head 映射到对应 key head，在一个 chunk 内定义：

```text
Aqk[i,j] = tril(q_i @ k_j^T * exp2(gk_i-gk_j)) * scale
Akk      = inv(I + tril((k_i @ k_j^T) * exp2(gk_i-gk_j) * beta_i, -1))
w        = Akk @ (k * beta * exp2(gk))
u        = Akk @ (v * beta)
kg       = k * exp2(-gk)
v_new    = u - w @ h_prev
h_next   = exp2(gk_last) * h_prev + kg_state^T @ v_new
o        = (qg @ h_prev + Aqk @ v_new) * scale
```

`gk` 位于 log2 空间，因此 kernel 以 `exp(x*ln2)` 实现 `exp2(x)`。`final_state`
固定为 FP32；partial chunk 的补齐行使用中性值参与固定 tile，公开输出的无效区域写零。

Python、aclnn 与直调入口均以同一逻辑 shape 解释输入。接口层转换只处理连续性、描述符或文档明确的
layout 规范化，不改变公式、边界 mask、head 映射或可选输入语义。

## 5. 整体架构

1. `op_host/*_def.cpp` 注册输入输出、dtype、属性和 A2/A3/A5。
2. InferShape、op_api 与 tiling host 共同按 README 校验必选参数、shape、dtype、layout、属性和可选输入组合，并构造或核对输出。
3. tiling processor 计算任务数、边界块、workspace 偏移和模板实例。
4. `op_kernel/` 按本算子的计算流程完成搬运、计算、同步和写回。
5. aclnn 两段式接口负责 contiguous、workspace/executor 和 stream 异步发射；输入按需转换到内部布局，最终输出由对应 kernel 直接按公开布局搬出。
6. `fla_npu.ops.ascendc` 仅通过 ctypes 调用 aclnn，不依赖 torch_npu dispatcher。

### 5.1 算子边界与 L2/L0 分工

公共 Python、aclnn 和直调接口只表达完整 KDA，不暴露数值阶段属性。`KdaChunkForward` 是 host 侧
L0 编排函数，不对应设备侧融合 kernel；它在同一 stream 上顺序加入以下独立算子：

| L0 算子 | 主要产物 |
| --- | --- |
| `ChunkKdaFwdPrepare` | `Aqk/Akk/qg` 及阶段种子 |
| `ChunkKdaFwdPostWu` | `w/u/kg/v_new` |
| `ChunkGatedDeltaRuleFwdH` | `h/v_new/final_state` |
| `ChunkKdaFwdFinalize` | `o` |

每个设备 kernel 只拥有自身阶段，不以 `SyncAll` 跨越其他算子。跨 chunk 状态传播复用独立
`ChunkGatedDeltaRuleFwdH` 的 Catlass kernel/scheduler，并固定 `use_exp2=true` 对齐 KDA log2 gate。
直调入口也按相同顺序发射四个 kernel，调用者无需构造阶段张量或阶段 tiling。

## 6. Tiling 设计

### 6.1 任务划分

PREPARE、POST_WU 和 OUTPUT 按 `(sequence, value_head, chunk)` 分配无跨 chunk 的矩阵任务；FWD_H
复用 ChunkGatedDeltaRuleFwdH 的调度并保持同一序列的 chunk 顺序。变长序列由 L2 生成紧凑的
`{seq, start, end, reserved}` chunk 元数据并放入 GM，PREPARE/POST_WU/OUTPUT 直接复用；tiling 只保存
shape、属性和执行资源等定长标量，避免随序列数膨胀。

### 6.2 Tiling Data

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `batch` | `int64_t` | host 计算并由 kernel 消费 |
| `seqNum` | `int64_t` | host 计算并由 kernel 消费 |
| `qHeadNum` | `int64_t` | host 计算并由 kernel 消费 |
| `vHeadNum` | `int64_t` | host 计算并由 kernel 消费 |
| `seqlen` | `int64_t` | host 计算并由 kernel 消费 |
| `kHeadDim` | `int64_t` | host 计算并由 kernel 消费 |
| `vHeadDim` | `int64_t` | host 计算并由 kernel 消费 |
| `chunkSize` | `int64_t` | host 计算并由 kernel 消费 |
| `totalChunks` | `int64_t` | host 计算并由 kernel 消费 |
| `scale` | `float` | host 计算并由 kernel 消费 |
| `hasInitialState` | `bool` | host 计算并由 kernel 消费 |
| `outputFinalState` | `bool` | host 计算并由 kernel 消费 |
| `isVarLen` | `bool` | host 计算并由 kernel 消费 |
| `safeGate` | `bool` | kernel 入口选择 `SAFE_GATE` 编译期模板实例 |
| `prepare*` | 阶段字段组 | PREPARE 核数、持久中间量和 scratch 偏移 |
| `postWu*` | 阶段字段组 | POST_WU 核数、输入种子和 scratch 偏移 |
| `output*` | 阶段字段组 | OUTPUT 核数、qg 输入和 FP32 scratch 偏移 |

host 侧为 PREPARE、POST_WU、FWD_H 和 OUTPUT 分别构造 tiling；每个 kernel 只读取自身字段。

序列边界不进入 tiling data。L2 根据 `cu_seqlens` 和可选 `chunk_indices` 生成每 chunk 一行的紧凑
元数据；kernel 只读取任务对应的一行控制信息，不在设备侧进行序列二分，也不复制大数组到 tiling。

### 6.3 模板化方案与 tiling key

数据类型不进入 tiling data：`DTYPE_Q`、`DTYPE_GK`、`DTYPE_BETA` 由算子编译实例生成，避免设备端
按整数 dtype 分派。各 KDA 阶段 kernel 只使用 key 1 和 mixed AIC:AIV task 类型；PREPARE 入口读取
`safeGate` 并选择 `<true>` 或 `<false>` 模板，chunk 热路径没有 runtime gate 分支。key 不编码
safe gate、B/H/T、layout、shape 或阶段，也不是公共接口属性。

`safe_gate` 不改变 KDA 数学公式。三方对标实现用它选择更稳定的 chunk 内数值路径；本实现的两个模板
实例都采用参考点因子化、FP32 16x16 对角求逆和块合并，因此 `false` 不会退化为容易溢出的实现，
`true` 也不需要在 AIV 热循环增加条件判断。

公共 aclnn API 没有 `stage` 参数。增加 dtype 或阶段时必须扩展受控模板实例和阶段 tiling，不允许恢复
runtime `dataType/gateDataType/stage` 分派。

## 7. Kernel 设计

### 7.1 计算流程

L2 先做 contiguous 与 layout 规范化。PREPARE 直接按 `GK_T/BETA_T` 读取 gate 和 beta，生成
Aqk/Akk/qg/kg/w seed，并在 kernel 内完成必要的 scale/cast；POST_WU 完成 Akk@W/U；FWD_H 更新
h/v_new/final_state；OUTPUT 计算 qg@h 与 Aqk@v_new 并合并 o。

KDA 没有 GDN 的逐 token 标量 gate `g`，只使用逐 key gate `gk`。host 为独立 FWD_H 构造全零
标量 gate 作为中性因子，同时传入 `gk` 并设置 `use_exp2=true`；该路径不把 `gk` 冒充为不同
shape/语义的 `g`。

### 7.2 内存规划

PREPARE user workspace 包含每 core 两槽、三 plane 的 score scratch，以及每 core 5 个
`chunk_size * chunk_size` FP32 solve slot；OUTPUT 使用两个 FP32 output plane。跨阶段中间 tensor 由
executor 显式持有。八个反向中间量始终采用 BNSD/NTD 内部布局并直接承担阶段间 GM 存储；
`return_intermediate=false` 时 aclnn 仍创建完整隐藏缓冲区供后续阶段读取，但 Python 边界返回 `None`。
`o` 由 OUTPUT kernel 在 GM 搬出时直接写成调用方 BSND/TND 或 BNSD/NTD 布局；`final_state` 由
FWD_H kernel 直接写公开状态布局，不再追加 layout-swap kernel。

| 层级/资源 | 生命周期与所有权 |
| --- | --- |
| GM/Workspace | 输入输出区间由 tiling 固定；可变输入、partial 或临时区均有唯一写 owner 和确定消费时点 |
| UB | 按 tile 或双缓冲 slot 独占；生产者写完再交给 Vector/搬出阶段，消费者结束后释放 |
| MTE2/V/MTE3 事件 | 每个 slot 的 load、compute、store 和反向复用事件闭环，EventID 不跨未完成轮次复用 |

### 7.3 流水与同步

PREPARE 的 AIV producer 与 AIC consumer 使用深度 2 的 ready/free 双向 cross-core flag；空 payload
也完成握手，队列排空后才复用 flag。MTE2/V/MTE3、Cube/Fixpipe 生命周期由事件闭环，
`--cce-auto-sync=off`。

四个 kernel 由同一 stream 的 launch 顺序建立阶段间 GM 依赖，每个 kernel 结束即形成阶段边界，
不需要跨阶段 `SyncAll`。阶段内部仍使用细粒度 ready/free、cross-core flag 和 pipe event；每个 kernel
独立初始化并闭环自身 flag 生命周期。A5 VEC 流水参考
[GroupNormSwishGrad regbase 实现](https://gitcode.com/cann/ops-nn/pull/4803/diffs)，使用双链路发射并保持
V/MTE3 依赖闭环。

### 7.4 边界处理

定长尾块与变长序列尾段均按每条逻辑序列的有效长度计算，任何补齐元素在参与指数、矩阵乘或归约前使用中性值或 mask，并按公开输出语义写零。非法累计长度和索引由 host 拦截。

## 8. 平台设计

| 平台 | SOC | 路径 | 验证要求 |
| --- | --- | --- | --- |
| A2 | `ascend910b` | 公共实现 | 构建、全量精度、性能、通路 |
| A3 | `ascend910_93` | 公共实现 | 构建、全量精度、性能、通路 |
| A5 | `ascend950` | `arch35/` 存在时使用特化，否则公共实现 | 构建、全量精度、性能、通路 |

平台差异不得改变公开 shape/dtype/layout 语义；若某模板在平台上不可用，应在 host 明确报错并同步更新 README 与 JSON。

A5 状态 kernel 的 UB 规划与 A2 不同。V=128/256 的 gate 临时区、h tile、v_new tile 和 workspace
subblock 使用互不重叠的固定区域；gate 行缩放使用 `Brcb + Mul`，宽 V tile 使用 16 行大块搬运。
任何 A5 本地内存调整都要覆盖 C64/C128、V128/V256、g/gk 与 FP16/BF16 gate 组合，并检查重复运行
二进制一致性，防止把 UB alias 或同步 race 误判为随机精度波动。

## 9. 精度设计

FP16/BF16 路径在矩阵乘或长归约中使用实现允许的高精度中间量；输出边界再转换到公开 dtype。逐项 reference、尾块、边界和组合模式按 JSON 阈值验证。 参考实现与阈值由 `tests/op_cases/chunk_kda_fwd.json` 固定。

## 10. 性能设计

以 JSON performance case 为固定矩阵，用 msopprof 记录设备侧 kernel duration、搬运与等待，并对比当前主线 Ascend C 基线检查性能回退。 不得使用 Python wall time 下结论。

## 11. 测试设计

`tests/op_cases/chunk_kda_fwd.json` 是唯一 case 规格。`tests/operators/chunk_kda_fwd/accuracy/` 运行主精度与泛化矩阵，
`routes/` 固化 aclnn 契约，`examples/fast_kernel_launch_example/tests/chunk_kda_fwd/` 执行真实 `<<<>>>`
通路。标签覆盖 accuracy、generalization、boundary、negative、route、performance、example；A2/A3/A5
使用相同 case ID，平台只决定编译与运行目标。静态 UT 额外检查公共 schema 不含阶段参数、tiling data
不含 runtime dtype/phase 字段，并确认四个 KDA phase 与 safe gate 都实例化为编译期模板。精度用例
覆盖 `safe_gate=true` 的大负累计 `gk`、中间量和最终输出，直调用例覆盖 true/false 两种入口值。

## 12. 已知限制与演进计划

- chunk_size 仅支持 64/128；K/V 均须在 [16,256] 且为 16 的倍数；交付矩阵覆盖 K=128、V=128/256。
- H_k/H_v 必须在 [1,128] 且 H_v % H_k == 0；TND 仅支持 H_k=1，多 head rank3 使用 NTD。
- 变长序列的 cu_seqlens 至少含首尾、非递减且末项等于 T；单次最多 1024 条逻辑序列。
- 显式 chunk_indices 必须完整、合法并严格采用 sequence-major 规范顺序。
- safe_gate 支持 false/true。raw gate 应先调用 kda_gate_cumsum；transpose_state_layout 当前必须为 false。

后续扩展任何限制时，必须同时修改 host 拦截、API 错误码、README、JSON 与三平台回归；模板实例增长前先评估二进制体积和编译耗时。
