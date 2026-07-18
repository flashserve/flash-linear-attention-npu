# <op_name> 设计方案

<!-- 将占位符替换为实际内容，删除不适用章节。Hint 仅用于展示填写粒度，完成文档后应删除。 -->

## 1. 背景

说明需求来源、现有实现及其问题。

> **Hint：** 以 `chunk_bwd_dv_local` 为例，应交代该算子属于 Gated Delta Rule 反向链路，负责计算 `dV_local`，并说明引入 Ascend C 实现要解决的功能覆盖或 Triton 性能问题。

## 2. 目标与非目标

### 2.1 目标

- `<functional target>`
- `<performance target>`
- `<platform target>`

### 2.2 非目标

- `<out-of-scope item>`

> **Hint：** `chunk_bwd_dv_local` 的目标可以包括 fixed/varlen、FP16/BF16、A2/A3/A5 和性能优于 Triton；非目标应明确列出当前不实现的 `g_gamma`、`A` 语义，而不是留空。

## 3. 能力边界

实现类型：`<ascendc|triton>`

说明支持的 shape、dtype、format/layout、属性、fixed/varlen、状态输入输出和不支持场景。

> **Hint：** `chunk_bwd_dv_local` 的实现类型为 `ascendc`。能力边界中的 shape 写为 `q/k=[B,H_k,T,K]`、`dO/dV=[B,H_v,T,V]`、`g=[B,H_v,T]`，并说明 `H_v % H_k == 0`；固定维度和 varlen batch 限制统一写入“已知限制与演进计划”。

## 4. 数学与接口语义

给出数学公式、维度关系、输入输出语义、边界行为和异常条件。

本文不重复定义 Shape 符号。公式和语义说明使用的变量统一引用[算子 README 的 Shape 变量说明附录](../README.md#shape-symbols)，并与所属模型的权威符号表保持一致。

> **Hint：** 除 `dV_local = Ws_gated @ dO` 外，还应写清 `qkHead = doHead / hRatio`、门控指数的行列方向、上三角是否含对角线，以及 partial chunk 的有效区域。

## 5. 整体架构

Ascend C 算子说明 op_host、InferShape、tiling、kernel、aclnn 和 `fla_npu` 各层职责及调用关系；Triton 算子说明 Python wrapper、Triton kernel、grid/config 和 launch 关系。

> **Hint：** `chunk_bwd_dv_local` 可说明 op_host 校验 shape 并生成 tiling，kernel 使用 AIC/AIV 混合任务：AIC 计算 `K @ Q^T`，AIV 完成 gating/mask，AIC 再计算 `Ws_gated @ dO`，workspace 用于阶段间数据交换。

### 5.1 算子边界与 L2/L0 分工

逐项说明：

- 一个完整 L0 入口内各语义 phase 的强生产消费关系，或拆成多个独立 L0 的必要性。
- 公共 aclnn、`fla_npu`、`<<<>>>`、legacy 接口是否完全隐藏内部 phase；不得出现要求调用者传入的 `stage`/`stage_id`。
- L2 负责的参数校验、layout/view、workspace/executor 和 launch 编排。
- 输入、阶段间和输出 cast 融入哪个生产者或消费者 kernel；L2 不得拼接独立 Cast L0。
- 中间 tensor/workspace owner、dtype、layout、stream 顺序和生命周期；不得用 stage 编号替代所有权协议。
- 同一 L0 内有多个紧耦合 phase 时，`SyncAll` 的参与核、schedule mode、A2/A3/A5 支持，以及 `TPipe::Reset()` 前排空和后续重建的 queue/buffer；Reset 与 SyncAll 的先后要按真实依赖说明。

> **反面样例：KDA 当前实现。** `ChunkKdaFwd(stage=1/3/2)` 把关系不强的分支放进同一 L0，直调调用者需要理解 stage，L2 还拼接 gate/阶段间 cast。新增算子不得照搬；整改应优先收敛为一个完整入口下的两个语义 phase，在 L0 内完成同步和 `TPipe::Reset()` 隔离，并把 cast 融进 kernel；若无共同归并语义或片上复用价值，再拆成独立 L0。

> **参考样例：** [ops-nn PR #4803 的 GroupNormSwishGrad A5 实现](https://gitcode.com/cann/ops-nn/pull/4803/diffs)对外保持一个完整 kernel 入口，内部第一 phase 生成输出和 workspace，调用 `pipe.Reset()` 释放本地资源，在第二 phase 消费 workspace 前执行 `SyncAll()`，随后重建 buffer 并在 kernel 内完成 reduce/cast。填写时需结合本算子的参与核和数据依赖说明，不能只写“参考该实现”。

## 6. Tiling 设计（仅 Ascend C 算子）

Triton 算子删除本章节，并在整体架构或 Kernel 设计中说明其 grid/config 选择策略。

### 6.1 任务划分

说明 block dim、核间任务分配、尾块和空任务处理。

> **Hint：** `chunk_bwd_dv_local` 的 fixed 场景可按 `B * ceil(T/chunk_size)` 个 chunk 分配任务，`usedCoreNum = min(chunkNumForT * B, coreNum)`；varlen 则由 `chunk_indices` 给出全局 chunk 列表。

### 6.2 Tiling Data

| 字段 | 类型 | 含义 | 取值范围 |
| --- | --- | --- | --- |
| `<field>` | `<type>` | `<description>` | `<range>` |

> **Hint：** `ChunkBwdDvLocalTilingData` 当前包含 `b`、`hQk`、`hDo`、`hRatio`、`headBufNum`、`t`、`k`、`v`、`chunkSize`、`chunkNumForT` 和 `scale`。模板中应逐字段解释其来源和消费者。

### 6.3 模板化方案

说明模板参数、实例选择条件，以及关键 shape/layout/dtype 分支如何映射到模板。

如使用 tiling key，必须说明模板化方案无法满足需求的原因、key 语义、组合数量、增长上限，以及对二进制体积和编译耗时的影响；未使用时写“未使用 tiling key”。

> **Hint：** `chunk_bwd_dv_local` 使用 `strategy`、`D_T_Q`、`D_T_G`、`V` 四个模板维度，分别覆盖 fixed/varlen、Q/K dtype、gate dtype 和受支持的 `V` 实例。若通过框架生成的 tiling key 选择模板实例，应说明它只承载这些有限模板维度，不能把普通 runtime shape 塞入 key。

## 7. Kernel 设计

### 7.1 计算流程

按执行顺序说明数据搬运、计算和写回过程。

> **Hint：** `chunk_bwd_dv_local` 可用“score 生成”“gate/mask 修饰”“dV 生成”等语义名称说明同一 kernel 内的 phase，并说明 `hRatio` 下 score 的共享方式；不要把 phase 编号设计成公开 `stage` 属性。

### 7.2 内存规划

| 内存层级 | Buffer | 大小/对齐 | 生命周期 | 用途 |
| --- | --- | --- | --- | --- |
| GM/Workspace | `<buffer>` | `<size>` | `<lifetime>` | `<usage>` |
| L1 | `<buffer>` | `<size>` | `<lifetime>` | `<usage>` |
| UB | `<buffer>` | `<size>` | `<lifetime>` | `<usage>` |
| L0A/L0B/L0C | `<buffer>` | `<size>` | `<lifetime>` | `<usage>` |

> **Hint：** `chunk_bwd_dv_local` 的 user workspace 大小可说明为 `2 * usedCoreNum * headBufNum * chunkSize * chunkSize` 字节，并解释每个 slot 在 AIC/AIV 阶段间的生产、消费和复用时机；不能只给总字节数。

### 7.3 流水与同步

Ascend C 算子说明 MTE/Cube/Vector/Fixpipe 流水、event/flag、buffer slot 复用、跨核依赖及 ready/free 协议，且 `--cce-auto-sync` 必须保持为 `off`。Triton 算子说明对应的 load/compute/store 流水和同步边界。

> **Hint：** 对 `chunk_bwd_dv_local`，应逐项写明 AIC 写 workspace 后如何通知 AIV、AIV 完成 gating 后如何通知第二阶段 AIC，以及 producer 复用 slot 前如何确认 consumer 已完成。实际 event/flag 名称和数量必须与源码一致。

若多个 phase 位于同一 L0 kernel，还必须写明 `AscendC::SyncAll<false>()` 的一致参与条件、schedule mode 和平台支持，以及 `pipe->Reset()` 前已完成的异步操作、被释放的本地资源和 Reset 后重新初始化的 queue/buffer。若拆成独立 L0，则说明同一 stream 顺序和显式中间量如何建立依赖，不写不存在的 kernel 间 `SyncAll`。

### 7.4 边界处理

说明非整除尾块、padding、无效区域、最小/最大 shape 和特殊值处理。

> **Hint：** `chunk_bwd_dv_local` 应说明 `T % chunk_size != 0` 时最后一个 chunk 的 mask、varlen 每段序列尾块、上三角 mask 和无效输出区域如何处理。

## 8. 平台设计

| 平台 | SOC | 实现策略 | 模板/分支差异 | 限制 |
| --- | --- | --- | --- | --- |
| A2 | `ascend910b` | `<strategy>` | `<difference>` | `<limitation>` |
| A3 | `ascend910_93` | `<strategy>` | `<difference>` | `<limitation>` |
| A5 | `ascend950` | `<strategy>` | `<difference>` | `<limitation>` |

> **Hint：** `chunk_bwd_dv_local` 的 A2/A3 可填写公共实现，A5 若命中 `arch35/` 特化，应说明 regbase Vector 实现与公共路径的差异，并分别给出构建、精度和性能结论。

## 9. 精度设计

说明累加精度、类型转换、数值稳定性风险、特殊值处理、标杆选择和阈值依据。

所有公开输入到内部计算 dtype、阶段间 dtype 和内部结果到公开输出 dtype 的 cast 都必须定位到具体 kernel 的读入/写回或片上计算步骤；不得写成由 L2 调用独立 Cast 完成。

> **Hint：** `chunk_bwd_dv_local` 需要关注 `exp(g_col-g_row)` 的溢出/下溢、FP16/BF16 输入的累加类型、FP32 gate 路径，以及 fixed/varlen 标杆的一致性。

## 10. 性能设计

说明目标 shape、Triton 基线、主要瓶颈、优化策略和 profiling 指标。Ascend C 替换 Triton 时，目标场景性能必须优于 Triton。

> **Hint：** `chunk_bwd_dv_local` 应分别记录全部受支持 `chunk_size`、全部受支持 `V`、不同 `hRatio` 和 fixed/varlen 的耗时，并分析 Cube 利用率、Vector gating 和 workspace 同步是否成为瓶颈。

## 11. 测试设计

关联 `tests/op_cases/<op_name>.json`，说明主精度、泛化、边界、异常、回归、通路、性能和 A2/A3/A5 测试矩阵。

> **Hint：** `chunk_bwd_dv_local` 的 case 标签至少应能筛出 fixed、varlen、tail、GVA/head-ratio、各受支持 `V`、FP32 gate、negative、route 和 performance 场景。

## 12. 已知限制与演进计划

- `<limitation>`
- `<follow-up plan>`

> **Hint：** `chunk_bwd_dv_local` 在这里集中说明 `K` 仅支持 128、`V` 仅支持 128/256、`chunk_size` 仅支持 64/128、varlen 仅支持 `B=1`，以及 `g_gamma`、`A` 当前必须为空；这些固定值不写入前文 Shape。
