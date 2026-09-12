# Chunk 间有依赖的算子开发参考

> 开发参考角色：`CHUNK_DEPENDENT`
>
> 开发参考版本：`V1`
>
> 当前来源：[PR #370](https://github.com/flashserve/flash-linear-attention-npu/pull/370)
>
> 当前参考算子：`ChunkGatedDeltaRuleFwdH`
>
> 当前参考目录：`fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/`
>
> 提炼来源 commit：`a14961e9dfe8c8264768d67fb447101873da4ecf`

本参考用于实现“后一个 chunk 需要前一个 chunk 的状态或中间结果”的算子。当前算子的公式、Stage、shape、资源数量和支持范围由已评审的 `docs/design.md` 决定；本参考提供从详设落到 host tiling、AIC/AIV 调度、workspace、同步和工程代码的方法。

## 1. 参考架构

当前参考算子的四个 Stage 为：

| Stage | 执行单元 | 计算 | 主要依赖 |
| --- | --- | --- | --- |
| S0 | Cube | `P = w @ h_prev` | 当前 chunk 的输入和上一 chunk 的状态 |
| S1 | Vector | `v_new = u - P`，并处理可选 gate | S0 |
| S2 | Cube | `delta_h = k_or_kg^T @ v_new` | S1 |
| S3 | Vector | `h_next = gate_last * h_prev + delta_h` | S2 |

实现中的关键约束是：同一序列的 `h_next` 产生后，下一 chunk 才能读取它。新算子把自己的 Stage 映射到相同结构时，先在设计文档中列清每个 Stage 的执行单元、输入落点、输出落点、同步条件和状态更新点，再按该表编写代码。

## 2. 任务与调度

1. host 侧把 fixed/varlen 输入统一转换为可直接执行的 chunk 任务，任务包含序列、chunk、token 起点、有效长度、head、状态和 workspace offset。
2. 同一序列的 chunk 按依赖顺序执行；不同序列或互不依赖的 head 可以并行。
3. active core 采用连续且均衡的 head 分片。当前参考先计算每核基础 head 数，再把余数依次分给前面的 core。
4. 每核按 head round 推进一个 chunk。当前参考每轮最多处理 4 个 head；一个 core 拥有更多 head 时执行多轮。
5. 当前 chunk 的全部 head round 完成状态更新后，再进入下一 chunk，避免下一 chunk 读到不完整状态。

目标算子的每轮 head 数、active core 数和任务顺序按自身容量与性能设计重新推导。

## 3. AIC/AIV 实现

Cube 与 Vector 使用独立的实现类和入口分支：

- AIC 按 round 执行 Cube Stage：等待当前窗口可复用，完成前一矩阵 Stage，通知 AIV；等待 Vector 中间结果后完成后一矩阵 Stage，再次通知 AIV。
- AIV 按相同任务顺序执行 Vector Stage：等待 AIC 输出，完成中间 Vector 计算并通知 AIC；等待后一 Cube 输出，完成状态更新并释放当前窗口。
- AIC 与 AIV 使用同一任务描述和相同的 chunk/head/round 次序，使固定 EventID 能按顺序复用。
- 矩阵 tail 继续走 Cube 主路径；无效区域通过 padding 或 mask 处理。Vector 只访问有效行并按设计写回有效区。

参考算子的核心顺序为：

```text
等待窗口空闲
  -> Cube S0
  -> 通知 S0 完成
  -> Vector S1
  -> 通知 S1 完成
  -> Cube S2
  -> 通知 S2 完成
  -> Vector S3
  -> 标记窗口可复用并形成 h_next
  -> 下一 chunk
```

## 4. workspace 与状态

当前参考采用两套窗口做 ping-pong，每套窗口为一轮中的 head 预留独立 slot；两套窗口共 8 个 workspace slot：

```text
slot = round_bank * 4 + round_head_offset
```

每套窗口维护四类跨核状态：

- `cube1Done`：前一 Cube Stage 已完成。
- `vec1Done`：中间 Vector Stage 已完成。
- `cube2Done`：后一 Cube Stage 已完成。
- `vec2Done`：状态更新已完成，窗口可以复用。

目标算子按自己的并发 head 数和中间张量大小计算窗口数、slot 数及 workspace 大小。每个 slot 在 producer 写入前等待上一消费者释放；最终状态与中间 workspace 使用不同 offset 和生命周期。

## 5. host tiling 与模板

host tiling 完成以下工作：

1. 校验 shape、dtype、layout、head 关系、chunk size、fixed/varlen 输入和状态张量。
2. 对 varlen 输入压缩空序列，计算每个有效 chunk 的 token 和状态 offset。
3. 计算 active core、每核 head 范围、round 数、窗口和各 workspace 区域。
4. 把 kernel 热路径需要的 offset、有效长度和模式写入 tiling data。
5. TilingKey 只编码会改变编译路径的组合。当前参考按 Vector tile、gate 模式和指数模式选择模板；dtype 通过生成宏进入模板。

模板实例与 host 选择条件保持一一对应。固定长度、变长、初始状态、最终状态和 tail chunk 共用同一任务描述格式。

## 6. 同步实现

1. 初始化每套窗口的 ready/free 状态，使首轮可以进入。
2. AIC 和 AIV 对每个 head round 使用相同的 flag 次序。
3. producer 完成 MTE3/Fixpipe 写入后再发布跨核 flag；consumer 等待 flag 后再读取目标地址。
4. consumer 完成最后一次读取后发布释放信号，下一轮才能覆盖该窗口。
5. kernel 退出前消费仍处于已发布状态的本地事件，使每个 EventID 的 Set/Wait 生命周期闭合。

设计文档中的同步表应能逐项对应代码中的初始化、等待、发布、消费和复用位置。

## 7. 从设计生成代码

按以下顺序实现：

1. 在 tiling struct 中定义任务、状态、workspace、模板和模式字段。
2. 在 host tiling 中完成参数校验、任务数量、head 分片、offset、workspace 和 TilingKey 计算。
3. 实现共享 scheduler，统一产生 AIC/AIV 使用的 chunk、head、round、window 和 slot。
4. 实现 Cube Stage 和其 L1/L0 double buffer，按同步表连接 Vector producer/consumer。
5. 实现 Vector Stage 和其 UB resident、mask、MTE2/MTE3 流水，完成状态更新和窗口释放。
6. 在 kernel 入口按 AIC/AIV 和 TilingKey 分发模板实例。
7. 接入 op_host、InferShape、op_api、schema、Python wrapper、构建和测试入口。

## 8. 定点查看当前实现

本参考缺少具体 API 或类名时，按问题读取当前来源中的对应文件：

| 需要确认的内容 | 文件 |
| --- | --- |
| 算子行为、shape 和调度总览 | `README.md` |
| 任务字段、offset 和数据结构 | `op_kernel/chunk_gated_delta_rule_fwd_h_struct.h` |
| chunk/head/round/window 调度 | `op_kernel/chunk_gated_delta_rule_fwd_h_scheduler.h` |
| AIC Stage、Cube 流水和同步 | `op_kernel/arch35/chunk_gated_delta_rule_fwd_h_cube.h` |
| AIV Stage、Vector 流水和同步 | `op_kernel/arch35/chunk_gated_delta_rule_fwd_h_vector.h` |
| host 校验、workspace 和模板选择 | `op_host/chunk_gated_delta_rule_fwd_h_tiling_processor.h` |
| kernel 模板分发 | `op_kernel/chunk_gated_delta_rule_fwd_h.cpp` |

## 9. 版本维护

同一来源中改进实现细节时，更新“提炼来源 commit”和受影响章节，开发参考版本保持不变。来源 PR、参考算子或核心调度架构更换时，开发参考版本升级为 `V2`、`V3`，并增加迁移说明。算子设计文档记录实际采用的版本、commit、采用内容和差异。
