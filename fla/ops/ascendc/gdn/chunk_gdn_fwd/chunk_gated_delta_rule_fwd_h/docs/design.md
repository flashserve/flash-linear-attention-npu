# ChunkGatedDeltaRuleFwdH 设计方案

## 1. 背景

Gated Delta Rule 的跨 chunk 状态推进算子。它根据 K/W/U、标量 gate 或逐 K 维 gate，从 initial_state 递推各 chunk 起始状态 H、修正值 V_new，并按需返回 final_state。 本实现通过统一 aclnn 与 ctypes 稳定入口接入 Gated Delta Network (GDN) 链路。

## 2. 目标与非目标

### 2.1 目标

- README 所列 `定长/变长序列、g/gk、可选 initial/final state` 场景精度与仓内参考实现一致。
- A2/A3/A5 均能构建和执行，`--cce-auto-sync=off`。
- 若本算子用于替换 Triton，同一 shape/dtype/layout 下 Ascend C 性能优于被替换实现。
- aclnn、`fla_npu.ops.ascendc`、`<<<>>>` 使用同一接口语义。

### 2.2 非目标

- 不扩展 README“已知限制”之外的 shape、预留参数或 layout。
- 不以 fallback、放宽阈值或跳过失败 case 替代 kernel 根因修复。

## 3. 能力边界

实现类型：`ascendc`。Dtype：K/W/U 为 FP16/BF16；gate/state 可为 FP32。Layout：BNSD；initial/final state 为 `[N,H_v,K,V]`。模式：定长/变长序列、g/gk、可选 initial/final state。
Shape 符号统一引用[GDN 模型符号表](../../../README.md#model-shape-symbols)。

## 4. 数学与接口语义

令 `S_c` 为 chunk c 的起始状态，`G_c` 为该 chunk 末端门控：

```text
V_new_c = U_c - W_c @ S_c
S_(c+1) = decay(G_c) * S_c + K_gated_c^T @ V_new_c
H[c]    = S_c
```

`g` 路径使用每 head 标量衰减，`gk` 路径对 K 维逐元素衰减；二者至少提供一个。

Python、aclnn 与直调入口均以同一逻辑 shape 解释输入。接口层转换只处理连续性、描述符或文档明确的
layout 规范化，不改变公式、边界 mask、head 映射或可选输入语义。

## 5. 整体架构

1. `op_host/*_def.cpp` 注册输入输出、dtype、属性和 A2/A3/A5。
2. InferShape、op_api 与 tiling host 共同按 README 校验必选参数、shape、dtype、layout、属性和可选输入组合，并构造或核对输出。
3. tiling processor 计算任务数、边界块、workspace 偏移和模板实例。
4. `op_kernel/` 按本算子的计算流程完成搬运、计算、同步和写回。
5. aclnn 两段式接口负责 contiguous、workspace/executor 和 stream 异步发射。
6. `fla_npu.ops.ascendc` 仅通过 ctypes 调用 aclnn，不依赖 torch_npu dispatcher。

## 6. Tiling 设计

### 6.1 任务划分

每个 `(sequence,value_head)` 是有序状态链；不同序列/head 并行，同一链的 chunk 按先后顺序执行并写 H。

### 6.2 Tiling Data

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `batch` | `int64_t` | host 计算并由 kernel 消费 |
| `seqlen` | `int64_t` | host 计算并由 kernel 消费 |
| `kNumHead` | `int64_t` | host 计算并由 kernel 消费 |
| `vNumHead` | `int64_t` | host 计算并由 kernel 消费 |
| `kHeadDim` | `int64_t` | host 计算并由 kernel 消费 |
| `vHeadDim` | `int64_t` | host 计算并由 kernel 消费 |
| `chunkSize` | `int64_t` | host 计算并由 kernel 消费 |
| `useInitialState` | `bool` | host 计算并由 kernel 消费 |
| `storeFinalState` | `bool` | host 计算并由 kernel 消费 |
| `dataType` | `int64_t` | host 计算并由 kernel 消费 |
| `gDataType` | `int64_t` | host 计算并由 kernel 消费 |
| `stateDataType` | `int64_t` | host 计算并由 kernel 消费 |
| `hasGk` | `bool` | host 计算并由 kernel 消费 |
| `isVariedLen` | `int64_t` | host 计算并由 kernel 消费 |
| `shapeBatch` | `int64_t` | host 计算并由 kernel 消费 |
| `tokenBatch` | `int64_t` | host 计算并由 kernel 消费 |
| `useGk` | `bool` | host 计算并由 kernel 消费 |
| `vWorkspaceOffset` | `int64_t` | host 计算并由 kernel 消费 |
| `vUpdateWorkspaceOffset` | `int64_t` | host 计算并由 kernel 消费 |
| `kDecayWorkspaceOffset` | `int64_t` | host 计算并由 kernel 消费 |
| `hWorkspaceOffset` | `int64_t` | host 计算并由 kernel 消费 |
| `numSeqWorkspaceOffset` | `int64_t` | host 计算并由 kernel 消费 |
| `numChunksWorkspaceOffset` | `int64_t` | host 计算并由 kernel 消费 |

### 6.3 模板化方案与 tiling key

仅保留 2 个 key：V<=128 选择 key 1 的 128 列 Cube tile，128<V<=256 选择 key 2 的 256 列 tile。必须使用 key 是因为两种 Cube tile 和 KERNEL_TASK_TYPE 在编译期不同；dtype、g/gk、initial/final、定长/变长序列均由 tiling data 处理，不继续扩张 key。

## 7. Kernel 设计

### 7.1 计算流程

AIC 计算 W@S 与 K^T@V_new，AIV 计算门控衰减、V_new 和状态合并；逐 chunk 推进，最后按属性写 final_state。

### 7.2 内存规划

状态 S 在 GM/工作区按 sequence/head 独占，L1/L0 放两个矩阵乘 tile，UB 保存 gate、V_new 和状态逐元素更新片段。

| 层级/资源 | 生命周期与所有权 |
| --- | --- |
| GM/Workspace | host 按 tiling 固定地址和容量；每个阶段写入区间与消费者、复用时点一一对应 |
| L1/L0A/L0B/L0C | 当前 Cube tile 独占；MTE1/Cube/Fixpipe 完成并由下一阶段消费后才允许复用 |
| UB | 当前 Vector tile 独占；MTE2/V/MTE3 或 AIC-AIV 交接完成后释放 |
| 事件/flag | 按 buffer slot 成对分配 ready/free；禁止未 wait 连续 set 或跨 slot 误复用 |

### 7.3 流水与同步

同一状态链由单一 owner 保证 chunk 顺序；AIC/AIV 通过 ready flag 交接 W@S 和 K^T@V。核内事件闭环，`--cce-auto-sync=off`。

### 7.4 边界处理

定长尾块与变长序列尾段均按每条逻辑序列的有效长度计算，任何补齐元素在参与指数、矩阵乘或归约前使用中性值或 mask，并按公开输出语义写零。非法累计长度和索引由 host 拦截。

## 8. 平台设计

| 平台 | SOC | 路径 | 验证要求 |
| --- | --- | --- | --- |
| A2 | `ascend910b` | 公共实现 | 构建、全量精度、性能、通路 |
| A3 | `ascend910_93` | 公共实现 | 构建、全量精度、性能、通路 |
| A5 | `ascend950` | `arch35/` 存在时使用特化，否则公共实现 | 构建、全量精度、性能、通路 |

平台差异不得改变公开 shape/dtype/layout 语义；若某模板在平台上不可用，应在 host 明确报错并同步更新 README 与 JSON。

## 9. 精度设计

FP16/BF16 路径在矩阵乘或长归约中使用实现允许的高精度中间量；输出边界再转换到公开 dtype。逐项 reference、尾块、边界和组合模式按 JSON 阈值验证。 参考实现与阈值由 `tests/op_cases/chunk_gated_delta_rule_fwd_h.json` 固定。

## 10. 性能设计

以 JSON performance case 为固定矩阵，用 msopprof 记录设备侧 kernel duration、搬运与等待，并对比当前主线 Ascend C 基线检查性能回退。 不得使用 Python wall time 下结论。

## 11. 测试设计

`tests/op_cases/chunk_gated_delta_rule_fwd_h.json` 是唯一 case 规格。`tests/operators/chunk_gated_delta_rule_fwd_h/accuracy/` 运行主精度与泛化矩阵，
`routes/` 固化 aclnn 与 `<<<>>>` 编译/发射契约。标签覆盖 accuracy、generalization、boundary、negative、route、performance、example；
A2/A3/A5 使用相同 case ID，平台只决定编译与运行目标。

## 12. 已知限制与演进计划

- `K` 仅支持 128，`V` 仅支持 128/256，`chunk_size` 仅支持 64/128。
- `g` 与 `gk` 至少提供一个；`H_v % H_k == 0`。
- 变长序列当前仅支持物理 `B=1`，索引必须完整且 sequence-major。
- `save_new_value=true`、`use_exp2=false`、`transpose_state_layout=false`。

后续扩展任何限制时，必须同时修改 host 拦截、API 错误码、README、JSON 与三平台回归；模板实例增长前先评估二进制体积和编译耗时。
