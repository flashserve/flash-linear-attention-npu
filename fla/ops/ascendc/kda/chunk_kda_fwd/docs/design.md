# ChunkKdaFwd 设计方案

## 1. 背景

Kimi Delta Attention 正向主算子。它消费已经按 chunk 累加的 key gate `gk`，分阶段生成 chunk 内矩阵项、递推状态和最终输出，并可返回完整中间量用于训练链路与精度定位。 本实现通过统一 aclnn 与 ctypes 稳定入口接入 Kimi Delta Attention (KDA) 链路。

## 2. 目标与非目标

### 2.1 目标

- README 所列 `dense/varlen、四种显式 layout、可选初始/最终状态、可选中间量` 场景精度与仓内参考实现一致。
- A2/A3/A5 均能构建和执行，`--cce-auto-sync=off`。
- 若本算子用于替换 Triton，同一 shape/dtype/layout 下 Ascend C 性能优于被替换实现。
- aclnn、`fla_npu.ops.ascendc`、`<<<>>>` 使用同一接口语义。

### 2.2 非目标

- 不扩展 README“已知限制”之外的 shape、预留参数或 layout。
- 不以 fallback、放宽阈值或跳过失败 case 替代 kernel 根因修复。

## 3. 能力边界

实现类型：`ascendc`。Dtype：q/k/v 为同一 FP16 或 BF16；gk/beta 为 FP32 或 BF16，当前实现在 L2 转 FP32；状态为 FP32。Layout：BSND/BNSD/TND/NTD；BNSD/NTD 为内部性能布局，BSND/TND 通过 KdaLayoutSwap12 转换。模式：dense/varlen、四种显式 layout、可选初始/最终状态、可选中间量。
Shape 符号统一引用[算子 README 的 Shape 变量说明](../README.md#shape-symbols)。

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
5. aclnn 两段式接口负责 contiguous、workspace/executor 和 stream 异步发射；当前内部布局中间结果通过 `ViewCopy` 回写，BSND/TND 外部布局转换调用 `KdaLayoutSwap12`，这些现状均需随架构整改重新收口。
6. `fla_npu.ops.ascendc` 仅通过 ctypes 调用 aclnn，不依赖 torch_npu dispatcher。

### 5.1 算子边界与 L2/L0 分工

当前实现通过 `stage=1/3/2` 复用同一 L0，并由 L2 拼接 gate cast、阶段间 cast/scale 和状态 kernel；
`<<<>>>` 调用者需要理解内部 stage，属于开发规范明确列出的反面样例。整改应优先收敛为一个完整入口下
的两个语义 phase，在 L0 内闭合必要的全核同步和 `TPipe::Reset()`，并把输入、阶段间和输出 cast
融入 kernel；若两个 phase 没有共同归并语义或片上复用价值，再拆成两个语义独立的 L0。

正面结构参考 [ops-nn PR #4803 的 GroupNormSwishGrad A5 实现](https://gitcode.com/cann/ops-nn/pull/4803/diffs)：
公共 kernel 不暴露 stage，第一 phase 生成输出和 workspace，`pipe.Reset()` 后在第二 phase 消费 workspace
前执行 `SyncAll()`，再重建 buffer 并在 kernel 内 reduce/cast。KDA 重构必须结合自身参与核和状态依赖
重新证明同步顺序，不能机械复制。

## 6. Tiling 设计

### 6.1 任务划分

stage1/3/2 按 (sequence,value_head,chunk) 分配无跨 chunk 的矩阵任务；状态传播复用 ChunkGatedDeltaRuleFwdH 并保持同一序列的 chunk 顺序。varlen tiling 保存每序列起止与累计 chunk offset，不按每个 chunk 膨胀。

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
| `dataType` | `int64_t` | host 计算并由 kernel 消费 |
| `gateDataType` | `int64_t` | host 计算并由 kernel 消费 |
| `usedCoreNum` | `int64_t` | host 计算并由 kernel 消费 |
| `stage` | `int64_t` | host 计算并由 kernel 消费 |
| `seqStart` | `int64_t[KDA_MAX_TILING_SEQUENCES]` | host 计算并由 kernel 消费 |
| `seqEnd` | `int64_t[KDA_MAX_TILING_SEQUENCES]` | host 计算并由 kernel 消费 |
| `seqChunkOffset` | `int64_t[KDA_MAX_TILING_SEQUENCE_OFFSETS]` | host 计算并由 kernel 消费 |

### 6.3 模板化方案与 tiling key

公开接口只设置 key=1，用于选择 AIC:AIV=1:2 的 mixed task kernel 类型；它不编码 B/H/T/C/layout，也不产生 shape 组合。设备源码中的 key=0/key=2 是历史保留分支，host 已明确不可达。保留 key=1 的原因是当前 Ascend C mixed task 发射需要通过 tiling key 绑定任务类型，不能仅由普通 tiling data 替代。

## 7. Kernel 设计

### 7.1 计算流程

L2 先做 contiguous、layout 规范化与 gate cast；stage1 生成 Aqk/Akk/qg/kg/w seed，stage3 完成 Akk@W/U，GDN fwd_h 更新 h/v_new/final_state，stage2 计算 qg@h 与 Aqk@v_new 并合并 o。

### 7.2 内存规划

stage1 user workspace 包含每 core 两槽、三 plane 的 score scratch，以及每 core 5 个 C*C FP32 solve slot；stage2 使用两个 FP32 output plane。中间 tensor 由 executor 显式持有，不能把后一 stage 读取的数据只作为原地输出参数。

| 层级/资源 | 生命周期与所有权 |
| --- | --- |
| GM/Workspace | 输入输出区间由 tiling 固定；可变输入、partial 或临时区均有唯一写 owner 和确定消费时点 |
| UB | 按 tile 或双缓冲 slot 独占；生产者写完再交给 Vector/搬出阶段，消费者结束后释放 |
| MTE2/V/MTE3 事件 | 每个 slot 的 load、compute、store 和反向复用事件闭环，EventID 不跨未完成轮次复用 |

### 7.3 流水与同步

stage1 的 AIV producer 与 AIC consumer 使用深度 2 的 ready/free 双向 cross-core flag；空 payload 也完成握手，队列排空后才复用 flag。MTE2/V/MTE3、Cube/Fixpipe 生命周期由事件闭环，`--cce-auto-sync=off`。

### 7.4 边界处理

dense/fixed 尾块与 varlen 尾段均按每条逻辑序列的有效长度计算，任何补齐元素在参与指数、矩阵乘或归约前使用中性值或 mask，并按公开输出语义写零。非法累计长度和索引由 host 拦截。

## 8. 平台设计

| 平台 | SOC | 路径 | 验证要求 |
| --- | --- | --- | --- |
| A2 | `ascend910b` | 公共实现 | 构建、全量精度、性能、通路 |
| A3 | `ascend910_93` | 公共实现 | 构建、全量精度、性能、通路 |
| A5 | `ascend950` | `arch35/` 存在时使用特化，否则公共实现 | 构建、全量精度、性能、通路 |

平台差异不得改变公开 shape/dtype/layout 语义；若某模板在平台上不可用，应在 host 明确报错并同步更新 README 与 JSON。

## 9. 精度设计

FP16/BF16 路径在矩阵乘或长归约中使用实现允许的高精度中间量；输出边界再转换到公开 dtype。逐项 reference、尾块、边界和组合模式按 JSON 阈值验证。 参考实现与阈值由 `tests/op_cases/chunk_kda_fwd.json` 固定。

## 10. 性能设计

以 JSON performance case 为固定矩阵，用 msopprof 记录设备侧 kernel duration、搬运与等待，并对比当前主线 Ascend C 基线检查性能回退。 不得使用 Python wall time 下结论。

## 11. 测试设计

`tests/op_cases/chunk_kda_fwd.json` 是唯一 case 规格。`tests/operators/chunk_kda_fwd/accuracy/` 运行主精度与泛化矩阵，
`routes/` 固化 aclnn 与 `<<<>>>` 编译/发射契约。标签覆盖 accuracy、generalization、boundary、negative、route、performance、example；
A2/A3/A5 使用相同 case ID，平台只决定编译与运行目标。

## 12. 已知限制与演进计划

- chunk_size 仅支持 64/128；K/V 均须在 [16,256] 且为 16 的倍数；交付矩阵覆盖 K=128、V=128/256。
- H_k/H_v 必须在 [1,128] 且 H_v % H_k == 0；TND 仅支持 H_k=1，多 head rank3 使用 NTD。
- varlen 的 cu_seqlens 至少含首尾、非递减且末项等于 T；单次最多 1024 条逻辑序列。
- 显式 chunk_indices 必须完整、合法并严格采用 sequence-major 规范顺序。
- safe_gate 与 transpose_state_layout 当前必须为 false；raw gate 应先调用 kda_gate_cumsum。

后续扩展任何限制时，必须同时修改 host 拦截、API 错误码、README、JSON 与三平台回归；模板实例增长前先评估二进制体积和编译耗时。
