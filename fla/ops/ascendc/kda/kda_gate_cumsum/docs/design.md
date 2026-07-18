# KdaGateCumsum 设计方案

## 1. 背景

把 KDA 的逐 token gate 转换为每个 chunk 内的 log2 累积 gate `gk`。既支持已计算好的 step gate，也支持带 A_log/dt_bias 的 safe-gate raw 输入。 本实现通过统一 aclnn 与 ctypes 稳定入口接入 Kimi Delta Attention (KDA) 链路。

## 2. 目标与非目标

### 2.1 目标

- README 所列 `定长/变长序列、step-gate/safe raw-gate、四种 layout、整块/尾块` 场景精度与仓内参考实现一致。
- A2/A3/A5 均能构建和执行，`--cce-auto-sync=off`。
- 若本算子用于替换 Triton，同一 shape/dtype/layout 下 Ascend C 性能优于被替换实现。
- aclnn、`fla_npu.ops.ascendc`、`<<<>>>` 使用同一接口语义。

### 2.2 非目标

- 不扩展 README“已知限制”之外的 shape、预留参数或 layout。
- 不以 fallback、放宽阈值或跳过失败 case 替代 kernel 根因修复。

## 3. 能力边界

实现类型：`ascendc`。Dtype：g 为 FP16/BF16/FP32；A_log/dt_bias/gk 为 FP32；cu_seqlens 为 INT64。Layout：rank4 使用 BSND/BNSD，rank3 使用 TND/NTD；layout 必须显式且不根据 shape 推导。模式：定长/变长序列、step-gate/safe raw-gate、四种 layout、整块/尾块。
Shape 符号统一引用[KDA 模型符号表](../../README.md#model-shape-symbols)。

## 4. 数学与接口语义

对每条序列、value head 和 K 维，累加在 chunk 边界重新置零：

```text
step = g                                             use_gate_in_kernel=false
x    = (g + dt_bias[h_v,k]) * exp(A_log[h_v])        raw safe-gate
step = lower_bound * sigmoid(x)                      raw safe-gate
gk[t] = sum(step[chunk_start:t]) / ln(2)
```

因下游使用 `exp2(gk_i-gk_j)`，输出统一为 FP32。safe-gate 的逐步值位于
`[lower_bound,0]`，合法的长 chunk 累积可达到较大负值，不能通过收窄输入范围掩盖写回或同步问题。

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

dense 按 (B,H_v,chunk) 分配，变长序列按 (sequence,H_v) 分配并在 core 内遍历该序列真实 chunk；每个 task 的 K 行按最多 256 元素的 UB 向量处理。

### 6.2 Tiling Data

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `batch` | `int64_t` | host 计算并由 kernel 消费 |
| `t` | `int64_t` | host 计算并由 kernel 消费 |
| `hv` | `int64_t` | host 计算并由 kernel 消费 |
| `k` | `int64_t` | host 计算并由 kernel 消费 |
| `rank` | `int64_t` | host 计算并由 kernel 消费 |
| `layout` | `int64_t` | host 计算并由 kernel 消费 |
| `chunkSize` | `int64_t` | host 计算并由 kernel 消费 |
| `seqNum` | `int64_t` | host 计算并由 kernel 消费 |
| `hasCuSeqlens` | `int64_t` | host 计算并由 kernel 消费 |
| `hasALog` | `int64_t` | host 计算并由 kernel 消费 |
| `hasDtBias` | `int64_t` | host 计算并由 kernel 消费 |
| `dataType` | `int64_t` | host 计算并由 kernel 消费 |
| `useGateInKernel` | `int64_t` | host 计算并由 kernel 消费 |
| `safeGate` | `int64_t` | host 计算并由 kernel 消费 |
| `lowerBound` | `float` | host 计算并由 kernel 消费 |
| `usedCoreNum` | `int64_t` | host 计算并由 kernel 消费 |

### 6.3 模板化方案与 tiling key

不使用 tiling key 组合。dataType 与 safeGate 由入口选择 `KdaGateCumsumKernel<T,SAFE_GATE>` 有限模板实例，shape/layout/chunk_size 保留在 tiling data；热循环内没有 dtype 分支。

## 7. Kernel 设计

### 7.1 计算流程

每个 task 清零 FP32 acc；逐 token MTE2 加载并转换 g，safe 模板可应用 dt_bias、exp(A_log) 和 sigmoid，再乘 1/ln2 累加，MTE3 写回当前行。chunk/序列切换时 acc 重新置零。

### 7.2 内存规划

UB 固定分配 row/acc/tmp/one 各 256 个 FP32、输入类型缓冲和两个 32-byte scalar 缓冲；无 user scratch，GM 输出按 task 不重叠。

| 层级/资源 | 生命周期与所有权 |
| --- | --- |
| GM/Workspace | 输入输出区间由 tiling 固定；可变输入、partial 或临时区均有唯一写 owner 和确定消费时点 |
| UB | 按 tile 或双缓冲 slot 独占；生产者写完再交给 Vector/搬出阶段，消费者结束后释放 |
| MTE2/V/MTE3 事件 | 每个 slot 的 load、compute、store 和反向复用事件闭环，EventID 不跨未完成轮次复用 |

### 7.3 流水与同步

MTE2->V、V->MTE3、MTE3->MTE2 与 MTE3->V 均显式闭环。最后一项尤其保护下一 task 的 Duplicate 不覆盖仍被 MTE3 读取的 acc；`--cce-auto-sync=off`。

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

FP16/BF16 路径在矩阵乘或长归约中使用实现允许的高精度中间量；输出边界再转换到公开 dtype。逐项 reference、尾块、边界和组合模式按 JSON 阈值验证。 参考实现与阈值由 `tests/op_cases/kda_gate_cumsum.json` 固定。

## 10. 性能设计

以 JSON performance case 为固定矩阵，用 msopprof 记录设备侧 kernel duration、搬运与等待，并对比当前主线 Ascend C 基线检查性能回退。 不得使用 Python wall time 下结论。

## 11. 测试设计

`tests/op_cases/kda_gate_cumsum.json` 是唯一 case 规格。`tests/operators/kda_gate_cumsum/accuracy/` 运行主精度与泛化矩阵，
`routes/` 固化 aclnn 与 `<<<>>>` 编译/发射契约。标签覆盖 accuracy、generalization、boundary、negative、route、performance、example；
A2/A3/A5 使用相同 case ID，平台只决定编译与运行目标。

## 12. 已知限制与演进计划

- K<=256，chunk_size 仅支持 32/64/128。
- use_gate_in_kernel=true 时 A_log 必须为 [H_v]、safe_gate 必须 true，dt_bias 若存在须为 [H_v*K] 或 [H_v,K]。
- lower_bound 仅支持 [-5,0)；use_gate_in_kernel=false 时 safe_gate 必须 false。
- use_gate_in_kernel=false 时 A_log 与 dt_bias 必须为空，避免未消费输入在不同通路产生歧义。
- rank4 变长序列物理 B 必须为 1；cu_seqlens 首项为 0、非递减且末项等于 T。

后续扩展任何限制时，必须同时修改 host 拦截、API 错误码、README、JSON 与三平台回归；模板实例增长前先评估二进制体积和编译耗时。
