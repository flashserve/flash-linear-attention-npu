# KdaLayoutSwap12 设计方案

## 1. 背景

KDA 内部的连续布局转置算子。rank3 交换维 0/1，rank>=4 交换维 1/2，其余维保持顺序；可选 dependency 只建立 executor 调度依赖，不参与数值计算。 本实现通过统一 aclnn 与 ctypes 稳定入口接入 Kimi Delta Attention (KDA) 链路。

## 2. 目标与非目标

### 2.1 目标

- README 所列 `rank3 与 rank>=4；对齐行 grouped copy 和非对齐/超长行 tiled copy` 场景精度与仓内参考实现一致。
- A2/A3/A5 均能构建和执行，`--cce-auto-sync=off`。
- 若本算子用于替换 Triton，同一 shape/dtype/layout 下 Ascend C 性能优于被替换实现。
- aclnn、`fla_npu.ops.ascendc`、`<<<>>>` 使用同一接口语义。

### 2.2 非目标

- 不扩展 README“已知限制”之外的 shape、预留参数或 layout。
- 不以 fallback、放宽阈值或跳过失败 case 替代 kernel 根因修复。

## 3. 能力边界

实现类型：`ascendc`。Dtype：FP16/BF16/FP32，输入输出一致。Layout：ND；rank3 交换维 0/1，rank>=4 交换维 1/2。模式：rank3 与 rank>=4；对齐行 grouped copy 和非对齐/超长行 tiled copy。
Shape 符号统一引用[算子 README 的 Shape 变量说明](../README.md#shape-symbols)。

## 4. 数学与接口语义

```text
rank(x) = 3: y[j,i,...]   = x[i,j,...]
rank(x) >= 4: y[b,j,i,...] = x[b,i,j,...]
```

这是精确的数据重排，不执行浮点算术。`dependency` 的 shape/dtype 不改变 y；它仅让 L2
图显式等待前序生产者，防止临时 tensor 生命周期被 executor 提前复用。

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

将交换前的 (batch,firstDim,secondDim) 行空间展平后按 core 轮转；当整行 32-byte 对齐且 stride 可编码时，把最多 64 行组合成一次 strided MTE2，再连续 MTE3 写回。

### 6.2 Tiling Data

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `batch` | `int64_t` | host 计算并由 kernel 消费 |
| `firstDim` | `int64_t` | host 计算并由 kernel 消费 |
| `secondDim` | `int64_t` | host 计算并由 kernel 消费 |
| `tailDim` | `int64_t` | host 计算并由 kernel 消费 |
| `dataType` | `int64_t` | host 计算并由 kernel 消费 |
| `usedCoreNum` | `int64_t` | host 计算并由 kernel 消费 |

### 6.3 模板化方案与 tiling key

必须保留 3 个有限 dtype key：0=FP32、1=BF16、2=FP16，因为元素类型决定 DataCopy 长度和 kernel 模板实例。key 不编码 rank、shape 或 dependency，组合固定为 3；这是使用 tiling key 的必要原因。

## 7. Kernel 设计

### 7.1 计算流程

host 折叠维 3 之后的 tailDim；kernel 优先 grouped row copy。不能 grouped 时，每行按 8192 元素 UB tile 搬入并写到交换后的 GM offset，非 32-byte 对齐使用 DataCopyPad。

### 7.2 内存规划

每 core 仅使用 8192 个元素的 UB copyBuf；无 user scratch。x/y 地址区间由交换映射一一对应，各 core 的输出行不重叠。

| 层级/资源 | 生命周期与所有权 |
| --- | --- |
| GM/Workspace | 输入输出区间由 tiling 固定；可变输入、partial 或临时区均有唯一写 owner 和确定消费时点 |
| UB | 按 tile 或双缓冲 slot 独占；生产者写完再交给 Vector/搬出阶段，消费者结束后释放 |
| MTE2/V/MTE3 事件 | 每个 slot 的 load、compute、store 和反向复用事件闭环，EventID 不跨未完成轮次复用 |

### 7.3 流水与同步

每个 tile 使用 MTE2_MTE3 event 0 和 MTE3_MTE2 event 1，写回完成后才复用 copyBuf。无 Vector 计算和跨核共享区，`--cce-auto-sync=off`。

### 7.4 边界处理

边界 shape、空维、非对齐搬运和可选输入组合严格按 README 的已知限制处理；host 在 launch 前拦截不支持组合，kernel 不用越界读取、静默截断或 fallback 改变公开语义。

## 8. 平台设计

| 平台 | SOC | 路径 | 验证要求 |
| --- | --- | --- | --- |
| A2 | `ascend910b` | 公共实现 | 构建、全量精度、性能、通路 |
| A3 | `ascend910_93` | 公共实现 | 构建、全量精度、性能、通路 |
| A5 | `ascend950` | `arch35/` 存在时使用特化，否则公共实现 | 构建、全量精度、性能、通路 |

平台差异不得改变公开 shape/dtype/layout 语义；若某模板在平台上不可用，应在 host 明确报错并同步更新 README 与 JSON。

## 9. 精度设计

FP16/BF16 路径在矩阵乘或长归约中使用实现允许的高精度中间量；输出边界再转换到公开 dtype。逐项 reference、尾块、边界和组合模式按 JSON 阈值验证。 参考实现与阈值由 `tests/op_cases/kda_layout_swap12.json` 固定。

## 10. 性能设计

以 JSON performance case 为固定矩阵，用 msopprof 记录设备侧 kernel duration、搬运与等待，并对比当前主线 Ascend C 基线检查性能回退。 不得使用 Python wall time 下结论。

## 11. 测试设计

`tests/op_cases/kda_layout_swap12.json` 是唯一 case 规格。`tests/operators/kda_layout_swap12/accuracy/` 运行主精度与泛化矩阵，
`routes/` 固化 aclnn 与 `<<<>>>` 编译/发射契约。标签覆盖 accuracy、generalization、boundary、negative、route、performance、example；
A2/A3/A5 使用相同 case ID，平台只决定编译与运行目标。

## 12. 已知限制与演进计划

- x rank 必须至少为 3；y rank、dtype 及交换后的每一维必须与 x 精确对应。
- 输入在 L2 中先 contiguous；当前 API 总是创建连续 y。
- dependency 只表达执行顺序，不提供数据、dtype 或 shape 语义，不得依赖其值改变输出。
- aclnn/Python 通过 executor 输入依赖排序；`<<<>>>` 直调不读取 dependency，调用者必须在同一 stream 上先发射其生产者。
- 所有维度必须为正；host 在进入 tiling 前拒绝空维度，kernel 的 usedCoreNum 至少为 1。

后续扩展任何限制时，必须同时修改 host 拦截、API 错误码、README、JSON 与三平台回归；模板实例增长前先评估二进制体积和编译耗时。
