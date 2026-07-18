# SolveTri 设计方案

## 1. 背景

对每个 chunk 的严格下三角矩阵 A 计算 `(I+A)^-1`，用于 WY 表示求解。输入最后一维保存当前 token 行的 chunk 列，输出保持相同布局。 本实现通过统一 aclnn 与 ctypes 稳定入口接入 Gated Delta Network (GDN) 链路。

## 2. 目标与非目标

### 2.1 目标

- README 所列 `dense BHTD/BSND 与变长序列 TND，支持尾块` 场景精度与仓内参考实现一致。
- A2/A3/A5 均能构建和执行，`--cce-auto-sync=off`。
- 若本算子用于替换 Triton，同一 shape/dtype/layout 下 Ascend C 性能优于被替换实现。
- aclnn、`fla_npu.ops.ascendc`、`<<<>>>` 使用同一接口语义。

### 2.2 非目标

- 不扩展 README“已知限制”之外的 shape、预留参数或 layout。
- 不以 fallback、放宽阈值或跳过失败 case 替代 kernel 根因修复。

## 3. 能力边界

实现类型：`ascendc`。Dtype：FP16/BF16。Layout：BHTD `[B,H_v,T,chunk_size]`、BSND `[B,T,H_v,chunk_size]`、TND `[T,H_v,chunk_size]`；layout 字符串必须小写。模式：dense BHTD/BSND 与变长序列 TND，支持尾块。
Shape 符号统一引用[算子 README 的 Shape 变量说明](../README.md#shape-symbols)。

## 4. 数学与接口语义

对每个 batch/head/chunk，取有效阶数 M 的严格下三角矩阵 A：

```text
Y = inverse(I_M + tril(A, diagonal=-1))
```

实现采用分块前代/三角逆；尾块只对有效 M 求逆，padding 列按接口约定写零。

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

按 `(batch/head/chunk)` 分配，chunk_size=128 时继续按内部 tile 求解，尾块使用局部 M。

### 6.2 Tiling Data

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `totalTiles` | `int64_t` | host 计算并由 kernel 消费 |
| `matrixSize` | `int64_t` | host 计算并由 kernel 消费 |
| `numHeads` | `int64_t` | host 计算并由 kernel 消费 |
| `seqLen` | `int64_t` | host 计算并由 kernel 消费 |
| `batchSize` | `int64_t` | host 计算并由 kernel 消费 |
| `isLower` | `int64_t` | host 计算并由 kernel 消费 |
| `hasCuSeqlens` | `int64_t` | host 计算并由 kernel 消费 |
| `tilesPerCore` | `int64_t` | host 计算并由 kernel 消费 |
| `chunkSize` | `int64_t` | host 计算并由 kernel 消费 |
| `numChunks` | `int64_t` | host 计算并由 kernel 消费 |
| `lastChunkValidSize` | `int64_t` | host 计算并由 kernel 消费 |
| `isVarlen` | `int64_t` | host 计算并由 kernel 消费 |
| `totalChunks` | `int64_t` | host 计算并由 kernel 消费 |
| `layoutMode` | `int64_t` | host 计算并由 kernel 消费 |
| `dtypeMode` | `int64_t` | 0=fp16, 1=bf16 |

### 6.3 模板化方案与 tiling key

固定使用 key 1 作为既有 kernel launch ABI，key 不承载 shape 或模式分派。chunk_size、dtype、layout、尾块和变长序列全部由 tiling data 处理，因此不会随 B/H/T 产生组合爆炸；保留 key 1 是因为 kernel 入口现有 TILING_KEY_IS(1) 编译契约。

## 7. Kernel 设计

### 7.1 计算流程

加载 A tile，清理上三角并注入单位对角，按对角块前代求逆，逐块更新剩余下三角并写回。

### 7.2 内存规划

UB/L1 保存当前三角 tile 和单位阵，L0/Cube 处理块乘更新；GM 输出与输入不别名。

| 层级/资源 | 生命周期与所有权 |
| --- | --- |
| GM/Workspace | host 按 tiling 固定地址和容量；每个阶段写入区间与消费者、复用时点一一对应 |
| L1/L0A/L0B/L0C | 当前 Cube tile 独占；MTE1/Cube/Fixpipe 完成并由下一阶段消费后才允许复用 |
| UB | 当前 Vector tile 独占；MTE2/V/MTE3 或 AIC-AIV 交接完成后释放 |
| 事件/flag | 按 buffer slot 成对分配 ready/free；禁止未 wait 连续 set 或跨 slot 误复用 |

### 7.3 流水与同步

每个矩阵由单 core/协作组按对角块顺序推进，核内 MTE/Vector/Cube 事件保护 tile 复用；`--cce-auto-sync=off`。

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

FP16/BF16 路径在矩阵乘或长归约中使用实现允许的高精度中间量；输出边界再转换到公开 dtype。逐项 reference、尾块、边界和组合模式按 JSON 阈值验证。 参考实现与阈值由 `tests/op_cases/solve_tri.json` 固定。

## 10. 性能设计

以 JSON performance case 为固定矩阵，用 msopprof 记录设备侧 kernel duration、搬运与等待；在相同输入和测量配置下逐项对比 `fla_npu.ops.triton.solve_tril_npu`，Ascend C 必须更快。 不得使用 Python wall time 下结论。

## 11. 测试设计

`tests/op_cases/solve_tri.json` 是唯一 case 规格。`tests/operators/solve_tri/accuracy/` 运行主精度与泛化矩阵，
`routes/` 固化 aclnn 与 `<<<>>>` 编译/发射契约。标签覆盖 accuracy、generalization、boundary、negative、route、performance、example；
A2/A3/A5 使用相同 case ID，平台只决定编译与运行目标。

## 12. 已知限制与演进计划

- 矩阵阶/最后一维 chunk_size 支持 16/32/64/128。
- layout 仅支持小写 `bhtd`、`bsnd`、`tnd`；TND 必须提供两个变长序列索引，定长布局 不接受变长序列索引。
- 输入必须表示严格下三角 A；对角线由算子加单位阵。

后续扩展任何限制时，必须同时修改 host 拦截、API 错误码、README、JSON 与三平台回归；模板实例增长前先评估二进制体积和编译耗时。
