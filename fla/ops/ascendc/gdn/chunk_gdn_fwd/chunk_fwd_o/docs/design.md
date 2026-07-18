# ChunkFwdO 设计方案

## 1. 背景

Gated Delta Rule 的 chunk 输出阶段。算子把当前 chunk 的 Q/K/V、chunk 起始状态 H 和累积 gate 合并，生成注意力输出 O；状态推进由 `chunk_gated_delta_rule_fwd_h` 单独完成。 本实现通过统一 aclnn 与 ctypes 稳定入口接入 Gated Delta Network (GDN) 链路。

## 2. 目标与非目标

### 2.1 目标

- README 所列 `fixed/varlen、GVA、整块/尾块；g 为必选标量 gate` 场景精度与仓内参考实现一致。
- A2/A3/A5 均能构建和执行，`--cce-auto-sync=off`。
- 若本算子用于替换 Triton，同一 shape/dtype/layout 下 Ascend C 性能优于被替换实现。
- aclnn、`fla_npu.ops.ascendc`、`<<<>>>` 使用同一接口语义。

### 2.2 非目标

- 不扩展 README“已知限制”之外的 shape、预留参数或 layout。
- 不以 fallback、放宽阈值或跳过失败 case 替代 kernel 根因修复。

## 3. 能力边界

实现类型：`ascendc`。Dtype：Q/K/V/H 为 FP16/BF16；g 可为 FP32。Layout：BNSD；状态使用 `[B,H_v,N_c,K,V]`。模式：fixed/varlen、GVA、整块/尾块；g 为必选标量 gate。
Shape 符号统一引用[算子 README 的 Shape 变量说明](../README.md#shape-symbols)。

## 4. 数学与接口语义

对 value head `h_v` 映射 key head `h_k=floor(h_v/(H_v/H_k))`：

```text
score = causal_mask((Q[h_k] @ K[h_k]^T) * scale * exp(g_row-g_col))
O[h_v] = Q[h_k] @ H_start[h_v] * scale + score @ V[h_v]
```

`H_start` 取当前 chunk 对应的状态切片；尾块只计算有效 token，输出保持 `[B,H_v,T,V]`。

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

按 `(batch,value_head,chunk)` 分配，Q/K score 在同 key head 的 value-head group 间复用；h 通过全局 chunk 序号定位。

### 6.2 Tiling Data

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `shapeBatch` | `int64_t` | host 计算并由 kernel 消费 |
| `seqlen` | `int64_t` | host 计算并由 kernel 消费 |
| `kNumHead` | `int64_t` | host 计算并由 kernel 消费 |
| `vNumHead` | `int64_t` | host 计算并由 kernel 消费 |
| `kHeadDim` | `int64_t` | host 计算并由 kernel 消费 |
| `vHeadDim` | `int64_t` | host 计算并由 kernel 消费 |
| `chunkSize` | `int64_t` | host 计算并由 kernel 消费 |
| `isVariedLen` | `int64_t` | host 计算并由 kernel 消费 |
| `tokenBatch` | `int64_t` | host 计算并由 kernel 消费 |
| `dataType` | `int64_t` | host 计算并由 kernel 消费 |
| `gDataType` | `int64_t` | host 计算并由 kernel 消费 |
| `vWorkspaceOffset` | `int64_t` | host 计算并由 kernel 消费 |
| `hWorkspaceOffset` | `int64_t` | host 计算并由 kernel 消费 |
| `attnWorkspaceOffset` | `int64_t` | host 计算并由 kernel 消费 |
| `aftermaskWorkspaceOffset` | `int64_t` | host 计算并由 kernel 消费 |
| `maskWorkspaceOffset` | `int64_t` | host 计算并由 kernel 消费 |
| `scale` | `float` | host 计算并由 kernel 消费 |

### 6.3 模板化方案与 tiling key

不使用 tiling key 分派 shape。主 dtype、gate dtype、fixed/varlen、K/V/C 均写入 tiling data，kernel 通过统一 Catlass 调度器处理，避免把 B/H/T 组合固化为 key。

## 7. Kernel 设计

### 7.1 计算流程

AIC 先计算 QK score 和 QH 状态项，AIV 应用 gate、因果/tail mask，AIC 再完成 score@V，最终与状态项相加写 O。

### 7.2 内存规划

workspace 保存 QK score 与 gated score 的阶段结果；L1/L0 承载 QK、QV 和 QH tile，UB 承载 gate、mask 与两个输出分支的逐元素合并。

| 层级/资源 | 生命周期与所有权 |
| --- | --- |
| GM/Workspace | host 按 tiling 固定地址和容量；每个阶段写入区间与消费者、复用时点一一对应 |
| L1/L0A/L0B/L0C | 当前 Cube tile 独占；MTE1/Cube/Fixpipe 完成并由下一阶段消费后才允许复用 |
| UB | 当前 Vector tile 独占；MTE2/V/MTE3 或 AIC-AIV 交接完成后释放 |
| 事件/flag | 按 buffer slot 成对分配 ready/free；禁止未 wait 连续 set 或跨 slot 误复用 |

### 7.3 流水与同步

AIC/AIV 通过按 core/head 划分的 workspace 槽交接 score，ready/free flag 成对使用；核内 MTE/Cube/Vector 事件闭环，`--cce-auto-sync=off`。

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

FP16/BF16 路径在矩阵乘或长归约中使用实现允许的高精度中间量；输出边界再转换到公开 dtype。reference 按 kernel 顺序模拟
QK、gate/mask、中间 dtype 转换、AV、QH、Exp 和输出缩放。逐项 reference、尾块、边界和组合模式按 JSON 阈值验证。
当前 `2e-2` 阈值继承 `tests/operators/chunk_fwd_o/routes/test_fast_kernel_chunk_fwd_o.py` 的 direct-launch 基线，
参考实现与阈值由 `tests/op_cases/chunk_fwd_o.json` 固定。

## 10. 性能设计

以 JSON performance case 为固定矩阵，用 msopprof 记录设备侧 kernel duration、搬运与等待，并对比当前主线 Ascend C 基线检查性能回退。 不得使用 Python wall time 下结论。

## 11. 测试设计

`tests/op_cases/chunk_fwd_o.json` 是唯一 case 规格。`tests/operators/chunk_fwd_o/accuracy/` 运行主精度与泛化矩阵，
`routes/` 固化 aclnn 与 `<<<>>>` 编译/发射契约。标签覆盖 accuracy、generalization、boundary、negative、route、performance、example；
A2/A3/A5 使用相同 case ID，平台只决定编译与运行目标。

## 12. 已知限制与演进计划

- `K` 仅支持 128，`V` 仅支持 128/256，`chunk_size` 仅支持 64/128。
- 必须满足 `H_v % H_k == 0`，h 的 chunk 数必须与索引推导一致。
- varlen 当前仅支持物理 `B=1`，两个索引必须同时提供。
- `g` 是 kernel 必选输入；`g_gamma` 必须为 None，`transpose_state_layout` 必须为 false。

后续扩展任何限制时，必须同时修改 host 拦截、API 错误码、README、JSON 与三平台回归；模板实例增长前先评估二进制体积和编译耗时。
