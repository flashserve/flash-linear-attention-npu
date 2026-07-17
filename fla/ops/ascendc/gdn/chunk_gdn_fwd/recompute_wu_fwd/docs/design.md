# RecomputeWUFwd 设计方案

## 1. 背景

按需重计算 WY 中间张量 W/U，减少前向保存显存。输入 K/V/beta/A 与标量 gate g，在每个 chunk 内输出 W 和 U。 本实现通过统一 aclnn 与 ctypes 稳定入口接入 Gated Delta Network (GDN) 链路。

## 2. 目标与非目标

### 2.1 目标

- README 所列 `fixed/varlen、GVA、标量 gate g` 场景精度与仓内参考实现一致。
- A2/A3/A5 均能构建和执行，`--cce-auto-sync=off`。
- 若本算子用于替换 Triton，同一 shape/dtype/layout 下 Ascend C 性能优于被替换实现。
- aclnn、`fla_npu.ops.ascendc`、`<<<>>>` 使用同一接口语义。

### 2.2 非目标

- 不扩展 README“已知限制”之外的 shape、预留参数或 layout。
- 不以 fallback、放宽阈值或跳过失败 case 替代 kernel 根因修复。

## 3. 能力边界

实现类型：`ascendc`。Dtype：主张量 FP16/BF16；beta/g/gk 可为 FP32。Layout：BNSD。模式：fixed/varlen、GVA、标量 gate g。
Shape 符号统一引用[算子 README 的 Shape 变量说明](../README.md#shape-symbols)。

## 4. 数学与接口语义

对 value head `h_v` 与映射后的 key head `h_k`：

```text
Vb = V[h_v] * beta[h_v]
Kb = K[h_k] * beta[h_v] * exp(g[h_v])
U  = A[h_v] @ Vb
W  = A[h_v] @ Kb
```

最后一个 chunk 使用实际有效行数，A 的其余列不参与结果。

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

按 value head/chunk 分配，两次矩阵乘共享 A tile；K 通过 GVA 映射读取。

### 6.2 Tiling Data

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `B` | `int64_t` | host 计算并由 kernel 消费 |
| `Hk` | `int64_t` | host 计算并由 kernel 消费 |
| `Hv` | `int64_t` | host 计算并由 kernel 消费 |
| `hvPerHk` | `int64_t` | host 计算并由 kernel 消费 |
| `T` | `int64_t` | host 计算并由 kernel 消费 |
| `K` | `int64_t` | host 计算并由 kernel 消费 |
| `V` | `int64_t` | host 计算并由 kernel 消费 |
| `chunkNum` | `int64_t` | host 计算并由 kernel 消费 |
| `chunkSize` | `int64_t` | host 计算并由 kernel 消费 |
| `vbVecRow` | `int64_t` | host 计算并由 kernel 消费 |
| `kbgExpVecRow` | `int64_t` | host 计算并由 kernel 消费 |
| `isVariable` | `int64_t` | host 计算并由 kernel 消费 |

### 6.3 模板化方案与 tiling key

保留 V=128 与 V=256 两个编译期 Cube tile 实例（key 1/2），普通 shape 和模式放 tiling data；组合固定为 2。

## 7. Kernel 设计

### 7.1 计算流程

AIV 生成 Vb/Kb，AIC 复用 A tile 分别计算 U/W，尾块由 valid row 控制写回。

### 7.2 内存规划

UB 保存 beta/gate 逐元素结果，L1/L0 复用 A tile，workspace 保存 AIV 到 AIC 的 Kb/Vb 中间块。

| 层级/资源 | 生命周期与所有权 |
| --- | --- |
| GM/Workspace | host 按 tiling 固定地址和容量；每个阶段写入区间与消费者、复用时点一一对应 |
| L1/L0A/L0B/L0C | 当前 Cube tile 独占；MTE1/Cube/Fixpipe 完成并由下一阶段消费后才允许复用 |
| UB | 当前 Vector tile 独占；MTE2/V/MTE3 或 AIC-AIV 交接完成后释放 |
| 事件/flag | 按 buffer slot 成对分配 ready/free；禁止未 wait 连续 set 或跨 slot 误复用 |

### 7.3 流水与同步

AIV 产出 Kb/Vb 后通知 AIC，AIC 完成两个矩阵乘后释放对应槽位；`--cce-auto-sync=off`。

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

FP16/BF16 路径在矩阵乘或长归约中使用实现允许的高精度中间量；输出边界再转换到公开 dtype。逐项 reference、尾块、边界和组合模式按 JSON 阈值验证。 参考实现与阈值由 `tests/op_cases/recompute_wu_fwd.json` 固定。

## 10. 性能设计

以 JSON performance case 为固定矩阵，用 msopprof 记录设备侧 kernel duration、搬运与等待，并对比当前主线 Ascend C 基线检查性能回退。 不得使用 Python wall time 下结论。

## 11. 测试设计

`tests/op_cases/recompute_wu_fwd.json` 是唯一 case 规格。`tests/operators/recompute_wu_fwd/accuracy/` 运行主精度与泛化矩阵，
`routes/` 固化 aclnn 与 `<<<>>>` 编译/发射契约。标签覆盖 accuracy、generalization、boundary、negative、route、performance、example；
A2/A3/A5 使用相同 case ID，平台只决定编译与运行目标。

## 12. 已知限制与演进计划

- 当前实现仅支持 `K=128`、`V=128/256`、`chunk_size=64/128`。
- 必须满足 `H_v % H_k == 0`，A 最后一维等于 chunk_size。
- g 必须提供；gk 当前未实现且必须为 None；varlen 物理 B=1。

后续扩展任何限制时，必须同时修改 host 拦截、API 错误码、README、JSON 与三平台回归；模板实例增长前先评估二进制体积和编译耗时。
