# ChunkBwdDvLocal 设计方案

## 1. 背景

Gated Delta Rule 反向过程的 Value 本地梯度算子。它在每个 chunk 内根据 `Q/K` score、门控差和 `dO` 生成 `dV_local`，不承担跨 chunk 状态梯度。 本实现通过统一 aclnn 与 ctypes 稳定入口接入 Gated Delta Network (GDN) 链路。

## 2. 目标与非目标

### 2.1 目标

- README 所列 `定长与变长序列；变长序列的两个索引必须同时提供` 场景精度与仓内参考实现一致。
- A2/A3/A5 均能构建和执行，`--cce-auto-sync=off`。
- 若本算子用于替换 Triton，同一 shape/dtype/layout 下 Ascend C 性能优于被替换实现。
- aclnn、`fla_npu.ops.ascendc`、`<<<>>>` 使用同一接口语义。

### 2.2 非目标

- 不扩展 README“已知限制”之外的 shape、预留参数或 layout。
- 不以 fallback、放宽阈值或跳过失败 case 替代 kernel 根因修复。

## 3. 能力边界

实现类型：`ascendc`。Dtype：q/k/d_o 为 FP16/BF16，g 可为 FP16/BF16/FP32。Layout：BNSD；变长序列在 T 维拼接。模式：定长与变长序列；变长序列的两个索引必须同时提供。
Shape 符号统一引用[算子 README 的 Shape 变量说明](../README.md#shape-symbols)。

## 4. 数学与接口语义

对 `h_v` 映射 `h_k=floor(h_v/(H_v/H_k))`，在每个 chunk 内：

```text
Ws       = K[h_k] @ Q[h_k]^T * scale
Ws_gated = triu(Ws * exp(g_col - g_row), diagonal=0)
dV_local = Ws_gated @ dO[h_v]
```

上三角包含对角线；partial chunk 的无效行列在乘法前清零。

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

定长以 `B*ceil(T/chunk_size)` 为 chunk 列表，变长序列直接消费规范化 `chunk_indices`；AIC 以 Q/K head 生成共享 score，AIV/AIC 以 value head 消费。

### 6.2 Tiling Data

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `b` | `int64_t` | host 计算并由 kernel 消费 |
| `hQk` | `int64_t` | host 计算并由 kernel 消费 |
| `hDo` | `int64_t` | host 计算并由 kernel 消费 |
| `hRatio` | `int64_t` | host 计算并由 kernel 消费 |
| `headBufNum` | `int64_t` | host 计算并由 kernel 消费 |
| `t` | `int64_t` | host 计算并由 kernel 消费 |
| `k` | `int64_t` | host 计算并由 kernel 消费 |
| `v` | `int64_t` | host 计算并由 kernel 消费 |
| `chunkSize` | `int64_t` | host 计算并由 kernel 消费 |
| `chunkNumForT` | `int64_t` | host 计算并由 kernel 消费 |
| `scale` | `float` | host 计算并由 kernel 消费 |

### 6.3 模板化方案与 tiling key

使用模板化 tiling：`strategy`(定长/变长序列)、`D_T_Q`、`D_T_G` 和 `V` 是有限编译期维度。key 只由模板实例生成，不承载 B/H/T 等普通 runtime shape；组合上限由 2 种策略、受支持 dtype 和 2 个 V 实例共同限定。

## 7. Kernel 设计

### 7.1 计算流程

Phase 1 AIC 计算 `K@Q^T`；Phase 1.5 AIV 扩展到 value head并应用 exp/mask；Phase 2 AIC 计算 gated score 与 dO 的矩阵乘并写 dV。

### 7.2 内存规划

user workspace 是 AIC/AIV 交接的 `chunk_size*chunk_size` score 环形槽；L1/L0A/L0B/L0C 放置两个矩阵乘 tile，UB 放置 gate、mask 和类型转换临时量。

| 层级/资源 | 生命周期与所有权 |
| --- | --- |
| GM/Workspace | host 按 tiling 固定地址和容量；每个阶段写入区间与消费者、复用时点一一对应 |
| L1/L0A/L0B/L0C | 当前 Cube tile 独占；MTE1/Cube/Fixpipe 完成并由下一阶段消费后才允许复用 |
| UB | 当前 Vector tile 独占；MTE2/V/MTE3 或 AIC-AIV 交接完成后释放 |
| 事件/flag | 按 buffer slot 成对分配 ready/free；禁止未 wait 连续 set 或跨 slot 误复用 |

### 7.3 流水与同步

score 槽采用 AIC-ready/AIV-done/second-AIC-ready 的阶段协议，生产者复用前等待消费者释放；核内事件覆盖 MTE2、Vector、MTE3 与 Cube/Fixpipe，构建固定 `--cce-auto-sync=off`。

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

FP16/BF16 路径在矩阵乘或长归约中使用实现允许的高精度中间量；输出边界再转换到公开 dtype。逐项 reference、尾块、边界和组合模式按 JSON 阈值验证。 参考实现与阈值由 `tests/op_cases/chunk_bwd_dv_local.json` 固定。

## 10. 性能设计

以 JSON performance case 为固定矩阵，用 msopprof 记录设备侧 kernel duration、搬运与等待，并对比当前主线 Ascend C 基线检查性能回退。 不得使用 Python wall time 下结论。

## 11. 测试设计

`tests/op_cases/chunk_bwd_dv_local.json` 是唯一 case 规格。`tests/operators/chunk_bwd_dv_local/accuracy/` 运行主精度与泛化矩阵，
`routes/` 固化 aclnn 与 `<<<>>>` 编译/发射契约。标签覆盖 accuracy、generalization、boundary、negative、route、performance、example；
A2/A3/A5 使用相同 case ID，平台只决定编译与运行目标。

## 12. 已知限制与演进计划

- `K` 仅支持 128，`V` 仅支持 128/256。
- `chunk_size` 仅支持 64/128。
- 必须满足 `H_v % H_k == 0`；变长序列当前仅支持物理 `B=1`。
- `g_gamma` 和 `A` 尚未实现，必须传 `None`。

后续扩展任何限制时，必须同时修改 host 拦截、API 错误码、README、JSON 与三平台回归；模板实例增长前先评估二进制体积和编译耗时。
