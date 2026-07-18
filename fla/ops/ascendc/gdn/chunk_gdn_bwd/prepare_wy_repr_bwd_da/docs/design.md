# PrepareWyReprBwdDa 设计方案

## 1. 背景

WY 表示反向的 dA 子算子。它根据 K/V、beta、前向 A、dW/dU 与 gate，计算 chunk 局部矩阵 A 的梯度，供完整 WY 反向继续生成 dK/dV/dBeta/dG。 本实现通过统一 aclnn 与 ctypes 稳定入口接入 Gated Delta Network (GDN) 链路。

## 2. 目标与非目标

### 2.1 目标

- README 所列 `定长与变长序列，支持 GVA` 场景精度与仓内参考实现一致。
- A2/A3/A5 均能构建和执行，`--cce-auto-sync=off`。
- 若本算子用于替换 Triton，同一 shape/dtype/layout 下 Ascend C 性能优于被替换实现。
- aclnn、`fla_npu.ops.ascendc`、`<<<>>>` 使用同一接口语义。

### 2.2 非目标

- 不扩展 README“已知限制”之外的 shape、预留参数或 layout。
- 不以 fallback、放宽阈值或跳过失败 case 替代 kernel 根因修复。

## 3. 能力边界

实现类型：`ascendc`。Dtype：K/V/A/dW/dU 为 FP16/BF16；beta/g 可为 FP32。Layout：BNSD；A/dA 最后一维为 C。模式：定长与变长序列，支持 GVA。
Shape 符号统一引用[算子 README 的 Shape 变量说明](../README.md#shape-symbols)。

## 4. 数学与接口语义

在每个 value head/chunk 内，先将对应 key head 按 GVA 关系广播，再组合两条矩阵链：

```text
dA = dU @ (V * beta)^T + dW @ (K * beta * exp(g))^T
dA = causal_mask(dA) + A-dependent triangular correction
```

`dA` 仅在 chunk 的有效因果区域有定义；尾块之外写零。

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

按 `(batch/value_head/chunk)` 划分；GVA 通过 `h_k=floor(h_v/(H_v/H_k))` 选择 K，尾块把无效行列屏蔽。

### 6.2 Tiling Data

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `B` | `int64_t` | host 计算并由 kernel 消费 |
| `HV` | `int64_t` | host 计算并由 kernel 消费 |
| `HK` | `int64_t` | host 计算并由 kernel 消费 |
| `T` | `int64_t` | host 计算并由 kernel 消费 |
| `K` | `int64_t` | host 计算并由 kernel 消费 |
| `V` | `int64_t` | host 计算并由 kernel 消费 |
| `chunkSize` | `int64_t` | host 计算并由 kernel 消费 |
| `chunkNum` | `int64_t` | host 计算并由 kernel 消费 |
| `rowNumKBetaG` | `int64_t` | host 计算并由 kernel 消费 |
| `rowNumVBeta` | `int64_t` | host 计算并由 kernel 消费 |
| `rowNumMDuDw` | `int64_t` | host 计算并由 kernel 消费 |
| `rowNumG` | `int64_t` | host 计算并由 kernel 消费 |
| `isVariable` | `int64_t` | host 计算并由 kernel 消费 |

### 6.3 模板化方案与 tiling key

当前仅使用 key=1 进入唯一模板化 kernel，不编码 runtime shape 或 dtype。该单值 key 是现有二进制入口约定，组合数为 1；后续 ABI 整理时应消除。

## 7. Kernel 设计

### 7.1 计算流程

AIC 分别计算 dU/V 与 dW/K 两条矩阵乘，AIV 应用 beta、gate、因果三角 mask 及 A 相关修正，最后写 dA。

### 7.2 内存规划

workspace 暂存两个 `C*C` 矩阵乘结果并按阶段复用；L1/L0 承载 K/V tile，UB 承载 beta/gate、mask 和 dA 合并片段。

| 层级/资源 | 生命周期与所有权 |
| --- | --- |
| GM/Workspace | host 按 tiling 固定地址和容量；每个阶段写入区间与消费者、复用时点一一对应 |
| L1/L0A/L0B/L0C | 当前 Cube tile 独占；MTE1/Cube/Fixpipe 完成并由下一阶段消费后才允许复用 |
| UB | 当前 Vector tile 独占；MTE2/V/MTE3 或 AIC-AIV 交接完成后释放 |
| 事件/flag | 按 buffer slot 成对分配 ready/free；禁止未 wait 连续 set 或跨 slot 误复用 |

### 7.3 流水与同步

AIC 产出矩阵片段后通知 AIV 合并，AIV 写回或释放槽位后才允许下一轮覆盖；核内事件覆盖 MTE/Cube/Vector，构建固定 `--cce-auto-sync=off`。

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

FP16/BF16 路径在矩阵乘或长归约中使用实现允许的高精度中间量；输出边界再转换到公开 dtype。逐项 reference、尾块、边界和组合模式按 JSON 阈值验证。 参考实现与阈值由 `tests/op_cases/prepare_wy_repr_bwd_da.json` 固定。

## 10. 性能设计

以 JSON performance case 为固定矩阵，用 msopprof 记录设备侧 kernel duration、搬运与等待，并对比当前主线 Ascend C 基线检查性能回退。 不得使用 Python wall time 下结论。

## 11. 测试设计

`tests/op_cases/prepare_wy_repr_bwd_da.json` 是唯一 case 规格。`tests/operators/prepare_wy_repr_bwd_da/accuracy/` 运行主精度与泛化矩阵，
`routes/` 固化 aclnn 与 `<<<>>>` 编译/发射契约。标签覆盖 accuracy、generalization、boundary、negative、route、performance、example；
A2/A3/A5 使用相同 case ID，平台只决定编译与运行目标。

## 12. 已知限制与演进计划

- `K` 仅支持 128，`V` 仅支持 128/256。
- `chunk_size` 仅支持 64/128，并须等于 A/dA 的最后一维。
- 必须满足 `H_v % H_k == 0`；变长序列当前仅支持物理 `B=1`。
- `cu_seqlens` 与 `chunk_indices` 必须同时提供或同时省略。

后续扩展任何限制时，必须同时修改 host 拦截、API 错误码、README、JSON 与三平台回归；模板实例增长前先评估二进制体积和编译耗时。
