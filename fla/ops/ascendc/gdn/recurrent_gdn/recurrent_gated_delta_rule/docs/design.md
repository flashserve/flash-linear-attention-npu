# RecurrentGatedDeltaRule 设计方案

## 1. 背景

面向 decode/投机推理的小步长 recurrent Gated Delta Rule。它按逻辑序列选择初始状态槽，逐 token 生成输出，并把每步状态写入 `ssm_state_indices` 指定的槽；支持标量 gate、逐 K gate 与接受 token 数。 本实现通过统一 aclnn 与 ctypes 稳定入口接入 Gated Delta Network (GDN) 链路。

## 2. 目标与非目标

### 2.1 目标

- README 所列 `变长 recurrent、g/gk 独立或组合、可选 accepted-token 状态选择、原地 state` 场景精度与仓内参考实现一致。
- A2/A3/A5 均能构建和执行，`--cce-auto-sync=off`。
- 若本算子用于替换 Triton，同一 shape/dtype/layout 下 Ascend C 性能优于被替换实现。
- aclnn、`fla_npu.ops.ascendc`、`<<<>>>` 使用同一接口语义。

### 2.2 非目标

- 不扩展 README“已知限制”之外的 shape、预留参数或 layout。
- 不以 fallback、放宽阈值或跳过失败 case 替代 kernel 根因修复。

## 3. 能力边界

实现类型：`ascendc`。Dtype：Q/K/V/beta/out 为 BF16；state 为 BF16/FP32；gate FP32；索引 INT32。Layout：TND + 独立状态槽布局 `[D_s,H_v,V,K]`。模式：变长 recurrent、g/gk 独立或组合、可选 accepted-token 状态选择、原地 state。
Shape 符号统一引用[GDN 模型符号表](../../../README.md#model-shape-symbols)。

## 4. 数学与接口语义

对逻辑序列 b，`actual_seq_lengths[0]` 是前置跳过 token 数，后续元素是各序列长度。令序列起点为 `p_b`；无 `num_accepted_tokens` 时从 `ssm_state_indices[p_b]` 读取初始状态，有该输入时从 `ssm_state_indices[p_b+a_b-1]` 读取。对序列内 token t：

```text
S_t = exp(g_t) * S_(t-1)                         # g 存在时
S_t = S_t * exp(gk_t)[None, :]                   # gk 存在时
delta_t = beta_t * (v_t - S_t @ k_t)
S_t = S_t + outer(delta_t, k_t)
o_t = S_t @ (q_t * scale)
state_ref[ssm_state_indices[t]] = S_t
```

`g` 与 `gk` 可独立使用、同时使用或同时为空；同时为空时不施加衰减。`state_ref` 是可变输入输出。

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

按序列/token/head 分配，映射到同一 state slot 的 token 保持程序顺序；不同槽可并行。

### 6.2 Tiling Data

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `shape/layout fields` | 整数 | 输入 shape、任务数、尾块与模式信息 |
| `workspace fields` | 整数 | workspace 偏移和阶段 buffer 大小 |

### 6.3 模板化方案与 tiling key

使用 tiling 模板注册但不使用数值 tiling key 分派；GetTilingKey 固定为 0。BF16 主路径、gate 组合、state dtype、token/head/dim 与 UB buffer profile 全部由 tiling data 描述。

## 7. Kernel 设计

### 7.1 计算流程

读取状态与 gate，AIC 计算 k@S/q@S 和 outer update，AIV 处理 beta、接受 mask 与 state 合并，写 out 并提交 state。

### 7.2 内存规划

state_ref 常驻 GM；UB 保存单步 gate/beta/mask，L1/L0 保存当前 state tile 和向量-矩阵计算片段。

| 层级/资源 | 生命周期与所有权 |
| --- | --- |
| GM/Workspace | host 按 tiling 固定地址和容量；每个阶段写入区间与消费者、复用时点一一对应 |
| L1/L0A/L0B/L0C | 当前 Cube tile 独占；MTE1/Cube/Fixpipe 完成并由下一阶段消费后才允许复用 |
| UB | 当前 Vector tile 独占；MTE2/V/MTE3 或 AIC-AIV 交接完成后释放 |
| 事件/flag | 按 buffer slot 成对分配 ready/free；禁止未 wait 连续 set 或跨 slot 误复用 |

### 7.3 流水与同步

同一 slot 的读改写由任务归属和 token 顺序串行化，提交 state 前等待 Cube/Vector 完成；`--cce-auto-sync=off`。

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

FP16/BF16 路径在矩阵乘或长归约中使用实现允许的高精度中间量；输出边界再转换到公开 dtype。逐项 reference、尾块、边界和组合模式按 JSON 阈值验证。 参考实现与阈值由 `tests/op_cases/recurrent_gated_delta_rule.json` 固定。

## 10. 性能设计

以 JSON performance case 为固定矩阵，用 msopprof 记录设备侧 kernel duration、搬运与等待，并对比当前主线 Ascend C 基线检查性能回退。 不得使用 Python wall time 下结论。

## 11. 测试设计

`tests/op_cases/recurrent_gated_delta_rule.json` 是唯一 case 规格。`tests/operators/recurrent_gated_delta_rule/accuracy/` 运行主精度与泛化矩阵，
`routes/` 固化 aclnn 与 `<<<>>>` 编译/发射契约。标签覆盖 accuracy、generalization、boundary、negative、route、performance、example；
A2/A3/A5 使用相同 case ID，平台只决定编译与运行目标。

## 12. 已知限制与演进计划

- 每条序列本次有效 token 数不超过 8。
- g/gk 均为空表示单位衰减；二者存在时 dtype 必须为 FP32，shape 分别为 `[T,H_v]`、`[T,H_v,K]`。
- state 槽索引必须位于 `[0,D_s)`；actual_seq_lengths 有效项之和等于 T。
- `H_k/H_v <= 256`、`K/V <= 512` 且 `H_v % H_k == 0`。
- state_ref 为原地更新参数，不能 require_grad；非连续 state 依赖 CANN >= 9.1。

后续扩展任何限制时，必须同时修改 host 拦截、API 错误码、README、JSON 与三平台回归；模板实例增长前先评估二进制体积和编译耗时。
