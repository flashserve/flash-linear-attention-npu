# CausalConv1d 设计方案

## 1. 背景

GDN 输入预处理的因果一维卷积。run_mode=0 执行定长/变长序列前向并维护卷积状态，run_mode=1 执行 decode/投机更新；可选 SiLU 激活和 head layout 转换。 本实现通过统一 aclnn 与 ctypes 稳定入口接入 Gated Delta Network (GDN) 链路。

## 2. 目标与非目标

### 2.1 目标

- README 所列 `定长/变长序列 forward、decode/update、投机接受 token` 场景精度与仓内参考实现一致。
- A2/A3/A5 均能构建和执行，`--cce-auto-sync=off`。
- 若本算子用于替换 Triton，同一 shape/dtype/layout 下 Ascend C 性能优于被替换实现。
- aclnn、`fla_npu.ops.ascendc`、`<<<>>>` 使用同一接口语义。

### 2.2 非目标

- 不扩展 README“已知限制”之外的 shape、预留参数或 layout。
- 不以 fallback、放宽阈值或跳过失败 case 替代 kernel 根因修复。

## 3. 能力边界

实现类型：`ascendc`。Dtype：x/weight/bias/state/y 为 FP16/BF16；元数据 INT64。Layout：输入 BSH/TH；head_num>0 时输出 BNSD/NTD。模式：定长/变长序列 forward、decode/update、投机接受 token。
Shape 符号统一引用[GDN 模型符号表](../../../README.md#model-shape-symbols)。

## 4. 数学与接口语义

对通道 d 和时间 t：

```text
z[t,d] = bias[d] + sum(j=0..W-1, weight[j,d] * x[t-j,d])
y[t,d] = z[t,d]                    activation_mode=0
y[t,d] = z[t,d] * sigmoid(z[t,d]) activation_mode=1
```

`t-j` 越过序列起点时读取该序列的 `conv_states`；update 模式只提交有效/已接受 token 并原地滚动状态。

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

forward 按 batch/sequence/channel tile 划分，update 按有效 state slot/token 划分；pad_slot_id 不产生输出和状态提交。

### 6.2 Tiling Data

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `shape/layout fields` | 整数 | 输入 shape、任务数、尾块与模式信息 |
| `workspace fields` | 整数 | workspace 偏移和阶段 buffer 大小 |

### 6.3 模板化方案与 tiling key

使用模板化三维选择：runModeKey(2)、widthKey(runtime/2/3/4) 和 fnPlanKey(CUTBS/CUTBSD)。实际选择表仅 7 个实例；width 编译期展开用于消除热循环，update 保留 runtime width。B/T/D 不进入 key。

## 7. Kernel 设计

### 7.1 计算流程

forward 路径滚动加载 W 个 token、执行 depthwise FMA、可选 SiLU并更新尾状态；update 路径按 cache index 读取/滚动 state，依据接受 mask 提交。

### 7.2 内存规划

conv_states 常驻 GM；UB 双缓冲 x/weight/输出 tile，workspace 仅用于 initial-state 同步或 layout 转换。

| 层级/资源 | 生命周期与所有权 |
| --- | --- |
| GM/Workspace | 输入输出区间由 tiling 固定；可变输入、partial 或临时区均有唯一写 owner 和确定消费时点 |
| UB | 按 tile 或双缓冲 slot 独占；生产者写完再交给 Vector/搬出阶段，消费者结束后释放 |
| MTE2/V/MTE3 事件 | 每个 slot 的 load、compute、store 和反向复用事件闭环，EventID 不跨未完成轮次复用 |

### 7.3 流水与同步

每个 state slot 的更新由唯一任务负责；MTE2/V/MTE3 事件保护滚动 buffer，存在 workspace 协作时使用明确 flag，`--cce-auto-sync=off`。

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

FP16/BF16 路径在矩阵乘或长归约中使用实现允许的高精度中间量；输出边界再转换到公开 dtype。逐项 reference、尾块、边界和组合模式按 JSON 阈值验证。 参考实现与阈值由 `tests/op_cases/causal_conv1d.json` 固定。

## 10. 性能设计

以 JSON performance case 为固定矩阵，用 msopprof 记录设备侧 kernel duration、搬运与等待，并对比当前主线 Ascend C 基线检查性能回退。 不得使用 Python wall time 下结论。

## 11. 测试设计

`tests/op_cases/causal_conv1d.json` 是唯一 case 规格。`tests/operators/causal_conv1d/accuracy/` 运行主精度与泛化矩阵，
`routes/` 固化 aclnn 与 `<<<>>>` 编译/发射契约。标签覆盖 accuracy、generalization、boundary、negative、route、performance、example；
A2/A3/A5 使用相同 case ID，平台只决定编译与运行目标。

## 12. 已知限制与演进计划

- 卷积宽度 W 仅支持 2/3/4，特征维 D 必须为 16 的倍数。
- activation_mode 仅支持 0/1，run_mode 仅支持 0/1。
- head_num>0 仅用于 forward，且必须整除 D；拆分后的 D/head_num 仍须为 16 的倍数。
- conv_states 的 L_s 至少为 W-1；rank-3 投机 update 还要求 L_s >= (W-1)+(T-1)。
- num_accepted_tokens 仅在 run_mode=1 且 W=4 时实现，值域为每条逻辑序列的 [0,token_count]。
- conv_states 为可变输入；非连续状态仅在 CANN >= 9.1 支持。
- rank-2 forward 必须提供 query_start_loc；query_start_loc、cache_indices、initial_state_mode 和接受数的长度必须与逻辑序列数一致。
- initial_state_mode 仅用于 run_mode=0 且元素只能为 0/1；cache_indices 只能选择有效状态槽或 pad_slot_id。

后续扩展任何限制时，必须同时修改 host 拦截、API 错误码、README、JSON 与三平台回归；模板实例增长前先评估二进制体积和编译耗时。
