# CausalConv1dBwd 设计方案

## 1. 背景

CausalConv1d 的训练反向算子。它支持定长/变长序列和五种公开 layout，根据 x/weight/dy 及可选预激活 y、initial_state、dht 计算 dx/dw/db/dh0。 本实现通过统一 aclnn 与 ctypes 稳定入口接入 Gated Delta Network (GDN) 链路。

## 2. 目标与非目标

### 2.1 目标

- README 所列 `定长/变长序列、无激活/SiLU、可选初末状态` 场景精度与仓内参考实现一致。
- A2/A3/A5 均能构建和执行，`--cce-auto-sync=off`。
- 若本算子用于替换 Triton，同一 shape/dtype/layout 下 Ascend C 性能优于被替换实现。
- aclnn、`fla_npu.ops.ascendc`、`<<<>>>` 使用同一接口语义。

### 2.2 非目标

- 不扩展 README“已知限制”之外的 shape、预留参数或 layout。
- 不以 fallback、放宽阈值或跳过失败 case 替代 kernel 根因修复。

## 3. 能力边界

实现类型：`ascendc`。Dtype：FP32/FP16/BF16，同次调用所有浮点输入一致。Layout：x/dx 始终逻辑 BSH/TH；y/dy 按 BSH、BSND、BNSD、TND 或 NTD。模式：定长/变长序列、无激活/SiLU、可选初末状态。
Shape 符号统一引用[算子 README 的 Shape 变量说明](../README.md#shape-symbols)。

## 4. 数学与接口语义

若启用 SiLU/Swish，先得到预激活梯度：

```text
dz = dy                                      activation=0
dz = dy * (sigmoid(y)+y*sigmoid(y)*(1-sigmoid(y))) activation=1/2
dx[t-j,d] += dz[t,d] * weight[j,d]
dw[j,d]   += dz[t,d] * x_or_state[t-j,d]
db[d]     += sum_t dz[t,d]
```

序列起点以前的 dx 贡献累积到 `dh0`，末状态梯度 `dht` 通过状态滚动关系反传。

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

dx 按 token/channel tile 计算，dw/db 按 channel/width 分核并在 workspace 归约；变长序列不跨序列边界读取。

### 6.2 Tiling Data

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `shape/layout fields` | 整数 | 输入 shape、任务数、尾块与模式信息 |
| `workspace fields` | 整数 | workspace 偏移和阶段 buffer 大小 |

### 6.3 模板化方案与 tiling key

kernel 入口不按 activation 模板化，固定使用 key=0；activation=0/1/2、layout、W、shape 和切分全部由 tiling data 在运行时选择，因此不存在按 shape 扩张的 key 组合。

## 7. Kernel 设计

### 7.1 计算流程

AIV 生成 dz 并处理 layout，随后计算 dx/dh0；dw/db 使用分核局部累加和 workspace 归约，最终转换到输出 dtype。

### 7.2 内存规划

workspace 保存各 core 的 dw/db FP32 partial；UB 保存 x/dz/weight tile，initial_state 与 dht 按序列边界加载。

| 层级/资源 | 生命周期与所有权 |
| --- | --- |
| GM/Workspace | 输入输出区间由 tiling 固定；可变输入、partial 或临时区均有唯一写 owner 和确定消费时点 |
| UB | 按 tile 或双缓冲 slot 独占；生产者写完再交给 Vector/搬出阶段，消费者结束后释放 |
| MTE2/V/MTE3 事件 | 每个 slot 的 load、compute、store 和反向复用事件闭环，EventID 不跨未完成轮次复用 |

### 7.3 流水与同步

局部 partial 写完后由归约 owner 读取，跨核阶段由明确计数/flag 管理；核内 MTE/V 事件成对，`--cce-auto-sync=off`。

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

FP16/BF16 路径在矩阵乘或长归约中使用实现允许的高精度中间量；输出边界再转换到公开 dtype。逐项 reference、尾块、边界和组合模式按 JSON 阈值验证。 参考实现与阈值由 `tests/op_cases/causal_conv1d_bwd.json` 固定。

## 10. 性能设计

以 JSON performance case 为固定矩阵，用 msopprof 记录设备侧 kernel duration、搬运与等待，并对比当前主线 Ascend C 基线检查性能回退。 不得使用 Python wall time 下结论。

## 11. 测试设计

`tests/op_cases/causal_conv1d_bwd.json` 是唯一 case 规格。`tests/operators/causal_conv1d_bwd/accuracy/` 运行主精度与泛化矩阵，
`routes/` 固化 aclnn 与 `<<<>>>` 编译/发射契约。标签覆盖 accuracy、generalization、boundary、negative、route、performance、example；
A2/A3/A5 使用相同 case ID，平台只决定编译与运行目标。

## 12. 已知限制与演进计划

- weight 必须 `[W,D]` 且 W 仅支持 2/3/4；D 或拆分后的 V 必须为 16 的倍数。
- activation=1/2 时 y 必须提供且与 dy 同 layout/shape。
- TND/NTD 必须提供合法 query_start_loc；累计长度非递减、首项为 0、末项为总 T，且总 token 数必须大于 0。
- initial_state/dht/dh0 的逻辑 shape 为 `[B,W,D]`。

后续扩展任何限制时，必须同时修改 host 拦截、API 错误码、README、JSON 与三平台回归；模板实例增长前先评估二进制体积和编译耗时。
