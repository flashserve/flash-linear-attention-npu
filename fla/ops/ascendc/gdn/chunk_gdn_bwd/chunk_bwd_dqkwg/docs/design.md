# ChunkBwdDqkwg 设计方案

## 1. 背景

Gated Delta Rule 分块反向链路的主梯度算子。它消费前向激活、chunk 状态和上游梯度，计算 `dQ`、`dK`、`dW` 与 `dG`，并支持 `H_v/H_k` 分组映射。 本实现通过统一 aclnn 与 ctypes 稳定入口接入 Gated Delta Network (GDN) 链路。

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

实现类型：`ascendc`。Dtype：主张量 FP16/BF16；gate 可为 FP32；输出跟随对应输入。Layout：BNSD；状态张量为 `[B,H_v,N_c,K,V]`。模式：定长与变长序列；变长序列的两个索引必须同时提供。
Shape 符号统一引用[算子 README 的 Shape 变量说明](../README.md#shape-symbols)。

## 4. 数学与接口语义

对 value head `h_v`，先映射 `h_k=floor(h_v/(H_v/H_k))`。算子按链式法则把
`dO`、`dH` 和 `dV` 对 chunk 内 score、状态项及门控项的贡献合并：

```text
dQ[h_k] += dS[h_v] @ K[h_k]
dK[h_k] += dS[h_v]^T @ Q[h_k] + dW[h_v] * beta/gate terms
dW[h_v]  = state/output branches reduced on V
dG[h_v]  = reverse cumulative reduction of gate-dependent terms
```

`dQ/dK` 在同一 key head 对应的多个 value head 上归约；尾块仅对有效 token 求值。
`g` 是沿 `T` 的 chunk-local 累积 gate，要求调用者提供与前向完全一致的值。

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

定长按 `B*N_c` 分配 chunk，变长序列按 `chunk_indices` 分配；每个 chunk 内再按 value head 划分 AIC/AIV 工作。

### 6.2 Tiling Data

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `B` | `uint64_t` | batch size |
| `HV` | `uint64_t` | value 侧 head 数 (v/g/h/do/dh/dv/dw/dg) |
| `HK` | `uint64_t` | key/query 侧 head 数 (q/k), HV = n_ratio * HK |
| `T` | `uint64_t` | sequence length |
| `K` | `uint64_t` | key/query dimension |
| `V` | `uint64_t` | value dimension |
| `BT` | `uint64_t` | chunk size (tile size in T dimension) |
| `numChunks` | `uint64_t` | T / BT |
| `scale` | `float` | 1.0 / sqrt(K) |
| `mul0RowNum` | `uint32_t` | host 计算并由 kernel 消费 |
| `aicCoreNum` | `uint32_t` | CV 深融合使用的 AIC blockDim (cube/vector 共用) |
| `wsDwOffset` | `uint64_t` | PartA: b_dw / 之后 PartC mm6 复用 |
| `wsBtxKSyncSlotsPerHead` | `uint64_t` | cross-stage group ring depth per core |
| `wsDgLastOffset` | `uint64_t` | PartA: b_dg_last 偏移 |
| `dgLastSize` | `uint64_t` | PartA: b_dg_last 大小, 32B 对齐 |
| `wsMm5Offset` | `uint64_t` | PartA: mm5 / GVA Part C: dq_inner / PartD: mm7 |
| `wsDsTempOffset` | `uint64_t` | PartB: b_ds_temp 偏移 |
| `wsMm6Offset` | `uint64_t` | PartC: mm6 / GVA D: dk_inner |
| `wsMm7Offset` | `uint64_t` | PartD: mm7 复用已释放的 wsMm5 |
| `wsMul1Offset` | `uint64_t` | independent short BT x BT ring for mul1 |
| `totalWorkspaceSize` | `uint64_t` | 总 workspace 大小 |
| `isVarLen` | `uint64_t` | 是否变长序列 |
| `B` | `int64_t` | host 计算并由 kernel 消费 |
| `HV` | `int64_t` | host 计算并由 kernel 消费 |
| `HK` | `int64_t` | host 计算并由 kernel 消费 |
| `T` | `int64_t` | host 计算并由 kernel 消费 |
| `K` | `int64_t` | host 计算并由 kernel 消费 |
| `V` | `int64_t` | host 计算并由 kernel 消费 |
| `BT` | `int64_t` | host 计算并由 kernel 消费 |
| `numChunks` | `int64_t` | host 计算并由 kernel 消费 |
| `scale` | `float` | host 计算并由 kernel 消费 |
| `mul0RowNum` | `int64_t` | host 计算并由 kernel 消费 |
| `wsDwOffset` | `int64_t` | host 计算并由 kernel 消费 |
| `wsDgLastOffset` | `int64_t` | host 计算并由 kernel 消费 |
| `dgLastSize` | `int64_t` | host 计算并由 kernel 消费 |
| `wsMm5Offset` | `int64_t` | host 计算并由 kernel 消费 |
| `wsDsTempOffset` | `int64_t` | host 计算并由 kernel 消费 |
| `totalWorkspaceSize` | `int64_t` | host 计算并由 kernel 消费 |
| `isVarLen` | `int64_t` | host 计算并由 kernel 消费 |

### 6.3 模板化方案与 tiling key

仅保留 key=1 选择当前 CV 深融合实现，不编码 runtime shape、layout 或普通属性。该单值 key 是历史二进制入口约定，不带组合增长；后续可在 ABI 允许时移除。

## 7. Kernel 设计

### 7.1 计算流程

Part A 生成 `dW` 和 gate 末端项；Part B 计算 score 梯度；Part C/D 通过 Cube 矩阵乘生成 `dQ/dK`，AIV 完成逐元素门控、尾块 mask 与 `dG` 归约。

### 7.2 内存规划

workspace 按 core/head 建立 `BT*K` 与 `BT*BT` 环形槽，`wsDwOffset/wsMm5Offset/wsMm6Offset/wsMul1Offset` 分阶段复用；L1/L0 承载矩阵乘 tile，UB 承载 gate 与归约临时量。

| 层级/资源 | 生命周期与所有权 |
| --- | --- |
| GM/Workspace | host 按 tiling 固定地址和容量；每个阶段写入区间与消费者、复用时点一一对应 |
| L1/L0A/L0B/L0C | 当前 Cube tile 独占；MTE1/Cube/Fixpipe 完成并由下一阶段消费后才允许复用 |
| UB | 当前 Vector tile 独占；MTE2/V/MTE3 或 AIC-AIV 交接完成后释放 |
| 事件/flag | 按 buffer slot 成对分配 ready/free；禁止未 wait 连续 set 或跨 slot 误复用 |

### 7.3 流水与同步

AIC 写 workspace 后用跨核阶段 flag 通知 AIV；AIV 消费后通过反向 free/ready 协议允许槽位复用。核内 MTE/Cube/Vector 事件成对使用，构建固定 `--cce-auto-sync=off`。

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

FP16/BF16 路径在矩阵乘或长归约中使用实现允许的高精度中间量；输出边界再转换到公开 dtype。逐项 reference、尾块、边界和组合模式按 JSON 阈值验证。 参考实现与阈值由 `tests/op_cases/chunk_bwd_dqkwg.json` 固定。

## 10. 性能设计

以 JSON performance case 为固定矩阵，用 msopprof 记录设备侧 kernel duration、搬运与等待，并对比当前主线 Ascend C 基线检查性能回退。 不得使用 Python wall time 下结论。

## 11. 测试设计

`tests/op_cases/chunk_bwd_dqkwg.json` 是唯一 case 规格。`tests/operators/chunk_bwd_dqkwg/accuracy/` 运行主精度与泛化矩阵，
`routes/` 固化 aclnn 与 `<<<>>>` 编译/发射契约。标签覆盖 accuracy、generalization、boundary、negative、route、performance、example；
A2/A3/A5 使用相同 case ID，平台只决定编译与运行目标。

## 12. 已知限制与演进计划

- `K` 仅支持 128，`V` 仅支持 128/256。
- `chunk_size` 仅支持 64/128，尾块按有效长度处理。
- 必须满足 `H_v % H_k == 0`；变长序列当前仅支持物理 `B=1`。
- `w`、`g_gamma` 当前为预留输入，必须传 `None`。
- `use_exp2` 与 `transpose_state_layout` 当前必须为 `false`。

后续扩展任何限制时，必须同时修改 host 拦截、API 错误码、README、JSON 与三平台回归；模板实例增长前先评估二进制体积和编译耗时。
