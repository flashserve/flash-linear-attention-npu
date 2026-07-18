# ChunkLocalCumsum 设计方案

## 1. 背景

在每个时间 chunk 内对 gate 或任意尾部特征做局部前缀/后缀累加。该 Ascend C 实现替换 Triton 预处理路径，并支持定长/变长序列与 scale。 本实现通过统一 aclnn 与 ctypes 稳定入口接入 Gated Delta Network (GDN) 链路。

## 2. 目标与非目标

### 2.1 目标

- README 所列 `定长/变长序列、forward/reverse、任意连续尾部 P` 场景精度与仓内参考实现一致。
- A2/A3/A5 均能构建和执行，`--cce-auto-sync=off`。
- 若本算子用于替换 Triton，同一 shape/dtype/layout 下 Ascend C 性能优于被替换实现。
- aclnn、`fla_npu.ops.ascendc`、`<<<>>>` 使用同一接口语义。

### 2.2 非目标

- 不扩展 README“已知限制”之外的 shape、预留参数或 layout。
- 不以 fallback、放宽阈值或跳过失败 case 替代 kernel 根因修复。

## 3. 能力边界

实现类型：`ascendc`。Dtype：输入输出仅 FP32；索引 INT64。Layout：head-first `[B,H_v,T,...]`。模式：定长/变长序列、forward/reverse、任意连续尾部 P。
Shape 符号统一引用[算子 README 的 Shape 变量说明](../README.md#shape-symbols)。

## 4. 数学与接口语义

将 token 后的尾部维展平为 P：

```text
out[b,h,t,p] = scale * sum(k=chunk_start..t) g[b,h,k,p]       reverse=false
out[b,h,t,p] = scale * sum(k=t..chunk_end-1) g[b,h,k,p]       reverse=true
```

变长序列的 chunk_start/chunk_end 在每条逻辑序列内计算，不跨 `cu_seqlens` 边界。

Python、aclnn 与直调入口均以同一逻辑 shape 解释输入。接口层转换只处理连续性、描述符或文档明确的
layout 规范化，不改变公式、边界 mask、head 映射或可选输入语义。

## 5. 整体架构

1. `op_host/*_def.cpp` 注册输入输出、dtype、属性和 A2/A3/A5。
2. InferShape、op_api 与 tiling host 共同按 README 校验必选参数、shape、dtype、layout、属性和可选输入组合，并构造或核对输出；变长序列元数据由 op_api 以 `aclIntArray` 校验，再由 executor 转为 value-depend tensor。
3. tiling processor 计算任务数、边界块、workspace 偏移和模板实例。
4. `op_kernel/` 按本算子的计算流程完成搬运、计算、同步和写回。
5. aclnn 两段式接口负责 contiguous、workspace/executor 和 stream 异步发射。
6. `fla_npu.ops.ascendc` 仅通过 ctypes 调用 aclnn，不依赖 torch_npu dispatcher。

## 6. Tiling 设计

### 6.1 任务划分

定长按 `(B*H_v,chunk,tail_tile)` 分配；变长序列按 `(seq_id,local_block_id,head,tail_tile)` 分配，每个处理块内再按 C 完成 scan。
令 `P` 为时间维后的连续尾部维乘积，host 使用
`B_T = next_pow2(max(floor(2^17 / (P * C)), 1))` 选择 processing block，并要求 `B_T >= C`。
变长序列的 `chunk_indices_out` 必须为每条序列列出 `ceil(sequence_length / B_T)` 个 block。

### 6.2 Tiling Data

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `shape/layout fields` | 整数 | 输入 shape、任务数、尾块与模式信息 |
| `workspace fields` | 整数 | workspace 偏移和阶段 buffer 大小 |

### 6.3 模板化方案与 tiling key

未使用多分支 tiling key；reverse、scale、tail P、定长/变长序列都由 tiling data 和 runtime 常量处理。

## 7. Kernel 设计

### 7.1 计算流程

MTE2 分段加载一个 chunk/tail tile，Vector 执行顺序或逆序 scan 与 scale，MTE3 写回；长 P 分 tile。

### 7.2 内存规划

UB 保存当前 scan tile和必要的 carry；不同 task 不共享输出区，无 user workspace 数据依赖。

| 层级/资源 | 生命周期与所有权 |
| --- | --- |
| GM/Workspace | 输入输出区间由 tiling 固定；可变输入、partial 或临时区均有唯一写 owner 和确定消费时点 |
| UB | 按 tile 或双缓冲 slot 独占；生产者写完再交给 Vector/搬出阶段，消费者结束后释放 |
| MTE2/V/MTE3 事件 | 每个 slot 的 load、compute、store 和反向复用事件闭环，EventID 不跨未完成轮次复用 |

### 7.3 流水与同步

MTE2-V-MTE3 事件按双缓冲槽闭环，同一 scan 由单 core 顺序处理；`--cce-auto-sync=off`。

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

FP16/BF16 路径在矩阵乘或长归约中使用实现允许的高精度中间量；输出边界再转换到公开 dtype。逐项 reference、尾块、边界和组合模式按 JSON 阈值验证。 参考实现与阈值由 `tests/op_cases/chunk_local_cumsum.json` 固定。

## 10. 性能设计

以 JSON performance case 为固定矩阵，用 msopprof 记录设备侧 kernel duration、搬运与等待；在相同输入和测量配置下逐项对比 `fla_npu.ops.triton.chunk_local_cumsum`，Ascend C 必须更快。 不得使用 Python wall time 下结论。

## 11. 测试设计

`tests/op_cases/chunk_local_cumsum.json` 是唯一 case 规格。`tests/operators/chunk_local_cumsum/accuracy/` 运行主精度与泛化矩阵，
`routes/` 固化 aclnn 与 `<<<>>>` 编译/发射契约。标签覆盖 accuracy、generalization、boundary、negative、route、performance、example；
A2/A3/A5 使用相同 case ID，平台只决定编译与运行目标。

## 12. 已知限制与演进计划

- g rank 至少 3，所有维度为正；head_first 当前必须 true。
- chunk_size 必须为 2 的幂，且满足 `B_T >= C`；交付矩阵至少覆盖 `P=1,C=64` 的变长序列尾块和
  `P=16,C=64` 的 dense 尾块。
- output_dtype 仅支持 FP32 别名。
- 变长序列物理 B=1，两个 host 整数数组必须同时提供；cu_seqlens 首项为 0、末项为 T 且非递减。
- chunk_indices_out 必须按 sequence-major 完整列出内部处理块；每条序列的索引对数为
  `ceil(sequence_length / B_T)`，处理块长度 B_T 由 UB、C 和 P 共同决定，不能直接按 C 构造。

后续扩展任何限制时，必须同时修改 host 拦截、API 错误码、README、JSON 与三平台回归；模板实例增长前先评估二进制体积和编译耗时。
