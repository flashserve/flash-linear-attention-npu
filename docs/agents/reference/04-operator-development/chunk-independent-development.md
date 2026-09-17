# Chunk 间无依赖的算子开发参考

> 开发参考角色：`CHUNK_INDEPENDENT`
>
> 开发参考版本：`V1`
>
> 当前来源：[PR #404](https://github.com/flashserve/flash-linear-attention-npu/pull/404)
>
> 当前参考算子：`ChunkGatedDeltaRuleBwdFinalize`
>
> 当前参考目录：`fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_finalize/`
>
> 提炼来源 commit：`a78efa9512dfa64c20b2a58a75e84e88a054d694`

本参考用于实现“每个 chunk 的输入在 kernel 启动前已经完备，各 chunk 可以独立完成”的算子。当前算子的公式、Stage、shape、资源数量和支持范围由已评审的 `docs/design.md` 决定；本参考提供从详设落到 task 分片、AIC/AIV 流水、workspace、同步和工程代码的方法。

## 1. 参考架构

当前参考算子把完整计算划分为 `Stage 0--15`：

- Vector 执行 S0、S2、S4、S6、S8、S10、S12、S14 和最终归并 S15。
- Cube 执行 S1、S3、S5、S7、S9、S11 和 S13。
- Vector 到 Cube 的中间量通过 GM/workspace 传递；Cube 到 Vector 使用 Fixpipe 写入 owner AIV 的 UB resident。
- 有直接依赖的 Stage 逐 head 握手；没有直接依赖的 Stage 成组执行并与另一引擎重叠。

新算子根据自己的设计文档建立同类执行表：

| 字段 | 必须写清的内容 |
| --- | --- |
| Stage | 编号、公式和执行单元 |
| 任务 | task/chunk/head/head group 映射 |
| 数据 | 输入、输出、dtype、shape、layout 和 offset |
| 存储 | GM、workspace、L1、L0、UB resident 及复用时刻 |
| 同步 | producer、consumer、flag/event、发布和释放位置 |
| tail | 有效行、padding、mask 和写回范围 |

## 2. 独立 task 分片

1. fixed 场景把 `[batch, chunk]` 展平为 task；当前参考的 `taskNum = B * ceil(T / chunkSize)`。
2. varlen 场景由 `chunk_indices` 的 `[seqIdx, chunkIdx]` 生成独立 task，并结合 `cu_seqlens` 计算 packed token offset 和有效长度。
3. AIC 采用静态跨步分片：`taskIdx = coreIdx, coreIdx + coreNum, ...`。当前参考的 `blockDim = min(AIC 核数, taskNum)`。
4. 每个 task 独立读取输入、使用独立 workspace 并写回输出，因此不同 core 可以任意交错执行 task。
5. task 内按 head group 推进。当前参考根据 `HV/HK` 选择 group 大小，并由两个 AIV 交错承包 head。

目标算子的 task 粒度、head group 大小和 blockDim 由自身任务数量、L1/UB 容量、AIC/AIV 比例和 profiling 结果决定。

## 3. AIC/AIV 流水

一个 AIC 顺序承包当前 task group 的全部 head，两个 AIV 使用相同的 head 顺序并按 owner 规则交错处理：

1. AIV 先成组完成 S0，逐 head 通知 AIC；AIC 逐 head 执行 S1。
2. AIV 在对应结果到达后执行 S2；AIC 可以同时为其他 head 推进后续 Cube Stage。
3. AIC 对同一类 Cube Stage 连续处理整个 group，使 MTE2、MTE1、Cube 和 Fixpipe 双缓冲持续轮转。
4. Cube 结果逐 head Fixpipe 到 owner AIV；两个 AIV 都按相同顺序消费通知，owner 执行 Vector Stage，non-owner 只闭合同一 EventID。
5. 没有直接依赖的分支可并行。例如当前参考的 S6 只依赖 S3，可以与 AIC 的 S5/S7 重叠；S11 与 S10 无数据依赖，可独立推进。
6. task group 末尾等待仍在使用共享 UB resident 的消费者完成，再进入下一 group。

按依赖链组织“成组执行”和“逐 head 握手”，避免在每个 Stage 后做整组 barrier。

## 4. 数据驻留与 workspace

当前参考使用以下方法减少重复搬运：

- Stage 首次读取时把后续多次复用的标量、向量或矩阵保留在 UB/L1 resident。
- 不同生命周期的中间量复用同一物理区域；覆盖前由本地 event 或跨核 flag 证明最后一个消费者已经完成。
- UB 地址布局使用编译期常量，并以 `static_assert` 校验边界和总容量。
- Vector 到 Cube 的中间量写入 GM/workspace，AIC 再搬入 L1；workspace slot 按 task 和 head/head group 隔离。
- 当前参考为每个 task、每个 HV 预留 8 个 vector 大小的 workspace 区域，使 AIV 推进下一 task 时不会覆盖 AIC 尚未消费的数据。

目标算子在 `docs/design.md` 中计算每个 resident、ping-pong bank、workspace slot 的字节数、地址范围和生命周期，再把这些结果写成常量、tiling 字段和容量断言。

## 5. host tiling 与模板

host tiling 完成以下工作：

1. 从 tensor descriptor 校验 shape、dtype、layout、属性、head 比例和可选输入。
2. 计算 fixed/varlen task 数、每个 task 的 token 起点、有效长度和状态索引。
3. 计算 task group、active AIC、blockDim 和 workspace 大小。
4. 把 kernel 热路径使用的 offset、有效长度、模式和模板字段写入 tiling data。
5. TilingKey 只表示真实执行路径。当前参考只生成主数据 BF16、gate/beta 为 BF16 或 FP32 的有效组合。

fixed/varlen 和 tail chunk 通过统一的 `ChunkInfo` 进入 kernel；无效 task 在访问数据前结束。

## 6. 同步实现

当前参考使用两类同步：

- 本核 pipe event：保护 GM/L1/L0/UB 搬运、矩阵计算和物理 buffer 复用。
- AIC/AIV cross-core flag：按 head 顺序传递 Vector-to-Cube ready、Cube-to-Vector ready 和 task group 释放状态。

实现时逐项满足：

1. 首轮开始前初始化所有可复用 buffer 的事件。
2. producer 的数据实际到达目标存储后再发布 ready。
3. consumer 等待 ready、完成最终读取，再发布 free 或下一 Stage ready。
4. 两个 AIV 对跨核 flag 使用相同顺序；non-owner 同样消费通知，保持 EventID 对齐。
5. 任务组边界只等待仍会阻止物理区域复用的消费者。
6. kernel 退出前闭合所有初始化或末轮遗留的本地 event。

跨核 ready 只表示跨引擎数据可用；本核 MTE3、VEC 或 buffer 覆盖仍使用对应的本地 event。

## 7. tail 与 head 归并

- `chunkLen = min(chunkSize, sequenceEnd - tokenStart)`，所有 Stage 使用同一有效长度。
- Cube 通过 padding、中性值或有效 M/N/K 配置保持矩阵主路径；Vector 使用 mask 限定有效元素和写回范围。
- head ratio 大于 1 时，每个 HV 先独立计算 partial，最终 Stage 再按 HK 归并。当前参考的 S15 由 AIV0 汇总 HV partial，并处理可选 norm backward。
- 归并前确保每个 partial 的 MTE3 已完成；归并输出按 HK 和 task 使用独立地址。

## 8. 从设计生成代码

按以下顺序实现：

1. 在 tiling struct 和 `ChunkInfo` 中定义 task、head group、offset、workspace 和模式字段。
2. 在 host tiling 中实现校验、task 展平、blockDim、group、workspace 和 TilingKey。
3. 实现 Vector 的偶数 Stage、resident 和 owner 规则。
4. 实现 Cube 的奇数 Stage、L1/L0 double buffer 和 Fixpipe copyout。
5. 按设计依赖把逐 head flag、成组循环和可并行分支连接起来。
6. 为物理地址布局、buffer 边界和模板约束增加编译期断言。
7. 在 kernel 入口按 AIC/AIV 和 TilingKey 分发模板实例。
8. 接入 op_host、InferShape、op_api、schema、Python wrapper、构建和测试入口。

## 9. 定点查看当前实现

本参考缺少具体 API 或类名时，按问题读取当前来源中的对应文件：

| 需要确认的内容 | 文件 |
| --- | --- |
| task、fixed/varlen、blockDim、workspace 和 TilingKey | `op_host/op_tiling/chunk_gated_delta_rule_bwd_finalize_tiling.cpp` |
| tiling data 和常量 | `op_kernel/arch35/chunk_gated_delta_rule_bwd_finalize_struct.h` |
| task 到 chunk/token/state 的映射 | `op_kernel/arch35/chunk_gated_delta_rule_bwd_finalize_common.h` |
| AIC Stage、L1/L0 流水、Fixpipe 和跨核握手 | `op_kernel/arch35/chunk_gated_delta_rule_bwd_finalize_cube.h` |
| AIV Stage、UB 布局、owner 规则和本地 event | `op_kernel/arch35/chunk_gated_delta_rule_bwd_finalize_vector.h` |
| kernel 模板分发 | `op_kernel/chunk_gated_delta_rule_bwd_finalize.cpp` |

## 10. 版本维护

同一来源中改进实现细节时，更新“提炼来源 commit”和受影响章节，开发参考版本保持不变。来源 PR、参考算子或核心调度架构更换时，开发参考版本升级为 `V2`、`V3`，并增加迁移说明。算子设计文档记录实际采用的版本、commit、采用内容和差异。
