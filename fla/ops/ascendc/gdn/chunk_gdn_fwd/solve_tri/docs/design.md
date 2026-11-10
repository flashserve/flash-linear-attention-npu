# SolveTri 实现概述

接口和支持范围由 [api.md](api.md) 定义。本文记录当前代码的主要数据流，供维护和后续设备验证使用；它不是 `docs/agents/03-方案设计.md` 所要求的完整 V2 资源评审记录。

## 分发与任务

`solve_tri_tiling.cpp` 根据 layout、shape、metadata 和 chunk size 生成任务数与 TilingKey。定长任务数为 `B * ceil(T/C) * H`，变长任务数为所有非空序列 chunk 数之和乘以 `H`。尾 chunk 的有效长度由序列终点或 `T` 确定。实际 AIC 数不超过设备 AIC 数；C=16/32 按每批 8/2 个 tile 分组，C=64/128 按 tile 均匀分配。

| SoC | chunk size | 主要路径 |
| --- | --- | --- |
| A2/A3 | 64 | `SolveTriCubeFp32` 和 `SolveTriVectorFp32` 协作，FP32 中间数据保存在 GM workspace。 |
| A2/A3 | 16、32、128 | `SolveTriVectorFullFp32` 在 AIV 上使用前向替代，输入转 FP32，求解后转回输入 dtype。 |
| A5 | 16、32、64、128 | `arch35/solve_tri_ascend950_*.h` 对应实现；片上和 workspace 的具体使用依 chunk size 而异。 |

## A2/A3 chunk 64 流水

每个 AIC 的工作域是 tiling 分配的连续 tile 区间。一个 batch 最多 16 个 tile，四个 batch window 按 `batch_id % 4` 复用。一个 tile 有四个 `64 x 64` FP32 矩阵 slot：`X`、`TMP`、`MNEG`、`Y`。每 AIC 用户 workspace 为 `4 * 16 * 4 * 64 * 64 * sizeof(float) = 4 MiB`；系统 workspace 另计。给定批内 tile 下标 `tile_in_batch`，矩阵起始地址的 float 偏移为 `core * (4 * 16 * 4 * 64 * 64) + ((batch_id % 4) * 16 + tile_in_batch) * (4 * 64 * 64) + matrix_slot * (64 * 64)`。

1. AIV 读取有效区域，在 FP32 中构造 `-A` 和 16 阶对角块的逆，写入该 batch 的 workspace，并通知 AIC。
2. AIC 等待输入就绪，执行 16 到 32 阶合并乘法并通知 AIV；AIV 将结果折叠到 `X`，再通知 AIC。
3. 启动时 AIV 最多预备三个 batch；处理中，AIV 在折叠后准备 `batch_id + 3`，与 AIC 的 32 到 64 阶合并重叠。AIC 等待折叠完成后将最终乘积写入 `Y` 并通知 AIV。
4. AIV 等待最终乘积，将 `Y` 折叠到 `X`，按输入 dtype 转换并写回有效区域。跨核 ready/free 使用 `CrossCoreFlagWithReverse`；局部搬运依赖使用硬件 event 和 pipeline barrier。

每个 batch 的 slot 直到消费完成后才能复用。chunk 64 分支在 Cube/Vector 算子及其局部资源全部析构后执行 `PIPE_ALL`，再将 `FFTS_BASE_ADDR` 恢复为默认零值，避免提前清理仍被跨核同步使用的寄存器。长流水回绕、竞态与内存检查的实际结果见算子 ATK README。

## 其他 workspace

A2/A3 的 C=16/32/128 路径不分配用户 workspace。A5 的 C=16 用户 workspace 为 0；C=32 按每 AIC 一个 `64 x 64` FP32 矩阵分配；C=64/128 按每 AIC 一个 `C x C` FP32 矩阵分配。各项按 512 字节对齐，并额外加上平台库所需的系统 workspace。

## 尚需设备证据

完整 V2 设计评审仍需补齐各路径 L1/UB 绝对地址、活跃区间和容量、逐 stage 搬运量与同步资源闭环，以及目标 shape 的同口径性能和额外显存测量。没有这些数据时，不应将本文视作资源、性能或同步正确性的设备验证结论。
