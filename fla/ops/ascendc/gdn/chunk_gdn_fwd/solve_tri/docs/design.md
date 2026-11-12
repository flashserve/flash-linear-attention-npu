# SolveTri 实现概述

接口和支持范围由 [api.md](api.md) 定义。本文记录当前代码的主要数据流，供维护和后续设备验证使用；它不是 `docs/agents/03-方案设计.md` 所要求的完整 V2 资源评审记录。

## 分发与任务

`solve_tri_tiling.cpp` 根据 layout、shape、metadata 和 chunk size 生成任务数与 TilingKey。定长任务数为 `B * ceil(T/C) * H`，变长任务数为所有非空序列 chunk 数之和乘以 `H`。尾 chunk 的有效长度由序列终点或 `T` 确定。实际 AIC 数不超过设备 AIC 数；C=16/32 按每批 8/2 个 tile 分组，C=64/128 按 tile 均匀分配。

| SoC | chunk size | 主要路径 |
| --- | --- | --- |
| A2/A3 | 64 | `SolveTriCubeFp32` 和 `SolveTriVectorFp32` 协作，FP32 中间数据保存在 GM workspace。 |
| A2/A3 | 16、32、128 | `SolveTriVectorFullFp32` 在 AIV 上使用前向替代，输入转 FP32，求解后转回输入 dtype。 |
| A5 | 16、32、64、128 | `arch35/solve_tri_ascend950_*.h` 对应实现；片上和 workspace 的具体使用依 chunk size 而异。 |

新增的参数及变长元数据校验仅在 A2/A3 启用：ACLNN 按运行时 `DAV_2201` 分支，tiling 按运行时 `ASCEND910B` / `ASCEND910_93` 分支。A5 保留 `546859c7` 的参数检查、元数据计数、TilingKey 回退和计算入口；`arch35/` 实现未修改。注册配置独立构造，A2/A3 使用 `PrecisionReduceFlag(false)`，A5 保持原值 `true`。

## A2/A3 chunk 64 流水

每个 AIC 的工作域是 tiling 分配的连续 tile 区间。一个 batch 最多 16 个 tile，四个 batch window 按 `batch_id % 4` 复用。一个 tile 有四个 `64 x 64` FP32 矩阵 slot：`X`、`TMP`、`MNEG`、`Y`。每 AIC 用户 workspace 为 `4 * 16 * 4 * 64 * 64 * sizeof(float) = 4 MiB`；系统 workspace 另计。给定批内 tile 下标 `tile_in_batch`，矩阵起始地址的 float 偏移为 `core * (4 * 16 * 4 * 64 * 64) + ((batch_id % 4) * 16 + tile_in_batch) * (4 * 64 * 64) + matrix_slot * (64 * 64)`。

1. AIV 读取有效区域，在 FP32 中构造 `-A` 和 16 阶对角块的逆，写入该 batch 的 workspace，并通知 AIC。
2. AIC 等待输入就绪，执行 16 到 32 阶合并乘法并通知 AIV；AIV 将结果折叠到 `X`，再通知 AIC。
3. 启动时 AIV 最多预备三个 batch；处理中，AIV 在折叠后准备 `batch_id + 3`，与 AIC 的 32 到 64 阶合并重叠。AIC 等待折叠完成后将最终乘积写入 `Y` 并通知 AIV。
4. AIV 等待最终乘积，直接从 `X` 和 `Y` 读取各自负责的最终结果，在 UB 中拼接后按输入 dtype 转换并写回有效区域；不再将最终 `Y` 乘积写回 GM 中的 `X`。跨核 ready/free 使用 `CrossCoreFlagWithReverse`；局部搬运依赖使用硬件 event 和 pipeline barrier。

### 最终结果读取与输出

32 到 64 阶合并的第二次 GEMM 仍由 AIC 写入独立的 `Y[0:32, 0:32]`，保持与输入 `X` 的地址隔离。AIC 完成 Fixpipe 后通知 AIV，AIV 保留最终的 `WaitAic()`，再将以下区间搬入已有的 `fp32LocalA_`。该 UB buffer 为 `32 x 64` FP32，行跨度为 64 个元素；表中区间均为行、列半开区间。

| 执行通道 | GM 来源 | UB 目的区间 | 读取量 |
| --- | --- | --- | --- |
| AIV0 | `X[0:32, 0:64]` | `[0:32, 0:64]` | 8 KiB |
| AIV1 | `Y[0:32, 0:32]` | `[0:32, 0:32]` | 4 KiB |
| AIV1 | `X[32:64, 32:64]` | `[0:32, 32:64]` | 4 KiB |

AIV1 的两次 strided GM 到 UB 搬运写入互不重叠的左右半块，共用一次 `V_MTE2` 设置与等待以及一次 `MTE2_V` 设置与等待；搬运完成后才开始 cast。AIV0 读取整条上半区。两个通道随后均沿用原有 cast、`V_MTE3`/`MTE3_V` 依赖和有效区域写回，不改变数值运算或舍入方式；尾块仍按有效行列裁剪输出，内部读取只访问已初始化的完整 workspace tile。

最终阶段每 tile 共读取 16 KiB 中间结果。相对先将 `Y` 搬入 UB、写回 `X` 再读取完整 `X` 的路径，省去 4 KiB 的 GM 写回及 4 KiB 的重复读取，共减少 8 KiB 逻辑 GM 流量。该路径复用原有 `fp32LocalA_`，不增加 UB、GM workspace 或 event 资源数量；`GetUserWorkspace` 和 FFTS 收尾保持原有契约。

最终 GM 中的 `X` 不再保存完整的 64 阶结果，其左下块不能作为最终结果读取。该 tile 在输出后没有其他算法消费者；同一 ring slot 只在当前输出完成后由后续 `PrepareBatch` 重新初始化。16 到 32 阶的 `Y` 到 `X` 折叠仍然保留，因为后续 AIC 合并必须读取它。

每个 batch 的 slot 直到消费完成后才能复用。chunk 64 分支在 Cube/Vector 算子及其局部资源全部析构后执行 `PIPE_ALL`，再将 `FFTS_BASE_ADDR` 恢复为默认零值，避免提前清理仍被跨核同步使用的寄存器。长流水回绕、竞态与内存检查的实际结果见算子 ATK README。

## 其他 workspace

A2/A3 的 C=16/32/128 路径不分配用户 workspace，C=64 按上述 4 MiB/AIC 分配。

A5 完整保留 `546859c7` 的 workspace 分配分支，包括原有编译宏条件。通常的 host 编译未定义 `__CCE_AICORE__`，按原公式分配：C=64 为 `65536 * N` 字节，其他 C 为 `(6 + 4 * N) * C * C` 字节，其中 N 为实际使用的 AIC 数。若 host 显式定义 `__CCE_AICORE__ == 310`，则沿用原有每核 FP32 矩阵公式（C=16 为零，C=32 使用 64 阶，其余使用 C 阶）。各项按 512 字节对齐，并额外加上系统 workspace。本次仅隔离 A2/A3 的新增分配，不修改或证明 A5 基线分配的容量正确性。

## 尚需设备证据

完整 V2 设计评审仍需补齐各路径 L1/UB 绝对地址、活跃区间和容量、逐 stage 搬运量与同步资源闭环，以及目标 shape 的同口径性能和额外显存测量。没有这些数据时，不应将本文视作资源、性能或同步正确性的设备验证结论。
