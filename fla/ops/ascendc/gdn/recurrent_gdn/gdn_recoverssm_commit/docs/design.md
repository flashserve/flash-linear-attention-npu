# GdnRecoverssmCommit Ascend 950 设计方案

方案规则版本：V2

## 1. 目标与范围

接口以 [api.md](api.md) 为唯一事实来源。实现 `[V,K]` checkpoint 的倒序
replay 组合、可选 boundary checkpoint 写回，并支持多个 layer 的独立基地址。

首版优先保证与 PR 20485 的 Triton 求值顺序和 page 选择行为一致。数值没有
Cube 计算，全部为 AIV 标量和向量操作。

## 2. 任务映射

```text
workPerRequest = Layer * HV * 2
task = request * workPerRequest + (layer * HV + head) * 2 + value_tile
core = task % blockDim
wave = task / blockDim
```

`blockDim = ceil(taskCount / ceil(taskCount / GetCoreNumAiv()))`，减少不足
一个 wave 时的空核。每个 task 处理一个 `(request, layer, head, value_tile)`，
其中 `value_tile` 覆盖 value 维的 64 行。`value_tile` 是最低维任务分量，
相邻的两个 tile 会优先分配到不同 AIV；不同 task 写不同 page/head/value
区间。每个 task 的主体按 `CopyIn -> Compute -> CopyOut` 封装，`Process`
只负责任务遍历、metadata 准备和双缓冲 slot 轮转。

`GdnRecoverssmCommitTilingData` 只保存 kernel 无法从运行上下文获得的信息：

- `taskCount`、`valueHeads`、`layerCount`：解码 request/layer/head/value tile；
- `stateHeadStride`、`stateValueStride`：读取和写回非连续 checkpoint；
- `replayHeadStride`、`replayPosStride`：读取非连续 replay。

block 数由 host 写入 `SetBlockDim`，kernel 通过 `GetBlockNum` 读取。dtype 和
`align_mode` 由模板参数表达；固定维度、UB 预算和请求级 metadata stride
不进入 tiling data。

## 3. Stage

### S0：读取 commit 计划

- 从 `commit_lens`、`state_indices`、`final_state_indices`、
  `boundary_state_indices`、`boundary_recovery_lens` 读取当前 request。
- 调用方保证 `1 <= n <= spec_query_len`，kernel 直接使用 `n`。
- 判定 `needBoundary`，并读取 layer 对应的 checkpoint/replay
  base address 与 block stride。
- page、地址表和长度均为合法输入，kernel 不增加运行期合法性判断或夹取。

### S1：倒序组合 replay

- 整个 window 的 replay records 使用一次多 block `DataCopyPad` 搬入当前
  record slot，block 间隔由 `replayPosStride - 257` 计算。每条 record 在 UB
  内补齐到 264 个 FP32，保证后续 VF 的 `LoadAlign` 地址满足 32B 对齐。
- 所有运算只由一个 `ProcessCommitValueTileVf` VF 函数完成；每次处理当前
  `value_tile` 的 64 个 value row。
- VF 每轮并行处理两个 value row，共享 key/decay load，并维护两组独立的
  correction/decay 寄存器链，使相互独立的乘加能够交错发射。
- boundary 按照 `boundaryLen` 拆成两个 reverse loop：先处理 final-only
  后缀，再处理 final+boundary 前缀。`needBoundary` 提升为 VF 模板参数，
  VF 内不保留运行时 boundary 判断。
- correction row、final/boundary accumulated decay 全部驻留寄存器。
- 对 `position = n-1 ... 0` 读取 `[correction, key, decay]`。
- `weighted_key = key * accumulated_decay`。
- 对每个 value row 执行
  `correction_row += correction_scalar * weighted_key`。
- `accumulated_decay *= decay`。
- align 且 boundary 有效时，对 `position < boundary_recovery_len`
  执行同一组操作，但更新独立 boundary 累加器。

### S2：合并初态并写回

- 初态转换、decay 乘法、correction 加法、BF16 输出 cast 均在同一个 VF 内完成。
- checkpoint 的当前 `64x128` value tile 一次搬入 UB，VF 内部遍历 64 行。
- `final = initial * final_decay + final_correction`。
- align 时在同一 row 的初态仍有效时同时计算 boundary 结果。
- boundary 先写回，final 后写回；BF16 checkpoint 只在最终写回时 cast。

## 4. UB 资源

一个 AIV 同时处理一个 head 的一个 value tile：

| 对象 | 大小 |
| --- | ---: |
| replay records 双缓冲 `2x32x264 FP32` | 66 KiB |
| source/final tile 双缓冲 `2x64x128 BF16/FP32` | 32 KiB/64 KiB |
| boundary tile 双缓冲 `2x64x128 BF16/FP32` | 32 KiB/64 KiB（仅 align） |
| 对齐和 guard | 4 KiB |

FP32 + align 的数据区约 `194 KiB`，计入 guard 约 `198 KiB`，低于 A5
可用 UB。final 复用 source UB，不额外分配矩阵；
correction 和 decay 不分配 UB 矩阵，
全部存放在 VF 寄存器中。无私有 workspace。

## 5. 同步

| 数据边 | 生产 | 消费 | 同步 |
| --- | --- | --- | --- |
| replay record | MTE2 | Vector | `GRSC_RECORD_MUTEX[slot]` |
| replay record 复用 | Vector | 下一次 MTE2 | `GRSC_RECORD_MUTEX[slot]` |
| source/final state tile | MTE2 | Vector | `GRSC_STATE_MUTEX[slot]` |
| final state tile 复用 | MTE3 | 下一次 MTE2 | `GRSC_STATE_MUTEX[slot]` |
| boundary tile | Vector | MTE3 | `GRSC_BOUNDARY_MUTEX[slot]` |
| boundary tile 复用 | MTE3 | 下一次 Vector | `GRSC_BOUNDARY_MUTEX[slot]` |

同一 Vector 内的矩阵更新和 decay 更新依靠 V pipe 程序顺序；必要时在复用同一
UB 行前使用 `PipeBarrier<PIPE_V>()`。不同 AIV 不共享 UB 或输出区间，不增加
跨核事件。

record 使用两个 task 级 slot 轮转。代码只描述当前 task 的搬运、VF 和写回，
不构造下一 task 元数据；后端按异步 MTE 队列和 mutex 自行形成重叠。record、
state 和 boundary 均使用两个 task 级 slot 轮转，final 直接复用 source UB。

## 6. TilingKey

模板参数为 checkpoint dtype、`align_mode`、plan metadata dtype 和
source metadata dtype：

```text
D_T_STATE       : BF16 | FP32
ALIGN_MODE      : 0 | 1
D_T_PLAN        : INT32 | INT64
D_T_STATE_INDEX : INT32 | INT64
```

host 使用 `ASCENDC_TPL_ARGS_DECL` / `ASCENDC_TPL_SEL` 声明组合，
`GET_TPL_TILING_KEY` 选择实例。每个实例只保留对应类型的
`GlobalTensor<planType>` 和 `GlobalTensor<stateIndexType>`，不生成
`finalIndices32Gm_`/`finalIndices64Gm_` 成对指针，也不在 kernel 内判断
metadata dtype。`spec_query_len` 只进入运行期循环上界，不手写数字 tiling key。

## 7. 正确性风险

- replay base/block stride 来自设备地址表，host 只能校验表布局。
- 多个 layer 共用 head/value/payload stride，调用方必须保证布局一致。
- 五个请求级 metadata 张量是一维连续输入，kernel 直接按 request 下标读取，
  不把逐张量 stride 写入 tiling data。
- source page 被 boundary 提前覆盖时，S2 已把同一 row 的 final 和 boundary
  结果都计算完成，因此结果仍等价于参考实现。
- BF16 初态在计算前只转换一次，组合过程不回写中间 BF16。
