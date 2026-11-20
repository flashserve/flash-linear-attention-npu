# GdnRecoverssmCommit Ascend 950 设计方案

方案规则版本：V2

## 1. 目标与范围

接口以 [api.md](api.md) 为唯一事实来源。实现 `[V,K]` checkpoint 的倒序
replay 组合、可选 boundary checkpoint 写回，并支持多个 layer 的独立基地址。

首版优先保证与 PR 20485 的 Triton 求值顺序和 page 选择行为一致。数值没有
Cube 计算，全部为 AIV 标量和向量操作。

## 2. 任务映射

```text
task = request * (Layer * HV) + layer * HV + head
core = task % blockDim
wave = task / blockDim
```

`blockDim = ceil(taskCount / ceil(taskCount / GetCoreNumAiv()))`，减少不足
一个 wave 时的空核。每个 task 独占一个 `(request, layer, head)`，不同 task
写不同 page/head 区间。

## 3. Stage

### S0：读取 commit 计划

- 从 `commit_lens`、`state_indices`、`final_state_indices`、
  `boundary_state_indices`、`boundary_recovery_lens` 读取当前 request。
- 计算有效长度 `n = min(max(commit_len, 0), spec_query_len)`。
- 判定 `needBoundary`，并读取 layer 对应的 checkpoint/replay
  base address 与 block stride。

### S1：倒序组合 replay

- correction 矩阵清零，final/boundary accumulated decay 初始化为 1。
- 对 `position = n-1 ... 0` 读取 `[correction, key, decay]`。
- `weighted_key = key * accumulated_decay`。
- 对每个 value row 执行
  `correction_row += correction_scalar * weighted_key`。
- `accumulated_decay *= decay`。
- align 且 boundary 有效时，对 `position < boundary_recovery_len`
  执行同一组操作，但更新独立 boundary 累加器。

### S2：合并初态并写回

- 逐 value row 读取 source checkpoint，转换为 FP32。
- `final = initial * final_decay + final_correction`。
- align 时在同一 row 的初态仍有效时同时计算 boundary 结果。
- boundary 先写回，final 后写回；BF16 checkpoint 只在最终写回时 cast。

## 4. UB 资源

一个 AIV 同时处理一个 head：

| 对象 | 大小 |
| --- | ---: |
| final correction `128x128 FP32` | 64 KiB |
| boundary correction `128x128 FP32` | 64 KiB（仅 align） |
| replay record `257 FP32` | 1028 B |
| final/boundary decay、weighted key | 2 KiB |
| source/final/boundary row临时量 | 2 KiB |
| 对齐和 guard | 4 KiB |

峰值约 `133 KiB`，低于 A5 可用 UB。`align_mode=False` 的模板不会分配 boundary
矩阵，峰值约 `69 KiB`。无私有 workspace。

## 5. 同步

| 数据边 | 生产 | 消费 | 同步 |
| --- | --- | --- | --- |
| replay record | MTE2 | Vector | `SetFlag/WaitFlag<MTE2_V>` |
| replay record 复用 | Vector | 下一次 MTE2 | `V_MTE2` |
| source state row | MTE2 | Vector | `MTE2_V` |
| final/boundary row | Vector | MTE3 | `V_MTE3` |
| row buffer 复用 | MTE3 | 下一次 MTE2/Vector | `MTE3_V` |

同一 Vector 内的矩阵更新和 decay 更新依靠 V pipe 程序顺序；必要时在复用同一
UB 行前使用 `PipeBarrier<PIPE_V>()`。不同 AIV 不共享 UB 或输出区间，不增加
跨核事件。

## 6. TilingKey

模板参数为 checkpoint dtype 和 `align_mode`：

```text
D_T_STATE  : BF16 | FP32
ALIGN_MODE : 0 | 1
```

host 使用 `ASCENDC_TPL_ARGS_DECL` / `ASCENDC_TPL_SEL` 声明组合，
`GET_TPL_TILING_KEY` 选择实例。`spec_query_len` 只进入运行期循环上界，
不手写数字 tiling key。

## 7. 正确性风险

- replay base/block stride 来自设备地址表，host 只能校验表布局。
- 多个 layer 共用 head/value/payload stride，调用方必须保证布局一致。
- source page 被 boundary 提前覆盖时，S2 已把同一 row 的 final 和 boundary
  结果都计算完成，因此结果仍等价于参考实现。
- BF16 初态在计算前只转换一次，组合过程不回写中间 BF16。
