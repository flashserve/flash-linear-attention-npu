# 流水与同步

本文描述 AIC/AIV 多阶段算子的核内事件、跨核 flag、双缓冲和 4-head window 背压协议。

## AIC 矩阵流水

典型数据路径为：

```text
GM -> L1A/L1B -> L0A/L0B -> MMAD -> L0C -> GM/L1/UB
```

- 非 resident operand 使用独立的 L1A 和 L1B scratch ping/pong。
- L0A/L0B 每个 slot 分别管理 ready/free，保持两个 operand 的 MTE1 流水独立。
- L0C 在容量允许时使用 ping/pong；单 tile 占满容量时使用单槽并明确覆盖点。
- 同一份 L0C 结果只执行一次 Fixpipe copyout。
- 独立 operand 先搬运，依赖 operand 的 wait 紧贴第一条真实消费。

## AIV Row 流水

UB 先扣除 resident、state 和 direct-path 固定区，再为 input、output 分别规划 ping/pong：

```text
CopyIn
  -> wait/cast
  -> vector or register compute
  -> cast output
  -> CopyOut
```

- 多路独立输入先并行发起 GM->UB，再按消费顺序 wait。
- input、output、state 和 direct-path slot 不共享未经证明的生命周期。
- output 的 V->MTE3 和 MTE3->V 在同一 slot 内闭环。
- state carry 使用独立 FP32 ping/pong，一个 slot 计算时另一个可搬入或写回。

## 核内 Event

每个跨 pipe 依赖写成五元组：

```text
(producer pipe, consumer pipe, buffer slot, access direction, reuse point)
```

- `SetFlag/WaitFlag` 的 HardEvent 类型和 EventID 完全匹配。
- 双缓冲每个 slot 使用对应事件，不跨生命周期误复用。
- EventID 在 kernel 生命周期内统一分配、收尾等待并释放。
- `PipeBarrier<PIPE_V>()` 只表达 V pipe 内顺序，不能代替 MTE、MMAD 或 Fixpipe 事件。

## 跨 AIC/AIV 数据边

跨核 flag 按数据边计数，而不是按 helper 或函数调用计数。对每条边记录：

| edge | payload owner | set pipe | wait side | ready count | free count |
|---|---|---|---|---:|---:|
| AIV -> AIC | owner Vector subblock | V/MTE3 | AIC | 每有效或协议 head | 与 ready 配平 |
| AIC -> AIV | AIC/Fixpipe | M/Fixpipe | 两个 Vector subblock | 每有效或协议 head | 与 ready 配平 |

MIX AIC 1:2 下，非 owner Vector subblock 即使没有 payload，也必须执行协议要求的配平或 drain。完整窗口、tail、空任务和跳过计算的分支都要核算次数。

## 4-Head Window 背压

4-head window 不能继续依赖“同一 raw flag 最多积压两次”的隐含假设。默认使用以下一种显式协议：

- 每个 `bank/headOffset/data-edge` 独立 ready/free。
- 每条数据边使用容量至少覆盖最大在飞深度的 credit/free 计数。
- 严格证明 producer/consumer 顺序后复用 raw flag，并记录最大未消费 set 深度。

推荐按两个 bank、每 bank 四个 head 管理逻辑 slot：

```text
bank 0: head slot 0..3
bank 1: head slot 0..3
```

ready 在 producer 对应 pipe 写入完成后发出；free 在最后消费者完成读取后归还。地址 bank、flag bank 和 generation 必须同步轮转。下一窗口覆盖 bank 前，全部 data edge 都已经 free。

## Wait 下沉

同一 stage 内先执行不依赖对端产物的工作：

```text
copy independent operands
prepare resident or addresses
wait dependent edge
consume dependent operand
compute and publish next edge
```

wait 过早会把原本可重叠的 MTE2、MTE1、M、V 和 Fixpipe 串行化。wait 过晚则可能形成 RAW。流水图应能说明每个 wait 紧贴真实消费者。

## 卡死定位

kernel launch 后使用明确 watchdog 区分加载问题和 kernel 卡死。卡死时依次检查：

1. task、window、active head、chunk 和 row 循环退出条件。
2. 每条数据边的 ready/free、set/wait 总数。
3. owner 与非 owner subblock 的配平路径。
4. bank、headOffset、buffer slot 和 EventID 是否同步轮转。
5. tail 和最后一个 tile 是否归还 free。
6. 多条同时在飞的数据边是否错误复用同一 flag。

race、越界、未初始化和同步问题使用对应 sanitizer，并确认运行时实际命中 sanitizer kernel。
