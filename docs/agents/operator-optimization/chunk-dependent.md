# Chunk 间有依赖算子

本文适用于 chunk 之间通过状态 carry 形成正向或反向递推的算子。

## 固定递推语义

典型递推形式如下：

```text
forward:  S[i + 1] = F(S[i], X[i])
backward: S[i]     = F(S[i + 1], X[i])
```

实现前必须明确：

1. 递推方向和第一个执行 chunk 的状态来源。
2. 当前 chunk 输出的是更新前还是更新后状态。
3. carry、workspace 和公开输出分别使用的 dtype。
4. 哪些轴共享状态，哪些轴完全独立。

## Task 所有权

共享同一 carry 的完整 chunk 链由一个逻辑 task 按依赖方向处理：

```text
task = independent_axes

for task = coreIdx; task < taskNum; task += blockDim:
    initialize state
    for chunk in dependency order:
        process chunk
    store final state
```

不要把共享 carry 的单个 chunk 直接分配给不同 core，除非方案明确引入全局状态、跨核同步或多次 kernel launch，并证明收益大于复杂度。并行度优先来自 sequence、batch、不共享 carry 的 head group 和可独立 state tile。

## 状态精度和布局

长递推链中的 carry 优先保持 FP32：

- GM workspace 按 FP32 字节规划状态段。
- UB 为 state 分配独立 ping/pong。
- 当前 row tile 搬入 UB、更新后写回 FP32 workspace。
- 只在公开输出边界转换为接口 dtype。

workspace 状态段必须明确 old/new state 的地址、更新顺序和最后消费点。不能让当前 chunk 在所有消费者读取 old state 前覆盖它。

## Fixed 与 Varlen

定长和变长差异收敛在 sequence/chunk offset helper。kernel 主 stage 只消费规范化后的 batch、token 起点、有效长度和输出 chunk 下标。

- sequence 前缀和 canonical chunk 映射在 task 粒度预计算。
- 热循环不重复扫描 `cu_seqlens` 或重建 chunk 映射。
- 非 canonical 输入进入显式低频路径，不污染主流水。
- 空 sequence、短尾 chunk 和反向递推起点单独覆盖。

## Head Window

Head window 只组织同一递推 task 内互不共享 carry 或具有明确共享关系的 head。每个 head 的完整状态责任由一个 Vector subblock 持有，避免 row tile 和 state owner 在两个 subblock 之间来回切换。

- 两个 Vector subblock 可以交替承包完整 head。
- 采用上述 owner 模型时先评估 2-head；只有服务时间和完整 timeline 证明额外在飞 head 能隐藏流水空洞时才扩大到 4-head。
- owner 生产 payload，非 owner 仍按协议完成必要的 flag 配平。
- 完整窗口和不足 `windowSize` 的 tail 使用相同状态与信号模型。
- 两个 window bank 轮转时规划 `2 * windowSize` 个逻辑 per-head workspace slot。
- carry slot 和普通矩阵中间量 slot 分开定义生命周期。

## Stage 结构

通用递推 stage 可以表达为：

```text
next/previous state ready
  -> save old state and prepare independent inputs
  -> AIC/AIV consume old state and exchange intermediates
  -> update current state
  -> next chunk in dependency order
```

阶段编号不能代替数据依赖。每条跨核同步必须对应一条明确的数据边，并在完整窗口、tail、空 payload 和所有分支中核算 set/wait 次数。

## 验证重点

- 正向和反向递推分别确认初始状态与输出状态语义。
- 单 chunk、两 chunk、长递推链和短尾 chunk。
- fixed/varlen、单 sequence、多 sequence 和空 sequence。
- 所选完整 head window、尾窗口及多个连续窗口。
- grouped/GVA、可选初始/最终状态和全部互斥分支。
- state 使用 FP32 carry 时检查每个有效输出，并确认无未初始化或提前覆盖。
