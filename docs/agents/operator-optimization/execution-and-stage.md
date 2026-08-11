# 执行模型与 Stage 设计

本文描述与 chunk 依赖类型和目标 SOC 无关的 task、stage、head window 及 AIC/AIV 分工方法。

## 先画数据依赖图

为每个中间量记录：

| 项目 | 内容 |
|---|---|
| producer | AIC、AIV、搬运 pipe 或递推 state |
| consumer | 当前或后续 stage、当前或后续 chunk |
| location | GM、workspace、UB、L1、L0 |
| dtype/layout | 数值精度和物理布局 |
| lifetime | 生产、ready、最后消费、free |
| reuse key | task、chunk、head、head group 或 tile |

先确定真实数据边，再放 stage 和同步。不要按公式行数或“一个矩阵乘一个 stage”机械切分。

## Stage 划分原则

- 同一 stage 优先放只依赖原始输入或在入口已经 ready 的计算。
- AIC 和 AIV 均只读取原始输入时可以并行启动。
- 跨侧中间量的 wait 紧贴消费者第一次真实读取，避免在 stage 入口过早阻塞独立搬运。
- row/tile 循环内不重复执行本可在 stage 级完成的同步。
- 同一逻辑 head 在所有 stage 中保持一致的 workspace slot 和 owner。
- stage 名称、数据语义和 slot 名称在方案、Tiling 和 kernel 中保持一致。

## AIC/AIV 分工

- AIC 负责矩阵乘、L0 tile 组织、矩阵累加和 Fixpipe 输出。
- AIV 负责逐元素 gate、exp、scale、归约、状态 row 更新和公开 dtype 转换。
- 中间量只由一侧生产，另一侧消费，不在两侧重复计算。
- 分工由数据规模和 pipe 能力决定，不能把矩阵主体改成 Vector/scalar 作为性能兜底。

## Head Window

窗口大小先按 [`README.md`](README.md) 中的服务时间模型与 profiling 选择，再按 stage 成组推进：

```text
for task in core_tasks:
    for headBase in range(0, headCount, windowSize):
        activeHeads = min(windowSize, headCount - headBase)
        bank = windowIndex & 1

        for head in activeHeads: stage_0(head, bank)
        for head in activeHeads: stage_1(head, bank)
        ...
        retire bank after the final consumers

        windowIndex += 1
```

标准约束：

- AIC 和 AIV 使用一致的 `task -> window -> stage -> head` 逻辑顺序。
- 一个 bank 提供 `windowSize` 个逻辑 per-head slot；双 bank 共 `2 * windowSize` 个。
- `activeHeads` 覆盖不足一个完整窗口的 tail，不能假设 head 数为窗口大小的倍数。
- window bank、GM workspace slot、UB ping/pong 和 L1/L0 slot 是不同层次的概念。
- 同组 head 共享数据时，按最小依赖键只生成一次，并保持到组内最后一次消费。
- 下一窗口覆盖 bank 前，所有数据边都必须完成最后消费和 free。

## 窗口选择证据

确定窗口大小前，必须给出：

1. 归一到单个 head 的 `T_C`、`T_V`、有效 `N_V`，以及 subblock 是否独立持有完整 head。
2. 1/2/4-head 候选的 UB、L1、L0、workspace 和 EventID 预算。
3. 完整窗口与 tail 的 set/wait、ready/free 和 owner 配平表。
4. 目标 shape 的 Task Duration、完整 timeline、wait 和主要 bound 对比。
5. 泛化范围、回退路径和多 SOC 影响。

两个 Vector subblock 各自持有完整 head 时，2-head 是首个候选。只有 Cube 明显快于聚合 Vector，且额外在飞 head 能消除 2-head 的流水空洞时，才选择 4-head；窗口扩大不能提升已经由 Vector 决定的稳态吞吐。

窗口大小不能按单个 shape 随意形成第二套 L0 路径；差异应由同一 L0 的 TilingData、模板或内部调度表达。

## 代码组织

- stage/head/row tile 的主顺序在调用点保持可见。
- 多处复用且承担复杂生命周期的搬运和 MMAD 才抽取 helper。
- helper 只表达一个所有权边界，避免隐藏 wait、slot 切换或 Fixpipe。
- Tiling 可得的 shape、stride、offset 和窗口参数在初始化阶段缓存，热循环不重复推导。
- 编译期模板只覆盖真正影响数据布局、指令选择或容量规划的维度。
