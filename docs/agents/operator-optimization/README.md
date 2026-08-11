# 算子优化指南

本目录按“依赖模型 + 优化类别 + SOC 能力”组织线性 attention 类 Ascend C 算子的通用优化方法。这里不收录具体算子案例、代码路径、固定变量名或单次性能结果。

## 阅读顺序

1. 先判断不同 chunk 是否共享递推 carry。
2. 无 carry 阅读 [`chunk-independent.md`](chunk-independent.md)，有 carry 阅读 [`chunk-dependent.md`](chunk-dependent.md)。
3. 所有优化任务阅读 [`execution-and-stage.md`](execution-and-stage.md)、[`memory-and-dataflow.md`](memory-and-dataflow.md) 和 [`pipeline-and-synchronization.md`](pipeline-and-synchronization.md)。
4. 涉及 tile、模板和物理布局时阅读 [`tiling-and-layout.md`](tiling-and-layout.md)。
5. 开始性能试验和分析 profiling 前阅读 [`performance-tuning.md`](performance-tuning.md)。
6. 根据目标平台阅读 [`soc/a2-a3.md`](soc/a2-a3.md) 或 [`soc/a5.md`](soc/a5.md)。
7. 交付前执行 [`checklist.md`](checklist.md) 和上层 [`../validation.md`](../validation.md)。

## 分类边界

- 依赖模型文档决定哪些轴必须串行、哪些轴可以并行。
- 技术分类文档描述与具体算子和 SOC 无关的优化机制。
- SOC 文档只描述容量、指令、数据通路、API 和流水能力差异，不重复定义通用机制。
- 具体算子的 README 和设计文档可以说明某次实现，但不能成为其他算子直接复制数字和同步协议的依据。

## 优化项写法

新增优化方法时至少写清楚：

| 字段 | 内容 |
|---|---|
| 目标 | 要消除的计算、搬运、同步或调度开销 |
| 适用依赖 | chunk-independent、chunk-dependent 或两者 |
| 必要前提 | 数据复用、容量、layout、API 和生命周期条件 |
| 设计方法 | task、stage、slot 和数据通路如何变化 |
| 风险 | 精度、覆盖、同步、容量和兼容性风险 |
| 回退 | 条件不满足时保留的同语义实现 |
| 验证 | 精度、流水、sanitizer 和性能证据 |

不要只写“可使用双缓冲”“可尝试驻留”一类口号，也不要把特定 shape 的 slot 数、容量结果或代码位置写成全仓规则。

## 默认调度基线

多 head 的线性 attention 类算子默认以 4-head window 组织 stage：

- 一个逻辑窗口最多处理 4 个 head。
- tail window 处理剩余 1 至 3 个 head。
- 两个 window bank 轮转时，规划 8 个逻辑 per-head workspace slot。
- AIC 和 AIV 使用一致的 `task -> window -> stage -> head` 顺序。
- 跨窗口覆盖前必须完成上一代 slot 的最后消费和 free。

4-head window 是设计基线，不是无需证明的常量。容量、事件或性能无法满足时可以缩小窗口，但必须记录预算、同步协议、实测对比和验证结果，不得静默退回。
