# 算子优化指南

本目录按“依赖模型 + 优化类别 + SOC 能力”组织线性 attention 类 Ascend C 算子的通用优化方法。这里不收录具体算子案例、代码路径、固定变量名或单次性能结果。

- `dependency/`：先判断 chunk 间是否存在 carry，确定并行与串行边界。
- `techniques/`：存放与具体算子和 SOC 无关的执行、内存、同步、Tiling 和调优方法。
- `soc/`：只补充各 SOC 落地时的能力和资源约束。
- `checklist.md`：汇总优化设计与交付检查项。

## 阅读顺序

1. 先判断不同 chunk 是否共享递推 carry。
2. 无 carry 阅读 [`dependency/chunk-independent.md`](dependency/chunk-independent.md)，有 carry 阅读 [`dependency/chunk-dependent.md`](dependency/chunk-dependent.md)。
3. 所有优化任务阅读 [`techniques/execution-and-stage.md`](techniques/execution-and-stage.md)、[`techniques/memory-and-dataflow.md`](techniques/memory-and-dataflow.md) 和 [`techniques/pipeline-and-synchronization.md`](techniques/pipeline-and-synchronization.md)。
4. 涉及 tile、模板和物理布局时阅读 [`techniques/tiling-and-layout.md`](techniques/tiling-and-layout.md)。
5. 开始性能试验和分析 profiling 前阅读 [`techniques/performance-tuning.md`](techniques/performance-tuning.md)。
6. 根据目标平台阅读 [`soc/a2-a3.md`](soc/a2-a3.md) 或 [`soc/a5.md`](soc/a5.md)。
7. 交付前执行 [`checklist.md`](checklist.md) 和 [`../development/validation.md`](../development/validation.md)。

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

## Head Window 选择基线

Head window 是 Tiling、模板或内部调度参数，不是固定为 4 的全仓常量。先定义：

- `T_C`：AIC/Cube 相关生产阶段归一到单个 head 的服务时间。
- `T_V`：单个 AIV/Vector subblock 处理一个完整 head 的服务时间。
- `N_V`：能够并行、独立且负载均衡地处理完整 head 的 Vector subblock 数。

只有各 subblock 独立持有完整 head 时，聚合 Vector 消费间隔才可近似为 `T_V / N_V`。如果 subblock 共同拆分一个 head、存在串行依赖或负载不均衡，应直接从完整 timeline 读取实际消费间隔，不能机械套用除法。

- 选择能够填满预期 owner 并形成稳定流水的最小窗口。两个 Vector subblock 各自承包完整 head 时，先评估 2-head；只有一个 owner 时可以先评估 1-head。
- 当 `T_C` 不小于聚合 Vector 消费间隔，或者 2-head 已无明显流水空洞时，不应仅为让 Cube 领先而扩大到 4-head。
- 只有 Cube 明显快于聚合 Vector，且 2-head timeline 中存在可由额外在飞 head 隐藏的空洞、背压或排空间隔时，才评估 4-head。
- 4-head 也可以由跨 head/group 驻留等独立复用收益驱动，但必须单独证明容量、生命周期和端到端收益。
- 不设固定比例阈值；在相同功能、shape 和环境下固定比较 1/2/4-head 的 Task Duration、完整 pipe、wait 和资源占用。

AIC 和 AIV 始终使用一致的 `task -> window -> stage -> head` 顺序。双 bank 轮转时逻辑 per-head workspace slot 数为 `2 * windowSize`；覆盖 bank 前必须完成上一代 slot 的最后消费和 free。
