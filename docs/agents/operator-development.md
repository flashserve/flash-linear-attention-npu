# 算子开发方法论

本文面向 AI coding agent 和开发者，说明在本仓开发 AscendC 算子时应该怎样拆问题、定边界、写实现和做验证。这里优先沉淀通用方法论；具体算子的设计细节应放在对应算子的 README 或设计文档中。

## 推荐开发顺序

不要从“先写一个 kernel 试试”开始。推荐顺序是：

```text
需求目标
  -> 对标语义和公开接口契约
  -> 本仓可复用 NPU 模块
  -> 数据依赖图和并行边界
  -> L2/L0 分层设计和 workspace 规划
  -> cube/vector/搬运/同步分工
  -> 小 shape 单算子精度
  -> 目标 shape 和组合路径精度
  -> 特殊值、极端值域、尾块、varlen
  -> profiling 性能定位
  -> 回归用例、文档和 PR 描述
```

精度第一现场还没定位时，不要先调阈值、缩小输入 range、删 case、把 cube 改成 scalar/vector 兜底，或提前做性能重构。

## 三类信息源

开发前先把输入信息分成三类，避免把不同层次的问题混在一起。

数学语义：来自论文、三方对标实现或参考 Python/Triton，用来回答“应该算什么”。重点确认公式、返回值顺序、中间量命名、dtype 语义、初始/最终状态和异常行为。

NPU 转写：来自本仓已有 AscendC 实现，用来回答“在昇腾上怎样组织计算”。重点看相邻算子的 tiling、layout、Catlass GEMM、blocked solve、状态传播、cross-core flag、AIC/AIV 分工和 workspace 规划。

工程边界：来自接口、文档、测试和 PR 规则，用来回答“交付时必须一致到什么程度”。重点包括 dtype、layout、shape、预留参数、报错文本、返回码、PyTorch schema、README、aclnn 文档和 CI/Example/ST。

## 先定能力边界

每次新增或修改算子，都先写清楚：

- 本次支持哪些 layout、dtype、shape、SOC、chunk、head 关系和状态参数。
- 哪些参数是预留但不支持，是否需要代码拦截。
- 哪些中间量是公开返回，哪些只是内部 workspace。
- 无效 token、padding、partial chunk 脏区是否有公开语义。
- 哪些场景只是 correctness fallback，哪些场景是性能目标。

不能验证的能力不要在文档或 PR 中宣称已经支持。限制条件应写成公开可理解的约束，不写个人环境、内部路径或临时日志。

## 画数据依赖图

融合不是把多个操作塞进一个 kernel。先把计算拆成三类：

- 无跨 chunk 或跨 task 依赖的大并行计算：优先按 batch/head/chunk/tile 切分，矩阵主路径走 AIC cube/Catlass。
- 有串行依赖的状态传播：单独设计阶段、调度和 workspace，必要时复用已有状态传播算子。
- 只负责 layout、cast、copy、边界适配的辅助逻辑：放在 aclnn L2 或独立轻量 L0，避免污染核心热路径。

如果 L2 组合多个 L0 算子，要区分“接口拼接”和“融合算子”。PyTorch 层调用多个 torch op 只能证明功能能串起来，不能替代 AscendC L2/L0 对 workspace、layout、dtype、同步和性能边界的控制。

## 多阶段协同

不少反向或局部计算算子天然会拆成“矩阵主路径 -> vector 修饰 -> 矩阵主路径”的协同流水。设计时要明确每个阶段的生产者、消费者和中间张量所有权：

- AIC 负责可复用的矩阵结果，例如 score、local attention、post GEMM 输入。
- AIV 负责 gate、mask、scale、cast、padding、三角区域处理等逐元素或逐行修饰。
- AIC 再消费 AIV 写好的中间结果继续 GEMM，生成最终输出。

这种模式下，workspace 不是随手申请一大片临时内存，而是 producer-consumer 队列。每个 slot 都要有清晰含义：哪个阶段写、哪个阶段读、可被哪些 head 或 chunk 复用、什么时候 free。若一个 Q/K head 对应多个输出 head，workspace 和调度要按 head ratio 扩展，不能默认所有 head 一一对应。

跨核协同时，ready/free flag 要成对设计。AIC 产出一个 tile 后通知 AIV；AIV 完成 gate/mask 后通知 AIC；生产者复用 slot 前必须确认消费者释放。空任务、tail chunk、varlen 无效区也要维持同样的计数协议，避免某一侧等待一个永远不会发送的 flag。

## 硬件分工

先按硬件能力决定实现路径：

- 矩阵乘、三角 solve、矩阵求逆、post GEMM：优先 AIC cube、Catlass 或 blocked solve。
- exp、scale、mask、cast、pad、逐元素变换：优先 AIV vector，使用大块 `DataCopy/DataCopyPad` 和 repeat。
- 少量 shape、offset、metadata：可以标量处理，但不要进入热路径内层循环。

目标矩阵计算不要轻易退回 scalar/vector。若有尾块或脏数据，优先 pad/clean 成中性值后继续走大块 cube/vector；只有非目标 correctness fallback 才考虑更简单但低性能的路径。

## 维度关系和策略封装

不要把 shape 当作互相独立的数字。线性注意力类算子经常有这些耦合：

- `H_out` 或 `H_do` 可能是 `H_qk` 的整数倍，需要显式推导 `hRatio`，并用 `outHead / hRatio` 映射回 Q/K head。
- `K`、`V`、`chunkSize` 往往决定模板、tile shape、UB/L1 预算和 workspace slot 数。
- fixed length 和 varlen 的 loop index 到 `(batch, token_start, chunk_len)` 映射不同，不应把两套 offset 逻辑散落在 kernel 内层。

推荐把 fixed/varlen 抽成 strategy：对外提供统一的 `calculate(loopIdx)`，返回当前 batch、token 起点和 chunk 有效长度。kernel 热路径只消费这个结果；host tiling 负责校验 `cu_seqlens`、`chunk_indices` 是否成对出现，以及当前实现是否限制 `B`、chunk 索引形状或尾块行为。

## 搬运和生命周期

搬运效率会直接限制 vector 上限。输入 layout 尽量让热路径连续读写；一次搬运尽量覆盖整行、整 tile 或多个 cache line；double buffer 必须真的让 MTE 与 VEC/CUBE 重叠。

同一块 UB slot 每次换 owner，都要闭合生命周期。`PipeBarrier<PIPE_V>()` 只约束 V pipe 内部顺序，不能替代 MTE/V、MTE/CUBE 或 MTE3 读写之间的硬事件。cross-core flag 要保证 set/wait 计数平衡，partial chunk 或空 payload 场景也不能让消费者死等。

怀疑同步或生命周期问题时，用固定输入多跑。如果同一输入多次结果不一致，优先检查跨 pipe event、cross-core flag、UB 提前复用和 workspace 写区重叠。

## 编译期模板和运行时 tiling

编译期模板适合消除热路径分支，例如 dtype、固定维度、safe gate 这类会改变内部计算路径的选项。运行时 tiling 适合存放规模、offset、layout、workspace、任务划分和调度信息。

推荐方式是使用参数模板化：host tiling 写入必要字段，kernel 入口根据字段选择模板实例，模板内部用 `if constexpr` 裁剪路径。不要依赖 tiling key 承载 dtype、layout、特性开关或维度组合的路径分发，也不要为了每个属性组合滥用 tiling key；除非算子框架或模板注册机制明确要求，tiling key 只应作为少量必要 kernel 变体的标识。

## 精度定位

不要只看最终输出。复杂算子应尽量支持逐阶段定位，例如 gate-only、stage-only、状态传播和最终输出分别对比。

先区分结构性错误和数值误差：

- 结构性错误：固定行、固定 chunk、固定 head、块状/条纹状误差、维度映射错误、NaN/Inf、固定输入多跑不一致。必须回到 kernel、layout、offset、mask、同步或 workspace 修复。
- 数值误差：误差随机分散、双标杆显示 test 和 benchmark 同数量级、没有固定结构模式。再评估迭代次数、fp32 workspace、阈值语义和性能取舍。

如果最终输出爆 NaN/Inf，先追第一处非有限值或第一处极大值。很多问题的第一现场在 gate、layout、padding、tail row 或中间 workspace，不一定在最终 GEMM。

## 性能定位

性能结论以 profiling 为准，不用 Python wall time 直接下结论。先看 bound，再改代码：

- MTE2、VEC、CUBE、MTE3 是可以并行形成流水的。长序列、多 tile、多循环且流水稳定时，通常需要某条流水线利用率大于 80% 才能认为它是主要 bound；如果没有任何一条流水线达到这个量级，极大概率还有搬运、计算、同步、任务粒度或 double buffer 方面的优化空间。
- Scalar bound：检查热路径是否有大量 `GetValue/SetValue` 或逐元素循环。
- MTE bound：检查搬运是否太碎、重复读写、layout 不连续或 `DataCopyPad` 粒度过小。
- VEC bound：检查 AIV 是否承担了本应由 cube 完成的矩阵工作，或 vector repeat 粒度过小。
- AIC/AIV wait：检查 producer-consumer 队列、cross-core flag、MTE3 写回、double buffer 和流水距离。

不要在 `for d` 里塞大量小搬运和小 vector 指令。优先整行/整 tile 搬入，一次 vector 指令处理大块数据，再整块写回。

## 交付闭环

交付前按根目录 `AGENTS.md` 的“算子开发交付 checklist”逐项核对。尤其注意：

- 接口契约、代码拦截、报错文本、返回码和文档约束必须一致。
- `op_host`、`op_api`、kernel、PyTorch schema、Python 导出、测试和示例要同步。
- 修改公共模块时，要列出受影响算子并扩大验证范围。
- 新增 bugfix 应有能稳定触发的回归用例。
- PR 只写公开测试项和结果，不暴露本地机器、账号、绝对路径、临时目录、日志路径或内部环境。
