# 算子开发方法论

本文面向 AI coding agent 和开发者，说明在本仓开发 AscendC 算子时应该怎样拆问题、定边界、写实现和做验证。这里优先沉淀通用方法论；具体算子的设计细节应放在对应算子的 README 或设计文档中。

首次参与本仓开发时先读 [`../foundation.md`](../foundation.md)。开始编写或重构代码前阅读 [`coding-standard.md`](coding-standard.md)。涉及性能设计、stage、resident、head window、流水或 SOC 优化时，按 [`../optimization/README.md`](../optimization/README.md) 加载对应分类文档。

## 推荐开发顺序

不要从“先写一个 kernel 试试”开始。推荐顺序是：

```text
需求目标
  -> 对标语义和公开接口契约
  -> 本仓可复用 NPU 模块
  -> 数据依赖图和并行边界
  -> L2/L0 分层设计和 workspace 规划
  -> 文件职责、模板分派和资源生命周期设计
  -> cube/vector/搬运/同步分工
  -> 小 shape 单算子精度
  -> 目标 shape 和组合路径精度
  -> 特殊值、极端值域、尾块、varlen
  -> profiling 性能定位
  -> 回归用例、文档和 PR 描述
```

在精度问题的第一现场尚未定位、根因尚未正面修复前，以下行为全部禁止：

- 调整或放宽精度阈值。
- 缩小输入 range。
- 删除、跳过或弱化失败 case。
- 把目标 cube 计算改成 scalar/vector 兜底。
- 提前进行与根因定位无关的性能重构。

上述五项都是禁止行为；特别是“把 cube 改成 scalar/vector 兜底”和“提前做性能重构”也绝不是推荐方案。

## 三类信息源

开发前先把输入信息分成三类，避免把不同层次的问题混在一起。

数学语义：来自论文、三方对标实现或参考 Python/Triton，用来回答“应该算什么”。重点确认公式、返回值顺序、中间量命名、dtype 语义、初始/最终状态和异常行为。

NPU 转写：来自本仓已有 AscendC 实现，用来回答“在昇腾上怎样组织计算”。重点看相邻算子的 tiling、layout、Catlass GEMM、blocked solve、状态传播、cross-core flag、AIC/AIV 分工和 workspace 规划。

工程边界：来自接口、文档、测试和 PR 规则，用来回答“交付时必须一致到什么程度”。重点包括 dtype、layout、shape、预留参数、报错文本、返回码、PyTorch schema、README、aclnn 文档和 CI/Example/ST。

## 算子原型和公开接口兼容性

本规则统一适用于仓库内所有算子。算子原型、aclnn 接口和 `fla_npu.ops.ascendc.<op_name>` 接口一旦确定，必须作为兼容性契约保持稳定。除非仓库管理账号 `@weinachuan` 针对本次具体变更明确下达命令，任何开发或修复都不得修改以下内容：

1. `op_host/*_def.cpp` 中已有的输入、输出和属性，包括名称、数量、顺序、必选/可选、默认值和既有语义。允许在不删除或改变现有 dtype 支持和语义的前提下增加 dtype，并同步校验、文档和测试。
2. 算子 aclnn 接口，包括参数名称、数量、数据类型、顺序、默认值语义和原始默认行为，必须保持 ABI 和源码调用兼容。
3. `fla_npu.ops.ascendc.<op_name>` 公开接口，包括参数名称、数量、顺序、默认值语义和原始默认行为，必须保证现有用户代码兼容。

任何接口改动的想法、方案或试验，都必须在实施、提交代码或修改文档前向 `@weinachuan` 说明：拟修改前后的完整接口、为什么一定要修改，以及为什么不能通过已有输入或属性、InferShape、origin/view/storage shape、tiling data、workspace、内部结构或内部算子获取所需信息。还必须说明 ABI、源码和用户调用兼容性影响，并等待明确命令；未确认时不得先改后问。

对于已经发布的原型或接口，默认优先保持原接口不变并新增独立 V2 实现，让旧接口继续维持原始默认行为。是否新增 V2 以及 V2 的具体原型也必须先由 `@weinachuan` 明确确认，不得直接在原接口上做不兼容演进。

## L0 单路径设计、模板覆盖和信息归属

L0 是 L2 与 kernel 之间的内部算子契约。设计时先确定稳定的 L0 语义和调用图，再在其内部承载平台、shape 和流水差异。

### 模板覆盖规则

1. 分别声明性能目标、功能支持范围和模板优势域，不能把性能目标组合直接当作功能支持边界。
2. 需求没有明确限制的维度必须保持泛化。只有核心计算结构或资源规划存在实质差异时才拆分内部模板，不能为了少数规则 shape 或普通边界形态缩小模板覆盖范围。
3. 同一模板应覆盖由相同计算结构和资源规划能够获得性能优势的完整场景集合。泛化正确性与目标性能分别验收，不能用低性能 fallback 代替其中任何一项。

### L0 路径规则

1. 同一算子在所有支持的 SOC 上使用相同的 L0 算子定义、输入输出原型和 L2 调用路径；L2 不按平台选择不同 L0 名称、定义或融合调用图。
2. host tiling 负责读取平台能力、tensor descriptor 和已有属性，计算任务划分、tile、workspace 和模板选择。
3. tiling data 承载运行期规模、offset、有效长度、调度和 workspace 信息；kernel 编译期模板参数或架构 trait 表达可静态裁剪的平台和算法差异。
4. 融合实现覆盖既有功能范围、通过精度验证并达到目标性能后，删除被替代的未融合 L0 定义、路由、注册和构建入口；仍需复用的算法迁入统一 L0 的内部模板或组件。

### L0 参数规则

1. L0 输入、输出和属性只表达不可由现有信息推导的算子语义。
2. 可从 tensor descriptor、已有属性或平台能力获得的 shape、layout、dtype、SOC 和模板选择信息，由 InferShape 或 host tiling 推导。
3. tile、block、任务划分、流水深度、workspace offset 等调度信息通过 tiling data、workspace 或编译期模板参数传递，不得增加为 L0 入参。
4. 仅为实现分支服务的布尔开关、枚举或重复 shape 标量属于冗余入参，不得固化到 L0 原型。

### 设计确认规则

新增 L0 原型、修改已有 L0 签名、增加 V2 L0、拆分或融合 L0，或保留多条 L0 调用路径，都必须在开始实现、修改文档或补测试前向 `@weinachuan` 提交设计并等待明确确认。设计说明至少包含：

1. 修改前后的 L2/L0 调用图、算子名称和完整输入输出原型。
2. 复用现有 L0 不可行的原因，以及通过 InferShape、tiling data、workspace 或内部模板承载需求的评估。
3. 性能目标、功能支持范围、模板优势域和各支持 SOC 的覆盖方式。
4. 每项新增信息的来源、推导位置、传递载体和 kernel 消费方式。
5. ABI、注册、构建、测试和维护影响，以及旧 L0 路径的迁移或删除计划。

默认评审方向是只保留一套 L0 路径。公开接口的 V2 兼容策略不能作为平台性能优化时复制 L0 的默认理由；确实无法统一时，必须先提供可验证的技术证据和后续收敛计划。

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

完整的依赖模型、head window、内存数据流、流水同步、Tiling、SOC 差异和调优顺序见 [`../optimization/README.md`](../optimization/README.md)。本文件只保留开发方法论，不复制具体优化机制。

## 交付闭环

交付前先按 [`coding-standard.md`](coding-standard.md) 检查代码结构，再按 [`checklist.md`](checklist.md) 逐项核对；性能优化同时执行 [`../optimization/checklist.md`](../optimization/checklist.md)。尤其注意：

- 接口契约、代码拦截、报错文本、返回码和文档约束必须一致。
- `op_host`、`op_api`、kernel、PyTorch schema、Python 导出、测试和示例要同步。
- 修改公共模块时，要列出受影响算子并扩大验证范围。
- 新增 bugfix 应有能稳定触发的回归用例。
- 性能优化 PR 要分别给出性能锚点和泛化矩阵；不能只用锚点 shape 证明模板可交付。
- 涉及平台优化时，要检查 A2/A3/A5 的 L2 调用是否仍落到同一 L0，并确认没有遗留平台专属或未融合 L0 路径。
- 对 L0 每个输入和属性记录其语义与信息来源，确认 shape、平台和模板选择没有被重复编码进原型。
- PR 只写公开测试项和结果，不暴露本地机器、账号、绝对路径、临时目录、日志路径或内部环境。
