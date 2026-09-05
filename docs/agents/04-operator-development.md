# 算子开发

本阶段依据已确认接口、CPU 标杆和评审通过的设计实现 Ascend C 算子。开发阶段负责把设计写成代码，接口语义、支持范围和测试标准沿用前面阶段的结论。

## 阶段输入

- [`01-interface-confirmation.md`](01-interface-confirmation.md) 输出的接口契约。
- [`02-reference-generation.md`](02-reference-generation.md) 输出的 CPU 标杆和对齐用例。
- [`03-solution-design.md`](03-solution-design.md) 输出并归档在算子目录的、记录当前规则版本且整体评审通过的完整设计文档。
- 与当前 chunk 依赖类型对应的开发参考：[`chunk-dependent-development.md`](reference/04-operator-development/chunk-dependent-development.md) 或 [`chunk-independent-development.md`](reference/04-operator-development/chunk-independent-development.md)。
- 仓库中相似算子的实现、代码生成输入、构建入口和代码规范。
- 接口或功能修改任务还需要已确认的差异清单、兼容策略和既有场景回归范围。
- 优化任务还需要经评审的瓶颈结论、优化前性能数据，以及优化前后必须保持一致的测试条件和统计方式。

上述输入缺失或互相冲突时，返回产生该输入的阶段修正；完整设计文档通过整体评审后开始编码，代码要覆盖已确认的全部支持范围和语义。

对于新算子，本阶段是首次查看仓内其他算子的代码、README 和设计文档。可以参考目录组织、注册、构建、API 封装、平台适配和编码方式；Stage、公式依赖、数据存放位置、资源规划、同步协议和支持范围按照已评审设计实现。参考代码显示设计缺少必要内容时，返回 `03` 补充设计并重新评审。

## 参考实现选择

1. 根据已评审设计判断 chunk 间是否存在顺序依赖：后一个 chunk 需要前一个 chunk 的状态或中间结果时，完整读取 [`chunk-dependent-development.md`](reference/04-operator-development/chunk-dependent-development.md)；各 chunk 的输入在启动前已经完备且可以独立完成时，完整读取 [`chunk-independent-development.md`](reference/04-operator-development/chunk-independent-development.md)。混合场景以有依赖参考组织整体调度，其中独立 Stage 可以采用无依赖参考的任务划分方式。
2. 参考文档中的调度、流水、同步、workspace 和工程接入方法用于实现已评审设计；当前算子的公式、Stage、shape、资源数量和支持范围仍以自己的 `docs/design.md` 为准。
3. 参考文档已覆盖开发所需细节时直接据此实现；需要确认具体类、API 或代码组织时，按文档中的“定点查看当前实现”只读取相关文件和方法。
4. 在当前算子的 `docs/design.md` 中记录开发参考角色、版本、来源 commit、采用内容和本算子的差异。参考文档版本更新时，既有算子评估版本变化说明并记录是否采用。

## 实现顺序

建议按依赖关系实施：

1. 更新算子定义、InferShape、参数校验和错误语义。
2. 实现 host tiling、tiling data、任务描述、workspace 计算和 TilingKey 选择。
3. 实现 L2/L0、stage 调度、kernel 模板、搬运与同步。
4. 同步 op_api/aclnn、schema、ctypes、Python wrapper 和公开导出。
5. 接入 CPU 标杆，运行最小合法精度用例并通过比较，确认当前实现可以进入完整测试。

修改生成文件前先查明 YAML、生成器或模板来源；需要修改生成输入时，更新源文件并重新生成结果。op_host、kernel、op_api、Python 和测试层中的参数名称、顺序、类型及默认值必须与接口契约一致。

## 接口或功能修改实施

- 只实现 `01` 和 `03` 已确认的特性差异，保持未受影响的接口、语义、支持场景和 CPU 标杆结果不变。
- 新增或修改参数、校验、tiling、kernel、op_api、schema、wrapper 和测试时，按设计同步所有受影响层级。
- 修复已支持场景时，以现有接口契约和 CPU 标杆为准，修正实现以匹配既定预期结果。
- 实施中发现实际修改超出已确认差异时，按变化内容返回相应阶段：接口或支持范围变化返回 `01`，CPU 标杆变化返回 `02`；Stage、数据流、地址、同步或资源规划变化返回 `03` 更新完整设计文档并重新评审。

## 优化任务实施

- 只实施实验达到目标且用户确认采用的优化方案，例如 tiling、任务划分、stage、workspace、同步、搬运、模板和流水修改。
- 修改算子代码时，同步更新设计文档中的实际实现、资源规划和实验结果，使文档与代码保持一致。
- 保持公开接口和 ABI、数学语义、输入输出与属性、支持范围、CPU 标杆和精度标准不变。
- 每轮只引入一项能够独立验证的优化，并保留相同条件下的优化前数据，以确认性能变化由哪项修改产生。
- 实现中发现瓶颈判断、Stage、资源预算或同步设计不成立时，先返回 `03-solution-design.md` 更新完整设计文档、证据和方案，再继续编码。
- 优化采用已评审的通用实现，既要达到目标场景的性能收益，也要保持既定支持范围、精度阈值和 CPU 标杆不变。
- 如需改变公开参数、返回值、默认值、dtype/layout 语义、数学语义或支持范围，立即停止优化任务，并转入接口变更或新功能流程。

## 复用与平台差异

从相邻 Ascend C 算子复用：

- host tiling、tiling data 和有限模板选择模式。
- AIC/AIV 分工、Catlass GEMM、blocked solve 和状态传播结构。
- workspace 生命周期、cross-core flag、pipe event 和写回协议。
- fixed/varlen、head ratio、tail chunk 和多 SoC 的工程处理方式。

具体算子的 tile、窗口、slot 数、同步计数和资源常量均从本算子设计推导。同一 L0 定义和调用路径服务所有支持 SoC；平台差异放入 tiling、workspace、kernel 模板或架构 trait。

## host tiling 与模板实现

- host 侧完整校验 shape、dtype、layout、属性、`cu_seqlens`、`chunk_indices` 和尾块约束。
- host 侧生成 kernel 所需的任务描述、offset、有效长度、workspace 和模板字段，使 kernel 的性能关键路径可以直接使用这些已经计算好的信息。
- kernel 入口只选择设计中列出的模板实例，模板内部使用 `if constexpr` 在编译期移除当前实例不需要的分支。
- TilingKey 与选择条件必须一一对应；支持范围之外的组合由 host 侧明确拦截。
- SoC、模板、workspace 和可推导信息保留在内部 tiling 与实现中；公开接口和 L0 参数只保留无法从这些信息推导的必要输入。

## kernel、搬运和同步实现

- 矩阵计算主路径使用适合的 Cube/Catlass 实现，Scalar/Vector 负责各自适用的非矩阵计算。
- 性能关键路径优先连续整行或整 tile 搬运，并采用批量访问 API。
- tail 和 partial chunk 优先使用 padding、中性值和有效区 mask 保持批量路径。
- double buffer、MTE、VEC/CUBE、MTE3 和 cross-core 同步按照设计实现；每个生产或通知事件都有对应的等待、消费和复用步骤。
- workspace slot 覆盖前确认消费者已释放；空任务和 varlen 无效区仍要遵守 ready/free 计数协议。
- `PipeBarrier<PIPE_V>()` 负责对应 pipe 内依赖，跨 pipe 依赖使用对应事件同步。

实现过程中发现原设计需要调整才能满足容量、同步或性能要求时，先在算子目录的 `docs/design.md` 中同步更新 Stage、逐 Stage 详设、全局资源分配和 `R01`–`R19` 检查表，完成整体评审后再修改代码。代码始终与最新评审结论一致。

## 开发期精度检查

开发期先使用 CPU 标杆完成最小合法用例和关键 Stage 对比。出现精度不一致时，按
[`05-operator-testing.md`](05-operator-testing.md) 中的 CT 流程定位问题，在本阶段修复算子实现；
修复后重新通过最小精度验证，再进入完整测试。

开发期保持 CPU 标杆、精度阈值、测试用例和已确认支持范围不变，不通过放宽阈值或屏蔽用例来
掩盖实现问题。

## 开发期性能定位

性能分析必须基于 profiling，并与设计目标对照：

- 主要受 Scalar 限制：检查热路径标量访问和逐元素循环。
- 主要受 MTE 限制：检查搬运是否过碎、重复、非连续或粒度过小。
- 主要受 VEC 限制：检查是否承担应由 Cube 完成的矩阵工作，或 repeat 粒度过小。
- 主要受 CUBE 限制：检查 tile、数据复用、矩阵形状和有效计算占比。
- 主要受 AIC/AIV 等待限制：检查 producer-consumer 队列、flag、MTE3 写回、double buffer 和流水距离。

目标 shape 达到性能要求后，还要验证设计中其他支持的 shape、dtype、layout 和 SoC。

## 阶段输出

进入完整测试前确认：

- 实现与已确认接口、CPU 标杆和设计一致。
- 实现中的每个 Stage 类型、公式、依赖、数据存放位置、L1/UB/GM 规划和 head 处理与已评审的完整设计文档一致。
- op_host、InferShape、tiling、kernel、op_api、schema、Python 导出同步。
- 所有设计中的模板、TilingKey、workspace 和同步路径均有对应代码。
- 最小合法用例和关键 Stage 已能与 CPU 标杆比较并通过。
- 实现中发现的新限制、风险或设计变化已更新到对应的 `01`、`02` 或 `03` 阶段文档。
- 接口或功能修改的每项代码变化都能对应已确认差异，未受影响场景保留兼容路径。
- 优化任务的实现改动能够逐项对应瓶颈证据和已评审方案。

完成开发期自检后，按 [`05-operator-testing.md`](05-operator-testing.md) 执行正式验证。
