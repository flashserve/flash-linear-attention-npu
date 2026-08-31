# ChunkGdnCoreFwd A5 性能优化因果决策

## 知识来源

- 来源：`D:/workspace/部门AI全流程专项/.codex/knowledge/causal-decisions/`
- 索引 SHA256：`8A39C978073539C5648B2CCDAC1131F82871AC567081EBA445CE91C3BC0DC426`
- 目标范围：A5 / dav-3510 / Ascend950；精确 CANN 版本待真机登记。

## 检索账本

| Card | 固定摘要 | 命中理由 | 范围/证据 | 下钻 | 当前含义 |
|---|---|---|---|---|---|
| `cann-version-architecture-evidence.md` | AscendC API、内部实现、保留资源和同步语义可能随架构、Kernel 类型、CANN 版本及编译选项变化；实现放行必须优先采用目标机器证据。 | 从独立 A5 单算子移植 RegBase、EventID 和流水机制到融合 MIX kernel。 | A5/dav-3510，E1 | 已读全文 | 参考代码只能提出候选；编译、同步和放行绑定目标 A5 安装包。 |
| `compile-time-template-specialization.md` | 启动前可确定且会改变热点路径、资源布局、同步或算法的差异，可用 Tiling Key 和模板特化隔离；但应防止模板组合爆炸。 | A5 路径、V128/V256、fixed/varlen 和 chunk pipeline 会改变布局与热点路径。 | AscendC 模板，E1 | 已读全文 | 优先复用统一 L0 和公共骨架，只特化能改变生成代码的维度。 |
| `regbase-local-memory-synchronization.md` | VF local scratch、Kernel 外 PIPE/Event 与跨核同步保护不同层级，可以组合但不能互相替代；当前仍需 A5 目标证据。 | `compute_w_u` 和 FwdO 的 A5 优化包含 RegBase/VF 及外部 AIC/AIV 协作。 | A5 RegBase，E0 | 已读全文 | 不因引入 RegBase 删除外层同步；分别验证 local、pipe/event 和 cross-core 边。 |
| `synchronization-from-data-dependencies.md` | 从数据版本和物理槽位上的 RAW、WAR、WAW 推导先行发生关系，再选择最小足够同步；核心是不串线、生命周期闭合和参与者一致。 | 两条参考优化复用 L1/workspace、重排 Cube/Vector 并占用事件槽。 | A2 E2；A5 待验证 | 已读全文 | 移植前先建立版本/槽位/参与者表；逐分支检查 init、drain 和事件配平。 |

## CD-001：A5 参考机制必须在融合核内重新验证

- Phase/status：设计 / accepted
- Affected files：`op_kernel/internal/operators/**`
- Signal/question：参考实现来自独立单算子，而目标是带私有任务映射和跨阶段同步的融合 MIX kernel。
- Mechanism：上下文和参与者变化 -> 资源/同步边变化 -> 文件覆盖可能破坏既有 varlen 与流水协议 -> 需要机制级移植。
- Alternatives：整目录替换，改动快但无法证明融合语义；机制级移植，成本较高但能维持统一 L0。
- Choice：只抽取最新 A5 优化机制，保留融合核的任务映射、varlen handoff、chunk pipeline 与架构分支。
- Validation：独立 WU/O A/B、fixed/varlen/V128/V256 精度、重复执行确定性以及目标模型 profiling。
- Result：pending
- Invalidation：若参考实现与融合私有副本已证明结构和同步完全等价，可缩小人工适配范围。

## CD-002：同步由数据版本和物理槽位推导

- Phase/status：设计 / accepted
- Affected files：WU/FwdO arch35 kernel、epilogue、block mmad 与 phase6 编排。
- Signal/question：L1 resident、ring/double buffer、三段 Cube overlap 和 AIC/AIV flag 会并存。
- Mechanism：共享槽位复用 -> 存在 RAW/WAR/WAW -> 缺边导致随机错误或 hang，过宽边导致流水串行化。
- Alternatives：保留全局屏障；直接复刻参考事件；按版本/槽位重新分配最小事件。
- Choice：先记录逻辑版本、物理槽、生产者/消费者和 init/drain，再决定事件；不得仅按参考 EventID 文本复制。
- Validation：静态事件表、目标 CANN 头文件核对、奇偶轮次/尾块/多序列压力重复测试、保守屏障 A/B。
- Result：pending
- Invalidation：目标实现或编译产物证明相关访问同步执行或物理地址不重叠。

## CD-003：平台和 shape 差异留在统一 L0 内部

- Phase/status：设计 / accepted
- Affected files：tiling、tiling key、arch35 私有实现。
- Signal/question：优化只面向 A5，但功能范围还包括 V256、fixed-length 和多序列 varlen。
- Mechanism：启动前已知的架构/shape 差异 -> tiling/template 可消除热点分支；过度特化会扩大组合和回归矩阵。
- Alternatives：新建第二套 L0；统一通用 kernel；统一 L0 + 少量 A5 策略特化。
- Choice：保持同一 L0 与调用路径，仅在现有 tiling/template/arch35 层表达实质差异。
- Validation：检查 key 域完备互斥；目标模型性能门禁；其他支持 case 功能与精度门禁。
- Result：pending
- Invalidation：二进制证明特化不改变生成代码或稳定性能，届时合并模板。

