# ChunkGdnCoreFwd A5 性能优化因果决策

## 知识来源

- 来源：本地固定的 AscendC 因果决策卡快照。
- 索引 SHA256：`8A39C978073539C5648B2CCDAC1131F82871AC567081EBA445CE91C3BC0DC426`
- 目标范围：A5 / dav-3510 / Ascend950 / CANN 9.1。
- 目标 CANN：由执行旁车固定的 CANN 9.1 安装；`kernel_event.h`
  SHA256 为
  `3b2fc26123e9fcaa011f77f1db67f2fef16909c6051f7930093845111d81aa52`。

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
- Result：WU 机制以 `dbf1c4b2` 集成，并在 `bf499b04` 修复 ring 槽复用 P1 与 A resident 未命中 P2；FwdO 机制以 `11bb0b4b` 集成，并在 `0a1d7ee6` 修复 chunk128 qkmask 下溢越界 P1。两项修复均经独立复核通过（0 critical / 0 warning）。尚未完成 Ascend950 编译和真机验证，不能放行。
- Invalidation：若参考实现与融合私有副本已证明结构和同步完全等价，可缩小人工适配范围。

## CD-002：同步由数据版本和物理槽位推导

- Phase/status：设计 / accepted
- Affected files：WU/FwdO arch35 kernel、epilogue、block mmad 与 phase6 编排。
- Signal/question：L1 resident、ring/double buffer、三段 Cube overlap 和 AIC/AIV flag 会并存。
- Mechanism：共享槽位复用 -> 存在 RAW/WAR/WAW -> 缺边导致随机错误或 hang，过宽边导致流水串行化。
- Alternatives：保留全局屏障；直接复刻参考事件；按版本/槽位重新分配最小事件。
- Choice：先记录逻辑版本、物理槽、生产者/消费者和 init/drain，再决定事件；不得仅按参考 EventID 文本复制。
- Validation：静态事件表、目标 CANN 头文件核对、奇偶轮次/尾块/多序列压力重复测试、保守屏障 A/B。
- Result：WU 改为显式 8-slot 状态机：flag3/4 为 AIV→AIC ready，flag5 为 AIC→两个 AIV 的 slot-free；task>=8 覆盖前等待，AIC 通过 `MTE2 -> MTE1 -> PIPE_MTE1 free` 链在最后消费后释放，尾部 drain 清空 credit。FwdO 的 Cube1/2 使用本地事件 0..3，Cube3 使用 4..7 和独立 `[192,384) KiB` L1 区域，scheduler 跨核 0..7 原 init/drain 保留；qkmask 两个 VF 已从 `max(gbrcStart,64)` 开始第二段循环。目标 CANN 实现、重复执行和边界 case 仍待 A5 证据。
- Invalidation：目标实现或编译产物证明相关访问同步执行或物理地址不重叠。

## CD-003：平台和 shape 差异留在统一 L0 内部

- Phase/status：设计 / accepted
- Affected files：tiling、tiling key、arch35 私有实现。
- Signal/question：优化只面向 A5，但功能范围还包括 V256、fixed-length 和多序列 varlen。
- Mechanism：启动前已知的架构/shape 差异 -> tiling/template 可消除热点分支；过度特化会扩大组合和回归矩阵。
- Alternatives：新建第二套 L0；统一通用 kernel；统一 L0 + 少量 A5 策略特化。
- Choice：保持同一 L0 与调用路径，仅在现有 tiling/template/arch35 层表达实质差异。
- Validation：检查 key 域完备互斥；目标模型性能门禁；其他支持 case 功能与精度门禁。
- Result：两项移植都限定在现有融合私有模块及 A5 arch35 分支；A2/非 arch35 路径、统一 L0、V128/V256 tiling key 和现有调用入口保持不变。已通过 12 项本地 ctypes ABI 测试和聚焦 diff 检查；编译/功能/性能待 A5。
- Invalidation：二进制证明特化不改变生成代码或稳定性能，届时合并模板。

## CD-004：用单一产物的编译期变体定位过宽同步

- Phase/status：实验 / in progress
- Affected files：Phase 6 host tiling、顶层 kernel、FwdH arch35 kernel 与 SolveTri64。
- Signal/question：FwdH 两处 bounded-MMAD 结束后和 SolveTri64 的
  `FixpipeL0cToL1` 之后使用 `PIPE_ALL`，需判断它们是否为当前 A5
  模型 shape 的可见性能瓶颈，以及是否能收窄而不破坏正确性。
- Mechanism：
  - FwdH 候选假设：`finalWaitFlags()` 已闭合当前 bounded MMAD
    生命周期，后续 `CrossCoreSetFlag<0x2, PIPE_FIX>` 承担向 AIV 发布结果的
    跨核边，因此两处 `PIPE_ALL` 可能是重复全流水排空。
  - SolveTri64 已确认的实际链路为
    `FIX -> GM workspace -> MTE2 -> l1_Y -> MTE1`。helper 内部的
    `FIX_MTE2` 不能替代 helper 返回后的 `MTE2_MTE1`；第一次 MMAD
    读 `l1_I/l1_X` 与 `l1_Y` 无关，第二次 MMAD 才首次读 `l1_Y`。
- Alternatives：直接删除所有 `PIPE_ALL`；用 `PIPE_FIX` 保留生产者发布语义；
  每个方案单独编译；一次构建多个 tiling key 变体。
- Choice：首轮 B0–B3 实验后先收窄为一次构建、六个编译期变体：
  B0 保留两处 FwdH `PIPE_ALL` 和 SolveTri64 `PIPE_ALL`；B1 仅将 FwdH C1
  收窄为 `PIPE_FIX`；B2 仅将 FwdH C2 收窄为 `PIPE_FIX`；B3 仅将
  SolveTri64 收窄为立即 `MTE2_MTE1`；B4 组合 C1/C2 `PIPE_FIX`、Solve
  保持 `PIPE_ALL`；B5 在 B4 上再将 SolveTri64 收窄为立即
  `MTE2_MTE1`。因此 B4→B5 可直接归因最终 FwdH 配置下的 Solve 增量。
  第二轮只扩展尚未隔离的 C1 发布与 Update 写回边，B6–B11 真值表如下；
  其中“event-only”只移除显式 `PipeBarrier`，保留紧随其后的定向事件或
  `CrossCoreSetFlag`，不是删除依赖协议。C2 和 Solve 在新增变体中均保持 B0，
  已有 B0–B5 语义不变。

  | 变体/key | C1 发布 | C2 发布 | SolveTri64 | Update FP32 final-state 写回 |
  |---|---|---|---|---|
  | B0/1 | `PIPE_ALL` | `PIPE_ALL` | `PIPE_ALL` | `PIPE_ALL` |
  | B1/11 | `PIPE_FIX` | `PIPE_ALL` | `PIPE_ALL` | `PIPE_ALL` |
  | B2/21 | `PIPE_ALL` | `PIPE_FIX` | `PIPE_ALL` | `PIPE_ALL` |
  | B3/31 | `PIPE_ALL` | `PIPE_ALL` | 立即 `MTE2_MTE1` | `PIPE_ALL` |
  | B4/41 | `PIPE_FIX` | `PIPE_FIX` | `PIPE_ALL` | `PIPE_ALL` |
  | B5/51 | `PIPE_FIX` | `PIPE_FIX` | 立即 `MTE2_MTE1` | `PIPE_ALL` |
  | B6/61 | `PIPE_ALL` | `PIPE_ALL` | `PIPE_ALL` | `PIPE_MTE3` |
  | B7/71 | `PIPE_ALL` | `PIPE_ALL` | `PIPE_ALL` | event-only，保留 `MTE3_MTE2`/`MTE3_V` |
  | B8/81 | event-only，保留 `CrossCoreSetFlag<0x2, PIPE_FIX>` | `PIPE_ALL` | `PIPE_ALL` | `PIPE_ALL` |
  | B9/91 | `PIPE_FIX` | `PIPE_ALL` | `PIPE_ALL` | `PIPE_MTE3` |
  | B10/101 | event-only，保留 `CrossCoreSetFlag<0x2, PIPE_FIX>` | `PIPE_ALL` | `PIPE_ALL` | `PIPE_MTE3` |
  | B11/111 | event-only，保留 `CrossCoreSetFlag<0x2, PIPE_FIX>` | `PIPE_ALL` | `PIPE_ALL` | event-only，保留 `MTE3_MTE2`/`MTE3_V` |

  第三轮以 B7 的 Update event-only 为共同基线，只隔离 H 的两条同步假设：

  | 变体/key | C1 发布 | WU→H 入口 | H initial_state 发布→C1 |
  |---|---|---|---|
  | B12/121 | `PIPE_ALL` | `SyncAll<false>()` | 跳过双侧 collective；保留每核 MTE3 drain、`vec2Done[0/1]` seed 与全部 C1 wait |
  | B13/131 | `PIPE_FIX` | `SyncAll<false>()` | 同 B12 |
  | B14/141 | `PIPE_ALL` | 所有参与者无条件本核 `PIPE_ALL` drain | 保留双侧 `SyncAll<false>()` |
  | B15/151 | `PIPE_FIX` | 同 B14 | 保留双侧 `SyncAll<false>()` |

  H-init bypass 依赖 AIC 与两个 AIV subblock 的物理 core/stream ownership
  一致；入口 local drain 则独立检验 WU→H 是否只需本核流水退休。两者不得组合，
  kernel 以 `static_assert` 固化该限制。B12–B15 只允许精确主 varlen shape：
  Ascend950、BF16、FP32 initial_state、B1/Hk16/Hv32/T11274/K128/V128/C64、
  `cu_seqlens` 长度 2、177 chunks、存在 initial_state 且
  `output_final_state=true`。output mask 不参与选择；同一请求在既有实验域的其他
  shape 回退 B7，实验域外回退 B0。

  第四轮继续以 B15 为共同基线，把入口本核全流水 drain 与高频 C2 发布边拆成
  两个原子变量，并增加一个组合变体验证可加性：

  | 变体/key | C1 发布 | C2 发布 | WU→H 入口 | H initial_state 发布→C1 |
  |---|---|---|---|---|
  | B16/161 | `PIPE_FIX` | `PIPE_ALL` | AIC `PIPE_FIX`、AIV `PIPE_MTE3` | 保留双侧 `SyncAll<false>()` |
  | B17/171 | `PIPE_FIX` | `PIPE_FIX` | 所有参与者本核 `PIPE_ALL` | 保留双侧 `SyncAll<false>()` |
  | B18/181 | `PIPE_FIX` | `PIPE_FIX` | AIC `PIPE_FIX`、AIV `PIPE_MTE3` | 保留双侧 `SyncAll<false>()` |

  B16/B18 的依赖闭环为：WU 的 AIC 最终 U/W 由 FIX 写 GM，AIV 的 ring
  workspace 由 MTE3 写并以同一 pipe 发布 ready；WU 尾部已经消费全部
  slot-free credit，随后保留的 H-init collective 继续负责参与者会合。因此只缩小
  本核 drain，不删除跨角色同步。`static_assert` 同时禁止 role-specific drain 与
  H-init bypass、以及两种入口 drain 策略并存。B17 只收窄 C2 的 FIX producer
  发布边，B18 用于检查两项变化是否可加。

  B16–B18 的生产 selector 仍只允许上述精确主 varlen shape。为让最短真实调用
  覆盖新 key，内部环境开关 `FLA_NPU_GDN_SYNC_T1_DIAGNOSTIC=1` 仅对 B16–B18
  放行严格诊断 shape：Ascend950、BF16、FP32 initial_state、
  B1/Hk16/Hv32/T1/K128/V128/C64、`cu_seqlens` 长度 2、1 个 chunk、
  `output_final_state=true`；output mask 仍不参与选择。未设置开关或任一 shape
  条件不符时不拓宽 key 域：未设置/设为 0 或 shape 不符时，既有实验域
  回退 B7、实验域外回退 B0；开关值非法则直接 fail-closed。

  已被 A5 硬件精度结果否定的 Solve deferred-wait 不重试；MIX kernel 的
  `SyncAll` 保护跨阶段 GM/workspace 可见性与参与者会合；B0–B11 全部保留，
  B12–B15 每次只隔离上表中的一条边。也不编译丢弃配套事件的裸删方案。
  B1–B18 的额外 key 仅在 A5、BF16 输入、
  FP32 initial_state 的编译配置中生成，并仅为 V128、chunk64
  模型主路径选择；
  其他合法 dtype/state/V256 形态回退 B0。host 在非 Ascend950 上
  fail-closed；不扩展公共 ABI/trailer。
- Validation：先冻结两轮 inference/full 均成功的 stable200；然后在同一
  CANN 9.1 产物中检查 key 集、实际路由、多设备首次/重复精度与确定性；
  只对通过的单项变体和 B4/B5 运行 stable200，最后用独占空闲卡做
  单项变体与 B0、B4→B5、winner 与交付基线的对称 ABBA，
  报告 median、p95 和峰值显存。
- Falsifiers：任一变体编译失败、路由不符、超时/hang、重复运行不确定、
  stable200 精度劣化，或独占卡 ABBA 无稳定收益，均否定对应候选机制。
- Result：旧 B0–B3 代码已以收敛配置完成 CANN 9.1 构建、部署和 key
  矩阵验收。B3 延后 wait 在 3/3 首调中造成大面积 O/final_state 错误，
  已否定。对模型推理 shape 的精确首调诊断显示：差异总是 final_state
  最后一个 value head 的 `[K=0:128,V=112:128]`，即 2048 个元素。
  device 6 上 B0 Phase6 5/5 产生差异、B1 Phase6 5/5 参考一致；
  device 5 上 B0/B1 Phase6 各 5/5 参考一致，但 legacy 路径 10 次中
  3 次产生同一差异。哨兵实验证明该分块被写覆盖，排除公共输出漏写。
  这些证据否定“比较失败可直接归因于当前 selector”；B3 延后 wait
  已被硬件结果否定。旧 B1 的两处裸删 Phase6 在两卡精确诊断中均得到
  同一参考一致哈希，因此保留为性能上界/正向观察；但由于它同时改变两条边且
  未保留 FIX 发布语义，未经进一步因果证明不作为交付候选。当前已切换到
  上述可独立归因的 B0–B11 矩阵。B6–B11 已通过源码真值表、selector、
  key 路由、实验域 fallback、非 A5 fail-closed 和 ABI-neutral 静态测试；
  B12–B15 已完成本地源码真值表、精确 selector、key 路由、分层 fallback、
  非 A5 fail-closed 和 ABI-neutral 静态门禁；均尚待同一产物构建及 A5 路由、
  精度、确定性和性能验证。B16–B18 已完成本地实现与静态门禁设计，新增 key
  仅在 A5/BF16/FP32 initial_state 编译域内出现；编译、T1 真命中、主 shape
  精度、确定性和性能仍待同一 A5 产物验证。
- Invalidation：若目标 CANN 头文件、生成代码或 profiling 证明上述生产者/消费者
  链路不成立，需回到 B0 并重建依赖图，不继续放宽同步。
