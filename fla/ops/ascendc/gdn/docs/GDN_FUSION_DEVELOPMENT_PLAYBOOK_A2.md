# A2 GDN 融合开发手册

本文档沉淀 Phase 2/3 的小步开发和验收方法，不承载当前进度。后续每个 Phase 开始前必须先阅读本文档，并按本文档执行；如需偏离，先在主计划中记录理由。

进行中的 A2 工作统一从 `GDN_CURRENT_STATUS_A2.md` 续跑。为降低上下文压缩、任务切换或中断造成的状态回退，任何小步开始前必须先读取该文件；完成、失败或中断后必须立即刷新证据、结论和唯一的“下一小步”，再继续工作。聊天记录不能替代该状态文件。

## 1. 核心结论

本次最大的时间和 token 消耗不在 kernel 编码本身，而在：

- 阶段边界反复校准；
- 远程源码、构建拷贝和安装包版本不确定；
- 过早优化 workspace 和同步，导致长链路调试；
- PowerShell、SSH、Bash 和内联 Python 多层拼接；
- 未先区分旧基线缺陷与融合回归。

后续优先缩短反馈链路，而不是一开始扩大融合范围。

## 2. 阶段启动卡

每个 Phase 开始前必须冻结以下内容：

| 项目 | 必须回答的问题 |
| --- | --- |
| 基线 | 和哪个已验收 variant 比较？ |
| 融合边界 | 本阶段仅吸收哪些计算？ |
| 输入/输出 | shape、dtype、layout、有效区域是什么？ |
| 保留项 | 哪些 GM workspace、transpose 或外部扩头暂时保留？ |
| 不做项 | 哪些规格明确不在本阶段同时修改？ |
| 验收矩阵 | dense/varlen、dtype、chunk、尾块、GVA 覆盖哪些用例？ |
| 成功标准 | 精度、latency、kernel 数目标是什么？workspace 如何记录和优化？ |

启动卡未冻结时不修改 kernel。实施中如果想顺手修改 transpose、原生 GVA、`V=256` 或 workspace 复用，默认拒绝，除非先更新启动卡和主计划。

## 3. 最短反馈流水线

固定按以下顺序执行，任一层未通过就停止向下一层扩展：

```text
本地语法/ABI/静态检查
    -> 最小源码同步
    -> 本地 = 远程 = build/binary 源码 SHA256
    -> 完整验收包构建一次
    -> 安装后 ACLNN 符号检查
    -> 1 个最小 smoke case
    -> 独立融合阶段精度矩阵
    -> 端到端消融矩阵
    -> 性能/workspace
    -> profiler/kernel 数
    -> JSON 报告与主计划归档
```

不在 smoke case 失败时继续跑全量用例，不在精度未通过时跑性能，不在版本哈希未确认时解释运行结果。
性能层未通过时只在当前冻结 case 上执行下文规定的有限优化轮次，不直接铺开矩阵。

## 4. 远程构建与版本规则

1. 远程命令写入固定 Bash 脚本，不通过 PowerShell 拼接多层 heredoc 或内联 Python。
2. 每次运行前比较本地源码、远程源码和 `build/binary/.../src` 拷贝的 SHA256。
3. 三份哈希不一致时，删除该算子的源码拷贝/编译 stamp 或做 clean operator build，不相信未验证的增量构建。
4. 验收包必须同时包含基线算子、新融合算子和端到端回归依赖。
5. 不向共享 OPP 根目录安装只含单个算子的局部包；它会覆盖 `libcust_opapi.so` 并丢失基线符号。
6. 安装后先用 `nm -D` 检查所有必需 ACLNN/GetWorkspaceSize 符号，再启动 Python。
7. 每次远程任务都设置 timeout，结束后清点本项目的悬挂进程，不终止其他用户或项目进程。

### 4.1 每个 Phase 一个不可变 ACLNN

1. 新 Phase 必须新增版本化 `aclnnGdnCoreFwdPhaseN` 和对应 Python wrapper，不能把上一 Phase 的 ACLNN 原地改造成新阶段。
2. 通用 `aclnnGdnCoreFwd` 只承担默认入口兼容，可指向当前推荐 Phase；性能结论必须使用版本化入口名称。
3. 同一个完整 wheel 必须同时导出基线和所有待比较 Phase 的 `GetWorkspaceSize`/执行符号。
4. benchmark 必须把每个版本化 ACLNN 作为独立命名 variant；报告记录实际调用的 ACLNN 符号，不能只写“composite”。
5. Phase 代码、测试、接口文档和验收报告同时归档后才能开始下一 Phase。验收包 SHA256 和源码提交/归档点必须可追溯。
6. 已验收 Phase 只接受有证据的 bug/ABI/build 修复；任何融合边界、数据布局、workspace 策略或性能行为变化都创建新 Phase。

当前固定映射见 `GDN_PHASE_VERSION_ARCHIVE_A2.md`。

### 4.2 每个 Phase 一个不可改写 Git 里程碑

1. 专用归档分支只通过普通 commit 向前推进；禁止 amend、rebase 和 force-push 已归档历史。
2. Phase 验收结束后再创建里程碑 commit，commit 中同时包含版本化 ACLNN、测试、文档和验收结论。
3. 下一 Phase 从上一里程碑 commit 继续开发并新增 commit，不覆盖、删除或重命名旧 Phase 的固定入口。
4. 报告同时记录 commit SHA 和 wheel/OPP SHA256。Git commit 证明源码版本，产物哈希证明实际验收包，二者不能互相替代。
5. 临时远程脚本、构建产物和原始调测输出不提交；稳定命令进入正式脚本，结构化结果进入验收报告。
6. 规则建立前 Phase 1/2 共处一个未提交工作区，因此允许首个归档 commit 同时固化二者；Phase 3 起不再例外。

## 5. 正确性规则

### 5.1 三级精度判定

1. 旧基线输出全部有限时，比较有效区域 bit-exact，并记录 max-abs。
2. 旧基线含 NaN/Inf 时，禁止将其作为 golden；改用 CPU FP64 或已验证的高精度参考。
3. 所有用例都要独立检查融合输出的 NaN/Inf，不能仅依赖 tolerance 比较。

只比较数学有效区域，但如果后续算子会读取 padding，则必须同时验证 padding 的定义和初始化。

### 5.2 正确性先于 workspace 优化

首版允许保留安全的 GM hand-off workspace。只有在以下两种设计之一被证明正确时，才可删除或原地复用：

- 所有 producer 和 consumer 之间有真正的全局阶段 barrier；
- tile-local producer/consumer 流水能证明每个别名区域的唯一所有权和时序。

dense 通过不能证明无竞态，必须用 varlen、尾块和多 core 并行场景回归。

## 6. A2 MIX Kernel 同步红线

- 优先复用仓内相同架构和 MIX 模式的已验证代码，不靠运行试错推测同步语义。
- 在 `MIX_AIC_1_2` 中，成对 AIV sub-block 分别执行 `CrossCoreSetFlag<0x2>` 后，配对 AIC 只等待一次；等待两次会死锁。
- core-group 内 ready flag 不等于全局 barrier，不能用它证明不同 core group 之间的 GM 原地覆写安全。
- 新同步方案先写最小同步实验，再接入完整数学计算。

## 7. 性能与验收规则

- 精度通过后才测性能。
- 同一输入、同一安装包、同一 device 上交替测量基线和融合 variant。
- 正式性能验收必须关闭 `ASCEND_LAUNCH_BLOCKING`；如果为诊断开启，结果只能标记为回归筛查。
- 每个报告必须记录 SoC/device、CANN、包哈希、环境变量、warmup/iteration、variant 顺序和计时方法。
- 固定 warmup 和 iteration，至少记录 median、P90 和 min；不用单次 latency 下结论。
- 报告必须列出每个实际运行的 case 及排除的失败/过期 case，同时单列未覆盖的规格矩阵，不得用代表 case 推导“全场景”。
- 同时记录 ACLNN 调用数、NPU kernel 数、workspace max/sum 和 peak allocated delta。
- 减少 ACLNN 数不等于真正融合；必须由 profiler 证明 NPU kernel 或 GM 往返确实减少。
- 一个阶段只有在精度、功能覆盖、性能、workspace 和 profiler 证据均归档后才能关闭。
- workspace 的绝对 `<=50 MB` 是优化目标和报告项，不是当前融合路线的硬门槛；优先保证性能，且应继续寻找不伤害性能的 workspace 降低方案。
- 性能不达标指冻结用例相对上一已验收 Phase 的稳定基线出现超出测量噪声的回退，或融合没有获得启动卡预期的收益；不预先用一个统一百分比替代 AB/BA、median/P90 和 profiler 证据。
- 性能不达标后，默认在同一路线上最多进行三轮优化。每轮必须有明确瓶颈假设，只改变一个主要变量，并重跑同一冻结 case、更新 profiler；单纯复测、增加 iteration 或确认噪声不计为一轮。
- 三轮后仍不达标，整理基线差距、受影响身份、profiler 瓶颈、三轮改动及结果和下一方案成本，反馈人工决策；不得自行增加按 dtype、chunk、layout 或 shape 的生产分支。

## 8. 控制时间和 Token

- 既有文档和测试先用 `rg` 定位，只读相关区段，避免重复遍历大仓库。
- 远程脚本、用例矩阵和 JSON 报告模板复用，不每轮重写。
- 长时间编译只启动一次；构建期间做本地检查，不并发覆盖同一远程安装环境。
- 每次实验只改一个变量，记录源码哈希、命令、结果和结论；不重复无法区分版本的试验。
- 优先输出结构化摘要，不将整份编译日志反复载入上下文。
- 发现新规则时更新本手册；单次错误细节记录在 `.learnings/ERRORS.md`，不让错误日志取代可执行流程。

## 9. Phase 3 历史启动清单（已执行完成）

Phase 3 吸收 `local_cumsum` 时固定以下边界：

- 已验收 `ChunkKktSolveTri` 是不可变基线。
- 先独立验证现有探索性 `ChunkCumsumKkt`，再决定如何接入 solve_tri。
- 首版允许保留 `g_cumsum` 或 KKT 私有 workspace，先证明 FP32 累加和 varlen 序列边界正确。
- 先做 `local_cumsum + KKT` 局部 A/B，通过后再与 Phase 2 串联。
- 不同时修改 transpose、原生 GVA、`K != V` 或 causal conv/RMSNorm。
- 如果没有性能收益，保留 Phase 2 路径，不为了减少算子数强行融合。

该清单后续已执行完成。早期 `(local_cumsum + KKT) + solve_tri` 拆分 core 候选因丢失
Phase 2 的 `KKT + solve_tri` 融合收益被淘汰，最终 Phase 3 为累积 `ChunkCumsumKktSolveTri`。

## 10. 后续融合边界选择规则

Phase 4 起不默认按前缀顺序机械吸收下一个算子。启动前可用静态分析和已有 profiler 比较相邻候选边界，但只选择其中一条进入实现，不并行铺设多条生产路线：

- 中间张量是否为公开输出；公开输出仍需写回，不能把理论上的全量 GM 消除计入收益；
- producer/consumer 的 task ownership、chunk/head 遍历顺序和 MIX 同步是否兼容；
- 能消除的是单次读取、完整写回+读取，还是仅 ACLNN/launch；
- 合并后是否仍需相同或更大的系统/user workspace；
- 候选能否沿单一生产路线继续演进，而不引入长期规格路由。

优先验证非公开大中间量的 producer/consumer 融合。允许按 Phase 线性形成两个已验收的融合组，再根据 profiler
决定是否合成单 kernel；不能把“累计前缀更长”或“kernel 数更少”本身当作性能结论。一个局部融合
只有同时通过独立 A/B 和完整 core A/B 才能成为正式 Phase。版本化 ACLNN 和旧 Phase kernel 仅用于
不可变对照与归档，不得被解释为默认入口需要按输入身份维护多条运行时分支。

`V=256`、原生 GVA、transpose/layout 和 workspace 别名均属于独立变量。若路线要求在两个融合
Phase 之间关闭规格缺口，必须分别建立启动卡和版本化 checkpoint，不得隐藏在下一融合 Phase 中。
