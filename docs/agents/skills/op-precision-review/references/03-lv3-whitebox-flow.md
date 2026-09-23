# lv3 白盒检视流程

lv3 的根因（同步/事件时序、tiling/分核错误、累加顺序/舍入）与算子实现细节强相关，因此不靠"看一眼代码"
判断，而是先用固定流程把方案还原出来，再做三项分析。缺任何一步都会退化成猜测。

## 1. 五步流程

### ① 读材料：接口文档 + AscendC 文档 + 硬件手册

1. 逐个读算子接口文档的每个参数说明与"约束"类条目：支持范围、shape/dtype/layout 限制、可选输入语义、
   返回码、预留参数。
2. 读 AscendC API 文档中本算子实际用到的 API 契约（声明、模板参数、入参单位、内存位置、对齐、异步语义）。
3. 读硬件手册中本算子涉及的片上资源规格（各层容量、对齐、指令吞吐与限制、跨核同步能力）。

输出：一句话的算子契约摘要 + 与本算子相关的 API/资源清单。材料缺失时记录缺什么，并说明因此无法判定的范围。

### ② PR 检视：从 PR 描述提取关键功能

1. 读 PR 详细描述与 diff 摘要，列出本次改动想实现的关键功能（新增分支、改变分核、改变内存布局、
   引入新的同步、改变累加方式等）。
2. 用 ① 的契约摘要过滤：改动是否触碰接口契约、支持范围、dtype/format、平台差异。
3. 输出"关键改动清单"，作为后续只看关键模板的依据。

整体白盒检视（非 PR）跳过本步，直接进入 ③。无 PR 描述时（描述缺失或只有标题），记录该缺失，
按 diff 自行归纳关键改动并标注推断依据。

### ③ 读算子详设

存在 `docs/design.md`、设计评审记录或开发期验证记录时：

1. 读 Stage 划分、每个 Stage 的执行单元、输入落点、输出落点、同步条件、状态更新点。
2. 读分核方案（每个 core 处理哪些 chunk/head/task）与内存分配（各层 buffer、份数、队列深度、workspace 区域表）。
3. 读累加序定义（归约顺序、中间精度、转换点）与流水设计（生产/消费、双缓冲、slot 生命周期）。

详设与代码不一致时，先记录差异点，再以"代码实际行为"继续分析，并把差异本身作为检视意见候选
（文档与实现不一致同样是问题）。

### ④ 无详设时按固定顺序读代码，生成三份报告

读代码顺序固定，不要跳步：

```text
op_host 中的 tiling 逻辑
  -> op_kernel 中的 tilingKey（选择条件与实例表）
  -> op_kernel 中的算子 inner 接口调用
  -> 统计有哪些模板实例/分支
  -> 按 ② 的关键改动清单挑出关键模板
  -> 只读关键模板中的 kernel 详细逻辑
  -> 生成 分核方案报告 / 内存分配报告 / 算子流水报告
```

#### 报告 1：分核方案报告

| 项 | 内容 |
| --- | --- |
| 任务定义 | 任务的粒度（chunk、head、token 行、tile）与数量来源 |
| 核心映射 | 每个 core 分到的任务、余数分配规则、是否均衡 |
| 依赖关系 | 任务之间是否存在顺序依赖（同序列前后 chunk、head 间归约） |
| 数据切分 | 每个任务读写的数据区间（含 varlen/tail 的实际有效长度） |
| 并行度证据 | `blockDim`、active core 数、每核任务数与总任务数的关系 |

#### 报告 2：内存分配报告

| 项 | 内容 |
| --- | --- |
| 片上 buffer | UB/L1/L0A/L0B/L0C/BT/FB 各 buffer 的用途、大小、份数、对齐后占用 |
| 容量结论 | 各层占用与规格的余量；是否存在超限或到达临界 |
| 队列与 slot | 队列深度、slot 数、每个 slot 的状态机（free/writing/ready/reading） |
| GM/workspace | 每个 region 的 offset、size、使用方、生命周期、是否与其他 region 重叠 |
| 别名 | 多个 `LocalTensor`/GM 区间重叠或别名的情况与保护方式 |

#### 报告 3：算子流水报告

| 项 | 内容 |
| --- | --- |
| Stage 序列 | Stage 编号、执行单元（AIC/AIV/Scalar/MTE）、公式、依赖 |
| 生产/消费对 | 每对生产-消费的 pipe、内存层级、buffer/slot、同步原语与 flag/event id |
| 同步闭环 | 每次 set 是否有对应 wait，wait 是否有对应 set；是否有反向 flag/credit 保护复用 |
| 流水重叠 | double buffer/multi-buffer 是否真的重叠（两份物理存储 + 实际并发证据） |
| 例外路径 | 首轮、尾块、空任务、varlen 无效区、异常退出路径上的同步是否仍闭环 |

三份报告就是 lv3 检视的证据。没有报告就给出 lv3 结论，视为无效检视。

### ⑤ 三项分析

依据 ④（或 ③）的结果，只回答三个问题：

1. **分核策略与内存分配是否冲突**
   - 每核任务数与 buffer 份数/队列深度是否匹配（任务多于 buffer 时代际复用是否正确）。
   - 同一 buffer 在同一 core 的连续任务之间是否可能被覆盖（未等待消费完成即复用）。
   - 不同 core 是否写同一 GM/workspace 区间，或写入区间是否可能重叠。
   - tail/varlen 下实际数据量小于 tiling 假设时，是否出现越界或写到其它任务的数据。
2. **算子各流水线之间同步是否合理**
   - 每个跨 pipe/跨核的生产-消费对是否有成对同步，且同步原语与触发 pipe 匹配。
   - 同步范围是否最小必要（用大范围 `SyncAll` 顶替细粒度握手是设计问题）。
   - 复用 buffer/slot 前是否有反向同步或 credit；计数器深度是否与流水距离匹配。
   - 分支内外的同步是否对称（某分支跳过 set/wait 会导致计数器错位或超时）。
3. **算子累加序是否合理**
   - 部分和顺序、归约树与详设/标杆是否一致；不一致是否会放大误差。
   - 中间累加的 dtype 与位宽是否足以承载累加规模（fp16 累加、bf16 累加、int 溢出）。
   - 累加与归约的边界（chunk 内、chunk 间、多核间归约）是否与算法的数学定义一致。
   - 是否引入可避免的额外舍入（提前转 dtype、重复归一化、先 scale 后累加）。

## 2. 按算子类别建立独立 skill

lv3 不追求一套通用清单。按算子类别分别维护独立 skill，同一类别内共享根因经验；公共流程仍是本文五步。
每个类别 skill 至少记录：本类别涉及的分核/内存/流水形态、该类别的常见 lv3 根因、判定所需的文档与手册章节、
以及典型误报边界。

示例（写具体到环节，不写"注意同步"这类泛化句）：

| 算子类别 | 常见 lv3 根因示例 |
| --- | --- |
| CV 混合（AIC+AIV 协作） | CV 核间同步缺少必要的反向同步；CV 核间同步的触发流水线设置错误（在错误 pipe 上 set/wait）；flag 参与核数与分工不匹配导致计数器错位；某一分支未参与握手导致超时 |
| Cube/Matmul 为主 | L0A/L0B 复用未等待上一次 cube 完成；baseM/baseN/baseK 与 tiling 假设不一致导致分块叠加越界；Fixpipe 写回与下一轮 MTE2 覆盖冲突 |
| Vector 为主 | MTE2→VEC、VEC→MTE3 事件方向写反；同一 UB 二次读写只用 `PipeBarrier` 而实际跨 pipe；多缓冲 slot 与 event id 生命周期不闭环 |
| 递推/带状态（chunk 间依赖） | 状态传递缺少顺序保证（下一 chunk 读到未完成状态）；状态 buffer 在同一 core 多轮之间被提前复用；首 chunk 与尾 chunk 的同步路径不对称 |
| 归约/累加密集 | 多核归约缺少跨核同步或重复累加；累加顺序与标杆不一致导致长序列误差放大；中间量按低精度 dtype 落盘 |

同一批 lv3 根因在仓库里出现过的实例（用于校准检视尺度，状态以仓库为准）：

| lv3 根因 | 相关 issue / PR |
| --- | --- |
| 同步/事件时序 | [PR #700](https://github.com/flashserve/flash-linear-attention-npu/pull/700)（Gate 标量暂存区读后写依赖缺失，UB WAR hazard）、[#325](https://github.com/flashserve/flash-linear-attention-npu/issues/325)（边界场景流水同步与尾块精度）、[#462](https://github.com/flashserve/flash-linear-attention-npu/issues/462)（特定 GQA head 配置下 kernel hang） |
| tiling/分核错误 | [#440](https://github.com/flashserve/flash-linear-attention-npu/issues/440)（varlen 路径输出 bitwise 不确定，分核/归约顺序导致）、[#508](https://github.com/flashserve/flash-linear-attention-npu/issues/508)（varlen 计数/位宽） |
| 累加顺序/舍入 | [#563](https://github.com/flashserve/flash-linear-attention-npu/issues/563)（A5 Stage5 先加后乘与参考实现不一致）、[#511](https://github.com/flashserve/flash-linear-attention-npu/issues/511)（不同中间精度的验收标准） |

完整的按条目归档见 [`../../../reference/04-operator-development/engineering-structure.md`](../../../reference/04-operator-development/engineering-structure.md) §9。

新类别按同样格式补充到本节，或单独建 skill 并由本文件链接。

## 3. 输出

lv3 检视的输出包含两部分：

1. 三份报告（分核方案、内存分配、算子流水），作为结算依据或附加在检视意见之后；
2. 检视意见，按 [`04-output-format.md`](04-output-format.md) 输出，`精度级别` 写 `lv3`，
   `细粒度根因` 写 13 类之一，并在"可能导致的结果"里写明依据哪份报告的哪一条。

只在报告里指出"存在风险"而不给出具体的文件行号、触发条件和影响，不算有效检视意见。
