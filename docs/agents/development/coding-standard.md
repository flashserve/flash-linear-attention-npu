# Ascend C 算子编码规范

本文规定本仓 Ascend C 算子的代码组织、命名、职责边界和资源生命周期表达。格式细节以仓库 `.clang-format` 为准；本文补充格式化工具无法判断的架构约束。

本规范抽取经过完整 host、kernel、AIC/AIV 和验证链路检验的稳定写法，但不要求复制任何具体算子的 shape、tile、flag 编号或展开因子。具体资源数字必须由当前算子的语义、目标 SOC 和容量重新推导。

## 文件分层

一个 L0 算子优先保持以下职责边界：

| 文件 | 责任 |
|---|---|
| `op_kernel/<op>.cpp` | kernel 入口、架构选择、tiling 解码和模板实例化 |
| `op_kernel/<op>_struct.h` | host/kernel 共用 tiling ABI、模板参数与支持组合 |
| `op_kernel/<op>_common.h` | 协议常量、任务信息结构和无状态公共 helper |
| `op_kernel/<op>_cube.h` | AIC/Cube 资源、搬运、MMAD 和写回 |
| `op_kernel/<op>_vector.h` | AIV/Vector 搬运、逐元素计算、状态更新和输出 |
| `op_kernel/arch*/...` | 特定架构的同构实现，不改变 L0 契约 |
| `op_host/op_tiling/*` | 参数校验、任务划分、workspace、tiling key 和 blockDim |

小算子可以合并没有独立职责的文件，但不得把 host 校验、架构选择、Cube 和 Vector 热路径混成一个不可审查的大文件。架构目录只表达内部实现差异，不得形成新的 L0 名称、原型或注册入口。

include 依赖保持单向：入口依赖 struct/common/Cube/Vector，Cube/Vector 依赖 struct/common，common 可以依赖 struct；struct 不得反向依赖 kernel 实现。头文件必须有 include guard，并保持项目许可证、`\file` 和 `\brief`。

## 命名与格式

- 使用仓库 `.clang-format`，4 空格缩进、禁用 tab、120 列上限，不手工制造另一套格式。
- 类型、类、struct 和公开方法使用 PascalCase，例如 `TaskInfo`、`Init`、`Process`。
- 局部变量和函数参数使用 camelCase，例如 `taskIdx`、`workspaceSlot`。
- 编译期常量和宏使用 UPPER_SNAKE_CASE；常量名必须带上资源或协议语义，不能只写数字序号。
- 类成员使用 camelCase 加尾缀 `_`；缓存状态同时包含 key、slot 和 valid，不用特殊数值兼任 valid。
- dtype 类型别名使用清晰的 `Type`/`Traits` 命名；不要在多个文件重复维护同一 dtype 映射。
- helper 名称使用动作加对象，例如 `CopyInStateRows`、`WorkspaceBase`、`InitEvents`、`DrainFlags`，避免 `Handle`、`DoWork` 一类模糊名称。

注释说明 invariant、硬件限制、同步原因和 fallback 条件，不逐行复述代码。性能数字、临时调试结论和个人环境不写入源码注释。

## Kernel 入口

kernel 入口只允许承担以下工作：

1. 选择目标架构对应的实现头文件。
2. 解码 tiling data，设置 kernel task type 和必要的运行时保护。
3. 将 dtype、固定维度和功能模式映射为编译期类型或模板参数。
4. 在 AIC/AIV 分支分别构造实现对象，依次调用 `Init()` 和 `Process()`。

入口不得包含 stage 主循环、复杂 offset、DataCopy、MMAD 或 Vector 公式。未使用的保留参数应显式标记；必需 workspace 为空时尽早返回，但不能在已进入同步协议后单边提前退出。

SOC 选择集中在入口 include/trait 层，禁止在 Cube/Vector 内层循环反复判断平台。所有 SOC 继续使用相同 L0 原型和 L2 调用路径。

## Tiling ABI 与模板

- tiling data 使用 host/kernel 都可稳定解释的定宽标量字段，不放 STL 容器、指针或生命周期不明确的对象。
- 字段按语义分组：公共 shape、任务划分、功能标志、workspace offset/size、tile 参数。新增字段时同步 host 写入和 kernel 读取。
- dtype、固定维度、layout 模式和会改变指令结构的功能开关使用模板参数；host 只注册真实支持的组合。
- 模板内使用 `if constexpr` 删除无效路径，避免在热循环保留 dtype 或模式判断。
- TilingKey 只标识框架要求的编译期结构组合；运行时规模、offset、有效长度和 task 数保留在 tiling data。
- unsupported 组合在 host 明确报错，不能静默映射到某个近似模板。

## Tiling 处理器

tiling 处理器公开接口保持小而稳定，通常只暴露 `Process()` 以及 workspace、blockDim、tilingKey 的只读结果。内部按以下顺序拆分并在失败时立即返回：

```text
PreCheck
  -> CommonTiling
  -> Fixed/Variable Tiling
  -> WorkspaceTiling
  -> TilingKey/BlockDim
```

- `PreCheck` 校验 required/optional、维数、dtype、互斥关系和属性范围，错误日志带公开输入名、期望和实际值。
- `CommonTiling` 只推导各模式共有的 shape、head ratio、chunk 和任务基础信息。
- fixed/varlen 的差异集中在任务映射 helper，不复制后续 workspace 和 kernel 数据流。
- `WorkspaceTiling` 使用命名 segment 逐段计算 size/offset，并按真实 dtype 与 API 要求对齐。
- blockDim 来自平台能力与任务模型；任务为空或平台信息异常时使用明确保护，不产生除零和负 offset。

## Kernel 类职责

Cube 和 Vector 类的公开接口收敛为：

- 默认构造函数。
- `Init(...)`：绑定 GM、缓存 tiling、初始化资源、buffer 和 event。
- `Process()`：按 `task -> window -> chunk -> head -> tile/stage` 编排主流程。

其余实现放入 private helper：

- 地址与 workspace helper。
- CopyIn/CopyOut、cast 和局部计算 helper。
- event/flag 到 slot 的映射。
- 初始化、drain 和 release。
- 可复用的 tile/MMAD 流程。

`Process()` 应能直接看出依赖方向、owner、stage 和同步边界。重复的搬运或 MMAD 流程提取 helper，但不要为了减少几行代码隐藏关键同步顺序。

## 任务和地址

- task owner、并行轴和串行轴必须由显式变量表达；不要从裸 block index 在多个位置重复推导不同语义。
- fixed/varlen 共用 `SeqInfo`、`ChunkInfo` 或等价结构，结构带 `valid`，主循环不复制两套 offset 分支。
- task、token、workspace 和 tensor offset 使用 `int64_t` 或足够宽的无符号类型；传给窄参数 API 前显式转换。
- 地址 helper 的单位必须清楚区分 element 和 byte；跨 dtype workspace 通过命名转换函数处理。
- head ratio、tail head、tail chunk、row/tile tail 和空序列显式处理，不能依赖 padding 恰好为零。
- optional 输出只有在 tiling 标志和地址同时有效时绑定；未参与计算的输出区域按公开语义初始化。

## 资源与 Workspace

- 逻辑 workspace bank/slot 与 UB、L1、L0 物理 ping/pong 分别命名和计数。
- 每个 workspace segment 声明 dtype、element 数、byte 对齐、producer、consumer 和最后消费点。
- 资源 offset、tile bytes、buffer count 和容量上限使用命名 `static constexpr` 集中定义。
- 可静态推导的 L1、L0C、UB 固定区执行 `static_assert`；动态容量由 host tiling 计算并保留 guard。
- resident、scratch、input、output 和 state 不共享一个模糊 buffer 名称；底层地址复用时也要维持独立生命周期说明。
- ping/pong slot 使用显式索引轮转；覆盖前等待 free，最后消费后归还，不能只靠循环奇偶隐式假设安全。
- cache 复用必须记录复用 key、valid、slot 和 release 条件，并在 key 变化前完成旧对象最后消费。

## 同步与生命周期

每个 event/flag 都必须能回答：谁生产、谁消费、保护哪个 slot、方向是什么、何时可复用。

- event ID 使用命名常量或数组，名称包含资源与 pipe 方向；热循环禁止散布裸 event 数字。
- `InitEvents/InitFlags` 建立初始 free 状态；`DrainEvents/ReleaseEvents` 等待所有在飞操作并成对释放动态 event。
- Copy helper 按固定顺序表达 `wait free -> 搬运/计算 -> set ready`，调用方显式传入或接收实际 slot。
- Cross-core ready/free 按 task、head、stage 和分支配平；非 owner、tail 和空 payload 也要执行协议动作。
- `PipeBarrier` 只解决同 pipe 内真实 RAW/WAR/WAW，不能代替 MTE/Cube/Vector event 或 cross-core flag。
- 不添加习惯性的 kernel 末尾 barrier；收尾同步由具体在飞资源的 drain 负责。
- direct path 与 GM fallback 保持相同公开语义和协议计数，不能让数据通路分支改变对端等待次数。

## Vector 与 RegBase

- 采用 RegBase/VF 时，helper 使用 `__simd_vf__ inline`，每个 helper 只表达一个公式或转换。
- mask、广播 scalar、Duplicate 和循环不变量移出内层循环；寄存器链保持独立，避免无必要的 RAW 串行。
- 完整向量使用 full mask，尾向量按有效元素更新 mask；不能用 full mask 写过有效边界。
- cast、exp、mul/add 等组合先保证公式和 dtype 语义一致，再通过 profiling 决定融合、成对处理或展开。
- `#pragma unroll`、pair loop 和硬件 loop 必须有固定基线证据；寄存器压力恶化时允许回退到更小展开。

## Cube 与 MMAD

- 优先复用仓库已有 Catlass/TLA layout、tensor、tile copy 和 MMAD 组件。
- GM->L1、L1->L0、MMAD、Fixpipe 的 slot 与事件分别表达，不把一次 barrier 当作整条流水生命周期。
- 归约循环显式标记 first/last tile；初始化累加只发生在 first，Fixpipe ready 只在 last 发布。
- L1/L0 resident operand 在最后一次 MMAD 后释放；跨 head/group 复用时以最小复用 key 管理 cache。
- L0C 双缓冲由容量决定，单个逻辑结果只执行一次 copyout。
- tail M/N/K 和硬件最小 tile 的 padding策略显式写出，padding 值必须是当前公式的中性值。

## 禁止项

- 在 kernel 入口堆放完整计算流程。
- 为 A5 或单个 shape 新建第二套 L0 契约。
- 把 dtype、SOC 和固定模式的运行时分支放进最内层循环。
- 使用无名称的 workspace offset、event ID、flag 编号和 buffer 数量。
- 只初始化 event 不 drain/release，或在 optional/tail 分支提前退出绕过回收。
- 用 `PipeBarrier` 掩盖跨 pipe、跨核或 buffer owner 不清的问题。
- fixed/varlen 各复制一套 kernel 主流程。
- 直接复制其他算子的 tile、slot、容量、flag 编号和展开因子而不重新推导。

## 提交前检查

交付时同时执行 [`checklist.md`](checklist.md)。性能优化还要执行 [`../optimization/checklist.md`](../optimization/checklist.md)，A5 实现额外执行 [`../optimization/soc/a5.md`](../optimization/soc/a5.md) 的编码与架构检查。
