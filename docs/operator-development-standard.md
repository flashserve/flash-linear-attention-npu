# 算子开发规范

本文定义本仓新增、替换或重要修改算子时的准入规范。目标是让每个算子在功能、性能、平台、调用通路、编码质量、测试覆盖和文档一致性上具备可评审、可验证、可维护的交付标准。

本文中的“必须”表示 PR 合入前应满足；“应”表示默认要求，如无法满足需要在算子 `docs/design.md` 或 PR 中说明原因、影响面和补齐计划；“可选”表示不作为合入硬门槛，但若已有兼容用户或历史入口，应尽量保留。

## 1. 适用范围

以下场景必须按本文执行：

- 新增 Ascend C 算子。
- 新增或重要修改 Triton 算子。
- 用 Ascend C 算子替换或加速已有 Triton 算子。
- 修改已有算子的公开接口、shape、dtype、layout、平台支持、tiling、kernel、workspace、精度、性能或调用通路。
- 修改公共模块并可能影响多个算子。

纯文档修复、注释修复或不改变行为的小范围清理，可按影响面裁剪验证，但不得削弱已有公开约束。

## 2. 算子能力基线

每个正式支持的算子必须在开发前明确能力边界，并在代码、测试和文档中保持一致。

### 2.1 功能语义

必须说明并验证：

- 输入、输出、属性、可选参数和预留参数语义。
- 支持的 dtype、format、layout、shape 范围、batch/head/chunk 关系和 varlen/dense 场景。
- 特殊值、空输入、尾块、padding、无效 token、partial chunk、初始状态和最终状态的语义。
- workspace 大小、对齐要求、输出别名、原地写入或禁止原地写入的约束。
- 不支持场景的拦截条件、返回码和报错文本。

文档中写明的“不支持”“必传”“仅支持”“预留参数不支持非空”等约束，代码必须有同等语义的拦截；代码新增或收紧拦截时，README、统一 API 文档和测试必须同步。

### 2.2 可用调用通路

算子必须先明确实现类型，再提供对应调用通路。`fla_npu.ops.ascendc` 与 `fla_npu.ops.triton` 是二选一关系，不要求同一个算子同时提供两者：

| 通路 | 要求 | 验收重点 |
| --- | --- | --- |
| `aclnn` | Ascend C 算子必须支持 | 提供 `aclnn*GetWorkspaceSize` 和执行接口，接口签名、返回码、workspace、stream 语义稳定；统一 API 文档与实现一致。 |
| Ascend C `<<<>>>` 直调 | Ascend C 算子必须支持 | 至少有可构建、可运行的 C++/示例或测试入口，用于验证 kernel launch、tiling data、workspace 和 stream 基础链路。 |
| `fla_npu.ops.ascendc` | Ascend C 算子必须支持 | 作为 Ascend C 算子的 Python 主入口和主测试入口；默认调用不得依赖 `torch_npu` dispatcher 或 legacy 扩展注册。 |
| `fla_npu.ops.triton` | Triton 算子必须支持 | 作为 Triton 算子的 Python 主入口和主测试入口。 |
| `torch.ops.npu` | 可选 | 仅作为 legacy 兼容入口；如保留，必须通过 `fla_npu.load_legacy_torch_ops()` 显式加载，不能成为默认新代码路径。 |

### 2.3 默认 Python 入口

Ascend C 算子使用：

```python
from fla_npu.ops.ascendc import op_name
```

Triton 算子使用：

```python
from fla_npu.ops.triton import op_name
```

实现类型对应的 Python 入口是该算子的主验证入口，应覆盖精度、边界和泛化场景。`torch.ops.npu.*` 是历史兼容路径，不得作为新增算子的主文档入口或主测试入口。

## 3. 平台支持

正式支持算子必须覆盖 A2 / A3 / A5：

| 产品 | `--soc` |
| --- | --- |
| A2 | `ascend910b` |
| A3 | `ascend910_93` |
| A5 | `ascend950` |

必须满足：

- 三个平台均可编译。
- 三个平台均有基础功能验证和主通路精度验证。
- 平台差异必须在代码、tiling、文档和测试中显式表达，不能通过沉默 fallback 掩盖。
- 若某个平台因 CANN 版本、硬件能力或需求边界暂不支持，必须在 issue、算子 `docs/design.md` 或 PR 中说明原因、用户影响、拦截行为和补齐计划，并经过维护者确认。

涉及平台差异的报错日志应包含当前平台类别和期望条件，避免只提示某个孤立参数错误。

## 4. 编码规范

### 4.1 交付件完整性

新增或重要修改算子时，不能只提交 kernel 或单一路径代码。应按影响范围同步以下交付件：

- Ascend C 算子定义：`op_host/*_def.cpp` 中的输入、输出、属性、dtype、format、required/optional 定义。
- Shape 和 tiling：InferShape、参数校验、tiling processor、tiling data、workspace 推导、block dim 和 SOC 分支。
- aclnn/op_api：`op_host/op_api/aclnn_*.h`、`aclnn_*.cpp`、必要的 L0/L2 op_api 包装和 CMake 配置。
- kernel：`op_kernel/` 下 kernel 入口、模板实现、arch 特化、tiling data 结构、CMake、A2/A3/A5 编译配置和必要的 binary/config 文件。
- `fla_npu` 适配层：按实现类型提供 `fla_npu.ops.ascendc` 或 `fla_npu.ops.triton` 稳定 Python 导出；Ascend C 路径还需同步 aclnn ctypes 包装、输出分配、正反向绑定、默认 import 行为和打包入口。
- PyTorch/legacy 适配：如需要保留 `torch.ops.npu`，同步 `torch_custom/fla_npu/*.yaml`、生成入口和显式 legacy 加载路径。
- Triton 适配：Triton 算子通过 `fla_npu.ops.triton` 导出；若 Ascend C 替换 Triton，Python 主入口和 example 默认路径迁移到 `fla_npu.ops.ascendc`。
- 测试：JSON 用例、实现类型对应的 Python 主精度测试、必要调用通路测试和 Example/ST；Ascend C 算子还需覆盖 aclnn 与 `<<<>>>`。
- 文档：算子 README、`docs/design.md`、统一 `docs/api.md` 和 PR 验证矩阵。

不涉及的交付件应在 PR 中说明“不涉及”的原因；涉及公共 runtime、公共 kernel helper、打包或 schema 时，应列出影响到的算子和回归范围。

### 4.2 `fla_npu` 适配层

`fla_npu.ops.ascendc` 是本仓 Ascend C 新代码的稳定 Python 主入口。新增或修改 Ascend C 算子时，适配层必须满足：

- 在 `torch_custom/fla_npu/fla_npu/ops/ascendc/` 下导出稳定函数名；公开名称应去掉内部 `npu_` 前缀，保留 raw op 时需说明用途。
- Python wrapper 的参数顺序、可选参数、默认值、输出数量和 dtype/shape 分配必须与 `aclnn_*.h`、README 和测试保持一致。
- wrapper 应通过默认解耦路径调用 `aclnn*GetWorkspaceSize` 和 `aclnn*`，不得把默认调用退回 `torch.ops.npu`、`torch.ops.load_library()` 或 PyTorch C++ extension。
- 默认 import `fla_npu` 或 `from fla_npu.ops.ascendc import op` 不得强制 import `torch_npu`；读取 NPU 私有 format 等信息只能走公共 runtime 的可选机制。
- wrapper 不得无说明地对输入做 `.contiguous()`、转置、reshape、CPU round-trip 或同步；确因 aclnn ABI 需要时，必须说明原因并补充 format/layout 测试。
- stream 使用当前 NPU stream 约定，不在单个算子 API 上新增零散 `stream=` 参数；如需适配外部 executor，应优先改公共 runtime。
- 异步 launch 依赖的输出、workspace、临时 tensor 或 int tensor 必须被保活到 kernel 消费完成，不能只依赖局部变量生命周期。
- 支持 autograd 的算子，应在 Python 入口层用 `torch.autograd.Function` 绑定 forward/backward；不满足反传语义的 decode、inplace cache 或状态更新模式应保留 raw 路径并测试。
- 新增正反向算子时，应同步 raw wrapper、公开导出名、forward/backward 映射、测试和文档。
- 修改 `_runtime.py`、打包逻辑、OPP 查找、stream、format、workspace 或保活机制时，必须同步更新 `docs/torch-npu-decoupled-architecture.md` 和相关回归测试。

`torch.ops.npu` 只作为 legacy 兼容入口。新增算子默认不应依赖该路径；确需保留时，必须通过 `fla_npu.load_legacy_torch_ops()` 显式启用，并在测试中只作为可选通路覆盖。

### 4.3 Tiling 设计

应优先使用“tiling 数据 + 编译期模板化”的方案，而不是为大量组合新增 tiling key。

推荐方式：

- host tiling 负责校验入参、推导 shape/layout/workspace/任务划分，并写入结构化 tiling data。
- kernel 入口根据少量字段选择模板实例。
- 模板内部使用 `if constexpr` 或模板参数裁剪 dtype、固定维度、layout、safe gate、是否 varlen 等热路径分支。
- fixed/varlen、layout、head ratio、chunk 映射等策略应封装为清晰的 strategy 或 helper，避免散落在 kernel 内层循环。

tiling key 只应用于少量确实需要独立 kernel 变体或框架注册机制要求的场景。若必须使用 tiling key，PR 中必须说明：

- 为什么模板化 tiling data 不能满足需求。
- tiling key 对应的语义、组合数量和增长上限。
- 每个 tiling key 覆盖的 dtype/layout/shape/SOC 范围。
- 是否会影响二进制体积、编译耗时、维护成本和后续泛化。

不得用 tiling key 承载本应由结构化 tiling data 表达的普通规模参数、offset、workspace 地址、runtime shape 或大量属性组合。

### 4.4 编译选项

`add_ops_compile_options` 中的 `--cce-auto-sync=off` 必须保持为 `off`，不得改为 `on`。

原因：

- 本仓 kernel 的跨 pipe、跨核和 workspace 生命周期应由显式 event、queue、barrier、flag 和 tiling 协议表达。
- 自动同步可能掩盖真实 RAW/WAR/WAW 依赖问题，破坏性能定位，也可能让不同 CANN/SOC 下行为难以复现。

如果新增同步，应说明生产 pipe、消费 pipe、内存层级、buffer slot、读写方向和复用时机，避免用大范围同步替代生命周期设计。

### 4.5 算子语义边界、阶段拆分与 L2/L0 分工

算子边界必须由数学语义、输入输出契约和可独立验证的功能确定，不能为了减少 launch 数量，把关系较弱的计算强行塞进同一个算子或同一个 L0 kernel。

必须遵守：

- aclnn、`fla_npu`、`<<<>>>`、legacy schema、README、API 文档、JSON 用例和 example 均不得暴露 `stage=1/2/3`、`stage_id` 等内部执行编号，也不得要求调用者理解或选择内部阶段。公开接口只表达完整算子语义。
- 内部阶段应使用计算语义命名，例如“chunk 内矩阵准备”“状态传播”“输出合成”，不得长期复用同一个 L0 kernel 并通过整数 `stage` 切换无强耦合分支。
- 完整算子天然包含“并行生成 -> 全核归并/转换/写回”这类强生产消费关系时，可以在一个 L0 kernel 内拆成两个按计算语义命名的 phase；kernel 入口仍只表达一次完整算子调用，不能增加公开 stage 属性。
- 同一 L0 的 phase 边界必须先闭合本阶段异步搬运和 queue 生命周期，再按真实依赖执行 `TPipe::Reset()` 与必要的 `AscendC::SyncAll<false>()`，随后初始化下一 phase 的 queue/buffer。Reset 与 SyncAll 的先后必须结合本地资源释放和跨核 GM/workspace 可见性证明，所有参与核必须走到同一同步点。
- 两段计算只有 GM 数据依赖、没有共同归并语义或片上复用价值时，应拆成两个语义独立的 L0 kernel。每个 L0 拥有闭合契约；两次 launch 依靠同一 stream 和显式中间量交接，不在 kernel 之间虚构 `SyncAll`。
- 中间 tensor/workspace 必须有明确 owner、dtype、layout、生命周期和消费时点，不得把整数 stage 当作所有权协议。
- 使用 `SyncAll`/`TPipe::Reset()` 时，`docs/design.md` 必须说明参与核类型和数量、schedule mode、A2/A3/A5 支持情况、同步前后的数据可见性、Reset 前已排空的队列和 Reset 后重新初始化的资源。不能用它们掩盖本可通过任务划分消除的依赖。

L2 与类型转换还必须遵守：

- L2 不得在核心 L0 之间调用独立 `Cast`，也不得先 cast 输入再交给 kernel，或在 kernel 输出后通过 cast 适配公开 dtype。
- 输入归一化、阶段间 dtype 转换、累加精度转换和输出转换应融合到生产者或消费者 kernel，在 GM 读入、片上计算或 GM 写回边界完成。
- 公开输入 dtype 与内部计算 dtype 不一致时，tiling、kernel 和设计文档必须共同说明转换位置、累加类型、舍入/饱和行为和精度验证；L2 只传递原始公开输入和目标输出描述符。
- 评审时若在 aclnn L2 看到通用 `Cast` L0，或在公共接口看到 stage 编号，应按架构红线处理，不能以“复合算子”或“便于复用”为由直接放行。

> **反面样例：当前 KDA 正向实现。** 现有 `ChunkKdaFwd` 通过 `stage=1/3/2` 复用同一 L0，并由 L2 拼接 gate cast、阶段间 cast/scale 和状态 kernel。各 stage 关系不足以支撑这种编号式强行融合，且 `<<<>>>` 直调需要理解 stage 语义，不符合本节要求。该实现只能作为历史整改对象，不能作为新增算子的参考架构。整改应优先收敛为一个完整入口下的两个语义 phase，在 L0 内完成必要的全核同步和 `TPipe::Reset()` 隔离，并把 cast 融入 kernel；若两个 phase 不具备共同归并语义或片上复用价值，再拆成两个语义独立的 L0。

> **正面参考：** [ops-nn PR #4803 GroupNormSwishGrad A5 实现](https://gitcode.com/cann/ops-nn/pull/4803/diffs)保持单一公开 kernel 入口，在 `Process()` 内先生成 `dx` 和归约 workspace，完成第一 phase 后调用 `pipe.Reset()`，各核在消费 workspace 前执行 `SyncAll()`，再初始化第二 phase buffer 并在 kernel 内完成 reduce/cast；host 对需要全核同步的路径设置 batch schedule mode。参考的是“公共语义不泄漏、phase 内聚、同步和 Reset 闭环、cast 内置”的结构，不能脱离具体参与核和数据依赖机械复制调用顺序。

### 4.6 Kernel 和内存生命周期

必须遵守：

- 不依赖 core 启动顺序、blockIdx 顺序或日志顺序表达跨核依赖。
- 跨核共享 GM/workspace 时，优先通过任务划分保证地址不重叠；确有依赖时使用成对的 ready/free flag 或阶段化 kernel launch。
- UB/L1/L0 buffer 复用前必须确认上一生产者和消费者完成。
- `SetFlag`/`WaitFlag`、`CrossCoreSetFlag`/`CrossCoreWaitFlag`、`PipeBarrier` 的模板参数、event id、flag id 和参与核条件必须成对一致。
- `PipeBarrier<PIPE_V>()` 只能约束 V pipe 内部顺序，不能替代 MTE/CUBE/FIXPIPE 之间的硬事件。
- 使用 `TPipe`、`TQue`、`LocalTensor` 时必须遵守 `AllocTensor -> EnQue -> DeQue -> FreeTensor` 所有权闭环。
- 手工 ring buffer 或 ping-pong buffer 必须按 slot 维护 free、being written、ready、being read 状态。

### 4.7 接口和错误处理

必须遵守：

- `op_host/*_def.cpp`、InferShape、Tiling、op_api、kernel、PyTorch schema、Python 导出、README 和统一 API 文档中的参数顺序、必选/可选、dtype、format、shape、返回值保持一致。
- 空指针、dtype/format、shape、属性值、layout 组合、预留参数等错误应返回与文档一致的错误码。
- 报错文本应能定位真实触发条件，必要时包含当前 shape、layout、dtype、平台、推导出的模式和期望值。
- 不得在默认路径中引入静默 fallback 来绕过 kernel 问题；fallback 只能用于明确声明的非目标兼容路径，并必须有测试和文档说明。

## 5. 精度要求

精度必须通过，且必须覆盖原问题和目标泛化场景。

必须满足：

- Ascend C 算子的主验证入口为 `fla_npu.ops.ascendc`，Triton 算子的主验证入口为 `fla_npu.ops.triton`。
- 对比基准应来自可信 PyTorch/Triton/数学参考实现，阈值来源应写入测试或算子文档。
- 所有公开输出都要验证，包括中间返回、最终状态、workspace 派生输出和可选输出。
- 检查 NaN/Inf、重复运行一致性、边界 shape、尾块、padding、varlen、head ratio、dtype 转换和极值范围。
- 修改 bug 时必须保留能稳定复现原问题的回归用例，并回到原始 shape/layout/dtype/调用链确认问题消失。

禁止通过以下方式制造精度通过：

- 收窄输入 range。
- 跳过失败 case。
- 降低覆盖强度。
- 屏蔽本应有语义的输出区域。
- 放宽阈值。
- 改写原用例以避开目标分支。
- 用非目标 fallback 替代 kernel 根因修复。

若失败集中在无效区，必须先明确无效区是否有公开语义；只有确认无语义要求时，才可在后处理中按文档约束屏蔽。

## 6. 性能要求

新增 Ascend C 算子替换已有 Triton 算子时，必须保证目标场景性能优于 Triton 实现。

必须满足：

- 对比同一 shape、dtype、layout、输入范围、warmup、迭代次数和环境下的 Triton 实现。
- 性能结论以 profiling 或稳定 benchmark 为准，不以单次 Python wall time 作为结论。
- 至少覆盖主推 shape、边界 shape 和会触发不同 tiling/template 的关键场景。
- 若 A2/A3/A5 使用不同策略，需要分别给出性能结论或说明当前平台差异。
- example 中原本使用 Triton 算子的路径，在 Ascend C 替换后必须改为使用 Ascend C 主入口；保留 Triton 只能作为对照或 fallback，并需在文档中说明。

性能优化不得牺牲精度、稳定性、平台支持或公开通路。若某些非目标 shape 只保证功能正确但不追求性能，应在能力边界中说明，并确保不会影响主场景。

## 7. 测试规范

### 7.1 测试资产归档路径

测试用例定义和测试执行代码统一归档，新增算子不得再在算子源码目录、临时脚本或多个 Python 文件中分别维护 shape、dtype 和属性组合。

```text
tests/
├── op_cases/
│   └── <op_name>.json
└── operators/
    └── <op_name>/
        ├── README.md
        ├── common/
        ├── accuracy/
        ├── routes/
        ├── ut/
        │   ├── op_host/
        │   └── op_kernel/
        ├── performance/
        └── st/
```

各目录职责如下：

| 路径 | 归档内容 |
| --- | --- |
| `tests/op_cases/<op_name>.json` | 算子的唯一用例设计来源，统一定义能力范围、输入 shape、dtype、layout、属性、标签、期望结果、精度阈值和适用平台。 |
| `tests/operators/<op_name>/README.md` | 测试入口、依赖、执行命令、JSON 筛选方式、标杆来源和结果判定方法。 |
| `tests/operators/<op_name>/common/` | 数据生成、reference/golden、结果比较、JSON 解析等各测试共用代码。 |
| `tests/operators/<op_name>/accuracy/` | 以实现类型对应的 `fla_npu.ops.ascendc` 或 `fla_npu.ops.triton` 为入口，执行主精度、泛化、边界和回归测试。 |
| `tests/operators/<op_name>/routes/` | 归档实现类型要求的调用通路测试；Ascend C 算子覆盖 aclnn、`<<<>>>`，实现 legacy 时再覆盖 `torch.ops.npu`。 |
| `tests/operators/<op_name>/routes/test_fast_kernel_<op_name>.py` | `examples/fast_kernel_launch_example` 构建出的 direct-launch Python 扩展回归；参数矩阵仍从唯一 JSON 读取。 |
| `tests/operators/<op_name>/ut/op_host/` | InferShape、参数校验、tiling、workspace、block dim 和平台分支单元测试。 |
| `tests/operators/<op_name>/ut/op_kernel/` | kernel 模板分支、tiling data 解析和可独立验证的 kernel 单元测试。 |
| `tests/operators/<op_name>/performance/` | 性能用例、profiling/benchmark 入口和结果汇总；Ascend C 替换 Triton 时保存同场景对比。 |
| `tests/operators/<op_name>/st/` | 算子组合、模型模块和仓内 example 的端到端回归测试。 |

测试脚本只负责读取 JSON、构造输入、调用被测通路和判断结果，不得在脚本中新增未登记的关键用例。算子源码目录及 `torch_custom`、example 等适配工程中不得保留主线算子的独立 `test/`、`tests/`、`ATK/` 用例目录或第二份参数矩阵；历史资产必须先逐条并入唯一 JSON、完成可逆性和执行资产校验，再删除旧文件夹。example 工程可保留构建器，但主线算子的执行 harness 归档到 `tests/operators/<op_name>/routes/`。

### 7.2 必测内容

每个算子至少应包含以下测试：

| 测试类别 | 必测内容 |
| --- | --- |
| 构建与平台 | A2（`ascend910b`）、A3（`ascend910_93`）、A5（`ascend950`）均可编译；检查各平台 kernel/tiling 分支和 `--cce-auto-sync=off`。 |
| op_host | Ascend C 算子覆盖输入输出数量、required/optional、dtype、format、shape 推导、workspace、block dim、tiling data、模板选择和非法参数拦截。 |
| 主精度 | 通过实现类型对应的 `fla_npu` 入口调用，逐项比较全部公开输出，并检查 NaN/Inf；覆盖支持的 dtype、layout、format、shape、属性和平台。 |
| 泛化 | 最小、典型、最大目标 shape，非整除尾块，batch/head/sequence/chunk/K/V 等关键维度组合，以及会切换 tiling/template 的场景。 |
| 功能分支 | fixed/varlen、正向/反向、初始/最终状态、可选输入为空或非空、padding、无效 token、partial chunk 等算子声明支持的分支。 |
| 边界与异常 | 空输入、单元素、维度上下界、非法 dtype/layout/shape/属性、空指针、预留参数非法值；校验实际返回码和错误类型。 |
| 回归 | 每个已修复问题保留稳定复现用例，保持原始 shape、dtype、layout、属性、随机种子和调用链。 |
| 调用通路 | Ascend C 算子覆盖 aclnn、`<<<>>>` 和 `fla_npu.ops.ascendc`；Triton 算子覆盖 `fla_npu.ops.triton`；`torch.ops.npu` 如实现则必须覆盖。 |
| 性能 | Ascend C 替换 Triton 时，使用相同 shape、dtype、layout、输入、warmup 和迭代配置对比；覆盖主场景及关键 tiling/template 分支。 |
| Example/ST | 覆盖仓内真实上层调用；Ascend C 替换 Triton 后，验证 example 已改走 `fla_npu.ops.ascendc`。 |

涉及手工 UB/L1/GM buffer 复用、跨 pipe 或跨核同步的 kernel 修改，还应按问题类型补充 sanitizer 检查，并确认实际运行的是 sanitizer 编译的 kernel。

三平台构建不能只检查 `OpDef::AddConfig` 或文档声明。应在对应 CANN 环境执行统一构建入口；命令返回成功且生成本次构建的新 run 包后，才可记录该平台编译通过：

```bash
python3 ci/run_operator_build_matrix.py --soc ascend910b
python3 ci/run_operator_build_matrix.py --soc ascend910_93
python3 ci/run_operator_build_matrix.py --soc ascend950
```

也可在能够同时编译三个目标的环境执行 `python3 ci/run_operator_build_matrix.py --soc all`。`--dry-run` 只用于检查命令和算子集合，不能作为编译通过依据。

### 7.3 各调用通路测试要求

| 通路 | 测试要求 | 固定归档文件 |
| --- | --- | --- |
| `fla_npu.ops.ascendc` | 仅用于 Ascend C 算子；作为主测试方式，承担 kernel 精度、泛化、边界、功能分支和回归用例。 | `tests/operators/<op_name>/accuracy/test_<op_name>.py` |
| `aclnn` | 仅用于 Ascend C 算子；验证 `GetWorkspaceSize`、workspace 分配、executor、stream、执行接口和返回码。 | `tests/operators/<op_name>/routes/test_aclnn_<op_name>.cpp` |
| Ascend C `<<<>>>` | 仅用于 Ascend C 算子；验证 tiling data、block dim、workspace 和 launch 成功。 | `tests/operators/<op_name>/routes/test_direct_<op_name>.cpp` |
| `fla_npu.ops.triton` | 仅用于 Triton 算子；作为主测试方式，承担精度、泛化、边界、功能分支和回归用例。 | `tests/operators/<op_name>/accuracy/test_<op_name>.py` |
| `torch.ops.npu` | 可选；实现该入口时验证显式加载和调用成功。 | `tests/operators/<op_name>/routes/test_legacy_<op_name>.py` |

### 7.4 JSON 用例格式

`tests/op_cases/<op_name>.json` 是该算子的用例清单。主精度、各调用通路、性能和 ST 测试均按标签读取同一份 JSON，不得各自复制一套 shape 列表。

推荐 JSON 结构：

```json
{
  "op": "op_name",
  "schema_version": 1,
  "implementation": "ascendc",
  "capability": {
    "soc": ["ascend910b", "ascend910_93", "ascend950"],
    "dtypes": ["float16", "bfloat16"],
    "layouts": ["BNSD"],
    "routes": ["ascendc", "aclnn", "direct_launch"]
  },
  "tolerances": {
    "default": {"rtol": 0.001, "atol": 0.001}
  },
  "cases": [
    {
      "id": "basic_001",
      "tags": ["accuracy", "generalization", "route", "a2", "a3", "a5"],
      "shape": {"B": 2, "T": 2048, "H": 8, "K": 128, "V": 128, "chunk_size": 64},
      "dtype": "float16",
      "layout": "BNSD",
      "attrs": {},
      "soc": ["ascend910b", "ascend910_93", "ascend950"],
      "run_on": ["ascendc", "aclnn", "direct_launch"],
      "reference": "torch",
      "expect": {"return_code": "ACL_SUCCESS"}
    }
  ]
}
```

字段约定：

- `capability` 描述算子公开支持范围。
- `implementation` 只能填写 `ascendc` 或 `triton`；前者对应 `fla_npu.ops.ascendc`，后者对应 `fla_npu.ops.triton`。
- 示例 JSON 展示的是 Ascend C 算子；Triton 算子应将 `implementation` 改为 `triton`，并将 `capability.routes` 和 `run_on` 改为仅包含 `triton` 及实际实现的可选通路，不包含 aclnn 或 `direct_launch`。
- `cases[].tags` 至少使用 accuracy、generalization、boundary、negative、regression、performance、route、example 等标签描述用途。
- `soc` 写明该用例需要覆盖的平台，不能只依赖执行脚本默认值。
- `run_on` 写明需要执行该用例的调用通路；主精度和泛化矩阵必须包含与 `implementation` 对应的 `ascendc` 或 `triton`，不能同时把二者声明为该算子的公开入口。
- `reference` 写明精度标杆实现或数据来源。
- 负向用例必须写明 `expect.return_code`、期望异常或关键错误文本。
- 性能用例必须标注对比基准、目标 shape 和是否替换 Triton。

### 7.5 精度与结果判定

精度测试必须满足：

- 标杆实现、输入生成范围、随机种子、比较公式、`rtol`/`atol` 或其他阈值在 JSON 或测试 README 中可追溯。
- 全部公开输出分别比较，不能只比较主输出；无语义的输出区域应在算子 README 中明确后再处理。
- 检查输出中的 NaN/Inf，并验证相同输入重复执行结果符合算子确定性约定。
- 精度失败必须定位 kernel、标杆或公开语义问题，不能通过缩小输入范围、删除用例、降低覆盖或放宽阈值处理。
- 测试报告至少记录算子版本、平台、用例总数、通过数、失败用例 ID、精度结论和性能结论；不得提交原始大体积输出、日志或本地环境信息。

实现类型对应的 `fla_npu.ops.ascendc` 或 `fla_npu.ops.triton` 必须覆盖完整主测试矩阵；其他必选通路完成对应调用链验证。

### 7.6 三平台泛化验收

每个正向泛化 case 的 `soc` 必须同时列出 `ascend910b`、`ascend910_93` 和 `ascend950`，并通过 canonical accuracy 入口真实构造 JSON 中的 shape 后调用算子。只解析 JSON、检查 tag 或把 case ID 传给不读取该字段的旧脚本，不算完成泛化测试。

安装当前平台构建产物后，分别执行：

```bash
python3 ci/run_operator_generalization.py --soc ascend910b --device 0
python3 ci/run_operator_generalization.py --soc ascend910_93 --device 0
python3 ci/run_operator_generalization.py --soc ascend950 --device 0
```

该入口逐算子筛选 `generalization`、`ascendc`、`ACLNN_SUCCESS` case，并通过 `FLA_NPU_CASE_IDS` 每次只运行一个
JSON case，检查 launch 成功、公开输出 shape 和 NaN/Inf。每个 case 必须设置超时并在超时时终止整个子进程组。
精度结论仍由各算子的主精度/reference 测试给出；泛化执行通过不能替代精度比较。任一平台编译失败、case 未执行或
输出不满足契约时，不得在 README、PR 或测试报告中写该平台已经验证通过。

主精度/reference backend 使用独立统一入口，逐算子隔离执行并汇总失败或超时：

```bash
python3 ci/run_operator_accuracy.py --soc ascend910b --device 0
python3 ci/run_operator_accuracy.py --soc ascend910_93 --device 0
python3 ci/run_operator_accuracy.py --soc ascend950 --device 0
```

每个 backend 必须显式检查比较结果并以非零返回码报告精度失败；只保存输出、打印误差或调用比较工具但忽略其 `success` 结果，不算通过。入口超时必须终止整个子进程组，不能遗留占用 CPU/NPU 的 runner。

仓库 NPU CI 按模式控制 `CI_RUN_OPERATOR_GENERALIZATION` 和 `CI_RUN_OPERATOR_ACCURACY`：`full` 未指定 `CI_OPS` 时运行全部
已注册算子的泛化和主精度矩阵；`quick` 只有显式指定 `CI_OPS` 时才运行所列算子的矩阵，未指定时跳过，避免把轻量验证退化为
全算子串行精度任务。容器入口必须透传这两个变量，`ci/run_checks.sh` 必须在安装当前 custom OPP 和 Python 适配后按开关运行
对应统一入口；不得只保留可手工执行的脚本。Example/ST 仍按 CI 配置独立执行，不能因跳过 quick 的全算子矩阵而一并跳过。

## 8. 文档和示例

### 8.1 文档归档路径

每个算子都必须在自身源码目录下归档 README、设计文档和一个统一 API 文档：

```text
fla/ops/ascendc/<domain>/<op_name>/
├── README.md
└── docs/
    ├── design.md
    └── api.md
```

如算子目录层级包含子模块，以实际算子根目录为准，但文件职责和命名应保持一致。不得再为 aclnn、Python、kernel 或 `torch.ops.npu` 分别创建 API 文档；所有接口说明和调用示例统一放入 `docs/api.md`。README 应链接设计文档和 API 文档。

统一模板位于：

```text
docs/templates/operator/
├── README.md
└── docs/
    ├── design.md
    └── api.md
```

新增算子必须以该目录中的模板为基础创建文档；不适用章节应说明原因或删除，不能保留未替换的占位符。模板中的“提示”和“示例”使用 `chunk_bwd_dv_local` 展示填写粒度，完成具体算子文档后应删除这些辅助内容。

模板入口：[算子 README 模板](templates/operator/README.md)、[设计文档模板](templates/operator/docs/design.md)、[统一 API 文档模板](templates/operator/docs/api.md)。

### 8.2 算子 README

每个算子的 `README.md` 至少包含：

- 算子功能、数学定义、输入输出关系和典型使用场景。
- 输入、输出和属性表，包括参数顺序、含义、required/optional、shape、dtype、format、取值范围、默认值和约束。
- Shape 使用 `B`、`H_k`、`H_v`、`T`、`K`、`V` 等符号变量表达，不在 Shape 中直接写固定数值；固定维度写入“已知限制”。枚举或离散属性必须在属性表的“取值范围”中完整列出合法值，平台差异和跨参数组合约束再补充到“已知限制”。
- A2/A3/A5 支持情况，以及各平台存在的限制或实现差异。
- 支持的 fixed/varlen、layout、状态输入输出、padding、无效区域和边界语义。
- 实现类型，以及对应的 `fla_npu.ops.ascendc` 或 `fla_npu.ops.triton` 入口概览。
- Ascend C 算子的 aclnn、`<<<>>>` 入口概览，以及可选 `torch.ops.npu`；如 Ascend C 替换 Triton，写明仓内 example 已迁移到 Ascend C。
- 精度标杆和阈值、性能测试范围、已知限制和不支持场景。
- 构建、安装、运行示例和执行测试的命令。
- `docs/design.md` 和 `docs/api.md` 的链接。
- 文档末尾的 Shape 变量语义附录，逐项解释 `B`、`H_k`、`H_v`、`T`、`K`、`V` 等符号。

### 8.3 算子设计文档

每个算子必须提供 `docs/design.md`，至少包含：

- 背景、目标、非目标和公开能力边界。
- 数学公式、输入输出语义、shape/layout/dtype/format 约束。
- 整体架构；Ascend C 算子说明 op_host、tiling、kernel、aclnn、`fla_npu` 各层职责和调用关系，Triton 算子说明 Python wrapper、Triton kernel、grid/config 和 launch 关系。
- Ascend C 算子说明 tiling 模板参数、任务划分、block dim、workspace、UB/L1/L0/GM 内存规划和尾块处理；Triton 算子说明对应的编译配置、任务划分和内存策略。
- A2/A3/A5 的实现策略、模板实例和平台差异。
- 流水、同步、buffer 生命周期及跨核依赖；Ascend C 算子如使用 tiling key，说明必须使用的原因和组合范围。
- 精度风险、数值稳定性处理、标杆选择和阈值依据。
- 性能目标、Triton 对比基线、关键 shape、主要瓶颈和优化方案。
- 测试矩阵与 `tests/op_cases/<op_name>.json` 的对应关系。
- 已知限制、异常拦截和后续演进计划。
- 公式和语义说明中指向算子 README Shape 变量附录的链接；`docs/design.md` 不重复维护符号表。

### 8.4 统一 API 文档

每个算子只提供一个 `docs/api.md`，统一说明该算子的全部公开接口和调用方式，不再单独维护 `aclnn<OpName>.md` 或其他通路专用文档。

`docs/api.md` 至少包含：

- API 总览：按实现类型列出 `fla_npu.ops.ascendc` 或 `fla_npu.ops.triton`；Ascend C 算子同时列出 aclnn、`<<<>>>`，实现时再列出可选的 `torch.ops.npu`。
- 每个 API 的完整签名、参数顺序、输入输出、属性、required/optional、shape、dtype、format、取值范围、默认值和约束。
- API 参数表中的 Shape 只使用符号变量，固定维度写入“已知限制”；枚举或离散属性的全部合法值写入属性表的“取值范围”。Shape 和接口语义处链接算子 README 的 Shape 变量附录，`docs/api.md` 不重复维护符号表。
- aclnn 的 `GetWorkspaceSize`、执行接口、workspace、executor、stream、异步执行、返回值和错误码，仅 Ascend C 算子需要。
- 与实现类型对应的 `fla_npu` 公开函数导入方式、参数和返回值；Ascend C 路径还需说明默认解耦调用约定。
- Ascend C `<<<>>>` 直调所需的 tiling data、block dim、workspace、stream 和参数顺序说明，仅 Ascend C 算子需要。
- A2/A3/A5 支持范围及平台差异。
- 各接口的限制、不支持场景和异常行为；“异常与返回码”必须逐项记录公开约束/触发条件、代码拦截位置、返回码或异常、错误信息关键字和 `tests/op_cases/<op_name>.json` 负向用例 ID。

“异常与返回码”必须与代码拦截双向一致：从文档逐项能够找到 `op_host`、InferShape、Tiling、aclnn 或 Python wrapper 中的对应拦截，从代码中的每个公开参数拒绝分支也能够反向找到文档条目。新增、删除或收紧约束时，代码拦截、错误日志、返回码、API 文档和负向测试必须在同一个改动中同步；不得只写文档限制而让非法输入进入 kernel，也不得让代码拒绝文档声明支持的输入。

### 8.5 调用示例

调用示例统一写入 `docs/api.md`，至少包含：

- 实现类型对应的 Python 入口：Ascend C 算子提供 `fla_npu.ops.ascendc` 示例，Triton 算子提供 `fla_npu.ops.triton` 示例。
- aclnn：仅 Ascend C 算子需要，必须在 API 文档内完整展开初始化 ACL、创建设备与 stream、构造 tensor、调用 `GetWorkspaceSize`、分配 workspace、执行算子、同步、输出回拷和资源释放代码；不得只链接仓内 example，也不得用文字步骤或伪代码替代。
- Ascend C `<<<>>>`：仅 Ascend C 算子需要，说明被发射函数的参数顺序，准备 tiling data、block dim、workspace 和 stream，使用 `op<<<blockDim, l2ctrl, stream>>>(...)` 发射，并同步、检查返回码和结果。
- `torch.ops.npu`：可选；仅当算子实现该入口时，给出显式加载和调用示例。

### 8.6 模型符号表一致性

同一模型或算法族下的所有算子必须使用同一套模型符号定义：

- 模型根目录 README 的“模型符号表”是该模型的权威来源，例如 GDN 使用 `fla/ops/ascendc/gdn/README.md`，KDA 使用 `fla/ops/ascendc/kda/README.md`。
- 每个算子只在自身 `README.md` 末尾保留 Shape 变量附录，注明模型/算法族、模型级符号表链接和符号表版本，并列出本算子实际使用的符号。
- `docs/design.md` 在数学公式和接口语义章节链接算子 README 附录；`docs/api.md` 在参数 Shape 和接口语义章节链接算子 README 附录。两者不得复制变量语义表。
- 同一模型内相同符号必须保持相同含义、大小写和下标形式。例如 GDN 各算子中的 `H_k` 不能在一个算子表示 Q/K head 数，在另一个算子表示单 head 特征维度。
- 不同语义必须使用不同符号，禁止通过局部文字重新解释已有模型符号。
- 修改模型符号名称或语义时，必须在同一个 PR 中同步模型根 README、所有受影响算子 README、公式、API Shape、JSON case 字段和测试代码。
- `tests/op_cases/<op_name>.json` 中的 Shape 字段名必须与所属模型符号表一致，不能为同一维度另起别名。

> **Hint：** GDN 和 KDA 分别维护自己的模型符号表。GDN 内所有相关算子共享一套 GDN 符号定义，KDA 内所有相关算子共享一套 KDA 符号定义；不要求 GDN 与 KDA 使用完全相同的符号集合，但同名符号不应表达相互冲突的基础语义。

示例必须是完整、可执行的最小调用，不得只给接口名或伪代码。新增算子时，README、`docs/design.md` 和 `docs/api.md` 必须与代码一起提交；修改接口、shape/dtype/layout/range、平台策略、tiling、workspace、同步或性能方案时，必须同步更新对应文档。

若新增 Ascend C 算子替换 Triton 算子：

- example 默认路径必须切到 Ascend C。
- 文档应说明 Ascend C 是推荐入口。
- Triton 实现如保留，应标注为参考、对照或兼容路径。
- 性能对比结果应作为 PR 验收信息，不得只写“理论更快”。

公开文档、PR、issue、评论和测试报告不得包含内网地址、机器名、用户名、绝对路径、临时目录、日志路径、token 或本地环境细节。

## 9. 交付检查清单

PR 提交前至少完成以下检查：

- [ ] 明确能力边界，代码、文档和测试一致。
- [ ] 已明确实现类型：Ascend C 算子支持 aclnn、`<<<>>>` 和 `fla_npu.ops.ascendc`；Triton 算子支持 `fla_npu.ops.triton`。
- [ ] 如保留 `torch.ops.npu`，仅作为显式加载的 legacy 可选入口。
- [ ] Ascend C 算子的 `op_host`、InferShape、Tiling、op_api、kernel、CMake 和 A2/A3/A5 配置已同步；Triton 算子的 wrapper、kernel、grid/config 和平台配置已同步。
- [ ] 实现类型对应的 `fla_npu` 适配层已同步；Ascend C 算子包括 raw wrapper、公开导出、输出分配、正反向绑定和打包入口。
- [ ] Ascend C 默认 Python 路径未新增强制 `torch_npu` import、`torch.ops.npu` dispatcher、`torch.ops.load_library()` 或 PyTorch C++ extension 依赖。
- [ ] 如涉及 legacy 入口，`torch_custom/fla_npu/*.yaml`、生成代码和显式加载路径已同步。
- [ ] A2 / A3 / A5 均可编译，并完成必要功能和精度验证。
- [ ] Ascend C 算子的 `--cce-auto-sync=off` 保持为 `off`。
- [ ] Ascend C 算子优先使用 tiling 模板化方案；如使用 tiling key，已说明必要性和覆盖范围。
- [ ] 公开接口、文档、example 和 JSON 未暴露内部 `stage` 编号；各 L0 以完整计算语义命名并拥有闭合契约。
- [ ] aclnn L2 未在核心 L0 前后或之间拼接独立 `Cast`；输入、阶段间和输出类型转换已融合进对应 kernel。
- [ ] 同一 L0 内存在多 phase 时，已说明 `SyncAll` 参与条件和 `pipe->Reset()` 前后的资源生命周期，并完成 A2/A3/A5 验证。
- [ ] 精度通过，且未通过缩小 range、跳 case、放宽阈值或 fallback 制造结论。
- [ ] 若 Ascend C 替换 Triton，目标场景性能优于 Triton，example 默认路径已切到 Ascend C。
- [ ] 泛化用例覆盖 shape、dtype、layout、SOC、边界、负向和真实上层调用。
- [ ] 用例设计已归一到 `tests/op_cases/<op_name>.json`，测试执行代码已归档到 `tests/operators/<op_name>/`。
- [ ] 算子源码、`torch_custom` 和 example 适配工程中没有残留主线算子的第二份用例目录或参数矩阵；历史 JSON 可从 manifest 逐字段物化。
- [ ] 实现类型对应的 `fla_npu.ops.ascendc` 或 `fla_npu.ops.triton` 覆盖主测试矩阵，其他必选通路完成链路验证。
- [ ] 算子 `README.md`、`docs/design.md` 和统一 `docs/api.md` 已新增或同步更新，未拆分 aclnn 专用文档。
- [ ] 新增算子文档已使用 `docs/templates/operator/` 模板，且不存在未替换占位符。
- [ ] Shape 已使用符号变量表达，固定取值已归入“已知限制”；只有算子 README 保留变量附录，设计文档和 API 文档已链接该附录且未复制符号表。
- [ ] 枚举或离散属性的全部合法值已在属性表“取值范围”中列出，未只写入“已知限制”或说明文字。
- [ ] 算子符号与所属模型根 README 的权威符号表一致；模型符号变更已同步所有受影响算子、JSON case 和测试。
- [ ] `docs/api.md` 已包含实现类型对应的 `fla_npu` 示例；Ascend C 算子还包含 aclnn、`<<<>>>` 示例，`torch.ops.npu` 实现时包含可选示例。
- [ ] `docs/api.md`“异常与返回码”已与代码拦截完成双向核对，每项约束均对应返回码、错误日志和 JSON 负向用例。
- [ ] Python 导出、schema、测试、仓内 Example/ST 和文档已同步。
- [ ] `git diff --check` 通过，且没有提交构建产物、日志、缓存或敏感信息。

## 10. 评审关注点

评审时应重点检查：

- 能力声明是否过宽；是否存在文档约束没有代码拦截，或代码已经拦截但文档仍声明支持的场景。
- 是否把 tiling key 用成普通参数分发。
- 是否通过自动同步、全局同步或 fallback 掩盖生命周期问题。
- 是否以整数 `stage` 将弱相关计算强行塞入同一个 L0，或让公共调用者理解内部阶段。
- L2 是否通过独立 `Cast` 连接核心 L0，而没有把类型转换融合到生产者/消费者 kernel 边界。
- 实现类型与 `fla_npu.ops.ascendc` / `fla_npu.ops.triton` 是否匹配，主测试是否覆盖实际实现。
- A2/A3/A5 的代码路径是否一致可维护，平台差异是否有测试。
- Ascend C 替换 Triton 是否真实带来性能收益，example 是否已经切换。
- JSON case 是否能作为唯一用例设计来源，脚本中是否仍散落未登记的关键 shape。
