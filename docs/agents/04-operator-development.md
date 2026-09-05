# 算子开发

本阶段依据已确认的接口、CPU 标杆和评审通过的设计实现 Ascend C 算子。接口语义、支持范围、
Stage 方案和测试标准沿用前面阶段的结论。

## 阶段输入

- 算子目录中的 `docs/api.md`、记录当前规则版本的 `docs/design.md`，以及开发期
  `docs/validation.md`。
- 已确认的 CPU 标杆、输入生成方式、精度阈值和对齐用例。
- 与 chunk 依赖类型对应的开发参考：
  [`chunk-dependent-development.md`](reference/04-operator-development/chunk-dependent-development.md)
  或
  [`chunk-independent-development.md`](reference/04-operator-development/chunk-independent-development.md)。
- 仓库中相似算子的目录、注册、构建、API 封装和平台适配代码。
- 接口或功能修改的差异清单、兼容策略和原有场景回归范围。
- 优化任务的瓶颈结论、优化前性能、目标和固定的测试条件。

输入缺失或互相冲突时，返回产生该输入的阶段修正。`docs/design.md` 整体评审通过后才能开始编码。

## 实现总顺序

按以下依赖顺序实现，每一步完成并通过对应检查后再进入下一步：

1. 锁定本次实现范围并选择开发参考。
2. 实现算子定义、参数校验和错误处理。
3. 实现 host tiling、TilingKey、任务描述和 workspace 计算。
4. 按 `docs/design.md` 的顺序完成逐 Stage 实现与验证、调用层接入、整算子验证和必要的性能迭代。

实现中发现接口或支持范围需要变化时返回 `01`，CPU 标杆需要变化时返回 `02`，Stage、数据流、
地址、同步或资源规划需要变化时返回 `03`。前置文档更新并重新评审后再继续编码。

## 1. 锁定实现范围与参考

先根据任务类型确定本次允许修改的范围：

| 任务类型 | 实现边界 |
| --- | --- |
| 新增算子 | 实现 `docs/api.md` 和 `docs/design.md` 已确认的全部内容 |
| 修改接口或功能 | 只实现差异清单中的变化，保持其余接口、语义、支持场景和 CPU 标杆结果不变 |
| 修复已有功能 | 以现有接口、CPU 标杆和设计为准修正实现，不改变既定预期 |
| 性能优化 | 只实现已评审并确认采用的方案，保持公开接口、ABI、数学语义、支持范围和精度标准不变 |

再根据 `docs/design.md` 判断 chunk 依赖类型：

- 后一个 chunk 需要前一个 chunk 的状态或中间结果时，完整读取有依赖开发参考。
- 各 chunk 的输入在启动前已经完备且可独立完成时，完整读取无依赖开发参考。
- 混合场景按有依赖方式组织整体调度，其中独立 Stage 可以采用无依赖参考的任务划分。

新算子在本步骤首次查看仓内其他算子的代码和 README，只复用工程组织和通用实现方式。当前算子
的接口与支持范围以 `docs/api.md` 为准，公式、Stage、任务映射、资源和同步以 `docs/design.md`
为准；tile、窗口、slot 数、同步计数和资源常量必须重新从当前设计推导，不能直接照搬。

参考文档已覆盖所需细节时直接按文档实现；只有需要确认具体类、API 或代码组织时，才按参考文档
中的“定点查看当前实现”读取相关代码。采用的参考角色、版本、来源 commit、复用内容和本算子
差异记录到 `docs/validation.md`。参考文档版本变化时，既有算子评估变化内容并记录是否采用。

## 2. 实现算子定义与校验

1. 实现或更新算子定义、InferShape、参数校验和错误语义。
2. 参数名称、顺序、类型、默认值和可选项与 `docs/api.md` 一致。
3. 接口或功能修改同步更新所有受影响的定义层；未变化的调用保持兼容。
4. 修改生成文件前找到对应的 YAML、生成器或模板，修改源文件并重新生成，不直接维护生成结果。

本步骤先固定后续 host、kernel 和调用层共同遵守的参数契约，具体实现不得再自行增加、删除或
推导替代原有输入。

## 3. 实现 host tiling 与模板

1. 校验 shape、dtype、layout、属性、`cu_seqlens`、`chunk_indices` 和尾块约束。
2. 按设计生成 kernel 直接使用的任务描述、offset、有效长度、workspace 和模板字段。
3. 计算每个 workspace 区域的大小、offset、使用方和生命周期。
4. TilingKey 与选择条件一一对应，kernel 入口只选择设计中列出的模板实例；模板内部使用
   `if constexpr` 移除当前实例不需要的分支。
5. 支持范围外的组合由 host 明确拦截；不同 SoC 的模板、workspace 和内部信息由 tiling 传递。

## 4. 按顺序实现、验证与迭代

### 4.1 统一开发闭环

每个开发目标都按以下闭环完成：

1. 明确本轮目标和受影响的接口、Stage、数据、地址、资源与同步设计。
2. 只修改本轮目标所需的代码。
3. 按目标 SoC 和下述开发期构建方式构建并安装当前代码，确认运行时加载的是本轮构建结果。
4. 对照 `docs/api.md` 和 `docs/design.md` 检查本轮实现；需要改变前置结论时，返回对应阶段更新并
   重新评审。
5. 运行本轮对应的精度或接入检查，并回归已经通过但受本轮修改影响的目标。
6. 精度不一致时按本节的“精度问题处理”定位和修复，再使用相同条件重新检查。
7. 全部检查通过后，将代码位置、用例和结果记录到 `docs/validation.md`，再开始下一个目标。

开发目标依次为 Stage 0 到最后一个 Stage、调用层与平台适配、整算子，以及每轮性能优化改动。
前一个目标未完成闭环时，不得开始后一个目标。

#### 开发期构建与运行控制

首次建立开发环境时，在独立的 conda 环境中编译并安装一次完整 wheel，并确认当前 `python` 和
`pip` 来自该环境。多人共用同一套 CANN 时，每个开发者使用各自的 conda 环境。每轮开发按
改动范围选择构建方式：

1. 只修改现有接口下的 host、tiling、kernel 或算子配置时，构建并安装单算子 run 包：

   ```sh
   bash build.sh \
     --soc=<soc> \
     --pkg \
     --vendor_name=fla_npu \
     --ops=<算子名>

   ./build_out/fla_npu_linux-*.run --install
   ```

   多个算子使用逗号分隔：`--ops=<算子1>,<算子2>`。

2. 修改算子参数、aclnn/schema 接口、Python wrapper 或公开导出时，重新构建并安装完整 wheel：

   ```sh
   FLA_NPU_SOC=<soc> \
     python -m pip wheel --no-build-isolation --no-deps . -w dist

   WHEEL_PATH=$(ls -t dist/flash_linear_attention_npu-*.whl | head -n 1)
   python -m pip install \
     --force-reinstall \
     --no-cache-dir \
     --no-deps \
     "$WHEEL_PATH"
   ```

构建产物安装完成后，重启测试 Python 进程，并确认运行时加载的是本轮构建产物。构建和安装的
详细说明见 [`开发者指南`](../开发者指南.md)。

开发期执行单个用例时，确认已经进入 NPU kernel 后开始计时，连续执行 60 秒仍未返回即判定为
kernel 超时。此时终止对应测试进程；普通终止无法退出时强制结束，并确认残留进程已经清理、
设备资源已经释放。将超时用例、当前 Stage、TilingKey、`blockDim` 和超时现象记录到
`docs/validation.md`，排查循环退出条件、任务分发和同步逻辑。修复并重新验证通过后再进入
下一 Stage。编译、CPU 标杆计算和 ATK 正式验收分别使用各自的耗时与超时配置。

### 4.2 逐 Stage 实现 kernel

从 Stage 0 开始按顺序执行以下循环，不能同时实现多个尚未通过精度验证的 Stage：

1. 只实现当前 Stage 及其运行所必需的调度、搬运、写回和同步。
2. 构建测试版本，并按设计核对当前 Stage 的公式、任务映射、数据地址、资源和同步实现。
3. 运行从 Stage 0 到当前 Stage 的最小合法用例。使用 CPU 标杆按相同计算顺序生成当前 Stage
   的预期结果，比较当前 Stage 输出，并回归已经通过的 Stage。
4. 当前 Stage 及已完成 Stage 的精度全部通过后，将代码位置、用例和结果记录到
   `docs/validation.md`，再开始实现下一个 Stage。

每个 Stage 的 kernel 实现遵守以下要求：

1. 每个 Stage 的执行单元、公式、任务映射、数据地址和实际执行顺序与设计一致。
2. 矩阵计算使用适合的 Cube/Catlass 实现，Scalar/Vector 完成对应的非矩阵计算。
3. 性能关键路径优先连续整行或整 tile 搬运，并采用批量访问 API。
4. tail 和 partial chunk 优先使用 padding、中性值和有效区 mask 保持批量路径。
5. double buffer、MTE、VEC/CUBE、MTE3 和 cross-core 同步按设计实现；每个生产或通知事件都有
   对应的等待、消费和复用步骤。
6. workspace slot 覆盖前确认消费者已释放；空任务和 varlen 无效区仍遵守 ready/free 计数。
7. `PipeBarrier<PIPE_V>()` 只处理对应 pipe 内依赖，跨 pipe 依赖使用对应事件同步。

声明 ping/pong 已实现时，必须同时具有两份独立物理存储、各自的 ready/free 闭环，以及
profiling 中搬运与计算实际重叠的证据。

涉及同一 core 多 chunk、ping/pong、slot 回绕或 workspace 复用时，测试版 kernel 在 workspace
中使用独立区域记录实际执行过程。记录区域不得与算子数据重叠，各 core 分开写入，记录过程
不得增加新的同步。记录至少包含运行时 `blockDim`、每个 core 执行的 task/chunk、使用的 slot、
该 slot 的使用次数和完成的同步步骤，并证明同一 core 确实发生了设计要求的连续处理、切换、
回绕和复用。每个 ready/free 闭环覆盖首次使用、切换、回绕和尾任务，记录生产、等待、消费和
释放的顺序。不能只根据 shape 推断，也不能用多次启动 kernel 或不同 core 分别执行来代替。
执行记录只证明目标过程实际发生，不能代替后续测试；正常测试和性能测试必须关闭执行记录。

实现每个 Stage 时同步更新 `docs/validation.md` 中适用的追溯项：

| 设计项 | 代码位置与实际实现 | 开发期证据 |
| --- | --- | --- |
| Stage、任务映射和操作顺序 | 调度入口、循环和分派位置 | 覆盖该顺序和映射的实际运行记录 |
| L1/UB/GM 区间、份数和队列深度 | 分配位置、大小、深度和复用条件 | 容量检查、边界用例和复用结果 |
| ready/free、event、flag 和 barrier | 生产、等待、消费、复位和异常路径 | 首次使用、切换、回绕和尾任务记录 |
| ping/pong 和流水重叠 | 两份物理存储及生产消费代码 | profiling 中对应 pipe 的实际重叠 |
| workspace slot 生命周期 | slot 计算、写入、末次读取和释放位置 | 同一 core 连续多 task 和 slot 回绕记录 |

减少存储份数或队列深度、扩大 barrier 范围、合并 ready、删除 free 或改变 slot 复用时机都属于
设计变更，必须先返回 `03` 更新并评审设计。

### 4.3 接入调用层并验证整算子

所有 Stage 完成闭环后，按以下顺序接入并验证：

1. 同步实现 op_api/aclnn、schema、ctypes、Python wrapper、公开导出和平台适配。
2. 各层参数名称、顺序、类型及默认值与第 2 节固定的参数契约一致。
3. 同一 L0 定义和调用路径服务所有支持 SoC；平台差异放入 tiling、workspace、kernel 模板或
   架构 trait，不复制整套算子。
4. 可以复用相邻算子的有限模板选择、AIC/AIV 分工、矩阵计算、状态传播、workspace 生命周期、
   同步和 fixed/varlen 处理方式，但所有数量和条件仍以当前设计为准。
5. 构建并确认实际加载版本，检查调用层参数、平台选择和 kernel 入口与接口及设计一致。
6. 使用已确认的 CPU 标杆和精度阈值运行整算子最小合法用例。精度不一致时，先定位首个出现
   差异的 Stage，再按“精度问题处理”定位和修复。
7. 整算子精度通过并将接入位置、用例和结果记录到 `docs/validation.md` 后，才能进行性能优化
   或进入 [`05-operator-testing.md`](05-operator-testing.md)。

### 4.4 精度问题处理

单 Stage 或整算子的 NPU 结果与 CPU 预期不一致时，统一按
[`精度对比与定位.md`](reference/精度对比与定位.md) 的“共同准备”和“算子实现定位”执行。
定位、修复和回归均属于当前开发目标的闭环，精度重新通过后才能继续。

### 4.5 性能优化迭代

本节只用于性能优化任务。每轮只引入一项能够独立验证的改动；修改已有 Stage 时，从最早受影响
的 Stage 开始重新执行统一开发闭环，逐 Stage 验证到整算子，精度通过后再按固定条件测量性能。
改进点、profiling 结果和优化前后数据记录到 `docs/validation.md`。未达到目标时保留真实结果，
根据当前瓶颈设计下一轮；方案发生变化时先返回 `03` 更新并评审设计。

性能分析按主要耗时位置检查：

- Scalar：热路径标量访问和逐元素循环。
- MTE：搬运是否过碎、重复、非连续或粒度过小。
- VEC：是否承担应由 Cube 完成的矩阵工作，或 repeat 粒度过小。
- CUBE：tile、数据复用、矩阵形状和有效计算占比。
- AIC/AIV 等待：生产消费队列、flag、MTE3 写回、double buffer 和流水距离。

目标 shape 达到要求后，再进入 [`05-operator-testing.md`](05-operator-testing.md) 验证其他支持的
shape、dtype、layout 和 SoC。

## 阶段输出

进入 05 算子测试阶段前确认：

- 算子定义、host tiling、kernel、调用层和平台适配均已按实现顺序完成。
- 实现与 `docs/api.md`、CPU 标杆和 `docs/design.md` 一致，追溯表中的代码位置和开发期证据完整。
- 每个 Stage 均按顺序实现并通过精度检查，整算子最小合法用例也已通过；性能优化任务已记录
  每轮改动和结果。
- 新限制、风险或设计变化已更新到对应文档并完成评审。

完成后按 [`05-operator-testing.md`](05-operator-testing.md) 执行正式验证。算子测试阶段归档有效
结论并删除开发期 `docs/validation.md`。
