# 内存规划与数据通路

本文描述与具体算子和 SOC 无关的 workspace、resident、缓存、清零和中间量落点设计。

## 从生命周期选择内存层级

不要先决定“放 L1”或“走 workspace”，而要先回答：

- 数据由谁生产、由谁消费。
- 复用范围是 tile、stage、head、head group、chunk 还是 task。
- 最后一次消费发生在哪条 pipe。
- 需要的 dtype、layout、对齐和容量是多少。
- 目标 SOC 是否支持所需直连通路。

一般选择如下：

| 使用方式 | 候选落点 |
|---|---|
| 跨 chunk carry | FP32 GM workspace，当前 row 在 UB |
| AIC 产生、AIV 紧接消费 | 目标 SOC 支持时使用 UB direct/CV，否则 GM workspace |
| AIV 产生、AIC 紧接消费 | 目标 SOC 支持时使用 L1 direct，否则 GM workspace |
| AIC 后续重复消费 | L1/L0 resident |
| AIV 后续重复消费的小 tensor | UB resident |
| 公开输出或长生命周期数据 | GM |

直连路径与 GM fallback 必须保持相同数学语义和 stage 边界。

## Workspace 是数据协议

workspace 不只是地址集合。每个 segment/slot 都要定义：

| 字段 | 含义 |
|---|---|
| address key | core、task、window bank、head offset、chunk 或 tile |
| producer/consumer | 唯一生产者和全部消费者 |
| dtype/layout | 实际存储精度、ND/NZ 和 padding |
| ready | 写入在哪条 pipe 完成后可见 |
| free | 最后一次读取何时完成 |
| reuse point | 哪个后续 window/task 可以覆盖 |

默认 4-head window 配合双 bank 时规划 8 个逻辑 per-head workspace slot。地址轮转和 ready/free 使用一致的 `bank/headOffset`，但不能把 workspace slot 与 UB、L1 或 L0 的物理 ping/pong 混为一谈。

混合 dtype workspace 按 byte 规划，再转换为对应类型指针。segment 起点满足真实 DataCopy、矩阵布局和原子访问的对齐要求。

## L1/L0 跨 Head/Group 驻留

跨 head/group 驻留是通用优化手段，与 SOC 名称无关。是否采用由数据复用键、容量和生命周期决定。

- 多个输出 head 共享只读输入时，按最小 head-group key 驻留 L1 或 L0。
- 同一 head 在多个 stage 重复使用的 operand 可以驻留 L1。
- resident 与 scratch 使用独立地址区和事件，不能借用短生命周期 scratch 后延长占用。
- L1 resident 在最后一次 L1->L0 搬运完成后释放。
- L0 resident 在最后一次 MMAD 消费完成后释放。
- L1A 和 L1B 的 resident、scratch 和双缓冲事件分别规划。
- 下一复用组覆盖前必须等待当前组最后消费者归还 free。

SOC 文档只记录容量、支持的数据通路和实现限制，不重新定义这套原则。

## 按最小依赖键缓存

中间量只依赖 `(task, chunk, key-head)` 时，不要按每个 value head 重复计算。先定义最小依赖键，再决定缓存范围：

- 缓存键必须包含所有影响结果的维度和属性。
- cache slot 的生命周期覆盖组内最后一次消费。
- 跨 window 复用前证明上一 bank 不再持有消费者。
- cache miss、tail group 和 group 切换路径使用相同语义。
- 缓存节省的计算要与新增 workspace、同步和容量成本一起评估。

## 小 Tensor 驻留

gate、scale、metadata、token 系数和重复使用的指数结果等小 tensor 可以驻留 UB：

- 在首次需要时搬入并转换到计算 dtype。
- 跨 stage 复用，避免重复 DataCopy、cast 和 exp。
- 按 4-head window 的 active head 分配或按共享键去重。
- resident slot 与普通 matrix input/output ping/pong 分离。
- tail 和变长场景只读取有效范围，padding lane 按消费者语义处理。

## 初始化与清零

- producer 完整覆盖消费者读取范围时，不做预清零。
- padding lane、尾块、对齐扩展区或跨 tile 残留会被读取时，只清零真实读取范围。
- 累加器首项能完整覆盖时直接写入，后续项再累加。
- 大输出整体清零按 core 划分连续范围，尾 core 处理剩余元素。
- 不允许通过清零隐藏未初始化读取或错误的生命周期。

## 消除 GM 中转

当目标 SOC 支持且容量、layout、生命周期可证明时，可评估：

- AIC 的矩阵结果直接写入 AIV 可消费的 UB slot。
- AIV 的中间矩阵直接写入 AIC 可消费的 L1 slot。
- AIC 的后续矩阵计算直接复用 L1/L0 resident。

引入直连时同步更新：

1. UB/L1/L0 容量和临时空间。
2. 每个 4-head window 的 per-head 或共享 slot。
3. 数据格式转换和真实物理跨度。
4. ready/free、tail drain 和 fallback。
5. sanitizer、流水和固定性能基线。
