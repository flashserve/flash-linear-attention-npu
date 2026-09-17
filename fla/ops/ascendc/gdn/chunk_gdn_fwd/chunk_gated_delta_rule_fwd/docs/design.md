# ChunkGatedDeltaRuleFwd Phase6 A2 局部设计

方案设计规则版本：V2

本文是既有 `ChunkGatedDeltaRuleFwd` 的局部实现设计，接口、支持范围和数学语义以
上级 [README](../README.md) 为准。本次只覆盖 A2（DAV_2201）Phase6 的 FP32 Solve
入口与私有 pipeline；不改变 host tiling、公开 ABI、其他架构或下游 H/O 计算。

## 1. 目标

- A2 的 `RunPhase6` 对所有合法 `BT=64/128`、dense/varlen、FP16/BF16、V128/V256、
  MHA/GVA 和既有 state 组合，统一执行
  `aWorkspace -> FP32 Solve -> A`。
- 删除 A2 BT64 varlen 的两次 BHT/TND 全量搬运及仅服务该搬运的外部 barrier。
- 私有 pipeline 固定使用 head-first `[B,H,T,W]` GM 布局；保留 `cu_seqlens` 任务描述、
  局部块偏移、现有 ready/free、阶段 barrier 和 `Run` 首尾同步。
- `BT=128` 的 64 层和 128 层 merge 是矩阵规模决定的算法路径，继续保留。
- 非 220 分支继续使用原有 TND fallback；A5 arch35、host/kernel ABI、locate 扫描、
  `full_convert` 和 ring 深度不在本次范围。

## 2. Stage 0–2 完整详设

### 全局符号与任务域

`B` 为物理 batch，`H` 为 `Hv`，`T` 为物理 token 数，`W` 为当前 Solve 矩阵宽，
`BT` 为 64 或 128。所有 Solve tensor 的 GM 元素偏移固定为：

```text
((b * H + h) * T + base + row) * W
```

行步长固定为 `W`。dense 任务域令 `sequences=0`，每个任务覆盖一条长度 `T` 的物理
序列；varlen 任务域按 `cu_seqlens[seq]` 和 `cu_seqlens[seq+1]` 得到 `base` 与
`length`，每个有效序列的每个 head 按 `span` 展开。`locate` 保持当前 grid-stride
所有权和顺序：同一 AIC 与两个 paired AIV 使用同一个本地 task 序号。

任务计数为：

```text
dense:  Q * H * ceil(T / span)
varlen: H * sum(ceil(Li / span))
```

其中 `span` 分别为 32、64、128。块内输入列仍用 `w.t % BT` 选择，避免非对齐序列
起点把全局 token 偏移当作块内偏移。

### Stage 0：Vector，KKT epilogue 产生 aWorkspace

KKT AIC 将 score 写入既有 `scoreWorkspace` 并发布 `SCORE_READY_FLAG`；KKT AIV
等待后执行 epilogue，将低精度系数写入 `aWorkspace`，其物理布局为
`[B,H,T,BT]` head-first。该阶段的 workspace offset、producer/consumer 和原有
同步不变。Stage 0 完成后由 Solve `Run` 的首部 `SyncAll` 统一建立 GM 可见性。

### Stage 1：Mixed，FP32 leaf/merge Solve

`GdnFp32Solve::Run<In, Out>` 使用 `aWorkspace` 作为 `raw` 输入；当输入不是 FP32
时，现有 `full_convert` 将连续元素写入 `solveFp32Input`，随后按原有阶段顺序执行：

1. `leaf_merge_pipeline<32>` 读取 `D16`，生成 `D32`。
2. `BT=64` 直接由 `stage64_merge<32, Out>` 写回 `A`。
3. `BT=128` 先由 `stage64_merge<32, float>` 生成 `D64`，再由
   `pipeline_merge<64, Out>` 写回 `A`。

每个 stage 继续使用既有 `solveD16Offset`、`solveD32Offset`、`solveD64Offset` 和
`solveWorkspaceOffset`。各 stage 的 local event、cross-core ready/free、双槽 scratch
和尾块 padding 不变；本次不新增 GM、UB、L1、L0、event 或 flag。

### Stage 2：Mixed，A 发布给下游 Recompute

Solve 末尾 `SyncAll` 完成最后的 AIV/MTE3 drain 后，`A` 已处于原有低精度
`[B,H,T,BT]` 输出布局，直接作为 Recompute 的输入。Phase6 不再在调用者与 Solve
之间插入 TND transpose 或额外的 staging barrier；下游 H/O 的同步仍按原实现执行。

### 同步与架构分支

`Run` 保持以下顺序：入口 `SyncAll`；必要的 FP32 conversion 后 `SyncAll`；leaf 后
`SyncAll`；`BT=128` 的 stage64 后 `SyncAll`；最终 A 写回后 `SyncAll`。pipeline 内部
的 ready/free flag 仍按原 slot 获取、消费、释放和回绕顺序闭合，零任务物理组也参加
既有 barrier。

DAV_2201 只编译并调用上述私有 pipeline。非 220 分支只在自身条件编译区域包含
`solve_layout_staging.h`、创建 TND 地址并保留原 `RunSolvePhase` fallback；这不改变
A3 行为，也不影响 A5 arch35。

### 验收边界

本次源码交付的静态检查包括：消费者唯一性搜索、条件编译分支检查、`git diff --check`
和源码/文档一致性检查。A2 构建、部署、精度、MSS、确定性和性能门禁依赖验证 Agent，
本设计不把未执行的硬件结果标为通过。
