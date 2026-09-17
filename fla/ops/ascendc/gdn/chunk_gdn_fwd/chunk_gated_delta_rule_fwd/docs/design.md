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

本次模型 case 为 A2、BF16、逻辑 B=28/32、每序列 S=1024、Hq=16、Hv=32、K=V=128、
BT=64、非零初态并输出终态；dense 一次调用的物理形状为 B 条序列，varlen 一次调用的物理
B=1、T=B*S。对照为优化前 main `9c0ffc175da6f924492282e7c07046639ddfc267` 和
ops-transformer `ba18b81f534b78163df54240db54cc1ba12e0e87`。

历史 B32 的完整公开调用设备 kernel 合计为 dense 17.831 ms、varlen 33.617 ms、
ops-transformer 18.274 ms；该数字只用于设计，不代表当前候选已通过。工程验证目标为
B28/B32 BT64 varlen 分别进入同输入 dense 与 transformer 耗时的 10% 范围，dense 和
BT128 控制组无超过测量波动且超过 5% 的回退。每个 case 至少两轮同卡交错对照，预热
不少于 100 次且不少于 3 秒，采样不少于 100 次；按 msprof op_summary 的
Task Duration(us) 汇总一次公开调用的全部 kernel，并单列融合 kernel。

## 2. 受影响阶段详设

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
dense:  B * H * ceil(T / span)
varlen: H * sum(ceil(Li / span))
```

其中 `span` 分别为 32、64、128。块内输入列仍用 `w.t % BT` 选择，避免非对齐序列
起点把全局 token 偏移当作块内偏移。

以 B32/S1024 为例，dense 的 B=32,T=1024 与 varlen 的 B=1,T=32768 在 span=32/64/128
时分别有 32768/16384/8192 个任务。实际启动的 AIC 数由 tiling 与硬件决定，不能在测试
中硬编码物理核数。

### Stage 0：Vector，KKT epilogue 产生求解输入 aWorkspace

KKT AIC 将 score 写入既有 `scoreWorkspace` 并发布 `SCORE_READY_FLAG`；KKT AIV
等待后执行 epilogue，将低精度系数写入 `aWorkspace`，其物理布局为
`[B,H,T,BT]` head-first。该阶段的 workspace offset、producer/consumer 和原有
同步不变。Stage 0 完成后由 Solve `Run` 的首部 `SyncAll` 统一建立 GM 可见性。

### Stage 1：Vector，转换 FP32 输入

`GdnFp32Solve::Run<In, Out>` 使用 `aWorkspace` 作为 `raw` 输入；当输入不是 FP32
时，现有 `full_convert` 将连续元素写入 `solveFp32Input`，转换后的 mixed barrier
保证后续参与核可见。该步骤只改变存储 dtype，不改变逻辑顺序。

### Stage 2：Vector，生成 D16

`LeafProducer` 从 FP32 输入读取每个有效 16 行对角块，按原有三角递推生成 D16，写入
`solveD16Offset`。配对 AIV 处理各自半块，尾部无效行不写回。

### Stage 3：Cube，计算非对角块

记当前半块宽度为 s，按原有顺序执行 `M = D1 @ L10`、`P = M @ D0`，实际 shape
由 n0/n1 有效行数裁剪，原生 FP32 GEMM、HF32 关闭、Fixpipe 和 scratch/result ring
保持不变。

### Stage 4：Vector，装配各层逆矩阵并发布 A

AIV 组装对角块与 `-P`，上三角填零；最后一层按基线 CAST_RINT 转为公开 dtype 写 A。
BT64 使用 16->32->64，BT128 继续 16->32->64->128；文档拆分不表示新增阶段屏障。

1. `leaf_merge_pipeline<32>` 读取 `D16`，生成 `D32`。
2. `BT=64` 直接由 `stage64_merge<32, Out>` 写回 `A`。
3. `BT=128` 先由 `stage64_merge<32, float>` 生成 `D64`，再由
   `pipeline_merge<64, Out>` 写回 `A`。

每个 stage 继续使用既有 `solveD16Offset`、`solveD32Offset`、`solveD64Offset` 和
`solveWorkspaceOffset`。各 stage 的 local event、cross-core ready/free、双槽 scratch
和尾块 padding 不变；本次不新增 GM、UB、L1、L0、event 或 flag。

### 下游边界：A 发布给 Recompute

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

| 数据/物理区 | 生产者 -> 消费者 | 依赖与复用条件 |
| --- | --- | --- |
| scoreWorkspace | KKT AIC -> KKT AIV | 原 SCORE_READY_FLAG；仍是 KKT 必需区，不再别名为 staging |
| aWorkspace | KKT AIV -> full_convert AIV | Run 入口 mixed SyncAll 保护跨组 RAW |
| solveFp32Input | full_convert -> leaf/merge | 转换后 mixed SyncAll；Solve 完成前不覆盖 |
| D16/D32/D64 | 各层 AIV -> 后继 AIC/AIV | 既有 ready/free 和层间 barrier；BT64 不消费 D64 |
| scratch/result ring | AIC/FIX -> AIV/MTE2 -> 下一轮 AIC 写 | 保留 ready/free 配对、尾部 drain 和 WAR 保护 |
| A | 最后 merge AIV/MTE3 -> Recompute | Run 末尾 mixed SyncAll，包括零任务物理组 |

本次资源增量为零，host tiling 序列化和各 workspace offset 不变。FP32 输入、D16、D32
和仅 BT128 使用的 D64 分别按 R*BT*4、R*16*4、R*32*4、R*64*4 字节规划，R=B*H*T；
scratch/ring、UB/L1/L0 和事件分配沿用基线。

每次 staging 转置读写 R*BT*sizeof(InputT) 字节，取消两次转置减少四倍该大小的 GM
搬运。B32 BF16 BT64 对应 512 MiB；以上是优化空间，必须保持 BT64 不变做同输入 A/B。

FP32 计算顺序、cast 时机和有效区 mask 不变。关键风险是非对齐序列起点、短尾块与跨任务
地址，需对 A、gCumsum、o、final_state 做同输入对照，并覆盖冻结双标杆、确定性和 MSS。
合法空序列产生零个 Solve task；不因零任务绕过阶段同步。

### 验收边界

本次源码交付的静态检查包括：消费者唯一性搜索、条件编译分支检查、`git diff --check`
和源码/文档一致性检查。A2 构建、部署、精度、MSS、确定性和性能门禁依赖验证 Agent，
本设计不把未执行的硬件结果标为通过。
