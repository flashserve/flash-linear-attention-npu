# chunk_bwd_dqkwg CV 深融合设计 (分支 cv_merge_new, 基线 new_dqkwg)

**本机无 CANN/NPU 头, 未编译/未测试。**

## 0. 现状 (重要)

经过多轮实验与诊断, 已确认:

1. **精度正确, 纯性能问题** (用户实测): 不是正确性 bug, 没有 denormal 等。
2. **new_dqkwg ≈ main (甚至更优)** (用户实测): 基线本身不慢。
3. 之前的"从零自研结构"(per-chunk 握手 + 合并 partA + 去 SyncAll + vector 4-stage 重构)
   **比 new_dqkwg 慢 ~70%**, 且**更换同步原语 (反转 flag ↔ raw 信用窗口4) 性能几乎无变化**
   => 瓶颈不在握手机制, 而在**结构本身破坏了 new_dqkwg 已有的良好 per-head 重叠**。

**结论**: 放弃自研结构。已将 5 个文件 (`common.h` / `cube.h` / `vector.h` / `tiling.{h,cpp}`)
**回退到 new_dqkwg**, 得到 ≈main 的无劣化基础。CV 深融合的**真正收益**只来自下面 §1 的最小改动。

## 1. CV 深融合优化方案 (待实现, 在 new_dqkwg 基础上的最小改动)

new_dqkwg 已是 5 段: `Part1(dw)`, `Part2(mm5 cube/ mul1 vector, 无握手)`, `Part3(ds)`,
`Part4+6(dq, 已融合)`, `Part5+7(dk, 已融合)`。每个**有握手的** part 内部已是 per-head 信用流水
(vector 预置 4 信用, cube 每 matmul `WaitCredit`→算→`SetReady`, vector 每 head `WaitReady`→处理→
`SetCredit`, part 末 cube drain 4), 且每 part 之间用 `SyncAll<false>()` 隔开。

**per-head 重叠已经很好**; 唯一的 CV 深融合收益 = **去掉 part 间 SyncAll 气泡**
(每个 SyncAll 强制 drain + 全局 barrier + 重新填流水, 估计 ~10-20% 总开销)。

### 改法 (连续信用流水)

- **去掉所有 `SyncAll<false>()`** (cube 4 处, vector 4 处)。
- **预置/drain 改为全局一次**: vector 只在最开头 (Part1 之前) 预置 4 信用; 删除 Part3/Part4/Part5
  的 per-part 预置和 cube 各 part 末的 drain 4。信用 flag (`SYNC_AIV_AIC_FLAG_0=3`/
  `SYNC_AIC_AIV_FLAG_0=5`) 跨 part 连续流动。
  - 计数平衡 (防死锁, 需静态核对): 每个有握手 part 内 cube `WaitCredit` 次数 == vector `SetCredit`
    次数 (Part1/3 = 1/head, Part4+6/5+7 = 2/head, 含 vector continue 分支的补发)。全局 cube 总
    `WaitCredit` == vector 总 `SetCredit`; 加一次全局预置 4 => 末尾余 4 信用 (无害, 无需 drain)。
  - `Part2` 无握手 (cube mm5 / vector mul1 自由跑) 是中间空档: 正确性由"全局窗口 ≤4 head + vector
    顺序处理"保证 (cube 领先 vector ≤4 head, 故 Part3 vector 读 mm5/mul1 时它们早已产出)。

### 必须同时修的 workspace 别名 (去 SyncAll 后才暴露)

new_dqkwg 靠 SyncAll 保证"全部消费完才被复用覆盖"; 去掉后, 以下 stride 不一致的复用在 BT≠K(=64)
时会数据竞争, **必须改为独立 workspace** (增加 HBM, 但保证正确):
- `mul1` 复用 `ptrDq` (stride BT vs dq 的 stride K) → 独立。
- `mm6` 复用 `mm5`/`dw`、`mm7` 复用 `mm5` (stride BT vs K) → 视 stride 是否一致决定是否独立;
  stride 一致 (都按 K) 且消费早于覆盖则可保留, 否则独立。

### 验证顺序

本地静态核对 (信用计数逐 part 平衡 + 花括号) → 远端编译 → 正确性 (CPU golden) →
性能 (对比 main / new_dqkwg)。先确认"≈ 基线无劣化", 再确认"去 SyncAll 拿到收益"。

## 2. 已确认有效/无害的改动 (可在优化时一并带上)

- host 用全部物理 AIC 核 `GetCoreNumAic()` (与基线一致, 不要因子限制核数)。
- `wsDgLast` 全工程只写不读 (Part1 写, Part5 本地重算不读) → 可删 Part1 的 dg_last 归约 (减 vector 负载)。

## 3. BT=128 小 kernel 劣化根因 (op_summary 实测定位, 2026-06-18)

case_10/12 的 op_summary pipe 分解 (cv 53d8037 vs new_dqkwg):
- `aic_mac` / `aic_mte1` **完全相同** → 矩阵乘法 FLOPs 与 L1→L0 搬运未变。
- `aic_mte2`(cube GM 读) **+44% (H=16) → +60% (H=32)**; `aic_fixpipe`(cube GM 写) **+24%→+37%**;
  `aiv_mte2`(vector GM 读) **+43%→+63%**; `aic_scalar`/`aiv_scalar` 大幅**下降** (cube 1.4k vs nd 10k)。
- 劣化 = 纯 GM/HBM 流量膨胀, **随 H 放大** (1.43×→1.61×) —— 这是判定根因的关键。

**根因: cube→vector 的 data-ready 握手粒度从 new_dqkwg 的 per-head 变成了 per-chunk。**
cube 算完一个 chunk 的**全部 H 个 head** 才 `SetCubeReady()` 一次; vector 必须等整块才能读 head 0。
此时 cube 已写出 H 个 head 的新数据, 把 head 0 的中间结果 (ds/dq_inner/dk_inner/mm6/mm7) 挤出 L2,
vector 的 MTE2 落到 HBM, 同时 cube 的写回(fixpipe)/再读与之争抢带宽 → 三条 GM pipe 同时膨胀。
H 越大挤得越狠, 故 case_12(H=32) 比 case_10(H=16) 更劣化。new_dqkwg 的 per-head 握手让
vector 紧跟 cube(差 ~1 head), 数据始终 L2-resident; 它"付出"的 scalar(per-head CrossCore flag)正是
cv 省掉的那部分 —— 用便宜的 scalar 换贵的 MTE2, 净赚。这也解释了为何 **"加深信用窗口 4→8 中性"**:
cube/vector 速率匹配(只领先 ~1 chunk), 窗口深度从不成为约束, 真正卡的是 per-chunk ready 耦合(读时数据已陈旧 H 个 head)。

**修复 (本次已实现, 待 NPU 验证): B/C/D stage 改回 per-head ready 握手, per-chunk 信用节流不变。**
- cube `chunk_bwd_dqkwg_cube.h`: B/C/D 的 `SetCubeReady()` 从 head 循环外(每 chunk 1 次)移入循环内(每 head 1 次)。
- vector `chunk_bwd_dqkwg_vector.h`: B/C/D 的 `WaitCubeReady()` 从 head 循环外移到循环体**最顶端**(在
  `++vecTaskIdx` / sub-block `continue` 之前 → 两个 AIV sub-block 各 wait H 次/chunk)。
- 计数平衡: cube `SetCubeReady` 每 chunk = {A:1, B/C/D:H}; 每 AIV sub-block `WaitCubeReady` 同 = {A:1, B/C/D:H};
  全局 cube 总 set == 每 sub-block 总 wait = M(1+3H)。`CrossCoreSetFlag<0x2>` 保持 1 set 对应每 sub-block 1 wait
  (与原 per-chunk 1:1 比例一致, 只是频率变 per-head) → 不会死锁。信用流(`WaitCredit`/`SetVecCredit`)完全未动。
- 只修了 B/C/D(单 head 循环, 同 stage 生产消费); **stage A 仍为 per-chunk**(cube 有 Part1/Part2 两个独立 head
  循环、vector 有常驻 mask, 需合并 head 循环, 更易出错)。若 B/C/D 验证后仍有残余劣化且定位到 A 的 dw 读, 再做 A。
- 预期: case_10/12 的 `aiv_mte2` 与 `aic_mte2/fixpipe` 应回落向 new_dqkwg 水平 (尤其 C/D 贡献部分)。

### 实测结果 (commit 2651550, B/C/D per-head)
- golden **104/104 PASS** (含 case_25 dq/dw 由 FAIL 转 PASS), 无死锁。
- 三个劣化 case 基本拉平: case_10 16.08→14.62ms(vs nd +12.3%→+2.1%), case_12 17.61→15.00ms(+20.3%→+2.5%),
  case_24 17.83→14.92ms(+21%→~0)。vs main(ff217ce) bf16 geomean 0.93(快 ~7%), 仅 case_06(H=4)+2.9% 微劣化。
- 残余 ~2% 来自 stage A 仍 per-chunk → 已补做 (见下)。

### stage A per-head (commit 待定, 已实现)
- A 同 stage 只有 dw(Part1 cube→Part1 vector)需要握手; mm5(Part2)由 B_vector 跨 stage 消费, 靠 "A 全 chunk 先于 B
  + credit 窗口" 保证就绪, 不需 ready 信号。dg_last/mul1 读输入或自算, 也不需。
- cube: `SetCubeReady()` 移入 Part1 head 循环(每 head 1 次), 删 Part2 后的整块通知。
- vector: `WaitCubeReady()` 移到 Part1 head 循环顶端(continue 前)。
- 计数: 全 stage 统一 cube SetCubeReady = 每 sub-block WaitCubeReady = 4H/chunk → 4HM, 平衡不死锁。

## 4. 进一步提速的可行性分析 (为何 ~20% vs main 难安全达成)

算子是 **memory-bound**, cube 端 **fixpipe-bound**(case_12 fixpipe ratio 0.99, MAC 仅 0.09)。每 (chunk,head) cube 必须
向 GM 写 **7 个中间结果**(dw/mm5/ds/dq_inner/dk_inner/mm6/mm7), 每个都要 vector 读回做逐元素门控/掩码后才得最终量。
- **L0C 累加(把成对 matmul 在累加器里合并, 省 1 写 1 读)被门控数学挡死**: dq = (dq_inner·exp(g)·scale) + mm6,
  dk = (dk_inner·exp(-g+g_last)) + mm7 — "inner" 项在加 mm6/mm7 前先被 exp 门控缩放, 故无法在 L0C 预先求和 →
  dq_inner/mm6、dk_inner/mm7 必须各自落 GM。(已读 C/D vector 确认。)
- **跨 stage 算子复用(q/k/v/do/h/dh 现各读 ~2 次, 减 cube mte2)**: 需把 stage-major 改成 chunk-major 把算子驻留 L1,
  会推翻整个 CV per-head 流水模型 → 高回归/正确性风险, 违背 "先保证不劣化"。且 cube 是 fixpipe-bound, 降 mte2 收益有限。
- **去掉 A/B 的 `PipeBarrier<PIPE_FIX>`**: A/B matmul 对象跨 head 复用同一 l0CBuf, 该 barrier 防 L0C 复用冲突, 盲删有正确性风险。
- **结论**: 7 次 fixpipe 写是本算法结构的下限, 门控不允许进一步融合 → **本结构的安全上限 ≈ new_dqkwg(约比 main 快 ~10-15%)**。
  真要 +20% 需算法级改动(把门控折进 cube 的 fixpipe/quant 通道以减少中间量, 或重算换带宽), 属研究性改动, 需单独立项 + 真机迭代验证。
- 小 H case 的 per-head 同步开销 > L2 收益(case_06 H=4 +2.9%): 若要严格零劣化, 可做 "H 大才 per-head" 的自适应粒度。

## 5. 实测 (commit d44fc6d, B/C/D + stage A per-head) 与小 H 劣化修复

### 实测 vs main (ff217ce/2807768, 同/跨 run)
- golden 104/104 PASS, 无死锁。fp16 用干净 main 2807768 同 run 对比 = **持平**(比值 0.96–1.004), 之前 fp16 9.5x 是污染坏点, 已排除。
- bf16 vs main: **15 个里 13 个比 main 快 7–15%**(case_10 0.91, case_12 0.85, case_24 0.85, case_08 0.91, case_20 0.89, case_26 0.88 ...)。
- **唯一比 main 慢的: case_06 (+2.7%) 与 case_25 (+22%, 但仅 0.1ms), 都是 H=4。**
- 残余: vs 历史 new_dqkwg(更快的 phased 内部分支, 非 main) case_10/12 仍 +2.5%/+2.75%, 来自 aic_mte2 +16%、aiv_mte2 +23%、
  aiv_mte3 +30%(per-head 把 cube/vector 压得更紧 -> HBM 读写争用上升; fixpipe 已被修到 ≈nd, -2.7%)。这是 **去 SyncAll 连续融合在
  memory-bound shape 上的固有争用代价**(new_dqkwg 有 SyncAll 分相, 争用低)。注意 run 间方差 ~7%, 该 2.5% 部分在噪声带内, 且 **不是 vs main 的劣化**。

### 小 H 劣化修复 (本次, 待验证): per-head 粒度按 H 自适应
- `common.h` 加 `DqkwgPerHeadMinH = 8`: 仅 H>=8 用 per-head; H<8 (如 H=4) 退回 per-chunk(整 chunk 一次握手)。
- cube 4 处 `SetCubeReady` 改 `if (params.H>=DqkwgPerHeadMinH || h==params.H-1)`; vector 4 处 `WaitCubeReady` 改 `if (H>=DqkwgPerHeadMinH || h==0)`。
- 两分支都是**已验证路径**: H>=8 = 本 run 验证过的 per-head; H<8 = 原基线 per-chunk(case_06 在老 per-chunk 下本就比 main 快)。计数各自平衡, 不死锁。
- 预期: case_06/25 回到 ≈/快于 main → **vs main 零劣化**; case_10/12/24 等大 H 保持 per-head 的快。
- 阈值 8 可调(H=8 的 case_14/18 在 per-head 下已快于 main, 故 per-head 下限取 8 安全)。
