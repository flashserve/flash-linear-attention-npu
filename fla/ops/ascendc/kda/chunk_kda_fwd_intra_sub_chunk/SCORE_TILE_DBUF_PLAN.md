# Score Tile 手写双缓冲详细方案（对齐 PR190）

> 范围：仅 **AIC `ComputeScoreTile`**（`Qg@Kg` / `W@Kg`），仍用手写 `TileMmadTla`。  
> 不改：Vec2Win CV 协议、Vector FwdSub、GM `depth=4`。  
> 参考：
>
> - PR190 Cube：`ref_pr190/.../prepare_wy_repr_bwd_cube.h`（L1 scratch/resident[2]、L0A/B[2]、`InitPipeFlags`）
> - 通用整理：`/workspace/fzy/code/kda/0723/a2_a3_common_optimization_notes.md` §2–3
> - 教训：`KDA_FORWARD_LESSONS.md`（DB 必须真重叠；禁空 pingpong）
> - 精度红线：W 的 MTE2 **不得**与 Fixpipe(Aqk) 重叠（已观测 H=32 间歇 `aqk_err≈7–13`）

---

## 1. 目标与非目标

### 1.1 目标

在 **不换 BlockMmad** 的前提下，用 PR190 同款「划槽 + 分事件 + TileMmad」实现：

| 代号 | 能力 | 期望掩盖 |
|------|------|----------|
| **P1** | L1A ping/pong（Qg / W）+ Kg L1B resident | MMAD1 ‖ MTE2(W) |
| **P2** | L0A/L0B ping/pong | MMAD1 ‖ MTE1(W→L0)；跨头 L1→L0 ‖ 上头发 Fix |
| **P3** | （可选）头间 Fix ‖ 下头 MTE2 | Fixpipe(Akk_h0) ‖ MTE2(Qg/Kg_h1) |

验收口径（与既有 perf 门禁一致）：

1. 精度：full suite + H=32/T=4096×≥15 无失败  
2. 性能：idle 卡 msprof Task Dur median，相对上一默认配置 **Δ ≤ −0.05 ms** 才 `default on`  
3. 可回退：每阶段独立宏，默认关（P1 修好 Wait 后若等价今日安全路径可保持开）

### 1.2 非目标

- 不引入 Cube MCH / SolveTri  
- 不加深 GM `scoreQueueDepth`（4→6）  
- 不在本方案内做 UB `TQue`（另案，看 AIV msprof）  
- 不把 `KdaDispatchPolicy=MmadPingpong` 当实现依赖（可删或保留声明）

---

## 2. 现状基线（串行点）

今日 `cube.h::ComputeScoreTile`（单 L1A / 单 L0）：

```text
MTE2 Kg→L1B ; MTE2 Qg→L1A
MTE1 → L0A/L0B
[可选] MTE2 W→L1A ‖ MMAD1          ← 单槽覆盖写
Wait W ; Fixpipe Aqk                 ← Wait 必须在 Fix 前（精度红线）
MTE1 W/Kg → L0 ; MMAD2 ; Fixpipe Akk
PIPE_ALL
```

瓶颈形态（待 msprof 确认）：小 tile 下 MAC 极轻，墙钟多在 **MTE2 / MTE1 / Fixpipe / 事件空等**。  
P1/P2 的价值在 **同窗双 MMAD 之间** 与 **同窗双头之间** 的气泡，而非「再开一条数学路径」。

---

## 3. 资源布局（对齐 PR190 划槽）

### 3.1 尺寸（锁死 DESIGN：BC=16，K≤128 热路径按 128）

```text
TILE_A_BYTES = BC * K * sizeof(T)     // 16*128*2 = 4096
TILE_B_BYTES = K * BC * sizeof(T)     // 同 4096（Kg 按 L1B 布局）
L0A_BYTES    = TILE_A_BYTES
L0B_BYTES    = TILE_B_BYTES
L0C          = 单槽（与 PR190 基线一致；本算子双 MMAD 串行写 C，暂不 L0C-DB）
```

### 3.2 L1 地图

```text
offset 0:              l1A[0]   // Qg  (scratch ping)
offset TILE_A_BYTES:   l1A[1]   // W   (scratch pong)
offset 2*TILE_A:       l1B[0]   // Kg  resident（单槽足够；跨 MMAD1/2 hold）
```

可选扩展（P3 跨头软预取时再开）：

```text
offset 2*TILE_A + TILE_B:  l1B[1]  // 预取下一头 Kg（默认不做）
```

`static_assert`：`L1_USED ≤ 512KiB`（本布局 ≪ 512KiB）。

### 3.3 L0 地图（P2）

```text
l0A[0], l0A[1] 各 L0A_BYTES
l0B[0], l0B[1] 各 L0B_BYTES
l0C 单槽
```

实例化形态（抄 PR190）：

```cpp
LocalTensor<ElementA> l1A[2] = {
    resource.l1Buf.template GetBufferByByte<ElementA>(0),
    resource.l1Buf.template GetBufferByByte<ElementA>(TILE_A_BYTES)};
LocalTensor<ElementB> l1B =
    resource.l1Buf.template GetBufferByByte<ElementB>(2 * TILE_A_BYTES);

LocalTensor<ElementA> l0A[2] = {
    resource.l0ABuf.template GetBufferByByte<ElementA>(0),
    resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_BYTES)};
LocalTensor<ElementB> l0B[2] = {
    resource.l0BBuf.template GetBufferByByte<ElementB>(0),
    resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_BYTES)};
LocalTensor<ElementC> l0C =
    resource.l0CBuf.template GetBufferByByte<ElementC>(0);
```

---

## 4. 事件协议（PR190 风格）

### 4.1 原则

| 缓冲 | 「可被 MTE2 写」 | 「可被 MTE1/M 读」 |
|------|------------------|-------------------|
| L1 槽 | Wait `MTE1_MTE2` → 写完 Set `MTE2_MTE1` | Wait `MTE2_MTE1` → 用完 Set `MTE1_MTE2` |
| L0 槽 | Wait `M_MTE1` → 写完 Set `MTE1_M` | Wait `MTE1_M` → 算完 Set `M_MTE1` |
| L0C / Fix | Wait `FIX_M`（若复用）/ `M_FIX` | Fix 完 Set `FIX_M` / `FIX_MTE2` |

Process 入口 **InitPipeFlags**（PR190 `InitPipeFlags`）：对每个 L1/L0 槽预先 `Set`「可写」侧，避免首轮死等。

### 4.2 建议事件 ID 表（避开 0/1 若仍担心 Catlass 残留；本路径无 BlockMmad 可用 0 起）

| ID | 用途 | 事件类型 |
|----|------|----------|
| 0 | L1A ping (Qg) | `MTE1_MTE2` / `MTE2_MTE1` |
| 1 | L1A pong (W) | 同上 |
| 2 | L1B Kg | 同上 |
| 0/2 | L0A ping/pong | `M_MTE1` |
| 1/3 | L0B ping/pong | `M_MTE1` |
| 0/1 | L0 ready ping/pong | `MTE1_M` |
| 0 | L0C / M↔FIX | `M_FIX` / `FIX_M` |
| 0/1 | FIX↔MTE2 ping/pong（P3） | `FIX_MTE2` |

**硬约束：**

1. 同一物理槽的 Set/Wait **成对、次数匹配**（含空头/早退路径）。  
2. **禁止**「W 的 MTE2 与 Aqk 的 Fixpipe 无屏障并发」——P1 在发 Fix(Aqk) 前必须 `Wait(MTE2_MTE1, L1A_W)`。  
3. CrossCore `s0Ready/cubeDone` 与 HardEvent ID 空间无关，勿混用语义。

---

## 5. P1 — L1A 双缓冲（必做第一刀）

### 5.1 语义

- `l1A[0]` 只承载 **Qg**，生命周期：MTE2 → MTE1→L0 →（可选）释放供下轮 Qg  
- `l1A[1]` 只承载 **W**，生命周期：在 Qg 已进 L0 后开始 MTE2，**MMAD1 期间完成**，Fix(Aqk) **之前** Wait 齐，再 MTE1→L0 供 MMAD2  
- `l1B`：**Kg 一次载入，MMAD1/2 共用**，MMAD2 的 L1→L0B 可重读同一 L1B（与今日 hold 一致）

相对今日 `USE_SCORE_MMAD1_LOAD_W=1`：

| | 今日 | P1 |
|--|------|-----|
| L1A | 单槽，W 覆盖 Qg | 双槽，无覆盖 |
| 事件 | `SCORE_EVT` + `SCORE_EVT_W` | 每槽独立 L1 事件 + InitPipeFlags |
| 精度 | Wait 提前后应稳 | 同红线，结构更清晰 |

### 5.2 伪代码（单 `ComputeScoreTile`）

```text
// --- load Kg + Qg ---
Wait(MTE1_MTE2, EVT_L1B)
copyGmToL1B(l1B, Kg);  Set(MTE2_MTE1, EVT_L1B)

Wait(MTE1_MTE2, EVT_L1A0)
copyGmToL1A(l1A[0], Qg);  Set(MTE2_MTE1, EVT_L1A0)

Wait(MTE2_MTE1, EVT_L1B); Wait(MTE2_MTE1, EVT_L1A0)
Wait(M_MTE1, EVT_L0A_*); Wait(M_MTE1, EVT_L0B_*)   // P1 可仍用单 L0；见 P2
copyL1ToL0B(l0B, l1B); copyL1ToL0A(l0A, l1A[0])
Set(MTE1_M, EVT_L0_READY)
Set(MTE1_MTE2, EVT_L1A0)   // Qg L1 槽可回收（数据已在 L0）

// --- MTE2(W) ‖ MMAD1 ---
Wait(MTE1_MTE2, EVT_L1A1)
copyGmToL1A(l1A[1], W);  Set(MTE2_MTE1, EVT_L1A1)

Wait(MTE1_M, EVT_L0_READY)
tileMmad(L0C, L0A, L0B, ...)           // Qg @ Kg
Set(M_MTE1, EVT_L0A_*); Set(M_MTE1, EVT_L0B_*)
Set(M_FIX, EVT_L0C)

Wait(MTE2_MTE1, EVT_L1A1)               // ★ 必须在 Fix(Aqk) 前
Wait(M_FIX, EVT_L0C)
copyL0CToGm(Aqk)
Set(FIX_M, EVT_L0C)                     // 或 FIX_MTE2，见下头策略

// --- MMAD2: W @ Kg ---
Wait(M_MTE1, ...); copyL1ToL0A(l0A, l1A[1]); copyL1ToL0B(l0B, l1B)
Set(MTE1_M, ...); Set(MTE1_MTE2, EVT_L1A1); Set(MTE1_MTE2, EVT_L1B) // 用完释放
Wait(MTE1_M, ...); tileMmad(...); Fixpipe(Akk)
```

### 5.3 宏与默认

```cpp
#ifndef USE_SCORE_L1A_DBUF
#define USE_SCORE_L1A_DBUF 0   // 落地验证后视门禁改 1
#endif
// 过渡：USE_SCORE_MMAD1_LOAD_W 在 L1A_DBUF=1 时忽略或强制走双槽路径
```

### 5.4 门禁

- 精度：suite + H32×15  
- 性能：相对「安全串行 W」（`MMAD1_LOAD_W=0`）应 ≤ 不退化；相对错误 ‖FIX 的旧代码不应再比精度  
- 若 vs 今日已修 Wait 的单槽路径 Δ≈0：仍建议合入双槽（可维护性 / 为 P2 铺路），性能门禁可放宽为「不回归 >0.05」

---

## 6. P2 — L0A/L0B 双缓冲（第二刀）

### 6.1 动机

P1 解决 **GM→L1(W) ‖ MMAD1**。  
P2 解决：**MMAD1 进行时预填 MMAD2 的 L0**，以及 **头0 Fix/MMAD 时预填头1 的 L0**。

### 6.2 同窗内流水（推荐）

```text
l0Idx0 = curL0_;  curL0_ ^= 1
L1→L0[l0Idx0](Qg,Kg); Set READY0;          // 不立刻 Wait READY0

Wait(MTE1_MTE2,L1A1); MTE2 W→l1A[1]; Set MTE2_MTE1

Wait READY0; MMAD1 on l0[l0Idx0]; Set M_MTE1(l0Idx0); Set M_FIX

// 与 MMAD1 尾 / Fix 前重叠：灌 MMAD2 的 L0
l0Idx1 = curL0_;  curL0_ ^= 1
Wait(MTE2_MTE1,L1A1)                        // ★ 仍在 Fix(Aqk) 前
Wait(M_MTE1, l0Idx1); L1→L0[l0Idx1](W,Kg); Set READY1

Wait M_FIX; Fixpipe(Aqk);                   // 此时 L0[l0Idx1] 已就绪或在飞

Wait READY1; MMAD2 on l0[l0Idx1]; Fixpipe(Akk); Set M_MTE1(l0Idx1)
```

注意：`Kg` 从同一 `l1B` 拷到 `l0B[0]` 与 `l0B[1]` 两次是允许的（L1B resident）；不要在 MMAD1 未释放 `l0B[0]` 前覆盖错误槽。

### 6.3 跨头（同窗 slot0→slot1）

`Process` 内连续两次 `ComputeScoreTile`：

```text
ComputeScoreTile(slot0): ... Fix Akk_h0 可「延迟 Wait FIX」
ComputeScoreTile(slot1): 开头 MTE2(Qg/Kg) ‖ 上一次未 Wait 的 FIX   // 即 P3 雏形
```

建议 **P2 先做同窗内 L0-DB**，跨头重叠放到 P3，避免一次改两维。

### 6.4 宏

```cpp
#ifndef USE_SCORE_L0_DBUF
#define USE_SCORE_L0_DBUF 0
#endif
// 依赖 USE_SCORE_L1A_DBUF==1（或内部强制）
```

---

## 7. P3 — 头间 Fix ‖ MTE2（可选，高风险）

对齐 sibling `USE_SCORE_FIX_MTE2_DBUF` / PR190 `EVENT_FIX_TO_MTE2_*`：

```text
state: akkFixPending_ + evt id
Tile(h):
  if (akkFixPending_) { /* 先发本头 MTE2 */ Wait(FIX_MTE2); akkFixPending_=false }
  ... MMAD1/2 ...
  Fixpipe(Akk); Set(FIX_MTE2); akkFixPending_=true  // 不在本函数末 Wait
Process 末 / 单头窗: drain Wait
```

**默认关。** 仅当 P1+P2 后 msprof 仍显示 `aic_fixpipe` 与 `aic_mte2` 互斥且占墙钟时再开。  
精度回归特征：大 HV 间歇 aqk 炸 → 立即关。

---

## 8. 与 Vec2Win / Vector 的边界

```text
AIV: Prefill S0×2 → [WaitCube → Post ‖ S0(w+2)]×W
AIC: WaitS0 → Score(h0)[, Score(h1)] → SetCubeDone
```

- DB **只发生在 AIC Score 内部与同窗双头之间**  
- **不**改变 `SetS0ReadyJoined` / SetFree bookend  
- AIV Post 仍一次读齐 `cmatWs`；Cube 必须在 `SetCubeDone` 前保证两头 Fix 对 consumer 可见（P3 延迟 Wait 时：`SetCubeDone` 前必须 drain 未完成 FIX）

---

## 9. 代码改动面

| 文件 | 改动 |
|------|------|
| `op_kernel/chunk_kda_fwd_intra_sub_chunk_cube.h` | L1/L0 划槽、`InitPipeFlags`、重写 `ComputeScoreTile`、宏 |
| `PERF_ITER_LOG.md` | 记录 P1/P2 Dur / 精度 |
| `DOUBLE_BUFFER_EVAL.md` | 指向本方案（摘要保留） |
| tiling / Vector | **无**（P1–P2） |

可选清理：`common.h` 中未使用的 `KdaDispatchPolicy` 注释标明「非热路径」。

---

## 10. 分阶段落地与门禁表

| Phase | 内容 | 宏默认 | 精度门禁 | 性能门禁 |
|-------|------|--------|----------|----------|
| **P0** | 修 Wait：禁 W-MTE2‖Fix(Aqk) | （行为修正） | H32 stress 全绿 | — |
| **P1** | L1A[2]+Kg resident+分事件 | `L1A_DBUF=0→门禁后1` | suite+stress | 不回归 >0.05；力争 −0.05 |
| **P2** | L0A/B[2] 同窗 | `L0_DBUF=0` | 同上 | Δ≤−0.05 才开 |
| **P3** | 头间 Fix‖MTE2 | `FIX_MTE2_DBUF=0` | 大 HV 加压 | 同上 |
| **P4** | （另案）UB DB | — | — | 看 AIV pipe |

每刀：**单变量**、可 `#if` 回退、先精度后 msprof。

---

## 11. 验证计划

1. **功能**：现有 `test_npu_chunk_kda_fwd_intra_sub_chunk.py` 全绿  
2. **加压**：`H=32,T=4096` × ≥15；再 `T=8192` model-sample  
3. **性能**：`prof_chunk_kda_fwd_intra_sub_chunk_model.py` + msprof PipeUtilization  
   - 关注：`aic_mte2_ratio`、`aic_fixpipe`（若有）、`aic_mac_ratio`、TaskWait  
4. **仿真（可选）**：T=1024 看 MTE2/MMAD/FIX 时间线是否出现预期重叠  
5. **反例**：故意把 W-Wait 挪回 Fix 后 → 应复现大 HV aqk 毛刺（作为回归用例记在 PERF log，不必合入 CI）

---

## 12. 风险与回滚

| 风险 | 表现 | 处置 |
|------|------|------|
| 事件次数不配对 | hang / timeout 507014 | 查 InitPipeFlags / 空头路径 |
| W‖Fix | 间歇 aqk_err≈7–13 | 保证 Fix 前 Wait L1A_W |
| L0 槽用错 | akk/aqk 静默错 | 单变量 + dump 首窗 |
| DB 空转（步步 Wait） | Dur 不降 | 对照 PR190：Set READY 后先发下一 MTE2 再 Wait READY |
| 墙钟在 AIV | Cube DB 无感 | 停 P2/P3，转 AIV sync/scalar |

回滚：宏置 0 即回到串行 Tile；P0 Wait 修正 **保留**（精度修复，非优化）。

---

## 13. 推荐实施顺序（执行清单）

```text
[ ] P0 合并：Fix 前 Wait W（已改 cube.h）→ stress/suite 绿 → 可提交 fix 提交
[ ] msprof C0 基线（Task Dur + pipe）
[ ] P1 实现 L1A[2] + InitPipeFlags + 伪代码流水 → 精度 → msprof → 门禁默认
[ ] P2 L0[2] 同窗重叠 → 精度 → msprof → 门禁默认
[ ] 若缺口仍大且 AIC fix/mte2 钉死 → 评估 P3；否则转 AIV 刀（mid Sub / sync hygiene）
```

---

## 14. 与「到 2.7 / 1.5」的关系

- Sibling ~2.7 与我们的差距 **首先对齐 msprof 口径**；Cube DB 是 **可选 AIC 刀**，预期单刀多在 0.05–0.2 ms 量级（tile 小）。  
- **1.5 ms** 不能指望 P1–P3 单独达成；DB 方案是「在手写 Tile 骨架上按 PR190 挖搬运气泡」的工程路径，不是换算法。
