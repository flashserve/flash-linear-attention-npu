# Double Buffer 支持评估（post-C0）

> 结论：**手写 `TileMmadTla` 完全可以开双缓冲**，不必切回 `BlockMmad` + `MmadPingpong`。  
> **详细落地方案：** [`SCORE_TILE_DBUF_PLAN.md`](./SCORE_TILE_DBUF_PLAN.md)（L1/L0 布局、事件表、P0–P3 伪代码与门禁）。  
> 参考：PR190 `prepare_wy_repr_bwd_cube.h` + `a2_a3_common_optimization_notes.md` §2–4。  
> 门禁：精度稳定后单变量开；Δ Task Dur ≥ 0.05 ms 才默认开。

## 0. 已有 vs 未开

| 层级 | 现状 | 机制 |
|------|------|------|
| GM score/cmat | **开** | `depth=4`，`slot=(w%2)*2+head`，`prefill=2` |
| Cube L1A 软件 DB | **P1 默认开** | `l1A[0]=Qg,l1A[1]=W`；MTE2(W)‖MMAD1；Wait W before Fix |
| Cube L0 软件 DB | **未开** | sibling P2 门禁未过；见 `SCORE_TILE_DBUF_PLAN` |
| `USE_SCORE_MMAD1_LOAD_W` | **关** | 单 L1A 覆盖写已否决（multi-HV aqk flake） |
| Catlass `MmadPingpong` | **声明未用** | 热路径是手写 Tile，不依赖该 policy |
| UB AscendC `TQue` | **未开** | 全 `TBuf` |

## 1. PR190 / 已有算子怎么开（手写 Tile）

PR190 Cube **不是**靠 BlockMmad 自动 pingpong，而是：

1. **L1 划槽**：`l1Scratch[2]` + 多组 `*Resident[2]`（`GetBufferByByte` 偏移）
2. **L0A/L0B 各 2 槽**：`l0A[2]` / `l0B[2]`，`curL0_ ^= 1`
3. **事件闭环**（每槽独立 ID）：
   - L1：`MTE1_MTE2`（可写）↔ `MTE2_MTE1`（可读）
   - L0：`M_MTE1`（可写）↔ `MTE1_M`（可算）
4. **仍用 `TileMmadTla`** 吃当前 `l0A[idx]/l0B[idx]`

要点（`KDA_FORWARD_LESSONS`）：只声明 ping/pong 变量、每步立刻 Wait → **仍是串行**；必须让 MTE2/MTE1 与 M/FIX **真重叠**。

代码锚点：

- L1/L0 实例化：`ref_pr190/.../prepare_wy_repr_bwd_cube.h:414–435`
- L0 双缓冲用法：`a2_a3_common_optimization_notes.md` §3（`curL0_` + 分槽事件）
- Vector UB 双缓冲：同文档 §4（matrix / beta-g / output 三组）

## 2. 落到本算子 Score Tile（16×16×K=128）

工作量小（单 tile 双 MMAD），DB 目标是 **藏 GM→L1 / L1→L0 / Fixpipe 气泡**，尤其 **同窗头1 相对头0、以及窗间**。

### A. L1A ping-pong（Qg / W）— **首选，对齐 PR190 scratch[2]**

```text
l1A[0] = Qg   l1A[1] = W
l1B     = Kg  (resident hold，跨 MMAD1/2，单槽即可)
```

流水：

```text
MTE2 Qg→l1A0 ‖ (prev) ...
MTE1 l1A0/l1B → L0 ; MTE2 W→l1A1 ‖ MMAD1(Qg@Kg)
Wait W ready（MTE2_MTE1）— 必须在 Fixpipe(Aqk) 之前
Fixpipe Aqk ; MTE1 l1A1/l1B → L0 ; MMAD2(W@Kg) ; Fixpipe Akk
```

相对今日 `MMAD1_LOAD_W`：从「单 L1A 覆盖写」升级为「双槽 + 分事件」，生命周期更清晰，也方便再叠 L0 DB。  
宏建议：`USE_SCORE_L1A_DBUF`（可先等价于修好 Wait 顺序的 MMAD1‖W）。

### B. L0A/L0B ping-pong — **次选，对齐 PR190 L0[2]**

同窗：MMAD1 用 `l0[0]`，同时把 W/Kg 灌进 `l0[1]`，MMAD2 立刻吃 `l0[1]`。  
跨头：头0 Fixpipe ‖ 头1 的 L1→L0（需 `FIX_MTE2` / `M_MTE1` 分 ID）。  
tile 很小，单窗内收益有限；**多头连打**时更有价值。

### C. Fixpipe ↔ 下一头 MTE2（sibling `USE_SCORE_FIX_MTE2_DBUF`）— **后置**

跨 `ComputeScoreTile` 挂起 Akk Fix。事件/状态机重；精度风险高。精度绿 + msprof 仍见 `aic_fixpipe` 钉死再试。

### D. 切 BlockMmad+MmadPingpong — **不优先**

PR190 证明手写 Tile + 软件 DB 足够。换框架回归面大，且会丢掉 Kg L1B hold 的显式控制。

### E. UB `TQue` / PR190 Vector 三组 DB — **AIV 侧另案**

与 Cube DB 解耦；看 C0 msprof `aiv_mte2/mte3` 再开。

## 3. L1 预算粗算（bf16，K=128, BC=16）

| 缓冲 | 单槽 | ×2 |
|------|------|-----|
| L1A (BC×K) | 4 KiB | 8 KiB |
| L1B Kg | 4 KiB | （可单槽 resident） |
| L0A/L0B | 各 4 KiB | 各 8 KiB |

远小于 512 KiB L1；**空间不是障碍，事件闭环才是**。

## 4. 落地顺序

```text
1) 精度：MMAD1‖W 禁止与 Fixpipe 重叠（Wait 提前）→ H32 stress + suite
2) msprof C0：aic_mte2 / fixpipe / aiv_* 
3) A：L1A[2] + 分事件（手写 Tile，PR190 scratch 形态）
4) 若 AIC 仍有 L1→L0 / 跨头气泡 → B：L0A/L0B[2]
5) C / UB-DB 仅在 3–4 不够时
```

## 5. 与 1.5 ms

DB 只削搬运气泡，不换骨架。主路径仍是 Vec2Win + Vector FwdSub。
