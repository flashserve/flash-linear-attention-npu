# ChunkKdaFwdIntraSubChunk · 迭代记录

> 模型 case：`B=1, T=8192, H=HV=32, K=128, BT=64, bf16`  
> 门禁：裸 `msprof` Task Duration **中位**（见 `MSPROF_GUIDE.md`）；Δ≤−0.05 ms 才 default on。  
> 硬目标：≤ **1.5 ms**（尚未达成）。  
> 当前已验证最优板端：**2.075 ms**（2026-07-25）。

---

## 1. 当前已验证最优（shipped）

| 层 | 配置 | 说明 |
|----|------|------|
| 流水 | Vec 2-win / 4 slot / raw 0x2 | `SetS0ReadyJoined`；SetFree 仅 Process bookend |
| Cube | **路径 A** | `L1A_DBUF=1` + `FIX_MTE2_DBUF=1`；`WIN_L1_RESIDENT=0`；Kg L1B 复用 |
| Vector Prep/Post | MTE2 合并 | S0：`mid‖q/k/g`；Post：`cmat‖beta`；各一次 Wait |
| Vector 卫生 | V-A | 去掉无依赖的冗余 `PipeBarrier<PIPE_V>` |
| FwdSub | 行广播 Mul + Add-fold | 无 brcd；两趟 col-tile Mul 后 **一次** BAR，再 fold；对角 `+I` 标量 |

**否决且已从 Vector 主路径删除**：per-col Mul barrier（V-B）、稀疏 partial Mul/fold（P2）、resident mask/+I（P3）、Post‖S0 MTE defer（V-D）。  
**Cube 保留但默认关**：`USE_SCORE_WIN_L1_RESIDENT`（精度 FAIL）。

---

## 2. 性能时间线（板端 Task Dur 中位）

| 阶段 | 日期 | Dur / wall | 相对 | 默认 | 摘要 |
|------|------|------------|------|------|------|
| B0 | 07-24 | **20.323 ms** | — | on | Tile GEMM + Kg L1B；W‖MMAD1（后因精度关掉单槽 overlap） |
| B1 | 07-24 | **7.589 ms** | −12.7 | on | FwdSub 向量化（去 O(i²) GetValue） |
| B2 | 07-24 | **4.610 ms** | −3.0 | on | `Brcb`→`Mul`→Add-fold；**Pattern::RA ~34.6 ms → reject** |
| C0 | 07-24 | wall ~3.04 | ~−1.6 wall | on | Vec 2-win dual-issue |
| P0 | 07-24 | — | — | — | 精度：关单槽 W‖MMAD；修 FIX/`PIPE_ALL` |
| P1 Cube | 07-24 | Task **~2.18 ms** / wall 2.97 | — | on | `L1A_DBUF`：MTE2(W)‖MMAD1 |
| C1 | 07-24 | **2.180 ms** | ≈0 vs P1 | on | Akk Fix ‖ 下 tile MTE2 |
| C2 | 07-24 | — | — | **off** | WIN L1 Prefetch：`aqk_err≈14` |
| V-A | 07-24 | **2.159 ms** | −0.021 | on | barrier 卫生 |
| V-C | 07-24 | **2.158 ms** | −0.001 | on | MTE2 merge |
| V-B / V-D | 07-24/25 | hang / 不稳 | — | **off→删** | 见 §3 |
| **FwdSub P1** | **07-25** | **2.075 ms** | **−0.083** | **on** | 行广播 Mul + 粗同步（当前最优） |
| P2 / P3 | 07-25 | — | — | **reject→删** | 见 §3 |

---

## 3. 否决清单（勿再 default on）

| 刀 | 现象 | 处置 |
|----|------|------|
| Pattern::RA / 库 ReduceSum on 16×16 | ~34.6 ms（远慢于 Add-fold） | 永久不用 |
| 单槽 `USE_SCORE_MMAD1_LOAD_W` | 多 HV aqk 抖 / fixp | 由 L1A_DBUF 取代 |
| C2 `WIN_L1_RESIDENT` | `aqk_err≈14` | 宏默认 0，代码保留于 cube |
| V-B per-col Mul+BAR | sim 好 UB2UB，板端 hang / 不稳 | 算法被 P1 粗同步取代后删除 |
| V-D Post MTE3 defer | suite 可过，裸 msprof hang | 删除 |
| P2 partial Mul/fold | suite 偶过；multi-iter aicore timeout；sim tick 回退 | 删除 |
| P3 resident mask/+I | 未过板端门禁；sim MOVEMASK 仅 ~3% | 删除 |

---

## 4. 仿真画像（T=1024，已验证最优）

产物目录：`prof_msprof_op_sim_t1024_p1_coarse`。

| 指标 | VA+VC (旧 brcd) | **当前最优 (coarse Mul)** |
|------|-----------------|---------------------------|
| Total tick | 157825 | **97847** |
| vec_med | 19.56 µs | **18.12 µs** |
| UB2UB% | 8.4% | **0.4%** |
| BAR%（cycle 占比，全 veccore） | ~60% | **~82.5%**（次数↓，但 fold 仍主导） |

解读：主矛盾仍是 **AIV BAR / Add-fold 层间 RAW**；Cube 路径 A 已基本吃满。距 1.5 ms 约差 **0.57 ms**，需新结构刀，而非再叠卫生 barrier。

---

## 5. 关键提交锚点（本分支）

分支：`20260725_150726_chunk-kda-fwd-intra-sub-chunk-pr`（相对 `origin/main`，作者均为 Coding-Pangolin）。  
下列 hash 与 `git log --reverse --oneline origin/main..HEAD` **从旧到新**一致，对应 §2 时间线。

| commit | 阶段 | 说明 |
|--------|------|------|
| `bce0e7a` | feat | 算子初版（FwdSub lockstep） |
| `f0028bc` | B0 | Tile GEMM + Kg L1B |
| `bd82ded` | B0 | 稳定 W load / MMAD1；基线 ~20.3 ms |
| `e279e78` | B1 | FwdSub 向量化 → 7.59 ms |
| `e822a6b` | B2 | Mul + Add-fold → 4.61 ms |
| `ab940ec` | C0 | Vec 2-win dual-issue |
| `cd9dd85` | P1+C1 Cube | 路径 A defaults；WIN L1 off |
| `ad63719` | docs | Phase F ship notes（950 / API / path A） |
| `f922b35` | docs | WIN L1 resident 标为 optional off |
| `5a0327f` | V-A | 合并冗余 Vector PipeBarrier → 2.159 ms |
| `081be05` | V-C | MTE2 merge on；V-B/V-D 宏 off → 2.158 ms |
| `349d299` | V-B | ScaleRowsByBeta slim（default off / 后否决） |
| `4094141` | **FwdSub P1** | 行广播 Mul + 粗同步 → **Dur 2.075 ms**（当前最优） |
| `00a89a6` | docs | DESIGN / ITERATION_LOG 对齐；剔除已否决 Vector 实验路径 |
| `d8a2f44` | docs | 同步 §5 提交锚点与本分支 git log |
| `d84c933` | docs | §5 补记 anchor-sync 提交 |

> 若 tip 仍有更新，以 `git log -1 --format=%h origin/main..HEAD` 为准。

明细测量目录与 PipeUtilization 见 §2–§4 与 `MSPROF_GUIDE.md`。

---

## 6. 采集与构建约定

- 构建：`FLA_NPU_SOC=ascend910b FLA_NPU_OPS=chunk_kda_fwd_intra_sub_chunk python -m pip wheel --no-build-isolation --no-deps . -w dist`
- 板端：`MSPROF_GUIDE.md`（勿单独使用会吞 interpreter 的 `msprof … -- python`）
- 仿真：`msprof op simulator`，`FLA_SIM_T=1024`，需 simulator `LD_LIBRARY_PATH`
- aicore timeout 后可能楔住设备；HCCS 联复位会重启全部卡，注意避让长任务

---

## 7. 后续方向（未开工）

1. 恢复/确认板端后，以 **2.075 ms** 为基线复采。  
2. 要逼近 1.5 ms：针对 **fold/BAR 结构** 做架构级方案（非再合并 Prep barrier；勿开 RA/ReduceSum）。  
3. C2 WIN L1：仅在精度修好并过 Dur 门禁后才可考虑。
