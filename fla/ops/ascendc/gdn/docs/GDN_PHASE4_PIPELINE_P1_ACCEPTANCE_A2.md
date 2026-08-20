# A2 GDN Phase 4 流水 P1 验收快照

> 验收日期：2026-07-29
> 状态：P1 冻结范围验收通过，独立 Git checkpoint 已归档
> 关系：本报告是 Phase 4 的后续性能 checkpoint，不覆盖或改写 `GDN_PHASE4_ACCEPTANCE_A2.md`

## 1. 结论

Phase 4 流水 P1 保持 Phase 4 的融合边界不变：

```text
Phase 4/P1: (A+B+C) + D + (E+F)
```

P1 只优化 `E+F` 的执行编排：

- dense C64/C128 使用 `H AIV -> IB -> O AIV -> 本地 O AIC` 的 chunk-ready 流水；
- 其余已验收 shape 使用 H 生产核与 O 消费核的任务亲和调度；
- 移除 H/O 之间的外层 `SyncAll<false>()`；
- 不修改数学路径、transpose/layout、workspace 别名、`V=256` 或原生 GVA。

正式结果：

- dense `4/4`、varlen `4/4`、state `1/1` 与 Phase 3 bit-exact，全部 finite；额外 `B=4`
  和四段 varlen 压力点通过；
- 8 个正式性能点中 5 个相对 Phase 3 提升，3 个轻微回退，正差均小于 1%；
- 简单平均 latency 约从 `0.97959 ms` 降至 `0.97308 ms`，约提升 `0.66%`；
- P1 相对 Phase 0 的 8 点改善为 `76.25%~80.74%`；
- 8 个性能点 workspace 全部下降；profiler 目标段保持 `2 -> 1` 个 kernel，代表点的 fused
  kernel 本体更快。

因此 P1 满足 Phase 4 的性能、精度、有限性、workspace、安装和归档要求，不启动没有明确
profiler 瓶颈假设的第三轮优化。

## 2. 版本与产物

- 不可变 Phase 3 基线：`gdn-a2-phase3^{commit}` =
  `7fb8f05b59ab56a8392e0f6c9bef071714894826`；
- Phase 4 基线保持不变：`gdn-a2-phase4^{commit}` =
  `9719f2701f62ec7ef3d67751af52d1a1ea3c9435`；
- P1 实现 checkpoint：`gdn-a2-phase4-pipeline-p1^{commit}` =
  `57c3ba03a3c15a797cedb9a712f02d3957de94f2`；
- P1 annotated tag object：`f0c7bb5876ccf9bbd4565a8ae47e72d1f5a4ce33`；
- 归档分支：`gdn-a2-phase-archive`，远端为 `chw/gdn-a2-phase-archive`；
- 验收硬件：A2 / Ascend 910B3，正式 P1 矩阵使用 device 7；CANN `9.1.0.beta1`，conda
  `chw-py11`。

P1 clean 产物：

- run 包 SHA256：`5525c89fbf7563c34fa5375a666326448928b1b06dd8759de731291af9795c86`；
- `libcust_opapi.so` SHA256：`8904e82de928a6a0eb137ddfcd457430cff85ef0679e3d052d4abc18e77f8fda`；
- opmaster SHA256：`771393f67bf931c47b2e20a9627b84fd2d5eae11c1d62525b207ffb716bbd47b`；
- 从 Git tag 重打的完整 wheel SHA256：
  `f374213f88ae3fae73179543fc097cbb95c2ac1ed2a122146b473833681bc550`。

旧 Phase4 tag 未移动，P1 通过新 annotated tag 独立归档。

## 3. 冻结范围与接口

| 项目 | P1 验收口径 |
| --- | --- |
| ACLNN / Python | 沿用 `aclnnGdnCoreFwdPhase4` / `gdn_core_fwd_phase4` |
| 融合边界 | `ChunkCumsumKktSolveTri -> recompute_w_u -> ChunkGatedDeltaRuleFwdHO` |
| dtype | FP16、BF16；`G=float/state=float` |
| chunk/value | C64、C128；`K==V==128` |
| layout/metadata | 现有 head-first ND、transpose/contiguous 和 canonical varlen metadata |
| GVA | 仍由入口外部扩头，core 内验收 `Hk==Hv` |
| state | 非空 initial state 和真实 final state |
| 不在本 checkpoint | `V=256`、原生 GVA、causal conv、RMSNorm/gate、backward、workspace 别名 |

P1 保留后续 `V=256` 的 tiling/调度表达能力，但只作为扩展口，不在本 checkpoint 做完整
精度或性能对齐。

## 4. 功能、精度与有限性

clean kernel 和 opmaster 安装后完成以下门禁：

| 矩阵 | 结果 | 结论 |
| --- | --- | --- |
| dense FP16/BF16 x C64/C128 | `4/4 PASS` | `o/g_cumsum/valid-A` 对 Phase 3 bit-exact，全部 finite |
| varlen FP16/BF16 x C64/C128 | `4/4 PASS` | canonical metadata、多序列边界通过，全部 finite |
| 非空 initial/真实 final state | `1/1 PASS` | `o/final_state/g_cumsum/A` bit-exact，`max_abs=0` |
| 额外 `B=4` 压力点 | `PASS` | 输出和 state 通过 |
| 四段 varlen 压力点 | `PASS` | 输出和 state 通过 |

含空序列的额外 varlen 点在进入 P1 前已被 Phase 3 workspace 查询以 `169109` 拒绝，因此不
计入 P1 新路径失败。

## 5. Phase 3 -> P1 正式性能

正式矩阵使用同设备、同进程、同输入、AB/BA 平衡轮转；每个 case 为 3 个独立进程、每阶段
池化 600 个样本，主判据为 median。变化按 `(P1-Phase3)/Phase3` 计算，负数代表更快。

| case | Phase 3 median (ms) | P1 median (ms) | 相对变化 |
| --- | ---: | ---: | ---: |
| D_BF16_C128 | `0.97274` | `0.98193` | `+0.9448%` |
| D_BF16_C64 | `1.03652` | `1.02740` | `-0.8799%` |
| D_FP16_C128 | `1.04739` | `1.03032` | `-1.6298%` |
| D_FP16_C64 | `0.98625` | `0.96510` | `-2.1445%` |
| V_BF16_C128 | `0.95938` | `0.96372` | `+0.4524%` |
| V_BF16_C64 | `0.95002` | `0.94677` | `-0.3421%` |
| V_FP16_C128 | `0.93540` | `0.91848` | `-1.8089%` |
| V_FP16_C64 | `0.94898` | `0.95091` | `+0.2034%` |

3 个正差均小于 1%，结合 AB/BA、长尾和 profiler 证据判为测量噪声范围；未发现需要继续
优化的明确瓶颈。

## 6. Workspace 与 profiler

| case | Phase 3 workspace (B) | P1 workspace (B) | 下降 |
| --- | ---: | ---: | ---: |
| D_BF16_C128 | `151,708,160` | `112,380,416` | `25.92%` |
| D_BF16_C64 | `136,917,504` | `103,617,536` | `24.32%` |
| D_FP16_C128 | `152,240,640` | `113,175,040` | `25.66%` |
| D_FP16_C64 | `137,055,744` | `104,017,920` | `24.11%` |
| V_BF16_C128 | `143,843,840` | `103,729,664` | `27.89%` |
| V_BF16_C64 | `130,229,760` | `94,570,496` | `27.38%` |
| V_FP16_C128 | `143,843,840` | `103,729,664` | `27.89%` |
| V_FP16_C64 | `130,229,760` | `94,570,496` | `27.38%` |

8 个性能点 workspace 全部下降；P1 最大 workspace 为 `113,175,040 B`，绝对 `50 MB` 仍是
后续持续优化项，不作为当前融合路线的硬阻断。

代表性 profiler：

| case | Phase 3 目标段 | P1 目标段 | 目标段变化 |
| --- | ---: | ---: | ---: |
| D_FP16_C64 | `FwdH + FwdO = 137.963 us` | `FwdHO = 132.703 us` | `-3.81%` |
| D_BF16_C128 | `FwdH + FwdO = 107.302 us` | `FwdHO = 105.762 us` | `-1.44%` |

两个代表点的目标段均由两个 kernel 变为一个 fused kernel；完整 case 的微小正差来自外围
launch、调度和测量波动，不改变 P1 的收口结论。

## 7. 构建、安装与回归

- A2 全新源码副本执行 `bash build.sh --pkg --soc=ascend910b --vendor_name=fla_npu` 成功；
- clean 产物同时包含 FP16/BF16、tiling key 1/2 和 Phase 1~4 版本化 ACLNN 符号；
- wheel + run 成对安装后，Python wrapper 与 tag 源码一致，host/opmaster/HO 文件 hash 一致，
  `site-packages` 可直接 import `gdn_core_fwd_phase4`；
- 安装态 dense FP16/C64 对 Phase 3 bit-exact；正式 Example/ST varlen forward/backward
  精度检查通过；
- qk-l2norm example 的 Triton/CANN runtime 枚举名不兼容属于 Phase 4 冻结范围外的环境门，
  不作为 P1 kernel 阻断项。

## 8. 证据索引与后续

正式证据目录：

- P1 正式性能：`gdn-phase0-p1-safe-formal-summary-d7-r1/`；
- P1 profiler：`gdn-phase4-p1-profiler-d7-r1/`；
- P1 clean 构建与 kernel 矩阵：`gdn-phase4-p1-clean-full-build-r2/`、
  `gdn-phase4-p1-clean-artifact-matrix-d7-r1/`；
- P1 wheel/安装回归：`gdn-phase4-p1-git-wheel-build-r5/`、
  `gdn-phase4-p1-clean-install-regression-d7-r2/`。

P1 关闭后，下一生产小步是在冻结 `V=128` 口径下启动 Phase 5：

```text
(A+B+C) + (D+E+F)
```

`V=256` 和原生 GVA 是独立的后续规格扩展，不是 Phase 5 的前置门槛；正式声明支持时
再分别启动规格闸门，避免与融合改动同时进行。
