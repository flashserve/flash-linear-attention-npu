# A2 GDN Phase 5 P0/P1 验收快照

> 验收日期：2026-07-30
> 范围：冻结 `V=128`、`K=128`，完成 `D + (E+F)` 融合及 Round2 调度优化
> 状态：P0 与 P1 Round2 已验收；Git 里程碑、tag 和交付归档已完成

## 1. 阶段边界

```text
Phase 4: (A+B+C) + D + (E+F)
Phase 5 P0: (A+B+C) + (D+E+F)
```

`D` 为 `RecomputeWUFwd`，`E+F` 为此前已经验收的 `ChunkGatedDeltaRuleFwdHO` 路径。
P0 保留 `w/u` 通过 GM workspace 交接，并沿用 Phase 4 已稳定的 H/O 实现模式。本阶段不修改
`ABC`、transpose/layout、`V=256`、原生 GVA、causal conv、RMSNorm、反向，也不新增运行时规格分支。

## 2. 功能与精度门禁

以下用例均在同一进程契约下与 Phase 4 对比。输出、有效 `A`、`g_cumsum`，以及存在时的
final state 均二进制一致；所有产出张量均为有限值。

| 用例 | 结果 |
| --- | --- |
| dense FP16/C64 | 通过 |
| dense BF16/C64 | 通过 |
| dense FP16/C128 | 通过 |
| varlen FP16/C64 | 通过 |
| initial/final state FP16/C64 | 通过 |

## 3. 性能门禁

测试使用 A2 设备 2，成对交替执行 NPU Event；预热 5 轮、实测 20 轮。下表三个 P0 性能契约
均使用 `T=128`：

| 用例 | Phase 4 中位数 | Phase 5 中位数 | 变化 | Phase 4 workspace | Phase 5 workspace |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense FP16/C64 | 1.04384 ms | 0.93200 ms | -10.71% | 91,497,472 B | 73,155,072 B |
| dense BF16/C64 | 0.90285 ms | 0.86039 ms | -4.70% | 91,497,472 B | 73,155,072 B |
| dense FP16/C128 | 1.30765 ms | 1.23015 ms | -5.93% | 97,540,608 B | 78,149,632 B |

FP16/C64 独立 profiler 显示设备任务数由 `7` 降至 `6`。这符合预期：原本分离启动的
`D` 与 `HO` 后缀被收敛为一次 Phase 5 融合启动。

首次 P1 预检复用了同一个 FP16/C64 独立 smoke。Phase 5 trace 中共有六个设备任务：已有的
transpose/cast/cumsum 前缀，加一个 `ChunkRecomputeWUFwdHO` 任务。融合后缀耗时为
`64.92 us`；Phase 4 的 `RecomputeWUFwd` 与 `ChunkGatedDeltaRuleFwdHO` 分别为
`26.94 us` 和 `43.12 us`。由于两份 trace 来自独立的干净进程样本，这组数值只用于判断方向，
不构成新的验收门禁。当前 kernel 仍将 `w/u` 写入 GM，并在 H 消费前等待；若要直接交接，
需要重新设计生产者/消费者映射，不能只删除 barrier。基于这次预检，后续开展了下述三轮
有界 P1 实验。

## 4. P1 有界优化验证

P1 保持 P0 的融合边界不变，没有新增 dtype、shape 或 layout 路由分支。

| 轮次 | 单变量改动 | 功能结果 | 性能结果 | 决策 |
| --- | --- | --- | --- | --- |
| 1 | 删除 recompute D 与 H 之间冗余的外层 barrier | FP16/C64 smoke 二进制一致且均为有限值 | 目标 kernel 小幅改善；没有稳定的端到端收益证据 | 保留，作为 Round2 的基础 |
| 2 | 将 Phase 5 recompute 工作从“一个 chunk 任务循环所有 value head”展平为 `chunk x value_head` 任务 | 五个 P0 契约全部二进制一致且均为有限值 | 目标 kernel 在 `T=128` 时由 `56.42` 降至 `55.40 us`，在 `T=1025` 时由 `163.84` 降至 `160.44 us`，提升约 `1.8%~2.1%`；完整归一化矩阵见下文 | **作为 P1 验收** |
| 3 | 两个 recompute 阶段使用独立 scratch buffer，并删除内部全核 barrier | FP16 smoke 通过，但 BF16/C64 二进制一致性失败，`output_max_abs=8.98e35` | 目标 kernel 快于 Round2，但正确性失败 | 拒绝并回退 |

Round2 的五用例门禁覆盖 dense FP16/C64、dense BF16/C64、dense FP16/C128、
varlen FP16/C64 和 initial/final-state FP16/C64。Round3 失败时，输出受到影响，但
`g_cumsum`、有效 `A` 和 final state 仍保持一致，因此该无 barrier 重叠被判定为真实的
时序竞争，而非容差问题。

Round2 不降低 P0 workspace。Round3 继续保持拒绝状态，不再开启 Round4。

### 4.1 Round2 全量整链路性能矩阵

P0 和 Round2 暴露同一个 Phase 5 ACLNN 符号；若不额外创建仅用于测量的算子分支，两者无法
在同一进程内并排加载。因此，每个干净包进程都通过交替 NPU Event，将各自的 Phase 5 与同一条
不可变 Phase 4 路径进行测量。P0 与 Round2 包进程在同一设备上按 AB/BA 顺序执行，报告的
增量计算公式为：

```text
(Round2_Phase5 / Round2_Phase4) / (P0_Phase5 / P0_Phase4) - 1
```

每个有效轮次预热 20 次，每个变体实测 200 个样本。首轮观察到明显的双峰进程后，在选择最终
矩阵前统一应用稳定性门禁：Phase 4 和 Phase 5 均须满足 `median/P90 >= 0.8`。任一进程未通过，
其所在的整组 P0/Round2 包轮次即判无效并重测。最终每个用例采用三个有效轮次，共计 48 份
有效 JSON 文件。

| 用例 | Round2 vs Phase4 整体中位数 | Round2 vs P0 中位数 | Round2 vs P0 P90 | 更快轮次 | P0/Round2 Phase5 workspace（相同） |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense BF16/C128 | -0.615% | -0.570% | -2.404% | 2/3 | 81,905,664 B |
| dense BF16/C64 | -1.579% | -0.741% | -1.206% | 2/3 | 78,746,112 B |
| dense FP16/C128 | -1.099% | +0.264% | +0.926% | 1/3 | 82,169,856 B |
| dense FP16/C64 | -1.547% | -0.591% | -1.389% | 3/3 | 79,010,304 B |
| varlen BF16/C128 | -1.110% | -0.218% | +0.827% | 2/3 | 79,478,784 B |
| varlen BF16/C64 | -1.780% | -0.003% | -1.677% | 2/3 | 74,746,368 B |
| varlen FP16/C128 | -2.994% | -2.207% | -1.667% | 2/3 | 79,478,784 B |
| varlen FP16/C64 | -1.702% | -1.212% | -0.574% | 2/3 | 74,746,368 B |

Round2 改善了 `7/8` 个用例的中位数，相对 P0 的矩阵中位增量为 `-0.581%`；P90 矩阵中位数
为 `-1.298%`，且 `7/8` 个用例在三轮中至少有两轮改善。唯一中位数为正的用例是
dense FP16/C128，增量为 `+0.264%`，低于 1%，同时其目标 kernel 已独立确认更快，因此判定为
测量噪声，而非生产性能回退。全部 48 份已接纳的性能 JSON 中，Phase 5 输出均为有限值。
P0 相对 Phase 4 的八个归一化用例中位数全部改善，幅度为 `0.496%~1.290%`；Round2 相对
Phase 4 的八个完整 GDN core ACLNN 用例也全部改善，幅度为 `0.615%~2.994%`，矩阵中位数为
`-1.563%`。这里的计时包含公共前缀 `(A+B+C)`，但不包含 GDN core ACLNN 外侧的 transpose、
causal conv 或 RMSNorm。

### 4.2 Round2 融合后缀局部 profiler 对比

为区分“完整 ACLNN 收益”和“融合边界自身收益”，补充仅覆盖下述设备任务的局部对比：

```text
Phase 4: RecomputeWUFwd + ChunkGatedDeltaRuleFwdHO = D + (E+F)
Round2:  ChunkRecomputeWUFwdHO                    = (D+E+F)
```

测试使用 A2 设备 4 和已验收的 Round2 vendor；dense FP16/C64，`V=128`、`K=128`、
`chunk_size=64`。每个规格执行三轮独立干净进程，按 AB/BA 顺序平衡 Phase 4 与 Round2；每个
进程先预热 5 次、Event 测量 10 次，再采集一份 profiler trace。Phase 4 局部耗时取同一 trace
中 `D` 与 `E+F` 两个串行 kernel 的任务时间之和，Round2 取融合 `(D+E+F)` kernel 的任务时间。

| 规格 | Phase 4 `D + (E+F)` 中位任务时间 | Round2 `(D+E+F)` 中位任务时间 | 中位任务时间变化 | 三轮延迟降幅 |
| --- | ---: | ---: | ---: | ---: |
| `T=128` | 68.961 us | 54.161 us | -21.462% | 11.175% ~ 27.047% |
| `T=1025` | 175.504 us | 164.723 us | -6.142% | 4.239% ~ 7.486% |

表中“中位任务时间变化”的负值表示延迟下降。两个规格的三轮结果均为 Round2 更快，且 12 份
profiler 报告的输出均为有限值。这说明融合边界
本身取得了明确的局部收益；进入完整 ACLNN 后，由于公共 `(A+B+C)` 前缀和其余固定任务共同占时，
最终整体收益被稀释为上一节的 `0.615%~2.994%`。

该局部表用于解释收益来源，不替代 4.1 节的正式整链路门禁：profiler 需要为每个 variant 使用
独立干净进程，因此这里报告三轮 AB/BA 的任务时间中位数，并不声称是同一进程内的 kernel 级
同步 A/B。

## 5. 证据与发布边界

- 精度 JSON：`.phase5_p0_accuracy_r2/`
- 成对性能 JSON：`.phase5_p0_perf_r1/`
- 独立 profiler trace：`.phase5_p0_perf_r1/phase5_one_aclnn_fused_recompute_wu_ho.json` 和
  `.phase5_p0_perf_r1/phase4_one_aclnn_fused_fwd_ho.json`
- A2 构建日志：`/opt/chw/phase5_p0_build_r11.log`
- A2 安装日志：`/opt/chw/phase5_p0_install_r5.log`
- 隔离 OPP 环境：`/opt/chw/phase5_p0_vendor_r5`
- Round2 精度 JSON：`.phase5_p0_accuracy_r2/phase5_p1_round2_accuracy_*_r1.json`
- Round2 成对性能/profiler 证据：`.phase5_p0_perf_r1/phase5_p1_round2_*`
- Round2 全量性能 JSON 与汇总：`.phase5_p1_round2_full_perf_r1/`
- Round2 全量性能归档 SHA256：`c474e0f19e880b67aae07878eec2d11486062519d7ff56dfc0df9caa67f6358b`
- Round2 融合后缀局部 profiler 与汇总：`.phase5_p1_round2_suffix_profile_r2/`
- Round2 融合后缀远端原始证据：`/opt/chw/phase5_p1_round2_suffix_profile_d4_r1`
- Round3 拒绝日志：`.phase5_p0_accuracy_r2/phase5_p1_round3_accuracy_bf16_c64_r1.log`
- 隔离的已接纳 Round2 vendor：`/opt/chw/phase5_p1_round2_vendor_r1`
- Round2 运行包 SHA256：`8dd6714f70a228253d842fe99c0a233691a4c2d0e383720ea7fed8a912ba1d9c`
- Round2 `libcust_opapi.so` SHA256：`2aff4ca34a3fce0f095919d1aa9ba266876125aa159f459b699ff5a18459bb2f`
- 仅保留为拒绝证据的 Round3 环境：`/opt/chw/phase5_p1_round3_vendor_r1`
- 归档前 clean build 证据：`/opt/chw/gdn-phase5-archive-clean-build-r1`；基于不可变
  `gdn-a2-phase4-pipeline-p1` 源码叠加当前 Phase 5 staged patch，ABI 单测 `11/11`
  （含 5 个 subtest）通过，Phase 1~5 和默认入口共 12 个 ACLNN 符号齐全，
  `ChunkRecomputeWUFwdHO` 的 `.o/.json` 各 8 份。
- clean run 包 SHA256：`f1d915bbb03b69b489f5e9ffd1391f1e8bae9a2f26a7ef55206986875ca445b5`；
  `libcust_opmaster_rt2.0.so` SHA256：
  `3c8cae5b9f61a42ac8935a84a0f87a90f111cc25b3913156cc29669a12a0fdde`；安装态
  `libcust_opapi.so` SHA256：
  `bf9f7d9d0165a551d838c2d4da54c2fb9320a7fee06131289e02fb97ddce8601`。
- clean 安装态最小回归在 A2 device 4 通过：dense FP16/C64 含 initial/final state
  和 dense BF16/C64 均与 Phase 4 bit-exact，且所有输出为有限值。BF16 用例专门覆盖了
  Round3 曾失败的正确性门禁。
- Phase 5 实现里程碑 commit：`8208f69e4bf359c3989823490121eb19dadfa157`，parent：
  `b2c69b6348bf1bc83fc5db56c2a15208d36fdf67`；annotated tag：`gdn-a2-phase5`，
  tag object：`7585de6416c102d68e9a5461d7bb31a84a6e4a66`。远端逐 SHA 回查确认 tag peeled
  commit 为上述里程碑，归档分支仅快进，Phase 2/3/4/P1 旧 tag 均未移动。

Phase 5 P1 在 Round2 收口验收并完成 Git/交付归档。P0 作为不可变证据保留，
不作为最终选定源码状态。下一步是 Phase 6 profiler 可行性闸门；`V=256`、
原生 GVA、transpose 和更外层算子仍属于后续独立门禁。
