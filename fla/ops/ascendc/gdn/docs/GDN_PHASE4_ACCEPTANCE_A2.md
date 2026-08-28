# A2 GDN Phase 4 验收快照

> 验收日期：2026-07-28
> 状态：冻结范围验收通过，Git 里程碑和 tag 已归档

## 1. 结论

Phase 4 在不可变 `aclnnGdnCoreFwdPhase3` 上融合 `fwd_h + fwd_o`：

```text
Phase 3: (A+B+C) + D + E + F
Phase 4: (A+B+C) + D + (E+F)
```

`E+F` 由单个 `ChunkGatedDeltaRuleFwdHO` MIX kernel 实现，并由版本化
`aclnnGdnCoreFwdPhase4` / `gdn_core_fwd_phase4` 调度。`h/v_new` 不再是 core executor
的中间 tensor，只存在于该 fused op 的 user workspace；首版仍为 `FwdH` 写 GM、
全核同步后 `FwdO` 读 GM，不宣称片上直通。

最终包通过 A2 干净构建、core dense/varlen `8/8` 加 state `1/1` 精度门禁、
dense/varlen `8/8` 生产性能矩阵、workspace/peak 和 profiler/kernel 数门禁。
密集 C128 两点的 median 差异在 `+0.44%~+0.68%`，配对中位差异为
`+0.38%~+0.46%`；目标 fused kernel 本体在 C128 profiler 中反而更快，因此按
AB/BA、长尾和 profiler 联合证据判为测量噪声范围，不启动无瓶颈假设的优化轮次。

## 2. 版本与产物

- 不可变基线：`gdn-a2-phase3^{commit}` =
  `7fb8f05b59ab56a8392e0f6c9bef071714894826`；
- Phase 4 Git 里程碑：`gdn-a2-phase4^{commit}` =
  `9719f2701f62ec7ef3d67751af52d1a1ea3c9435`；
- 硬件：A2，Ascend 910B3，验收使用 device 1；
- CANN：`9.1.0.beta1`；conda：`chw-py11`；
- 干净构建日志：`/opt/chw/gdn-phase4-pilot1-build-r2.log`，返回码 `0`；
- 完整 run 包：`fla-npu-fla_npu_linux-aarch64.run`，`8,236,673 B`，SHA256
  `a297168f3b5d14a09afd23acd060d3ab546bab9cbd71e7c8d301ebe3ce9b9206`；
- 安装 host 库：`libcust_opapi.so` / `libopapi.so`，均为 `527,736 B`，SHA256
  `6f67757282030b90f95f92f403beaf1a3bdeeee9cd52ccf9de87f58226c5a23d`，`cmp` 通过；
- 同包导出 `aclnnGdnCoreFwdPhase1/2/3/4` 的 workspace/launch 符号，
  `ChunkGatedDeltaRuleFwdHO` 的 8 组 `.o/.json` 均进入构建产物；
- 默认 `aclnnGdnCoreFwd` 继续保持 Phase 2 兼容行为，Phase 4 结论只使用
  版本化入口，不用默认别名代替历史 A/B。

## 3. 冻结路由

| 版本 | ACLNN / Python | 内部路径 |
| --- | --- | --- |
| Phase 3 | `aclnnGdnCoreFwdPhase3` / `gdn_core_fwd_phase3` | `ChunkCumsumKktSolveTri -> recompute_w_u -> fwd_h -> fwd_o` |
| Phase 4 | `aclnnGdnCoreFwdPhase4` / `gdn_core_fwd_phase4` | `ChunkCumsumKktSolveTri -> recompute_w_u -> ChunkGatedDeltaRuleFwdHO` |

Phase 4 不修改 Phase 1/2/3 的数学、ABI 或调度边界，不新增按 dtype、chunk、layout
或 shape 的生产分支。

## 4. 功能与精度

Phase 4 对同包 Phase 3 逐位比较：

| 矩阵 | 结果 | 覆盖 |
| --- | --- | --- |
| dense FP16/BF16 x C64/C128 | `4/4 PASS` | 整 chunk/尾块，`o/g_cumsum/valid-A` bit-exact，全部有限 |
| varlen FP16/BF16 x C64/C128 | `4/4 PASS` | canonical metadata 和多序列边界，同上 |
| 非空 initial/真实 final state | `1/1 PASS` | `o/final_state/g_cumsum/A` bit-exact，`max_abs=0` |

本地证据镜像为 `.phase4_dense_accuracy_r1/`、`.phase4_varlen_accuracy_r1/`、
`.phase4_smoke_r1.log` 和 `.phase4_state_smoke_r1.log`；原始执行位于 A2 `/opt/chw/`
对应的 `gdn-phase4-pilot1-*` 目录。

## 5. 完整 core 生产性能

比较对象为同包 `aclnnGdnCoreFwdPhase3` 与 `aclnnGdnCoreFwdPhase4`；正式运行关闭
`ASCEND_LAUNCH_BLOCKING`，使用 NPU Event、AB/BA 交替顺序和固定输入。首个
dense/varlen 代表点使用 2 轮干净进程、每变体合并 100 样本；剩余点使用
同进程 Phase 3/4 交替测量，dense 每点合并 3 个独立进程的 600 样本，
varlen 每点 200 样本。主判据为 median，P90 长尾不被隐去。

| case | Phase 3 median | Phase 4 median | median 变化 | P90 变化 |
| --- | ---: | ---: | ---: | ---: |
| dense FP16/C64 | `0.991740 ms` | `0.983540 ms` | `-0.83%` | `-1.52%` |
| dense BF16/C64 | `0.974270 ms` | `0.967570 ms` | `-0.69%` | `-1.95%` |
| dense FP16/C128 | `0.983820 ms` | `0.988150 ms` | `+0.44%` | `+1.36%` |
| dense BF16/C128 | `0.974930 ms` | `0.981580 ms` | `+0.68%` | `+2.70%` |
| varlen FP16/C64 | `0.950280 ms` | `0.905000 ms` | `-4.76%` | `-7.18%` |
| varlen BF16/C64 | `1.042500 ms` | `1.029260 ms` | `-1.27%` | `-0.15%` |
| varlen FP16/C128 | `0.962640 ms` | `0.946080 ms` | `-1.72%` | `-38.34%` |
| varlen BF16/C128 | `0.912510 ms` | `0.906950 ms` | `-0.61%` | `-1.45%` |

dense C128 在跨进程独立测量中曾显示 `2%~3%` 回退，但存在明显批次/顺序漂移。
三个新鲜同进程配对测量后，FP16/BF16 C128 合并 median 只差
`+0.44%/+0.68%`，逐轮配对 median 为 `+0.46%/+0.38%`；同时 profiler 的目标
kernel 为 `115.52 -> 112.24 us`。因此不将旧跨进程漂移解读成 kernel 回退。

## 6. Workspace 与 peak

| case | Phase 3 workspace | Phase 4 workspace | 变化 |
| --- | ---: | ---: | ---: |
| dense FP16/C64 | `137055744 B` | `104017920 B` | `-24.11%` |
| dense BF16/C64 | `136917504 B` | `103617536 B` | `-24.32%` |
| dense FP16/C128 | `152240640 B` | `113175040 B` | `-25.66%` |
| dense BF16/C128 | `151708160 B` | `112380416 B` | `-25.92%` |
| varlen FP16/BF16 C64 | `130229760 B` | `94570496 B` | `-27.38%` |
| varlen FP16/BF16 C128 | `143843840 B` | `103729664 B` | `-27.89%` |

8 个性能点的 workspace 全部下降；peak allocated delta 也全部下降。最大 Phase 4
workspace 为 `113175040 B`，仍高于绝对 `50 MB`；按已冻结规则，这是持续优化项，
不是压过性能的硬门槛。

## 7. Profiler 与真实融合

| 项目 | Phase 3 | Phase 4 |
| --- | ---: | ---: |
| 完整 core NPU 任务数 | `8` | `7` |
| C64 目标段 | `FwdH 98.00 + FwdO 44.18 = 142.18 us` | `FwdHO 146.38 us` |
| C128 目标段 | `FwdH 73.28 + FwdO 42.24 = 115.52 us` | `FwdHO 112.24 us` |

目标段从两个设备 kernel 降为一个，且 core executor 不再分配 `h/v_new` 中间
tensor，证明这不是仅减少外层 ACLNN 数量的假融合。C64/C128 的 kernel body 差异
方向不同，所以本验收只宣称完整 core 的生产性能和中间张量生命周期收益，
不外推为所有 shape 的 fused kernel 本体都更快。

## 8. 证据位置

- dense/varlen 精度：`.phase4_dense_accuracy_r1/`、`.phase4_varlen_accuracy_r1/`；
- dense 性能：`.phase4_perf_r1/`、`gdn-phase4-pilot1-dense-paired-r1/`；
- varlen 性能：`.phase4_varlen_perf_r1/`、`gdn-phase4-pilot1-varlen-paired-r1/`；
- profiler：`.phase4_perf_r1/traces/`、`.phase4_c128_profiler_r1/`；
- A2 原始路径：`/opt/chw/gdn-phase4-pilot1-*`。

上述原始 JSON、trace、临时 runner 和构建目录按归档规则不进入 Phase commit；
正式 benchmark 接口、可复现参数和汇总结论进入仓库。

## 9. 范围与后续

本次验收仅代表：

- A2 / Ascend 910B3，`K==V==128`；
- dense/varlen、FP16/BF16、C64/C128、尾块；
- 外部扩头后 core 内 `Hk==Hv`；
- 现有 head-first ND、transpose/layout 和 canonical varlen metadata；
- forward GDN core，含非空 initial state 与真实 final state 精度。

未完成且不得由本报告外推：`V=256`、原生 GVA、transpose/layout 融合、
causal conv、RMSNorm/gate、backward、完整 Demo/模型性能，以及单 ACLNN 绝对
workspace `<=50 MB`。

Phase 4 归档后不直接吸收 `D`。按冻结路线依次关闭 `V=256` 和原生 GVA
两个独立规格闸门，每次只引入一个变量；之后才能启动 Phase 5
`(A+B+C) + (D+E+F)`。
