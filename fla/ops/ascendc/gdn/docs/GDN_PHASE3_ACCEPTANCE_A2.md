# A2 GDN Phase 3 验收报告

> 文档定位：Phase 3 关闭时的冻结验收快照，不承载当前进度。当前状态只见 `GDN_CURRENT_STATUS_A2.md`。

## 1. 结论

Phase 3 于 2026-07-26 在 A2（Ascend 910B3）按冻结范围完成工程验收。最终版本不是早期的
`ChunkCumsumKkt -> Cast -> SolveTri` 拆分候选，而是累积融合：

```text
raw FP32 g
    -> chunk-local FP32 forward cumsum
    -> KKT
    -> low-precision A hand-off
    -> solve_tri
```

该边界由单个 `ChunkCumsumKktSolveTri` MIX kernel 实现，并由版本化
`aclnnGdnCoreFwdPhase3` 调度。不可变 Phase 2 基线继续使用
`ChunkLocalCumsum -> ChunkKktSolveTri`；Phase 1、Phase 2 和默认入口没有被 Phase 3 原地覆盖。

最终包通过独立融合算子 `80/80` exact、GDN core dense/varlen `8/8` 加 state `1/1` 精度门禁、
共享 helper 局部微基准 `8/8`、完整 GDN core 生产性能 `8/8`、workspace/peak 和 profiler/kernel 数门禁。
Phase 3 在已验收矩阵上的主判据 median 均不劣于 Phase 2，完整 core NPU kernel 数 `9 -> 8`。

本次收口仍只代表冻结的 `K==V==128`、head-first ND、外部 GVA 扩头和现有 transpose/layout
边界。原生 `Hk != Hv`、`V=256`、causal conv、RMSNorm/gate、完整 Demo 性能、backward 及单 ACLNN
绝对 workspace `<=50 MB` 均未被纳入 Phase 3，也不由本报告推导为已完成。

## 2. 冻结实现与版本边界

### 2.1 固定入口

| Phase | ACLNN / Python | 固定内部路径 |
| --- | --- | --- |
| Phase 1 | `aclnnGdnCoreFwdPhase1` / `gdn_core_fwd_phase1` | `local_cumsum -> KKT -> cast -> solve_tri -> recompute_w_u -> fwd_h -> fwd_o` |
| Phase 2 | `aclnnGdnCoreFwdPhase2` / `gdn_core_fwd_phase2` | `local_cumsum -> ChunkKktSolveTri -> recompute_w_u -> fwd_h -> fwd_o` |
| Phase 3 | `aclnnGdnCoreFwdPhase3` / `gdn_core_fwd_phase3` | `ChunkCumsumKktSolveTri -> recompute_w_u -> fwd_h -> fwd_o` |

独立 `aclnnChunkCumsumKkt` 保留为 Phase 3 的局部验证入口，输出公开 FP32 `g_cumsum` 和 FP32
`A_raw`。core 最终路由不调用它，而是直接调用累积融合 `ChunkCumsumKktSolveTri` 并取得 FP32
`g_cumsum` 与已求解、同 `k` dtype 的 `A`。

### 2.2 保留和排除项

- 保留公开 `g_cumsum` GM 输出、FP32 score、低精度 A hand-off 和 per-core solve workspace；
- 保持 head-first ND、现有 transpose/contiguous、外部 GVA 扩头和 canonical varlen metadata；
- 不修改 Phase 1/2 的数学、布局、workspace 或版本化 ABI；
- 不做输出/workspace 原地别名；正确性继续依赖已验证的 MIX barrier 和 MTE3 staging 顺序；
- 不扩 `K != V`、`V=256`、原生 GVA、causal conv、RMSNorm/gate 或 backward。

## 3. 构建、安装与产物身份

- Phase 2 不可变代码基线：`gdn-a2-phase2^{commit}`（`f2a4b46`）；
- Phase 2 性能证据基线：commit `2b8161d`；
- Phase 3 不可变 Git 里程碑：`gdn-a2-phase3^{commit}` =
  `7fb8f05b59ab56a8392e0f6c9bef071714894826`；完整 tag 身份见 `GDN_PHASE_VERSION_ARCHIVE_A2.md`；
- 硬件：A2，Ascend 910B3；正式验证使用 device 1；
- CANN：`9.1.0.beta1`；conda 环境：`chw-py11`；
- 最终完整 run 包：`build_out/fla-npu-fla_npu_linux-aarch64.run`；
- run 包 SHA256：`837742c6731143ec0ea55c517338e0daca3d3f295b9b7a71f079805b9b62bfdb`；
- 安装 host 库 SHA256：`645336f9f65c522a06d74cf8b3df0ca6db2ae493fcd6fec980719eca7c8af07f`；
- 安装目录的兼容库名 `libcust_opapi.so` 和 `libopapi.so` 均为 `527,736 B`、SHA256 相同，远端
  `cmp` 逐字节通过；
- Phase 1/2/3 与独立 `ChunkCumsumKkt` 的执行及 `GetWorkspaceSize` 共 8 个 host 符号齐全；
- 独立/累积 Phase 3 的 4 个 device object、4 个 per-kernel JSON 和 2 个 config JSON 共 `10/10`
  与 CPack 逐字节一致；
- A2 ctypes ABI `11/11 PASS`，CPU contract `2/2 PASS`。

报告中的性能和精度均来自上述最终安装包，不混用早期 split、ping/pong 或 V_MTE2 候选包。

## 4. 功能与精度

### 4.1 独立 `ChunkCumsumKkt`

最终共享 MTE3 staging helper 上完成 8 个完整路由身份：

```text
dense/varlen × FP16/BF16 × C64/C128
```

每个身份运行 10 个冻结 contract，共 `80/80 exact PASS`。覆盖 `T=1/2/63/64/65/127/128/129/193/257`、
多 batch/多头、整 chunk、尾块、单/多序列、1-token 序列和序列边界。每例同时证明：

- 公开 FP32 `g_cumsum` 对独立 `ChunkLocalCumsum` 全张量逐位一致；
- FP32 `A_raw` 对两小算子 NPU 拼接基线全张量逐位一致；
- 所有输出有限；
- 对角、上三角、无效行列和尾块 padding 为零。

最终 8 个 exact 证据 SHA256（dense FP16/C64 为固定执行 wrapper，其余为运行日志）：

| identity | SHA256 |
| --- | --- |
| dense FP16/C64 | `d075a420b1cea81d8f0e471f5a7e74601b0570be1c2329b4cb92feccd638ab74` |
| dense BF16/C64 | `0d98164c825d57ce96f63a04fc2552d886ff1e3b911d705bf26b4116e8c72f8c` |
| dense FP16/C128 | `0c27b35391842be73552a9309f6cda7e0fc58a530ae4e94bc437d7780f629cdc` |
| dense BF16/C128 | `9517bd38b40d889151e3a7bf9e3c523f4ca613ccec4e021b7b027369bc2e945a` |
| varlen FP16/C64 | `a7ccc69b11ffc1b942147abb014c5465fb79aa4bbf10422e1e9280c8018b5198` |
| varlen BF16/C64 | `627a5f178ca7c326a58d0b552ec68244e15b0c48fc223c53ba8363d80d34c9f4` |
| varlen FP16/C128 | `6ab30c7e817f64317cc83b2fe0fbe795349a54143a2286335ad1e4f838039f90` |
| varlen BF16/C128 | `fcccf2efc527a329bd017a4ebfc5d120956520e12155a6c8f66fef9fb62e3cf3` |

### 4.2 `aclnnGdnCoreFwdPhase3`

最终 core 对不可变 Phase 2 完成：

| 矩阵 | 结果 | 比较内容 |
| --- | --- | --- |
| dense FP16/BF16 × C64/C128 | `4/4 PASS` | output、公开 `g_cumsum`、有效 solved A、final-state presence bit-exact/有限 |
| varlen FP16/BF16 × C64/C128 | `4/4 PASS` | 同上，并覆盖 canonical 序列边界 |
| 非空 initial/真实 final state | `1/1 PASS` | output、`g_cumsum`、有效 A、真实 final state bit-exact/有限 |

结构化摘要 SHA256：dense
`10ecd0b89ae8a2791cba09a8324b0f01973cd2e0c35eaa9d3d4cfcf26b02e521`，varlen
`95d7095a217fe4cc5209b211ef8df5f211ed8604748e6934bd1729c87f13d6d5`，state
`ecccd0ca365418bddb4423c8640193bd3d1d16183bcf1e973fcf19b13dc74fb0`。

## 5. 性能与显存

### 5.1 测量口径

- 正式性能关闭 `ASCEND_LAUNCH_BLOCKING`；
- 使用 NPU Event，warmup `10`、每个 variant 每进程采样 `50`；
- 共享 helper 局部微基准在同进程逐轮 AB/BA 交替；
- core 使用 4 轮 AB/BA、8 个干净子进程，每 Phase 聚合 `200` 个样本；
- 同一 case 使用同一输入、同一安装包和同一 device；
- 主判据为 median，同时记录 P90、min、workspace max/sum 与 peak allocated delta。

### 5.2 共享 helper 局部微基准（辅助证据）

基线为 `ChunkLocalCumsum -> ChunkScaledDotKkt` 两个 ACLNN，融合为一个
`ChunkCumsumKkt` ACLNN。公开输出在计时前通过 bit-exact/有限性检查，对外 ACLNN 数 `2 -> 1`。

这是 `local_cumsum + KKT`（A+B）共享实现的独立证据，用于证明该 helper 在接入累积融合前
既正确又有局部收益。它不包含 `solve_tri`，不是 Phase 3 最终路由，也不代替 5.3 的
完整 GDN core 生产性能结论。

| case | baseline median | fused median | median 变化 | baseline P90 | fused P90 | P90 变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense FP16/C64 | `0.942500 ms` | `0.640060 ms` | `-32.089%` | `1.272520 ms` | `0.662520 ms` | `-47.936%` |
| dense BF16/C64 | `1.071500 ms` | `0.728780 ms` | `-31.985%` | `1.532840 ms` | `0.855600 ms` | `-44.182%` |
| dense FP16/C128 | `1.120740 ms` | `0.758620 ms` | `-32.311%` | `1.526040 ms` | `1.169340 ms` | `-23.374%` |
| dense BF16/C128 | `1.062760 ms` | `0.715710 ms` | `-32.656%` | `1.096060 ms` | `0.735940 ms` | `-32.856%` |
| varlen FP16/C64 | `1.353440 ms` | `0.865230 ms` | `-36.072%` | `1.447820 ms` | `0.925440 ms` | `-36.080%` |
| varlen BF16/C64 | `1.261460 ms` | `0.839060 ms` | `-33.485%` | `1.391360 ms` | `0.917340 ms` | `-34.069%` |
| varlen FP16/C128 | `1.176600 ms` | `0.763910 ms` | `-35.075%` | `1.246180 ms` | `0.791780 ms` | `-36.463%` |
| varlen BF16/C128 | `1.177090 ms` | `0.764890 ms` | `-35.019%` | `1.210440 ms` | `0.784240 ms` | `-35.210%` |

8/8 median 与 P90 均改善。workspace max 持平；workspace sum 每例减少 `16,777,728 B`，peak
allocated delta 每例减少 `16,778,240 B`。最大 fused workspace max 为 `22,544,896 B`，最大
fused peak delta 为 `27,300,864 B`，均低于局部相对额外 `50 MB` 门槛。

原始证据位于 A2 `/opt/chw/gdn-phase3-staging-local-perf-r2`。八份 JSON 的 SHA256 为：

```text
D_BF16_C128  3c8a0efba9f1b2ee91e69c175081e4bcd32b009268c3923c7ada28d29337a34f
D_BF16_C64   4a51baec04675888ef2981c4dd119c8e4ad1c34d97661168381668b3c1c14adf
D_FP16_C128  70f90dc73d82643891542941834f33e2bddf1bf6b0ad6779d65936eddfb06207
D_FP16_C64   ac2058169fd621bef86a0cc5c6930187f16bcfb4a27b4cc80bcae7255e15e991
V_BF16_C128  2c0a0004580e564f6d555d7ed88a37eac20bb75dc34f09e2d75b7e14c0462b60
V_BF16_C64   30d4a46677a403c84c7255c97be717bb4c3fbd2ce0dc4b5d3a1ad3bf7cac3dd3
V_FP16_C128  65704f6b6e7071ad267a43d07f00fa3b24e93f39155fd26eda9a96266a5aa9af
V_FP16_C64   45b13dfc2232ab797e9d78cc92ba231487322f895cd59ecff308f72bd33b93ba
```

本地只读审计为 8 case、0 failure，摘要 SHA256
`a7665e632e6207db0bfd7b1a743b890cc7fc11b83291c46ef0e55b2b7a086a33`。

### 5.3 完整 GDN core 生产性能（Phase 2 vs Phase 3）

这是 Phase 3 的最终性能结论：在同一安装包中比较版本化 `aclnnGdnCoreFwdPhase2`
和 `aclnnGdnCoreFwdPhase3`，覆盖原六个 GDN core 逻辑阶段组成的完整 forward core 链路。
该口径不包含 causal conv、RMSNorm/gate 或完整 Demo/模型。

| case | Phase 2 median | Phase 3 median | median 变化 | P90 变化 |
| --- | ---: | ---: | ---: | ---: |
| dense FP16/C64 | `1.007370 ms` | `0.973270 ms` | `-3.385%` | `-0.922%` |
| dense BF16/C64 | `0.946660 ms` | `0.925350 ms` | `-2.251%` | `+5.305%` |
| dense FP16/C128 | `0.959930 ms` | `0.934970 ms` | `-2.600%` | `+1.623%` |
| dense BF16/C128 | `0.965750 ms` | `0.939910 ms` | `-2.676%` | `+2.336%` |
| varlen FP16/C64 | `0.931980 ms` | `0.897620 ms` | `-3.687%` | `-4.371%` |
| varlen BF16/C64 | `0.950730 ms` | `0.932310 ms` | `-1.937%` | `+0.202%` |
| varlen FP16/C128 | `0.941230 ms` | `0.921650 ms` | `-2.080%` | `-2.607%` |
| varlen BF16/C128 | `0.939280 ms` | `0.920690 ms` | `-1.979%` | `-0.712%` |

主判据 median `8/8` 改善。P90 中 dense BF16/C64、dense FP16/C128、dense BF16/C128 和 varlen
BF16/C64 回退 `0.202%–5.305%`；这些长尾结果被保留，不用 median 隐去。所有运行的 contract、
输出和状态有限。Phase 2/3 在每个相同 case 上 workspace max/sum 与 peak 完全相同，Phase 3 没有
引入相对额外显存。

dense FP16/C64 summary SHA256 为
`6ab6abb75e086da3af6d3b227fc4e6ad2deb461369f07d4fd0241141b293a5a6`，其余三个 dense identity
汇总为 `33bc664e8c8bd7444749a773cabd51c555362a08eb593dc16f64ca4575380380`；varlen 四点 summary
SHA256 为 `203034e59d162c62b608a49be1caec5be5bc5eb06ebeaf95e489ede47ca22731`。

完整 core 的绝对 workspace 约为 `130–152 MB`，仍大于 50 MB；因此本报告只能宣称相对 Phase 2
没有增加，不能宣称单 ACLNN 绝对 workspace `<=50 MB`。

## 6. Profiler 与真实融合证明

最终二进制在 `D_FP16_C64` 上分别以干净进程 profile：

| 项目 | Phase 2 | Phase 3 |
| --- | ---: | ---: |
| 完整 core NPU kernel 数 | `9` | `8` |
| device duration 合计 | `340.067 us` | `319.486 us` |
| 目标段 | `ChunkLocalCumsum 45.301 us + ChunkKktSolveTri 83.262 us` | `ChunkCumsumKktSolveTri 106.302 us` |
| 目标段合计 | `128.563 us` | `106.302 us` |

目标段改善 `22.260 us`（`17.314%`）。Phase 3 trace 不含独立 `ChunkLocalCumsum`、
`ChunkCumsumKkt`、`Cast` 或 `SolveTri`，证明最终路由是累积单 kernel 融合，而不是减少 ACLNN 数但
仍保留拆分 NPU kernel。

结构化 profiler summary SHA256 为
`ae726a91fb800de97e05e4632591b486c462abc74d1aa09d451a10abeffd27ac`；Phase 2/3 trace SHA256
分别为 `f8b6d42bcb3a8fef5da99b6ccbcd7996dfc5ff784eb3334ac8cf88af340f0077` 和
`69e65a04d29fa2062bce6cef17dcf2016f84e7e48f0ba55724b662a808791fa3`。

## 7. 失败候选与最终选择

Phase 3 曾验证并淘汰多个候选：早期 core 把 `local_cumsum + KKT` 融合后又拆出 Cast/SolveTri，
导致 kernel 数 `9 -> 10`、生产 median 回退 `3.682%`；V_MTE2 和 ping/pong 单变量候选也未通过
性能门禁。最终 MTE3 staging 方案在不改变数学顺序、公开输出、tiling/workspace 和 Phase ABI 的
前提下解决 cumsum 写回瓶颈，并在完整精度矩阵恢复后通过生产性能和 profiler 门禁。

这些失败只用于说明设计选择，不计入最终验收结果，也不与最终包哈希混用。

## 8. 范围边界与后续

Phase 3 已完成的范围：

- `K==V==128`；
- dense/varlen、FP16/BF16、C64/C128；
- 外部将 q/k 扩展到 value head 数的 GVA contract；
- 现有 head-first ND 和 transpose/layout；
- forward GDN core，含非空 initial state 与真实 final state；
- 共享 helper 的独立精度/局部微基准，以及完整 core 的精度、生产性能、workspace/peak 与 profiler。

未完成且不得由本报告外推的范围：

- 原生 `Hk != Hv`；
- `K=128,V=256` 或其他 K/V 组合；
- causal conv、RMSNorm、门控和完整 Demo 单 kernel；
- backward；
- 完整 Demo 生产性能；
- 单 ACLNN 绝对 workspace `<=50 MB`；
- Phase 4 及以后对 `recompute_w_u/fwd_h/fwd_o` 的吸收。

Phase 3 归档后，下一阶段只能从不可变 `gdn-a2-phase3` 里程碑追加新 commit 和新版本化入口，
不能移动 Phase 1/2/3 tag，也不能原地改变已验收的 Phase 3 融合边界。
