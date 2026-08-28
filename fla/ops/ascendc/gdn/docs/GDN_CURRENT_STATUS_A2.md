# GDN 当前进度 A2

更新时间：2026-08-17（Asia/Shanghai）

## 新版 main Cumsum/KKT 三路线重新定基线（2026-08-17，当前规划）

- 当前新版基线固定为 upstream 仓库
  `https://github.com/flashserve/flash-linear-attention-npu.git` 的 `main` 分支提交
  `3434816c1da8082996962e61e070f9947541d89b`，冻结时间为 2026-08-17；提交说明为
  `fix(gdn): KKT tail precision and fixed BT128 crash on top of PR #283 (#319)`。
- 该提交包含新版 `ChunkLocalCumsum` BHT fast path、KKT score/pipeline/A5 优化，以及 KKT tail
  precision 和 BT128 crash 修复。后续测试、构建和性能报告均使用该完整 commit，不再使用动态 `main`。
- 旧 `main@8a63cf3eb28807440abf3aa88adedca2e5152862`、旧 `chw`/Phase6 三路线结果继续保留为历史归档，
  不与新版结果拼接或混用。
- 新分支 `chw_new_cumsum_kkt` 的新版 main 集成检查点为
  `67b797491d611b627227998196e972219fcc6ee5`：它合入了上述固定 main，且独立
  `ChunkLocalCumsum` 与 `ChunkScaledDotKkt` 均保持 upstream 源码版本。为保持既有复合 ACLNN
  路径可构建，旧的“融合 cumsum + KKT epilogue”仅作为私有 helper 保留；它不是独立
  `ChunkScaledDotKkt` 的回退，也尚未宣称吸收新版 KKT Catlass 流水。
- 该检查点只通过了合并完整性、冲突标记、diff 空白和 Python 打包脚本语法检查；尚未执行 A5 构建。
  A5 首轮需重新构建六算子与 Phase6 运行时，并记录源码、wheel、OPP 和 `libcust_opapi.so` hash。
- 新版三路线口径固定为：
  1. `main_new6`：新版 `main@3434816c1da8082996962e61e070f9947541d89b` 六 ACLNN；
  2. `chw_new6`：`chw_new_cumsum_kkt` 分支六 ACLNN；
  3. `phase6_new`：同一新分支的 `aclnnGdnCoreFwdPhase6`。
- 三路线复用现有固定测试合同和输入，先完成 route、shape/dtype、finite、fresh-process 确定性及精度门禁，
  通过后再执行同设备 AB/BA NPU Event 和 Level2 profiling。当前尚无新版三路线 NPU 精度或性能结论。

## A2：最新固定模型三路线门禁结果（2026-08-16，当前有效）

- 正式合同固定为 `B=2,Hk=Hv=32,T=8192,K=V=128,chunk=64,BF16,varlen=False`，device 0，输入文件
  SHA256 为 `3a4ae595403b429321732cf2f30b3dae2bb6802ee99e9f4720652af5de1ee822`。三路线为干净
  `main@8a63cf3` 六 ACLNN、同提交锚点 `chw@68a1a45e476d` 六 ACLNN、以及该 chw 工作树的
  `aclnnGdnCoreFwdPhase6`。
- 全量十算子 OPP 已完成构建，并通过隔离安装、公共 Python API、ACLNN 符号、10 个 kernel 目录及
  45 object/45 JSON 校验；当前有效 release 为
  `/opt/chw/gdn-phase6-deployments-latest-main-20260816/releases/20260816_161252_68a1a45e476d_dirty`。
  `libcust_opapi.so` SHA256 为 `d2de6e70ac7e40951935e5b07eb17300ab4130d0ae1cca22a919075b2e07b2be`。
- 有效运行证据根：
  `/opt/chw/gdn-model-case-a2-results/20260816_162529_b2_hk32_hv32_t8192_kt4096_vt4096_k128_v128_c64_d0`。
  三路线均完成 3 次 fresh-process 预检；route、shape、dtype、finite 均通过。main 与 chw 各三次均只有
  一个输出 raw SHA256，且两条 baseline 的 hash 相同：
  `6b73455ad329d3b3190f5e5679fa7ef6fbb17f31b1a7497933a0cfb1ab83acdf`。
- **门禁失败**：Phase6 三次 fresh process 分别得到三个不同的输出 raw SHA256：
  `2d71a788...d271a99c`、`eb655d05...33143750`、`f94eced1...fa448393`；
  `determinism_comparison.json.status=FAIL`。全部输出有限，但 Phase6 不具备重复 bit-exact 性。
- 对保存的首次输出进行离线量化后，main -> chw 完全 bit-exact；chw -> Phase6 有
  `4,702,626 / 67,108,864` 个 BF16 raw-bit mismatch（`7.0075%`），`36,647` 个元素超过
  `atol=rtol=0.01`，最大绝对误差 `0.0457000732`，RMSE `0.000646592`。详见同目录
  `accuracy_comparison.json`。
- 脚本在重复确定性门禁处主动停止，**没有**产生本轮 AB/BA NPU Event、正式性能结论或 profiler CSV；
  任何旧绝对性能数据均不得与本轮拼接。此前 systemd 缺少 `HOME` 的 r2 和前台 SSH 超时的 r1 均为无效中断轮，
  不纳入上述结论。
- 当前状态：`THREE_ROUTE_PHASE6_NONDETERMINISTIC_AND_ACCURACY_FAIL_NO_PERF`。冻结性能扩展、profiling
  和更大矩阵；下一步只能在用户批准新的单变量 Phase6 根因诊断计划后恢复 NPU 实验。

## A2：Phase6 中间量定位（2026-08-16，r5，有效）

- 这是对固定模型三路线失败的单变量定位实验，不是性能实验：同一输入文件、device 0、同一隔离
  release，先运行一次同分支六 ACLNN reference，再运行 3 次 fresh-process `aclnnGdnCoreFwdPhase6`。
  输入 SHA256 仍为 `3a4ae595403b429321732cf2f30b3dae2bb6802ee99e9f4720652af5de1ee822`；三次 Phase6
  route 均明确捕获为 `aclnnGdnCoreFwdPhase6`，workspace 均为 `1054642688` bytes，全部有限。
- 证据根：`/opt/chw/gdn-phase6-diagnostics/20260816_intermediate_r5`；汇总为
  `component_comparison.json`，脚本为 `gdn_phase6_intermediate_probe_verify.sh`。
- `g_cumsum`：Phase6 三次 raw SHA256 完全相同，且三次均与六 ACLNN reference bit-exact（0 mismatch）。
- `A`：Phase6 三次 raw SHA256 完全相同，且三次均与六 ACLNN reference bit-exact（0 mismatch）。
- `o`：Phase6 三次分别为 3 个 raw SHA256，重复 bit-exact 失败；相对 reference 的 BF16 raw mismatch
  分别为 `4,676,252 (6.9682%)`、`4,740,985 (7.0646%)`、`4,626,922 (6.8947%)`，最大绝对误差分别为
  `0.0462494`、`0.0388489`、`0.0417480`。这与三路线正式门禁的漂移结论一致。
- 当前可证明的首个**可观测**分歧在 `A/g_cumsum` 之后、公开 `o` 之前：Cumsum、KKT、SolveTri
  这条 ABC 产物链在本 case 上不是漂移源。`A` 之后还经过 Recompute W/U、FwdH、FwdO，
  仅凭公开输出不能在这三个阶段之间继续定位；因此根因状态仍为
  `PHASE6_DRIFT_AFTER_A_BEFORE_O_UNRESOLVED`，不宣称已定位到具体 kernel。
- 仍冻结 AB/BA NPU Event、profiler、1000-case、V=256 和其它 shape；下一条实验必须是经过批准的
  Recompute W/U/H/O 内部单变量观测或替代路径对照。

## A2：HO 单变量同步探针门禁（2026-08-16）

- 本次只验证一个变量：在 `RunPhase6` 的 solve/recompute 边界插入一处 `SyncAll`，并只替换
  BF16 固定变体 `ChunkGdnCoreFwd_e9ff32ae361a136aa58ab0f8fa63b7a5`；其余 host、OPP、变体和全局
  `current` 均保持不变。隔离 release：
  `/opt/chw/gdn-phase6-deployments/releases/20260816_122612_68a1a45e476d_e9ff32ae361a136aa58ab0f8fa63b7a5_probe`。
- 固定输入为既有 `B=2,H=32,T=8192,K=V=128,BF16,C=64`，输入文件 SHA256 为
  `3a4ae595403b429321732cf2f30b3dae2bb6802ee99e9f4720652af5de1ee822`，device 1，fresh process。
- 结果：repeat 1 `PASS`、finite，output raw SHA256
  `a514557e883edc5baea3e17438c527531014b06a834c610c9b13ee8fb43cd406`；repeat 2 在输出检查阶段
  报 `RuntimeError: non-finite GDN output`。因此该单变量变体未通过稳定性门禁，不能扩大到更多重复或 case，
  也不能宣称 `SyncAll` 修复有效。
- 原始证据：`/opt/chw/gdn-phase6-probes/a2_ho_sync_bf16_r2`；验证 unit 已结束且无残留 probe 进程。
- 当前门禁：`PHASE6_HO_SYNCALL_PROBE_FAIL_NONFINITE`。1000-case、V=256、Phase3/4 扩展及其它设备实验继续冻结，
  仅保留后台完整编译作为构建证据，不激活其产物。

## A2：固定模型三路线正式门禁（2026-08-16）

- 运行根：`/opt/chw/gdn-model-case-a2-results/three_route_20260816_r1`；合同为
  `B=2,Hk=Hv=32,T=8192,K=V=128,chunk=64,BF16,varlen=False`，device 1，输入文件 SHA256
  `3a4ae595403b429321732cf2f30b3dae2bb6802ee99e9f4720652af5de1ee822`。
- `main_baseline@8a63cf3` 和 `chw_baseline@68a1a45` 均完成 fresh-process 3/3，六 ACLNN 路由、输出 shape/dtype、finite 门禁通过。
- 旧 Phase6 release `20260816_023919_68a1a45e476d_dirty` 首次 Phase6 preflight 触发
  `507015` AICore MPU 非法地址，正式脚本在性能扩展前停止；无 Event/profiler 数据可用。
- 旧 release 的 `source_status` 未包含当前本地 core/tiling 配套改动。已将本地 dirty tree（基底仍为
  `68a1a45e476d`）应用到远端隔离 clone `/opt/chw/self/code/phase6-latest-main-20260816`，并启动独立构建 unit
  `gdn-phase6-latest-main-build-20260816-r1.service`，deployment root 为
  `/opt/chw/gdn-phase6-deployments-latest-main-20260816`；该 unit 不改动正式 `current`。
- 这是当时的历史构建状态；后续完整包已发布并完成三路线复测，当前有效结论以文首
  “最新固定模型三路线门禁结果”为准：Phase6 重复确定性与容差均失败，未恢复 AB/BA 或 profiling。

## A2：baseline repair2 case585 根因与 R19 修复（2026-08-15）

- 当前首要功能门禁已从 R18 全 1000-case 首轮收敛到唯一重复异常 case585：
  `state_initial_final_0185_fp16_b1_lb16_hk4_hv8_c64_t1136`。
- 已定位首个错误算子为 `aclnnChunkGatedDeltaRuleFwdH`，首个错误输出为
  `final_state`；同次 `h` 和 `v_new` 不漂移。
- 根因是 scalar-g 模板 `kGated=false` 的 `blockTokens < 16` TailH 无条件读
  `gmKDecayWorkspace`，而该 workspace 只在 `kGated=true` 有生产者；因此将未定义
  workspace 内容带入 final-state 更新。
- R19 最小修复：`kGated=true` 保持读 K-decay workspace，`kGated=false` 改读真实
  `gmK`。R18 -> R19 只有这一个功能变量。
- R19 FwdH 同进程 20/20 finite 且 `h/v_new/final_state` 各一个 raw hash；官方六 ACLNN
  整网 fresh-process 5/5 finite、outer/core 门禁逐次通过、所有定义组件跨次 bit-exact。
- 为排除低概率不确定性，随后在同一 A2 device 3、同一共享 VLLM 环境中追加 100 次
  fresh-process 重放：100/100 `PASS`、100/100 finite、outer/core 门禁 100/100 通过；
  输入六路 hash、整网五类定义组件 raw hash 和 26 个 `valid_a_chunks` 均各只有一个状态。
  汇总证据：
  `/opt/chw/gdn-main-vs-phase6-full1000-r1/baseline_repair2/analysis_case585_r19_full_r100/evidence/r100_validation_summary.json`。
- 该结果证明 case585 在 R19 隔离候选上的 100 次重放未观察到跨次不确定性；尚不等同于
  统一生产候选的 1000-case 三轮验收，也不覆盖其它 SOC。
- CPU 公式对照进一步证明 R19 恢复了正确 K 项：差异点仅在短尾序列的 K 行
  63/127，R19 误差比 R18 主要坏态下降约 50--1000 倍。
- 当前状态：`CASE585_SCALAR_TAIL_K_R19_TARGETED_PASS_R100`。尚未将隔离 R19 宣称为生产版，
  下一门禁是带回统一候选后用原 1000 条保存输入做三轮 fresh-process 验收。
- 详细证据见
  [`GDN_FWDH_SCALAR_TAIL_K_R19_A2.md`](evidence/phase6_original_h_provenance_20260803/GDN_FWDH_SCALAR_TAIL_K_R19_A2.md)；
  远程原始证据根为 `/opt/chw/gdn-case585-scalar-tailk-20260815-r19/evidence`。

## A2：三路线正式性能实验准备状态（2026-08-15）

- 本轮正式口径固定为同一输入、同一卡、同进程隔离的三条路线：干净
  `main@8a63cf3eb288` 六 ACLNN、`chw@68a1a45e476d` 同分支六 ACLNN、同一
  `chw` 分支的 Phase6。报告拆分为总体收益（main -> Phase6）、旧算子修改收益
  （main -> chw 六算子）和纯融合收益（chw 六算子 -> Phase6）。
- `main_baseline` release
  `/opt/chw/gdn-baseline-deployments/releases/20260815_114734_8a63cf3eb288` 已核验为上述固定
  commit，且六 ACLNN public/workspace 符号门禁通过。上游 main 后续推进（当前可见 tip 为
  `73798c9`）不改变此 case 的定义。
- 当前 `chw` 工作树以 `68a1a45e476d` 为提交锚点。此前
  `/opt/chw/gdn-phase6-deployments/releases/20260815_115317_68a1a45e476d` 是旧隔离 release；
  2026-08-15 正在通过 `gdn_phase6_build_deploy.sh` 从当前工作树生成完整 OPP 包。此前一次
  `--ops=chunk_gdn_core_fwd` 的局部包缺少
  `aclnnGdnCoreFwdPhase6GetWorkspaceSize`，因此不可用于本轮正式测试；完整包尚未激活前，
  三路线 NPU 测试不得启动。
- 首次 NPU 执行因脚本误加入未批准的 recurrent 第四路线，在
  `aclnnRecurrentGatedDeltaRuleGetWorkspaceSize` 返回 `aclnnStatus=561103` 后按
  失败即停止；三条正式路线尚未启动，不得据此写入性能或精度结论。失败证据：
  `/opt/chw/gdn-three-route-results/20260815_125611_b1_hk2_hv4_t16384_kt256_vt512_k128_v128_c64_d4`。
- 修正脚本后再次启动时，device 4 在 `main_baseline` 前被 `VLLMWorker_TP` 占用，且
  device 0-7 全部有同类进程；空卡门禁再次停止执行，未发射任何 GDN kernel。失败证据：
  `/opt/chw/gdn-three-route-results/20260815_141103_b1_hk2_hv4_t16384_kt256_vt512_k128_v128_c64_d4`。
- 已将本地和远端 `/opt/chw/self` 的测试入口收敛为三路线，静态 `bash -n`、Python AST
  和路线残留检查均通过。用户已授权 A2 共享卡运行，固定模型入口默认
  `GDN_REQUIRE_IDLE=0`，但会保存运行前 `npu-smi` 快照并在结论中标注常驻 VLLM 竞争；待完整
  Phase6 OPP 的符号门禁通过后，先做固定输入预检，再做 AB/BA 与 profiler。

## A2：Phase6 原生 GVA + 任意 dense T 最小闭环（2026-08-14）

- 当前状态：`PHASE6_GVA_DENSE_T_MINIMAL_PASS`。
- Phase6 生产路径已支持物理 `Hq=Hk`、`Hv % Hk == 0`，不在 ACLNN/Python 路径展开 q/k；
  `Hk/Hv/hvPerHk` 进入 kernel tiling，KKT/FwdH/FwdO 均执行 `Hv -> Hk` 映射，workspace、output、
  `g_cumsum`、`A` 和 state 均按 Hv 维组织。
- dense T 的旧白名单已从新 host tiling 移除；同时修正 `a_storage` 的 host 校验为
  `[B,Hv,T,chunk_size]`。
- A2 device0 最小实机通过：`BF16,B=1,Hk=2,Hv=8,T=130,K=V=128,C=64/128`。两个 chunk_size 下，
  原生 GVA 与等价 q/k 展开参考的 `output/final_state/g_cumsum/A` 四项逐位一致；C64 的 shape 分别为
  `[1,8,130,128]/[1,8,128,128]/[1,130,8]/[1,8,130,64]`，且全部有限。
- 完整 kernel/package、完整 Phase opapi 与闭合 host 已构建成功；隔离根为
  `/opt/chw/gdn-phase6-gva-dense-t-20260814-r1`。详细构建哈希、门禁和 smoke 原始输出见
  [`phase6_gva_dense_t_20260814`](evidence/phase6_gva_dense_t_20260814/README.md)。
- 边界：当前只完成 A2 最小功能/精度门禁，尚不代表 GVA 全矩阵、性能、`T=8192` 或 A5 已验收。

## A2：A5 固定输入的 baseline / Phase6 运行性门禁（2026-08-14）

- 固定合同为 `B=2,H=32,T=8192,K=V=128,BF16,chunk=64,scale=1.0`；A5 原始输入文件 SHA256 为
  `5eb53f5079d8236f73e47c20faba28c4deb6b9ebdc7fdc0611d2574dab08969b`。虽然 A2 的 PyTorch 序列化
  文件 SHA256 因版本不同而不同，但 `q/k/v/g/beta` 五个实际输入张量的 raw-byte SHA256 均与 A5 逐项一致。
- A2 device4 上的官方六 ACLNN baseline 通过：两次预热后一次整网 NPU Event 为 `69.994820 ms`；14 条
  kernel task 总和为 `11,846.437 us`，其中六个核心算子为 `10,966.539 us`。采样时 device utilization=0%，
  但有 VLLM 常驻约 62% HBM，故仅作为共享卡运行性/性能样本。
- 同输入、同外层网络切换到实际 Phase6 overlay 后，在
  `aclnnGdnCoreFwdPhase6GetWorkspaceSize` 返回 `169112`；同进程 tiling 日志明确为 BF16 dense 仅支持
  `T=128/1024`，本次 `T=8192` 未发射 kernel。不得将 baseline 与 Phase6 计算百分比或宣称该 shape 的
  Phase6 性能。版本边界必须保留：本次加载的是远端既有 task-affinity overlay（SHA256
  `8b75c4a986d706e1ec3d5dd1e4d040e40add50d325535a0609748c2fedce50f6`），不是当前本地工作树的
  最新未提交源码。该条是旧 overlay 的历史失败记录；后续新隔离 overlay 已完成 `Hk=2,Hv=8,T=130`
  最小构建与实机验证，但尚未据此把本条 `T=8192` 场景升级为通过。
- 独立证据见 [`a2_same_input_baseline_phase6_profile_20260814`](evidence/a2_same_input_baseline_phase6_profile_20260814/README.md)。
  该旁路运行性门禁不改变 Phase6-R11 精度闭环、1000-case 或 V=256 的现有冻结状态。

## A5 model.csv：原始 main baseline 与既有 Phase6 归档（2026-08-13）

- 原始 main baseline 已完成，不重复运行；来源为干净 `main@ded35453915d100b7404f57b6643c81672388e10`，A5 device 2，
  `B=2,H=32,T=8192,K=V=128,BF16,chunk=64`，200 次 NPU Event 平均 `10.293165 ms`、中位数 `10.032376 ms`。
- 此前已完成的 Phase6 性能与 profiling 证据也已归档到本地唯一目录：
  [`model_csv_main_vs_phase6_a5_20260813`](evidence/model_csv_main_vs_phase6_a5_20260813/README.md)。
- 归档目录中的 `baseline_original_main/` 只含干净 main 报告、固定输入及其 runner；`phase6_completed/` 原样保存此前
  Phase6 的 paired、formal trials、Level2、profiling、trace 和日志。刚才对当前 Phase6 环境的失败复核不纳入正式结果。
- baseline 与历史 Phase6 不是同一轮同设备 AB/BA 交替测量，不能直接宣称严格加速比；详细比较边界见归档 README。
- 对应 CSV 已一并归档：`model.csv`、Phase6 的 `phase6_task_time.csv`、`phase6_op_summary.csv`、`phase6_api_statistic.csv`。

## 旁路状态：V128 baseline 1000-case（2026-08-13）

- 路径：`/opt/chw/gdn-main-vs-phase6-full1000-128and256-r1/v128-baseline`，当前停止于
  `412/1000`，其中 `322` 条 self-bit-exact、`90` 条 self-nonexact、`1` 条 non-finite、
  `1` 条 outer/core anomaly；这组统计不是 Phase6 验收结论。
- 最新断点为 `index=412`：前三次 repeat 均已完成，第三次 fresh Python 进程在
  `torch.npu.set_device(0)` 阶段报 CANN `507033/E39007`（`TsdOpen` / device subprocess
  startup timeout）。当时没有执行 kernel；随后 `npu-smi` 显示 device0 `Health=OK` 且无残留进程，
  因此归类为一次性设备初始化基础设施故障，不归因于算子或精度。
- 远端 v128 baseline runner 已加入受限重试：仅当日志匹配 `507033`、`E39007`、`TsdOpen failed`
  或 `Starting a subprocess on the device timed out` 时，单个 fresh capture 最多重试 3 次；
  其它 kernel、非有限值、outer/core anomaly 和返回码仍立即失败。runner 已通过 `bash -n`，未自动重启。
- 断点续跑会跳过已有完整 case，并将未完成的 `index=412` 目录保留后重做；当前应先确认分配卡仍健康，
  再用 `GDN_START_FULL1000=1 GDN_REPEATS=3 GDN_DEVICE=<空闲卡>` 启动。

## Ascend950 A5 适配状态

- 950 目标 wheel 已在 A5 构建成功：
  `flash_linear_attention_npu-26.7.0.dev0-950.x86_64-py3-none-any.whl`，SHA256=
  `00043f792fff2e1764795b8e2856cc72e567bee7e38026c3ac84488fc42b4e4c`。
- 隔离安装后的 Phase6 符号检查通过；最小 BF16 `B=1,H=8,T=128,K=V=128,C=64`
  smoke 通过，Phase6 对 Phase5 逐位一致且全部有限。
- 同形状 paired NPU Event 计时通过：Phase5 median `0.243865 ms`，Phase6 median
  `0.243976 ms`，变化约 `+0.046%`；该数据仅为运行时门禁，不替代 A5 model.csv 正式性能。
- 证据目录：[`phase6_a5_950_modelcase_20260813`](evidence/phase6_a5_950_modelcase_20260813/README.md)。

> 阶段 7 启动门禁：路线已进入 Phase 7 准备态，但 transpose/layout 设备探针暂不启动。
> 必须先把 FwdH R11 候选从固定 case3 扩展验证并形成生产收口；在此之前继续冻结 Phase6
> 全矩阵、1000-case、V=256 和其它 layout 变量，避免把同步修复与布局收益混测。

## 当前获准实验：Phase6-R11 × case198 50 次

- 用户已批准把 R11 的两处最小时序改动带入 Phase6 大融合 kernel，并固定
  `dense_0198_bf16_b1_lb1_hk32_hv32_c64_t1024` 重放 50 次。
- 只移动 `vec1Done` 通知：从 `vnewdecay` 写回后移动到最终 `vnewOutput` GM 写回后；
  不整文件替换，不引入 R11 实验源码中的其它功能差异。
- 正式输入保持为 `/opt/chw/gdn-case198-phase6-repeat20-20260812-r2/evidence/input.pt`，
  SHA256=`84f689bca852712ccacb13d27fa68e124c4bd732f238855a85e717991a1a3354`。
- 官方 six-ACLNN 参考保持为同目录 `baseline_reference.pt`，
  SHA256=`7b172e7dff860cd8e3b8bef7ad56580f91c5ae1f2beee36de1511fbeb3f8b613`。
- 新实验根：`/opt/chw/gdn-phase6-r11-case198-20260813-r1`；仅在隔离源码副本中构建，
  只替换独立 overlay 的 `chunk_gdn_core_fwd` kernel，正式 Phase6 overlay 和历史证据只读。
- 运行合同：50 次均为 fresh Python process；验收要求为输入 50/50 一致、全部 finite、
  outer/core 50/50 一致、Phase6 仅一个 output raw hash，且 50/50 与冻结 six-ACLNN 参考逐位一致。
- 隔离构建已完成：`/opt/chw/gdn-phase6-r11-case198-20260813-r1/build_evidence/status.txt=COMPLETE`；
  运行前已确认正式源码哈希未变、正式 `op_api` 哈希保持 `8b75c4a9...50f6`、非 core overlay
  与正式 overlay 完全一致，卡 2 当时无其它进程。
- 50 次运行与汇总已完成：远端 `acceptance.json.status=PASS`；50/50 输入一致、finite、
  outer/core 门禁通过，Phase6 只有 1 个 output raw hash，50/50 与冻结 six-ACLNN
  reference 逐位一致；paired nonexact=0。
- 当前状态：`PHASE6_R11_CASE198_REPEAT50_PASS`。本实验已停止，卡 2 无残留进程；
  不自动扩大到其它 case 或矩阵，等待下一条明确的回归授权。

## 下一步实验：Phase6 repair1 × 原 73 条 self-nonexact

- 本次固定集合不是历史 89 条，也不是按 `historical_bit_exact=false` 过滤；它严格来自
  `/opt/chw/gdn-main-vs-phase6-full1000-r1/phase6-only/archive-r1/evidence/failure_index.json`，
  哈希为 `6dda16da...f10085d`，共 73 个原 Phase6 self-nonexact index。
- 每条直接复制原 Phase6 archive 的 immutable `input.pt`，不重新物化、不随机生成、不调用 baseline；
  每条使用 R11 修复 overlay，3 次 fresh Python process，保存新 self-check。
- 新隔离根：`/opt/chw/gdn-main-vs-phase6-full1000-r1/phase6_repair1`；原 `phase6-only`
  archive、1000-case 结果和 case198 50 次证据只读。
- 主要门禁：73 条全部完成，统计 `old self-nonexact -> new self-exact` 修复数、剩余非 exact、
  finite、outer/core gate 和 NPU Event 支持情况。
- 当前状态：`RUNNING_PHASE6_REPAIR1_73`；A2 已启动
  `/opt/chw/run_phase6_repair1_73.sh`，实时进度以远端
  `/opt/chw/gdn-main-vs-phase6-full1000-r1/phase6_repair1/evidence/progress.txt`
  和 `progress.json` 为准；截至 2026-08-13 20:55（A2），已完成 20/73，
  其中 10 条从原 self-nonexact 修复为新 self-exact、10 条仍为新 self-nonexact；
  case333 曾发生一次 A2 NPU 启动超时（507033），保留现场后已断点续跑成功，
  当前正在 case337。
  repair1 运行期间不扩大到其它 case、1000-case、V=256 或 layout 矩阵。

## 唯一当前结论

## A2：固定模型三路线正式包重建（2026-08-16）

- 正式合同固定为 `B=2,Hk=Hv=32,T=8192,K=V=128,chunk_size=64,BF16,varlen=False`；三条
  路线为干净 `main@8a63cf3eb288` 六 ACLNN、同基底本地 `chw@68a1a45e476d` 六 ACLNN，及同一
  `chw` 工作树的 `aclnnGdnCoreFwdPhase6`。
- 两条 baseline 已完成三次 fresh-process 预检：路线调用、shape/dtype、全部有限值门禁均通过。
  旧 Phase6 release 在首个预检报 `507015`，设备日志为 `ChunkGdnCoreFwd` AICore MPU 非法访存；
  因此旧 release 未进入 Event 或 profiler，不能据其产生任何性能结论。
- 本地最新未提交工作树已以 `68a1a45e476d` 为提交锚点同步到
  `/opt/chw/self/code/phase6-latest-main-20260816`；`git apply --check` 与 `git diff --check` 通过。
  此处“最新”只指本地工作树，不指持续推进的 upstream main tip。
- 第一轮完整包构建在 protobuf 编译向共享 `/tmp` 写临时汇编时遇到 `No space left on device`；这是基础设施
  失败，不是源码/算子编译错误，失败 staging 和日志均保留在
  `/opt/chw/gdn-phase6-deployments-latest-main-20260816/.staging_20260816_125831_68a1a45e476d_dirty_801142`。
- 历史记录：第二轮隔离重建当时运行于 systemd 单元
  `gdn-phase6-latest-main-build-20260816-r2.service`，部署根
  `/opt/chw/gdn-phase6-deployments-latest-main-20260816`，临时目录
  `/opt/chw/gdn-phase6-build-tmp-20260816-r2`，并行度 `-j2`。它不覆盖任何旧 release/current。
  截至记录时 `SolveTri`、`ChunkCumsumKkt`、`ChunkCumsumKktSolveTri`、`ChunkFwdO` 和部分 `FwdH`
  已完成，正在编译 `FwdH/HO` 变体；服务为 `active/running`、编译器满核，未见新的错误。
- 新包通过安装与符号/内核目录校验后，才允许用
  `PHASE6_DEPLOY_ROOT=/opt/chw/gdn-phase6-deployments-latest-main-20260816`
  运行 `gdn_test_model_case_a2.sh`。测试入口已确认先做三路 3 次 fresh-process 预检，之后才执行 AB/BA
  NPU Event，最后每路采集 Level2 + PipeUtilization 和 5 类 CANN CSV。
- 当前构建余量已由生成参数文件核实：`ChunkGatedDeltaRuleFwdHO` 共 8 个参数族，5 个已落盘、3 个仍在
  编译；`ChunkGdnCoreFwd` 共 4 个参数族，尚未开始。故当前长时间运行属于全量 OPP 展开，不是无日志卡死。
- 该构建随后生成完整 `.run` 包；包装脚本首次因 `-j2` 续行异常未发布，复用同一 `.run` 后已完成
  隔离安装和符号/内核/API 校验。当前状态已更新为文首的
  `THREE_ROUTE_PHASE6_NONDETERMINISTIC_AND_ACCURACY_FAIL_NO_PERF`。

- 当前定位对象是原 main 六 ACLNN baseline 的自非 bit-exact，不是 Phase6 独有问题。
- 首个漂移输出已定位到 `aclnnChunkGatedDeltaRuleFwdH.v_new`；后续 `ChunkFwdO.output` 只是传播。
- 固定保存输入为 case3：`dense_0003_bf16_b1_lb1_hk16_hv32_c128_t128`，输入 raw hash 在所有实验中一致。
- 根因已定位到 FwdH `vnewdecay` 生产完成后的跨核通知时序错误：消费者等待 `vec1Done` 后读取 `gmVUpdateWorkspace`，原实现可能在生产侧完成必要的 V/MTE3 依赖前就放行消费者。
- r7/r8/r11 均在固定 case3 上得到同一个稳定正确 raw hash；其中 r11 只把通知延后到最终 `vnewOutput` GM 写回之后、保持原 `PIPE_MTE3`，10/10 重入稳定。因此“通知时序”已被因果验证；`PIPE_V` 仍是需要在双分支生产方案中继续验证的候选，不再把它单独宣称为唯一根因。
- r10 双分支 `PIPE_V` 候选未通过 raw-bit 稳定性门禁：4 次均有结果但输出 hash 为 4 个状态，不能收口。
- r12 进一步证明：仅把 fast-path 通知移到现有 `WaitFlag<MTE3_V>` 后、仍用 `PIPE_MTE3` 也不够；10 次得到 10 个 `v_new/o/output` raw hash，`h` 保持稳定且全部有限。
- 2026-08-13 已把 R11 两处“最终 `vnewOutput` GM 写回后通知”最小源码改动同步到本地，同时保留已有 zero-row EVENT0 hand-off 修复；该本地状态是待扩展验证候选，不是已验收 Phase 7 生产版本。
- 2026-08-13 Phase6-R11 隔离回归已通过固定 case198 50 次：修复版 output raw hash
  `e87bc5cdef80da277b92b8ab1d3676c0208bad92eb62e906dc807232e7bb071c` 与冻结
  six-ACLNN reference 相同；详细摘要见
  [`GDN_PHASE6_R11_CASE198_REPEAT50_A2.md`](evidence/phase6_original_h_provenance_20260803/GDN_PHASE6_R11_CASE198_REPEAT50_A2.md)。

## 已完成的因果实验

| 实验 | 源码变量 | 结果 | 结论 |
| --- | --- | --- | --- |
| r6 | 强制走 slow path；保留原 `PIPE_MTE3` 通知 | 4 次中 2 次失败；每次恰有 2 个完整 BF16 行变为 0，共 256 元素 | slow path 可独立复现生产者/消费者竞争 |
| r7 | r6 基础上，把通知延后到 `vnewOutput` GM 写回之后 | 10/10 exact | 延迟通知可消除竞争，但同步范围偏大 |
| r8 | r6 基础上仅把 slow path 通知 `PIPE_MTE3 -> PIPE_V`，通知位置不变 | 10/10 exact；每次 `v_new` mismatch=0、全零行=0；最终输出跨次 exact | 唯一单变量证明 slow path 根因是通知 pipe 归属错误 |
| r9 | 生产代码只修 slow path，不强制分支 | 4/4 失败；每次出现 6、16、6、20 条完整全零行，`v_new` mismatch 为 768、2048、768、2560 | 直接证明正常 case3 实际走 fast path，只修 slow 不足 |
| r10 | fast 通知移动到既有 `WaitFlag<MTE3_V>` 后并改用 `PIPE_V`；slow 改用 `PIPE_V` | 4 次均完成但 `output/v_new/o` 各出现 4 个 raw hash 状态 | 双分支 `PIPE_V` 候选未通过稳定性门禁，不能收口 |
| r11 | fast/slow 通知均保持 `PIPE_MTE3`，但移动到最终 `vnewOutput` GM 写回之后 | 10/10 `PASS`；`output/core/v_new/h/o` 各 1 个 raw hash，且与 r7/r8 完全一致；全部有限 | 证明延后通知时序可消除固定 case3 的 FwdH 漂移，但同步范围偏大，尚非最终生产修复 |
| r12 | 只把 fast-path 通知移到已有 `WaitFlag<MTE3_V>` 之后，保持 `PIPE_MTE3`；slow 不改 | 10/10 均完成且有限，但 `output/core/v_new/o` 各 10 个 raw hash，`h` 只有 1 个 hash | V 侧等待之后立即从 MTE3 发通知仍不能建立所需 happens-before；需要继续定位后续 MTE3 侧排序点 |

r6 与 r8 基于同一 commit、同一强制 slow path，源码唯一额外差异就是 `PIPE_MTE3 -> PIPE_V`。r6 的一个正确 repeat、r7、r8 和 r11 的正确结果已保存；r7/r8/r11 的固定 case3 输出 raw hash 完全一致。r11 说明仅改变通知时序也能恢复该正确态，因此当前结论必须同时保留 pipe 与时序两个变量，不能把 `PIPE_V` 单独升级为充分条件。

## 关键源码事实

- 源码 commit：`ded35453915d100b7404f57b6643c81672388e10`
- 文件：`block_epilogue_gdn_fwdh_vnew.hpp`
- slow path：`DataCopy(vnewdecay)` 后已有 `SetFlag/WaitFlag<MTE3_V>`，原通知绑定 `PIPE_MTE3`；r8 改为 `PIPE_V` 后稳定，r11 延后到最终输出写回后也稳定。
- fast path：原通知发生在 `DataCopy(vnewdecay)` 之后、现有 `WaitFlag<MTE3_V>` 之前。r11 将通知移到最终 `vnewOutput` GM 写回之后并保持 `PIPE_MTE3`，固定 case3 10/10 稳定；静态审计确认这覆盖了消费者读取 `gmVUpdateWorkspace` 前的可见性风险，但仍需更小范围的“仅移到 `WaitFlag<MTE3_V>` 后、保持 `PIPE_MTE3`”实验来判断是否可以避免延后到最终输出写回。
- CANN 9.1.0.beta1 DAV220 实现确认：`CrossCoreSetFlag<..., pipe>` 最终调用 `ffts_cross_core_sync(pipe, ...)`，pipe 参数是实际调度依赖，不是无意义标签。

## 证据位置

- 原始 baseline 定位：`/opt/chw/gdn-main-vs-phase6-h-stage-sync-case3-r1/evidence`
- r6 复现：`/opt/chw/gdn-fwdh-v128-tiled-case3-20260812-r6/evidence`
- r7 延后通知对照：`/opt/chw/gdn-fwdh-vnew-order-case3-20260812-r7/evidence`
- r8 单变量 pipe 修复：`/opt/chw/gdn-fwdh-vnew-pipev-case3-20260812-r8/evidence`
- r8 正式摘要：`analysis_10_summary.json`
- r6/r7/r8 正确结果交叉比对：`cross_variant_exact.json`
- r9 最小生产 slow-only 负对照：`/opt/chw/gdn-fwdh-vnew-pipev-prod-case3-20260812-r9/evidence/analysis_4.json`
- r10 双分支最小生产候选：`/opt/chw/gdn-fwdh-vnew-pipev-full-case3-20260812-r10`
- r11 时序对照（保持 `PIPE_MTE3`，通知延后到最终输出写回）：[`GDN_FWDH_AFTER_OUTPUT_R11_A2.md`](evidence/phase6_original_h_provenance_20260803/GDN_FWDH_AFTER_OUTPUT_R11_A2.md)；远端原始证据 `/opt/chw/gdn-fwdh-after-output-case3-20260812-r11/evidence`
- r12 fast-path after-wait 对照：`/opt/chw/gdn-fwdh-vnew-afterwait-mte3-case3-20260812-r12/evidence`

## 后续门禁

1. 本次 50/50 只证明 R11 时序修复覆盖固定 case198；仍需用户明确授权后，才可做 case3 与规格矩阵回归，形成生产收口。
2. 其它 Phase6、1000-case、V=256 和 layout 工作继续冻结，不能把本次固定 case 结果外推成全矩阵结论。
3. 若后续获准回归再次出现非 exact，必须保留首个异常 repeat 和完整 raw-bit 比较，并恢复 D/E/F 同次边界定位。
4. r13 更小同步范围探索暂缓，待 R11 固定 case 结果纳入回归门禁后再决定。
