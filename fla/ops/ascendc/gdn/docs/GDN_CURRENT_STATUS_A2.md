# GDN 当前进度 A2

更新时间：2026-08-14（Asia/Shanghai）

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
