# SolveTri A2 补充验证

本页对应 A2 `ascend910b`。原 A5 的 200 条精度、6 条性能、6 条 MSS 清单，以及默认 YAML、generator、executor 和执行命令保留在 [README.md](README.md)。公共 `run_test_cpu.sh -op=solve_tri` 使用原 A5 文件，`-soc` 不会切换清单；A2 使用本页的独立入口。

## 文件与覆盖

| 文件 | 用途 |
| --- | --- |
| `gen_solve_tri_a2.py` | 仅生成带 `_a2` 名称的清单，不写原 A5 文件。 |
| `executor_solve_tri_a2.py` | A2 NPU DUT 与 CPU FP64 `linalg.inv`，保留混合空序列支持。 |
| `atk_solve_tri_a2.json` | 208 条精度：原有 200 种输入在 A2 的覆盖，加 200-205 混合空序列及 206-207 长流水。 |
| `atk_solve_tri_a2_perf.json` | 208/209 两个目标模型场景，BF16、chunk 64。 |
| `atk_solve_tri_a2_mss.json` | 6 条基本分支及长流水 206/207，共 8 条，供确定性和 sanitizer 使用。 |
| `atk_solve_tri_a2_negative.json` | 31 条真实 ACLNN 负例、1 条 TND/NTD 正向契约。 |
| `atk_solve_tri_a2_resources.json` | 208/209 的额外分配器显存测量。 |

A2 精度使用 `mixed_tolerance_bm`、相同量化输入上的 CPU FP64 golden，输入值域和精度规则沿用此前验证。`x/out` 为 FLOAT16/BFLOAT16，chunk 为 16/32/64/128，布局为 BSND/BNSD/TND/NTD。混合空序列要求总长度为正；空序列不生成 chunk。

| Case | 输入 | 目的 |
| --- | --- | --- |
| 200-205 | TND/NTD，前缀、中间、尾部空序列 | 空序列兼容。 |
| 206 | BSND `[1,8193,32,64]` | 定长尾块及多批 ring 复用。 |
| 207 | TND `[8193,32,64]`，seqlens `[4096,4097]` | 变长尾块及多批 ring 复用。 |
| 208 | BSND `[1,11264,32,64]` | Issue #660 定长目标场景。 |
| 209 | TND `[65536,32,64]`，64 x 1024 | 平均长度 1K 的可复现变长代表。 |

`HK/K/V` 不属于 SolveTri 输入，算子输入 H 对应 GDN 的 HV=32。209 的长度分布是代表场景，不是未知业务长度分布的还原。

## 生成与执行

在仓库根目录生成 A2 清单：

```bash
python tests/atk/solve_tri/gen_solve_tri_a2.py
```

加载与 A2 匹配的 CANN 和本轮普通 wheel/OPP 后，在 `tests/atk/solve_tri` 执行。以下命令显式选择 A2 文件，不修改公共 runner，也不导出 `PYTHONPATH`。ATK 至少使用仓库要求的版本；历史结果来自 26.7.8，不能据此降低默认版本门禁。

```bash
atk node --name npu_dut --backend npu --devices 0 \
  --output_path /absolute/results/a2_accuracy \
  node --name cpu_golden --backend cpu \
  --output_path /absolute/results/a2_accuracy \
  task -c atk_solve_tri_a2.json -p executor_solve_tri_a2.py \
  --task accuracy --bm_device cpu -sp -to 14400

atk node --name npu_dut --backend npu --devices 0 \
  --output_path /absolute/results/a2_determinism \
  task -c atk_solve_tri_a2_mss.json -p executor_solve_tri_a2.py \
  --task accuracy_dc --dc_loop_nums 50 -sp -to 3600

python scripts/check_aclnn_contract.py --check-json
atk node --name npu_contract --backend npu --devices 0 \
  --output_path /absolute/results/a2_contract \
  task -c atk_solve_tri_a2_negative.json -p executor_solve_tri_a2_contract.py --task run -sp
```

契约测试直接调用 ACLNN 第一阶段，每条负例检查返回码 161002、结构化错误码、参数名、实际值和期望值，拒绝后不得返回 workspace/executor。正向项通过 Stable 入口检查 TND/NTD 空序列、合法索引重排和压缩元数据的输出一致。必须核对全部 32 项成功，不能只看 ATK 退出码。

## 性能与显存

先以 `atk_solve_tri_a2_perf.json` 和 `executor_solve_tri_a2.py` 执行 CPU golden 的 `accuracy`，随后使用相同输入种子测性能：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
atk node --name npu_dut --backend npu --devices 0 \
  --output_path /absolute/results/a2_performance \
  task -c atk_solve_tri_a2_perf.json -p executor_solve_tri_a2.py \
  --task performance_device --performance_data 3,20,20 --save_data profile -sp

SOLVE_TRI_MEMORY_OUTPUT=/absolute/results/a2_memory_raw \
atk node --name npu_dut --backend npu --devices 0 \
  --output_path /absolute/results/a2_resources \
  task -c atk_solve_tri_a2_resources.json -p executor_solve_tri_a2_resources.py --task run -sp
python scripts/check_performance.py memory --input-root /absolute/results/a2_memory_raw \
  --output /absolute/results/a2_memory.json
```

预热 3 次、采样 20 次；CPU 线程限制避免输入生成过度并行，版本对照使用相同配置。资源测量为 `max_memory_allocated - baseline - output_storage_bytes`，包含 torch_npu 分配器内的 Stable workspace，不包含分配器以外的驱动内存。额外显存上限为 200000000 bytes。

性能及精度 worker 已加载算子库且仍存活时，分别采集：

```bash
python scripts/performance_provenance.py --pid PERFORMANCE_WORKER_PID --output /absolute/results/performance_worker.json
python scripts/performance_provenance.py --pid ACCURACY_WORKER_PID --output /absolute/results/accuracy_worker.json
python scripts/check_performance.py check-a2 --manifest /absolute/results/evidence.json \
  --output /absolute/results/a2_gate.json
```

`performance_evidence.json` 是模板。填写真实构建、CANN、设备、报告、profile、worker 加载证明和内存报告。精度报告保留原任务的 `report/*.xlsx` 与 `log/atk.log` 及文件名，门禁绑定任务、PID、设备和用例，核对三类任务的 opapi/kernel 资产哈希。每例至少 20 条 SolveTri Task Duration，其均值必须与 XLSX 一致；缺证据或显存超限均失败。资产哈希仍需 profile 证明实际命中的 kernel。

`check-a2` 不判定跨平台加速比。完整 A2/A3/A5 对照另执行 `check --max-duration-ratio ...`，比值为 A2/A3 平均 Task Duration 除以同输入 A5 平均 Task Duration。A5 比较数据必须通过显式独立任务采集，不能覆盖原 A5 默认性能清单；缺平台或基线不会因 `not_key` 自动通过。

## 长流水检测

仓库根目录单独构建插桩产物并加载对应 OPP，性能测量继续使用普通产物：

```bash
FLA_NPU_OPS=solve_tri FLA_NPU_SOC=ascend910b \
python scripts/build_wheel.py --sanitizer -g --no-progress
```

随后在算子 ATK 目录执行：

```bash
python scripts/check_pipeline.py run --device 0 \
  --core-profile /absolute/a2_profile/kernel_details.csv \
  --report-root /absolute/results/new_a2_pipeline_run
```

该入口只读取 A2 MSS 清单、加载 A2 executor，以 racecheck/memcheck 检测 206/207。核数 profile 应来自相同设备、目标规模 chunk-64 任务。门禁检查实际启动、同 kernel 的结束配对、完整用例契约及 ATK 检测结果；跳过检测、unsupported、越界、FFTS 未复位、缺失或误配证据均失败。

## 平台拆分复测

2026-11-10 使用新的 `_a2` 清单及注册名，复用下节相同普通 wheel，在 A2 910B3、CANN 9.2.0、ATK `26.7.8.dev20260730` 环境复测。精度、模型场景和资源在 NPU 2，契约在同机同型号 NPU 5；命令使用 `-sp`，逐条核对 XLSX，不能仅依据 ATK 进程退出码。

| 项目 | 本次结果 |
| --- | --- |
| A2 全量精度 | 208/208 SUCCESS，208/208 精度通过，包含混合空序列及长流水。 |
| 模型场景 208/209 精度 | 2/2 SUCCESS，2/2 精度通过。 |
| ACLNN 契约 | 补齐公共辅助文件后的 `contract_retry` 为 32/32 SUCCESS。 |
| A2 资源入口 | 2/2 SUCCESS；208/209 额外分配器显存各 100664832 bytes，低于 200000000 bytes。 |
| 本地测试资产检查 | 60 项通过，包含 A5 六文件固定基线哈希和 A2/A5 双向生成隔离。 |

报告保存在外部工作区 `report/pr1085_a2_split_20261110/results/solve_tri_a2_split.VbAEBmqW/`；首轮契约任务因上传遗漏公共辅助文件失败，已保留日志但不计入通过结果。原 A5 三份清单、generator、executor、YAML 与 PR 基线 `546859c7d36f0c01e81e97789b11bb02b38aa8ac` 内容一致。

本次确认入口拆分及上述用例结果；ATK 版本低于仓库 `26.8.8` 门禁，不能替代该版本要求的正式验收。未重跑性能采样、确定性和 sanitizer，也未在 A5 设备重跑；下节继续保留拆分前的历史证据和性能代价。

## 优化前基线对照

优化前基线固定为main分支原始代码，在 A2 上构建和运行。

两版本均使用本轮相同的 `atk_solve_tri_a2_perf.json`、`executor_solve_tri_a2.py` 及 common 输入工具，208/209 的 seed 分别为 `20261025`、`20261026`。在同一台 A2 910B3 的 NPU 2 顺序运行独立 ATK 进程；每例预热 3 次、采样 20 次。

| Case | 优化前基线 (us) | 基线验证候选 (us) | 候选/基线耗时比 | 加速比 | 耗时降低 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 208 定长 | 5738.73555 | 810.34130 | 0.141206 | 7.0819x | 85.88% |
| 209 变长 | 33443.87925 | 4539.59975 | 0.135738 | 7.3671x | 86.43% |

统计口径为原始 `op_summary*.csv` 中 SolveTri 的 `Task Duration(us)` 算术均值，四组均为 20 条，与各自 XLSX 均值一致；两版本各 2/2 执行成功。固定输入清单和生成器字节摘要，并额外保存共享 CPU 量化输入的 SHA256；后者是可复现 fixture 的指纹，不冒充每个 ATK worker 的输入转储。性能 profile 的 PID、设备和启动记录已与实际 worker 加载的独立安装目录 `libcust_opapi.so` 及 kernel 资产摘要关联，未复用前一日候选耗时。两版日志均存在 profiler schedule-stop 告警，已逐例核对完整的 20 条原始采样及 XLSX 一致性，不仅依据 ATK 退出码判断。

基线 wheel SHA256 为 `189964d4607904fdf883187bf93f132c11d21f7ed34c728513b6e27f597020c8`，基线验证候选为提交 `3b244a91`、wheel `54efbfd645b68b266efa30a4bb5347d7399093726e82a58fbdf37ec9cb8d2fdc`。原始报告、采样、构建记录及分析脚本保存在外部工作区 `report/pr1085_main_baseline_20261111/`，远端隔离目录为 `/opt/solve_tri_main_compare.0wHv2X0m/`；`comparison.json` 保存全部采样、分布统计和证据哈希。

本次数据证明候选在这两个 A2 场景相对用户指定的优化前基线有明确提升。该基线是在 A2 设备运行的优化前实现，不能作为 A5 设备测量值；Issue #660 的 A2/A3 对 A5 目标比值、必需性能 CI 检查和 ATK 版本要求仍需分别完成。本次未修改原 A5 测试资产。

## 最终输出融合复测

2026-11-11，保留 `GetUserWorkspace`、AIC 独立写 `Y`、16 到 32 阶折叠、跨核等待及 FFTS 收尾，仅将最终 `Y -> X -> UB` 改为从 `Y` 和 `X` 直接拼接到已有 UB 后转换输出。每 tile 少 4 KiB GM 写回及 4 KiB 重复读取，不增加 workspace 或 UB。

本轮以融合前普通 wheel `54efbfd645b68b266efa30a4bb5347d7399093726e82a58fbdf37ec9cb8d2fdc` 为对照，重新采样；其算子源码与 `34a1d319` 一致，该提交相对 `3b244a91` 仅更新本页。候选为 `34a1d319` 加输出融合，kernel header SHA256 为 `b69eb11d65e83a8f6c649b5d5d1cb3a0d6e3f78251427ce5dc8f1d698e39f242`。

### 同轮融合前后对比

加速比为融合前均值除以融合后均值；耗时降低为 `1 - 融合后均值 / 融合前均值`。

| Case | 融合前 (us) | 融合后 (us) | 加速比 | 耗时降低 |
| --- | ---: | ---: | ---: | ---: |
| 208 定长 | 813.70525 | 719.57635 | 1.1308x | 11.57% |
| 209 变长 | 4513.98715 | 3929.74565 | 1.1487x | 12.94% |

同一 A2 910B3、NPU 2、CANN 9.2.0、ATK `26.7.8.dev20260730`；相同 `_a2` 输入清单、executor、种子和单线程配置，两个普通产物顺序测量。每例预热 3 次、采样 20 次，统计原始 `Task Duration(us)` 均值；四组均实际启动 20 AIC，与 XLSX 均值误差不超过 `0.00015 us`。208 的样本标准差为融合前 `2.45460 us`、融合后 `1.54986 us`，209 为 `1.89366/11.40097 us`，两例耗时降低均明显大于本轮样本波动。两目标均优于融合前版本，达到本轮减少正确实现耗时的目标；不能据此认定 Issue #660 的跨平台目标达标。

### 最终版本与优化前基线对比

将前节优化前基线实验与本轮最终版本的已验证均值汇总如下。耗时比为最终版本除以优化前基线，加速比为其倒数。

| Case | 优化前基线 (us) | 最终输出融合 (us) | 最终/基线耗时比 | 加速比 | 耗时降低 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 208 定长 | 5738.73555 | 719.57635 | 0.125389 | 7.9752x | 87.46% |
| 209 变长 | 33443.87925 | 3929.74565 | 0.117503 | 8.5104x | 88.25% |

这是跨实验批次的参考对照：优化前数值来自 `report/pr1085_main_baseline_20261111/comparison.json`，最终数值来自 `report/pr1085_fused_output_20261111/comparison.json`。两批均使用同机 NPU 2、20 AIC、相同 CANN/ATK、A2 输入清单、种子及 3 次预热和 20 次采样；本轮没有重新运行优化前 main 基线。输出融合这项改动的直接收益以上面的同轮融合前后对比为准。

### 与历史最快中间候选对比

| Case | 历史中间候选 (us) | 最终输出融合 (us) | 耗时差 (us) | 耗时变化 |
| --- | ---: | ---: | ---: | ---: |
| 208 定长 | 708.36520 | 719.57635 | +11.21115 | +1.58% |
| 209 变长 | 3897.08485 | 3929.74565 | +32.66080 | +0.84% |

历史中间候选来自 2026-11-10 的修复前记录，本轮未重新构建和采样；上述差值只作历史参考，不能作为同轮严格回归结论，也不能宣称最终版本已完全恢复到该耗时。最终版本保留正确的 workspace 起点、独立 `Y` 结果和 FFTS 收尾，本轮验证结果如下。

本轮普通精度 208/208、模型精度 2/2 通过；另用不改默认清单的独立补充任务，验证 FP16/BF16、四种布局、33/63 行尾块和各三个种子，共 48/48 通过。插桩同源构建在 NPU 2 对 206/207 执行 racecheck、memcheck，各两次实际启动及匹配结束，均无内核 warning、skipped、越界、竞态或 FFTS 未复位；核数 profile 也来自本轮 NPU 2。独立采集的实际 worker PID、加载 API/kernel 摘要与报告关联，双工具原始日志及 XLSX 已在本地再次复核。

普通候选 wheel SHA256 为 `3335623cbc8328c3d5749f6c112378ea79d732b78e2dd0f665a2e469c0fb5890`，插桩版为 `c3f556903ff7ac5cba69f16d55c9dee73e5e34411f8d0435d80919ce76073951`。两版安装中的 header 与候选一致；两版本构建依赖逐文件摘要一致。证据、补充尾块清单、脚本和 `comparison.json` 保存在外部工作区 `report/pr1085_fused_output_20261111/`，远端目录 `/opt/solve_tri_fused_output.cmgaqw9F/`。

本次未改变原 A5 六份测试资产，未在 A3/A5 重测，也未重新执行显存峰值与确定性测试；ATK 版本限制仍适用，不能替代 `26.8.8` 正式验收。精度任务的设备关联来自同 PID 单进程日志中的唯一 NPU 2 配置；性能及 sanitizer 另有设备 profile 或 worker 记录。sanitizer 启动输出中的目录权限及 `LD_PRELOAD` 提示不属于内核检测告警；原始日志完整保留。

## A5 隔离范围

2026-11-12，本轮将新增的参数及变长元数据校验、结构化错误、workspace 计算、L0 转换失败保护、注册精度标志修改和 kernel 优化限定在 A2/A3。A5 保留 `546859c7d36f0c01e81e97789b11bb02b38aa8ac` 的原有 host 行为、workspace 计算、kernel 入口、`arch35` 实现及 `PrecisionReduceFlag(true)`；A2/A3 使用独立注册配置和 `false`。A5 原有输入约束仍须满足，本轮未增加其非法输入防御，也不据源码隔离宣称其原有 workspace 容量充分。

原 A5 六份测试资产与该基线逐字节一致；`arch35` 六份实现及 A5 kernel 入口按 LF 规范化后内容一致。原生 C++ 对照直接执行基线和候选的 tiling 函数，在普通 host 编译与显式 `__CCE_AICORE__=310` 编译下共完成 328 次 A5 比较，状态、key、分核、16 个 tiling 字段及 workspace 均一致。A2/A3 共 668 次状态检查通过，合法输入的 tiling 字段及新增 workspace 公式均符合预期。该测试使用 GERT/platform mock，不覆盖真实 SDK ABI 或设备计算；原始证据见外部工作区 `report/pr1085_a5_isolation_20261112/tiling_mock_results/` 和 `baseline_scope.json`。

A2、A3、A5 均已使用本轮冻结源码在 CANN 9.2.0 下构建通过。A2 910B3 的 NPU 6 上，全量精度 208/208、ACLNN 契约 32/32、模型场景精度 2/2 通过，逐项核对原始 XLSX，并关联实际 worker 加载的本轮 API 和 kernel 摘要。A2 安装中的两份 kernel 对象及两份 metadata 与隔离前输出融合版本的 SHA256 全部一致。

同轮在 NPU 6 顺序测量隔离前输出融合版本与本轮隔离版本，使用相同 `_a2` 输入、executor、种子、20 AIC、3 次预热及 20 次采样。统计原始 `Task Duration(us)` 均值，每例均为 20 条且与 XLSX 一致；本轮不重新测量优化前 main。

| Case | 隔离前 (us) | 隔离后 (us) | 耗时变化 |
| --- | ---: | ---: | ---: |
| 208 定长 | 717.47335 | 718.40725 | +0.13% |
| 209 变长 | 3935.28870 | 3920.93830 | -0.36% |

本轮两例观测耗时变化均小于 0.4%；208 样本标准差为隔离前 `4.90500 us`、隔离后 `7.71607 us`，209 为 `15.57844/20.10920 us`。A2 普通 wheel SHA256 为 `9bbe5b11679ed417caa0753a5f3e63627f0a424a2bcff924602bb5faffee883f`。原始报告、244 份逐文件校验后的证据及 `comparison.json` 位于外部工作区 `report/pr1085_a5_isolation_20261112/`，远端为 `/opt/solve_tri_a5_isolation.1yBECY6C/`；首轮因 NPU 2 被占用而在空闲检查处停止，未计入设备验证。

A3/A5 本轮仅交叉构建，尚无设备执行、精度或性能验证；未重跑 sanitizer、显存峰值或确定性。ATK 仍为 `26.7.8.dev20260730`，低于仓库 `26.8.8` 正式验收要求。全算子静态扫描仍报出原有 DFX guard/日志及旧测试位置问题，不将其描述为全部门禁通过。本节不改变上文已经归档的性能数据及其验证边界。

## 历史 A2 结果

2026-11-10，A2 910B3、CANN 9.2.0、ATK 26.7.8：31 条负例和 1 条正向契约通过；206/207、208/209 及 FP16 尾块 21/110 精度通过；长流水 racecheck/memcheck 各两次真实启用并通过，无 warning/skipped。普通 wheel SHA256 为 `41ff1f5b2be220c4934e4da93b808070d2fb10f7be9ba9cfb71c8a742e99dcde`。

| 场景 | 中间候选 (us) | 完整修复候选 (us) | 耗时变化 | 额外分配器显存 (bytes) |
| --- | ---: | ---: | ---: | ---: |
| 208 定长 | 708.36520 | 807.44605 | +13.99% | 100664832 |
| 209 变长 | 3897.08485 | 4524.89945 | +16.11% | 100664832 |

各性能均值来自 20 次原始设备采样，不能宣称修复没有性能代价。普通精度、性能和资源在 NPU 2，sanitizer 在同机同型号 NPU 3。核数 profile 来自 NPU 2，20 AIC 下推算最少每核 195 tile/13 batch，不能作为 sanitizer 在 NPU 3 实际分核的直接证据。原始证据位于外部工作区 `report/pr1085_kernel_recheck_20261110/ffts_candidate/`，不随仓库提交。

这里的差值包含 workspace 和最终写回等修复，不能单独归因于 FFTS 复位。未加入 FFTS 收尾的组合版本已测得 208/209 为 `813.93925/4541.15485 us`；加入收尾后的 `807.44605/4524.89945 us` 不支持 FFTS 本身造成上述回退的结论。

上述报告产生于本次平台拆分之前，使用旧文件名及注册名；它们保留为历史结果，不冒充 `_a2` 入口重新运行后的报告。A3/A5 和 A5 性能比值仍未验证。
