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

按main分支原始代码作为优化前基线。

两版本均使用本轮相同的 `atk_solve_tri_a2_perf.json`、`executor_solve_tri_a2.py` 及 common 输入工具，208/209 的 seed 分别为 `20261025`、`20261026`。在同一台 A2 910B3 的 NPU 2 顺序运行独立 ATK 进程；每例预热 3 次、采样 20 次。

| Case | 优化前基线 (us) | 当前候选 (us) | 候选/基线耗时比 | 加速比 | 耗时降低 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 208 定长 | 5738.73555 | 810.34130 | 0.141206 | 7.0819x | 85.88% |
| 209 变长 | 33443.87925 | 4539.59975 | 0.135738 | 7.3671x | 86.43% |

统计口径为原始 `op_summary*.csv` 中 SolveTri 的 `Task Duration(us)` 算术均值，四组均为 20 条，与各自 XLSX 均值一致；两版本各 2/2 执行成功。固定输入清单和生成器字节摘要，并额外保存共享 CPU 量化输入的 SHA256；后者是可复现 fixture 的指纹，不冒充每个 ATK worker 的输入转储。性能 profile 的 PID、设备和启动记录已与实际 worker 加载的独立安装目录 `libcust_opapi.so` 及 kernel 资产摘要关联，未复用前一日候选耗时。两版日志均存在 profiler schedule-stop 告警，已逐例核对完整的 20 条原始采样及 XLSX 一致性，不仅依据 ATK 退出码判断。

基线 wheel SHA256 为 `189964d4607904fdf883187bf93f132c11d21f7ed34c728513b6e27f597020c8`，候选为 `54efbfd645b68b266efa30a4bb5347d7399093726e82a58fbdf37ec9cb8d2fdc`。原始报告、采样、构建记录及分析脚本保存在外部工作区 `report/pr1085_main_baseline_20261111/`，远端隔离目录为 `/opt/solve_tri_main_compare.0wHv2X0m/`；`comparison.json` 保存全部采样、分布统计和证据哈希。

本次数据证明候选在这两个 A2 场景相对用户指定的优化前基线有明确提升。该基线是在 A2 设备运行的优化前实现，不能作为 A5 设备测量值；Issue #660 的 A2/A3 对 A5 目标比值、必需性能 CI 检查和 ATK 版本要求仍需分别完成。本次未修改原 A5 测试资产。

## 历史 A2 结果

2026-11-10，A2 910B3、CANN 9.2.0、ATK 26.7.8：31 条负例和 1 条正向契约通过；206/207、208/209 及 FP16 尾块 21/110 精度通过；长流水 racecheck/memcheck 各两次真实启用并通过，无 warning/skipped。普通 wheel SHA256 为 `41ff1f5b2be220c4934e4da93b808070d2fb10f7be9ba9cfb71c8a742e99dcde`。

| 场景 | 中间候选 (us) | FFTS 修复候选 (us) | 耗时变化 | 额外分配器显存 (bytes) |
| --- | ---: | ---: | ---: | ---: |
| 208 定长 | 708.36520 | 807.44605 | +13.99% | 100664832 |
| 209 变长 | 3897.08485 | 4524.89945 | +16.11% | 100664832 |

各性能均值来自 20 次原始设备采样，不能宣称修复没有性能代价。普通精度、性能和资源在 NPU 2，sanitizer 在同机同型号 NPU 3。核数 profile 来自 NPU 2，20 AIC 下推算最少每核 195 tile/13 batch，不能作为 sanitizer 在 NPU 3 实际分核的直接证据。原始证据位于外部工作区 `report/pr1085_kernel_recheck_20261110/ffts_candidate/`，不随仓库提交。

上述报告产生于本次平台拆分之前，使用旧文件名及注册名；它们保留为历史结果，不冒充 `_a2` 入口重新运行后的报告。A3/A5 和 A5 性能比值仍未验证。
