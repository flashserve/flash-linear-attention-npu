> 性能范围纠正（2026-11-25）：H96历史结果不对应详设模型。M1 HK16/HV128/T11264中位2473.7195us；M2用户授权16×1024变长探针中位3610.0920us，各100样本。最新报告：/home/yqw/atk/prepare_models_design/design_model_performance_report.md。

# chunk_kda_bwd_prepare GVA 开发验证报告

日期：2026-11-24；设备：179.183.2.2，Ascend910B3（A2），设备 0。
源码目录：`/home/yqw/chunk_kda_bwd_prepare_gva`；分支：`codex/bwd-prepare-gva`。
基础提交：`c4ca968446089b4293248483eecd56a0eaa66a18`；本轮变更单独保存为补丁。

## 实现与范围

新增 keyword-only `gva_ratio=1`，贯通 Python/Stable ABI、ACLNN/L0、OpDef、tiling 和 kernel。
校验整数 R∈[1,16] 且 HV%R=0，tiling 保存 HK=HV/R。R=1 保留原 key，R=2..16 走独立编译 GVA 分支。
四个输入与三个输出均保持 HV 域，不广播或归并 head。保留原四 head Cube/Vector 算法、缓冲和事件协议。
完整 backward 的旧调用方显式传 1；本轮不宣称完整 backward 支持原始 HK<HV 的 Q/K 输入。
新增属性改变 C ABI，交付的 host/kernel/Stable/wheel 为同步重编译产物。

按用户最终要求：精度 **200 条×1 seed**，确定性 **20 条**，内存 **20 条**，性能 **2 条**。
精度 R=2..16，HK=1/3，最大 HV=48、T=257，覆盖四布局、定/变长、两个状态方向和尾块。
DC/MSS 为同一份 20 条精简资产，覆盖全部 A2 GVA key 16..23。
用户更新后的反向标杆来源：`/home/yqw/atk/prepare`；原包保留不动。
输入 BF16 正态分布 std=0.05、输入种子与原生双标杆 max/avg/RMS=5/1.5/1.5、小值域错误数比例=2 均未放宽。
FP32 Bench 与 FP64 Golden 统一使用独立 CPU 数学标杆；全量三路输出比较，包含上三角和尾列。

## 验证结果

| 检查 | 总数 | 通过 | 失败 | 未执行 |
|---|---:|---:|---:|---:|
| GVA 精度（每条一个固定 seed） | 200 | 200 | 0 | 0 |
| 确定性（每条相同输入重复 3 次） | 20 | 20 | 0 | 0 |
| MSS memcheck（插桩包） | 20 | 20 | 0 | 0 |

额外显存峰值：**19,494,400 B = 18.5913 MiB**，低于 200,000,000 B。
测量对象为上述 20 条：输入常驻并同步，重置 allocator 峰值，再扣除三路必需输出的精确元素字节数；保留 allocated/reserved 口径。
不能将此小用例峰值声称为两个模型 shape 的实测峰值。A2 自定义 workspace 为 65536×blockDim B，外加系统 workspace 和元数据。

MSS 报告 Pass，未报告越界等内存错误；原始日志另有 **690 条 FFTS_BASE_ADDR 未复位警告**。
这些寄存器警告完整保留，未屏蔽、未归因成已确认的误报；本结果不等同“无任何工具警告”。

补充：CPU 精确独立 oracle 16 个布局/模式/方向组合通过；R=1/2/8 同 HV 张量逐位等价与非法 ratio 检查通过；
30 项原生 ACLNN 错误与公开异常传播通过；host R=1..16 key/非法属性检查通过；公开 backend=stable，无 fallback。
最终正常包重新构建的 kernel 与本轮精度/性能所用 kernel SHA256 完全一致。

## msprof 性能

保留更新包中的两个典型 case 原始 shape、seed、scale、state 属性，R=8，HK=12，HV=96。
ATK `performance_device --performance_data 10,20,20 --save_data profile` 发起采集，随后逐份执行
`msprof --export=on --output=<PROF目录>` 导出原始 op_summary。
统计目标是单个 `ChunkKdaBwdPrepare` 主 kernel 的 **Task Duration(us)**，每条预热 10 次、有效采样 20 次，1 轮。

| case | B/HV/T | state_v_first | 中位数 us | 均值 us | p95 us |
|---|---|---|---:|---:|---:|
| 198 | 1/96/8192 | False | 1340.777 | 1339.683 | 1343.926 |
| 199 | 1/96/16384 | True | 2810.106 | 2811.207 | 2820.236 |

无已冻结的原生算子基线，因此只报告实测耗时，不宣称 0.4x 或跨平台性能达标。
A3/A5 尚无本轮设备验收；测试文件的 `_a3` 后缀继承交付包命名，实机结果只对应 A2。

## 产物与复现

- 正式测试资产：`/home/yqw/chunk_kda_bwd_prepare_gva/tests/atk/chunk_kda_bwd_prepare_a3/`
- 唯一可编辑数学标杆：`tests/atk/chunk_kda_bwd_prepare/reference.py`；其它入口只做无公式转发。
- 运行目录与原始报告：`/home/yqw/atk/prepare_gva200/`
- 正常 wheel：`/home/yqw/chunk_kda_bwd_prepare_gva/dist/flash_linear_attention_npu_a2-26.10.0+codex.bwd.prepare.gva.devc4ca968-py3-none-manylinux_2_34_aarch64.whl`
- 插桩 wheel：同源码初次构建的 `dist-sanitizer/` 包；实际加载的插桩包在运行目录 `runtime-sanitizer/`。
- 结构化汇总：`accuracy_summary.json`、`determinism_summary.json`、`memory_summary.json`、`peak_memory20.json`、`performance_summary.json`。
- 版本与二进制证据：`environment.json`、`final_binary_verification.json`、`delivery_manifest.json`。

执行：`bash run_accuracy200_accepted.sh` → `bash run_determinism20.sh` → `bash run_memcheck20.sh`
→ `bash run_performance.sh`，再执行 `export_msprof.py`。脚本使用隔离包，不覆盖其他任务安装环境。
MSS 脚本通过 `--log-file` 与 ATK `-msl` 对接同一份原始日志。
生成器默认且仅允许 `--seed-count 1`，三个正式 JSON 的生成器校验均通过（200/20/2）。

历史三 seed 展开、比较方向排查和数值路径实验仅作历史记录；均不作为最终范围、最终实现或验收结论。
历史额外 seed 试验确有 3 个有效输入组合触发原生小值域错误数比例失败，未被修复或改判为通过；具体 case/seed 见 historical_extra_seed_failures.json。用户明确将最终范围限定为 200 条单 seed，本报告的通过结论不覆盖这些额外 seed。
ATK 原生双标杆的首节点为 Actual：NPU 在前，CPU Bench 在后；报告比较列以第二节点名称命名。
