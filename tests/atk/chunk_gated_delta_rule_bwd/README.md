# ChunkGatedDeltaRuleBwd ATK 测试说明

本目录验证 `chunk_gated_delta_rule_bwd` 融合接口。NPU 节点只调用一次公开融合接口；
CPU 节点使用同一组冻结输入，独立串联四个阶段，并提供 FP64 golden 与 FP32
同精度 benchmark 两条 CPU 标杆；NPU DUT 与两条标杆均比较最终有效梯度。

## 标杆与输出

CPU 标杆依次计算 `ChunkGdnBwdIntra`、`ChunkFwdH`、
`ChunkGatedDeltaRuleBwdDhu` 和 `ChunkGatedDeltaRuleBwdFinalize`。FP64 golden 保留
双精度阶段计算，FP32 benchmark 按 BF16 输入、FP32 累加和阶段落盘规则复现同精度链，
避免把 NPU 中间结果作为标杆。

精度使用 ATK 原生 `cv_fused_double_benchmark`，采用 L1 误差比标准：
`max_re_ratio=5`、`avg_re_ratio=1.5`、`root_mean_squared_ratio=1.5`，不使用
`mixed_tolerance_bm`。

有效输出顺序为 `dq, dk, dv, d_beta, d_g`；传入 `initial_state` 时额外比较
`dh0`。其中 `d_beta` 和 `d_g` 固定使用 BSND `[B,T,HV]`，其余有效输出跟随
对应输入布局。保留接口输出 `d_a_log` 和 `d_dt_bias` 当前为空，不参与数值比较。

## 覆盖范围

- A2/A3 与 A5，输入主类型为 BF16，`K=V=128`，`chunk_size=64`。
- 覆盖 `BNSD/BSND/NTD/TND`，定长与变长，以及完整块、尾块和多块。
- 覆盖 `G=HV/HK` 的 `1/2/3/4`。
- 覆盖 BF16/FP32 标量输入、初始状态、末状态梯度和 `state_v_first`。
- 覆盖 Q/K L2Norm backward、beta sigmoid backward 及保留输入。
- `use_exp2=true`、`use_gate_in_kernel=false`。

## 用例矩阵

`gen_chunk_gated_delta_rule_bwd.py` 定义 12 个确定性 profile。矩阵以较小 shape
覆盖接口分支和四阶段组合语义，避免大 shape 使 CPU 标杆成为测试瓶颈。

## 执行

```bash
GEN_CASES_DTYPE_NUMBERS=12 bash tests/atk/run_test_cpu.sh \
  -op=chunk_gated_delta_rule_bwd -scope=gen_cases

# CPU 双标杆：NPU DUT + CPU FP32 benchmark + CPU FP64 golden
bash tests/atk/chunk_gated_delta_rule_bwd/scripts/run_double_benchmark.sh 0
```

双标杆脚本会检查三路角色数量和 XLSX 中的逐用例精度结果；任一执行或精度失败均返回非零。
本算子的精度入口使用上述脚本，以同时启动三个角色。

脚本根据 DUT 型号选择 FP32 benchmark 的中间存储顺序：arch22 使用
RecomputeWUFwd + ChunkBwdDvLocal，arch35 使用 Intra；两者的 BF16 舍入边界不同。
arch22 的 W/U 先舍入 `k*beta*exp(g)`、`v*beta` 再乘 A，local-dV 先舍入 KQ；
不能用 arch35 的 `A*beta` 舍入顺序替代。FP64 golden 保持完整精度计算，三路输入
在统一的输入 dtype 下生成后再提升计算精度。输入值域和原生双标杆 L1 阈值保持不变。

arch35 的 Intra、FwdH、Finalize 向量 BF16 写回使用 `CAST_ROUND`，恰好位于
中点时向远离零的方向舍入；对应 CPU FP32 benchmark 按该模式量化。Cube 写回及
DHU 的 `CAST_RINT` 保持偶数舍入，FP64 golden 不引入这些中间量化。
直接调用 ATK（包括 `accuracy_lt`）时，显式设置 `GDN_ATK_BENCHMARK_ARCH=arch22`
或 `arch35`；CPU benchmark 节点拒绝缺少该配置的调用。

布局改造另用同一组输入对比 BNSD/BSND、NTD/TND 的逐字节结果，同时检查输出 shape、dtype、
连续性和有限值。补充场景覆盖 T=1/63/64/65、G=3、多 batch、空序列、混合标量 dtype、
两种指数模式、T=H、非连续输入和可追踪数值；这项对照不替代 CPU 双标杆精度测试。

空序列的旧实现会将 dh0 留为零，与已有 dh0=dht 契约不符。本次同时修复该问题。
与旧包冻结数据比较时，额外传入 `--baseline-empty-dh0-fix`，仅对空序列 dh0 使用冻结
输入中的 dht 作为预期（无 dht 时为零）；其它区域仍逐字节比较旧包输出。CPU 双标杆
保持完整状态梯度比较，不屏蔽空序列。

```bash
python tests/atk/chunk_gated_delta_rule_bwd/scripts/check_layout_equivalence.py \
  --device 0 --save-dir <new_result_dir>
# 安装旧主线包时冻结输入和 head-major 输出，安装新包后读取并逐字节对照
python tests/atk/chunk_gated_delta_rule_bwd/scripts/check_layout_equivalence.py \
  --device 0 --write-control-dir <baseline_control_dir>
python tests/atk/chunk_gated_delta_rule_bwd/scripts/check_layout_equivalence.py \
  --device 0 --read-control-dir <baseline_control_dir> --save-dir <new_result_dir>
python tests/atk/chunk_gated_delta_rule_bwd/scripts/check_layout_equivalence.py \
  --emit-supplemental-cases <supplemental_json>
GDN_ATK_CASE_JSON=<supplemental_json> \
  bash tests/atk/chunk_gated_delta_rule_bwd/scripts/run_double_benchmark.sh 0
```

公开 C ABI 的参数拦截和非连续输出兼容检查：

```bash
python tests/atk/chunk_gated_delta_rule_bwd/scripts/check_aclnn_layout_contract.py --device 0
# A2/A3 额外回归 dq/dk/dv/dh0 非连续输出及非零 storage offset
python tests/atk/chunk_gated_delta_rule_bwd/scripts/check_aclnn_layout_contract.py --device 0 --output-views
```


## 布局模板、内存与性能回归

`generate_layout_key_cases.py` 生成 64 组模板分支，覆盖两种标量 dtype、L2Norm、
sigmoid、两种指数模式、G=1/G=3 调度和两种物理布局。`--include-layout-regression`
同时加入原 12 组和补充 14 组的两种布局，共 116 组；不修改输入值域或双标杆阈值。
可使用 `--seed-offset 0/1000/2000` 生成三组独立数值样本。
追加 `--complete-public-keys` 时，再补充 BF16 fixed G=2/4、packed G=1/2/3/4
的两种布局，以及 FwdH 无初态、指数模式和标量 dtype 的剩余组合，共 140 组；
原 116 个 case id 保持不变。融合接口会将 g/beta 提升为共同计算 dtype，该矩阵据此
覆盖 Intra 的固定/变长、G、标量 dtype 和布局模板，以及 FwdH 的可达模板组合。

```bash
python tests/atk/chunk_gated_delta_rule_bwd/scripts/generate_layout_key_cases.py \
  --out <key_cases_json> --include-layout-regression --complete-public-keys
GDN_ATK_CASE_JSON=<key_cases_json> \
  bash tests/atk/chunk_gated_delta_rule_bwd/scripts/run_double_benchmark.sh 0
# 内存检查须先编译并安装匹配当前源码的 sanitizer 内核；不使用该构建测性能。
mssanitizer --tool=memcheck -- atk node --name npu_dut --backend npu --devices 0 task \
  -c <key_cases_json> -p tests/atk/chunk_gated_delta_rule_bwd/executor_chunk_gated_delta_rule_bwd.py \
  --task run --mssanitizer -msl <memcheck_log>
```

性能脚本默认执行原 12 组及 4 组固定输入的布局对照，预热三次、采集一次。
每次 executor 调用后追加 Add 标记，仅汇总融合接口前缀下的计算和转换任务，
排除输入构造和 Add 标记。性能依据为 msprof 的 `Task Duration(us)`，不使用 Python wall time。
新旧版本使用相同脚本、输入和可见设备，分别保留完整 profiler CSV。

```bash
msprof --output=<profile_dir> python \
  tests/atk/chunk_gated_delta_rule_bwd/scripts/profile_layout_regression.py --out-dir <manifest_dir>
python tests/atk/chunk_gated_delta_rule_bwd/scripts/summarize_layout_profile.py \
  <op_summary_csv> <manifest_dir>/manifest.json --out <summary_csv>
python tests/atk/chunk_gated_delta_rule_bwd/scripts/compare_layout_profiles.py \
  <before_summary_csv> <after_summary_csv> --out <comparison_csv>
```

比较脚本同时验证 BSND/TND 每次调用各减少 6 个 Transpose，BNSD/NTD 的转置数保持一致。
增量编译前应确认构建目录的 kernel 源码副本与仓库源码一致，并刷新受影响内核的编译缓存；
仅重装已有 run 包或刷新 opapi 不能证明内核头文件改动已经生效。

### 实际模板选择记录

使用 140 组矩阵各执行一次，按 `task_time` 中的真实 kernel 名称建立 key 到 case id 的映射：
[arch22](layout_key_coverage.arch22.json)、[arch35](layout_key_coverage.arch35.json)。
映射同时保存原始 task CSV 和 case manifest 的 SHA256。

| 执行链 | 可达模板轴 | 实测 key 数 |
| --- | --- | ---: |
| arch22 WU | 两种 qkv 布局 | 2 |
| arch22 LocalDV | 指数模式、qkv 布局 | 4 |
| arch22 FwdH | gate dtype、指数模式、状态 dtype、dense T16、qkv 布局 | 32 |
| arch35 Intra | fixed/packed、G=1/2/3/4、共同标量 dtype、qkv 布局 | 32 |
| arch35 FwdH | gate dtype、指数模式、状态 dtype、qkv 布局 | 16 |
| DHU | gate dtype、qkv 布局 | 每平台 4 |
| arch22 Finalize | gate dtype、L2Norm、sigmoid、指数模式、owner、qkv 布局 | 64 |
| arch35 Finalize | gate dtype、L2Norm、sigmoid、指数模式、qkv 布局 | 32 |

```bash
python tests/atk/chunk_gated_delta_rule_bwd/scripts/summarize_layout_key_coverage.py \
  <task_time_csv> <manifest_json> --arch arch22 --out <coverage_json>
```
