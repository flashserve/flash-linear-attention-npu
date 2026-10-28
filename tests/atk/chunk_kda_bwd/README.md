# ChunkKdaBwd public V2（Ascend950）

## 范围与版本

本包验收 `fla_npu.ops.ascendc.chunk_kda_bwd(..., implementation="optimized")` 的 public V2，
不测试或改写 `auto/legacy/optimized` 路由。代码基线为 upstream/main
`c9e2f1dbe6733d7d74c6bc4e1567e136f77da509`；候选增加 Finalize 五组 helper unroll-2、
A5/V2 原生 head-major db/dg、Stage7→9 FP32 中心因子复用，以及 A5 tiling ABI 修复。
本次复测环境为 CANN 9.1.0、ATK 26.9.8、Ascend950DT；安装目录和结果目录由下文变量指定，不固定本机临时路径。
只声明 A5/BF16 token、K=V=128、chunk=64、safe gate、`use_exp2=True` 的本包覆盖；
不覆盖 unsafe、FP16、initial state/dht，也不声称 A2/A3 已在设备上验收。

反向输出保持 `(dq, dk, dv, db, dg, dh0, dA, dbias)`；`dh0=None` 被过滤。
dense 输出为 `[B,H,T,*]`，packed 输出为 `[H,T,*]`。V2 直接产生公开 head-major db/dg；
独立 Finalize 默认仍为 token-major，未改变公开输出合同。

## 输入准备与 CPU 标杆

DUT 初始化显式调用公开 `npu_chunk_kda_fwd_prepare`、`npu_chunk_fwd_h` 和
`npu_chunk_kda_fwd_finalize`，构造 Aqk/Akk 与 saved caches，并检查前向输出有限。
这些调用只是准备反向输入，不用旧融合 `chunk_kda_fwd` 的小形状前向门控替代或覆盖本包结果。
saved case 传入完整 `gk/w/qg/kg/v_new/h`；recompute case 六项均传 `None`，只保留 Aqk/Akk。
norm-on 使用 Prepare 导出的 q_hat/k_hat/q_rstd/k_rstd。初始化、有限值检查不计入反向性能。

CPU golden 是独立 FP64 recurrent + autograd，最终返回 FP32。norm-on 时对原始 q/k
执行 `epsilon=1e-6` 的 L2Norm，按公开 q_hat/k_hat 的 BF16 dtype 量化后转回 FP64，
再做 recurrent/autograd；归一化反向与 cast 均保留。输入沿用已校准值域：
q/k/v std=0.05、raw_g std=1.25、beta∈[0.1,0.9]、A_log std=0.12、
dt_bias std=1.65 且均值 -3、dO std=51.2，先量化到输入 dtype。
精度使用 ATK 原生 `mixed_tolerance_bm` 和 BF16 output dtype overrides，不增加自定义阈值或 cosine 判定。

## 精度矩阵

每个逻辑 case 使用固定种子 `0 / 20260630 / 7`，共 12 逻辑 case、36 个 `(case, seed)`。
`atk_chunk_kda_bwd.json` 与 `gen_chunk_kda_bwd.py` 的 36 条真实输入/seed 完全一致；
CPU golden 与 NPU DUT 均消费 ATK 提供的张量，独立 CPU 自校验才使用无 provided 回退。
正常偶数头 H=8；recompute 满足 H≤256 且 H%8=0；另覆盖 odd-head、尾块、B2 和无 bias。

| 逻辑 id | accuracy id | 路径 | layout / norm | B/H/T 或 cu | beta / dt_bias |
| --- | --- | --- | --- | --- | --- |
| 0 saved_dense | 0–2 | saved | dense / off | B1/H8/T64 | BF16 / 有 |
| 1 saved_packed | 3–5 | saved | packed / off | H8/cu=[0,64,128] | BF16 / 有 |
| 2 saved_dense_norm_beta32 | 6–8 | saved | dense / on | B1/H8/T100 | FP32 / 有 |
| 3 saved_packed_norm | 9–11 | saved | packed / on | H8/cu=[0,97,130] | BF16 / 有 |
| 4 recompute_dense | 12–14 | recompute | dense / off | B1/H8/T100 | BF16 / 有 |
| 5 recompute_packed | 15–17 | recompute | packed / off | H8/cu=[0,64,128] | BF16 / 有 |
| 6 recompute_dense_norm_beta32 | 18–20 | recompute | dense / on | B1/H8/T64 | FP32 / 有 |
| 7 recompute_packed_norm | 21–23 | recompute | packed / on | H8/cu=[0,97,130] | BF16 / 有 |
| 8 saved_dense_odd_norm | 24–26 | saved | dense / on | B1/H3/T100 | BF16 / 有 |
| 9 saved_packed_odd | 27–29 | saved | packed / off | H3/cu=[0,97,130] | BF16 / 有 |
| 10 saved_dense_b2_nobias_tail | 30–32 | saved | dense / off | B2/H8/T100 | BF16 / None |
| 11 recompute_dense_b2_nobias_tail | 33–35 | recompute | dense / off | B2/H8/T100 | BF16 / None |

## TilingKey 与实际选择

V2 host 链路为 Prepare→Dhu→Finalize；recompute 之前再执行 Recompute→FwdH。
Finalize key 为 `(variable ? 2 : 1) + (hasRstd ? 2 : 0)`；新增内部 gate layout 属性不新增 key。
`_mss.json` 对 12 个逻辑场景各保留 seed=0 一条。下表保留 key 与用例映射；实际选择日志应随本次运行
归档到 `$RESULT_ROOT`，不沿用旧环境的硬编码日志路径。

| key | 条件 | accuracy 逻辑 id | mss id |
| --- | --- | --- | --- |
| 1 | dense，无 rstd | 0,4,10,11 | 0,4,10,11 |
| 2 | packed，无 rstd | 1,5,9 | 1,5,9 |
| 3 | dense，有 rstd | 2,6,8 | 2,6,8 |
| 4 | packed，有 rstd | 3,7 | 3,7 |

这些是 public V2 组合内的 key/用例映射，不是内存检测通过结论，也不替代各子算子独立全 key 验收。
旧直接基准的小形状前向非有限值不属于本包的反向失败；本包使用上述三阶段输入准备。
ABI 修复在 A5 设置 raw tiling 为 `sizeof(TilingData)+8`，与仓内 `MergeFwdBwd` 的
CANN `GET_TILING_DATA opParaSize` 处理一致，arch22 路径不变。

## 正式结果与证据

`RESULT_ROOT` 指向本次 `pr913` 结果目录；下文证据也可按工作区相对路径
`runs/pr_delivery_20261028/pr913` 解读。除特别注明外，结果来自本次 A5 实机复测。

| 动作 | 本次真实结果 | 证据 |
| --- | --- | --- |
| A5 build | 通过 | `runs/pr_delivery_20261028/pr913_build.log` |
| A2/A3 交叉编译 | 均通过；未执行 A2/A3 实机测试 | `$RESULT_ROOT/a2_finalize_build.log`；`$RESULT_ROOT/a3_finalize_build.log` |
| wheel 安装匹配 | **1318/1318** 个 `fla_npu` 文件匹配，安装差异为 0 | `$RESULT_ROOT/install_verify.log` |
| fresh 基线对象比较 | 非 Finalize 的 **183** 个对象名称/hash 全同；两侧 Finalize 各 **4** 个对象；scoped 比较通过 | `runs/pr_delivery_20261028/base/object_compare_final.log` |
| smoke | **20/20 bit-exact**（saved 10/10、recompute 10/10） | `$RESULT_ROOT/smoke/compare.log` |
| public V2 ATK accuracy | **36/36 PASS**，12 逻辑 case × 3 seeds | `$RESULT_ROOT/atk/public_gm/accuracy_case_detail.json` |
| saved/optimized 扩展精度 | **63/63 PASS**；排除由 legacy/拒绝路径看护的 9 个不适用 id | `$RESULT_ROOT/atk/saved_opt63/case_detail.json`；`$RESULT_ROOT/atk/saved_opt63/atk_run.log` |
| accuracy_dc | **12/12 × 50 PASS** | `$RESULT_ROOT/atk/public_gm/determinism_check_actual.log` |
| CPU reference | 单步 norm/packed 与确定性输入构造自校验通过 | `$RESULT_ROOT/atk/cpu_reference.log` |
| CPU recurrent | 六项输出通过既定容差 | `$RESULT_ROOT/numeric/accuracy.log` |
| 大形状位比较 | **15/15** 张量逐位一致 | `$RESULT_ROOT/numeric/compare.log` |
| 数值矩阵 | 16 case × 15 张量，**240/240 bit-exact** | `$RESULT_ROOT/numeric/matrix_compare.log` |
| saved/recompute rstd layout | 8 case、**80/80** 张量逐位一致 | `$RESULT_ROOT/layout_norm/compare.json` |
| 独立默认布局 direct | 20 case、**140/140** 张量逐位一致 | `$RESULT_ROOT/direct_matrix/compare.json` |

确定性结论来自真实报告的复核，未以脚本退出前的“未找到报告（跳过）”或 `atk task success`
代替报告核验。合法 saved 集合排除 `{42,43,44,51,52,53,57,58,59}`，不将整包 72 例冒充 V2 正例。
旧 gatekeeper 默认值为 legacy，因此 saved 扩展验收必须显式设置 `KDA_GK_IMPL=optimized`，不能只读
63/63 数字推断路由。

### 性能模型 case

`atk_chunk_kda_bwd_perf.json` 保留 B1/H96/T8192/K=V=128/C64，BF16 token、FP32 raw gate、
safe gate、dense、norm off，saved/recompute 各一条。ATK 的前向缓存构造不计入性能。
同输入、同口径的 `msprof op_summary.Task Duration(us)` 原始结果在
`$RESULT_ROOT/perf/perf_saved.json` 和 `$RESULT_ROOT/perf/perf_recompute.json`；saved 为 6 轮、
recompute 为 3 轮。bwd kernel sum 包含 Prepare/Dhu/Finalize/附属转换，recompute 另含 Recompute/FwdH。

| 路径 | Finalize 基线中位数 (µs) | Finalize 候选中位数 (µs) | Finalize 配对差中位数 (µs) | Finalize 比值变化 | bwd 配对差中位数 (µs) |
| --- | ---: | ---: | ---: | ---: | ---: |
| saved | 5357.4075 | 5110.0340 | -247.2200 | -4.617411% | -656.0965 |
| recompute | 5359.6560 | 5106.4650 | -253.1910 | -4.724016% | -665.7660 |

“配对差中位数”是逐轮 `candidate - baseline` 差值的中位数；“比值变化”直接取 JSON 的
`kernel_delta_percent`，即两侧 Finalize 中位数的比值减 1。二者不是同一统计量，不能用配对差中位数
除以基线中位数替代。bwd 一栏同样只报告 JSON 中的逐轮配对差中位数，不额外推算比值。

### E2E 结果

`$RESULT_ROOT/e2e/summary.json` 的 saved/recompute 各有 3 轮、每轮 20 次采样。下表转录其
`summary` 的五类指标；`event`/`wall` 的基线→候选为两侧中位数（ms），随后分别是逐轮配对差中位数
（ms）和 JSON 的中位数比值变化，三者不混用。

| 路径 | 指标 | event 基线→候选 (ms) | event 配对差中位数 (ms) | event 比值变化 | wall 基线→候选 (ms) | wall 配对差中位数 (ms) | wall 比值变化 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| saved | fwd | 3.474484→3.641146 | +0.154445 | +4.796729% | 3.886780→3.997450 | +0.093235 | +2.847344% |
| saved | bwd | 8.450121→7.868250 | -0.562238 | -6.885957% | 9.035080→8.210520 | -0.824560 | -9.126205% |
| saved | bwd_with_beta_chain | 8.453660→7.881665 | -0.530789 | -6.766246% | 9.150370→8.275275 | -0.790330 | -9.563492% |
| saved | bwd_full | 8.873215→8.312673 | -0.593772 | -6.317243% | 9.600885→8.720840 | -0.709985 | -9.166292% |
| saved | e2e | 12.077152→11.538548 | -0.533109 | -4.459696% | 12.572150→11.904250 | -0.732710 | -5.312536% |
| recompute | fwd | 3.496078→3.491073 | -0.005974 | -0.143133% | 3.998615→3.922730 | -0.085430 | -1.897782% |
| recompute | bwd | 10.084047→9.358140 | -0.697776 | -7.198562% | 11.011455→9.890170 | -1.121285 | -10.182895% |
| recompute | bwd_with_beta_chain | 10.227564→9.463698 | -0.763866 | -7.468699% | 10.918485→9.922955 | -0.995530 | -9.117838% |
| recompute | bwd_full | 10.526616→9.912820 | -0.684624 | -5.830893% | 11.190110→10.378255 | -0.800120 | -7.255113% |
| recompute | e2e | 13.834198→13.188031 | -0.634184 | -4.670793% | 14.452500→13.716265 | -0.734200 | -5.094170% |

### 显存

生产形状下先构造并保留对应前向缓存，三次反向预热后清空缓存、重置峰值，再测一次公开反向。
四份原始结果为 `$RESULT_ROOT/memory/base_saved.json`、`$RESULT_ROOT/memory/pr_saved.json`、
`$RESULT_ROOT/memory/base_recompute.json` 和 `$RESULT_ROOT/memory/pr_recompute.json`。
dg 在两侧均为 `[1,96,8192,128]` FP32，输出布局没有变化。

| 路径 | 基线 allocated / reserved 峰值 | 候选 allocated / reserved 峰值 | allocated / reserved 均下降 |
| --- | ---: | ---: | ---: |
| saved | 5997.473 / 6018.0 MiB | 5669.473 / 5690.0 MiB | **328 / 328 MiB** |
| recompute | 5669.476 / 5690.0 MiB | 5477.476 / 5498.0 MiB | **192 / 192 MiB** |

## 执行

以下命令从仓库根目录执行。`ENV_DIR` 为安装环境根目录，`ATK_ROOT` 为 ATK 根目录，
`RESULT_ROOT` 为本次结果目录；三者按实际环境设置，不在示例中写本机临时路径。
`CANN_SET_ENV` 默认使用标准 CANN 路径，也可覆盖。

```bash
: "${ENV_DIR:?set ENV_DIR to the installed fla_npu environment}"
: "${ATK_ROOT:?set ATK_ROOT to the ATK root}"
: "${RESULT_ROOT:?set RESULT_ROOT to the pr913 result directory}"
CANN_SET_ENV="${CANN_SET_ENV:-/usr/local/Ascend/cann-9.1.0/set_env.sh}"
source "$CANN_SET_ENV"
export PATH="$ATK_ROOT/bin:$PATH"
export PYTHONPATH="$ATK_ROOT:$ENV_DIR:${PYTHONPATH:-}"
export ASCEND_CUSTOM_OPP_PATH="$ENV_DIR/fla_npu/opp/vendors/fla_npu_transformer:$ENV_DIR/fla_npu/opp/vendors/fla_npu_transformer/op_api/lib"
export FLA_NPU_DISABLE_PTH=1
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
NPU_DEVICE_ID="${NPU_DEVICE_ID:-0}"
export ATK_OUTPUT_ROOT="$RESULT_ROOT/atk/public_gm"

GEN_CASES_DTYPE_NUMBERS=36 bash tests/atk/run_test_cpu.sh -op=chunk_kda_bwd -scope=gen_cases -soc=ascend950
# 按阶段顺序执行，避免 ATK 端口复用；mss 结果必须另看设备日志，不能只看退出码。
bash tests/atk/run_test_cpu.sh -op=chunk_kda_bwd -npu_device_id="$NPU_DEVICE_ID" -scope=accuracy -soc=ascend950
bash tests/atk/run_test_cpu.sh -op=chunk_kda_bwd -npu_device_id="$NPU_DEVICE_ID" -scope=performance -soc=ascend950
bash tests/atk/run_test_cpu.sh -op=chunk_kda_bwd -npu_device_id="$NPU_DEVICE_ID" -scope=determinism -soc=ascend950
```

正式 accuracy 使用默认 GM-on；saved/optimized 扩展精度另需显式设置 `KDA_GK_IMPL=optimized`。
运行生成的 `atk_output/`、`result/`、日志和 `__pycache__` 均不提交。

## 未闭环项

- **整链 msSanitizer 尚未闭环**，不能将当前候选标为整链 memcheck/racecheck 通过或正式全量验收完成。
  上述普通执行、bit-exact 和显存峰值结果不能替代完整 sanitizer 证据。
- A2/A3 仅完成交叉编译，**未跑 A2/A3 实机**；**整体 ST 未跑**。
