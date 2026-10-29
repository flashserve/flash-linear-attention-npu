# ChunkKdaBwd public V2（Ascend950）

## 范围与版本

本包验收 `fla_npu.ops.ascendc.chunk_kda_bwd(..., implementation="optimized")` 的 public V2，
不测试或改写 `auto/legacy/optimized` 路由。代码基线为 upstream/main
`c9e2f1dbe6733d7d74c6bc4e1567e136f77da509`，当前生产 head 为
`1c132d95`。性能优化为 Finalize 五组 VF 同一成对循环内按操作 A/B 交错并在循环外保留独立单尾、
Stage7→9 FP32 中心因子复用，以及 A5/V2 原生 head-major db/dg。A5 raw tiling ABI 修复、arch22
布局拒绝、host/ATK 测试与文档整改均属非性能修改，不分配性能收益。
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
CPU golden 与 NPU DUT 均消费 `input_data.kwargs` 中的真实 ATK 张量和属性，独立 CPU 自校验才使用
`provided=None` 回退。用例设计与 TilingKey/mss 映射详见 `test_case_design_report.md`。
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
CANN `GET_TILING_DATA opParaSize` 处理一致，arch22 token-major 转发不变。arch22 对
`gate_output_head_major=true` 使用 `OP_LOGE_WITH_INVALID_ATTR` 拒绝，不用 scoped build 缺宏的
`OP_LOGE_FOR_INVALID_VALUE`；规范 host 测试位于 `tests/operators/chunk_kda_bwd_finalize`。

## 正式结果与证据

`RESULT_ROOT` 指向 `runs/pr_final_20261029/pr913`，`REGRESSION_ROOT` 指向
`$RESULT_ROOT/final_regression_r3`，`DFX_ROOT` 指向
`runs/pr913_finalize_dfx_20261029/build_tests_operators`，`REVIEW_ROOT` 指向
`runs/pr_final_20261029/review`。除特别注明外，结果来自当前生产 head 的最终 A5 组合复测。

| 动作 | 最终真实结果 | 证据 |
| --- | --- | --- |
| A5 build | 通过 | `$RESULT_ROOT/build_final.log` |
| A2/A3 交叉编译 | 均通过；未执行 A2/A3 实机测试 | `$RESULT_ROOT/a2_finalize_build_r2.log`；`$RESULT_ROOT/a3_finalize_build_r2.log` |
| 最终 wheel 与路由 | SHA256 `21568e8a0f6d5530e47320211765faf6ed3d7f21acc88831a72781d5d089fc14`；**1318/1318** 文件匹配、差异为 0、路由不变 | `$RESULT_ROOT/install_verify_final.log` |
| 对象与运行时隔离 | **183** 个非 Finalize 对象名称/hash 全同；隔离 `fla_npu`/OPP/opapi 已核验 | `$RESULT_ROOT/install_verify_final.log`；`$RESULT_ROOT/runtime_final.log` |
| host tiling 正负测 | **6/6 PASS**，CTest **1/1 PASS**；不替代 A2/A3 实机 | `$DFX_ROOT/test_binary.log`；`$DFX_ROOT/ctest.log` |
| smoke | **20/20 bit-exact**（saved 10/10、recompute 10/10） | `$REGRESSION_ROOT/smoke/compare.json` |
| public V2 ATK accuracy | **36/36 PASS**，12 逻辑 case × 3 seeds | `$REGRESSION_ROOT/public/accuracy_case_detail.json` |
| saved/optimized 扩展精度 | **63/63 PASS**；显式 optimized，排除 9 个 legacy/拒绝路径看护 id | `$REGRESSION_ROOT/saved_opt63/case_detail.json`；`$REGRESSION_ROOT/saved_opt63/atk_run.log` |
| accuracy_dc | **12/12 × 50 PASS** | `$REGRESSION_ROOT/public/determinism_check.log` |
| CPU reference | **4/4** 自校验通过 | `$RESULT_ROOT/cpu_reference.log` |
| CPU recurrent | **6/6** 输出通过既定容差 | `$REGRESSION_ROOT/numeric/accuracy.log` |
| 大形状位比较 | **15/15** 张量逐位一致 | `$REGRESSION_ROOT/numeric/compare.log` |
| 数值矩阵 | 16 case × 15 张量，**240/240 bit-exact** | `$REGRESSION_ROOT/numeric/matrix_compare.log` |
| saved/recompute rstd layout | 8 case、**80/80** 张量逐位一致 | `$REGRESSION_ROOT/layout_norm_compare.json` |
| 独立默认布局 direct | 20 case、**140/140** 张量逐位一致 | `$REGRESSION_ROOT/direct_matrix_compare.json` |

确定性结论来自真实报告的复核，未以脚本退出前的“未找到报告（跳过）”或 `atk task success`
代替报告核验。合法 saved 集合排除 `{42,43,44,51,52,53,57,58,59}`，不将整包 72 例冒充 V2 正例。
旧 gatekeeper 默认值为 legacy，因此 saved 扩展验收必须显式设置 `KDA_GK_IMPL=optimized`，不能只读
63/63 数字推断路由。

PR967 最终白盒无新增确认问题，issue corpus 为 191 completed issues；静态报告的
209 errors / 2 warnings 为存量，不宣称静态检查全绿。证据为
`$REVIEW_ROOT/pr913_final_diff.json`、`issue_sync_check.log` 和 `issue_coverage_final.log`。

### 最终组合 kernel 性能

`atk_chunk_kda_bwd_perf.json` 保留 B1/H96/T8192/K=V=128/C64，BF16 token、FP32 raw gate、
safe gate、dense，saved/recompute 各一条。最终实测为 core/norm off 且 beta sigmoid=False；
两条路径均做六轮交替 A/B，每次 profile 取 tail-3 中位数，再取六轮中位数与配对差中位数。
证据为 `$RESULT_ROOT/final_perf/kernel/perf_{saved,recompute}.json`，不是独立 ATK `performance_device`。

| 路径 | Finalize 基线→候选 (µs) | 中位数变化 | Finalize 配对差 (µs) | bwd 基线→候选 (µs) | bwd 配对差 (µs) | Finalize MFU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| saved | 5354.0265→5085.2025 | -5.020969% | -271.9465 | 8031.492→7373.545 | -657.8565 | 13.9096673%→14.6449875% |
| recompute | 5350.994→5076.1455 | -5.136401% | -276.3540 | 9951.6945→9287.9475 | -678.0500 | 13.9175501%→14.6711175% |

V2 外层两个 Transpose 在 saved 为 **407.706→0 µs**、recompute 为 **407.5415→0 µs**。
它们属于 Finalize 之后的布局恢复，不是 Finalize 单 kernel 内部收益；bwd kernel sum 也不是完整 e2e。
“配对差”是逐轮 `candidate-baseline` 的中位数，与两侧中位数的比值变化分开统计。

独立 O1-only 数据不混入上述组合结果。其候选为 `fe6c9360` 加与最终版等价的五组 vector 修改，
仅 4 个 Finalize.o 和 4 个 json 变化，其余 1319 个安装文件不变；证据根为 `$O1_ROOT`，对应
`runs/pr_final_20261029/pr913/o1_interleaved/acceptance`。

| 路径 | Finalize 基线→候选 (µs) | 中位数变化 / 配对差 | bwd 基线→候选 (µs) | bwd 配对差 (µs) | Finalize MFU |
| --- | ---: | ---: | ---: | ---: | ---: |
| saved | 5354.677→5181.794 | -3.228635% / -171.3715 | 8032.8695→7860.708 | -175.0515 | 13.9079775%→14.3719969% |
| recompute | 5353.75→5184.3825 | -3.163530% / -171.4870 | 9967.9405→9818.8800 | -145.2205 | 13.9103857%→14.3648211% |

O1-only saved/recompute smoke 均 10/10 bit-exact，allocated/reserved 均 0/0 MiB；
性能证据为 `$O1_ROOT/perf/perf_{saved,recompute}.json`，显存证据为
`$O1_ROOT/memory_current/summary.json`。O2/F1 历史增量继续保留 `$INCREMENTAL_ROOT` 口径。

### core/norm off 与 norm on e2e

两种语义均为 3 轮紧配对，每轮预热 3 次、采集 20 个公开 forward+backward event；
core/norm off 与 norm on 都保持 beta sigmoid=False，不包含额外 beta 链或模型布局恢复。
event 的基线→候选为两侧中位数，配对差由逐轮 `candidate-baseline` 计算；MFU 只针对 Finalize。

| 语义 | 路径 | event 基线→候选 (ms) | 中位数变化 | 配对差中位 (ms) | Finalize MFU |
| --- | --- | ---: | ---: | ---: | ---: |
| core/norm off | saved | 11.4248438→10.7700992 | -5.730885% | -0.6547446 | 13.8972950%→14.6605773% |
| core/norm off | recompute | 13.1426930→12.4079571 | -5.590452% | -0.6562715 | 13.9007528%→14.6602974% |
| norm on | saved | 11.6627965→11.0080752 | -5.613759% | -0.6632800 | 13.5028989%→14.2244650% |
| norm on | recompute | 13.4452333→12.7647848 | -5.060890% | -0.6804485 | 13.4886480%→14.2219496% |

norm on 仅开启 Q/K L2Norm；对应 Finalize 为 saved **5515.314→5235.538 µs**、
recompute **5521.141→5236.464 µs**。证据分别为
`$RESULT_ROOT/final_e2e_memory/e2e_normoff/summary_{saved,recompute}.json` 和
`$RESULT_ROOT/final_normon_memory/e2e_normon/summary_{saved,recompute}.json`。

### 显存

生产形状下先构造并保留对应前向缓存，三次反向预热后清空缓存、重置峰值，再测一次公开反向。
dg 在两侧均为 `[1,96,8192,128]` FP32，输出布局没有变化；证据为
`$RESULT_ROOT/final_normon_memory/memory_summary.json`。

| 路径 | 基线 allocated / reserved 峰值 | 候选 allocated / reserved 峰值 | allocated / reserved 均下降 |
| --- | ---: | ---: | ---: |
| saved | 5997.473 / 6018.0 MiB | 5669.473 / 5690.0 MiB | **328 / 328 MiB** |
| recompute | 5669.476 / 5690.0 MiB | 5477.476 / 5498.0 MiB | **192 / 192 MiB** |

最终显存下降归因于 O3 删除 token-major 临时量与布局恢复；O1-only 为 0/0 MiB，
O2/F1 历史增量保持对应实现口径。

## 执行

以下命令从仓库根目录执行，输出到独立复现目录，不覆盖归档证据。`ENV_DIR` 为隔离安装环境，
`ATK_ROOT` 为 ATK 根目录，`RESULT_ROOT` 为上述最终结果目录；`CANN_SET_ENV` 可覆盖。

```bash
: "${ENV_DIR:?set ENV_DIR to the installed fla_npu environment}"
: "${ATK_ROOT:?set ATK_ROOT to the ATK root}"
: "${RESULT_ROOT:?set RESULT_ROOT to the pr913 result directory}"
: "${LEGACY_ROOT:?set LEGACY_ROOT to the legacy saved ATK operator directory}"
REPRO_ROOT="${REPRO_ROOT:-$RESULT_ROOT/reproduce}"
CANN_SET_ENV="${CANN_SET_ENV:-/usr/local/Ascend/cann-9.1.0/set_env.sh}"
source "$CANN_SET_ENV"
export PATH="$ATK_ROOT/bin:$PATH"
export PYTHONPATH="$ATK_ROOT:$ENV_DIR:${PYTHONPATH:-}"
export ASCEND_CUSTOM_OPP_PATH="$ENV_DIR/fla_npu/opp/vendors/fla_npu_transformer:$ENV_DIR/fla_npu/opp/vendors/fla_npu_transformer/op_api/lib"
export FLA_NPU_DISABLE_PTH=1
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
NPU_DEVICE_ID="${NPU_DEVICE_ID:-0}"

GEN_CASES_DTYPE_NUMBERS=36 bash tests/atk/run_test_cpu.sh -op=chunk_kda_bwd -scope=gen_cases -soc=ascend950
export ATK_OUTPUT_ROOT="$REPRO_ROOT/public_accuracy"
bash tests/atk/run_test_cpu.sh -op=chunk_kda_bwd -npu_device_id="$NPU_DEVICE_ID" -scope=accuracy -soc=ascend950
export ATK_OUTPUT_ROOT="$REPRO_ROOT/public_determinism"
bash tests/atk/run_test_cpu.sh -op=chunk_kda_bwd -npu_device_id="$NPU_DEVICE_ID" -scope=determinism -soc=ascend950
export KDA_GK_IMPL=optimized
ids=()
for ((i=0; i<72; i++)); do
  case "$i" in 42|43|44|51|52|53|57|58|59) ;; *) ids+=("$i") ;; esac
done
mkdir -p "$REPRO_ROOT/saved_opt63/accuracy"
python3 -m atk node --name npu_dut --backend npu --devices "$NPU_DEVICE_ID" \
  --output_path "$REPRO_ROOT/saved_opt63/accuracy" \
  node --name cpu_golden --backend cpu --output_path "$REPRO_ROOT/saved_opt63/accuracy" \
  task -c "$LEGACY_ROOT/atk_chunk_kda_bwd.json" --task accuracy --bm_device cpu \
  -p "$LEGACY_ROOT/executor_chunk_kda_bwd.py" -sp -mt 1 -to 1800 \
  -wl "[$(IFS=,; echo "${ids[*]}")]"
unset KDA_GK_IMPL
```

正式 accuracy 使用默认 GM-on；saved/optimized 扩展精度必须显式 `KDA_GK_IMPL=optimized`。
host tiling 从零构建和运行见 `tests/operators/chunk_kda_bwd_finalize/README.md`；已有构建目录可执行
`ctest --test-dir "$DFX_ROOT" --output-on-failure` 和
`"$DFX_ROOT/chunk_kda_bwd_finalize_tiling_test"`。运行生成的 `atk_output/`、`result/`、日志和
`__pycache__` 均不提交。

## 未闭环项

- 完整 Finalize 及 saved+recompute 整链 msSanitizer **未运行**；普通执行、bit-exact、确定性和
  显存结果不能替代 memcheck/racecheck。
- A2/A3 实机、整体 ST、当前 head 7 项 NPU CI **未运行**；交叉编译和 host 模拟不等于设备通过。
- 独立 ATK `performance_device` **未运行**；本文性能来自最终紧配对 msprof/e2e 证据。
- 最终硬件双发射**未证明**；没有 IR/反汇编，只声明源码为编译器提供调度机会。
- optional attr、内部 L0 重载、TilingData 与 raw tiling `sizeof+8` 仍需 ABI owner 复核。
