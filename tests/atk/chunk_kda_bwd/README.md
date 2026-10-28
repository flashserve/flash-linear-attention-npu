# ChunkKdaBwd public V2（Ascend950）

## 范围与版本

本包验收 `fla_npu.ops.ascendc.chunk_kda_bwd(..., implementation="optimized")` 的 public V2，
不测试或改写 `auto/legacy/optimized` 路由。代码基线为 upstream/main
`c9e2f1dbe6733d7d74c6bc4e1567e136f77da509`；候选增加 Finalize 五组 helper unroll-2、
A5/V2 原生 head-major db/dg、Stage7→9 FP32 中心因子复用，以及 A5 tiling ABI 修复。
被测环境为 `/home/vllm_ascend_m/envs/v2_delivery_c9e2_1027`，CANN 9.1.0、ATK 26.9.8、Ascend950DT。
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
`atk_chunk_kda_bwd.json` 与 `gen_chunk_kda_bwd.py` 的 36 条 spec/seed 完全一致。
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
`_mss.json` 对 12 个逻辑场景各保留 seed=0 一条。`$T` 为
`/home/vllm_ascend_m/kda_opt_workspace/runs/v2_opt_1027/c9e2/atk/tiling_selection`；
`$K` 为同级 `key_selection/driver.log`。

| key | 条件 | accuracy 逻辑 id | mss id | 当前候选实际选择记录 |
| --- | --- | --- | --- | --- |
| 1 | dense，无 rstd | 0,4,10,11 | 0,4,10,11 | `$K:148257` |
| 2 | packed，无 rstd | 1,5,9 | 1,5,9 | `$K:161682` |
| 3 | dense，有 rstd | 2,6,8 | 2,6,8 | saved `$T/p0_r0_bf16.log:146950`；recompute `$T/p0_r1_bf16.log:147146` |
| 4 | packed，有 rstd | 3,7 | 3,7 | saved `$T/p1_r0_bf16.log:145796`；recompute `$T/p1_r1_bf16.log:146508` |

这些是 public V2 组合内实际命中记录，不是内存检测通过结论，也不替代各子算子独立全 key 验收。
旧直接基准的小形状前向非有限值不属于本包的反向失败；本包使用上述三阶段输入准备。
ABI 修复在 A5 设置 raw tiling 为 `sizeof(TilingData)+8`，与仓内 `MergeFwdBwd` 的
CANN `GET_TILING_DATA opParaSize` 处理一致，arch22 路径不变。

## 正式结果与证据

`$O` 为 `/home/vllm_ascend_m/kda_opt_workspace/runs/v2_opt_1027/c9e2/atk`；
`$D` 为同级 `delivery`。除特别注明外，精度/确定性使用物理卡7，进程内设备0。

| 动作 | 当前候选结果 | 证据 |
| --- | --- | --- |
| public V2 accuracy | **36/36 PASS**，12 逻辑 case × 3 seeds | `$O/public_v2/accuracy/atk_output/atk_chunk_kda_bwd_2026-10-27-10-50-45-759135/report/atk_chunk_kda_bwd_reports_2026-10-27-10-50-46.xlsx` |
| saved/optimized 扩展精度 | **63/63 PASS**；排除由 legacy/拒绝路径看护的 9 个不适用 id | `$O/saved_opt63/case_detail.json`；`KDA_GK_IMPL=optimized` 见 `$O/saved_opt63/atk_run.log:13` |
| performance_device | 2/2 已执行；无预设性能目标，不标为“达标” | `$O/public_v2/perf/atk_output/atk_chunk_kda_bwd_perf_2026-10-27-10-52-38-127440/report/atk_chunk_kda_bwd_perf_reports_2026-10-27-10-52-38.xlsx` |
| accuracy_dc | **12/12 × 50 PASS** | `$O/public_v2/runtime_atk_output/atk_chunk_kda_bwd_mss_2026-10-27-10-53-37-624822/report/atk_chunk_kda_bwd_mss_reports_2026-10-27-10-53-38.xlsx` |
| 大形状位比较 | 15/15 张量逐位一致 | `$D/compare.log` |
| 数值矩阵 | 16 case × 15 张量，**240/240 bit-exact** | `$D/matrix_compare.log` |
| 独立默认布局 | 20 case、140 张量逐位一致 | `$D/direct_matrix_compare.json` |
| saved/recompute rstd | 8 case、80 张量逐位一致 | `$D/layout_norm_matrix_compare.json` |
| CPU recurrent | 六项输出通过既定容差 | `$D/accuracy.log` |
| CPU 标杆自校验 | 单步 norm/packed/无 bias 与确定性输入构造通过 | `$O/cpu_reference.log` |
| 安装对象 | wheel 的 1318 个 fla_npu 文件与安装环境一致；非 Finalize 的 183 个对象名称/hash 全同；Finalize 均为 4 个对象；路由 Python 不变 | `$D/verify_install_final.json` |

确定性输出由统一脚本写在工作目录，已原样移至 `$O/public_v2/runtime_atk_output`；
随后用未修改的 `check_atk_result.py --type determinism` 指定该真实输出根目录复核为 12/12 Pass。
没有以脚本退出前的“未找到报告（跳过）”或 `atk task success` 代替报告核验。
`generator_compare.json` 证明正式 JSON 与生成器的 36 条 spec/seed 完全一致。
合法 saved 集合排除 `{42,43,44,51,52,53,57,58,59}`，不将整包 72 例冒充 V2 正例。
旧 gatekeeper 默认值为 legacy，因此 saved 扩展验收必须显式设置 `KDA_GK_IMPL=optimized`，不能只读 63/63 数字推断路由。

### 性能模型 case

`atk_chunk_kda_bwd_perf.json` 保留 B1/H96/T8192/K=V=128/C64，BF16 token、FP32 raw gate、
safe gate、dense、norm off，saved/recompute 各一条。ATK 的前向缓存构造不计入性能。

| perf id | 路径 | ATK device_perf | NPU std deviation |
| --- | --- | ---: | ---: |
| 0 model_h96_saved | saved | 7335.267050 µs | 9.866881 µs |
| 1 model_h96_recompute | recompute | 9270.823600 µs | 10.609632 µs |

同输入、同物理卡6的 `msprof op_summary.Task Duration(us)` 另作同口径主线对照。
bwd kernel sum 包含 Prepare/Dhu/Finalize/附属转换，recompute 另含 Recompute/FwdH；
saved 六轮正反顺序交替，recompute 三轮。下表为各轮尾三次中位数之后再取中位数。

| 路径 | kernel | 主线中位数 | 候选中位数 | 中位数差值 | 比值变化 |
| --- | --- | ---: | ---: | ---: | ---: |
| saved | Finalize | 5375.450 µs | 5113.037 µs | -262.413 µs | -4.88% |
| saved | bwd kernel sum | 8105.302 µs | 7445.102 µs | -660.200 µs | -8.15% |
| recompute | Finalize | 5367.947 µs | 5114.570 µs | -253.377 µs | -4.72% |
| recompute | bwd kernel sum | 9909.605 µs | 9242.302 µs | -667.303 µs | -6.73% |

逐轮配对差值的中位数分别为 saved Finalize -261.143µs、bwd -675.323µs；
recompute Finalize -255.717µs、bwd -665.756µs。它与“先取两侧中位数再相减”是不同统计量，不混用。
原始结果在 `$D/perf_saved.json`、`$D/perf_recompute.json`；逐项 profile 汇总在 `$D/timeline_compare.json`。

### 显存

生产形状下先构造并保留对应前向缓存，三次反向预热后清空缓存、重置峰值，再测一次公开反向。
`/home/vllm_ascend_m/kda_opt_workspace/runs/v2_opt_1027/c9e2/memory_routes.py`
与 `$D` 下四份 `memory_{saved,recompute}_{main,candidate}.json` 为原始证据。
dg 在两侧均为 `[1,96,8192,128]` FP32，输出布局没有变化。

| 路径 | 主线 allocated / reserved 峰值 | 候选 allocated / reserved 峰值 | 两者均下降 |
| --- | ---: | ---: | ---: |
| saved | 5997.473 / 6018.0 MiB | 5669.473 / 5690.0 MiB | **328 MiB** |
| recompute | 5669.476 / 5690.0 MiB | 5477.476 / 5498.0 MiB | **192 MiB** |

## 执行

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
V=/home/vllm_ascend_m/kda_opt_workspace/atk_kb1_gatekeeper/vendor/atk2698
E=/home/vllm_ascend_m/envs/v2_delivery_c9e2_1027
export PATH="$V/bin:$PATH"
export PYTHONPATH="$V:$E:${PYTHONPATH:-}"
export ASCEND_CUSTOM_OPP_PATH="$E/fla_npu/opp/vendors/fla_npu_transformer:$E/fla_npu/opp/vendors/fla_npu_transformer/op_api/lib"
export FLA_NPU_DISABLE_PTH=1 ASCEND_RT_VISIBLE_DEVICES=7
export ATK_OUTPUT_ROOT=/home/vllm_ascend_m/kda_opt_workspace/runs/v2_opt_1027/c9e2/atk/public_v2

GEN_CASES_DTYPE_NUMBERS=36 bash tests/atk/run_test_cpu.sh -op=chunk_kda_bwd -scope=gen_cases -soc=ascend950
# 按阶段顺序执行，避免本机 ATK 端口复用；mss 结果必须另看设备日志，不能只看退出码。
bash tests/atk/run_test_cpu.sh -op=chunk_kda_bwd -npu_device_id=0 -scope=accuracy -soc=ascend950
bash tests/atk/run_test_cpu.sh -op=chunk_kda_bwd -npu_device_id=0 -scope=performance -soc=ascend950
bash tests/atk/run_test_cpu.sh -op=chunk_kda_bwd -npu_device_id=0 -scope=determinism -soc=ascend950
```

正式 accuracy 使用默认 GM-on。运行生成的 `atk_output/result/atk.log/__pycache__` 均已移至 `$O`，不提交。

## 内存与竞争检测：未完成

**不能把当前候选标为 memcheck/racecheck 通过或正式全量验收完成。**

- 当前候选的 saved/norm 代表用例（逻辑 id 2，key 3）在 CANN 9.1.0 / msSanitizer 26.1.0 下，
  即使设置 `PYTORCH_NO_NPU_MEMORY_CACHING=1` 并仅筛选 `ChunkKdaBwdFinalize`，仍出现 `507015`。
  运行时 plog 为 device error 263、`mix_aic+0x2a738`，随后 ATK host `double free`。
  见 `$O/mss_nocache_case2/` 及 `$O/plogs/plog-3759181_20261027104322613.log`。
- 已从当前候选源码构建 `--debug --sanitizer` 包，JSON 确认为 `ccec_g,sanitizer`，
  不是复用旧 571d 包。静态代表用例仍有同一偏移和 `507015`，没有可据以修复 kernel 的源码行号。
  见 `$O/static_c9e2_case2/` 及 `$O/plogs/plog-3795650_20261027105634913.log`。
  日志有原始 LOAD/STORE 记录且没有对应非法访问报告，但设备异常使其不能成为通过证据。
- 用同一 c9e2 主线、只加入 ABI 修复的静态包复测相同代表用例，仍在 `mix_aic+0x2a738`
  出现 `507015` 和 host 异常（`$O/static_base_abi_case2/`、
  `$O/plogs/plog-3832197_20261027111804067.log`）。
  因而该现象不是只有性能候选才出现，但主线和候选均未因此获得内存验收通过结论。
- racecheck 代表用例虽写 `No error detected`，同一运行仍有 `507015` 和 host 异常，故无效；
  见 `$O/race_nocache_case2/`。CANN 26.2 的 DBI fallback/无真实记录也不作为绿色证据。
- tiling ABI 修复只处理 host/device tiling 尺寸契约，不能替代上述整链检测。
  目前没有有效的完整 12 例内存报告，也没有完整的双路径 racecheck 通过证据。

因此当前交付结论是：**数值、ATK 精度/确定性及双路径性能已实测；内存/竞争验收仍阻塞。**
普通执行与 bit-exact 结果不能证明不存在内存错误，插桩异常也尚不能等同于已定位的原 kernel 越界。
