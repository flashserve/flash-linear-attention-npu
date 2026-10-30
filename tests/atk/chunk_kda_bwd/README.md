# ChunkKdaBwd public V2（Ascend950）

> **当前精度未全绿：42 条全部执行，26/42 PASS、16/42 FAIL；失败全部为 FP32 dg 的 `output_4.pt`。** 三版本输出位一致只证明 O1/O2 未引入新差异，不能替代 CPU 精度验收，也不能写成 42/42 通过。

## 范围、版本与运行环境

本包执行 `fla_npu.ops.ascendc.chunk_kda_bwd(..., implementation="optimized")` 的 public V2，
覆盖 saved/recompute、dense/packed、norm off/on、raw_g/beta 的 BF16/FP32 组合。
范围限定为 Ascend950、K=V=128、chunk=64、safe gate、`use_exp2=True`、BF16 token；
不覆盖 unsafe、FP16、initial state/dht，不把本包结果外推为其他路由的验收结论。

性能机制仅为 Finalize 的 O1 五组 VF 同一成对循环内 A/B 交错与独立单尾，以及
O2 Stage7→9 FP32 bandExp 复用。A5 raw tiling、DFX、host/ATK 资产和文档修改不计性能收益。

结果根记为 `$R=/home/vllm_ascend_m/kda_opt_workspace/runs/pr913_pr993_20261030`。
被测 wheel 的生产提交为 `b39f223e`；逐输出比较合同对应测试提交 `4f190ea2`。

| 项目 | 当前实测或环境 | 证据 |
| --- | --- | --- |
| A5 wheel | 构建、安装成功；SHA256 `2a1895900aca858b19405f97bb48c4bac1228b1af3f36aafb3181b474a55327f` | `$R/build_locked_wheel.log`、`install/final_wheel.sha256`、`install/final_pip.log` |
| 安装完整性 | **1326/1326** 文件匹配 | `$R/install/final_install_verify.log` |
| 软件环境 | CANN 9.1.0、Python 3.11.10、torch 2.10.0+cpu、torch_npu 2.10.0.post6、ATK 26.9.8 | `$R/build_locked_preflight.log`、`install/final_runtime.log`、`atk_serial_accuracy.log` |
| 设备 | Ascend950DT_9584；四组配对性能使用物理卡 7 | `$R/install/final_runtime.log`、`perf/*.json` 及对应 msprof 记录 |
| base 独立环境 | `/home/vllm_ascend_m/envs/kda_pr993_base_20261030` | `$R/install/base_runtime.log` |
| O1 独立环境 | `/home/vllm_ascend_m/envs/kda_pr993_o1_20261030` | `$R/install/o1_runtime.log` |
| final 独立环境 | `/home/vllm_ascend_m/envs/kda_pr993_final_20261030`；运行时 `fla_npu` 与 opapi 均来自该目录 | `$R/install/final_runtime.log` |

## PR993 布局、输入准备与 CPU 标杆

公开接口只有一个 `layout`，合法值为 `BSND/BNSD/TND/NTD`；当前 ATK 执行入口使用
dense `BSND` 和 packed `TND`。q/k/v 及 dq/dk/dv 随 layout，有成对 rstd 时 q/k 为
前向保存的 head-major q_hat/k_hat。公开 do/raw_g/beta 与 dg/db 固定 token-major。
dq/dk/dv 为 BF16，dg 跟随 raw_g 的真实 dtype，db 跟随 beta，A_log/dt_bias 及其梯度固定 FP32。

ACLNN 将 do 转置一次供 Prepare/Dhu 共用，普通 beta 转置一次供 Recompute/Finalize 共用；
Finalize 内部 beta 仍为 head-major。独立 Finalize 的 `output_layout` 控制全部五个逐 token
梯度，内部可选 `model_layout` 默认空字符串表示 standalone，不用 shape 猜测 `T == H` 的布局。

DUT 初始化调用公开 `npu_chunk_kda_fwd_prepare`、`npu_chunk_fwd_h` 和
`npu_chunk_kda_fwd_finalize` 构造 Aqk/Akk 与缓存。saved 传入完整
`gk/w/qg/kg/v_new/h` 六项 cache；recompute 六项均为 None，只保留 Aqk/Akk。
norm-on 使用 Prepare 导出的 q_hat/k_hat/q_rstd/k_rstd。前向准备与有限值检查不计入反向性能。

真实 ATK 输入保持 head-major，与 CPU golden 一致；executor 不重建另一套随机输入。
NPU 调用前将 q/k/v/raw_g/beta/d_o 转为公开 BSND/TND，输出比较前再把五个逐 token
梯度转回 CPU golden 的 head-major 排列。CPU golden 是独立 FP64 recurrent + autograd，
最终返回 FP32；norm-on 保留 L2Norm、BF16 q_hat/k_hat 量化、归一化反向和 cast。
输入值域固定为 q/k/v std=0.05、raw_g std=1.25、beta∈[0.1,0.9]、A_log std=0.12、
dt_bias std=1.65 且均值 -3、dO std=51.2。

`cpu_reference_r2.log` 记录四个 one-step 场景、确定性输入构造、BF16 raw_g 量化与
provided-input 消费、dense/packed layout roundtrip 均为 OK。

## 逐输出精度合同与当前失败

ATK 可见输出为 `(dq,dk,dv,db,dg,dA,dbias)`，无 bias 时过滤 dbias。
**比较 override 只选择逐输出精度判据，不是公开输出 dtype 声明。** executor 独立断言
公开存储 dtype：dq/dk/dv 为 BF16，db 跟随 beta，dg 跟随 raw_g，dA/dbias 为 FP32。

| 可见输出 | 存储 dtype | mixed-tolerance 比较判据 |
| --- | --- | --- |
| dq/dk/dv | BF16 | 已验证的 BF16 校准 |
| db | 跟随 beta，BF16/FP32 | 已验证的 BF16 校准 |
| dA/dbias | FP32 | 已验证的 BF16 校准 |
| dg，raw_g=BF16 | BF16 | 按 BF16 |
| dg，raw_g=FP32 | FP32 | **严格按 FP32** |

当前串行 accuracy 证据为 `$R/atk_serial/accuracy_summary.json`、其对应 XLSX 和
`$R/atk_serial_accuracy.log`：42 条执行成功，**26 通过、16 失败**，失败输出全部为
`output_4.pt`，即 FP32 dg。失败 case id 为：

```text
6, 7, 8, 9, 10, 11, 13, 18, 19, 20, 21, 22, 23, 24, 25, 26
```

FP32 dg 的精度差距可在 PR993 base 复现；base/O1/final 三版本位一致说明 O1/O2 未引入
新差异，但不能掩盖上述 16 条失败。既有错误注入检查表明，FP32 dg 套用 BF16 判据会
漏掉应报告的误差，因此不得用该方式规避失败。当前没有经过误差预算和负例共同验证的
替代 FP32 标准，不能把问题称作“已校准”或“只是容差问题”。

补充 571d→c9e2→PR993 base→final 的同输入、同缓存、同 CPU golden A/B：case 0/6/13 均使用正式
ATK `input.bin` 绑定输入，各版本的 dg 及全部七项输出逐位一致。case 6/13 的 FP32 dg
失败在 571d 和 c9e2 已存在，**不是 PR968、PR993 或 O1/O2 引入的差异**；case 0 通过。
证据为 `$R/pre993_probe/r3/results/exact_summary.json` 和
`$R/pre993_probe/r4/results/exact_summary.json`。该对照仅覆盖三个代表用例，不能确定更早的
最初引入点，也不能外推全部 16 个失败或改变 accuracy 26/42 的结果。

## 资产矩阵与 TilingKey 映射

固定种子为 `0 / 20260630 / 7`。当前有 **14 个逻辑 profile × 3 seeds = 42 条 accuracy
资产、14 条 mss 资产、2 条 perf 资产**。真实输入、shape、dtype 和 seed 与 generator 一致。
下表的 dtype 顺序为 raw_g/beta；完整输入设计见 `test_case_design_report.md`。

| 逻辑 id / profile | accuracy id | mss id | 路径 | layout / norm | B/H/T 或 packed cu | raw_g/beta | Key |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 saved_dense | 0–2 | 0 | saved | BSND / off | B1/H8/T64 | FP32/BF16 | 1 |
| 1 saved_packed | 3–5 | 1 | saved | TND / off | H8/cu=[0,64,128] | FP32/BF16 | 2 |
| 2 saved_dense_norm_beta32 | 6–8 | 2 | saved | BSND / on | B1/H8/T100 | FP32/FP32 | 3 |
| 3 saved_packed_norm | 9–11 | 3 | saved | TND / on | H8/cu=[0,97,130] | FP32/BF16 | 4 |
| 4 recompute_dense | 12–14 | 4 | recompute | BSND / off | B1/H8/T100 | FP32/BF16 | 1 |
| 5 recompute_packed | 15–17 | 5 | recompute | TND / off | H8/cu=[0,64,128] | FP32/BF16 | 2 |
| 6 recompute_dense_norm_beta32 | 18–20 | 6 | recompute | BSND / on | B1/H8/T64 | FP32/FP32 | 3 |
| 7 recompute_packed_norm | 21–23 | 7 | recompute | TND / on | H8/cu=[0,97,130] | FP32/BF16 | 4 |
| 8 saved_dense_odd_norm | 24–26 | 8 | saved | BSND / on | B1/H3/T100 | FP32/BF16 | 3 |
| 9 saved_packed_odd | 27–29 | 9 | saved | TND / off | H3/cu=[0,97,130] | FP32/BF16 | 2 |
| 10 saved_dense_b2_nobias_tail | 30–32 | 10 | saved | BSND / off | B2/H8/T100 | FP32/BF16 | 1 |
| 11 recompute_dense_b2_nobias_tail | 33–35 | 11 | recompute | BSND / off | B2/H8/T100 | FP32/BF16 | 1 |
| 12 saved_dense_t_eq_h_raw_g_bf16 | 36–38 | 12 | saved | BSND / off | B1/H2/T2 | BF16/BF16 | 1 |
| 13 recompute_packed_t_eq_h_norm_raw_g_bf16 | 39–41 | 13 | recompute | TND / on | H8/cu=[0,8] | BF16/BF16 | 4 |

profile 12 覆盖 H=T=2、norm off 和 BF16 raw_g；profile 13 覆盖 H=T=8、
`cu_seqlens=[0,8]`、norm on 和 BF16 raw_g，H=8 满足 recompute 的 8 倍数限制。
逻辑 10/11 无 dt_bias，其余有 dt_bias。Finalize key 为
`(variable ? 2 : 1) + (hasRstd ? 2 : 0)`，内部 `model_layout` 不增加 key。

| Key | 条件 | 逻辑 profile id | mss id |
| --- | --- | --- | --- |
| 1 | dense，无 rstd | 0,4,10,11,12 | 0,4,10,11,12 |
| 2 | packed，无 rstd | 1,5,9 | 1,5,9 |
| 3 | dense，有 rstd | 2,6,8 | 2,6,8 |
| 4 | packed，有 rstd | 3,7,13 | 3,7,13 |

这些是资产到预期 key 的设计映射，不是全部实际选 key 日志，也不是 msSanitizer 通过结论。

## 其他验证结果

| 项目 | 当前结果 | 证据 |
| --- | --- | --- |
| host tiling 与 DFX | **15/15 PASS**，CTest **1/1 PASS**；A5 TilingData=184、raw size=192，非法 `model_layout` 报告参数名/实际值/合法集合 | `$R/host_test_dfx.log`、`host_ctest_dfx.log` |
| A2/A3 构建 | 两个构建日志均成功；**不代表 A2/A3 实机通过** | `$R/a2_finalize_build.log`、`a3_finalize_build.log` |
| ATK determinism | **14 个 mss case × 50 次循环，14/14 PASS**；真实 XLSX 已复核 | `$R/atk_serial/determinism_verified.json`、`determinism.log` 及对应 XLSX |
| ATK `performance_device` | **2/2 执行完成**；saved `device_perf=8588.387750 µs`，recompute `10575.810750 µs`；未给出性能达标判定 | `$R/atk_serial/performance.log` 及对应 XLSX |
| 三版本输出等价 | **84/84 整数位模式一致**，12 组 × 七种梯度，mismatch=0，最终 `nonfinite_tensors=[]` | `$R/dumps/summary.json` |
| CPU 参考自检 | one-step、输入确定性、BF16 raw_g、layout roundtrip 均为 OK | `$R/cpu_reference_r2.log` |
| issue 覆盖 | **188 completed issues / 21 rules** 检查通过 | `$R/review/issue_coverage.log` |
| Finalize 静态扫描 | **42 errors / 0 warnings**，均为 legacy/generic guard 存量；不宣称静态全绿 | `$R/review/finalize.json` |

上述 84 个张量比较项来自 12 组组合：saved/recompute × core/customer ×
`base→O1`、`O1→final`、`base→final`，每组七项梯度。它证明这些输入下优化前后的
整数位模式相同，包括 NaN payload 的比较方式；本次最终非有限值列表为空。
**位等价不替代 CPU 精度。**

## 配对性能实测

测量对象为 `B1/H96/T8192/K=V128/C64`、dense BSND、BF16 token、FP32 raw_g、safe gate。
core 为 norm off；customer 为 norm on 并包含外部 beta-sigmoid backward，A5 算子内仍保持
`use_beta_sigmoid_in_kernel=False`。证据为 `$R/perf/{saved_core,saved_customer,recompute_core,recompute_customer}.json`。

三版本为 PR993 base、仅 O1、O1+O2 final。O1 比较 `base→O1`，O2 比较 `O1→final`，
combined 比较 `base→final`。每组共 **6 轮三版本交替**，奇数轮 base→O1→final，
偶数轮反向；每版本 warmup 3 次、采集 20 次，profiler 序列取 **tail-3 中位数**，
再取六轮中位数。`Δk=median_kernel_delta_us`，`Δbwd=median_bwd_delta_us`，
均为逐轮 `candidate-baseline` 后取中位数，不是两侧中位数直接相减；负值表示候选更快。
bwd kernel sum 与 public forward+backward event 是不同统计对象。

### core / norm off

| 对比 | 路径 | Finalize base→candidate (µs) | Δk (µs) | Δk (%) | bwd base→candidate (µs) | Δbwd (µs) | Finalize MFU base→candidate | e2e base→candidate (ms) | Δe2e (ms) | Δe2e (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| O1 | saved | 5361.0105→5186.5470 | -162.6215 | -3.254303 | 7804.1675→7622.4585 | -172.1625 | 13.891547%→14.358826% | 11.150669→10.971033 | -0.183818 | -1.610984 |
| O1 | recompute | 5353.0920→5184.3965 | -169.1525 | -3.151366 | 9806.1915→9626.1170 | -188.1690 | 13.912096%→14.364782% | 12.908335→12.727046 | -0.180792 | -1.404435 |
| O2 | saved | 5186.5470→5123.2630 | -68.0435 | -1.220157 | 7622.4585→7571.7045 | -55.3300 | 14.358826%→14.536191% | 10.971033→10.904788 | -0.064149 | -0.603818 |
| O2 | recompute | 5184.3965→5118.2100 | -70.8865 | -1.276648 | 9626.1170→9556.7260 | -63.9160 | 14.364782%→14.550542% | 12.727046→12.650078 | -0.076849 | -0.604757 |
| combined | saved | 5361.0105→5123.2630 | -237.4295 | -4.434752 | 7804.1675→7571.7045 | -229.5230 | 13.891547%→14.536191% | 11.150669→10.904788 | -0.243831 | -2.205075 |
| combined | recompute | 5353.0920→5118.2100 | -239.8670 | -4.387782 | 9806.1915→9556.7260 | -249.4655 | 13.912096%→14.550542% | 12.908335→12.650078 | -0.257174 | -2.000699 |

### norm on / customer

| 对比 | 路径 | Finalize base→candidate (µs) | Δk (µs) | Δk (%) | bwd base→candidate (µs) | Δbwd (µs) | Finalize MFU base→candidate | e2e base→candidate (ms) | Δe2e (ms) | Δe2e (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| O1 | saved | 5507.8485→5335.6605 | -171.8130 | -3.126230 | 7878.5190→7707.5865 | -175.9225 | 13.521201%→13.957546% | 11.446321→11.268313 | -0.178123 | -1.555151 |
| O1 | recompute | 5506.5430→5334.9805 | -170.7835 | -3.115612 | 10006.3205→9837.0760 | -170.7835 | 13.524407%→13.959325% | 13.504430→13.333792 | -0.162993 | -1.263575 |
| O2 | saved | 5335.6605→5276.2135 | -57.7015 | -1.114145 | 7707.5865→7644.5400 | -63.2645 | 13.957546%→14.114805% | 11.268313→11.219630 | -0.042398 | -0.432036 |
| O2 | recompute | 5334.9805→5272.4955 | -60.2260 | -1.171232 | 9837.0760→9752.6105 | -80.7615 | 13.959325%→14.124759% | 13.333792→13.273439 | -0.059616 | -0.452630 |
| combined | saved | 5507.8485→5276.2135 | -229.1445 | -4.205544 | 7878.5190→7644.5400 | -233.7425 | 13.521201%→14.114805% | 11.446321→11.219630 | -0.231646 | -1.980468 |
| combined | recompute | 5506.5430→5272.4955 | -234.2975 | -4.250353 | 10006.3205→9752.6105 | -251.9315 | 13.524407%→14.124759% | 13.504430→13.273439 | -0.228962 | -1.710486 |

MFU 仅针对 Finalize 单 kernel wall 时间：FLOPs=`322122547200`，
BF16 peak=`432.5376 TFLOP/s`。即 `FLOPs/(Finalize_us×1e-6×peak×1e12)×100%`，
不是 Prepare/Dhu/Recompute、bwd kernel sum 或整条 forward+backward 链路的 MFU。
ATK `performance_device` 默认 MFU=0 未用于上述表格；执行完成也不等于性能达标。

## 显存实测

同形状、同缓存语义下保留前向缓存，三次反向预热后执行 `empty_cache`、重置峰值，
再测一次公开反向。证据为 `$R/memory/summary.json`。allocated/reserved 单位均为 MiB。

| 语义 | 路径 | base allocated / reserved | O1 allocated / reserved | final allocated / reserved | O1 Δallocated | O2 Δallocated | combined Δallocated / Δreserved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| core / norm off | saved | 5866.0947 / 5882.0 | 5866.0947 / 5882.0 | 5866.0947 / 5882.0 | 0.0 | 0.0 | 0.0 / 0.0 |
| core / norm off | recompute | 5674.0947 / 5690.0 | 5674.0947 / 5690.0 | 5674.0947 / 5690.0 | 0.0 | 0.0 | 0.0 / 0.0 |
| norm on / customer | saved | 6259.0947 / 6266.0 | 6259.0947 / 6266.0 | 6259.0947 / 6266.0 | 0.0 | 0.0 | 0.0 / 0.0 |
| norm on / customer | recompute | 6067.0947 / 6074.0 | 6067.0947 / 6074.0 | 6067.0947 / 6074.0 | 0.0 | 0.0 | 0.0 / 0.0 |

O1、O2 及合计的 GM allocated/reserved 差均为 **0 MiB**；UB 内复用不能写成 GM 显存下降。

## 复现入口

以下命令使用结果根中的现有脚本和隔离环境，只在独立输出目录生成证据。
执行完成后必须读取真实报告；脚本结束不代表 accuracy 或性能达标。

```bash
R=/home/vllm_ascend_m/kda_opt_workspace/runs/pr913_pr993_20261030
BASE_ENV=/home/vllm_ascend_m/envs/kda_pr993_base_20261030
O1_ENV=/home/vllm_ascend_m/envs/kda_pr993_o1_20261030
FINAL_ENV=/home/vllm_ascend_m/envs/kda_pr993_final_20261030
export BASE_ENV O1_ENV FINAL_ENV

# 42 条 accuracy、14×50 determinism、2 条 performance_device
FINAL_ENV="$FINAL_ENV" CARD=<card-id> bash "$R/run_atk.sh"

# 三版本、两种路径、两种语义的 dump 与整数位模式比较
CARD=<card-id> bash "$R/run_dumps.sh"

# 四组六轮三版本交替性能
for ROUTE in saved recompute; do
  for SEMANTIC in core customer; do
    ROUTE="$ROUTE" SEMANTIC="$SEMANTIC" ROUNDS=6 CARD=<card-id> \
      bash "$R/run_kernel_ab.sh"
  done
done

# 四种 route/semantic 的 allocated/reserved 峰值
CARD=<card-id> bash "$R/run_memory.sh"
```

host CMake 复现方式如下，`UT_COMMON` 必须指向已有的 host UT common 目录：

```bash
CANN_ROOT=/usr/local/Ascend/cann-9.1.0
UT_COMMON=/absolute/path/to/existing/ut/common
BUILD_DIR="$R/host_repro"

cmake -S tests/operators/chunk_kda_bwd_finalize -B "$BUILD_DIR" \
  -DASCEND_CANN_PACKAGE_PATH="$CANN_ROOT" \
  -DUT_COMMON_DIR="$UT_COMMON"
cmake --build "$BUILD_DIR" -j4
export LD_LIBRARY_PATH="$CANN_ROOT/$(uname -m)-linux/lib64:$CANN_ROOT/lib64:${LD_LIBRARY_PATH:-}"
ctest --test-dir "$BUILD_DIR" --output-on-failure
"$BUILD_DIR/chunk_kda_bwd_finalize_tiling_test"
```

## 未闭环项

- **16 条 FP32 dg accuracy 失败未闭环**；没有经验证的替代 FP32 标准，不能用 BF16 判据漏检。
- **Finalize 及 saved/recompute 整链 msSanitizer 未通过、未闭环**。14 条 mss 资产、普通执行、
  位等价和 determinism 均不能替代 memcheck/racecheck。
- A2/A3 实机、整体 ST、当前 head 的 7 项 NPU CI **未执行**；交叉编译和 host 模拟不等于设备通过。
- `performance_device` 虽已 2/2 执行完成，但没有据此形成性能达标结论；配对性能也仅限本文
  形状、layout、dtype、route 和 semantic。
- 硬件双发射**未证明**；没有 IR、反汇编或硬件计数证据时，只描述源码级交错与实测耗时。
- PR993 optional attr、内部 L0 `modelLayout`、TilingData=184 与 raw tiling `sizeof+8=192`
  的接口/ABI owner 复核仍需完成。
