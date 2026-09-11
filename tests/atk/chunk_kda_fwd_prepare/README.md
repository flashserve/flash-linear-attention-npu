# ChunkKdaFwdPrepare ATK 工程

本目录通过稳定入口 `fla_npu.ops.ascendc.chunk_kda_fwd_prepare` 验证 Prepare
算子，不加载 `torch_npu` dispatcher 或 `torch.ops.npu` legacy 入口。接口固定保留
以下 13 个公开输出槽位：

```text
gk, aqk, akk, w, u, qg, kg, qg_scaled,
q_hat, k_hat, q_rstd, k_rstd, beta_eff
```

`output_mode` 只控制对应槽位是否发生真实搬出，未搬出的槽位返回 `None`：

| 模式 | 实际输出 |
| --- | --- |
| `none` | `gk/aqk/w/u/kg/qg_scaled` |
| `recompute` | `none` 模式输出，加 `akk/q_hat/k_hat/q_rstd/k_rstd/beta_eff` |
| `save` | 全部 13 个输出，额外保存 `qg` |

## 标杆语义

`executor_chunk_kda_fwd_prepare.py` 独立实现 CPU FP64 标杆，覆盖 Q/K L2Norm、
chunk 内 gate cumsum、S=4 的 16 行局部参考缩放、因果 `Aqk/Lkk`、32x32
分块三角逆、`W/U` 和 post-WU 输出。标杆显式保留 `qg_scaled` 与
`K_beta_g` 对已舍入 BF16 中间值的二次消费，避免把两次 cast 合并为一次。

输入由固定 seed 的 FP32 随机数生成，先量化到接口声明的 BF16/FP32，再分别送入
CPU 和 NPU 节点。ATK 使用原生 `mixed_tolerance_bm` 比较当前模式下真实返回的
输出，并检查全部 13 个槽位的 `None`、shape 和 dtype 合同。性能和内存检查不执行
输出 D2H；确定性由 ATK 拉回输出并做 50 轮逐位比较。

## 用例矩阵

| 文件 | 用例数 | 覆盖内容 |
| --- | ---: | --- |
| `atk_chunk_kda_fwd_prepare.json` | 200 | 56 个逻辑场景，每场景 3 或 4 个互异固定 seed；覆盖四种 layout、dense/varlen、tail、GVA、三档输出策略及全部功能分支；精度用例实际覆盖 17/432 个不同 TilingKey |
| `atk_chunk_kda_fwd_prepare_perf.json` | 10 | 用户给定的 10 个模型 shape；统一使用连续 BNSD 输入和纯推理输出模式 |
| `atk_chunk_kda_fwd_prepare_mss.json` | 432 | 每个可达 TilingKey 恰好一条，用于确定性和四类 sanitizer 检查 |

### 精度逻辑场景到 case ID

每个逻辑场景对应连续的固定 seed case，完整映射如下：

| 逻辑场景 | case ID | 逻辑场景 | case ID |
| --- | ---: | --- | ---: |
| `dense_bnsd_min` | 0-3 | `dense_bsnd_tail15` | 4-7 |
| `dense_ntd_tail16` | 8-11 | `dense_tnd_tail17` | 12-15 |
| `dense_bnsd_tail31` | 16-19 | `dense_bnsd_tail32` | 20-23 |
| `dense_bnsd_tail33` | 24-27 | `dense_bnsd_tail47` | 28-31 |
| `dense_bnsd_tail48` | 32-35 | `dense_bnsd_tail49` | 36-39 |
| `dense_bnsd_full_chunk` | 40-43 | `dense_two_chunks_tail1` | 44-47 |
| `dense_bsnd_batch2` | 48-51 | `dense_gva_ratio5_cross_wave` | 52-55 |
| `dense_gva_ratio8` | 56-59 | `dense_head_partition_33` | 60-63 |
| `dense_hv160_no_artificial_limit` | 64-67 | `varlen_tnd_auto_indices` | 68-71 |
| `varlen_ntd_explicit_indices` | 72-75 | `varlen_bnsd_rank4` | 76-79 |
| `varlen_more_than_1024_sequences` | 80-83 | `gate_softplus_without_bias` | 84-87 |
| `gate_softplus_with_bias` | 88-91 | `gate_safe_exp2` | 92-95 |
| `beta_sigmoid` | 96-99 | `beta_two_sigmoid` | 100-103 |
| `l2norm_small_epsilon` | 104-107 | `dense_bnsd_tail63` | 108-111 |
| `dense_bsnd_full_chunk` | 112-115 | `dense_ntd_full_chunk` | 116-119 |
| `dense_tnd_full_chunk` | 120-123 | `dense_bsnd_two_chunks_tail1` | 124-127 |
| `dense_ntd_two_chunks_tail1` | 128-130 | `dense_tnd_two_chunks_tail1` | 131-133 |
| `dense_bnsd_exact_two_chunks` | 134-136 | `dense_bsnd_three_chunks_tail1` | 137-139 |
| `dense_ntd_three_chunks_tail1` | 140-142 | `dense_tnd_three_chunks_tail1` | 143-145 |
| `dense_batch4_small` | 146-148 | `dense_gva_ratio2` | 149-151 |
| `dense_gva_ratio3` | 152-154 | `dense_gva_ratio4` | 155-157 |
| `dense_gva_ratio6` | 158-160 | `dense_head_partition_5` | 161-163 |
| `dense_head_partition_6` | 164-166 | `dense_head_partition_7` | 167-169 |
| `varlen_bnsd_auto_with_empty` | 170-172 | `varlen_bsnd_explicit_with_empty` | 173-175 |
| `varlen_ntd_auto_mixed` | 176-178 | `varlen_tnd_explicit_mixed` | 179-181 |
| `varlen_exact_chunk_boundaries` | 182-184 | `gate_softplus_exp2_with_bias` | 185-187 |
| `gate_safe_exp_without_bias` | 188-190 | `beta_sigmoid_bf16_output_none` | 191-193 |
| `beta_two_sigmoid_fp32_output_recompute` | 194-196 | `l2norm_exp2_small_epsilon_output_none` | 197-199 |

### 用户模型到性能 case ID

10 条性能用例均使用 BF16 输入、`K=V=128`、`chunk_size=64`、连续 BNSD、
`scale=0.08838`、融合 norm/gate/beta、safe gate、exp2 和 `output_mode=none`。
变长 case 仍以连续 BNSD tensor 为输入，只额外提供 `cu_seqlens`；case 2 同时提供
由 `cu_seqlens` 生成的 `chunk_indices`。已知基线和目标均为 msopprof 单次 kernel
duration；未提供的数据不臆造。

| case ID | case key | `B/HK/HV/T` | 序列 | 已知基线 | 目标 | 实测性能 | 结论 |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| 0 | `model_b2_hk16_hv32_t11264` | `2/16/32/11264` | dense | 未提供 | 未提供 | 未测试 | 未验收 |
| 1 | `model_b1_hk16_hv32_t11264` | `1/16/32/11264` | dense | 1494.040 us | 未提供 | 未测试 | 未验收 |
| 2 | `model_b1_hk32_hv32_t65536_varlen` | `1/32/32/65536` | 64-sequence varlen | 未提供 | 未提供 | 未测试 | 未验收 |
| 3 | `model_b4_hk96_hv96_t128` | `4/96/96/128` | dense | 未提供 | 未提供 | 未测试 | 未验收 |
| 4 | `model_b1_hk32_hv32_t160` | `1/32/32/160` | dense | 未提供 | 未提供 | 未测试 | 未验收 |
| 5 | `model_b6_hk6_hv6_t1084` | `6/6/6/1084` | dense | 未提供 | 未提供 | 未测试 | 未验收 |
| 6 | `model_b1_hk12_hv12_t1084` | `1/12/12/1084` | dense | 未提供 | 未提供 | 未测试 | 未验收 |
| 7 | `model_b1_hk96_hv96_t8192_inference` | `1/96/96/8192` | dense | 3221.331 us | 2700 us | 未测试 | 待验收（已知基线高于目标） |
| 8 | `model_b1_hk96_hv96_t16384_varlen_inference` | `1/96/96/16384` | 64-sequence varlen | 7131.371 us | 5600 us | 未测试 | 待验收（已知基线高于目标） |
| 9 | `model_b1_hk8_hv24_t32768` | `1/8/24/32768` | dense | 未提供 | 未提供 | 未测试 | 未验收 |

当前提交未提供上述 10 个 case 的本轮 msopprof 实测结果；因此性能验收整体结论为
`未完成`，不能用已知基线或部分 case 代替其余 case。

所有矩阵轴、逻辑场景、性能 case、默认值和随机种子均声明在
`tests/op_cases/chunk_kda_fwd_prepare.json` 的 `atk_generation` 字段中。本目录的
生成器只负责展开声明，不维护第二份 shape 或属性列表。

## TilingKey 覆盖

432 个可达组合来自：

逐 key 的冻结映射和覆盖缺口见
[`tiling_key_matrix.md`](./tiling_key_matrix.md)。该附录由
`scripts/generate_tiling_key_matrix.py` 从两份冻结 JSON 机械生成，列出每个 key 的
精度普通/边界 case、确定性 case、四类 sanitizer case、规格 SoC 和实际选择证据；
其中 `未覆盖`、`未提供` 明确表示尚未完成，不等价于测试通过。

```text
2 gate dtype x 2 beta dtype x 2 norm mode x
3 beta mode x 3 legal gate mode x 2 exp domain x 3 output mode = 432
```

- gate/beta dtype：BF16、FP32；Q/K/V 固定为 BF16。
- beta mode：raw、sigmoid、two-sigmoid。
- gate mode：预计算 step、softplus、safe sigmoid；`safe_gate` 只在 safe sigmoid
  置位，不生成数学等价的重复组合。
- exp domain：`exp`、`exp2`。
- output mode：`none`、`recompute`、`save`，各 144 条。

冻结矩阵按以下顺序编号：

```text
case_id = 216 * gate_dtype_index
        + 108 * beta_dtype_index
        +  54 * norm_index
        +  18 * beta_mode_index
        +   6 * gate_mode_index
        +   3 * exp_index
        +       output_mode_index
```

各轴 index 固定为：`gate_dtype/beta_dtype: bf16=0, fp32=1`，
`norm/use_exp2: false=0, true=1`，`beta_mode: raw=0, sigmoid=1,
two_sigmoid=2`，`gate_mode: precomputed=0, softplus=1, safe=2`，
`output_mode: none=0, recompute=1, save=2`。

每个 case ID 对应的实际 TilingKey 按 kernel 编码直接计算：

```text
tiling_key = gate_dtype_token
           + (beta_dtype_token << 8)
           + (norm_index << 16)
           + (beta_mode_index << 17)
           + (gate_mode_index << 19)
           + (use_exp2_index << 21)
           + ((gate_mode == safe) << 22)
           + (output_mode_index << 23)
```

其中 dtype token 为 `bf16=10, fp32=30`。因此上述两个公式共同给出全部
432 个“期望 TilingKey <-> `_mss.json` case ID”的一一映射；这只是冻结的输入映射，
不是运行时选择证据。冻结的 `(case ID, TilingKey)` 顺序摘要为
`f28a869f07d8e3f65e8d0c85768759bde635759a09d047cd78b4a9b7136d3b19`。

12 个运行 profile 通过正交轮转映射到 432 个模板组合，每个 profile 恰好出现
36 次。profile 覆盖 BNSD/BSND/NTD/TND、dense/varlen、自动/显式 chunk
indices、空序列、尾块、GVA、head 分核、chunk-only 分核、128 core 以上的
grid-stride，以及 1/2/3/4/5/8/12 个 active head 和 slot 复用。融合 gate 路径还
独立覆盖 `(exp/exp2) x (有/无 dt_bias)`。

静态检查同时冻结 case/key 对的 SHA256，防止生成顺序、取值或 key 集合静默漂移：

```bash
python3 tests/atk/chunk_kda_fwd_prepare/scripts/check_coverage.py
python3 tests/atk/chunk_kda_fwd_prepare/scripts/generate_tiling_key_matrix.py --check
python3 -m unittest tests.operators.chunk_kda_fwd_prepare.ut.test_atk_generation
```

确定性和每一种内存检查都开启 OP 调试日志。`verify_matrix.py` 分别提取：

1. Host `PrintTiling` 计算的 `tilingKey`；
2. `op_executor` 在实际 Launch 前使用的 `Tiling Key`；
3. MSS JSON 中的 `expected_tiling_key`。

三组集合必须各有 432 个唯一值、互相完全相等且无缺失或额外 key，不能用安装包
静态 metadata 代替真实运行命中证据。矩阵根目录还会冻结实际加载的 op_api、kernel
对象、metadata、Python 包装器和测试程序指纹，并记录 ATK 版本及可执行文件指纹；
内存矩阵额外记录 mssanitizer/msopscommon revision 和 mssanitizer 可执行文件指纹。
SoC、二进制、工具链、超时、循环次数或 GM 预填充配置变化时拒绝复用旧分片。

## 执行方式

200 条精度和 432 条确定性/内存矩阵使用可恢复分片入口。每个 ATK 进程只下发一条
case，并使用多进程任务模式，使 `-to 60` 真实约束当前 case，不把排队时间算入其他
case。正式精度矩阵关闭 ATK 的整卡 GM 预填充；该步骤不受单进程 `-to` 约束，在大
显存设备上不能满足 60 秒合同。未初始化、越界、竞争和同步问题由四种 sanitizer 的
完整 432-key 矩阵覆盖；确定性矩阵固定执行 50 轮。

内存检查必须先安装通过 `--sanitizer` 构建的单算子 wheel。该选项同时启用
`sanitizer`、`dump_cce` 和 `--op_debug_level=1`：

```bash
# A2 使用 ascend910b，A3 使用 ascend910_93，A5 使用 ascend950
export KDA_PREPARE_ATK_SOC=ascend950
FLA_NPU_SOC="$KDA_PREPARE_ATK_SOC" FLA_NPU_OPS=chunk_kda_fwd_prepare \
  python3 scripts/build_wheel.py --sanitizer --wheel-dir dist
```

`--sanitizer` 通过 `--bisheng_flags` 保留 `asc_opc` 调试配置；V2 构建链还会
将同一配置映射为 `-g/-sanitizer` Bisheng 编译选项。仅出现
`asc_opc --op_debug_config=sanitizer` 不代表对象已经插桩；正式矩阵仍会用
`nm` 检查安装包内每个 kernel 对象的 sanitizer 符号。

`FLA_NPU_SOC` 必须与后续 `KDA_PREPARE_ATK_SOC` 完全一致；安装该 wheel 并加载其
custom OPP 后再执行内存矩阵，禁止用其他平台或普通优化包代替 sanitizer 包。正式
矩阵必须显式设置 `KDA_PREPARE_ATK_SOC`，不接受 `auto`；该值会同时约束运行时
kernel 指纹和实际选中物理卡的 SoC。

`run_matrix.sh` 会在下发前逐个检查 metadata 引用的 `.o` 含 sanitizer 符号；同一
TilingKey 可以由多个 hashed 编译对象提供候选，但它们可能分别支持不同的 `g/beta`
dtype。运行时清单从 metadata 的 `supportInfo.inputs` 提取调度签名；每个 case 必须按
`gate_dtype/beta_dtype` 命中唯一匹配对象，且 Start/Finish 名称及次数闭合。运行后从
实际命中 kernel 名提取
TilingKey，并要求其集合与该分片的期望 key 完全一致，同时拒绝任何目标 kernel 的
`No active sanitizer tool` 记录。每个分片的
`<tool>.log` 同时是 ATK `-msl` 和外层 mssanitizer 的原始日志；分片摘要记录其哈希，
续跑和聚合时重新检查 clean-finish、诊断行、文件哈希及 case/kernel 调度绑定摘要；
四种工具还必须命中同一组完整绑定，不能只凭 ATK xlsx 判定通过。

```bash
# 200 条精度，每次独立执行 1 条
bash tests/atk/chunk_kda_fwd_prepare/scripts/run_matrix.sh accuracy 0

# 432 个 key，每条重复 50 次，每次独立执行 1 条
bash tests/atk/chunk_kda_fwd_prepare/scripts/run_matrix.sh determinism 0

# 同一 432-key 矩阵分别执行四种内存/同步检查
MSS_TOOL=memcheck \
  bash tests/atk/chunk_kda_fwd_prepare/scripts/run_matrix.sh mssanitizer 0
MSS_TOOL=racecheck \
  bash tests/atk/chunk_kda_fwd_prepare/scripts/run_matrix.sh mssanitizer 0
MSS_TOOL=initcheck \
  bash tests/atk/chunk_kda_fwd_prepare/scripts/run_matrix.sh mssanitizer 0
MSS_TOOL=synccheck \
  bash tests/atk/chunk_kda_fwd_prepare/scripts/run_matrix.sh mssanitizer 0

# 四种工具完成后，校验它们使用同一 runtime 且各自覆盖全部 432 个 key
python3 tests/atk/chunk_kda_fwd_prepare/scripts/verify_matrix.py \
  sanitizer-suite \
  --case-file tests/atk/chunk_kda_fwd_prepare/atk_chunk_kda_fwd_prepare_mss.json \
  --soc "$KDA_PREPARE_ATK_SOC" \
  --test-artifact tests/atk/chunk_kda_fwd_prepare/executor_chunk_kda_fwd_prepare.py \
  --test-artifact tests/atk/chunk_kda_fwd_prepare/chunk_kda_fwd_prepare.yaml \
  --test-artifact tests/atk/chunk_kda_fwd_prepare/gen_chunk_kda_fwd_prepare.py \
  --test-artifact tests/op_cases/chunk_kda_fwd_prepare.json \
  --test-artifact tests/atk/common/_ascendc_common_executor.py \
  --test-artifact tests/atk/common/check_atk_result.py \
  --test-artifact tests/atk/run_test_cpu.sh \
  --test-artifact tests/atk/chunk_kda_fwd_prepare/scripts/check_coverage.py \
  --test-artifact tests/atk/chunk_kda_fwd_prepare/scripts/run_matrix.sh \
  --test-artifact tests/atk/chunk_kda_fwd_prepare/scripts/verify_matrix.py \
  --aggregate <memcheck根目录>/aggregate_summary.json \
  --aggregate <racecheck根目录>/aggregate_summary.json \
  --aggregate <initcheck根目录>/aggregate_summary.json \
  --aggregate <synccheck根目录>/aggregate_summary.json \
  --output <汇总目录>/sanitizer_suite_summary.json
```

性能继续使用统一入口；正式执行显式关闭 `-sp`、开启 ATK
`--fluctuation_check`，并把每条 case 的超时限制为 60 秒。开启波动校验后，
报告中的“性能波动校验结果”列才具有明确的 Pass/Fail 语义。统一入口只接受本次
调用开始后生成或更新的 xlsx；case 7/8 还会分别硬校验 NPU Device 性能不超过
2700/5600 us，缺少耗时列或测量值同样失败：

```bash
ATK_SINGLE_PROCESS=off PERFORMANCE_TIMEOUT=60 \
  bash tests/atk/run_test_cpu.sh \
  -op=chunk_kda_fwd_prepare -npu_device_id=0 -scope=performance

# 分段只跑 case 7、8；范围为半开区间 [7, 9)
PERFORMANCE_START=7 PERFORMANCE_END=9 \
  ATK_SINGLE_PROCESS=off PERFORMANCE_TIMEOUT=60 \
  bash tests/atk/run_test_cpu.sh \
  -op=chunk_kda_fwd_prepare -npu_device_id=0 -scope=performance
```

正式矩阵的分片大小固定为 1，各阶段超时参数只能设为 1 到 60 秒，正式确定性矩阵固定
执行 50 轮。续跑时同时设置既有
`KDA_PREPARE_ATK_MATRIX_ROOT` 和 `KDA_PREPARE_ATK_MATRIX_START`；已有分片只有在
scope、工具、范围、用例哈希、运行参数、二进制指纹和 case ID 均匹配，且重新解析的
原始 xlsx/console/sanitizer 日志与原摘要完全一致时才会复用。最终
`aggregate_summary.json` 必须显示完整 case 集合闭合且 `passed=true`。

重建三份冻结 JSON：

```bash
python3 tests/atk/chunk_kda_fwd_prepare/gen_chunk_kda_fwd_prepare.py \
  --output-dir tests/atk/chunk_kda_fwd_prepare \
  --summary
```

A2、A3、A5 的 JSON 规格可以共用 `soc=all` 作为平台无关的输入描述，但实际编译和
运行必须按平台分别执行，不能用任一平台的结果替代其他平台。每个平台都必须单独
记录 host/launch TilingKey、精度普通/边界 case、确定性、四类 sanitizer 及实际加载
的插桩对象证据；当前文档不把 `soc=all` 视为三平台已覆盖。

平台覆盖状态（本提交不虚构未执行结果）：

| SoC | 精度 200 条 | 确定性 432 key | 四类 sanitizer 432 key | 实际选择/对象证据 |
| --- | --- | --- | --- | --- |
| A2 (`ascend910b`) | 200/200 Pass | 未完成 | 未完成 | 精度汇总已核验；其余矩阵待补 |
| A3 (`ascend910_93`) | 未提供 | 未提供 | 未提供 | 未提供 |
| A5 (`ascend950`) | 未提供 | 未提供 | 未提供 | 未提供 |
