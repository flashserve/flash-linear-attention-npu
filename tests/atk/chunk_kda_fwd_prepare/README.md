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
| `atk_chunk_kda_fwd_prepare.json` | 200 | 56 个逻辑场景，每场景 3 或 4 个固定 seed；覆盖四种 layout、dense/varlen、tail、GVA、三档输出策略及全部功能分支 |
| `atk_chunk_kda_fwd_prepare_perf.json` | 9 | 纯推理输出模式下的模型 shape、跨 wave GVA、layout、tail 和 varlen |
| `atk_chunk_kda_fwd_prepare_mss.json` | 432 | 每个可达 TilingKey 恰好一条，用于确定性和四类 sanitizer 检查 |

两个重点性能用例为：

| case ID | Shape | 目标 |
| ---: | --- | ---: |
| 7 | `B=1, HK=HV=96, T=8192, BNSD` | 2700 us |
| 8 | `B=1, HK=HV=96, T=16384, BNSD varlen` | 5600 us |

所有矩阵轴、逻辑场景、性能 case、默认值和随机种子均声明在
`tests/op_cases/chunk_kda_fwd_prepare.json` 的 `atk_generation` 字段中。本目录的
生成器只负责展开声明，不维护第二份 shape 或属性列表。

## TilingKey 覆盖

432 个可达组合来自：

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

12 个运行 profile 通过正交轮转映射到 432 个模板组合，每个 profile 恰好出现
36 次。profile 覆盖 BNSD/BSND/NTD/TND、dense/varlen、自动/显式 chunk
indices、空序列、尾块、GVA、head 分核、chunk-only 分核、128 core 以上的
grid-stride，以及 1/2/3/4/5/8/12 个 active head 和 slot 复用。融合 gate 路径还
独立覆盖 `(exp/exp2) x (有/无 dt_bias)`。

静态检查同时冻结 case/key 对的 SHA256，防止生成顺序、取值或 key 集合静默漂移：

```bash
python3 tests/atk/chunk_kda_fwd_prepare/scripts/check_coverage.py
python3 -m unittest tests.operators.chunk_kda_fwd_prepare.ut.test_atk_generation
```

确定性和每一种内存检查都开启 OP 调试日志。`verify_matrix.py` 分别提取：

1. Host `PrintTiling` 计算的 `tilingKey`；
2. `op_executor` 在实际 Launch 前使用的 `Tiling Key`；
3. MSS JSON 中的 `expected_tiling_key`。

三组集合必须各有 432 个唯一值、互相完全相等且无缺失或额外 key，不能用安装包
静态 metadata 代替真实运行命中证据。矩阵根目录还会冻结实际加载的 op_api、kernel
对象及 metadata 指纹；SoC、二进制、超时、循环次数或 GM 预填充配置变化时拒绝复用
旧分片。

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

`--sanitizer` 会同时经 `--op_debug_config` 生成 `-g/-sanitizer`
Bisheng 编译选项，并保留 `asc_opc` 的 sanitizer 调试配置。仅出现
`asc_opc --op_debug_config=sanitizer` 不代表对象已经插桩；正式矩阵仍会用
`nm` 检查安装包内每个 kernel 对象的 sanitizer 符号。

`FLA_NPU_SOC` 必须与后续 `KDA_PREPARE_ATK_SOC` 完全一致；安装该 wheel 并加载其
custom OPP 后再执行内存矩阵，禁止用其他平台或普通优化包代替 sanitizer 包。正式
矩阵必须显式设置 `KDA_PREPARE_ATK_SOC`，不接受 `auto`；该值会同时约束运行时
kernel 指纹和实际选中物理卡的 SoC。

`run_matrix.sh` 会在下发前逐个检查 metadata 引用的 `.o` 含 sanitizer 符号；运行后
从工具启动日志提取实际 kernel 名中的 TilingKey，并要求其集合与该分片的期望 key
完全一致，同时拒绝任何目标 kernel 的 `No active sanitizer tool` 记录。

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
  --test-artifact tests/atk/run_test_cpu.sh \
  --test-artifact tests/atk/chunk_kda_fwd_prepare/scripts/run_matrix.sh \
  --test-artifact tests/atk/chunk_kda_fwd_prepare/scripts/verify_matrix.py \
  --aggregate <memcheck根目录>/aggregate_summary.json \
  --aggregate <racecheck根目录>/aggregate_summary.json \
  --aggregate <initcheck根目录>/aggregate_summary.json \
  --aggregate <synccheck根目录>/aggregate_summary.json \
  --output <汇总目录>/sanitizer_suite_summary.json
```

性能继续使用统一入口：

```bash
bash tests/atk/run_test_cpu.sh \
  -op=chunk_kda_fwd_prepare -npu_device_id=0 -scope=performance
```

正式矩阵的分片大小固定为 1，三类超时参数只能设为 1 到 60 秒，正式确定性矩阵固定
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

A2、A3、A5 使用同一份 `soc=all` 矩阵。编译、精度、确定性及内存结果必须按平台
分别记录，不能用任一平台的结果替代其他平台；sanitizer 结论还必须确认实际加载了
插桩后的算子对象，日志出现对应工具启动记录。
