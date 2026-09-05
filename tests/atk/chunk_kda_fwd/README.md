# chunk_kda_fwd ATK 交付件

本目录提供当前主线 `chunk_kda_fwd` 的 ATK 精度、确定性和内存测试资产。测试
executor 使用现代 raw `g`、`A_log`、`dt_bias` 接口，与当前算子实现保持同一 ABI。

## 交付清单

| 文件 | 用途 | 规模 |
| --- | --- | --- |
| `atk_chunk_kda_fwd.json` | 混合容差精度 | 200 条正向用例（ID 0--199） |
| `atk_chunk_kda_fwd_mss.json` | `accuracy_dc` 与 mssanitizer | 7 条：4 条 key 覆盖、2 条 A5 同步回归与 1 条变长 tail 回归 |
| `atk_chunk_kda_fwd_perf.json` | 性能采样 | 2 条：每个 tiling key 一条 |
| `gen_chunk_kda_fwd.py` | 生成并校验上述清单 | canonical seed 为 `20260831` |
| `executor_chunk_kda_fwd.py` | CPU golden 与 NPU DUT | `ascendc` 主通路 |
| `stress_npu_determinism.py` | 独立逐 bit 确定性诊断 | 默认遍历 MSS 清单 |
| `scripts/validate_manifests.py` | 无 NPU 静态校验 | 校验源码选择、私有依赖边界与 JSON 漂移 |

精度清单使用 `soc=all`，同一份 200 条矩阵应分别在 A2（`ascend910b`）、A3
（`ascend910_93`）和 A5（`ascend950`）设备上执行，不把同一矩阵复制成 600 条。

## Tiling key 覆盖

主线 host 只根据 `chunk_size/K/V` 选择 key：

| key | 选择条件 | MSS 普通 | MSS 边界 | 性能 |
| --- | --- | --- | --- | --- |
| 1 | 除 key2 条件外的合法组合 | ID 0（`V=256`） | ID 1（`chunk=128`） | ID 0 |
| 2 | `chunk_size=64` 且 `K=128` 且 `V=128` | ID 2 | ID 3（`T=65`） | ID 1 |

MSS ID 0--3 保持原有两个 key 的普通/边界矩阵。ID 4 原样克隆 accuracy case 4，
覆盖 A5 unsafe BF16 key1 的单次 full launch；ID 5 原样克隆 accuracy case 24，覆盖同一路径
的四阶段 staged launch 和 63-token tail。两条克隆保留源用例的 seed、shape、layout、dtype、
输入 range 与输出策略，只增加来源和 launch-mode 元数据；validator 会逐字段核对并从 host
分阶段条件重新推导 launch mode。
ID 6 保留 `T=63/HV=96/K=V=128` 的 key2 变长 tail 回归，使用固定 seed 4 和
`model_h96` 输入分布，覆盖双 AIV 对原位 `W` 的 K 列分片与从高行到低行写回。

精度清单在 100 个结构 profile 上分别使用 BF16/FP16，固定为 200 条。结构矩阵覆盖四种
layout（`BSND/BNSD/TND/NTD`）、dense/varlen、tail、GQA、initial/final state，以及：

- 五种 gate 组合：已激活 gate、raw unsafe gate 有/无 `dt_bias`、raw safe gate 有/无
  `dt_bias`；raw `g` 同时覆盖 FP32/BF16。
- `disable_recompute` 与 `return_intermediate_states` 的四种组合，以及
  `state_v_first=false/true`。
- `K=16/128/256`、`V=128/256`、`chunk_size=64/128` 和 key1/key2。
- 重复 `cu_seqlens` 形成的空序列：无 initial state 时空 final-state 槽清零；有 initial
  state 时空槽原样透传，并验证非空序列使用原始 batch state 而非压缩后的索引。

profile 28 的 BF16 用例（ID 56）固定为 A5 key2 融合候选：dense/aligned、FP32 raw gate、
safe gate、偶数 `HV`，且不导出 QG/VNew/H。profile 5 的 BF16 用例（ID 10）固定保留已复现
卡死结构：`B=1,H=2,HV=2,T=65,K=128,V=256,chunk=64,BNSD`，dense、无 initial state、
返回 final state，并开启 safe/raw gate、`dt_bias`、`disable_recompute` 和 intermediate state。
validator 对两个固定点执行精确存在性检查。profile 的 key 由同一 host 条件计算并写入
`case_spec`。`soc=all` 是平台通配标记，实际运行时仍需针对每个物理 SoC 单独执行。

profile 41（ID 82/83）固定覆盖无 initial state 的交错空序列，profile 56（ID 112/113）
固定覆盖有 initial state、显式 `chunk_indices` 与 `state_v_first=true` 的交错空序列。
validator 对两组 BF16/FP16 成对结构和 kernel 的 compact-to-original state 映射执行检查。

## 精度复检

case ID `4`、`12`、`14`、`52`、`54` 的单轮 `mixed_tolerance_bm` 结果需要复检。生成器在这些
用例的 `tags` 中写入 `needs_accuracy_lt_recheck`，validator 会精确校验该 ID 集合。

主清单和当前 executor 使用 ATK 26.8 `mixed_tolerance_bm` 单标杆：executor 固定读取
`case_spec.seed`，且没有同精度 CPU control。因此不能直接对这组资产执行 50 轮后将结果
作为 CT 双标杆结论。复检必须使用兼容的独立双标杆 fixture，并同时满足：

- 与主清单的 shape、layout、dtype、range 和其他结构参数逐字段一致；
- `accuracy_lt` 每轮按 runtime case ID 派生新 seed，同一轮三路输入保持一致；
- 明确区分 NPU DUT、同精度 CPU control 和 FP64 CPU benchmark；
- 执行 50 轮且不使用 `-sp`，最终使用 CT L2 聚合，同时保留单轮输出定位结构性错误。

复检不能替代执行错误、非有限值或 sanitizer 问题的修复，也不能把当前单标杆报告解释为
双标杆报告。

## 输入与输出约定

- `q/k/v` 使用相同的 BF16 或 FP16；`g` 使用 FP32 或 BF16；`beta` 使用 FP32 或 BF16。
- `K`、`V` 为 16 的倍数，范围为 16--256；精度矩阵覆盖 `K=16/128/256`、
  `V=128/256`。
- `layout` 支持 `BSND`、`BNSD`、`TND`、`NTD`。变长输入的 `cu_seqlens` 从 0 开始并以
  `T` 结束，`chunk_indices` 使用 sequence-major canonical 顺序。
- `use_gate_in_kernel=true` 时提供 FP32 `A_log`；若提供 `dt_bias`，其形状为 `[HV*K]`。
- CPU 节点以 FP64 计算 golden，executor 只在 golden 输出边界转为 FP32；NPU 节点保留
  算子原始输出 dtype，由 `mixed_tolerance_bm` 单标杆统一比较。

## 运行

先加载 CANN、算子 OPP、Python 包和 ATK 26.8.8 或更高版本，再从仓库根目录运行：

```bash
# 精度：每个物理 SoC 都执行完整 200 条
bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd -soc=ascend910b -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd -soc=ascend910_93 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd -soc=ascend950 -scope=accuracy

# 确定性：复用 7 条 MSS 清单，覆盖 key1/key2、unsafe full/staged 与变长 tail 回归
bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd -soc=ascend950 -scope=determinism

# 内存：需要 sanitizer/debug 算子包；默认使用 memcheck
bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd -soc=ascend950 -scope=mssanitizer

# 无 NPU 的清单与源码选择条件检查
python3 tests/atk/chunk_kda_fwd/scripts/validate_manifests.py
```

卡死判定采用单次 kernel launch 口径：任一 case 的日志进入 `KERNEL_LAUNCH` 后
60 秒仍未返回，即判定该 case 卡死并记为失败。公共 runner 的 `-to` 是从 ATK 每个
执行阶段开始计算的软超时，不以 `KERNEL_LAUNCH` 为计时起点，也不能替代该判据；在
接入精确的 launch watchdog 前，运行人员必须按日志时间监控并终止卡死 case，不能通过
延长等待时间把它记录为通过。

确定性阶段使用 ATK `accuracy_dc`，内存阶段使用 mssanitizer 包裹的 ATK `run`；二者
均消费 `atk_chunk_kda_fwd_mss.json`。独立诊断脚本按 MSS 本地 ID 工作，默认遍历全部
7 条用例，也可用 `--case-id` 只定位一条；staged tail 同步回归使用 MSS ID 5，
`T=63/HV=96` 的 key2 变长 tail 回归使用 MSS ID 6：

```bash
python3 tests/atk/chunk_kda_fwd/stress_npu_determinism.py \
  --device 0 --soc ascend950 --repeats 100
python3 tests/atk/chunk_kda_fwd/stress_npu_determinism.py \
  --device 0 --soc ascend950 --case-id 5 --repeats 100
python3 tests/atk/chunk_kda_fwd/stress_npu_determinism.py \
  --device 0 --soc ascend950 --case-id 6 --repeats 100
```

脚本每轮从固定输入的 clone 发起调用，检查所有非空输出的 shape、dtype 和 bitwise
一致性；任一用例异常或不一致都会以非零状态退出。mssanitizer 只有在确认 debug
对象实际命中 sanitizer 版本并看到对应工具启动信息后，才能作为内存结论。

内存回归至少对 ID 0--6 全部执行 `memcheck`；A5 上再对 ID 4/5 分别执行
`racecheck`、`synccheck` 和 `initcheck`，可通过
`MSS_START=4 MSS_END=6 MSS_TOOL=<tool>` 限制到两条 unsafe 同步回归。ID 6 是双 AIV
原位写回竞争回归，需在 A2、A3、A5 分别执行 `racecheck`，范围为
`MSS_START=6 MSS_END=7 MSS_TOOL=racecheck`。staged ID 5 的四个物理 launch 和 ID 6
都必须出现对应 sanitizer 启动标记；出现 inactive 提示时不能形成“未发现问题”的结论。

## 重新物化

canonical 清单由生成器冻结为 200 条精度、7 条 MSS、2 条性能用例。重新生成并校验：

```bash
python3 tests/atk/chunk_kda_fwd/gen_chunk_kda_fwd.py \
  --out-dir tests/atk/chunk_kda_fwd --positive 200 --negative 0 \
  --seed 20260831 --print-summary
python3 tests/atk/chunk_kda_fwd/scripts/validate_manifests.py
```

不要提交 ATK 运行产生的 `atk_output`、报告、日志、profiling 数据或 Python 缓存。
