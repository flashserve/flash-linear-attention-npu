# RecurrentGatedDeltaRule ATK 工程

本目录提供 `recurrent_gated_delta_rule` 的 ATK 单算子工程。测试通过稳定入口 `fla_npu.ops.ascendc.recurrent_gated_delta_rule` 调用算子，并同时比较注意力输出和原地更新后的状态。

## 输入约束

- `query/key` 的 shape 为 `(T, Nk, Dk)`，`value/out` 的 shape 为 `(T, Nv, Dv)`。
- `beta` 的 shape 为 `(T, Nv)`；`stateRef` 为原地输入输出，shape 为 `(BlockNum, Nv, Dv, Dk)`。
- `query/key/value/beta/out` 为 BF16；`stateRef` 支持 BF16 和 FP32。
- `actualSeqLengths` 为 `(B+1,)` 的 INT32 张量，首元素是不参与计算的无效前缀长度，所有元素之和等于 `T`。
- `ssmStateIndices` 为 `(T,)` 的 INT32 张量，取值范围为 `[0, BlockNum)`。
- `g` 如提供，shape 为 `(T, Nv)`；`gk` 如提供，shape 为 `(T, Nv, Dk)`。两者均为 FP32，且至少提供一个。
- `numAcceptedTokens` 如提供，shape 为 `(B,)`，每项取值范围为 `[1, actualSeqLengths[i+1]]`。
- `Nv` 必须是 `Nk` 的整数倍；`Nk/Nv <= 256`，`Dk/Dv <= 512`，每条有效序列长度不超过 8。

## 标杆与比较对象

CPU 标杆在 `executor_recurrent_gated_delta_rule.py` 中按 batch 分段执行递推，覆盖 GQA 头映射、`g/gk` 衰减、接受 token 对初始状态索引的选择，以及逐 token 状态写回。高精度标杆使用 FP64 计算，同精度标杆保留原始输入量化和输出 dtype。

ATK 比较以下两个结果：

- `out`：`(T, Nv, Dv)` 的注意力输出；无效前缀没有公开数值语义，比较前统一置零。
- `stateRef`：算子执行后原地更新的完整状态张量。

参考实现同时核对了算子目录下的 `tests/pta/golden.py` 和 kernel 状态更新流程。

## 冻结用例

`atk_recurrent_gated_delta_rule.json` 由 `gen_recurrent_gated_delta_rule.py` 生成，固定包含以下 10 条用例：

| Case ID | 场景 | 关键覆盖 |
|---:|---|---|
| 0 | 最小 BF16 state | 单 token、`g` |
| 1 | 基础 FP32 state | 双 token、`g` |
| 2 | 非整块尾维 | `Dk=80`、`Dv=96`、仅 `gk` |
| 3 | 变长 GQA | 双 batch、`g+gk` |
| 4 | 接受 token | 三 batch、GQA、FP32 state |
| 5 | 无效前缀 | 前缀长度 2、变长、`g+gk` |
| 6 | 最大 MTP | 单序列 8 token、`Dv=256` |
| 7 | 零长度 batch | 中间 batch 长度为 0 |
| 8 | 非连续多头 `gk` | `Nk=4`、`Nv=8`、FP32 state、transpose 非连续布局 |
| 9 | 最大状态维度 | `Dk=Dv=512`、`g+gk` |

冻结用例使用 `soc=all`，同一矩阵用于 A2、A3、A5；实际执行平台通过统一脚本的 `-soc` 参数记录。

## 执行方式

执行前先加载 CANN 环境和待测 `fla_npu_transformer` OPP 环境。若使用一体化 wheel，可将 `FLA_NPU_ENV` 指向 wheel 内 `fla_npu/opp/vendors/fla_npu_transformer/bin/set_env.bash`；统一脚本会在启动 ATK 进程前加载它，避免 op_api 与 kernel binary 来自不同安装包。
ATK 需要同时支持原生 `cv_fused_double_benchmark` 和目标 SOC 的 `performance_device` 后处理；A2 四项调用使用 ATK 26.4.30 验证，A5 使用 ATK 26.7.8 验证。

```bash
export FLA_NPU_ENV=<fla_npu_transformer>/bin/set_env.bash
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -npu_device_id=0 -scope=accuracy
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -npu_device_id=0 -scope=performance
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -npu_device_id=0 -scope=determinism
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -npu_device_id=0 -scope=mssanitizer
```

不传 `-scope` 时默认依次执行上述四项。mssanitizer 必须使用带 sanitizer 信息的 debug OPP 包，并确认日志实际命中目标 kernel。
精度 scope 会在当前 ATK 支持时自动加入 `--gm_init_flag`，A5 默认跳过该选项以避免 ATK 按空闲 GM 规模申请内存；不支持该选项的 ATK 版本仍执行 CPU 高精度与同精度双标杆比较。可通过 `ATK_GM_INIT_MODE=on/off` 显式覆盖默认策略。
内存检查会兼容探测 ATK 的 `--mssanitizer/-msl` 参数；无论 ATK 是否提供内存报告后处理，都由外层 mssanitizer 日志确认目标 kernel 启动、结束且未检测到异常。

用例生成和冻结 JSON 复核命令如下：

```bash
bash tests/atk/run_test_cpu.sh -op=recurrent_gated_delta_rule -scope=gen_cases
python3 tests/atk/recurrent_gated_delta_rule/gen_recurrent_gated_delta_rule.py --summary
```

`gen_cases` 用于检查 ATK 生成器接入；提交的冻结 JSON 由生成器 CLI 重建，便于直接复核 case 数量和具体参数。
